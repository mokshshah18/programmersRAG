// Idle detection: CPU load of other processes and AC power.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#endif

#include "prag/tune/pipeline.hpp"

namespace prag::tune {
namespace {

struct CpuSample {
  double busy = 0.0;   // system-wide busy time, arbitrary unit
  double total = 0.0;  // system-wide total time, same unit
  double own = 0.0;    // this process's CPU time, same unit
  bool ok = false;
};

#ifdef _WIN32
double ticks(const FILETIME& f) {
  return static_cast<double>((static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime);
}

CpuSample sample_cpu() {
  CpuSample s;
  FILETIME idle, kernel, user, create, exit_time, pk, pu;
  if (!GetSystemTimes(&idle, &kernel, &user)) return s;
  if (!GetProcessTimes(GetCurrentProcess(), &create, &exit_time, &pk, &pu)) return s;
  s.total = ticks(kernel) + ticks(user);  // kernel time includes idle time
  s.busy = s.total - ticks(idle);
  s.own = ticks(pk) + ticks(pu);
  s.ok = true;
  return s;
}

bool on_ac_power() {
  SYSTEM_POWER_STATUS status{};
  if (!GetSystemPowerStatus(&status)) return true;
  return status.ACLineStatus != 0;  // 255 (unknown) counts as AC
}
#else
CpuSample sample_cpu() {
  CpuSample s;
  std::ifstream stat("/proc/stat");
  std::string label;
  double v[8] = {};
  if (!(stat >> label) || label != "cpu") return s;
  for (double& x : v) stat >> x;
  if (!stat) return s;
  const double idle = v[3] + v[4];  // idle + iowait
  for (double x : v) s.total += x;
  s.busy = s.total - idle;
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  const double hz = static_cast<double>(sysconf(_SC_CLK_TCK));
  s.own = (static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
           static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) * 1e-6) * hz;
  s.ok = true;
  return s;
}

bool on_ac_power() { return true; }
#endif

std::string seconds_text(double seconds) {
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%.0f s", seconds);
  return buffer;
}

}  // namespace

IdleState probe_idle(double seconds_since_activity, const IdlePolicy& policy) {
  IdleState state;
  state.seconds_since_activity = seconds_since_activity;

  const CpuSample a = sample_cpu();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const CpuSample b = sample_cpu();
  if (a.ok && b.ok && b.total > a.total) {
    const double system = (b.busy - a.busy) / (b.total - a.total);
    const double own = (b.own - a.own) / (b.total - a.total);
    state.cpu_load = std::clamp(system - own, 0.0, 1.0);
  }
  state.on_ac_power = on_ac_power();

  char buffer[96];
  if (seconds_since_activity < policy.idle_seconds) {
    state.reason = "user active " + seconds_text(seconds_since_activity) + " ago";
  } else if (state.cpu_load > policy.max_cpu_load) {
    std::snprintf(buffer, sizeof buffer, "CPU load %.0f%% above %.0f%%", state.cpu_load * 100.0,
                  policy.max_cpu_load * 100.0);
    state.reason = buffer;
  } else if (policy.require_ac_power && !state.on_ac_power) {
    state.reason = "running on battery";
  } else {
    state.idle = true;
    state.reason = "idle for " + seconds_text(seconds_since_activity);
  }
  return state;
}

}  // namespace prag::tune
