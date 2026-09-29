// Parts of the llm module that are the same with and without llama.cpp.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <thread>
#include <vector>

#include "detail/util.hpp"
#include "prag/llm/model.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace prag::llm {

double Likelihood::perplexity() const {
  if (tokens <= 0) return std::numeric_limits<double>::quiet_NaN();
  return std::exp(nll / static_cast<double>(tokens));
}

namespace detail {

namespace {

int hardware_threads() {
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

int detect_physical_cores() {
#if defined(_WIN32)
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes != 0) {
    std::vector<char> buffer(bytes);
    auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes)) {
      int cores = 0;
      for (DWORD offset = 0; offset < bytes;) {
        const auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (info->Relationship == RelationProcessorCore) ++cores;
        if (info->Size == 0) break;
        offset += info->Size;
      }
      if (cores > 0) return cores;
    }
  }
#elif defined(__linux__)
  // One entry per distinct set of hyper-thread siblings.
  std::set<std::string> cores;
  for (unsigned cpu = 0; cpu < 4096; ++cpu) {
    std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
    if (!in) break;
    std::string line;
    if (std::getline(in, line)) cores.insert(line);
  }
  if (!cores.empty()) return static_cast<int>(cores.size());
#elif defined(__APPLE__)
  int cores = 0;
  std::size_t length = sizeof(cores);
  if (sysctlbyname("hw.physicalcpu", &cores, &length, nullptr, 0) == 0 && cores > 0) return cores;
#endif
  // Unknown topology: assume two hardware threads per core on larger machines.
  const int hw = hardware_threads();
  return hw <= 4 ? hw : hw / 2;
}

}  // namespace

int physical_cores() {
  static const int cores = std::max(1, detect_physical_cores());
  return cores;
}

int resolve_threads(int requested) {
  if (requested > 0) return requested;
  return std::clamp(physical_cores(), 1, 8);
}

std::size_t utf8_complete_prefix(std::string_view bytes) {
  // A sequence is at most four bytes long, so an incomplete one starts within
  // the last three bytes.
  const std::size_t n = bytes.size();
  const std::size_t window = std::min<std::size_t>(n, 3);
  for (std::size_t back = 1; back <= window; ++back) {
    const auto b = static_cast<unsigned char>(bytes[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: keep looking for its lead byte
    std::size_t length = 1;
    if ((b & 0xE0) == 0xC0) {
      length = 2;
    } else if ((b & 0xF0) == 0xE0) {
      length = 3;
    } else if ((b & 0xF8) == 0xF0) {
      length = 4;
    }
    return length > back ? n - back : n;
  }
  return n;
}

std::string_view utf8_trim_partial(std::string_view bytes) {
  std::size_t first = 0;
  while (first < bytes.size() && first < 3 && (static_cast<unsigned char>(bytes[first]) & 0xC0) == 0x80) ++first;
  bytes.remove_prefix(first);
  return bytes.substr(0, utf8_complete_prefix(bytes));
}

std::string file_stem(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  if (slash != std::string_view::npos) path.remove_prefix(slash + 1);
  const std::size_t dot = path.rfind('.');
  if (dot != std::string_view::npos && dot != 0) path = path.substr(0, dot);
  return std::string(path);
}

std::string to_upper(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

std::string to_lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

}  // namespace detail

}  // namespace prag::llm
