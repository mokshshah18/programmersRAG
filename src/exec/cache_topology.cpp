#include "prag/exec/cache_topology.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>

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

namespace prag::exec {

namespace {

void add_class(std::vector<CacheClass>& classes, const CacheClass& seen) {
  for (CacheClass& c : classes) {
    if (c.level == seen.level && c.type == seen.type && c.size_bytes == seen.size_bytes &&
        c.sharing_threads == seen.sharing_threads) {
      ++c.instances;
      return;
    }
  }
  classes.push_back(seen);
  classes.back().instances = 1;
}

void summarise(CacheTopology& t) {
  std::size_t l1d_min = 0;
  std::size_t l1d_per_thread_min = 0;
  std::size_t l2_min = 0;
  for (const CacheClass& c : t.classes) {
    const bool holds_data = c.type == CacheType::Data || c.type == CacheType::Unified;
    if (!holds_data || c.size_bytes == 0) continue;
    if (c.level == 1) {
      const std::size_t per_thread = c.size_bytes / std::max(1u, c.sharing_threads);
      l1d_min = l1d_min == 0 ? c.size_bytes : std::min(l1d_min, c.size_bytes);
      l1d_per_thread_min = l1d_per_thread_min == 0 ? per_thread : std::min(l1d_per_thread_min, per_thread);
      if (c.line_bytes != 0) t.line_bytes = c.line_bytes;
    } else if (c.level == 2) {
      l2_min = l2_min == 0 ? c.size_bytes : std::min(l2_min, c.size_bytes);
    }
  }
  if (l1d_min != 0) {
    t.l1d_min_bytes = l1d_min;
    t.l1d_per_thread_min_bytes = l1d_per_thread_min;
    t.detected = true;
  }
  if (l2_min != 0) t.l2_min_bytes = l2_min;
  std::sort(t.classes.begin(), t.classes.end(), [](const CacheClass& a, const CacheClass& b) {
    if (a.level != b.level) return a.level < b.level;
    if (a.type != b.type) return static_cast<int>(a.type) < static_cast<int>(b.type);
    return a.size_bytes > b.size_bytes;
  });
}

#ifdef _WIN32

void query(CacheTopology& t) {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
  if (length == 0) return;
  std::vector<std::uint8_t> buffer(length);
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationCache, first, &length)) return;

  for (DWORD offset = 0; offset < length;) {
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
    if (info->Size == 0) break;
    if (info->Relationship == RelationCache) {
      const CACHE_RELATIONSHIP& cache = info->Cache;
      CacheClass c;
      c.level = cache.Level;
      c.type = cache.Type == CacheData          ? CacheType::Data
               : cache.Type == CacheInstruction ? CacheType::Instruction
                                                : CacheType::Unified;
      c.size_bytes = cache.CacheSize;
      c.line_bytes = cache.LineSize;
      c.associativity = cache.Associativity;
      c.sharing_threads = std::max(1, std::popcount(static_cast<std::uint64_t>(cache.GroupMask.Mask)));
      if (cache.Type != CacheTrace) add_class(t.classes, c);
    }
    offset += info->Size;
  }
}

#elif defined(__linux__)

std::string read_line(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  if (in) std::getline(in, line);
  return line;
}

// "0-3,8,10-11" -> 7
unsigned count_cpu_list(const std::string& list) {
  unsigned count = 0;
  std::stringstream ss(list);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) continue;
    const std::size_t dash = part.find('-');
    try {
      if (dash == std::string::npos) {
        (void)std::stoul(part);
        ++count;
      } else {
        const unsigned long lo = std::stoul(part.substr(0, dash));
        const unsigned long hi = std::stoul(part.substr(dash + 1));
        if (hi >= lo) count += static_cast<unsigned>(hi - lo + 1);
      }
    } catch (...) {
    }
  }
  return std::max(1u, count);
}

std::size_t parse_size(const std::string& s) {
  if (s.empty()) return 0;
  try {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(s, &consumed);
    const char unit = consumed < s.size() ? s[consumed] : '\0';
    if (unit == 'K' || unit == 'k') return static_cast<std::size_t>(value) * 1024;
    if (unit == 'M' || unit == 'm') return static_cast<std::size_t>(value) * 1024 * 1024;
    return static_cast<std::size_t>(value);
  } catch (...) {
    return 0;
  }
}

void query(CacheTopology& t) {
  // Each physical cache is reported once per sharing CPU; count an instance only
  // at the first CPU of its shared list.
  for (unsigned cpu = 0; cpu < 4096; ++cpu) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/";
    if (read_line(base + "index0/level").empty()) {
      if (cpu == 0) return;
      break;
    }
    for (unsigned index = 0; index < 16; ++index) {
      const std::string dir = base + "index" + std::to_string(index) + "/";
      const std::string level = read_line(dir + "level");
      if (level.empty()) break;
      const std::string shared = read_line(dir + "shared_cpu_list");
      unsigned first_cpu = cpu;
      try {
        first_cpu = static_cast<unsigned>(std::stoul(shared));
      } catch (...) {
      }
      if (first_cpu != cpu) continue;

      CacheClass c;
      try {
        c.level = static_cast<unsigned>(std::stoul(level));
      } catch (...) {
        continue;
      }
      const std::string type = read_line(dir + "type");
      c.type = type == "Data" ? CacheType::Data : type == "Instruction" ? CacheType::Instruction : CacheType::Unified;
      c.size_bytes = parse_size(read_line(dir + "size"));
      c.line_bytes = parse_size(read_line(dir + "coherency_line_size"));
      if (c.line_bytes == 0) c.line_bytes = 64;
      c.associativity = static_cast<unsigned>(parse_size(read_line(dir + "ways_of_associativity")));
      c.sharing_threads = count_cpu_list(shared);
      add_class(t.classes, c);
    }
  }
}

#elif defined(__APPLE__)

std::size_t sysctl_size(const char* name) {
  std::uint64_t value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return 0;
  return static_cast<std::size_t>(value);
}

void query(CacheTopology& t) {
  const std::size_t line = sysctl_size("hw.cachelinesize");
  const char* l1_names[] = {"hw.perflevel0.l1dcachesize", "hw.perflevel1.l1dcachesize", "hw.l1dcachesize"};
  const char* l2_names[] = {"hw.perflevel0.l2cachesize", "hw.perflevel1.l2cachesize", "hw.l2cachesize"};
  for (const char* name : l1_names) {
    const std::size_t size = sysctl_size(name);
    if (size == 0) continue;
    CacheClass c;
    c.level = 1;
    c.type = CacheType::Data;
    c.size_bytes = size;
    c.line_bytes = line == 0 ? 64 : line;
    add_class(t.classes, c);
  }
  for (const char* name : l2_names) {
    const std::size_t size = sysctl_size(name);
    if (size == 0) continue;
    CacheClass c;
    c.level = 2;
    c.type = CacheType::Unified;
    c.size_bytes = size;
    c.line_bytes = line == 0 ? 64 : line;
    add_class(t.classes, c);
  }
}

#else

void query(CacheTopology&) {}

#endif

}  // namespace

CacheTopology detect_cache_topology() {
  CacheTopology t;
  const unsigned hw = std::thread::hardware_concurrency();
  t.logical_cpus = hw == 0 ? 1 : hw;
  try {
    query(t);
    summarise(t);
  } catch (...) {
    t = CacheTopology{};
    t.logical_cpus = hw == 0 ? 1 : hw;
  }
  return t;
}

const CacheTopology& cache_topology() {
  static const CacheTopology topology = detect_cache_topology();
  return topology;
}

std::size_t recommended_block_bytes(const CacheTopology& topology, double fraction) {
  constexpr std::size_t kKiB = 1024;
  constexpr std::size_t kMin = 4 * kKiB;
  constexpr std::size_t kMax = 64 * kKiB;
  if (!(fraction > 0.0)) fraction = 0.5;
  if (fraction > 1.0) fraction = 1.0;
  const double by_capacity = fraction * static_cast<double>(topology.l1d_min_bytes);
  const double by_share = static_cast<double>(topology.l1d_per_thread_min_bytes);
  const double raw = std::min(by_capacity, by_share);
  // The epsilon absorbs floating-point rounding of fraction * capacity.
  const std::size_t kib = static_cast<std::size_t>(raw / static_cast<double>(kKiB) + 1e-9);
  return std::clamp(kib * kKiB, kMin, kMax);
}

std::string describe(const CacheTopology& t) {
  std::string out;
  char line[256];
  std::snprintf(line, sizeof(line), "logical CPUs: %u, cache line: %zu B, detected: %s\n", t.logical_cpus,
                t.line_bytes, t.detected ? "yes" : "no (defaults)");
  out += line;
  for (const CacheClass& c : t.classes) {
    const char* type = c.type == CacheType::Data ? "data" : c.type == CacheType::Instruction ? "instr" : "unified";
    std::snprintf(line, sizeof(line), "L%u %-7s %7zu KiB  x%-3u shared by %u thread(s), %u-way\n", c.level, type,
                  c.size_bytes / 1024, c.instances, c.sharing_threads, c.associativity);
    out += line;
  }
  std::snprintf(line, sizeof(line), "L1d per thread (min): %zu KiB -> block size %zu KiB\n",
                t.l1d_per_thread_min_bytes / 1024, recommended_block_bytes(t) / 1024);
  out += line;
  return out;
}

}  // namespace prag::exec
