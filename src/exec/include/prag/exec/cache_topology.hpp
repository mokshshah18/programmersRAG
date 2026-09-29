// Cache topology discovery and the block-size rule derived from it.
//
// User-space code cannot place data in a particular cache level. What it can do
// is keep the working set of each thread below the L1 data capacity available
// to that thread, so that after the first pass over a block every further
// access is an L1 hit. The block size is therefore chosen as
//
//     B = floor_kib( min( fraction * min_c L1d(c),  min_c L1d(c) / smt(c) ) )
//
// over the core classes c of the machine, where smt(c) is the number of logical
// CPUs sharing one L1d. The first term keeps the block within `fraction` (one
// half by default) of the smallest L1d, leaving the rest for the query, the
// outputs and the stack; the second keeps it within the share of one logical
// CPU when its hyper-thread sibling is busy. On the reference machine (8 P-cores:
// 48 KiB shared by 2 hyper-threads, 8 E-cores: 32 KiB unshared) this gives
// min(32 / 2, min(24, 32)) KiB = 16 KiB.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace prag::exec {

enum class CacheType { Data, Instruction, Unified };

struct CacheClass {
  unsigned level = 0;            // 1, 2 or 3
  CacheType type = CacheType::Data;
  std::size_t size_bytes = 0;
  std::size_t line_bytes = 64;
  unsigned associativity = 0;    // 0 = unknown, 0xFF = fully associative
  unsigned sharing_threads = 1;  // logical CPUs sharing one instance
  unsigned instances = 0;        // number of caches of this class in the machine
};

struct CacheTopology {
  bool detected = false;  // false when the OS query failed and defaults are reported
  unsigned logical_cpus = 1;
  std::size_t line_bytes = 64;
  std::size_t l1d_min_bytes = 32 * 1024;             // smallest L1d of any core class
  std::size_t l1d_per_thread_min_bytes = 16 * 1024;  // min over classes of L1d / sharing_threads
  std::size_t l2_min_bytes = 256 * 1024;
  std::vector<CacheClass> classes;  // distinct (level, type, size, sharing) combinations
};

// Queries the operating system (GetLogicalProcessorInformationEx on Windows,
// sysfs on Linux, sysctl on macOS). Never throws; falls back to conservative defaults.
CacheTopology detect_cache_topology();

// Cached result of detect_cache_topology().
const CacheTopology& cache_topology();

// Block size in bytes from the rule above, rounded down to a whole number of
// KiB and clamped to [4 KiB, 64 KiB].
std::size_t recommended_block_bytes(const CacheTopology& topology, double fraction = 0.5);

std::string describe(const CacheTopology& topology);

}  // namespace prag::exec
