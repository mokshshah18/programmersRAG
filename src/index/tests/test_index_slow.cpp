// Measurements of the scan: throughput by thread count and by block size.
// They take about a minute and 1.5 GiB of memory, so they run only when the
// environment variable PRAG_SLOW_TESTS is set to 1. Lines that start with
// "MEASURE" hold the numbers. Times are medians over the repetitions; the
// smallest time of the run inside the executor is given as well (min_ms),
// because other processes on the machine only ever add to a time.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "index_testing.hpp"
#include "prag/core/clock.hpp"
#include "prag/exec/cache_topology.hpp"

using namespace prag;
using namespace prag::index;
using namespace prag::index::testing;

namespace {

bool slow_tests_enabled() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

struct Timing {
  double call_ms = 0.0;  // the whole call: encoding of the query, scan, result
  double run_ms = 0.0;   // inside the executor: the blocks only
  double work_ms = 0.0;  // summed over the workers
  double gate_ms = 0.0;  // waiting at the admission gate, summed over the workers
  double min_ms = 0.0;   // smallest run_ms
};

// Median times of `repetitions` scans after one that is not counted.
Timing time_scan(const VectorStore& s, std::span<const float> queries, exec::OrderedExecutor& executor,
                 int repetitions, const std::vector<ScanResult>& expected) {
  std::vector<double> call, run, work, gate;
  for (int repeat = -1; repeat < repetitions; ++repeat) {
    const Stopwatch watch;
    const std::vector<ScanResult> results = scan_batch(s, queries, executor);
    const double elapsed = watch.elapsed_ms();
    REQUIRE(results.size() == expected.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
      // Estimates do not depend on threads or blocks; the statistics may differ in the last bits.
      CHECK(same_bits(results[i].scores, expected[i].scores));
      CHECK(same_bits(results[i].error, expected[i].error));
      CHECK_NEAR(results[i].mean, expected[i].mean, 1e-12);
    }
    CHECK_EQ(results.front().stats.admission_violations, std::uint64_t{0});
    CHECK_EQ(results.front().stats.commit_violations, std::uint64_t{0});
    if (repeat < 0) continue;
    call.push_back(elapsed);
    run.push_back(static_cast<double>(results.front().stats.wall_ns) / 1e6);
    work.push_back(static_cast<double>(results.front().stats.work_ns) / 1e6);
    gate.push_back(static_cast<double>(results.front().stats.gate_wait_ns) / 1e6);
  }
  return Timing{median(call), median(run), median(work), median(gate), *std::min_element(run.begin(), run.end())};
}

}  // namespace

PRAG_TEST(scan_throughput_and_block_size_sweep) {
  if (!slow_tests_enabled()) {
    std::printf("  skipped: set PRAG_SLOW_TESTS=1 to run the measurements\n");
    return;
  }
  const std::size_t n = 100000;
  const std::size_t dim = 1024;
  const std::size_t batch = 64;

  const exec::CacheTopology& topology = exec::cache_topology();
  std::printf("MEASURE machine logical_cpus=%u l1d_min_bytes=%zu l1d_per_thread_min_bytes=%zu l2_min_bytes=%zu "
              "recommended_block_bytes=%zu kernel=%s\n",
              topology.logical_cpus, topology.l1d_min_bytes, topology.l1d_per_thread_min_bytes, topology.l2_min_bytes,
              exec::recommended_block_bytes(topology), simd_kernel());

  Stopwatch watch;
  SyntheticOptions options;
  options.n = n;
  options.dim = dim;
  options.clusters = 64;
  options.seed = 2026;
  const SyntheticData data = synthetic_embeddings(options);
  const std::vector<std::uint32_t> layout = layout_by_cluster(data);
  const std::vector<float> queries = synthetic_queries(data, batch, 0.5, 9, nullptr);
  const std::span<const float> single(queries.data(), dim);
  std::printf("MEASURE corpus n=%zu dim=%zu clusters=%zu generate_s=%.1f\n", n, dim, options.clusters,
              watch.elapsed_s());

  exec::OrderedExecutor all(24);
  {
    // The exact scan of the float32 rows on one thread, for comparison.
    VectorStore s;
    BuildOptions build;
    build.codes = CodeKind::Int8;
    s.build(data.rows, dim, layout, data.cluster, build, all);
    std::vector<double> times;
    for (int repeat = -1; repeat < 5; ++repeat) {
      watch.reset();
      const std::vector<float> exact = exact_scores(s, single);
      const double elapsed = watch.elapsed_ms();
      CHECK_EQ(exact.size(), n);
      if (repeat >= 0) times.push_back(elapsed);
    }
    std::printf("MEASURE exact_float32 threads=1 ms=%.3f vectors_per_s=%.3e bytes_per_vector=%zu\n", median(times),
                static_cast<double>(n) / (median(times) / 1e3), dim * sizeof(float));
  }

  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    BuildOptions build;
    build.codes = kind;
    build.block_bytes = 16 * 1024;
    VectorStore s;
    watch.reset();
    s.build(data.rows, dim, layout, data.cluster, build, all);
    const double build_s = watch.elapsed_s();
    REQUIRE(s.size() == n);
    std::printf("MEASURE build codes=%s threads=24 seconds=%.2f bytes_per_vector=%zu code_bytes=%zu side_bytes=8 "
                "vectors_per_16KiB_block=%zu blocks=%zu\n",
                to_string(kind), build_s, s.bytes_per_code(), s.bytes_per_code() - 8,
                std::size_t{16 * 1024} / s.bytes_per_code(), s.blocks().size());

    exec::OrderedExecutor reference_executor(1);
    const std::vector<ScanResult> expected_single = scan_batch(s, single, reference_executor);
    const std::vector<ScanResult> expected_batch = scan_batch(s, queries, reference_executor);

    for (const unsigned threads : {1u, 8u, 24u}) {
      exec::OrderedExecutor executor(threads);
      const Timing t = time_scan(s, single, executor, 5, expected_single);
      std::printf("MEASURE throughput codes=%s threads=%u block_kib=16 queries=1 call_ms=%.3f run_ms=%.3f "
                  "min_ms=%.3f work_ms=%.3f gate_wait_ms=%.3f vectors_per_s_call=%.3e vectors_per_s_run=%.3e\n",
                  to_string(kind), threads, t.call_ms, t.run_ms, t.min_ms, t.work_ms, t.gate_ms,
                  static_cast<double>(n) / (t.call_ms / 1e3), static_cast<double>(n) / (t.run_ms / 1e3));
    }

    exec::OrderedExecutor eight(8);
    for (const std::size_t kib : {std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{32}, std::size_t{64},
                                  std::size_t{256}, std::size_t{1024}}) {
      s.reblock(kib * 1024);
      const Timing one = time_scan(s, single, eight, 3, expected_single);
      const Timing many = time_scan(s, queries, eight, 3, expected_batch);
      std::printf("MEASURE sweep codes=%s threads=8 block_kib=%zu blocks=%zu vectors_per_block=%zu | queries=1 "
                  "call_ms=%.3f run_ms=%.3f min_ms=%.3f work_ms=%.3f vectors_per_s_run=%.3e | queries=%zu "
                  "call_ms=%.3f run_ms=%.3f min_ms=%.3f work_ms=%.3f pairs_per_s_run=%.3e\n",
                  to_string(kind), kib, s.blocks().size(), s.blocks().front().count, one.call_ms, one.run_ms,
                  one.min_ms, one.work_ms, static_cast<double>(n) / (one.run_ms / 1e3), batch, many.call_ms,
                  many.run_ms, many.min_ms, many.work_ms, static_cast<double>(n * batch) / (many.run_ms / 1e3));
    }
  }
}
