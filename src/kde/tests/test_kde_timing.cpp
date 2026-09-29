// Wall-time measurements of the kde module. They take a few minutes and only
// run when the environment variable PRAG_SLOW_TESTS is set to 1; otherwise every
// case reports that it was skipped and passes.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "prag/core/clock.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/kde/directional.hpp"
#include "sampling.hpp"

using namespace prag::kde;
using namespace prag::kde::testing;
using prag::exec::OrderedExecutor;

namespace {

constexpr std::size_t kKiB = 1024;
constexpr std::size_t kPoints = 2000;
constexpr std::size_t kDimension = 32;
constexpr int kRepetitions = 3;

bool enabled() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  const bool on = value != nullptr && value[0] == '1';
  if (!on) std::printf("  skipped: set PRAG_SLOW_TESTS=1 to run the measurements\n");
  return on;
}

// 2000 points in 32 dimensions: twelve components of unequal weight and
// concentration, the shape of a corpus with function families of different
// size and tightness.
Sample corpus() {
  Rng rng(2026);
  Mixture mixture = separated_mixture(rng, kDimension, 12, 120.0);
  double total = 0.0;
  for (std::size_t c = 0; c < mixture.means.size(); ++c) {
    mixture.kappas[c] = 120.0 * std::pow(4.0, 2.0 * rng.uniform() - 1.0);
    mixture.weights[c] = 0.3 + rng.uniform();
    total += mixture.weights[c];
  }
  for (double& w : mixture.weights) w /= total;
  return sample_mixture(rng, mixture, kPoints, true);
}

template <class Body>
double median_ms(Body&& body) {
  std::vector<double> times;
  for (int r = 0; r < kRepetitions; ++r) {
    prag::Stopwatch watch;
    body();
    times.push_back(watch.elapsed_ms());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

}  // namespace

PRAG_TEST(wall_time_by_thread_count) {
  if (!enabled()) return;
  const Sample sample = corpus();
  const prag::exec::CacheTopology& topology = prag::exec::cache_topology();
  std::printf("  %s\n", prag::exec::describe(topology).c_str());
  std::printf("  recommended block: %zu KiB; n = %zu, m = %zu, median of %d runs, block 16 KiB\n",
              prag::exec::recommended_block_bytes(topology) / kKiB, kPoints, kDimension, kRepetitions);

  OrderedExecutor setup(8);
  const double cross_validated = select_kappa(sample.rows, kDimension, setup, 16 * kKiB).kappa;
  const double neighbourhood = neighbourhood_kappa(sample.rows, kDimension, setup, 16 * kKiB);
  std::printf("  cross-validated kappa %.3f, neighbourhood kappa %.3f\n", cross_validated, neighbourhood);

  MeanShiftOptions fixed;
  fixed.kappa = std::min(cross_validated, neighbourhood);
  const Modes reference = find_modes(sample.rows, kDimension, fixed, setup);
  double mean_steps = 0.0;
  for (const std::uint16_t steps : reference.iterations) mean_steps += steps;
  mean_steps /= static_cast<double>(kPoints);
  std::printf("  find_modes at kappa %.3f: %zu modes, adjusted Rand index %.4f, %.1f steps per point, %zu not converged\n",
              fixed.kappa, reference.count(), adjusted_rand_index(reference.assignment, sample.labels), mean_steps,
              reference.not_converged);

  std::printf("  threads   select_kappa   neighbourhood_kappa   find_modes(kappa given)   find_modes(automatic)\n");
  for (const unsigned threads : {1u, 8u, 24u}) {
    OrderedExecutor executor(threads);
    const double select_ms = median_ms([&] {
      CHECK_EQ(select_kappa(sample.rows, kDimension, executor, 16 * kKiB).kappa, cross_validated);
    });
    const double neighbourhood_ms = median_ms([&] {
      CHECK_EQ(neighbourhood_kappa(sample.rows, kDimension, executor, 16 * kKiB), neighbourhood);
    });
    const double modes_ms = median_ms([&] {
      const Modes modes = find_modes(sample.rows, kDimension, fixed, executor);
      CHECK(modes.assignment == reference.assignment);
    });
    const double automatic_ms = median_ms([&] {
      const Modes modes = find_modes(sample.rows, kDimension, MeanShiftOptions{}, executor);
      CHECK(modes.assignment == reference.assignment);
    });
    std::printf("  %7u   %9.1f ms   %16.1f ms   %20.1f ms   %18.1f ms\n", threads, select_ms, neighbourhood_ms,
                modes_ms, automatic_ms);
  }
}

PRAG_TEST(block_size_sweep) {
  if (!enabled()) return;
  const Sample sample = corpus();
  OrderedExecutor executor(8);
  MeanShiftOptions options;
  options.kappa = std::min(select_kappa(sample.rows, kDimension, executor, 16 * kKiB).kappa,
                           neighbourhood_kappa(sample.rows, kDimension, executor, 16 * kKiB));
  const Modes reference = find_modes(sample.rows, kDimension, options, executor);
  std::printf("  find_modes, 8 threads, n = %zu, m = %zu, kappa %.3f, median of %d runs\n", kPoints, kDimension,
              options.kappa, kRepetitions);
  std::printf("  block      tile rows   seeds per batch   batches   wall time\n");
  for (const std::size_t kib : {4u, 8u, 16u, 32u, 64u, 256u, 1024u}) {
    options.block_bytes = kib * kKiB;
    const double ms = median_ms([&] {
      const Modes modes = find_modes(sample.rows, kDimension, options, executor);
      CHECK(modes.assignment == reference.assignment);
    });
    // The layout rule of the module: half a block of data rows, half a block of seed state.
    const std::size_t tile_rows = std::max<std::size_t>(1, options.block_bytes / 2 / (kDimension * sizeof(float)));
    const std::size_t seed_bytes = 2 * kDimension * sizeof(double) + 3 * sizeof(double) + 2 * sizeof(std::uint32_t) + 2;
    const std::size_t seeds = std::max<std::size_t>(1, (options.block_bytes - options.block_bytes / 2) / seed_bytes);
    std::printf("  %4zu KiB   %9zu   %15zu   %7zu   %7.1f ms\n", kib, std::min(tile_rows, kPoints),
                std::min(seeds, kPoints), (kPoints + seeds - 1) / seeds, ms);
  }
}

PRAG_TEST(reduction_wall_time) {
  if (!enabled()) return;
  // 2000 rows of dimension 1024 (the size of the embedding model), reduced to 32.
  Rng rng(7);
  const std::size_t d = 1024;
  std::vector<double> scales;
  for (int j = 0; j < 60; ++j) scales.push_back(3.0 / std::sqrt(1.0 + j));
  const std::vector<double> directions = orthonormal_rows(rng, scales.size(), d);
  std::vector<float> rows(kPoints * d);
  for (std::size_t i = 0; i < kPoints; ++i) {
    std::vector<double> x(d, 0.0);
    x[0] = 2.0;  // common direction
    for (std::size_t j = 0; j < scales.size(); ++j) {
      const double z = rng.normal() * scales[j];
      for (std::size_t t = 0; t < d; ++t) x[t] += z * directions[j * d + t];
    }
    double norm2 = 0.0;
    for (std::size_t t = 0; t < d; ++t) {
      x[t] += 0.05 * rng.normal();
      norm2 += x[t] * x[t];
    }
    for (std::size_t t = 0; t < d; ++t) rows[i * d + t] = static_cast<float>(x[t] / std::sqrt(norm2));
  }
  Reduction reduction;
  const double fit_ms = median_ms([&] { reduction = fit_reduction(rows, d, 32, 0, 1); });
  std::vector<float> reduced;
  const double reduce_ms = median_ms([&] { reduced = reduce(reduction, rows); });
  const double anisotropy_ms = median_ms([&] { CHECK(mean_pairwise_cosine(rows, d) > 0.0); });
  double hubness = 0.0;
  const double hubness_ms = median_ms([&] { hubness = hubness_skewness(rows, d, 10); });
  std::printf("  n = %zu, in_dim = %zu, out_dim = %zu (single thread, median of %d runs)\n", kPoints, d,
              reduction.out_dim, kRepetitions);
  std::printf("  fit_reduction %.1f ms, reduce %.1f ms, mean_pairwise_cosine %.2f ms, hubness_skewness %.1f ms\n", fit_ms,
              reduce_ms, anisotropy_ms, hubness_ms);
  std::printf("  explained variance %.4f; mean pairwise cosine %.4f before, %.4f after; skewness of N_10 %.3f before, "
              "%.3f after\n",
              reduction.explained_variance_ratio(), mean_pairwise_cosine(rows, d),
              mean_pairwise_cosine(reduced, reduction.out_dim), hubness,
              hubness_skewness(reduced, reduction.out_dim, 10));
  CHECK_EQ(reduction.out_dim, std::size_t{32});
}
