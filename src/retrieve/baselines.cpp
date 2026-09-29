#include "prag/retrieve/baselines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ranking.hpp"

namespace prag::retrieve {

namespace {

using detail::ranks_before;
using detail::scored_unit;
using detail::Standardiser;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

double non_negative(double weight) noexcept { return weight > 0.0 && std::isfinite(weight) ? weight : 0.0; }

// Number of ids below n. Targets are sorted ascending, so they form a prefix.
std::size_t count_below(std::span<const UnitId> ids, std::size_t n) noexcept {
  const auto end = std::lower_bound(ids.begin(), ids.end(), n, [](UnitId id, std::size_t bound) { return id < bound; });
  return static_cast<std::size_t>(end - ids.begin());
}

}  // namespace

std::vector<TracedUnit> largest_gap_cutoff(std::span<const float> scores, double mean, double stddev,
                                           const Calibration& calibration, std::size_t k_max, std::size_t buffer) {
  std::vector<TracedUnit> out;
  std::vector<UnitId> order;
  order.reserve(scores.size());
  for (std::size_t v = 0; v < scores.size(); ++v) {
    if (!std::isnan(scores[v])) order.push_back(static_cast<UnitId>(v));
  }
  const std::size_t n = order.size();
  if (n == 0 || k_max == 0) return out;

  // Positions 1 .. last are candidates for the cut; position i needs s_(i+1).
  const std::size_t last = std::min(k_max, n - 1);
  const std::size_t extra = std::min(buffer, n);  // also keeps the sums below from overflowing
  const std::size_t sorted = std::min(n, last + 1 + extra);
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(sorted), order.end(),
                    [&](UnitId a, UnitId b) { return ranks_before(scores[a], a, scores[b], b); });

  std::size_t cut = 1;
  float widest = -1.0f;
  for (std::size_t i = 1; i <= last; ++i) {
    const float gap = scores[order[i - 1]] - scores[order[i]];
    if (gap > widest) {  // strict: the first of equal gaps wins, and an undefined gap never does
      widest = gap;
      cut = i;
    }
  }

  const Standardiser standardise(mean, stddev);
  const std::size_t keep = std::min(n, cut + extra);
  out.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    out.push_back(scored_unit(order[i], scores[order[i]], standardise, calibration));
  }
  return out;
}

std::vector<double> personalized_pagerank(const graph::CallGraph& calls, std::span<const double> seeds,
                                          const PageRankOptions& options, PageRankStats* stats) {
  PageRankStats local;
  const std::size_t n = seeds.size();
  std::vector<double> restart(n, 0.0);
  if (n == 0) {
    if (stats != nullptr) *stats = local;
    return restart;
  }

  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    restart[i] = non_negative(seeds[i]);
    total += restart[i];
  }
  if (total > 0.0 && std::isfinite(total)) {
    for (double& p : restart) p /= total;
  } else {
    std::fill(restart.begin(), restart.end(), 1.0 / static_cast<double>(n));
  }

  const double alpha = std::isnan(options.continuation) ? 0.5 : std::clamp(options.continuation, 0.0, 0.999999);
  const double forward = non_negative(options.callee_weight);
  const double backward = non_negative(options.caller_weight);

  std::vector<double> out_weight(n);
  for (std::size_t u = 0; u < n; ++u) {
    const UnitId id = static_cast<UnitId>(u);
    out_weight[u] = forward * static_cast<double>(count_below(calls.callees(id), n)) +
                    backward * static_cast<double>(count_below(calls.callers(id), n));
  }

  std::vector<double> rank = restart;
  std::vector<double> next(n);
  for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
    for (std::size_t i = 0; i < n; ++i) next[i] = (1.0 - alpha) * restart[i];
    double dangling = 0.0;
    for (std::size_t u = 0; u < n; ++u) {
      if (rank[u] == 0.0) continue;
      if (!(out_weight[u] > 0.0)) {
        dangling += rank[u];
        continue;
      }
      const UnitId id = static_cast<UnitId>(u);
      const double share = alpha * rank[u] / out_weight[u];
      if (forward > 0.0) {
        for (const UnitId v : calls.callees(id)) {
          if (v < n) next[v] += share * forward;
        }
      }
      if (backward > 0.0) {
        for (const UnitId v : calls.callers(id)) {
          if (v < n) next[v] += share * backward;
        }
      }
    }
    if (dangling > 0.0) {
      for (std::size_t i = 0; i < n; ++i) next[i] += alpha * dangling * restart[i];
    }

    double change = 0.0;
    for (std::size_t i = 0; i < n; ++i) change += std::fabs(next[i] - rank[i]);
    rank.swap(next);
    local.iterations = iteration + 1;
    local.change = change;
    if (change < options.tolerance) {
      local.converged = true;
      break;
    }
  }

  // The iteration preserves the total mass up to rounding; remove the rounding.
  double sum = 0.0;
  for (const double r : rank) sum += r;
  if (sum > 0.0) {
    for (double& r : rank) r /= sum;
  }
  if (stats != nullptr) *stats = local;
  return rank;
}

std::vector<TracedUnit> top_k_by_rank(std::span<const double> rank, std::span<const float> scores, double mean,
                                      double stddev, const Calibration& calibration, std::size_t k) {
  const std::size_t n = std::min(rank.size(), scores.size());
  const std::size_t keep = std::min(k, n);
  std::vector<TracedUnit> out;
  if (keep == 0) return out;

  std::vector<UnitId> order(n);
  for (std::size_t v = 0; v < n; ++v) order[v] = static_cast<UnitId>(v);
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(),
                    [&](UnitId a, UnitId b) { return ranks_before(rank[a], a, rank[b], b); });

  const Standardiser standardise(mean, stddev);
  out.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    out.push_back(scored_unit(order[i], scores[order[i]], standardise, calibration));
  }
  return out;
}

double conformal_threshold(std::span<const double> z_relevant, double alpha) {
  if (std::isnan(alpha)) return -kInfinity;
  if (alpha >= 1.0) return kInfinity;

  std::vector<double> sorted;
  sorted.reserve(z_relevant.size());
  for (const double z : z_relevant) {
    if (std::isfinite(z)) sorted.push_back(z);
  }
  const std::size_t n = sorted.size();
  if (n == 0 || alpha <= 0.0) return -kInfinity;

  const double product = static_cast<double>(n + 1) * (1.0 - alpha);
  const double needed = std::ceil(product - 1e-9 * std::max(1.0, product));
  if (needed > static_cast<double>(n)) return -kInfinity;
  if (needed < 1.0) return kInfinity;
  const std::size_t k = static_cast<std::size_t>(needed);

  // The k-th largest score.
  const auto kth = sorted.begin() + static_cast<std::ptrdiff_t>(k - 1);
  std::nth_element(sorted.begin(), kth, sorted.end(), [](double a, double b) { return a > b; });
  return *kth;
}

}  // namespace prag::retrieve
