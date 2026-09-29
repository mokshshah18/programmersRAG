// Shared by the tests of the index module: corpora, reference computations in
// double precision, and the statistics of the estimation error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "prag/exec/ordered_executor.hpp"
#include "prag/index/synthetic.hpp"
#include "prag/index/vector_store.hpp"

namespace prag::index::testing {

inline exec::OrderedExecutor& pool() {
  static exec::OrderedExecutor executor(8);
  return executor;
}

struct Corpus {
  SyntheticData data;
  std::vector<std::uint32_t> layout;  // cluster by cluster
  std::vector<float> queries;         // noisy copies of rows
  std::vector<std::uint32_t> targets;
  std::size_t query_count = 0;

  std::span<const float> query(std::size_t i) const {
    return {queries.data() + i * data.dim, data.dim};
  }
};

inline Corpus make_corpus(std::size_t n, std::size_t dim, std::size_t queries, std::uint64_t seed) {
  Corpus corpus;
  SyntheticOptions options;
  options.n = n;
  options.dim = dim;
  options.clusters = 32;
  options.seed = seed;
  corpus.data = synthetic_embeddings(options);
  corpus.layout = layout_by_cluster(corpus.data);
  corpus.queries = synthetic_queries(corpus.data, queries, 0.5, seed + 1000, &corpus.targets);
  corpus.query_count = queries;
  return corpus;
}

// The corpus used by most tests: 2000 rows, 100 queries, one per dimension.
inline const Corpus& corpus(std::size_t dim) {
  static std::map<std::size_t, Corpus> cache;
  auto found = cache.find(dim);
  if (found == cache.end()) found = cache.emplace(dim, make_corpus(2000, dim, 100, 11 + dim)).first;
  return found->second;
}

inline VectorStore build_store(const Corpus& corpus, const BuildOptions& options,
                               exec::OrderedExecutor& executor = pool()) {
  VectorStore store;
  store.build(corpus.data.rows, corpus.data.dim, corpus.layout, corpus.data.cluster, options, executor);
  return store;
}

// Stores of the shared corpora with default options (16 KiB blocks, seed 1).
inline const VectorStore& store(std::size_t dim, CodeKind kind) {
  static std::map<std::pair<std::size_t, int>, VectorStore> cache;
  const auto key = std::make_pair(dim, static_cast<int>(kind));
  auto found = cache.find(key);
  if (found == cache.end()) {
    BuildOptions options;
    options.codes = kind;
    found = cache.emplace(key, build_store(corpus(dim), options)).first;
  }
  return found->second;
}

// Inner product in double precision, summed in index order.
inline double exact_dot(std::span<const float> a, std::span<const float> b) {
  double sum = 0.0;
  for (std::size_t j = 0; j < a.size(); ++j) sum += static_cast<double>(a[j]) * static_cast<double>(b[j]);
  return sum;
}

inline bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

inline bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

inline bool same_scan(const ScanResult& a, const ScanResult& b) {
  return same_bits(a.scores, b.scores) && same_bits(a.error, b.error) && same_bits(a.mean, b.mean) &&
         same_bits(a.stddev, b.stddev);
}

struct ErrorStatistics {
  std::size_t pairs = 0;
  double mean = 0.0;             // of (estimate - exact)
  double stddev = 0.0;           // of (estimate - exact)
  double naive_se = 0.0;         // stddev / sqrt(pairs): pairs taken as independent
  double clustered_se = 0.0;     // two-way cluster-robust, see measure_errors
  double coverage = 0.0;         // fraction of pairs with |estimate - exact| <= half-width
  double mean_half_width = 0.0;
  double max_half_width = 0.0;
  double max_abs_error = 0.0;
  double max_ratio = 0.0;        // largest |estimate - exact| / half-width
  double rms_ratio = 0.0;        // root mean square of (estimate - exact) / half-width
};

// Errors of scan_batch against cosines computed in double precision from the
// float32 rows, for every (query, unit) pair.
//
// The standard error of the mean. Pairs are not independent: the errors of one
// unit share its code, the errors of one query share the query, and units of
// one group have similar codes, as queries aimed at one group are similar.
// `unit_group` and `query_group` name the group of every unit and of every
// query (empty: each unit, each query, is a group of its own). The two-way
// cluster-robust variance (Cameron, Gelbach and Miller 2011) is
//     V = V_a + V_b - V_ab,     V_x = sum over the clusters g of x of (sum of r over g)^2 / N^2,
// with a the groups of units, b the groups of queries, ab their intersections,
// r = error - mean and N the number of pairs. The variance for independent
// pairs is used when it is larger.
inline ErrorStatistics measure_errors(const VectorStore& store, std::span<const float> queries,
                                      std::span<const std::uint32_t> unit_group = {},
                                      std::span<const std::uint32_t> query_group = {},
                                      exec::OrderedExecutor& executor = pool()) {
  ErrorStatistics out;
  const std::size_t dim = store.dim();
  const std::size_t n = store.size();
  if (dim == 0 || n == 0) return out;
  const std::size_t count = queries.size() / dim;
  if (!unit_group.empty() && unit_group.size() != n) return out;
  if (!query_group.empty() && query_group.size() != count) return out;
  const std::vector<ScanResult> results = scan_batch(store, queries, executor);
  if (results.size() != count) return out;

  std::vector<double> errors(n * count);
  double sum = 0.0;
  double width_sum = 0.0;
  double ratio_sum2 = 0.0;
  std::size_t covered = 0;
  for (std::size_t j = 0; j < count; ++j) {
    const std::span<const float> q = queries.subspan(j * dim, dim);
    for (std::size_t i = 0; i < n; ++i) {
      const double exact = exact_dot(q, store.vector(static_cast<UnitId>(i)));
      const double error = static_cast<double>(results[j].scores[i]) - exact;
      const double width = results[j].error[i];
      errors[j * n + i] = error;
      sum += error;
      width_sum += width;
      if (std::fabs(error) <= width) ++covered;
      out.max_half_width = std::max(out.max_half_width, width);
      out.max_abs_error = std::max(out.max_abs_error, std::fabs(error));
      if (width > 0.0) {
        out.max_ratio = std::max(out.max_ratio, std::fabs(error) / width);
        ratio_sum2 += (error / width) * (error / width);
      }
    }
  }
  const double pairs = static_cast<double>(n * count);
  out.pairs = n * count;
  out.mean = sum / pairs;
  out.coverage = static_cast<double>(covered) / pairs;
  out.mean_half_width = width_sum / pairs;
  out.rms_ratio = std::sqrt(ratio_sum2 / pairs);

  std::map<std::uint32_t, double> by_unit;
  std::map<std::uint32_t, double> by_query;
  std::map<std::pair<std::uint32_t, std::uint32_t>, double> by_both;
  double residual2 = 0.0;
  for (std::size_t j = 0; j < count; ++j) {
    const std::uint32_t b = query_group.empty() ? static_cast<std::uint32_t>(j) : query_group[j];
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t a = unit_group.empty() ? static_cast<std::uint32_t>(i) : unit_group[i];
      const double r = errors[j * n + i] - out.mean;
      by_unit[a] += r;
      by_query[b] += r;
      by_both[{a, b}] += r;
      residual2 += r * r;
    }
  }
  double unit2 = 0.0;
  double query2 = 0.0;
  double both2 = 0.0;
  for (const auto& [group, sum_r] : by_unit) unit2 += sum_r * sum_r;
  for (const auto& [group, sum_r] : by_query) query2 += sum_r * sum_r;
  for (const auto& [group, sum_r] : by_both) both2 += sum_r * sum_r;
  out.stddev = std::sqrt(residual2 / std::max(1.0, pairs - 1.0));
  out.naive_se = out.stddev / std::sqrt(pairs);
  out.clustered_se = std::sqrt(std::max(unit2 + query2 - both2, residual2)) / pairs;
  return out;
}

// The k best units: score descending, unit id ascending among equal scores.
inline std::vector<UnitId> top_k(const std::vector<float>& scores, std::size_t k) {
  std::vector<UnitId> order(scores.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<UnitId>(i);
  k = std::min(k, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                    [&](UnitId a, UnitId b) { return scores[a] != scores[b] ? scores[a] > scores[b] : a < b; });
  order.resize(k);
  return order;
}

// Units whose interval [score - error, score + error] reaches `threshold`.
inline std::vector<UnitId> reaching(const ScanResult& result, float threshold) {
  std::vector<UnitId> units;
  for (std::size_t i = 0; i < result.scores.size(); ++i) {
    if (result.scores[i] + result.error[i] >= threshold) units.push_back(static_cast<UnitId>(i));
  }
  return units;
}

}  // namespace prag::index::testing
