// Agreement between two labelings: normalised mutual information and adjusted Rand index.
#include <algorithm>
#include <cmath>
#include <utility>

#include "prag/kde/directional.hpp"

namespace prag::kde {

namespace {

// Contingency table of two labelings in sparse form: the sizes of the clusters
// of each labeling and of their non-empty intersections.
struct Contingency {
  std::size_t n = 0;
  std::vector<std::size_t> rows;   // cluster sizes of the first labeling
  std::vector<std::size_t> columns;  // cluster sizes of the second labeling
  std::vector<std::size_t> cells;  // sizes of the non-empty intersections
};

std::vector<std::size_t> run_lengths(std::vector<std::uint32_t> labels) {
  std::sort(labels.begin(), labels.end());
  std::vector<std::size_t> sizes;
  for (std::size_t i = 0; i < labels.size();) {
    std::size_t j = i;
    while (j < labels.size() && labels[j] == labels[i]) ++j;
    sizes.push_back(j - i);
    i = j;
  }
  return sizes;
}

Contingency tabulate(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
  Contingency t;
  if (a.size() != b.size() || a.empty()) return t;
  t.n = a.size();
  t.rows = run_lengths(std::vector<std::uint32_t>(a.begin(), a.end()));
  t.columns = run_lengths(std::vector<std::uint32_t>(b.begin(), b.end()));
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs(t.n);
  for (std::size_t i = 0; i < t.n; ++i) pairs[i] = {a[i], b[i]};
  std::sort(pairs.begin(), pairs.end());
  for (std::size_t i = 0; i < pairs.size();) {
    std::size_t j = i;
    while (j < pairs.size() && pairs[j] == pairs[i]) ++j;
    t.cells.push_back(j - i);
    i = j;
  }
  return t;
}

// sum_c p_c log p_c with p_c = size_c / n, negated.
double entropy(const std::vector<std::size_t>& sizes, std::size_t n) {
  double h = 0.0;
  for (const std::size_t size : sizes) {
    const double p = static_cast<double>(size) / static_cast<double>(n);
    h -= p * std::log(p);
  }
  return h;
}

double pairs_within(const std::vector<std::size_t>& sizes) {
  double total = 0.0;
  for (const std::size_t size : sizes) total += 0.5 * static_cast<double>(size) * static_cast<double>(size - 1);
  return total;
}

}  // namespace

double normalized_mutual_information(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
  const Contingency t = tabulate(a, b);
  if (t.n == 0) return 0.0;
  const double ha = entropy(t.rows, t.n);
  const double hb = entropy(t.columns, t.n);
  if (t.rows.size() == 1 && t.columns.size() == 1) return 1.0;
  if (t.rows.size() == 1 || t.columns.size() == 1) return 0.0;  // one labeling carries no information
  // I(a; b) = H(a) + H(b) - H(a, b)
  const double mutual = std::max(0.0, ha + hb - entropy(t.cells, t.n));
  return std::min(1.0, 2.0 * mutual / (ha + hb));
}

double adjusted_rand_index(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
  const Contingency t = tabulate(a, b);
  if (t.n == 0) return 0.0;
  const double together = pairs_within(t.cells);  // pairs joined in both labelings
  const double in_a = pairs_within(t.rows);
  const double in_b = pairs_within(t.columns);
  const double all = 0.5 * static_cast<double>(t.n) * static_cast<double>(t.n - 1);
  if (all == 0.0) return 1.0;  // a single point
  const double expected = in_a * in_b / all;
  const double maximum = 0.5 * (in_a + in_b);
  if (maximum == expected) return 1.0;  // both partitions are one cluster, or both are all singletons
  return (together - expected) / (maximum - expected);
}

}  // namespace prag::kde
