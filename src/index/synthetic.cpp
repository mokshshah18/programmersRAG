// Synthetic embeddings (see synthetic.hpp). Every row has its own generator,
// seeded from the seed of the set and the number of the row, so a row does not
// depend on how many rows are generated.
#include "prag/index/synthetic.hpp"

#include <algorithm>
#include <cmath>

#include "prag/core/hash.hpp"
#include "rng.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)  // results must not depend on the instruction set the compiler may use
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace prag::index {

namespace {

constexpr std::uint64_t kOffsetStream = 0x6f6666736574ULL;
constexpr std::uint64_t kCentreStream = 0x63656e747265ULL;
constexpr std::uint64_t kRowStream = 0x726f77ULL;
constexpr std::uint64_t kQueryStream = 0x7175657279ULL;

// Fills `out` with a direction drawn uniformly from the unit sphere.
void random_direction(Gaussian& normal, std::vector<double>& out) {
  double norm2 = 0.0;
  do {
    norm2 = 0.0;
    for (double& v : out) {
      v = normal.next();
      norm2 += v * v;
    }
  } while (!(norm2 > 0.0));
  const double inverse = 1.0 / std::sqrt(norm2);
  for (double& v : out) v *= inverse;
}

// Writes x / ||x|| as float32; a zero vector becomes the first basis vector.
void store_normalised(const std::vector<double>& x, float* out) {
  double norm2 = 0.0;
  for (const double v : x) norm2 += v * v;
  if (!(norm2 > 0.0)) {
    std::fill(out, out + x.size(), 0.0f);
    out[0] = 1.0f;
    return;
  }
  const double inverse = 1.0 / std::sqrt(norm2);
  for (std::size_t j = 0; j < x.size(); ++j) out[j] = static_cast<float>(x[j] * inverse);
}

}  // namespace

SyntheticData synthetic_embeddings(const SyntheticOptions& options) {
  SyntheticData data;
  const std::size_t dim = options.dim;
  if (dim == 0) return data;
  const std::size_t n = options.n;
  const std::size_t clusters = std::max<std::size_t>(1, options.clusters);
  data.n = n;
  data.dim = dim;
  data.rows.resize(n * dim);
  data.cluster.resize(n);
  data.centers.resize(clusters * dim);
  data.offset.resize(dim);

  std::vector<double> direction(dim);
  std::vector<double> offset(dim);
  {
    Gaussian normal(hash_combine(options.seed, kOffsetStream));
    random_direction(normal, offset);
    for (std::size_t j = 0; j < dim; ++j) data.offset[j] = static_cast<float>(offset[j]);
  }
  std::vector<double> centres(clusters * dim);
  for (std::size_t k = 0; k < clusters; ++k) {
    Gaussian normal(hash_combine(hash_combine(options.seed, kCentreStream), k));
    random_direction(normal, direction);
    for (std::size_t j = 0; j < dim; ++j) {
      centres[k * dim + j] = direction[j];
      data.centers[k * dim + j] = static_cast<float>(direction[j]);
    }
  }

  std::vector<double> x(dim);
  for (std::size_t i = 0; i < n; ++i) {
    SplitMix64 pick(hash_combine(hash_combine(options.seed, kRowStream), i));
    const std::size_t k = static_cast<std::size_t>(pick.below(clusters));
    Gaussian normal(pick.next());
    random_direction(normal, direction);
    const double* centre = centres.data() + k * dim;
    for (std::size_t j = 0; j < dim; ++j) {
      x[j] = options.offset_weight * offset[j] + options.cluster_weight * centre[j] +
             options.noise_weight * direction[j];
    }
    store_normalised(x, data.rows.data() + i * dim);
    data.cluster[i] = static_cast<std::uint32_t>(k);
  }
  return data;
}

std::vector<float> synthetic_queries(const SyntheticData& data, std::size_t count, double noise_weight,
                                     std::uint64_t seed, std::vector<std::uint32_t>* targets) {
  std::vector<float> queries;
  if (targets != nullptr) targets->clear();
  if (data.dim == 0 || data.n == 0) return queries;
  const std::size_t dim = data.dim;
  queries.resize(count * dim);
  if (targets != nullptr) targets->resize(count);

  std::vector<double> direction(dim);
  std::vector<double> x(dim);
  for (std::size_t i = 0; i < count; ++i) {
    SplitMix64 pick(hash_combine(hash_combine(seed, kQueryStream), i));
    const std::size_t target = static_cast<std::size_t>(pick.below(data.n));
    Gaussian normal(pick.next());
    random_direction(normal, direction);
    const float* row = data.rows.data() + target * dim;
    for (std::size_t j = 0; j < dim; ++j) x[j] = static_cast<double>(row[j]) + noise_weight * direction[j];
    store_normalised(x, queries.data() + i * dim);
    if (targets != nullptr) (*targets)[i] = static_cast<std::uint32_t>(target);
  }
  return queries;
}

std::vector<std::uint32_t> layout_by_cluster(const SyntheticData& data) {
  std::vector<std::uint32_t> layout(data.n);
  for (std::size_t i = 0; i < data.n; ++i) layout[i] = static_cast<std::uint32_t>(i);
  std::stable_sort(layout.begin(), layout.end(),
                   [&](std::uint32_t a, std::uint32_t b) { return data.cluster[a] < data.cluster[b]; });
  return layout;
}

}  // namespace prag::index
