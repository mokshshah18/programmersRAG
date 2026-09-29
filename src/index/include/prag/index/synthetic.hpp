// Synthetic embeddings with the geometry of language-model embeddings.
//
// Embeddings produced by language models are not spread evenly over the
// sphere: they share a common direction (anisotropy, mean pairwise cosine well
// above zero) and they form groups. The generator reproduces both:
//
//     x = normalise( w_offset * u0  +  w_cluster * m_k  +  w_noise * g / ||g|| )
//
// with one common unit vector u0, a unit vector m_k per cluster and isotropic
// Gaussian noise g. In high dimension the three parts are almost orthogonal,
// so the cosine of two points is close to w_offset^2 / W for different
// clusters and (w_offset^2 + w_cluster^2) / W inside a cluster, with
// W = w_offset^2 + w_cluster^2 + w_noise^2.
//
// Everything is a function of the seed; no model is needed. Used by the tests
// and by the throughput experiments.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prag::index {

struct SyntheticOptions {
  std::size_t n = 1000;
  std::size_t dim = 1024;
  std::size_t clusters = 32;
  double offset_weight = 0.70;
  double cluster_weight = 0.55;
  double noise_weight = 0.45;
  std::uint64_t seed = 1;
};

struct SyntheticData {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<float> rows;             // n * dim, L2-normalised
  std::vector<std::uint32_t> cluster;  // cluster of each row
  std::vector<float> centers;          // clusters * dim, unit vectors
  std::vector<float> offset;           // dim, the common direction
};

SyntheticData synthetic_embeddings(const SyntheticOptions& options);

// Queries that ask for a particular row: query i is
// normalise(row(target_i) + noise_weight * g / ||g||) with target_i drawn
// uniformly, so its cosine with the target is about 1 / sqrt(1 + noise_weight^2).
// Returns count * dim floats; `targets`, when given, receives the chosen rows.
std::vector<float> synthetic_queries(const SyntheticData& data, std::size_t count, double noise_weight,
                                     std::uint64_t seed, std::vector<std::uint32_t>* targets = nullptr);

// Storage order that lists the rows cluster by cluster (stable inside a
// cluster), the way kde::layout_by_mode orders units by density mode.
std::vector<std::uint32_t> layout_by_cluster(const SyntheticData& data);

}  // namespace prag::index
