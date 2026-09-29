// Kernel density estimation and mode seeking for directional data.
//
// Data are unit vectors x_1..x_n in R^m. The estimator uses the von Mises-Fisher
// kernel with concentration kappa (bandwidth h = kappa^{-1/2}):
//
//   fhat(x) = (1/n) * sum_i C_m(kappa) * exp(kappa * <x_i, x>).
//
// Modes of fhat are found with the directional mean-shift fixed-point iteration
//
//   x <- sum_i w_i x_i / || sum_i w_i x_i ||,     w_i = exp(kappa * (<x_i, x> - 1)),
//
// started from every data point. Points whose iterations end at the same mode
// form one group. Kernel sums run over the data in cache-sized blocks that are
// committed in order, so every result is bit-identical for any thread count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "prag/exec/ordered_executor.hpp"

namespace prag::kde {

// ---- dimensionality reduction ----------------------------------------------

// Centre on the corpus mean, project on principal directions, re-normalise to
// the sphere. Centring (and optionally discarding the few dominant directions)
// counters the anisotropy of language-model embeddings, which otherwise pulls
// every density estimate toward the common mean direction.
struct Reduction {
  std::size_t in_dim = 0;
  std::size_t out_dim = 0;
  std::size_t dropped = 0;      // leading principal directions discarded before `basis`
  std::vector<float> mean;      // in_dim
  std::vector<float> basis;     // out_dim x in_dim, orthonormal rows, by decreasing variance
  std::vector<float> variance;  // out_dim eigenvalues of the covariance
  double total_variance = 0.0;  // trace of the covariance

  // y must hold out_dim floats. A point that projects to zero maps to e_1.
  void apply(const float* x, float* y) const;
  double explained_variance_ratio() const;
};

// rows: n * in_dim floats. The basis holds principal directions number
// drop_top + 1 .. drop_top + out_dim; out_dim is clamped so that
// drop_top + out_dim <= min(in_dim, n - 1).
// Deterministic for a given seed (randomised subspace iteration).
Reduction fit_reduction(std::span<const float> rows, std::size_t in_dim, std::size_t out_dim,
                        std::size_t drop_top = 0, std::uint64_t seed = 1);

// Diagnostics of embedding geometry.
// Anisotropy: mean cosine over distinct pairs (0 for an isotropic cloud).
double mean_pairwise_cosine(std::span<const float> rows, std::size_t dim);
// Hubness: skewness of the k-occurrence distribution N_k (how often each point
// appears among the k nearest neighbours of the others).
double hubness_skewness(std::span<const float> rows, std::size_t dim, std::size_t k = 10);

// Applies the reduction to every row; returns n * out_dim floats.
std::vector<float> reduce(const Reduction& reduction, std::span<const float> rows);

// ---- density ---------------------------------------------------------------

// log fhat(x) for each query row. `exclude_self` evaluates the leave-one-out
// density when queries and data are the same array (query i skips data i and
// the sum is divided by n - 1 instead of n).
// The result does not depend on the number of threads nor on block_bytes: each
// query adds the data rows in index order, whatever the tiling.
// block_bytes == 0 selects exec::recommended_block_bytes().
std::vector<double> log_density(std::span<const float> data, std::span<const float> queries, std::size_t m,
                                double kappa, bool exclude_self, exec::OrderedExecutor& executor,
                                std::size_t block_bytes);

// ---- bandwidth -------------------------------------------------------------

// Plug-in rule of thumb for the vMF kernel (Garcia-Portugues 2013): the AMISE
// optimal bandwidth when the data themselves are vMF, with the concentration
// of that reference density estimated by maximum likelihood. With q = m - 1 and
// k the estimated concentration,
//
//   h^(4+q) = 4 sqrt(pi) I_{(q-1)/2}(k)^2
//             / ( k^{(q+1)/2} [ 2 q I_{(q+1)/2}(2k) + (2 + q) k I_{(q+3)/2}(2k) ] n ),   kappa = h^-2.
//
// Returns 0 when the rule is undefined (fewer than two dimensions, no data, or
// a sample without a preferred direction, for which the optimal kernel is flat).
double kappa_rule_of_thumb(std::span<const float> data, std::size_t m);

struct KappaSelection {
  double kappa = 0.0;                 // selected concentration
  double rule_of_thumb = 0.0;
  double score = 0.0;                 // mean leave-one-out log-likelihood at `kappa`
  std::vector<std::pair<double, double>> grid;  // (kappa, mean LOO log-likelihood)
};

// Likelihood cross-validation: maximises the mean leave-one-out log-likelihood
// over a geometric grid centred on the rule of thumb (ratio sqrt(2), extended
// while the maximum sits on its border), then refines by golden-section search
// on log kappa to a relative precision of 1e-3.
// Rows that coincide with x_i (cosine above 1 - 1e-6) are left out together with
// x_i: exact duplicates would otherwise make the likelihood unbounded in kappa.
// `grid` lists the grid points in increasing order of kappa.
KappaSelection select_kappa(std::span<const float> data, std::size_t m, exec::OrderedExecutor& executor,
                            std::size_t block_bytes);

// The largest concentration at which the kernels still reach the neighbours of
// the data points. With
//
//   N_i(kappa) = sum_{j != i} exp(kappa * (<x_i, x_j> - 1)),
//
// the weight of the other points at x_i in units of the weight of x_i itself
// (fhat(x_i) is proportional to 1 + N_i), the function returns the largest
// kappa for which at most a fraction `quantile` of the points have N_i < 1.
// N_i decreases as kappa grows; a point with N_i well below 1 is a local
// maximum of fhat on its own, and mean shift returns it as a mode with a
// single member. Relative precision 0.5 %. Same result for every thread count
// and block size.
double neighbourhood_kappa(std::span<const float> data, std::size_t m, exec::OrderedExecutor& executor,
                           std::size_t block_bytes, double quantile = 0.05);

// ---- mode seeking ----------------------------------------------------------

struct MeanShiftOptions {
  double kappa = 0.0;           // 0 = select_kappa(), limited as described below
  double tolerance = 1e-8;      // stop when 1 - <x_new, x_old> falls below this
  int max_iterations = 500;
  double merge_cosine = 0.999;  // end points closer than this belong to one mode
  std::size_t min_group = 1;    // smaller groups are merged into the nearest larger mode
  std::size_t block_bytes = 16 * 1024;  // 0 = exec::recommended_block_bytes()
  // Only when kappa == 0: use min(select_kappa().kappa, neighbourhood_kappa()).
  // The concentration that maximises the likelihood suits the density, but in
  // more than a few dimensions it is too large for mode seeking. On planted
  // mixtures it left 22 to 31 % of the points as modes with a single member in
  // 8 dimensions, 99 to 100 % in 16 and 74 to 77 % in 32 (none to 0.5 % with the
  // limit; in 3 dimensions the limit is not reached). Set to false for the
  // cross-validated value as it is.
  bool limit_to_neighbourhood = true;
};

struct Modes {
  std::size_t m = 0;
  double kappa = 0.0;
  std::vector<float> centers;             // count() * m, unit vectors, largest group first
  std::vector<std::uint32_t> assignment;  // mode index per data point
  std::vector<std::uint32_t> sizes;       // points per mode
  std::vector<float> log_density;         // log fhat at each data point
  std::vector<float> center_log_density;  // log fhat at each mode
  std::vector<std::uint16_t> iterations;  // mean-shift steps per data point
  std::size_t not_converged = 0;          // points that hit max_iterations
  // Steps after which the density was lower than before the step. The iteration
  // ascends the density, so anything but 0 indicates a numerical problem.
  std::size_t density_decreases = 0;

  std::size_t count() const noexcept { return sizes.size(); }
};

// End points are merged greedily in order of decreasing density: an end point
// joins the existing mode with which its cosine is largest, provided that
// cosine exceeds merge_cosine, and founds a new mode otherwise. A group with
// fewer than min_group points is merged into the mode whose centre is nearest
// to its own; when no group reaches min_group, the largest one is kept. Modes
// are numbered by decreasing size, then decreasing density of the centre, then
// lowest member index.
// Assignments do not depend on the number of threads nor on block_bytes.
Modes find_modes(std::span<const float> data, std::size_t m, const MeanShiftOptions& options,
                 exec::OrderedExecutor& executor);

// The path of one seed, computed by the same kernel as find_modes (diagnostic).
struct Trajectory {
  std::size_t m = 0;
  std::vector<double> points;       // (steps() + 1) * m: the start, then the position after each step
  std::vector<double> log_density;  // log fhat at each of these positions
  bool converged = false;

  std::size_t steps() const noexcept { return log_density.empty() ? 0 : log_density.size() - 1; }
};

// Mean shift started from data point `seed`. Requires options.kappa > 0;
// returns an empty trajectory otherwise or when `seed` is out of range.
Trajectory mean_shift_trajectory(std::span<const float> data, std::size_t m, std::size_t seed,
                                 const MeanShiftOptions& options);

// Storage order that lists points mode by mode (largest mode first) and, inside
// a mode, by decreasing density, so that neighbours in the index are neighbours
// in embedding space.
std::vector<std::uint32_t> layout_by_mode(const Modes& modes);

// Agreement between two labelings of the same points (label values are
// arbitrary). Both return 0 when the labelings differ in length or are empty.
//
// NMI = 2 I(a; b) / (H(a) + H(b)), in [0, 1]; 1 when both labelings consist of
// a single cluster.
double normalized_mutual_information(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b);
// ARI (Hubert and Arabie 1985): 1 for identical partitions, 0 in expectation
// for independent ones, negative when the agreement is below chance; 1 when
// the index is undefined because both partitions are trivial in the same way.
double adjusted_rand_index(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b);

}  // namespace prag::kde
