// Density evaluation and selection of the concentration.
#include <algorithm>
#include <cmath>
#include <limits>

#include "blocking.hpp"
#include "kernels.hpp"
#include "prag/kde/directional.hpp"
#include "prag/kde/vmf.hpp"

namespace prag::kde {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kLogPi = 1.14472988584940017414;

// Concentrations outside this range are never proposed by the selection.
constexpr double kMinKappa = 1e-3;
constexpr double kMaxKappa = 1e7;
// The grid is centred on the rule of thumb, but not below this value: centred
// data have almost no preferred direction, and their rule of thumb tends to 0.
constexpr double kMinGridCentre = 1.0;
constexpr int kGridHalfWidth = 6;   // grid points on each side of the centre
constexpr int kGridExtension = 6;   // points added when the maximum is on the border
constexpr int kMaxExtensions = 12;
constexpr double kGoldenTolerance = 1e-3;  // width of the final bracket in log kappa
// Two rows with a cosine above this value are copies of each other up to the
// rounding of float32 normalisation.
constexpr double kDuplicateCosine = 1.0 - 1e-6;
// neighbourhood_kappa(): interior points per subdivision of the bracket, and
// number of subdivisions (each divides the width in log kappa by kBracketPoints + 1).
constexpr int kBracketPoints = 8;
constexpr int kBracketPasses = 2;

struct DensityScratch {
  std::vector<double> reference;  // running maximum of the exponent, per query of the batch
  std::vector<double> sum;        // sum of exp(exponent - reference)
};

// Kernel sums between every data point and all the others, for several
// concentrations in one pass over the pairs: the inner products do not depend
// on kappa, only the exponentials do. For point i and concentration k the pass
// produces
//   best_i       the largest inner product with another row,
//   used_i       the number of rows that entered the sum,
//   sums_i(k)    sum_j exp(k (<x_i, x_j> - best_i)),
// from which both the leave-one-out likelihood and the weight of the
// neighbourhood follow.
class PairwisePass {
 public:
  PairwisePass(std::span<const float> data, std::size_t m, exec::OrderedExecutor& executor, std::size_t block_bytes)
      : data_(data.data()), m_(m), n_(data.size() / m), executor_(executor), block_bytes_(block_bytes) {}

  // Mean leave-one-out log-likelihood for each concentration; rows that
  // coincide with the point are left out with it. -infinity when no point has
  // a row left to be compared with.
  std::vector<double> loo_scores(const std::vector<double>& kappas) {
    const std::size_t g = kappas.size();
    std::vector<double> mean(g, -kInf);
    if (g == 0 || n_ < 2) return mean;
    std::vector<double> log_normalizer(g);
    for (std::size_t k = 0; k < g; ++k) log_normalizer[k] = log_vmf_normalizer(m_, kappas[k]);

    std::vector<double> total(g, 0.0);
    std::size_t counted = 0;
    // The sum over points is formed in commit, point by point in index order, so
    // that it is the same number for every thread count and every block size.
    run(kappas, true, [&](const exec::Block& batch) {
      for (std::size_t i = batch.first; i < batch.first + batch.count; ++i) {
        if (used_[i] == 0) continue;
        const double log_count = std::log(static_cast<double>(used_[i]));
        for (std::size_t k = 0; k < g; ++k) {
          total[k] += log_normalizer[k] - log_count + kappas[k] * best_[i] + std::log(sums_[i * g + k]);
        }
        ++counted;
      }
    });
    if (counted == 0) return mean;
    for (std::size_t k = 0; k < g; ++k) mean[k] = total[k] / static_cast<double>(counted);
    return mean;
  }

  // For each concentration, the `quantile` quantile over the points of
  //   N_i(k) = sum_{j != i} exp(k (<x_i, x_j> - 1)),
  // the weight of the other points at x_i in units of the weight of x_i itself.
  std::vector<double> neighbour_weight_quantiles(const std::vector<double>& kappas, double quantile) {
    const std::size_t g = kappas.size();
    std::vector<double> out(g, 0.0);
    if (g == 0 || n_ < 2) return out;
    run(kappas, false, [](const exec::Block&) {});
    const std::size_t rank = std::min(n_ - 1, static_cast<std::size_t>(quantile * static_cast<double>(n_)));
    std::vector<double> weights(n_);
    for (std::size_t k = 0; k < g; ++k) {
      for (std::size_t i = 0; i < n_; ++i) {
        weights[i] = used_[i] == 0 ? 0.0 : std::exp(kappas[k] * (best_[i] - 1.0)) * sums_[i * g + k];
      }
      std::nth_element(weights.begin(), weights.begin() + static_cast<std::ptrdiff_t>(rank), weights.end());
      out[k] = weights[rank];
    }
    return out;
  }

 private:
  struct Scratch {
    std::vector<double> best;
    std::vector<double> sums;
    std::vector<std::uint32_t> used;
  };

  template <class Commit>
  void run(const std::vector<double>& kappas, bool skip_duplicates, Commit&& commit) {
    const std::size_t g = kappas.size();
    // State of one point: its row (read in place), the largest inner product
    // met so far, the number of rows that contributed, and one sum per kappa.
    const std::size_t state_bytes = m_ * sizeof(float) + (g + 2) * sizeof(double);
    const detail::Layout layout = detail::make_layout(n_, m_ * sizeof(float), n_, state_bytes, block_bytes_);
    // One scratch area per worker for all the passes of a selection; it grows
    // when a pass carries more concentrations than the ones before.
    scratch_.resize(executor_.threads());
    best_.assign(n_, -kInf);
    used_.assign(n_, 0u);
    sums_.assign(n_ * g, 0.0);

    const auto work = [&](const exec::Block& batch, unsigned worker) {
      Scratch& s = scratch_[worker];
      if (s.best.size() < layout.max_batch) {
        s.best.resize(layout.max_batch);
        s.used.resize(layout.max_batch);
      }
      if (s.sums.size() < layout.max_batch * g) s.sums.resize(layout.max_batch * g);
      std::fill_n(s.best.begin(), batch.count, -kInf);
      std::fill_n(s.sums.begin(), batch.count * g, 0.0);
      std::fill_n(s.used.begin(), batch.count, 0u);

      for (const exec::Block& tile : layout.tiles) {
        const float* rows = data_ + tile.first * m_;
        for (std::size_t q = 0; q < batch.count; ++q) {
          const std::size_t self = batch.first + q;
          const float* x = data_ + self * m_;
          double* sums = s.sums.data() + q * g;
          double best = s.best[q];
          std::uint32_t used = s.used[q];
          for (std::size_t r = 0; r < tile.count; ++r) {
            if (tile.first + r == self) continue;
            const double d = detail::dot(rows + r * m_, x, m_);
            if (skip_duplicates && d > kDuplicateCosine) continue;
            if (used == 0) {
              for (std::size_t k = 0; k < g; ++k) sums[k] = 1.0;
              best = d;
            } else if (d > best) {
              const double shift = best - d;
              for (std::size_t k = 0; k < g; ++k) sums[k] = sums[k] * std::exp(kappas[k] * shift) + 1.0;
              best = d;
            } else {
              const double gap = d - best;
              for (std::size_t k = 0; k < g; ++k) sums[k] += std::exp(kappas[k] * gap);
            }
            ++used;
          }
          s.best[q] = best;
          s.used[q] = used;
        }
      }
      // Each point owns its slots of the result, so the worker writes them directly.
      for (std::size_t q = 0; q < batch.count; ++q) {
        const std::size_t self = batch.first + q;
        best_[self] = s.best[q];
        used_[self] = s.used[q];
        std::copy_n(s.sums.begin() + static_cast<std::ptrdiff_t>(q * g), g,
                    sums_.begin() + static_cast<std::ptrdiff_t>(self * g));
      }
      return 0;
    };
    executor_.run(layout.batches, work, [&](const exec::Block& batch, int&&) { commit(batch); });
  }

  const float* data_;
  std::size_t m_;
  std::size_t n_;
  exec::OrderedExecutor& executor_;
  std::size_t block_bytes_;
  std::vector<Scratch> scratch_;
  std::vector<double> best_;
  std::vector<std::uint32_t> used_;
  std::vector<double> sums_;
};

double log_sum_exp(double a, double b) {
  const double hi = std::max(a, b);
  const double lo = std::min(a, b);
  if (hi == -kInf) return -kInf;
  return hi + std::log1p(std::exp(lo - hi));
}

}  // namespace

std::vector<double> log_density(std::span<const float> data, std::span<const float> queries, std::size_t m,
                                double kappa, bool exclude_self, exec::OrderedExecutor& executor,
                                std::size_t block_bytes) {
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  const std::size_t n_queries = m == 0 ? 0 : queries.size() / m;
  std::vector<double> out(n_queries, -kInf);
  if (n_queries == 0 || n == 0) return out;

  const double log_normalizer = log_vmf_normalizer(m, kappa);
  const std::size_t state_bytes = m * sizeof(float) + 2 * sizeof(double);
  const detail::Layout layout = detail::make_layout(n, m * sizeof(float), n_queries, state_bytes, block_bytes);
  std::vector<DensityScratch> scratch(executor.threads());
  const float* rows_begin = data.data();
  const float* queries_begin = queries.data();

  const auto work = [&](const exec::Block& batch, unsigned worker) {
    DensityScratch& s = scratch[worker];
    if (s.reference.empty()) {
      s.reference.resize(layout.max_batch);
      s.sum.resize(layout.max_batch);
    }
    std::fill_n(s.reference.begin(), batch.count, -kInf);
    std::fill_n(s.sum.begin(), batch.count, 0.0);

    for (const exec::Block& tile : layout.tiles) {
      const float* rows = rows_begin + tile.first * m;
      for (std::size_t q = 0; q < batch.count; ++q) {
        const std::size_t self = batch.first + q;
        const float* x = queries_begin + self * m;
        double reference = s.reference[q];
        double sum = s.sum[q];
        for (std::size_t r = 0; r < tile.count; ++r) {
          if (exclude_self && tile.first + r == self) continue;
          const double e = kappa * detail::dot(rows + r * m, x, m);
          // Running log-sum-exp: the sum is kept relative to the largest exponent so far.
          if (e > reference) {
            sum = sum * std::exp(reference - e) + 1.0;
            reference = e;
          } else {
            sum += std::exp(e - reference);
          }
        }
        s.reference[q] = reference;
        s.sum[q] = sum;
      }
    }
    // Each query owns its slot of the result, so the worker writes it directly.
    for (std::size_t q = 0; q < batch.count; ++q) {
      const std::size_t self = batch.first + q;
      const std::size_t terms = n - ((exclude_self && self < n) ? 1 : 0);
      if (terms == 0) continue;
      out[self] = log_normalizer - std::log(static_cast<double>(terms)) + s.reference[q] + std::log(s.sum[q]);
    }
    return 0;
  };
  executor.run(layout.batches, work, [](const exec::Block&, int&&) {});
  return out;
}

double kappa_rule_of_thumb(std::span<const float> data, std::size_t m) {
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  if (n == 0 || m < 2) return 0.0;

  std::vector<double> resultant(m, 0.0);
  std::size_t used = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = data.data() + i * m;
    const double norm2 = detail::dot(x, x, m);
    if (!(norm2 > 0.0)) continue;
    detail::axpy(1.0 / std::sqrt(norm2), x, resultant.data(), m);
    ++used;
  }
  if (used == 0) return 0.0;
  const double rbar =
      std::sqrt(detail::dot(resultant.data(), resultant.data(), m)) / static_cast<double>(used);
  double concentration = vmf_kappa_mle(m, std::min(rbar, 1.0 - 1e-12));
  if (!(concentration > 0.0)) return 0.0;
  concentration = std::min(concentration, kMaxKappa);

  const double q = static_cast<double>(m - 1);
  const double k = concentration;
  const double bracket = log_sum_exp(std::log(2.0 * q) + log_bessel_i(0.5 * (q + 1.0), 2.0 * k),
                                     std::log((2.0 + q) * k) + log_bessel_i(0.5 * (q + 3.0), 2.0 * k));
  // log of h^(4+q)
  const double log_power = std::log(4.0) + 0.5 * kLogPi + 2.0 * log_bessel_i(0.5 * (q - 1.0), k) -
                           0.5 * (q + 1.0) * std::log(k) - bracket - std::log(static_cast<double>(used));
  const double kappa = std::exp(-2.0 * log_power / (4.0 + q));
  return std::isfinite(kappa) ? kappa : 0.0;
}

KappaSelection select_kappa(std::span<const float> data, std::size_t m, exec::OrderedExecutor& executor,
                            std::size_t block_bytes) {
  KappaSelection out;
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  out.rule_of_thumb = kappa_rule_of_thumb(data, m);
  const double centre = std::clamp(out.rule_of_thumb, kMinGridCentre, kMaxKappa / 64.0);
  out.kappa = centre;
  out.score = -kInf;
  if (n < 2 || m < 2) return out;

  PairwisePass pass(data, m, executor, block_bytes);
  const double ratio = std::sqrt(2.0);

  std::vector<double> kappas;
  for (int j = -kGridHalfWidth; j <= kGridHalfWidth; ++j) kappas.push_back(centre * std::pow(ratio, j));
  std::vector<double> scores = pass.loo_scores(kappas);
  if (scores.front() == -kInf) return out;  // every row is a copy of every other row

  // `kappas` stays sorted; the first maximum wins, so equal scores select the smaller concentration.
  const auto best_index = [&] {
    std::size_t best = 0;
    for (std::size_t i = 1; i < scores.size(); ++i) {
      if (scores[i] > scores[best]) best = i;
    }
    return best;
  };
  std::size_t best = best_index();
  for (int extension = 0; extension < kMaxExtensions; ++extension) {
    const bool at_top = best + 1 == kappas.size() && kappas.back() * ratio <= kMaxKappa;
    const bool at_bottom = best == 0 && kappas.front() / ratio >= kMinKappa;
    if (!at_top && !at_bottom) break;
    std::vector<double> more;
    for (int j = 1; j <= kGridExtension; ++j) {
      const double k = at_top ? kappas.back() * std::pow(ratio, j) : kappas.front() / std::pow(ratio, j);
      if (k > kMaxKappa || k < kMinKappa) break;
      more.push_back(k);
    }
    if (more.empty()) break;
    const std::vector<double> more_scores = pass.loo_scores(more);
    if (at_top) {
      kappas.insert(kappas.end(), more.begin(), more.end());
      scores.insert(scores.end(), more_scores.begin(), more_scores.end());
    } else {
      kappas.insert(kappas.begin(), more.rbegin(), more.rend());
      scores.insert(scores.begin(), more_scores.rbegin(), more_scores.rend());
    }
    best = best_index();
  }

  out.grid.reserve(kappas.size());
  for (std::size_t i = 0; i < kappas.size(); ++i) out.grid.emplace_back(kappas[i], scores[i]);
  out.kappa = kappas[best];
  out.score = scores[best];
  if (best == 0 || best + 1 == kappas.size()) return out;  // maximum on the border of the admissible range

  // Golden-section search for the maximum over log kappa inside the bracket
  // formed by the neighbours of the best grid point.
  const double golden = 0.5 * (std::sqrt(5.0) - 1.0);
  double a = std::log(kappas[best - 1]);
  double b = std::log(kappas[best + 1]);
  double c = b - golden * (b - a);
  double d = a + golden * (b - a);
  const std::vector<double> first = pass.loo_scores({std::exp(c), std::exp(d)});
  double fc = first[0];
  double fd = first[1];
  const auto consider = [&](double log_kappa, double score) {
    if (score > out.score) {
      out.score = score;
      out.kappa = std::exp(log_kappa);
    }
  };
  consider(c, fc);
  consider(d, fd);
  while (b - a > kGoldenTolerance) {
    if (fc > fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - golden * (b - a);
      fc = pass.loo_scores({std::exp(c)})[0];
      consider(c, fc);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + golden * (b - a);
      fd = pass.loo_scores({std::exp(d)})[0];
      consider(d, fd);
    }
  }
  return out;
}

double neighbourhood_kappa(std::span<const float> data, std::size_t m, exec::OrderedExecutor& executor,
                           std::size_t block_bytes, double quantile) {
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  if (n < 2) return kMaxKappa;
  quantile = std::clamp(quantile, 0.0, 1.0);
  PairwisePass pass(data, m, executor, block_bytes);
  const double ratio = std::sqrt(2.0);

  // The quantile of the neighbour weights decreases as kappa grows. Walk a
  // geometric grid, upward or downward from sqrt(m), until two neighbouring
  // grid points enclose the value one ...
  double lo = 0.0;  // largest concentration known to keep the quantile at one or above
  double hi = 0.0;  // smallest concentration known to bring it below one
  double up = std::sqrt(static_cast<double>(m));
  double down = up / ratio;
  bool upward = true;
  for (int round = 0; round < 2 * kMaxExtensions && (lo == 0.0 || hi == 0.0); ++round) {
    std::vector<double> kappas;
    for (int j = 0; j < 2 * kGridHalfWidth; ++j) {
      const double k = upward ? up * std::pow(ratio, j) : down / std::pow(ratio, j);
      if (k > kMaxKappa || k < kMinKappa) break;
      kappas.push_back(k);
    }
    if (kappas.empty()) break;
    const std::vector<double> weights = pass.neighbour_weight_quantiles(kappas, quantile);
    for (std::size_t i = 0; i < kappas.size(); ++i) {
      if (weights[i] >= 1.0) {
        lo = std::max(lo, kappas[i]);
      } else if (hi == 0.0 || kappas[i] < hi) {
        hi = kappas[i];
      }
    }
    if (upward) {
      up = kappas.back() * ratio;
    } else {
      down = kappas.back() / ratio;
    }
    upward = hi == 0.0;  // below one nowhere yet: continue upward; otherwise look below the grid
  }
  if (hi == 0.0) return kMaxKappa;  // the neighbours outweigh the point at every admissible concentration
  if (lo == 0.0) return kMinKappa;

  // ... then subdivide the bracket.
  for (int refinement = 0; refinement < kBracketPasses; ++refinement) {
    std::vector<double> kappas;
    const double step = std::log(hi / lo) / static_cast<double>(kBracketPoints + 1);
    for (int j = 1; j <= kBracketPoints; ++j) kappas.push_back(lo * std::exp(step * j));
    const std::vector<double> weights = pass.neighbour_weight_quantiles(kappas, quantile);
    double next_lo = lo;
    double next_hi = hi;
    for (std::size_t i = 0; i < kappas.size(); ++i) {
      if (weights[i] >= 1.0) {
        next_lo = std::max(next_lo, kappas[i]);
      } else {
        next_hi = std::min(next_hi, kappas[i]);
      }
    }
    lo = next_lo;
    hi = next_hi;
  }
  return lo;
}

}  // namespace prag::kde
