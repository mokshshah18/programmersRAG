// One-dimensional Gaussian kernel density estimation and isotonic regression,
// used to turn similarity scores into calibrated relevance probabilities.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace prag::retrieve {

// Silverman's rule: 0.9 * min(sd, IQR / 1.34) * n^{-1/5}. Returns a small
// positive value when the sample is degenerate.
//
// sd is the sample standard deviation (divisor n - 1) and IQR the difference
// of the quartiles, each interpolated linearly between order statistics at
// position (n - 1) q. When the quartiles coincide but the sample is not
// constant, sd alone is used. Degenerate means empty, a single value or all
// values equal; the result is then 1e-3 * max(1, |value|). Values that are not
// finite are ignored.
double silverman_bandwidth(std::span<const double> samples);

inline constexpr std::size_t kKeepAll = static_cast<std::size_t>(-1);

// f(x) = 1 / (n h) * sum_i phi((x - x_i) / h), phi the standard normal density.
class Kde1d {
 public:
  Kde1d() = default;

  // bandwidth <= 0 selects Silverman's rule. Values that are not finite are
  // ignored.
  static Kde1d fit(std::span<const double> samples, double bandwidth = 0.0);

  // Zero for an estimate without samples and for x that is not finite.
  double pdf(double x) const;
  // ln pdf(x) by log-sum-exp. Stays finite where pdf(x) underflows to zero, so
  // the ratio of two densities is defined in the tails. -infinity for an
  // estimate without samples and for x that is not finite.
  double log_pdf(double x) const;

  // Local log-quadratic likelihood estimate of ln f(x) with a Gaussian kernel
  // of width h. Around x the log-density is modelled as
  // a + b (t - x) + c (t - x)^2 / 2 and fitted by maximising the local
  // likelihood
  //
  //     1/n sum_i K_h(x_i - x) ln f(x_i)  -  integral K_h(t - x) f(t) dt,
  //
  // whose solution for the Gaussian kernel is explicit:
  //
  //     ln f(x) = ln f_h(x) + ln(h / s) - m^2 / (2 s^2),
  //
  // with f_h the kernel estimate of bandwidth h, and m and s^2 the mean and
  // the variance of the offsets x_i - x under the weights K_h(x_i - x). For a
  // normal density the estimate converges to the true value at every fixed
  // width h, where the kernel estimate converges to the density smoothed by
  // the kernel; this is what permits a wide kernel where samples are sparse.
  // pdf() is the local-constant member of the same family. Falls back to the
  // kernel estimate of bandwidth h where the data do not determine a curve:
  // when the weights amount to fewer than three effective samples,
  // (sum w)^2 / sum w^2 < 3, as happens far outside the data with a narrow
  // kernel, and when the weighted variance is not positive.
  // h <= 0 selects bandwidth().
  //
  // leave_out: index into samples() of a sample that takes no part, for
  // leave-one-out cross-validation; kKeepAll (or any index >= size()) keeps
  // all samples.
  double log_pdf_local(double x, double h = 0.0, std::size_t leave_out = kKeepAll) const;

  // Distance from x to its k-th nearest sample (k = 1: the nearest), or to the
  // farthest sample when there are fewer than k. Zero without samples.
  double neighbour_distance(double x, std::size_t k, std::size_t leave_out = kKeepAll) const;

  double bandwidth() const noexcept { return bandwidth_; }
  std::size_t size() const noexcept { return samples_.size(); }
  const std::vector<double>& samples() const noexcept { return samples_; }  // sorted ascending

 private:
  std::vector<double> samples_;
  double bandwidth_ = 1.0;
};

// Pool-adjacent-violators: the non-decreasing sequence closest to y in weighted
// least squares. Empty weights mean unit weights.
//
// Weights must be non-negative; negative and undefined weights count as zero,
// and a weight vector whose length differs from y is ignored. A point of
// weight zero does not influence the other points; it receives the value of
// the block it falls into.
std::vector<double> isotonic_nondecreasing(std::span<const double> y, std::span<const double> weights = {});

}  // namespace prag::retrieve
