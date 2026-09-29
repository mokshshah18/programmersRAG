#include "prag/retrieve/kde1d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace prag::retrieve {

namespace {

constexpr double kSqrt2Pi = 2.50662827463100050242;
// exp(-0.5 u^2) is exactly zero in double precision for |u| > 38.7, so a sum
// restricted to |u| <= kKernelReach equals the sum over all samples.
constexpr double kKernelReach = 39.0;
// Weights below exp(-kWeightReach) of the largest one are left out of the
// sums taken in the logarithm: a million of them together change a sum by
// less than one part in 10^15.
constexpr double kWeightReach = 50.0;

double degenerate_bandwidth(double location) { return 1e-3 * std::max(1.0, std::fabs(location)); }

std::vector<double> sorted_finite(std::span<const double> samples) {
  std::vector<double> out;
  out.reserve(samples.size());
  for (const double x : samples) {
    if (std::isfinite(x)) out.push_back(x);
  }
  std::sort(out.begin(), out.end());
  return out;
}

double quantile_sorted(const std::vector<double>& sorted, double q) {
  const double position = q * static_cast<double>(sorted.size() - 1);
  const std::size_t below = static_cast<std::size_t>(position);
  const std::size_t above = std::min(below + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(below);
  return sorted[below] + fraction * (sorted[above] - sorted[below]);
}

double silverman_sorted(const std::vector<double>& sorted) {
  const std::size_t n = sorted.size();
  if (n == 0) return degenerate_bandwidth(0.0);
  if (n == 1 || sorted.front() == sorted.back()) return degenerate_bandwidth(sorted.front());

  double mean = 0.0;
  for (const double x : sorted) mean += x;
  mean /= static_cast<double>(n);
  double squares = 0.0;
  for (const double x : sorted) squares += (x - mean) * (x - mean);
  const double sd = std::sqrt(squares / static_cast<double>(n - 1));

  const double iqr = quantile_sorted(sorted, 0.75) - quantile_sorted(sorted, 0.25);
  const double spread = iqr > 0.0 ? std::min(sd, iqr / 1.34) : sd;
  const double h = 0.9 * spread * std::pow(static_cast<double>(n), -0.2);
  if (!(h > 0.0) || !std::isfinite(h)) return degenerate_bandwidth(mean);
  return h;
}

// Kernel-weighted moments of the offsets x_i - x. The weight of sample i is
// exp(peak - u_i^2 / 2), u_i = (x_i - x) / width: the largest weight, that of
// the nearest sample, is factored out, so the sums stay in range where the
// kernel itself underflows.
struct Moments {
  double peak = 0.0;       // u^2 / 2 of the nearest sample
  double total = 0.0;      // sum of the weights, >= 1
  double mean = 0.0;       // weighted mean of the offsets
  double variance = 0.0;   // weighted variance of the offsets
  double effective = 0.0;  // effective number of samples, (sum w)^2 / sum w^2
  std::size_t count = 0;   // samples taking part: n, or n - 1 with one left out
};

// A curve with three parameters needs three points.
constexpr double kMinEffectiveSamples = 3.0;

Moments moments(const std::vector<double>& samples, double x, double width, std::size_t leave_out) {
  Moments m;
  const std::size_t n = samples.size();
  m.count = leave_out < n ? n - 1 : n;
  if (m.count == 0) return m;
  const double inverse = 1.0 / width;

  // Nearest sample on either side, stepping over the one left out.
  std::size_t above = static_cast<std::size_t>(std::lower_bound(samples.begin(), samples.end(), x) - samples.begin());
  std::size_t below = above;
  if (above < n && above == leave_out) ++above;
  if (below > 0 && below - 1 == leave_out) --below;
  double nearest = std::numeric_limits<double>::infinity();
  if (above < n) nearest = std::min(nearest, std::fabs(samples[above] - x));
  if (below > 0) nearest = std::min(nearest, std::fabs(x - samples[below - 1]));
  const double u_min = nearest * inverse;
  m.peak = 0.5 * u_min * u_min;

  const double reach = std::sqrt(u_min * u_min + 2.0 * kWeightReach) * width;
  const std::size_t first =
      static_cast<std::size_t>(std::lower_bound(samples.begin(), samples.end(), x - reach) - samples.begin());
  const std::size_t last =
      static_cast<std::size_t>(std::upper_bound(samples.begin(), samples.end(), x + reach) - samples.begin());

  // Incremental update of the weighted mean and scatter: one pass, and free of
  // the cancellation in E[d^2] - E[d]^2.
  double scatter = 0.0;
  double squares = 0.0;
  for (std::size_t i = first; i < last; ++i) {
    if (i == leave_out) continue;
    const double offset = samples[i] - x;
    const double u = offset * inverse;
    const double w = std::exp(m.peak - 0.5 * u * u);
    if (!(w > 0.0)) continue;
    m.total += w;
    squares += w * w;
    const double delta = offset - m.mean;
    m.mean += (w / m.total) * delta;
    scatter += w * delta * (offset - m.mean);
  }
  // Rounding in the window bounds may drop the nearest sample when it lies
  // exactly on the edge; its weight is 1 by construction.
  if (!(m.total >= 1.0)) m.total = 1.0;
  m.variance = scatter / m.total;
  m.effective = squares > 0.0 ? m.total * m.total / squares : 1.0;
  return m;
}

// ln of the kernel estimate of bandwidth `width` from the moments.
double kernel_log_density(const Moments& m, double width) {
  return -m.peak + std::log(m.total) - std::log(static_cast<double>(m.count) * width * kSqrt2Pi);
}

}  // namespace

double silverman_bandwidth(std::span<const double> samples) { return silverman_sorted(sorted_finite(samples)); }

Kde1d Kde1d::fit(std::span<const double> samples, double bandwidth) {
  Kde1d kde;
  kde.samples_ = sorted_finite(samples);
  kde.bandwidth_ = (bandwidth > 0.0 && std::isfinite(bandwidth)) ? bandwidth : silverman_sorted(kde.samples_);
  return kde;
}

double Kde1d::pdf(double x) const {
  if (samples_.empty() || !std::isfinite(x)) return 0.0;
  const double reach = kKernelReach * bandwidth_;
  const auto first = std::lower_bound(samples_.begin(), samples_.end(), x - reach);
  const auto last = std::upper_bound(first, samples_.end(), x + reach);
  const double inverse = 1.0 / bandwidth_;
  double sum = 0.0;
  for (auto it = first; it != last; ++it) {
    const double u = (x - *it) * inverse;
    sum += std::exp(-0.5 * u * u);
  }
  return sum / (static_cast<double>(samples_.size()) * bandwidth_ * kSqrt2Pi);
}

double Kde1d::log_pdf(double x) const {
  if (samples_.empty() || !std::isfinite(x)) return -std::numeric_limits<double>::infinity();
  const Moments m = moments(samples_, x, bandwidth_, kKeepAll);
  return kernel_log_density(m, bandwidth_);
}

double Kde1d::log_pdf_local(double x, double h, std::size_t leave_out) const {
  if (!std::isfinite(x)) return -std::numeric_limits<double>::infinity();
  const double width = (h > 0.0 && std::isfinite(h)) ? h : bandwidth_;
  const Moments m = moments(samples_, x, width, leave_out);
  if (m.count == 0) return -std::numeric_limits<double>::infinity();
  const double constant = kernel_log_density(m, width);
  if (!(m.effective >= kMinEffectiveSamples)) return constant;
  if (!(m.variance > 1e-24 * width * width) || !std::isfinite(m.variance)) return constant;
  return constant + std::log(width) - 0.5 * std::log(m.variance) - 0.5 * m.mean * m.mean / m.variance;
}

double Kde1d::neighbour_distance(double x, std::size_t k, std::size_t leave_out) const {
  if (k == 0 || !std::isfinite(x)) return 0.0;
  const std::size_t n = samples_.size();
  const bool skipping = leave_out < n;
  const std::size_t count = skipping ? n - 1 : n;
  if (count == 0) return 0.0;
  // The samples that take part, in ascending order.
  const auto at = [&](std::size_t i) { return samples_[skipping && i >= leave_out ? i + 1 : i]; };

  // The nearest `wanted` samples are consecutive in that order. Their run
  // starts at the smallest index for which the sample following the run is not
  // nearer to x than the first sample of the run.
  const std::size_t wanted = std::min(k, count);
  std::size_t low = 0;
  std::size_t high = count - wanted;
  while (low < high) {
    const std::size_t mid = low + (high - low) / 2;
    if (x - at(mid) > at(mid + wanted) - x) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return std::max(x - at(low), at(low + wanted - 1) - x);
}

std::vector<double> isotonic_nondecreasing(std::span<const double> y, std::span<const double> weights) {
  const std::size_t n = y.size();
  const bool weighted = weights.size() == n && n != 0;

  // A block is a run of pooled points. Its level is the weighted mean of its
  // values, or their plain mean when the total weight is zero. The smallest
  // and the largest value are kept to return a block of equal values exactly
  // (w y / w need not be y), which makes the regression idempotent to the bit.
  struct Block {
    double weighted_sum = 0.0;
    double weight = 0.0;
    double plain_sum = 0.0;
    double smallest = 0.0;
    double largest = 0.0;
    std::size_t count = 0;
    double level() const noexcept {
      if (smallest == largest) return smallest;
      const double mean = weight > 0.0 ? weighted_sum / weight : plain_sum / static_cast<double>(count);
      return std::clamp(mean, smallest, largest);
    }
  };

  std::vector<Block> blocks;
  blocks.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    double w = weighted ? weights[i] : 1.0;
    if (!(w > 0.0) || !std::isfinite(w)) w = 0.0;
    Block block{w * y[i], w, y[i], y[i], y[i], 1};
    while (!blocks.empty() && blocks.back().level() > block.level()) {
      const Block& previous = blocks.back();
      block.weighted_sum += previous.weighted_sum;
      block.weight += previous.weight;
      block.plain_sum += previous.plain_sum;
      block.smallest = std::min(block.smallest, previous.smallest);
      block.largest = std::max(block.largest, previous.largest);
      block.count += previous.count;
      blocks.pop_back();
    }
    blocks.push_back(block);
  }

  // The levels are the values the loop compared, so they are non-decreasing
  // as computed, not only up to rounding.
  std::vector<double> out;
  out.reserve(n);
  for (const Block& block : blocks) out.insert(out.end(), block.count, block.level());
  return out;
}

}  // namespace prag::retrieve
