#include "prag/retrieve/calibration.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "prag/retrieve/kde1d.hpp"
#include "prag/retrieve/logodds.hpp"

namespace prag::retrieve {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kDefaultMidpoint = 4.0;
constexpr double kDefaultSlope = 1.5;
constexpr double kDefaultCandidates[] = {0.0, 0.03, 0.1, 0.3, 1.0};

// Value at z of the piecewise-linear function through (grid, values), constant
// outside the grid. With `geometric`, values are interpolated in the logarithm.
// The grid has at least two points and is ascending.
double interpolate(const std::vector<double>& grid, const std::vector<double>& values, double z, bool geometric) {
  const std::size_t m = grid.size();
  if (!(z > grid.front())) return values.front();  // below the grid, or NaN
  if (z >= grid.back()) return values.back();

  // Uniform grids are hit by the first guess; the two loops make the lookup
  // correct for every ascending grid and for rounding at the nodes.
  const double scale = static_cast<double>(m - 1) / (grid.back() - grid.front());
  std::size_t i = std::min(m - 2, static_cast<std::size_t>((z - grid.front()) * scale));
  while (i > 0 && z < grid[i]) --i;
  while (i + 2 < m && z >= grid[i + 1]) ++i;

  const double width = grid[i + 1] - grid[i];
  const double w = width > 0.0 ? (z - grid[i]) / width : 0.0;
  const double a = values[i];
  const double b = values[i + 1];
  const double value = geometric ? a * std::pow(b / a, w) : a + w * (b - a);
  // Rounding must not carry the value outside the segment: that would break
  // monotonicity at the nodes.
  return std::clamp(value, std::min(a, b), std::max(a, b));
}

// Doubles mapped to unsigned integers in the same order, so that a bisection
// over all representable values takes at most 64 steps. Negative values have
// all bits inverted, the others get the top bit set.
constexpr std::uint64_t kSignBit = std::uint64_t{1} << 63;

std::uint64_t to_ordered(double x) noexcept {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  return (bits & kSignBit) != 0 ? ~bits : bits | kSignBit;
}

double from_ordered(std::uint64_t key) noexcept {
  return std::bit_cast<double>((key & kSignBit) != 0 ? key & ~kSignBit : ~key);
}

// Smallest double z in (low, high] with reaches(z), given !reaches(low),
// reaches(high) and a predicate that is monotone in z.
template <class Predicate>
double first_reaching(double low, double high, Predicate&& reaches) {
  std::uint64_t lo = to_ordered(low);
  std::uint64_t hi = to_ordered(high);
  while (hi - lo > 1) {
    const std::uint64_t mid = lo + (hi - lo) / 2;
    if (reaches(from_ordered(mid))) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return from_ordered(hi);
}

bool finite_vector(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

// ---- local density estimate ---------------------------------------------------

// k = max(k_min, ceil(a n)), between 1 and n.
std::size_t neighbours(double fraction, std::size_t n, std::size_t min_neighbours) noexcept {
  const double share = std::isfinite(fraction) ? std::clamp(fraction, 0.0, 1.0) : 0.0;
  const std::size_t wanted = static_cast<std::size_t>(std::ceil(share * static_cast<double>(n)));
  return std::clamp<std::size_t>(std::max(min_neighbours, wanted), 1, std::max<std::size_t>(1, n));
}

double local_log_density(const Kde1d& kde, double z, std::size_t k, std::size_t leave_out = kKeepAll) {
  const double width = std::max(kde.bandwidth(), kde.neighbour_distance(z, k, leave_out));
  return kde.log_pdf_local(z, width, leave_out);
}

// ln f(z) of one class under the chosen estimate.
class ClassDensity {
 public:
  ClassDensity(const Kde1d& kde, DensityEstimate estimate, double fraction, std::size_t min_neighbours)
      : kde_(kde),
        local_(estimate == DensityEstimate::LocalQuadratic),
        neighbours_(neighbours(fraction, kde.size(), min_neighbours)) {}

  double operator()(double z) const {
    return local_ ? local_log_density(kde_, z, neighbours_) : kde_.log_pdf(z);
  }

 private:
  const Kde1d& kde_;
  bool local_;
  std::size_t neighbours_;
};

// ---- choice of the neighbour fractions ---------------------------------------

// At most `limit` scores, evenly spaced in rank, both extremes included.
Kde1d thinned(const Kde1d& kde, std::size_t limit) {
  const std::size_t n = kde.size();
  if (limit < 2 || n <= limit) return kde;
  std::vector<double> kept(limit);
  for (std::size_t i = 0; i < limit; ++i) kept[i] = kde.samples()[i * (n - 1) / (limit - 1)];
  return Kde1d::fit(kept);
}

struct Fractions {
  double relevant = 0.0;
  double other = 0.0;
};

// Log-densities of both classes at the scores that enter the cross-validation,
// for one number of neighbours per class. `own` is estimated without the score
// itself.
struct Evaluated {
  std::vector<double> own;    // density of the class the score belongs to
  std::vector<double> cross;  // density of the other class
};

Fractions select_fractions(const Kde1d& relevant_all, const Kde1d& other_all, double prior,
                           const FitOptions& options) {
  const Kde1d relevant = thinned(relevant_all, options.cv_max_samples);
  const Kde1d other = thinned(other_all, options.cv_max_samples);
  const std::size_t n1 = relevant.size();
  const std::size_t n0 = other.size();

  // Candidates per class, ascending: from local to global.
  std::vector<double> listed(options.candidates.begin(), options.candidates.end());
  std::erase_if(listed, [](double a) { return !(a >= 0.0) || !(a <= 1.0); });
  if (listed.empty()) listed.assign(std::begin(kDefaultCandidates), std::end(kDefaultCandidates));
  std::sort(listed.begin(), listed.end());
  listed.erase(std::unique(listed.begin(), listed.end()), listed.end());
  const auto fixed = [](double a) { return a >= 0.0 && a <= 1.0; };
  const std::vector<double> for_relevant =
      fixed(options.fraction_relevant) ? std::vector<double>{options.fraction_relevant} : listed;
  const std::vector<double> for_other =
      fixed(options.fraction_nonrelevant) ? std::vector<double>{options.fraction_nonrelevant} : listed;
  if (for_relevant.size() == 1 && for_other.size() == 1) return {for_relevant.front(), for_other.front()};

  // The scores where the posterior is away from 0 and 1: the lowest relevant
  // and the highest non-relevant ones.
  const std::size_t points = std::max<std::size_t>(1, options.cv_points);
  const std::size_t t1 = std::min(points, n1);
  const std::size_t t0 = std::min(points, n0);
  const std::size_t first0 = n0 - t0;

  // Densities of the relevant class at all evaluation scores, per candidate;
  // likewise for the non-relevant class. A candidate that gives the same
  // number of neighbours as the one before shares its result.
  const auto evaluate = [&](const Kde1d& kde, const std::vector<double>& fractions, std::size_t own_first,
                            std::size_t own_count, const Kde1d& foreign, std::size_t foreign_first,
                            std::size_t foreign_count) {
    std::vector<Evaluated> out(fractions.size());
    std::size_t previous_k = 0;
    for (std::size_t c = 0; c < fractions.size(); ++c) {
      const std::size_t n = kde.size();
      const std::size_t k = neighbours(fractions[c], n, options.min_neighbours);
      if (c > 0 && k == previous_k) {
        out[c] = out[c - 1];
        continue;
      }
      previous_k = k;
      const std::size_t k_without = neighbours(fractions[c], n - 1, options.min_neighbours);
      out[c].own.resize(own_count);
      for (std::size_t i = 0; i < own_count; ++i) {
        const std::size_t index = own_first + i;
        out[c].own[i] = local_log_density(kde, kde.samples()[index], k_without, index);
      }
      out[c].cross.resize(foreign_count);
      for (std::size_t j = 0; j < foreign_count; ++j) {
        out[c].cross[j] = local_log_density(kde, foreign.samples()[foreign_first + j], k);
      }
    }
    return out;
  };
  // f1 at the relevant scores (left out) and at the non-relevant scores; f0 likewise.
  const std::vector<Evaluated> f1 = evaluate(relevant, for_relevant, 0, t1, other, first0, t0);
  const std::vector<Evaluated> f0 = evaluate(other, for_other, first0, t0, relevant, 0, t1);

  // Loss terms of one pair of candidates: one per evaluation score.
  const double prior_logit = logit(prior);
  const double weight1 = prior / static_cast<double>(n1);
  const double weight0 = (1.0 - prior) / static_cast<double>(n0);
  const auto loss_terms = [&](std::size_t c1, std::size_t c0) {
    std::vector<double> terms;
    terms.reserve(t1 + t0);
    for (std::size_t i = 0; i < t1; ++i) {
      const double ratio = std::clamp(f1[c1].own[i] - f0[c0].cross[i], -kMaxLogRatio, kMaxLogRatio);
      terms.push_back(weight1 * softplus(-clamp_logit(prior_logit + ratio)));  // -ln P
    }
    for (std::size_t j = 0; j < t0; ++j) {
      const double ratio = std::clamp(f1[c1].cross[j] - f0[c0].own[j], -kMaxLogRatio, kMaxLogRatio);
      terms.push_back(weight0 * softplus(clamp_logit(prior_logit + ratio)));  // -ln(1 - P)
    }
    return terms;
  };
  const auto total = [](const std::vector<double>& terms) {
    double sum = 0.0;
    for (const double t : terms) sum += t;
    return sum;
  };

  std::size_t best1 = 0;
  std::size_t best0 = 0;
  double best_loss = kInfinity;
  for (std::size_t c1 = 0; c1 < for_relevant.size(); ++c1) {
    for (std::size_t c0 = 0; c0 < for_other.size(); ++c0) {
      const double loss = total(loss_terms(c1, c0));
      if (loss < best_loss) {
        best_loss = loss;
        best1 = c1;
        best0 = c0;
      }
    }
  }

  // One-standard-error rule: a wider pair is acceptable when its loss exceeds
  // the smallest loss by no more than `cv_tolerance` standard errors of the
  // difference, estimated from the paired terms.
  const std::vector<double> best_terms = loss_terms(best1, best0);
  const double tolerance =
      options.cv_tolerance > 0.0 && std::isfinite(options.cv_tolerance) ? options.cv_tolerance : 0.0;
  const auto acceptable = [&](std::size_t c1, std::size_t c0) {
    const std::vector<double> terms = loss_terms(c1, c0);
    const std::size_t m = terms.size();
    double difference = 0.0;
    for (std::size_t k = 0; k < m; ++k) difference += terms[k] - best_terms[k];
    if (difference <= 0.0) return true;
    if (m < 2) return false;
    const double mean = difference / static_cast<double>(m);
    double squares = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      const double deviation = (terms[k] - best_terms[k]) - mean;
      squares += deviation * deviation;
    }
    const double standard_error = std::sqrt(squares * static_cast<double>(m) / static_cast<double>(m - 1));
    return difference <= tolerance * standard_error;
  };
  std::size_t chosen0 = best0;
  for (std::size_t c0 = for_other.size(); c0-- > best0 + 1;) {
    if (acceptable(best1, c0)) {
      chosen0 = c0;
      break;
    }
  }
  std::size_t chosen1 = best1;
  for (std::size_t c1 = for_relevant.size(); c1-- > best1 + 1;) {
    if (acceptable(c1, chosen0)) {
      chosen1 = c1;
      break;
    }
  }
  return {for_relevant[chosen1], for_other[chosen0]};
}

DensityEstimate density_from_string(const std::string& name) {
  return name == to_string(DensityEstimate::Kernel) ? DensityEstimate::Kernel : DensityEstimate::LocalQuadratic;
}

}  // namespace

const char* to_string(DensityEstimate estimate) {
  switch (estimate) {
    case DensityEstimate::Kernel: return "kernel";
    case DensityEstimate::LocalQuadratic: return "local_quadratic";
  }
  return "local_quadratic";
}

Calibration::Calibration() : prior_(sigmoid(-kDefaultSlope * kDefaultMidpoint)) {}

Calibration Calibration::fallback(double midpoint, double slope) {
  Calibration c;
  c.fallback_midpoint_ = std::isfinite(midpoint) ? midpoint : kDefaultMidpoint;
  c.fallback_slope_ = (slope > 0.0 && std::isfinite(slope)) ? slope : kDefaultSlope;
  c.prior_ = sigmoid(-c.fallback_slope_ * c.fallback_midpoint_);
  return c;
}

Calibration Calibration::fit(std::span<const double> z_relevant, std::span<const double> z_nonrelevant, double prior,
                             std::size_t grid_points) {
  FitOptions options;
  options.grid_points = grid_points;
  return fit(z_relevant, z_nonrelevant, prior, options);
}

Calibration Calibration::fit(std::span<const double> z_relevant, std::span<const double> z_nonrelevant, double prior,
                             const FitOptions& options) {
  const Kde1d relevant = Kde1d::fit(z_relevant);
  const Kde1d other = Kde1d::fit(z_nonrelevant);
  if (relevant.size() == 0 || other.size() == 0 || !(prior > 0.0) || !(prior < 1.0)) return fallback();

  double low = std::min(relevant.samples().front(), other.samples().front());
  double high = std::max(relevant.samples().back(), other.samples().back());
  if (!(high > low)) {
    const double pad = std::max(relevant.bandwidth(), other.bandwidth());
    low -= pad;
    high += pad;
  }

  Calibration c;
  c.fitted_ = true;
  c.prior_ = prior;
  c.n_relevant_ = relevant.size();
  c.n_nonrelevant_ = other.size();
  c.density_ = options.density;
  if (options.density == DensityEstimate::LocalQuadratic) {
    const Fractions chosen = select_fractions(relevant, other, prior, options);
    c.fraction_relevant_ = chosen.relevant;
    c.fraction_nonrelevant_ = chosen.other;
  }
  const ClassDensity density_relevant(relevant, options.density, c.fraction_relevant_, options.min_neighbours);
  const ClassDensity density_other(other, options.density, c.fraction_nonrelevant_, options.min_neighbours);

  const std::size_t m = std::max<std::size_t>(2, options.grid_points);
  c.grid_.resize(m);
  c.ratio_.resize(m);
  std::vector<double> raw(m);
  std::vector<double> mass(m);
  const double prior_logit = logit(prior);
  for (std::size_t i = 0; i < m; ++i) {
    const double z = i + 1 == m ? high : low + (high - low) * (static_cast<double>(i) / static_cast<double>(m - 1));
    const double log_f1 = density_relevant(z);
    const double log_f0 = density_other(z);
    const double log_ratio = std::clamp(log_f1 - log_f0, -kMaxLogRatio, kMaxLogRatio);
    c.grid_[i] = z;
    c.ratio_[i] = std::exp(log_ratio);
    raw[i] = sigmoid(clamp_logit(prior_logit + log_ratio));
    mass[i] = prior * std::exp(log_f1) + (1.0 - prior) * std::exp(log_f0);
  }

  c.table_ = isotonic_nondecreasing(raw, mass);
  for (double& p : c.table_) p = std::clamp(p, 0.0, 1.0);
  return c;
}

double Calibration::posterior(double z) const {
  if (fitted_) return interpolate(grid_, table_, z, false);
  if (std::isnan(z)) return 0.0;
  return sigmoid(fallback_slope_ * (z - fallback_midpoint_));
}

double Calibration::likelihood_ratio(double z) const {
  if (fitted_) return interpolate(grid_, ratio_, z, true);
  if (std::isnan(z)) return std::exp(-kMaxLogRatio);
  return std::exp(std::clamp(fallback_slope_ * z, -kMaxLogRatio, kMaxLogRatio));
}

double Calibration::threshold(double tau) const {
  if (std::isnan(tau)) return kInfinity;
  const auto reaches = [&](double z) { return posterior(z) >= tau; };

  if (fitted_) {
    if (table_.front() >= tau) return -kInfinity;
    if (!(table_.back() >= tau)) return kInfinity;
    // First node at or above tau; the crossing lies in the segment that ends there.
    const auto node = std::lower_bound(table_.begin(), table_.end(), tau);
    const std::size_t i = static_cast<std::size_t>(node - table_.begin());
    return first_reaching(grid_[i - 1], grid_[i], reaches);
  }

  if (tau <= 0.0) return -kInfinity;
  if (tau >= 1.0) return kInfinity;  // a logistic curve never reaches 1
  const double guess = fallback_midpoint_ + (std::log(tau) - std::log1p(-tau)) / fallback_slope_;
  // Bracket the crossing around the analytic solution, then settle the last bits.
  double reach = 1e-9 * std::max(1.0, std::fabs(guess));
  for (int attempt = 0; attempt < 8; ++attempt, reach *= 64.0) {
    const double low = guess - reach;
    const double high = guess + reach;
    if (!reaches(low) && reaches(high)) return first_reaching(low, high, reaches);
  }
  return guess;
}

nlohmann::json Calibration::to_json() const {
  nlohmann::json j;
  j["fitted"] = fitted_;
  j["prior"] = prior_;
  j["relevant_samples"] = n_relevant_;
  j["nonrelevant_samples"] = n_nonrelevant_;
  j["fallback_midpoint"] = fallback_midpoint_;
  j["fallback_slope"] = fallback_slope_;
  j["density"] = to_string(density_);
  j["fraction_relevant"] = fraction_relevant_;
  j["fraction_nonrelevant"] = fraction_nonrelevant_;
  j["grid"] = grid_;
  j["table"] = table_;
  j["ratio"] = ratio_;
  return j;
}

Calibration Calibration::from_json(const nlohmann::json& j) {
  try {
    if (!j.is_object()) return fallback();
    const Calibration base = fallback(j.value("fallback_midpoint", kDefaultMidpoint),
                                      j.value("fallback_slope", kDefaultSlope));
    if (!j.value("fitted", false)) return base;

    Calibration c = base;
    c.fitted_ = true;
    c.prior_ = j.at("prior").get<double>();
    c.n_relevant_ = j.value("relevant_samples", std::size_t{0});
    c.n_nonrelevant_ = j.value("nonrelevant_samples", std::size_t{0});
    c.density_ = density_from_string(j.value("density", std::string(to_string(DensityEstimate::LocalQuadratic))));
    c.fraction_relevant_ = j.value("fraction_relevant", 0.0);
    c.fraction_nonrelevant_ = j.value("fraction_nonrelevant", 0.0);
    c.grid_ = j.at("grid").get<std::vector<double>>();
    c.table_ = j.at("table").get<std::vector<double>>();
    c.ratio_ = j.at("ratio").get<std::vector<double>>();

    const std::size_t m = c.grid_.size();
    const bool consistent =
        c.prior_ > 0.0 && c.prior_ < 1.0 && m >= 2 && c.table_.size() == m && c.ratio_.size() == m &&
        finite_vector(c.grid_) && finite_vector(c.table_) && finite_vector(c.ratio_) &&
        std::is_sorted(c.grid_.begin(), c.grid_.end()) && c.grid_.front() < c.grid_.back() &&
        std::is_sorted(c.table_.begin(), c.table_.end()) && c.table_.front() >= 0.0 && c.table_.back() <= 1.0 &&
        std::all_of(c.ratio_.begin(), c.ratio_.end(), [](double r) { return r > 0.0; });
    return consistent ? c : base;
  } catch (const nlohmann::json::exception&) {
    return fallback();
  }
}

Reliability reliability(const Calibration& calibration, std::span<const LabeledScore> samples, std::size_t bins) {
  const std::size_t b = std::max<std::size_t>(1, bins);
  Reliability out;
  out.bin_confidence.assign(b, 0.0);
  out.bin_accuracy.assign(b, 0.0);
  out.bin_count.assign(b, 0);
  if (samples.empty()) return out;

  double brier = 0.0;
  double log_loss = 0.0;
  for (const LabeledScore& sample : samples) {
    const double p = calibration.posterior(sample.z);
    const double y = sample.relevant ? 1.0 : 0.0;
    const std::size_t bin = std::min(b - 1, static_cast<std::size_t>(p * static_cast<double>(b)));
    out.bin_confidence[bin] += p;
    out.bin_accuracy[bin] += y;
    ++out.bin_count[bin];
    brier += (p - y) * (p - y);
    // -ln p = softplus(-l) and -ln(1 - p) = softplus(l) for l = logit p.
    const double l = logit(p);
    log_loss += softplus(sample.relevant ? -l : l);
  }

  const double n = static_cast<double>(samples.size());
  double ece = 0.0;
  for (std::size_t i = 0; i < b; ++i) {
    if (out.bin_count[i] == 0) continue;
    const double count = static_cast<double>(out.bin_count[i]);
    ece += std::fabs(out.bin_accuracy[i] - out.bin_confidence[i]);  // = count * |accuracy - confidence|
    out.bin_confidence[i] /= count;
    out.bin_accuracy[i] /= count;
  }
  out.expected_calibration_error = ece / n;
  out.brier = brier / n;
  out.log_loss = log_loss / n;
  return out;
}

}  // namespace prag::retrieve
