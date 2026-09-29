// Calibration of similarity scores into relevance probabilities.
//
// For a query q the scan yields a score s_v for every unit v. Scores are first
// standardised against the background of the query itself,
//
//     z_v = (s_v - mean_q) / stddev_q,
//
// which removes the query-dependent offset and scale. The class-conditional
// densities f1(z) = p(z | relevant) and f0(z) = p(z | not relevant) are
// estimated by Gaussian kernel density estimation on a calibration set, and
// Bayes' rule with prior pi gives
//
//     P(relevant | z) = pi f1(z) / ( pi f1(z) + (1 - pi) f0(z) ).
//
// The posterior is tabulated on a grid and made non-decreasing in z by isotonic
// regression, so that "walk down the ranking until the probability drops below
// tau" has exactly one stopping point.
//
// Details of the estimate:
//
//   * Densities. Relevant units are rare, so the posterior rises from 0 to 1 in
//     the far upper tail of f0, where a calibration set holds a handful of
//     scores. A kernel estimate with a bandwidth chosen for the bulk is too
//     noisy there (DensityEstimate::Kernel is kept for comparison). The default
//     is the local log-quadratic likelihood estimate (Kde1d::log_pdf_local)
//     with the kernel width
//
//         h(z) = max( Silverman bandwidth, d_k(z) ),   k = max(k_min, ceil(a n)),
//
//     d_k(z) the distance from z to its k-th nearest score of the class. The
//     neighbour fraction a runs from local (a = 0: k_min neighbours) to global
//     (a = 1: every score of the class has a weight of at least exp(-1/2), and
//     the estimate is close to one normal curve fitted to the class).
//   * Choice of a. Unless given, a is chosen per class by leave-one-out
//     cross-validation of the quantity that matters, the log loss of the
//     posterior under the class proportions of the prior,
//
//         L = - pi / n1 sum_i ln P(z_i)  -  (1 - pi) / n0 sum_j ln(1 - P(z_j)),
//
//     i over relevant and j over non-relevant calibration scores, the density
//     of the own class estimated without the score itself. The sums run over
//     the lowest relevant and the highest non-relevant scores, where the
//     posterior is away from 0 and 1. Among the candidates whose loss is
//     within `cv_tolerance` standard errors of the smallest, the widest is
//     taken (one-standard-error rule), first for the non-relevant class, then
//     for the relevant one.
//   * The grid is uniform and spans the pooled calibration scores, from the
//     smallest to the largest. Outside of it the posterior keeps the value at
//     the nearer end; no density estimate is extrapolated beyond the data.
//   * The posterior is computed as sigmoid(logit(pi) + ln f1 - ln f0) from
//     log-densities, so it is defined where the densities underflow.
//   * The isotonic regression weights grid point i by the mixture density
//     pi f1(z_i) + (1 - pi) f0(z_i). The table is therefore the non-decreasing
//     function closest to the Bayes posterior in mean squared error under the
//     distribution of the scores the calibration will be applied to.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <nlohmann/json.hpp>

namespace prag::retrieve {

struct LabeledScore {
  double z = 0.0;
  bool relevant = false;
};

// How the class-conditional densities are estimated.
enum class DensityEstimate {
  Kernel,          // Gaussian kernel estimate, Silverman bandwidth (Kde1d::log_pdf)
  LocalQuadratic,  // local log-quadratic likelihood, nearest-neighbour width (Kde1d::log_pdf_local)
};

const char* to_string(DensityEstimate estimate);

struct FitOptions {
  std::size_t grid_points = 257;  // raised to 2 when smaller
  DensityEstimate density = DensityEstimate::LocalQuadratic;

  // The remaining options apply to LocalQuadratic.
  // Neighbour fraction per class, in [0, 1]. A negative value (the default)
  // lets the fit choose among `candidates` by cross-validation.
  double fraction_relevant = -1.0;
  double fraction_nonrelevant = -1.0;
  std::size_t min_neighbours = 20;  // k_min
  // Neighbour fractions to choose from. Empty selects {0, 0.03, 0.1, 0.3, 1}.
  std::vector<double> candidates;
  std::size_t cv_points = 256;         // scores per class entering the cross-validation
  std::size_t cv_max_samples = 20000;  // larger classes are thinned (evenly in rank) for the choice of a
  double cv_tolerance = 1.0;           // in standard errors of the loss difference; 0 = smallest loss
};

class Calibration {
 public:
  Calibration();  // the uncalibrated default, see fallback()

  // prior: P(relevant) for a unit drawn at random from the corpus.
  // Returns fallback() when a class has no finite score or when the prior is
  // not inside (0, 1).
  static Calibration fit(std::span<const double> z_relevant, std::span<const double> z_nonrelevant, double prior,
                         std::size_t grid_points = 257);
  static Calibration fit(std::span<const double> z_relevant, std::span<const double> z_nonrelevant, double prior,
                         const FitOptions& options);

  // Logistic posterior used before any calibration data exist:
  // P = 1 / (1 + exp(-slope * (z - midpoint))).
  // Its prior() is the posterior of an average unit, P(z = 0), which makes its
  // likelihood ratio exp(slope * z). A slope that is not positive and finite,
  // or a midpoint that is not finite, selects the default value.
  static Calibration fallback(double midpoint = 4.0, double slope = 1.5);

  // NaN counts as the lowest possible score.
  double posterior(double z) const;         // non-decreasing in z, in [0, 1]
  // The estimate of the ratio as tabulated on the grid, interpolated linearly
  // in ln(f1 / f0) and constant outside the grid. It is the raw estimate:
  // unlike posterior() it is not forced to be monotone. Bounded to
  // exp(+-kMaxLogRatio) (logodds.hpp).
  double likelihood_ratio(double z) const;  // f1(z) / f0(z)
  // Smallest z whose posterior reaches tau; +infinity when no z does and
  // -infinity when every z does. Consistent with posterior():
  // posterior(threshold(tau)) >= tau, and posterior(z) < tau for the largest
  // double below the threshold.
  double threshold(double tau) const;

  double prior() const noexcept { return prior_; }
  bool fitted() const noexcept { return fitted_; }
  std::size_t relevant_samples() const noexcept { return n_relevant_; }
  std::size_t nonrelevant_samples() const noexcept { return n_nonrelevant_; }
  const std::vector<double>& grid() const noexcept { return grid_; }
  const std::vector<double>& table() const noexcept { return table_; }

  // The density estimate behind a fitted calibration, and the neighbour
  // fractions it used (chosen or given; 0 for DensityEstimate::Kernel).
  DensityEstimate density() const noexcept { return density_; }
  double fraction_relevant() const noexcept { return fraction_relevant_; }
  double fraction_nonrelevant() const noexcept { return fraction_nonrelevant_; }

  // Numbers are written with the shortest representation that reads back to
  // the same double, so from_json(to_json()) reproduces every value exactly,
  // also through dump() and parse(). from_json never throws: a document that
  // is not a consistent calibration yields fallback().
  nlohmann::json to_json() const;
  static Calibration from_json(const nlohmann::json& j);

 private:
  bool fitted_ = false;
  double prior_ = 0.0;
  std::size_t n_relevant_ = 0;
  std::size_t n_nonrelevant_ = 0;
  double fallback_midpoint_ = 4.0;
  double fallback_slope_ = 1.5;
  std::vector<double> grid_;        // ascending z
  std::vector<double> table_;       // posterior at grid_, non-decreasing
  std::vector<double> ratio_;       // likelihood ratio at grid_
  DensityEstimate density_ = DensityEstimate::LocalQuadratic;
  double fraction_relevant_ = 0.0;
  double fraction_nonrelevant_ = 0.0;
};

// Reliability of a calibration on held-out labelled scores.
struct Reliability {
  double expected_calibration_error = 0.0;  // sum_b (n_b / n) |accuracy_b - confidence_b|
  double brier = 0.0;                        // mean (p - y)^2
  double log_loss = 0.0;                     // mean -ln P(observed label), logits clamped (logodds.hpp)
  std::vector<double> bin_confidence;        // mean predicted probability per bin
  std::vector<double> bin_accuracy;          // observed fraction relevant per bin
  std::vector<std::size_t> bin_count;
};

// Bins are equal-width in probability. Bin b holds the samples with
// b / bins <= p < (b + 1) / bins; p = 1 belongs to the last bin. Empty bins
// report zero confidence and accuracy. bins = 0 is treated as 1.
Reliability reliability(const Calibration& calibration, std::span<const LabeledScore> samples,
                        std::size_t bins = 10);

}  // namespace prag::retrieve
