#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "prag/core/clock.hpp"
#include "prag/retrieve/calibration.hpp"
#include "prag/retrieve/kde1d.hpp"
#include "prag/retrieve/logodds.hpp"
#include "sampling.hpp"

using namespace prag;
using namespace prag::retrieve;

namespace {

const double kInfinity = std::numeric_limits<double>::infinity();

// The synthetic model of the acceptance tests: relevant scores N(5, 1),
// non-relevant scores N(0, 1), prior 0.002.
constexpr double kPrior = 0.002;
constexpr double kShift = 5.0;

double analytic_posterior(double z, double prior = kPrior, double shift = kShift) {
  // ln f1 / f0 = shift z - shift^2 / 2 for unit variances.
  const double l = std::log(prior / (1.0 - prior)) + shift * z - 0.5 * shift * shift;
  return 1.0 / (1.0 + std::exp(-l));
}

struct Sample {
  std::vector<double> relevant;
  std::vector<double> other;
};

Sample draw(std::uint64_t seed, std::size_t n_relevant = 400, std::size_t n_other = 20000) {
  std::mt19937_64 rng(seed);
  Sample s;
  s.relevant.resize(n_relevant);
  s.other.resize(n_other);
  for (double& z : s.relevant) z = test::normal(rng, kShift);
  for (double& z : s.other) z = test::normal(rng);
  return s;
}

// Fresh labelled scores with the class proportions of the prior.
std::vector<LabeledScore> draw_labelled(std::uint64_t seed, std::size_t n) {
  std::mt19937_64 rng(seed);
  std::vector<LabeledScore> out(n);
  for (LabeledScore& s : out) {
    s.relevant = test::uniform01(rng) < kPrior;
    s.z = test::normal(rng, s.relevant ? kShift : 0.0);
  }
  return out;
}

// Largest |fitted - analytic| where the analytic posterior is in [0.05, 0.95].
double worst_error(const Calibration& c) {
  double worst = 0.0;
  for (double z = -6.0; z <= 10.0; z += 0.001) {
    const double truth = analytic_posterior(z);
    if (truth >= 0.05 && truth <= 0.95) worst = std::max(worst, std::fabs(c.posterior(z) - truth));
  }
  return worst;
}

// One fit shared by the tests that only read it.
const Sample& sample() {
  static const Sample s = draw(20260929);
  return s;
}

const Calibration& fitted() {
  static const Calibration c = Calibration::fit(sample().relevant, sample().other, kPrior);
  return c;
}

// Measurements run in full when the environment variable PRAG_SLOW_TESTS is 1.
bool slow_tests() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  return value != nullptr && std::string(value) == "1";
}

}  // namespace

// ---- log-odds helpers ----------------------------------------------------------

PRAG_TEST(logodds_helpers_are_finite_and_consistent) {
  CHECK_EQ(logit(0.0), -kMaxLogit);
  CHECK_EQ(logit(1.0), kMaxLogit);
  CHECK_EQ(logit(-0.5), -kMaxLogit);
  CHECK_EQ(logit(2.0), kMaxLogit);
  CHECK_EQ(logit(std::numeric_limits<double>::quiet_NaN()), -kMaxLogit);
  CHECK_EQ(logit(0.5), 0.0);
  CHECK_EQ(clamp_logit(kInfinity), kMaxLogit);
  CHECK_EQ(clamp_logit(-kInfinity), -kMaxLogit);
  for (const double p : {1e-12, 0.002, 0.25, 0.5, 0.9, 1.0 - 1e-9}) {
    CHECK_NEAR(sigmoid(logit(p)), p, 1e-15 + 1e-12 * p);
    CHECK_NEAR(logit(p), std::log(p / (1.0 - p)), 1e-9);
    CHECK_NEAR(-softplus(logit(p)), std::log1p(-p), 1e-12);
    CHECK_NEAR(-softplus(-logit(p)), std::log(p), 1e-12);
    CHECK_NEAR(logit_from_log_survival(std::log1p(-p)), logit(p), 1e-9);
  }
  CHECK_EQ(sigmoid(-kInfinity), 0.0);
  CHECK_EQ(sigmoid(kInfinity), 1.0);
  CHECK_EQ(sigmoid(-800.0), 0.0);
  CHECK_NEAR(softplus(800.0), 800.0, 1e-12);
  CHECK_NEAR(softplus(-800.0), 0.0, 1e-300);
  CHECK_NEAR(log1m_exp(-1e-20), std::log(1e-20), 1e-9);
  CHECK_NEAR(log1m_exp(-50.0), -std::exp(-50.0), 1e-30);
  CHECK_EQ(logit_from_log_survival(0.0), -kMaxLogit);
  CHECK_EQ(logit_from_log_survival(-kInfinity), kMaxLogit);
  // Prior shift in log-odds equals the closed form of the posterior under a
  // changed base rate, p' = b' (p - p b) / (b - p b + b' p - b b').
  for (const double p : {0.01, 0.3, 0.8}) {
    for (const double b : {0.002, 0.1}) {
      for (const double b2 : {0.05, 0.5}) {
        const double closed = b2 * (p - p * b) / (b - p * b + b2 * p - b * b2);
        CHECK_NEAR(sigmoid(logit(p) - logit(b) + logit(b2)), closed, 1e-14);
      }
    }
  }
}

// ---- fallback -------------------------------------------------------------------

PRAG_TEST(fallback_is_the_documented_logistic_curve) {
  const Calibration standard;
  CHECK(!standard.fitted());
  CHECK(standard.grid().empty());
  CHECK_EQ(standard.relevant_samples(), std::size_t{0});
  const Calibration same = Calibration::fallback();
  const Calibration other = Calibration::fallback(2.5, 0.8);
  for (double z = -30.0; z <= 40.0; z += 0.37) {
    CHECK_NEAR(standard.posterior(z), 1.0 / (1.0 + std::exp(-1.5 * (z - 4.0))), 1e-15);
    CHECK_EQ(same.posterior(z), standard.posterior(z));
    CHECK_NEAR(other.posterior(z), 1.0 / (1.0 + std::exp(-0.8 * (z - 2.5))), 1e-15);
    CHECK_NEAR(other.likelihood_ratio(z), std::exp(0.8 * z), 1e-12 * std::exp(0.8 * z));
  }
  CHECK_EQ(standard.posterior(4.0), 0.5);
  // The prior is the posterior of an average unit, so that Bayes' rule with the
  // likelihood ratio gives the posterior back.
  CHECK_EQ(other.prior(), other.posterior(0.0));
  for (const double z : {-3.0, 0.0, 2.5, 7.0}) {
    const double odds = other.prior() / (1.0 - other.prior()) * other.likelihood_ratio(z);
    CHECK_NEAR(odds / (1.0 + odds), other.posterior(z), 1e-14);
  }
  CHECK_EQ(standard.posterior(kInfinity), 1.0);
  CHECK_EQ(standard.posterior(-kInfinity), 0.0);
  CHECK_EQ(standard.posterior(std::numeric_limits<double>::quiet_NaN()), 0.0);

  // Thresholds: the analytic inverse, consistent with posterior() to the last bit.
  for (const double tau : {1e-6, 0.1, 0.25, 0.5, 0.9, 1.0 - 1e-9}) {
    const double t = other.threshold(tau);
    // Near 0 and 1 the curve is flat, and one step of the posterior in the
    // last bit corresponds to a longer stretch of z.
    const double resolution = 1e-9 + 4e-16 / (0.8 * tau * (1.0 - tau));
    CHECK_NEAR(t, 2.5 + std::log(tau / (1.0 - tau)) / 0.8, resolution);
    CHECK(other.posterior(t) >= tau);
    CHECK(other.posterior(std::nextafter(t, -kInfinity)) < tau);
  }
  CHECK(other.threshold(0.0) == -kInfinity);
  CHECK(other.threshold(-1.0) == -kInfinity);
  CHECK(other.threshold(1.0) == kInfinity);
  CHECK(other.threshold(std::numeric_limits<double>::quiet_NaN()) == kInfinity);

  // Parameters that make no curve select the defaults.
  const Calibration repaired = Calibration::fallback(std::numeric_limits<double>::quiet_NaN(), -2.0);
  CHECK_EQ(repaired.posterior(1.0), standard.posterior(1.0));
}

PRAG_TEST(fit_without_usable_data_is_the_fallback) {
  const std::vector<double> some = {1.0, 2.0, 3.0};
  const std::vector<double> undefined = {std::numeric_limits<double>::quiet_NaN(), kInfinity};
  const Calibration standard;
  for (const Calibration& c : {Calibration::fit({}, some, 0.1), Calibration::fit(some, {}, 0.1),
                               Calibration::fit(some, undefined, 0.1), Calibration::fit(some, some, 0.0),
                               Calibration::fit(some, some, 1.0), Calibration::fit(some, some, -0.2),
                               Calibration::fit(some, some, std::numeric_limits<double>::quiet_NaN())}) {
    CHECK(!c.fitted());
    CHECK_EQ(c.posterior(3.3), standard.posterior(3.3));
  }
}

// ---- the fit, step by step --------------------------------------------------------

// The table must be what the header says: Bayes' rule on the class densities at
// the nodes of a uniform grid over the pooled scores, then the isotonic
// regression weighted by the mixture density.
PRAG_TEST(fit_is_bayes_rule_followed_by_isotonic_regression) {
  const Sample s = draw(3, 60, 900);
  const double prior = 0.05;
  const Kde1d f1 = Kde1d::fit(s.relevant);
  const Kde1d f0 = Kde1d::fit(s.other);
  const double low = std::min(f1.samples().front(), f0.samples().front());
  const double high = std::max(f1.samples().back(), f0.samples().back());

  for (const bool local : {false, true}) {
    FitOptions options;
    options.grid_points = 65;
    options.density = local ? DensityEstimate::LocalQuadratic : DensityEstimate::Kernel;
    options.fraction_relevant = 0.5;  // 30 of 60 neighbours
    options.fraction_nonrelevant = 0.1;  // 90 of 900
    const Calibration c = Calibration::fit(s.relevant, s.other, prior, options);
    REQUIRE(c.fitted());
    REQUIRE(c.grid().size() == 65u);
    CHECK_EQ(c.prior(), prior);
    CHECK_EQ(c.relevant_samples(), std::size_t{60});
    CHECK_EQ(c.nonrelevant_samples(), std::size_t{900});
    CHECK(c.density() == options.density);
    CHECK_EQ(c.fraction_relevant(), local ? 0.5 : 0.0);
    CHECK_EQ(c.fraction_nonrelevant(), local ? 0.1 : 0.0);
    CHECK_EQ(c.grid().front(), low);
    CHECK_EQ(c.grid().back(), high);

    std::vector<double> raw;
    std::vector<double> mass;
    std::vector<double> ratio;
    for (std::size_t i = 0; i < 65; ++i) {
      const double z = c.grid()[i];
      CHECK_NEAR(z, low + (high - low) * static_cast<double>(i) / 64.0, 1e-12);
      const double h1 = std::max(f1.bandwidth(), f1.neighbour_distance(z, 30));
      const double h0 = std::max(f0.bandwidth(), f0.neighbour_distance(z, 90));
      const double l1 = local ? f1.log_pdf_local(z, h1) : f1.log_pdf(z);
      const double l0 = local ? f0.log_pdf_local(z, h0) : f0.log_pdf(z);
      const double log_ratio = std::clamp(l1 - l0, -kMaxLogRatio, kMaxLogRatio);
      ratio.push_back(std::exp(log_ratio));
      raw.push_back(sigmoid(clamp_logit(std::log(prior / (1.0 - prior)) + log_ratio)));
      mass.push_back(prior * std::exp(l1) + (1.0 - prior) * std::exp(l0));
    }
    const std::vector<double> expected = isotonic_nondecreasing(raw, mass);
    for (std::size_t i = 0; i < 65; ++i) {
      CHECK_NEAR(c.table()[i], expected[i], 1e-12);
      CHECK_NEAR(c.posterior(c.grid()[i]), expected[i], 1e-12);
      CHECK_NEAR(c.likelihood_ratio(c.grid()[i]), ratio[i], 1e-9 * ratio[i]);
    }
    // Where the densities are well above zero, Bayes' rule in plain arithmetic.
    for (std::size_t i = 0; i < 65; ++i) {
      if (local) break;
      const double a = prior * f1.pdf(c.grid()[i]);
      const double b = (1.0 - prior) * f0.pdf(c.grid()[i]);
      if (a > 1e-200 && b > 1e-200) CHECK_NEAR(raw[i], a / (a + b), 1e-12);
    }

    // Between the nodes: linear for the posterior, geometric for the ratio;
    // outside the grid: the value at the end.
    const double middle = 0.5 * (c.grid()[20] + c.grid()[21]);
    CHECK_NEAR(c.posterior(middle), 0.5 * (expected[20] + expected[21]), 1e-12);
    CHECK_NEAR(c.likelihood_ratio(middle), std::sqrt(ratio[20] * ratio[21]), 1e-9 * ratio[20]);
    CHECK_EQ(c.posterior(low - 100.0), c.table().front());
    CHECK_EQ(c.posterior(high + 100.0), c.table().back());
    CHECK_EQ(c.posterior(-kInfinity), c.table().front());
    CHECK_EQ(c.posterior(kInfinity), c.table().back());
    CHECK_EQ(c.posterior(std::numeric_limits<double>::quiet_NaN()), c.table().front());
    CHECK_EQ(c.likelihood_ratio(high + 100.0), c.likelihood_ratio(high));
  }
}

PRAG_TEST(fit_handles_degenerate_samples) {
  // Both classes constant, and at the same place: the grid gets a width and
  // every value stays finite.
  const std::vector<double> ones(10, 1.0);
  const Calibration same = Calibration::fit(ones, ones, 0.3);
  REQUIRE(same.fitted());
  CHECK(same.grid().front() < same.grid().back());
  for (const double p : same.table()) CHECK(p >= 0.0 && p <= 1.0);
  CHECK_NEAR(same.posterior(1.0), 0.3, 1e-9);  // equal densities leave the prior

  // One score per class.
  const std::vector<double> high = {4.0};
  const std::vector<double> low = {0.0};
  const Calibration two = Calibration::fit(high, low, 0.5);
  REQUIRE(two.fitted());
  CHECK(two.posterior(0.0) < 0.01);
  CHECK(two.posterior(4.0) > 0.99);
  CHECK(std::is_sorted(two.table().begin(), two.table().end()));
  CHECK(std::isfinite(two.likelihood_ratio(2.0)));
  CHECK(two.likelihood_ratio(0.0) > 0.0);

  // Tiny classes pass through the cross-validation.
  std::mt19937_64 rng(5);
  for (const std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{25}}) {
    std::vector<double> a(n);
    std::vector<double> b(3 * n);
    for (double& z : a) z = test::normal(rng, 3.0);
    for (double& z : b) z = test::normal(rng);
    const Calibration c = Calibration::fit(a, b, 0.1);
    REQUIRE(c.fitted());
    CHECK(std::is_sorted(c.table().begin(), c.table().end()));
    for (const double p : c.table()) CHECK(p >= 0.0 && p <= 1.0);
  }

  // Fewer than two grid points are raised to two.
  const Calibration coarse = Calibration::fit(high, low, 0.5, std::size_t{0});
  CHECK_EQ(coarse.grid().size(), std::size_t{2});
}

PRAG_TEST(fit_is_well_defined_on_awkward_samples) {
  std::mt19937_64 rng(404);
  for (int trial = 0; trial < 120; ++trial) {
    const std::size_t n1 = 1 + static_cast<std::size_t>(rng() % 60);
    const std::size_t n0 = 1 + static_cast<std::size_t>(rng() % 300);
    // Offsets and scales over many orders of magnitude, scores rounded to a
    // coarse step (ties), classes overlapping, nested or far apart.
    const double scale = std::pow(10.0, static_cast<double>(rng() % 9) - 4.0);
    const double offset = (trial % 4 == 0 ? 1e6 : 0.0) * scale;
    const double separation = static_cast<double>(rng() % 5) - 1.0;
    const double step = trial % 3 == 0 ? 0.5 : 0.0;
    const auto score = [&](double mean, double sd) {
      double z = test::normal(rng, mean, sd);
      if (step > 0.0) z = std::round(z / step) * step;
      return offset + scale * z;
    };
    std::vector<double> relevant(n1);
    std::vector<double> other(n0);
    for (double& z : relevant) z = score(separation, trial % 5 == 0 ? 3.0 : 1.0);
    for (double& z : other) z = score(0.0, 1.0);
    const double prior = trial % 2 == 0 ? 0.002 : 0.3;

    for (const DensityEstimate estimate : {DensityEstimate::Kernel, DensityEstimate::LocalQuadratic}) {
      FitOptions options;
      options.density = estimate;
      options.grid_points = 33;
      const Calibration c = Calibration::fit(relevant, other, prior, options);
      REQUIRE(c.fitted());
      REQUIRE(c.table().size() == 33u);
      CHECK(std::is_sorted(c.grid().begin(), c.grid().end()));
      CHECK(c.grid().front() < c.grid().back());
      CHECK(std::is_sorted(c.table().begin(), c.table().end()));
      for (std::size_t i = 0; i < 33; ++i) {
        CHECK(c.table()[i] >= 0.0 && c.table()[i] <= 1.0);
        const double ratio = c.likelihood_ratio(c.grid()[i]);
        CHECK(std::isfinite(ratio) && ratio > 0.0);
      }
      const double t = c.threshold(0.25);
      if (std::isfinite(t)) {
        CHECK(c.posterior(t) >= 0.25);
        CHECK(c.posterior(std::nextafter(t, -kInfinity)) < 0.25);
      } else if (t > 0.0) {
        CHECK(c.table().back() < 0.25);
      } else {
        CHECK(c.table().front() >= 0.25);
      }
      // What is written can be read.
      CHECK(Calibration::from_json(nlohmann::json::parse(c.to_json().dump())).table() == c.table());
    }
  }
}

// ---- acceptance: synthetic scores ---------------------------------------------------

PRAG_TEST(fitted_posterior_is_close_to_the_analytic_posterior) {
  const Calibration& c = fitted();
  REQUIRE(c.fitted());
  CHECK_EQ(c.grid().size(), std::size_t{257});
  CHECK_EQ(c.relevant_samples(), std::size_t{400});
  CHECK_EQ(c.nonrelevant_samples(), std::size_t{20000});
  CHECK(c.density() == DensityEstimate::LocalQuadratic);

  // Within 0.1 wherever the analytic posterior is between 0.05 and 0.95
  // (z from 3.15 to 4.33).
  std::size_t compared = 0;
  for (double z = -6.0; z <= 10.0; z += 0.001) {
    const double truth = analytic_posterior(z);
    if (truth < 0.05 || truth > 0.95) continue;
    CHECK_NEAR(c.posterior(z), truth, 0.1);
    ++compared;
  }
  CHECK(compared > 1000u);
  CHECK(worst_error(c) < 0.1);

  // Far from the transition the posterior is settled.
  CHECK(c.posterior(0.0) < 1e-4);
  CHECK(c.posterior(2.0) < 0.01);
  CHECK(c.posterior(6.0) > 0.999);
  // The threshold of the trace: the analytic posterior is 0.25 at z = 3.523.
  CHECK_NEAR(c.threshold(0.25), (std::log(0.25 / 0.75) - std::log(kPrior / (1.0 - kPrior)) + 12.5) / 5.0, 0.1);
}

PRAG_TEST(fitted_posterior_is_monotone) {
  const Calibration& c = fitted();
  CHECK(std::is_sorted(c.table().begin(), c.table().end()));
  CHECK(std::is_sorted(c.grid().begin(), c.grid().end()));
  for (const double p : c.table()) CHECK(p >= 0.0 && p <= 1.0);
  double previous = -1.0;
  std::size_t reversals = 0;
  for (double z = c.grid().front() - 1.0; z <= c.grid().back() + 1.0; z += 0.0005) {
    const double p = c.posterior(z);
    if (p < previous) ++reversals;
    previous = p;
  }
  CHECK_EQ(reversals, std::size_t{0});
  // Also on the finest scale: consecutive doubles around the nodes.
  for (std::size_t i = 1; i + 1 < c.grid().size(); i += 7) {
    double z = c.grid()[i];
    for (int step = 0; step < 4; ++step) z = std::nextafter(z, -kInfinity);
    double last = c.posterior(z);
    for (int step = 0; step < 8; ++step) {
      z = std::nextafter(z, kInfinity);
      const double p = c.posterior(z);
      CHECK(p >= last);
      last = p;
    }
  }
}

PRAG_TEST(threshold_is_consistent_with_posterior) {
  const Calibration& c = fitted();
  for (const double tau : {1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double t = c.threshold(tau);
    REQUIRE(std::isfinite(t));
    CHECK(c.posterior(t) >= tau);
    CHECK(c.posterior(std::nextafter(t, -kInfinity)) < tau);
    // Everything above reaches tau, everything below does not.
    CHECK(c.posterior(t + 0.01) >= tau);
    CHECK(c.posterior(t - 0.01) < tau);
  }
  CHECK(c.threshold(0.1) < c.threshold(0.25));
  CHECK(c.threshold(0.25) < c.threshold(0.5));
  // Every z reaches a threshold at or below the smallest table value; none
  // reaches one above the largest.
  CHECK(c.threshold(c.table().front()) == -kInfinity);
  CHECK(c.threshold(0.0) == -kInfinity);
  CHECK(c.threshold(-1.0) == -kInfinity);
  CHECK(c.threshold(std::nextafter(c.table().back(), 2.0)) == kInfinity);
  CHECK(c.threshold(1.5) == kInfinity);
  CHECK(c.threshold(std::numeric_limits<double>::quiet_NaN()) == kInfinity);
  // A segment that reaches from negative to positive scores.
  nlohmann::json wide = c.to_json();
  wide["grid"] = std::vector<double>{-3000.0, -26.5, 202.25, 4000.0};
  wide["table"] = std::vector<double>{0.0, 0.2, 0.3, 0.9};
  wide["ratio"] = std::vector<double>{1.0, 2.0, 3.0, 4.0};
  const Calibration across = Calibration::from_json(wide);
  REQUIRE(across.fitted());
  CHECK_NEAR(across.threshold(0.25), 0.5 * (202.25 - 26.5), 1e-12);
  CHECK(across.posterior(across.threshold(0.25)) >= 0.25);
  CHECK(across.posterior(std::nextafter(across.threshold(0.25), -kInfinity)) < 0.25);
  CHECK_NEAR(across.threshold(0.1), -3000.0 + 0.5 * (3000.0 - 26.5), 1e-12);

  // A plateau: the threshold is where it begins.
  nlohmann::json j = c.to_json();
  j["grid"] = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  j["table"] = std::vector<double>{0.1, 0.4, 0.4, 0.8};
  j["ratio"] = std::vector<double>{1.0, 2.0, 2.0, 3.0};
  const Calibration steps = Calibration::from_json(j);
  REQUIRE(steps.fitted());
  // (To the rounding of the interpolation: the posterior may reach the value
  // of a node a few bits before the node.)
  CHECK_NEAR(steps.threshold(0.4), 1.0, 1e-14);
  CHECK_NEAR(steps.threshold(0.25), 0.5, 1e-14);
  CHECK_NEAR(steps.threshold(0.6), 2.5, 1e-14);
  CHECK_NEAR(steps.threshold(0.8), 3.0, 1e-14);
  for (const double tau : {0.25, 0.4, 0.6, 0.8}) {
    const double t = steps.threshold(tau);
    CHECK(steps.posterior(t) >= tau);
    CHECK(steps.posterior(std::nextafter(t, -kInfinity)) < tau);
  }
  CHECK_EQ(steps.posterior(1.5), 0.4);
}

PRAG_TEST(json_round_trip_is_exact) {
  const Calibration& c = fitted();
  const std::string text = c.to_json().dump();
  const Calibration back = Calibration::from_json(nlohmann::json::parse(text));
  REQUIRE(back.fitted());
  CHECK(back.grid() == c.grid());  // bit for bit
  CHECK(back.table() == c.table());
  CHECK_EQ(back.prior(), c.prior());
  CHECK_EQ(back.relevant_samples(), c.relevant_samples());
  CHECK_EQ(back.nonrelevant_samples(), c.nonrelevant_samples());
  CHECK(back.density() == c.density());
  CHECK_EQ(back.fraction_relevant(), c.fraction_relevant());
  CHECK_EQ(back.fraction_nonrelevant(), c.fraction_nonrelevant());
  for (double z = -7.0; z <= 11.0; z += 0.0137) {
    CHECK_EQ(back.posterior(z), c.posterior(z));
    CHECK_EQ(back.likelihood_ratio(z), c.likelihood_ratio(z));
  }
  for (const double tau : {0.1, 0.25, 0.5}) CHECK_EQ(back.threshold(tau), c.threshold(tau));
  CHECK_EQ(back.to_json().dump(), text);

  // The fallback keeps its parameters.
  const Calibration curve = Calibration::fallback(3.1, 0.7);
  const Calibration curve_back = Calibration::from_json(nlohmann::json::parse(curve.to_json().dump()));
  CHECK(!curve_back.fitted());
  for (double z = -5.0; z <= 12.0; z += 0.5) CHECK_EQ(curve_back.posterior(z), curve.posterior(z));
  CHECK_EQ(curve_back.prior(), curve.prior());
}

PRAG_TEST(from_json_rejects_inconsistent_documents) {
  const Calibration standard;
  const auto is_fallback = [&](const nlohmann::json& j) {
    const Calibration c = Calibration::from_json(j);
    return !c.fitted() && c.posterior(3.0) == standard.posterior(3.0);
  };
  CHECK(is_fallback(nlohmann::json()));
  CHECK(is_fallback(nlohmann::json::array()));
  CHECK(is_fallback(nlohmann::json::object()));
  CHECK(is_fallback(nlohmann::json("calibration")));

  const nlohmann::json good = fitted().to_json();
  CHECK(Calibration::from_json(good).fitted());
  const auto broken = [&](const char* key, const nlohmann::json& value) {
    nlohmann::json j = good;
    j[key] = value;
    return !Calibration::from_json(j).fitted();
  };
  CHECK(broken("grid", std::vector<double>{0.0, 1.0}));          // lengths differ
  CHECK(broken("table", "text"));                                // wrong type
  CHECK(broken("prior", 0.0));
  CHECK(broken("prior", 1.5));
  CHECK(broken("prior", "half"));
  nlohmann::json j = good;
  j.erase("ratio");
  CHECK(!Calibration::from_json(j).fitted());
  j = good;
  j["table"][10] = 2.0;                                          // not a probability, not monotone
  CHECK(!Calibration::from_json(j).fitted());
  j = good;
  j["grid"][5] = j["grid"][200];                                 // not ascending
  CHECK(!Calibration::from_json(j).fitted());
  j = good;
  j["ratio"][0] = -1.0;
  CHECK(!Calibration::from_json(j).fitted());
  j = good;
  j["table"][3] = nullptr;                                       // what a NaN is written as
  CHECK(!Calibration::from_json(j).fitted());
}

// ---- reliability ------------------------------------------------------------------

PRAG_TEST(reliability_hand_example) {
  // Posterior = logistic function of z. Four scores in two bins of width 1/2.
  const Calibration c = Calibration::fallback(0.0, 1.0);
  const double z_low = std::log(0.2 / 0.8);
  const double z_high = std::log(0.9 / 0.1);
  const std::vector<LabeledScore> samples = {{z_low, false}, {z_low, true}, {z_high, true}, {z_high, true}};
  const Reliability r = reliability(c, samples, 2);
  REQUIRE(r.bin_count.size() == 2u);
  CHECK_EQ(r.bin_count[0], std::size_t{2});
  CHECK_EQ(r.bin_count[1], std::size_t{2});
  CHECK_NEAR(r.bin_confidence[0], 0.2, 1e-12);
  CHECK_NEAR(r.bin_confidence[1], 0.9, 1e-12);
  CHECK_NEAR(r.bin_accuracy[0], 0.5, 1e-12);
  CHECK_NEAR(r.bin_accuracy[1], 1.0, 1e-12);
  CHECK_NEAR(r.expected_calibration_error, 0.5 * 0.3 + 0.5 * 0.1, 1e-12);
  CHECK_NEAR(r.brier, (0.04 + 0.64 + 0.01 + 0.01) / 4.0, 1e-12);
  CHECK_NEAR(r.log_loss, -(std::log(0.8) + std::log(0.2) + 2.0 * std::log(0.9)) / 4.0, 1e-12);

  // p = 1 belongs to the last bin; a certain and wrong prediction has a finite loss.
  const std::vector<LabeledScore> certain = {{1000.0, false}, {-1000.0, false}};
  const Reliability wrong = reliability(c, certain, 10);
  CHECK_EQ(wrong.bin_count[9], std::size_t{1});
  CHECK_EQ(wrong.bin_count[0], std::size_t{1});
  CHECK_NEAR(wrong.brier, 0.5, 1e-12);
  CHECK(std::isfinite(wrong.log_loss));
  CHECK_NEAR(wrong.log_loss, 0.5 * kMaxLogit, 1e-9);

  const Reliability empty = reliability(c, {}, 10);
  CHECK_EQ(empty.bin_count.size(), std::size_t{10});
  CHECK_EQ(empty.expected_calibration_error, 0.0);
  CHECK_EQ(reliability(c, samples, 0).bin_count.size(), std::size_t{1});
}

PRAG_TEST(reliability_on_fresh_samples) {
  const Calibration& c = fitted();
  const std::vector<LabeledScore> fresh = draw_labelled(77, 300000);
  std::size_t relevant = 0;
  for (const LabeledScore& s : fresh) relevant += s.relevant ? 1 : 0;
  CHECK(relevant > 500u && relevant < 700u);  // 0.002 of 300000

  const Reliability r = reliability(c, fresh);
  CHECK(r.expected_calibration_error < 0.05);
  CHECK(r.expected_calibration_error < 0.001);  // what was measured is far below the bound
  REQUIRE(r.bin_count.size() == 10u);
  std::size_t total = 0;
  double ece = 0.0;
  for (std::size_t b = 0; b < 10; ++b) {
    total += r.bin_count[b];
    ece += static_cast<double>(r.bin_count[b]) * std::fabs(r.bin_accuracy[b] - r.bin_confidence[b]);
    if (r.bin_count[b] == 0) continue;
    CHECK(r.bin_confidence[b] >= static_cast<double>(b) / 10.0);
    CHECK(r.bin_confidence[b] <= static_cast<double>(b + 1) / 10.0);
  }
  CHECK_EQ(total, fresh.size());
  CHECK_NEAR(r.expected_calibration_error, ece / static_cast<double>(fresh.size()), 1e-15);
  // The bins where the decisions fall are populated and honest: observed
  // frequency within 0.15 of the stated probability wherever there are at
  // least 50 scores.
  for (std::size_t b = 0; b < 10; ++b) {
    if (r.bin_count[b] >= 50) CHECK_NEAR(r.bin_accuracy[b], r.bin_confidence[b], 0.15);
  }

  // The analytic posterior as a yardstick for the two proper scores.
  double brier = 0.0;
  double log_loss = 0.0;
  for (const LabeledScore& s : fresh) {
    const double p = analytic_posterior(s.z);
    brier += (p - (s.relevant ? 1.0 : 0.0)) * (p - (s.relevant ? 1.0 : 0.0));
    log_loss -= std::log(s.relevant ? p : 1.0 - p);
  }
  brier /= static_cast<double>(fresh.size());
  log_loss /= static_cast<double>(fresh.size());
  CHECK_NEAR(r.brier, brier, 0.05 * brier);
  CHECK_NEAR(r.log_loss, log_loss, 0.05 * log_loss);

  // A calibration fitted to scores that were 2.5 too low: it promises relevance
  // where there is little.
  std::vector<double> relevant_low = sample().relevant;
  std::vector<double> other_low = sample().other;
  for (double& z : relevant_low) z -= 2.5;
  for (double& z : other_low) z -= 2.5;
  FitOptions options;
  options.fraction_relevant = c.fraction_relevant();
  options.fraction_nonrelevant = c.fraction_nonrelevant();
  const Calibration shifted = Calibration::fit(relevant_low, other_low, kPrior, options);
  for (const double z : {3.0, 3.5, 4.0}) CHECK_NEAR(shifted.posterior(z - 2.5), c.posterior(z), 1e-9);
  const Reliability bad = reliability(shifted, fresh);
  CHECK(bad.expected_calibration_error > 0.05);
  CHECK(bad.expected_calibration_error > 100.0 * r.expected_calibration_error);
  CHECK(bad.brier > 10.0 * r.brier);
  CHECK(bad.log_loss > 10.0 * r.log_loss);
}

// ---- choice of the density estimate --------------------------------------------------

namespace {

// A pair of class densities of unit variance, five standard deviations apart,
// with the posterior they imply. Normal tails fall like exp(-z^2 / 2), those of
// the logistic distribution like exp(-1.81 z).
struct Model {
  const char* name;
  bool normal;

  double log_density(double x) const {
    if (normal) return -0.5 * x * x - 0.5 * std::log(2.0 * 3.14159265358979323846);
    const double s = std::sqrt(3.0) / 3.14159265358979323846;  // scale of unit variance
    return -x / s - 2.0 * softplus(-x / s) - std::log(s);
  }
  double posterior(double z) const {
    return sigmoid(logit(kPrior) + log_density(z - kShift) - log_density(z));
  }
  Sample draw(std::uint64_t seed) const {
    if (normal) return ::draw(seed);
    std::mt19937_64 rng(seed);
    Sample out;
    out.relevant.resize(400);
    out.other.resize(20000);
    for (double& z : out.relevant) z = test::logistic(rng, kShift);
    for (double& z : out.other) z = test::logistic(rng);
    return out;
  }
  // Largest |fitted - true| where the true posterior is in [0.05, 0.95].
  double worst_error(const Calibration& c) const {
    double worst = 0.0;
    for (double z = -6.0; z <= 12.0; z += 0.001) {
      const double truth = posterior(z);
      if (truth >= 0.05 && truth <= 0.95) worst = std::max(worst, std::fabs(c.posterior(z) - truth));
    }
    return worst;
  }
  // Integral of (fitted - true)^2 against the density of the scores: what the
  // Brier score of the fit exceeds that of the true posterior by.
  double excess_brier(const Calibration& c) const {
    double sum = 0.0;
    const double step = 0.002;
    for (double z = -14.0; z <= 20.0; z += step) {
      const double mixture = kPrior * std::exp(log_density(z - kShift)) + (1.0 - kPrior) * std::exp(log_density(z));
      const double error = c.posterior(z) - posterior(z);
      sum += mixture * error * error * step;
    }
    return sum;
  }
};

const Model kNormal{"normal", true};
const Model kLogistic{"logistic", false};

}  // namespace

PRAG_TEST(window_is_chosen_by_cross_validation) {
  // Normal classes: the global fit of the non-relevant class is not
  // contradicted and is chosen. (The relevant class has 400 scores, and its
  // choice varies with the sample: 0.1 here, 1 for most other samples.)
  const Calibration& normal = fitted();
  CHECK_EQ(normal.fraction_nonrelevant(), 1.0);
  CHECK(normal.fraction_relevant() >= 0.1);
  CHECK_NEAR(kNormal.posterior(3.7), analytic_posterior(3.7), 1e-12);

  // Classes with exponential tails: a normal curve fitted to the non-relevant
  // class underestimates its upper tail, where the decisions fall, and the
  // cross-validation keeps the window local.
  const Sample s = kLogistic.draw(31);
  const Calibration heavy = Calibration::fit(s.relevant, s.other, kPrior);
  CHECK_EQ(heavy.fraction_nonrelevant(), 0.0);
  FitOptions global;
  global.fraction_relevant = 1.0;
  global.fraction_nonrelevant = 1.0;
  const Calibration forced = Calibration::fit(s.relevant, s.other, kPrior, global);
  CHECK_EQ(forced.fraction_nonrelevant(), 1.0);
  CHECK(kLogistic.excess_brier(heavy) < 0.2 * kLogistic.excess_brier(forced));
  CHECK(kLogistic.worst_error(heavy) < 0.5 * kLogistic.worst_error(forced));

  // Options: fixed fractions are taken as given, candidates are honoured, and
  // a tolerance of zero takes the smallest loss.
  FitOptions options;
  options.fraction_relevant = 0.25;
  options.candidates = {0.5, 0.05};
  const Calibration mixed = Calibration::fit(sample().relevant, sample().other, kPrior, options);
  CHECK_EQ(mixed.fraction_relevant(), 0.25);
  CHECK(mixed.fraction_nonrelevant() == 0.5 || mixed.fraction_nonrelevant() == 0.05);
  options = FitOptions{};
  options.candidates = {0.2, -3.0, 7.0, std::numeric_limits<double>::quiet_NaN()};  // one valid entry
  options.cv_tolerance = 0.0;
  const Calibration single = Calibration::fit(sample().relevant, sample().other, kPrior, options);
  CHECK_EQ(single.fraction_relevant(), 0.2);
  CHECK_EQ(single.fraction_nonrelevant(), 0.2);
  CHECK(std::string(to_string(DensityEstimate::Kernel)) == "kernel");
  CHECK(std::string(to_string(DensityEstimate::LocalQuadratic)) == "local_quadratic");
}

// ---- measurements ----------------------------------------------------------------

// How often does the criterion of the acceptance test hold over independent
// calibration sets of 400 relevant and 20000 non-relevant scores? Reports the
// kernel estimate the design started from, the local estimate with a fixed
// window, and the default, for normal classes and for classes with exponential
// tails. PRAG_SLOW_TESTS=1.
PRAG_TEST(measure_accuracy_over_many_calibration_sets) {
  if (!slow_tests()) {
    std::printf("  skipped: set PRAG_SLOW_TESTS=1 to run the measurement\n");
    return;
  }
  const int sets = 100;
  struct Variant {
    const char* name;
    FitOptions options;
  };
  std::vector<Variant> variants(4);
  variants[0].name = "kernel estimate, Silverman bandwidth";
  variants[0].options.density = DensityEstimate::Kernel;
  variants[1].name = "local estimate, 20 neighbours";
  variants[1].options.fraction_relevant = 0.0;
  variants[1].options.fraction_nonrelevant = 0.0;
  variants[2].name = "local estimate, all neighbours";
  variants[2].options.fraction_relevant = 1.0;
  variants[2].options.fraction_nonrelevant = 1.0;
  variants[3].name = "local estimate, cross-validated (default)";

  for (const Model* model : {&kNormal, &kLogistic}) {
    const double true_threshold = [&] {
      double z = 0.0;
      while (model->posterior(z) < 0.25) z += 1e-4;
      return z;
    }();
    std::printf("  %s classes; the true posterior is 0.25 at z = %.3f\n", model->name, true_threshold);
    for (const Variant& variant : variants) {
      std::vector<double> worst;
      std::vector<double> thresholds;
      double brier = 0.0;
      double seconds = 0.0;
      int global = 0;
      int local = 0;
      for (int set = 0; set < sets; ++set) {
        const Sample s = model->draw(1000 + static_cast<std::uint64_t>(set));
        const Stopwatch watch;
        const Calibration c = Calibration::fit(s.relevant, s.other, kPrior, variant.options);
        seconds += watch.elapsed_s();
        worst.push_back(model->worst_error(c));
        thresholds.push_back(c.threshold(0.25));
        brier += model->excess_brier(c) / sets;
        global += c.fraction_nonrelevant() == 1.0 ? 1 : 0;
        local += c.fraction_nonrelevant() == 0.0 ? 1 : 0;
      }
      std::sort(worst.begin(), worst.end());
      std::sort(thresholds.begin(), thresholds.end());
      const auto within = std::count_if(worst.begin(), worst.end(), [](double w) { return w <= 0.1; });
      std::printf("    %-42s within 0.1 in %3d of %d sets; worst error median %.3f, 90%% %.3f, max %.3f\n",
                  variant.name, static_cast<int>(within), sets, worst[worst.size() / 2], worst[worst.size() * 9 / 10],
                  worst.back());
      std::printf("    %-42s z(0.25) from %.3f to %.3f; excess Brier score %.2e; %.2f s per fit\n", "",
                  thresholds.front(), thresholds.back(), brier, seconds / sets);
      if (variant.options.density == DensityEstimate::LocalQuadratic && variant.options.fraction_relevant < 0.0) {
        std::printf("    %-42s window of the non-relevant class: global in %d sets, 20 neighbours in %d\n", "", global,
                    local);
      }
    }
  }
}
