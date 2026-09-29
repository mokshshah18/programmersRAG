#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "prag/retrieve/kde1d.hpp"
#include "sampling.hpp"

using namespace prag::retrieve;

namespace {

const double kSqrt2Pi = std::sqrt(2.0 * 3.14159265358979323846);

std::vector<double> normal_sample(std::size_t n, double mean, double sd, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<double> out(n);
  for (double& x : out) x = prag::test::normal(rng, mean, sd);
  return out;
}

// The definition, summed over every sample.
double direct_pdf(const std::vector<double>& samples, double h, double x) {
  double sum = 0.0;
  for (const double s : samples) {
    const double u = (x - s) / h;
    sum += std::exp(-0.5 * u * u);
  }
  return sum / (static_cast<double>(samples.size()) * h * kSqrt2Pi);
}

// Trapezoid rule with step h / 32 over everything the kernels reach.
double integral(const Kde1d& kde) {
  const double step = kde.bandwidth() / 32.0;
  const double low = kde.samples().front() - 12.0 * kde.bandwidth();
  const double high = kde.samples().back() + 12.0 * kde.bandwidth();
  const std::size_t steps = static_cast<std::size_t>(std::ceil((high - low) / step));
  double sum = 0.5 * (kde.pdf(low) + kde.pdf(low + static_cast<double>(steps) * step));
  for (std::size_t i = 1; i < steps; ++i) sum += kde.pdf(low + static_cast<double>(i) * step);
  return sum * step;
}

double squared_error(const std::vector<double>& fit, const std::vector<double>& y, const std::vector<double>& w) {
  double sum = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) sum += w[i] * (fit[i] - y[i]) * (fit[i] - y[i]);
  return sum;
}

bool non_decreasing(const std::vector<double>& v) { return std::is_sorted(v.begin(), v.end()); }

}  // namespace

// ---- kernel density estimate -------------------------------------------------

PRAG_TEST(density_integrates_to_one) {
  const Kde1d silverman = Kde1d::fit(normal_sample(300, 2.0, 1.5, 11));
  CHECK_NEAR(integral(silverman), 1.0, 1e-9);
  const Kde1d given = Kde1d::fit(normal_sample(50, -1.0, 0.2, 12), 0.05);
  CHECK_EQ(given.bandwidth(), 0.05);
  CHECK_NEAR(integral(given), 1.0, 1e-9);
  // Two far clusters: the window of the sum must not lose either of them.
  std::vector<double> split = normal_sample(40, 0.0, 0.1, 13);
  for (const double x : normal_sample(40, 500.0, 0.1, 14)) split.push_back(x);
  const Kde1d two = Kde1d::fit(split, 0.05);
  double sum = 0.0;
  for (double x = -2.0; x <= 2.0; x += 0.05 / 32.0) sum += two.pdf(x) * (0.05 / 32.0);
  CHECK_NEAR(sum, 0.5, 1e-9);
}

PRAG_TEST(density_equals_the_direct_formula) {
  const std::vector<double> samples = normal_sample(200, 0.0, 1.0, 21);
  const Kde1d kde = Kde1d::fit(samples);
  CHECK(std::is_sorted(kde.samples().begin(), kde.samples().end()));
  CHECK_EQ(kde.size(), samples.size());
  for (double x = -9.0; x <= 9.0; x += 0.37) {
    const double expected = direct_pdf(samples, kde.bandwidth(), x);
    CHECK_NEAR(kde.pdf(x), expected, 1e-13 * expected + 1e-300);
    if (expected > 1e-290) CHECK_NEAR(kde.log_pdf(x), std::log(expected), 1e-11);
  }
  // A narrow kernel far from the data: the density underflows, its logarithm does not.
  const Kde1d narrow = Kde1d::fit(samples, 0.01);
  const double top = narrow.samples().back();
  const double x = top + 1.0;  // 100 bandwidths above the largest sample
  CHECK_EQ(narrow.pdf(x), 0.0);
  CHECK_EQ(direct_pdf(samples, 0.01, x), 0.0);
  // Only the largest sample counts at this distance from the second largest.
  const double second = narrow.samples()[narrow.size() - 2];
  REQUIRE(top - second > 0.003);
  const double expected = -0.5 * 100.0 * 100.0 - std::log(200.0 * 0.01 * kSqrt2Pi);
  CHECK_NEAR(narrow.log_pdf(x), expected, 1e-9);
}

PRAG_TEST(density_handles_undefined_input) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> dirty = {1.0, nan, 2.0, inf, 3.0, -inf};
  const Kde1d kde = Kde1d::fit(dirty);
  CHECK_EQ(kde.size(), std::size_t{3});
  CHECK_EQ(kde.pdf(nan), 0.0);
  CHECK_EQ(kde.pdf(inf), 0.0);
  CHECK(kde.log_pdf(nan) == -inf);
  CHECK(kde.pdf(2.0) > 0.0);

  const Kde1d empty = Kde1d::fit({});
  CHECK_EQ(empty.size(), std::size_t{0});
  CHECK_EQ(empty.pdf(0.0), 0.0);
  CHECK(empty.log_pdf(0.0) == -inf);
  CHECK(empty.log_pdf_local(0.0) == -inf);
  CHECK_EQ(empty.neighbour_distance(0.0, 3), 0.0);
  CHECK(empty.bandwidth() > 0.0);
}

// ---- Silverman's rule --------------------------------------------------------

PRAG_TEST(silverman_bandwidth_on_known_samples) {
  // 1..9: sd = sqrt(7.5), quartiles 3 and 7, so IQR / 1.34 = 2.985 > sd.
  const std::vector<double> ramp = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const double n_factor = std::pow(9.0, -0.2);
  CHECK_NEAR(silverman_bandwidth(ramp), 0.9 * std::sqrt(7.5) * n_factor, 1e-12);
  CHECK_NEAR(silverman_bandwidth(ramp), 1.58828, 1e-5);  // the same number, worked out by hand

  // The order of the samples does not matter.
  const std::vector<double> shuffled = {7, 1, 9, 3, 5, 2, 8, 4, 6};
  CHECK_EQ(silverman_bandwidth(shuffled), silverman_bandwidth(ramp));

  // An outlier inflates sd; the quartiles (0 and 1) decide: IQR / 1.34 = 0.7463.
  const std::vector<double> outlier = {0, 0, 0, 0, 1, 1, 1, 1, 100};
  CHECK_NEAR(silverman_bandwidth(outlier), 0.9 * (1.0 / 1.34) * n_factor, 1e-12);

  // Quartiles interpolate between order statistics: for 1, 2, 4, 8 they are
  // 1.75 and 5, IQR / 1.34 = 2.4254, and sd = 3.0957 is the larger one.
  const std::vector<double> four = {1, 2, 4, 8};
  CHECK_NEAR(silverman_bandwidth(four), 0.9 * (3.25 / 1.34) * std::pow(4.0, -0.2), 1e-12);

  // Coinciding quartiles, but not a constant sample: sd alone.
  const std::vector<double> spike = {5, 5, 5, 5, 5, 5, 5, 5, 9};
  const double sd = std::sqrt((8.0 * (4.0 / 9.0) * (4.0 / 9.0) + (32.0 / 9.0) * (32.0 / 9.0)) / 8.0);
  CHECK_NEAR(silverman_bandwidth(spike), 0.9 * sd * n_factor, 1e-12);

  // A large normal sample: close to 0.9 sigma n^{-1/5}.
  const std::vector<double> large = normal_sample(20000, 0.0, 2.0, 31);
  CHECK_NEAR(silverman_bandwidth(large), 0.9 * 2.0 * std::pow(20000.0, -0.2), 0.01);
}

PRAG_TEST(degenerate_samples_give_a_positive_bandwidth) {
  const std::vector<double> one = {3.0};
  const std::vector<double> equal = {2.0, 2.0, 2.0, 2.0};
  const std::vector<double> far = {-5000.0, -5000.0};
  for (const std::vector<double>* sample : {&one, &equal, &far}) {
    const double h = silverman_bandwidth(*sample);
    CHECK(h > 0.0);
    CHECK(std::isfinite(h));
    CHECK(h < 0.01 * std::max(1.0, std::fabs(sample->front())));
    const Kde1d kde = Kde1d::fit(*sample);
    CHECK_EQ(kde.bandwidth(), h);
    CHECK(std::isfinite(kde.pdf(sample->front())));
    CHECK(kde.pdf(sample->front()) > 0.0);
    CHECK_NEAR(integral(kde), 1.0, 1e-9);
  }
  CHECK(silverman_bandwidth({}) > 0.0);
  const std::vector<double> undefined = {std::numeric_limits<double>::quiet_NaN()};
  CHECK(silverman_bandwidth(undefined) > 0.0);
}

// ---- local likelihood estimate -------------------------------------------------

PRAG_TEST(neighbour_distance_matches_a_sorted_search) {
  const std::vector<double> samples = normal_sample(97, 0.0, 1.0, 41);
  const Kde1d kde = Kde1d::fit(samples);
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 200; ++trial) {
    // Half of the trials sit exactly on a sample.
    const double x = trial % 2 == 0 ? prag::test::uniform(rng, -4.0, 4.0)
                                    : kde.samples()[static_cast<std::size_t>(trial) % kde.size()];
    const std::size_t leave_out = trial % 3 == 0 ? static_cast<std::size_t>(trial) % kde.size() : kKeepAll;
    std::vector<double> distances;
    for (std::size_t i = 0; i < kde.size(); ++i) {
      if (i != leave_out) distances.push_back(std::fabs(kde.samples()[i] - x));
    }
    std::sort(distances.begin(), distances.end());
    for (const std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{96}, std::size_t{500}}) {
      const double expected = distances[std::min(k, distances.size()) - 1];
      CHECK_EQ(kde.neighbour_distance(x, k, leave_out), expected);
    }
  }
  CHECK_EQ(kde.neighbour_distance(0.0, 0), 0.0);
}

// The closed form must solve the equations it was derived from. With
// K(u) = phi(u / h) / h and the local model f(t) = exp(a + b u + c u^2 / 2),
// u = t - x, the local likelihood is stationary when the kernel-weighted
// moments of order 0, 1 and 2 of the sample equal those of the model:
//
//     1/n sum_i K(x_i - x) (x_i - x)^j  =  integral K(u) u^j f(x + u) du.
//
// a comes from the implementation; b and c follow from the weighted mean and
// variance of the offsets; the integrals are evaluated by quadrature.
PRAG_TEST(local_estimate_solves_the_local_likelihood_equations) {
  const std::vector<double> samples = normal_sample(60, 1.0, 2.0, 51);
  const Kde1d kde = Kde1d::fit(samples);
  const double n = static_cast<double>(samples.size());
  int fallbacks = 0;
  for (const double x : {-4.5, -1.0, 0.7, 3.0, 8.0}) {
    for (const double h : {0.4, 1.3, 5.0}) {
      double m0 = 0.0;
      double m1 = 0.0;
      double m2 = 0.0;
      double squares = 0.0;
      for (const double s : samples) {
        const double u = s - x;
        const double k = std::exp(-0.5 * (u / h) * (u / h)) / (h * kSqrt2Pi);
        m0 += k / n;
        m1 += k * u / n;
        m2 += k * u * u / n;
        squares += k * k / (n * n);
      }
      // Fewer than three effective samples do not determine a curve; the
      // estimate is then the kernel estimate of that width. This happens with
      // the narrower kernels at the edges of the data.
      if (m0 * m0 / squares < 3.0) {
        CHECK(h < 5.0 && (x == -4.5 || x == 8.0));
        CHECK_NEAR(kde.log_pdf_local(x, h), std::log(m0), 1e-12);
        ++fallbacks;
        continue;
      }
      const double mean = m1 / m0;
      double variance = 0.0;  // centred, to keep its digits where the weights concentrate on one sample
      for (const double s : samples) {
        const double u = s - x;
        variance += std::exp(-0.5 * (u / h) * (u / h)) / (h * kSqrt2Pi) * (u - mean) * (u - mean) / (n * m0);
      }
      const double a = kde.log_pdf_local(x, h);
      const double b = mean / variance;
      const double c = 1.0 / (h * h) - 1.0 / variance;

      // K(u) f(x + u) is a normal curve of mean `mean` and variance `variance`.
      const double sd = std::sqrt(variance);
      const double step = sd / 64.0;
      double i0 = 0.0;
      double i1 = 0.0;
      double i2 = 0.0;
      for (double u = mean - 14.0 * sd; u <= mean + 14.0 * sd; u += step) {
        const double value = std::exp(-0.5 * (u / h) * (u / h) + a + b * u + 0.5 * c * u * u) / (h * kSqrt2Pi) * step;
        i0 += value;
        i1 += value * u;
        i2 += value * u * u;
      }
      CHECK_NEAR(i0, m0, 1e-8 * m0);
      CHECK_NEAR(i1, m1, 1e-8 * (std::fabs(m1) + m0));
      CHECK_NEAR(i2, m2, 1e-8 * m2);
    }
  }
  CHECK(fallbacks >= 1 && fallbacks <= 4);  // of 15 combinations
}

PRAG_TEST(local_estimate_has_no_tail_bias_on_normal_data) {
  // A million draws from N(0, 1) and a wide kernel (h = 1). The kernel estimate
  // converges to the density of N(0, 1 + h^2): at x = 3.5 that is 15 times the
  // true value. The local estimate is consistent for every h.
  const Kde1d kde = Kde1d::fit(normal_sample(1000000, 0.0, 1.0, 61), 1.0);
  for (const double x : {0.0, 1.5, 2.5, 3.5}) {
    const double truth = -0.5 * x * x - std::log(kSqrt2Pi);
    const double smoothed = -0.25 * x * x - std::log(std::sqrt(2.0) * kSqrt2Pi);
    CHECK_NEAR(kde.log_pdf(x), smoothed, 0.03);
    CHECK_NEAR(kde.log_pdf_local(x), truth, 0.1);
  }
  CHECK(kde.log_pdf(3.5) - kde.log_pdf_local(3.5) > 2.5);
}

PRAG_TEST(local_estimate_leave_one_out_and_fallback) {
  const std::vector<double> samples = normal_sample(40, 0.0, 1.0, 71);
  const Kde1d kde = Kde1d::fit(samples);
  // Leaving sample i out equals an estimate fitted without it, at equal width.
  for (const std::size_t i : {std::size_t{0}, std::size_t{17}, std::size_t{39}}) {
    std::vector<double> rest = kde.samples();
    rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(i));
    const Kde1d without = Kde1d::fit(rest, kde.bandwidth());
    for (const double x : {kde.samples()[i], -2.0, 0.3, 4.0}) {
      CHECK_NEAR(kde.log_pdf_local(x, 0.8, i), without.log_pdf_local(x, 0.8), 1e-12);
    }
  }
  // The default width is the bandwidth of the estimate.
  CHECK_EQ(kde.log_pdf_local(0.5), kde.log_pdf_local(0.5, kde.bandwidth()));
  CHECK_EQ(kde.log_pdf_local(0.5, -1.0), kde.log_pdf_local(0.5, kde.bandwidth()));

  // Without spread there is nothing to fit a curve to: the kernel estimate.
  const std::vector<double> equal = {1.0, 1.0, 1.0};
  const Kde1d constant = Kde1d::fit(equal, 0.5);
  CHECK_NEAR(constant.log_pdf_local(1.2), constant.log_pdf(1.2), 1e-12);
  const std::vector<double> single = {1.0};
  const Kde1d one = Kde1d::fit(single, 0.5);
  CHECK_NEAR(one.log_pdf_local(0.0), one.log_pdf(0.0), 1e-12);
  CHECK(one.log_pdf_local(0.0, 0.5, 0) == -std::numeric_limits<double>::infinity());
}

// ---- isotonic regression -----------------------------------------------------

PRAG_TEST(isotonic_hand_examples) {
  const auto check = [](const std::vector<double>& y, const std::vector<double>& w,
                        const std::vector<double>& expected) {
    const std::vector<double> fit = isotonic_nondecreasing(y, w);
    REQUIRE(fit.size() == expected.size());
    for (std::size_t i = 0; i < fit.size(); ++i) CHECK_NEAR(fit[i], expected[i], 1e-12);
  };
  check({}, {}, {});
  check({4.0}, {}, {4.0});
  check({1, 2, 3}, {}, {1, 2, 3});
  check({1, 3, 2, 4}, {}, {1, 2.5, 2.5, 4});
  check({3, 2, 1}, {}, {2, 2, 2});
  check({1, 1, 1}, {}, {1, 1, 1});
  // A violation that reaches back over two blocks: 5, 6 and 1 pool to 4; 3
  // stays below them and 7 above.
  check({3, 5, 6, 1, 7}, {}, {3, 4, 4, 4, 7});
  // Here the pooled block drops below its predecessor and takes it in as well.
  check({4.5, 5, 6, 1, 7}, {}, {4.125, 4.125, 4.125, 4.125, 7});
}

PRAG_TEST(isotonic_weights) {
  const auto check = [](const std::vector<double>& y, const std::vector<double>& w,
                        const std::vector<double>& expected) {
    const std::vector<double> fit = isotonic_nondecreasing(y, w);
    REQUIRE(fit.size() == expected.size());
    for (std::size_t i = 0; i < fit.size(); ++i) CHECK_NEAR(fit[i], expected[i], 1e-12);
  };
  check({3, 1}, {1, 3}, {1.5, 1.5});
  check({3, 1}, {3, 1}, {2.5, 2.5});
  check({1, 3, 2}, {1, 1, 3}, {1, 2.25, 2.25});
  // Equal weights of any size behave like unit weights.
  check({1, 3, 2, 4}, {7, 7, 7, 7}, {1, 2.5, 2.5, 4});
  // A point of weight zero follows its block and moves nobody.
  check({3, 1, 2}, {1, 0, 1}, {2.5, 2.5, 2.5});
  check({1, 5, 2}, {1, 0, 1}, {1, 2, 2});
  // Negative and undefined weights count as zero; a wrong length means unit weights.
  check({1, 5, 2}, {1, -4, 1}, {1, 2, 2});
  check({1, 5, 2}, {1, std::numeric_limits<double>::quiet_NaN(), 1}, {1, 2, 2});
  check({1, 3, 2, 4}, {1, 100}, {1, 2.5, 2.5, 4});
}

PRAG_TEST(isotonic_is_idempotent) {
  std::mt19937_64 rng(81);
  for (int trial = 0; trial < 200; ++trial) {
    const std::size_t n = 1 + static_cast<std::size_t>(rng() % 60);
    std::vector<double> y(n);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = prag::test::uniform(rng, -5.0, 5.0);
      w[i] = prag::test::uniform(rng, 0.1, 4.0);
    }
    const std::vector<double> once = isotonic_nondecreasing(y, w);
    const std::vector<double> twice = isotonic_nondecreasing(once, w);
    CHECK(non_decreasing(once));
    CHECK(once == twice);  // bit for bit
    // The weighted mean is preserved.
    double before = 0.0;
    double after = 0.0;
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      before += w[i] * y[i];
      after += w[i] * once[i];
      total += w[i];
    }
    CHECK_NEAR(after / total, before / total, 1e-12);
  }
}

PRAG_TEST(isotonic_is_optimal_on_small_inputs) {
  std::mt19937_64 rng(91);
  const std::vector<double> weights_pool = {0.5, 1.0, 2.0, 3.0};
  const int levels = 17;  // candidates take values k / 16
  int instances = 0;
  for (int trial = 0; trial < 240; ++trial) {
    const std::size_t n = 2 + static_cast<std::size_t>(rng() % 5);  // 2 .. 6
    std::vector<double> y(n);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = static_cast<double>(rng() % 9) / 8.0;
      w[i] = trial % 2 == 0 ? 1.0 : weights_pool[rng() % weights_pool.size()];
    }
    const std::vector<double> fit = isotonic_nondecreasing(y, trial % 2 == 0 ? std::vector<double>{} : w);
    REQUIRE(fit.size() == n);
    CHECK(non_decreasing(fit));
    const double error = squared_error(fit, y, w);

    // Every non-decreasing sequence over the levels, by an odometer that keeps
    // the digits sorted.
    std::vector<int> digits(n, 0);
    std::vector<double> candidate(n);
    double best_candidate = std::numeric_limits<double>::infinity();
    for (;;) {
      for (std::size_t i = 0; i < n; ++i) candidate[i] = static_cast<double>(digits[i]) / 16.0;
      best_candidate = std::min(best_candidate, squared_error(candidate, y, w));
      std::size_t position = n;
      while (position > 0 && digits[position - 1] == levels - 1) --position;
      if (position == 0) break;
      const int next = digits[position - 1] + 1;
      for (std::size_t i = position - 1; i < n; ++i) digits[i] = next;
    }
    CHECK(error <= best_candidate + 1e-12);

    // The exact optimum is constant on consecutive blocks, each at its weighted
    // mean: search all 2^(n-1) partitions and keep the non-decreasing ones.
    double best_partition = std::numeric_limits<double>::infinity();
    for (std::uint32_t cuts = 0; cuts < (1u << (n - 1)); ++cuts) {
      std::vector<double> blocks(n);
      std::size_t begin = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const bool last = i + 1 == n || ((cuts >> i) & 1u) != 0;
        if (!last) continue;
        double sum = 0.0;
        double total = 0.0;
        for (std::size_t k = begin; k <= i; ++k) {
          sum += w[k] * y[k];
          total += w[k];
        }
        for (std::size_t k = begin; k <= i; ++k) blocks[k] = sum / total;
        begin = i + 1;
      }
      if (non_decreasing(blocks)) best_partition = std::min(best_partition, squared_error(blocks, y, w));
    }
    CHECK_NEAR(error, best_partition, 1e-12);
    ++instances;
  }
  CHECK(instances >= 200);
}
