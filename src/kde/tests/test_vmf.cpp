#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vector>

#include "bessel_reference.hpp"
#include "prag/kde/vmf.hpp"

using namespace prag::kde;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kOrders[] = {0.0, 0.5, 1.0, 3.5, 15.5, 31.0, 127.5, 255.5, 511.0};

// Relative agreement of two logarithms. The absolute floor covers the rounding
// of the reference itself where the logarithm is close to zero.
bool log_close(double value, double reference, double relative = 1e-9) {
  return std::fabs(value - reference) <= relative * std::fabs(reference) + 4e-15;
}

// n points spaced geometrically from lo to hi, both included.
std::vector<double> geometric(double lo, double hi, int n) {
  std::vector<double> out;
  for (int i = 0; i < n; ++i) out.push_back(lo * std::pow(hi / lo, static_cast<double>(i) / (n - 1)));
  return out;
}

// log of int_0^pi exp(kappa (cos t - 1)) sin^power(t) dt by the composite Simpson rule.
double log_polar_integral(double kappa, int power, int intervals) {
  const double h = kPi / intervals;
  double sum = 0.0;
  for (int i = 0; i <= intervals; ++i) {
    const double t = h * i;
    const double weight = (i == 0 || i == intervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
    sum += weight * std::exp(kappa * (std::cos(t) - 1.0)) * std::pow(std::sin(t), power);
  }
  return std::log(sum * h / 3.0);
}

// log C_m(kappa) from the definition: 1 / C_m = area(S^{m-2}) int_0^pi exp(kappa cos t) sin^{m-2}(t) dt.
double log_normalizer_by_quadrature(std::size_t m, double kappa) {
  const double log_equator = m == 2 ? std::log(2.0) : log_sphere_area(m - 1);
  return -(log_equator + kappa + log_polar_integral(kappa, static_cast<int>(m) - 2, 200000));
}

}  // namespace

PRAG_TEST(log_bessel_i_agrees_with_the_standard_library) {
  std::vector<double> arguments = geometric(1e-3, 700.0, 97);
  // Both sides of every regime boundary of the implementation.
  for (double x : {31.999, 32.0, 32.001, 5.999, 6.001, 24.49, 24.51, 480.4, 480.6, 22.6, 22.7, 45.2, 45.3})
    arguments.push_back(x);
  int compared = 0;
  double worst = 0.0;
  for (double nu : kOrders) {
    for (double x : arguments) {
      const double reference = std::cyl_bessel_i(nu, x);
      // Outside the normal range the library value is 0, infinite or subnormal
      // (a subnormal carries few significant digits).
      if (!std::isfinite(reference) || reference < 1e-300) continue;
      const double expected = std::log(reference);
      const double value = log_bessel_i(nu, x);
      CHECK(log_close(value, expected));
      worst = std::fmax(worst, std::fabs(value - expected) / std::fmax(1.0, std::fabs(expected)));
      ++compared;
    }
  }
  CHECK(compared > 600);
  CHECK(worst < 1e-12);
  std::printf("  compared %d points, largest error %.2e (relative to max(1, |log I|))\n", compared, worst);
}

PRAG_TEST(log_bessel_i_agrees_with_high_precision_values) {
  double worst = 0.0;
  for (const testing::BesselReference& row : testing::kBesselReference) {
    const double value = log_bessel_i(row.nu, row.x);
    CHECK(log_close(value, row.log_i));
    worst = std::fmax(worst, std::fabs(value - row.log_i) / std::fmax(1.0, std::fabs(row.log_i)));
  }
  CHECK(worst < 1e-13);
  std::printf("  largest error %.2e over %zu reference values\n", worst,
              sizeof(testing::kBesselReference) / sizeof(testing::kBesselReference[0]));
}

PRAG_TEST(log_bessel_i_half_integer_closed_forms) {
  for (double x : geometric(1e-3, 1e6, 61)) {
    // I_{1/2}(x) = sqrt(2 / (pi x)) sinh(x),  I_{3/2}(x) = sqrt(2 / (pi x)) (cosh(x) - sinh(x) / x).
    const double log_sinh = x < 1e-2 ? std::log(x) + std::log1p(x * x / 6.0 + x * x * x * x / 120.0)
                                     : x + std::log(-0.5 * std::expm1(-2.0 * x));
    const double half = 0.5 * std::log(2.0 / (kPi * x)) + log_sinh;
    CHECK(log_close(log_bessel_i(0.5, x), half, 1e-12));
    if (x >= 1.0) {
      const double coth = 1.0 / std::tanh(x);
      const double three_halves = half + std::log(coth - 1.0 / x);
      CHECK(log_close(log_bessel_i(1.5, x), three_halves, 1e-12));
    }
  }
}

PRAG_TEST(log_bessel_i_satisfies_the_recurrence_for_large_arguments) {
  // I_{nu-1}(x) - I_{nu+1}(x) = (2 nu / x) I_nu(x), divided by I_nu(x):
  //   exp(L_{nu-1} - L_nu) - exp(L_{nu+1} - L_nu) = 2 nu / x.
  // Each L is of size x and carries a rounding error of a few units in its last
  // place, which sets the tolerance.
  for (double nu : {1.0, 3.5, 15.5, 31.0, 127.5, 255.5, 511.0}) {
    for (double x : geometric(700.0, 1e6, 25)) {
      const double below = log_bessel_i(nu - 1.0, x);
      const double centre = log_bessel_i(nu, x);
      const double above = log_bessel_i(nu + 1.0, x);
      const double lhs = std::exp(below - centre) - std::exp(above - centre);
      const double tolerance = 16.0 * kEps * std::fabs(centre) + 1e-15;
      CHECK_NEAR(lhs, 2.0 * nu / x, tolerance);
      // The same differences agree with the ratio function, which does not
      // pass through logarithms of size x.
      CHECK_NEAR(above - centre, std::log(bessel_ratio(nu, x)), tolerance);
      CHECK_NEAR(centre - below, std::log(bessel_ratio(nu - 1.0, x)), tolerance);
    }
  }
}

PRAG_TEST(log_bessel_i_increases_with_the_argument_at_the_right_rate) {
  // d/dx log I_nu(x) = I_{nu+1}(x) / I_nu(x) + nu / x.
  for (double nu : kOrders) {
    double previous = -std::numeric_limits<double>::infinity();
    for (double x : geometric(1e-3, 1e6, 181)) {
      const double value = log_bessel_i(nu, x);
      CHECK(value > previous);
      previous = value;
      const double h = 1e-3 * x;
      const double slope = (log_bessel_i(nu, x + h) - log_bessel_i(nu, x - h)) / (2.0 * h);
      const double expected = bessel_ratio(nu, x) + nu / x;
      CHECK(std::fabs(slope - expected) <= 2e-6 * expected);
    }
  }
}

PRAG_TEST(log_bessel_i_is_continuous_across_regime_boundaries) {
  for (double nu : {0.0, 0.5, 3.5, 4.5, 15.5, 25.0, 31.0, 31.999, 32.0, 32.001, 127.5, 511.0}) {
    for (double boundary : {32.0, 2.0 * nu * nu, 2.0 * std::sqrt(nu + 1.0)}) {
      if (boundary <= 0.0) continue;
      const double lo = boundary * (1.0 - 1e-9);
      const double hi = boundary * (1.0 + 1e-9);
      const double jump = log_bessel_i(nu, hi) - log_bessel_i(nu, lo);
      const double expected = (bessel_ratio(nu, boundary) + nu / boundary) * (hi - lo);
      const double scale = std::fmax(1.0, std::fabs(log_bessel_i(nu, boundary)));
      CHECK_NEAR(jump, expected, 1e-13 * scale);
    }
  }
  // Across the order at which the uniform expansion takes over.
  for (double x : {0.5, 8.0, 40.0, 300.0, 5000.0}) {
    const double below = log_bessel_i(32.0 - 1e-9, x);
    const double above = log_bessel_i(32.0 + 1e-9, x);
    const double centre = log_bessel_i(32.0, x);
    CHECK(below > centre);
    CHECK(centre > above);
    CHECK_NEAR(below - centre, centre - above, 1e-13 * std::fmax(1.0, std::fabs(centre)));
  }
}

PRAG_TEST(log_bessel_i_special_arguments) {
  CHECK_EQ(log_bessel_i(0.0, 0.0), 0.0);
  CHECK(log_bessel_i(0.5, 0.0) == -std::numeric_limits<double>::infinity());
  CHECK(log_bessel_i(511.0, 0.0) == -std::numeric_limits<double>::infinity());
  CHECK(std::isnan(log_bessel_i(-1.0, 1.0)));
  CHECK(std::isnan(log_bessel_i(1.0, -1.0)));
  CHECK(std::isnan(log_bessel_i(std::numeric_limits<double>::quiet_NaN(), 1.0)));
  // Far below the smallest double that I_nu(x) itself could represent.
  CHECK(log_close(log_bessel_i(511.0, 1e-6), -1.00937462513871850e4, 1e-12));
  CHECK_EQ(bessel_ratio(3.0, 0.0), 0.0);
  CHECK(std::isnan(bessel_ratio(-0.5, 1.0)));
}

PRAG_TEST(bessel_ratio_matches_closed_forms_and_bounds) {
  for (double x : geometric(1e-3, 1e6, 91)) {
    // I_{3/2} / I_{1/2} = coth(x) - 1/x (the Langevin function). Below 0.1 the
    // closed form cancels, so its Taylor series is the reference there:
    // x/3 - x^3/45 + 2 x^5/945 - x^7/4725 + 2 x^9/93555 - ...
    const double x2 = x * x;
    const double series = x / 3.0 * (1.0 - x2 / 15.0 * (1.0 - x2 * 2.0 / 21.0 * (1.0 - x2 / 10.0 * (1.0 - x2 * 10.0 / 99.0))));
    const double langevin = x < 0.1 ? series : 1.0 / std::tanh(x) - 1.0 / x;
    CHECK(std::fabs(bessel_ratio(0.5, x) - langevin) <= 2e-13 * langevin);
    for (double nu : kOrders) {
      const double ratio = bessel_ratio(nu, x);
      CHECK(ratio > 0.0);
      CHECK(ratio < 1.0);
      // Amos (1974): x / (nu + 1/2 + sqrt(x^2 + (nu + 3/2)^2)) <= ratio <= x / (nu + 1/2 + sqrt(x^2 + (nu + 1/2)^2)).
      CHECK(ratio >= x / (nu + 0.5 + std::hypot(x, nu + 1.5)) * (1.0 - 1e-14));
      CHECK(ratio <= x / (nu + 0.5 + std::hypot(x, nu + 0.5)) * (1.0 + 1e-14));
      // Recurrence between neighbouring ratios: 1 / r_{nu} - r_{nu+1} = 2 (nu + 1) / x.
      const double next = bessel_ratio(nu + 1.0, x);
      const double expected = 2.0 * (nu + 1.0) / x;
      CHECK(std::fabs(1.0 / ratio - next - expected) <= 1e-13 * (expected + 1.0));
    }
  }
}

PRAG_TEST(sphere_area_closed_forms) {
  CHECK_NEAR(log_sphere_area(1), std::log(2.0), 1e-15);
  CHECK_NEAR(log_sphere_area(2), std::log(2.0 * kPi), 1e-15);
  CHECK_NEAR(log_sphere_area(3), std::log(4.0 * kPi), 1e-15);
  CHECK_NEAR(log_sphere_area(4), std::log(2.0 * kPi * kPi), 1e-15);
  CHECK_NEAR(log_sphere_area(6), std::log(kPi * kPi * kPi), 1e-15);
  CHECK(std::isnan(log_sphere_area(0)));
}

PRAG_TEST(vmf_normalizer_closed_form_on_the_ordinary_sphere) {
  // m = 3: C_3(kappa) = kappa / (4 pi sinh kappa).
  for (double kappa : geometric(1e-3, 1e5, 81)) {
    const double log_sinh = kappa < 1e-2
                                ? std::log(kappa) + std::log1p(kappa * kappa / 6.0 + std::pow(kappa, 4) / 120.0)
                                : kappa + std::log(-0.5 * std::expm1(-2.0 * kappa));
    const double expected = std::log(kappa) - std::log(4.0 * kPi) - log_sinh;
    CHECK(log_close(log_vmf_normalizer(3, kappa), expected, 1e-12));
  }
  // m = 1: two points with weights proportional to exp(+-kappa).
  for (double kappa : {0.0, 0.3, 2.0, 50.0, 800.0}) {
    const double expected = kappa < 300.0 ? -std::log(2.0 * std::cosh(kappa)) : -kappa;
    CHECK(log_close(log_vmf_normalizer(1, kappa), expected, 1e-12));
  }
}

PRAG_TEST(vmf_normalizer_agrees_with_quadrature) {
  for (std::size_t m : {std::size_t{2}, std::size_t{4}, std::size_t{5}, std::size_t{16}}) {
    for (double kappa : {0.01, 0.5, 3.0, 20.0, 150.0, 2000.0}) {
      const double expected = log_normalizer_by_quadrature(m, kappa);
      CHECK(log_close(log_vmf_normalizer(m, kappa), expected, 1e-10));
    }
  }
}

PRAG_TEST(vmf_normalizer_tends_to_the_uniform_density) {
  for (std::size_t m : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{8}, std::size_t{32},
                        std::size_t{384}, std::size_t{1024}}) {
    const double uniform = -log_sphere_area(m);
    CHECK_NEAR(log_vmf_normalizer(m, 0.0), uniform, 1e-15 * std::fabs(uniform));
    CHECK_NEAR(log_vmf_normalizer(m, 1e-9), uniform, 1e-15 * std::fabs(uniform));
    // First correction: C_m(kappa) = C_m(0) / (1 + kappa^2 / (2 m) + ...).
    const double kappa = 1e-3;
    const double drop = uniform - log_vmf_normalizer(m, kappa);
    if (m > 1) CHECK_NEAR(drop, kappa * kappa / (2.0 * static_cast<double>(m)), 1e-12);
    // Decreasing in kappa until the exponential takes over is not required;
    // what is required is a finite value for every dimension and concentration.
    for (double k : {1.0, 100.0, 1e4, 1e6}) CHECK(std::isfinite(log_vmf_normalizer(m, k)));
  }
  CHECK(std::isnan(log_vmf_normalizer(0, 1.0)));
  CHECK(std::isnan(log_vmf_normalizer(3, -1.0)));
}

PRAG_TEST(mean_resultant_length_closed_forms) {
  for (double kappa : {0.2, 1.0, 7.0, 90.0, 4000.0}) {
    CHECK_NEAR(vmf_mean_resultant_length(1, kappa), std::tanh(kappa), 1e-15);
    CHECK_NEAR(vmf_mean_resultant_length(3, kappa), 1.0 / std::tanh(kappa) - 1.0 / kappa, 1e-14);
  }
  CHECK_EQ(vmf_mean_resultant_length(16, 0.0), 0.0);
  // Small concentration: A_m(kappa) = kappa / m - kappa^3 / (m^2 (m + 2)) + ...
  for (std::size_t m : {std::size_t{2}, std::size_t{16}, std::size_t{1024}}) {
    const double d = static_cast<double>(m);
    const double kappa = 1e-3;
    const double expected = kappa / d - kappa * kappa * kappa / (d * d * (d + 2.0));
    CHECK(std::fabs(vmf_mean_resultant_length(m, kappa) - expected) <= 1e-12 * expected);
  }
}

PRAG_TEST(kappa_mle_inverts_the_mean_resultant_length) {
  double worst = 0.0;
  for (std::size_t m : {std::size_t{2}, std::size_t{3}, std::size_t{16}, std::size_t{64}, std::size_t{1024}}) {
    for (double kappa : geometric(0.01, 1e5, 57)) {
      const double rbar = vmf_mean_resultant_length(m, kappa);
      REQUIRE(rbar > 0.0);
      REQUIRE(rbar < 1.0);
      const double recovered = vmf_kappa_mle(m, rbar);
      const double error = std::fabs(recovered - kappa) / kappa;
      worst = std::fmax(worst, error);
      CHECK(error <= 1e-9);
    }
  }
  std::printf("  largest relative error of the recovered concentration %.2e\n", worst);
  CHECK_EQ(vmf_kappa_mle(16, 0.0), 0.0);
  CHECK(std::isinf(vmf_kappa_mle(16, 1.0)));
  CHECK_NEAR(vmf_kappa_mle(1, std::tanh(1.5)), 1.5, 1e-13);
  // Table 1 of Banerjee et al. (2005) lists the approximation, not the root;
  // the root must satisfy the equation it approximates.
  const double root = vmf_kappa_mle(500, 0.8);
  CHECK_NEAR(vmf_mean_resultant_length(500, root), 0.8, 1e-15);
}
