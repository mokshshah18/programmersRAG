#include "prag/kde/vmf.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace prag::kde {

namespace {

constexpr double kLogPi = 1.14472988584940017414;     // log(pi)
constexpr double kLogTwoPi = 1.83787706640934548356;  // log(2 pi)
constexpr double kLogTwo = 0.69314718055994530942;
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Regimes of log_bessel_i:
//
//   order        argument                     method
//   nu >= 32     x^2 / 4 <= nu + 1            power series
//   nu >= 32     larger                       uniform (Debye) expansion
//   nu <  32     x <= 32                      power series
//   nu <  32     x >= max(32, 2 nu^2)         large-argument (Hankel) expansion
//   nu <  32     in between                   recurrence down from an order >= 32
//
// kLargeOrder: the uniform expansion is an expansion in 1 / r, r = sqrt(nu^2 + x^2),
// whose k-th term is bounded by h_k / r^k with h_k the k-th coefficient of the
// large-argument expansion of I_0 (the polynomial factor is largest at p = 0;
// checked numerically for k <= 24). h_14 / 32^14 = 1.0e-16, so fourteen terms
// reach double precision for every argument once nu >= 32.
constexpr double kLargeOrder = 32.0;
// Below the large order the power series is used up to this argument (at most
// about 70 terms, all positive).
constexpr double kSeriesMaxArgument = 32.0;
constexpr int kSeriesMaxTerms = 400;
constexpr int kDebyeTerms = 16;
constexpr int kHankelMaxTerms = 80;
constexpr int kFractionMaxTerms = 4000;

// Coefficients of the polynomials of the uniform asymptotic expansion,
//   U_k(p) = sum_{j=0..k} c[k][j] p^(k + 2j),
// from U_0 = 1 and
//   U_{k+1}(p) = 1/2 p^2 (1 - p^2) U_k'(p) + 1/8 int_0^p (1 - 5 t^2) U_k(t) dt.
// The sign of c[k][j] is (-1)^j, so both contributions to a coefficient have the
// same sign and the recurrence loses no precision in floating point.
struct DebyeTable {
  std::array<std::array<double, kDebyeTerms + 1>, kDebyeTerms + 1> c{};

  DebyeTable() {
    c[0][0] = 1.0;
    for (int k = 0; k < kDebyeTerms; ++k) {
      for (int j = 0; j <= k; ++j) {
        const double e = static_cast<double>(k + 2 * j);
        const std::size_t row = static_cast<std::size_t>(k);
        const std::size_t col = static_cast<std::size_t>(j);
        c[row + 1][col] += c[row][col] * (0.5 * e + 0.125 / (e + 1.0));
        c[row + 1][col + 1] -= c[row][col] * (0.5 * e + 0.625 / (e + 3.0));
      }
    }
  }
};

const DebyeTable& debye_table() {
  static const DebyeTable table;
  return table;
}

// log( sum_{k>=0} (x^2/4)^k / (k! (nu+1)_k) ) = log( Gamma(nu+1) (x/2)^(-nu) I_nu(x) ).
// Every term is positive. The caller keeps x^2 / 4 small enough for the sum not to overflow.
double log_series_sum(double nu, double x) {
  const double q = 0.25 * x * x;
  double term = 1.0;
  double tail = 0.0;  // terms k >= 1; kept apart from the leading 1 for log1p
  for (int k = 1; k <= kSeriesMaxTerms; ++k) {
    term *= q / (static_cast<double>(k) * (nu + static_cast<double>(k)));
    tail += term;
    if (term <= 0.125 * kEps * (1.0 + tail)) break;
  }
  return std::log1p(tail);
}

// Uniform asymptotic expansion for large order (Debye):
//   I_nu(x) ~ exp(E) / sqrt(2 pi r) * sum_k U_k(p) / nu^k,
//   r = sqrt(nu^2 + x^2),  p = nu / r,  E = r + nu log( x / (nu + r) ) = r - nu asinh(nu / x).
// U_k(p) / nu^k = V_k(p^2) / r^k with V_k(s) = sum_j c[k][j] s^j, which is the form
// evaluated here. Accurate to double precision for r >= 32.
//
// Returns log( sum_k U_k(p) / nu^k ).
double uniform_log_sum(double nu, double r) {
  const DebyeTable& table = debye_table();
  const double p = nu / r;
  const double s = p * p;
  double sum = 0.0;  // terms k >= 1
  double inverse_power = 1.0;
  for (std::size_t k = 1; k <= static_cast<std::size_t>(kDebyeTerms); ++k) {
    inverse_power /= r;
    const auto& c = table.c[k];
    double v = c[k];
    for (std::size_t j = k; j-- > 0;) v = v * s + c[j];
    sum += v * inverse_power;
    // |V_k| is largest at s = 0, so c[0] / r^k bounds this term for every p.
    if (c[0] * inverse_power < 0.25 * kEps) break;
  }
  return std::log1p(sum);
}

double log_bessel_i_uniform(double nu, double x) {
  const double r = std::hypot(nu, x);
  const double exponent = r + nu * std::log(x / (nu + r));
  return exponent - 0.5 * (kLogTwoPi + std::log(r)) + uniform_log_sum(nu, r);
}

// log( I_{nu+1}(x) / I_nu(x) ) from the uniform expansion, arranged so that the
// large parts of the two exponents cancel analytically:
//   E(nu+1) - E(nu) = (r1 - r0) - asinh((nu+1)/x) - nu (asinh((nu+1)/x) - asinh(nu/x)),
//   r1 - r0 = (2 nu + 1) / (r0 + r1),
//   asinh(b) - asinh(a) = asinh( (2 nu + 1) / ((nu+1) r0 + nu r1) )   for a = nu/x, b = (nu+1)/x.
// For x >> nu the result is -(nu + 1/2) / x + ..., with a relative error of a few
// units of rounding, which a difference of two logarithms of size x cannot give.
double log_bessel_ratio_uniform(double nu, double x) {
  const double r0 = std::hypot(nu, x);
  const double r1 = std::hypot(nu + 1.0, x);
  const double odd = 2.0 * nu + 1.0;
  const double exponent =
      odd / (r0 + r1) - std::asinh((nu + 1.0) / x) - nu * std::asinh(odd / ((nu + 1.0) * r0 + nu * r1));
  return exponent - 0.25 * std::log1p(odd / (r0 * r0)) + (uniform_log_sum(nu + 1.0, r1) - uniform_log_sum(nu, r0));
}

// Gauss continued fraction
//   I_{nu+1}(x) / I_nu(x) = x / ( b_1 + x^2 / ( b_2 + x^2 / ( b_3 + ... ) ) ),   b_k = 2 (nu + k),
// evaluated by the modified Lentz algorithm. All partial numerators and
// denominators are positive, so there is no cancellation. The number of terms
// grows like 6 sqrt(x) and the rounding error with its square root, which is why
// large arguments are left to the uniform expansion.
double bessel_ratio_fraction(double nu, double x) {
  const double a = x * x;
  double f = 2.0 * (nu + 1.0);
  double c = f;
  double d = 0.0;
  for (int k = 2; k <= kFractionMaxTerms; ++k) {
    const double b = 2.0 * (nu + static_cast<double>(k));
    d = 1.0 / (b + a * d);
    c = b + a / c;
    const double delta = c * d;
    f *= delta;
    if (std::fabs(delta - 1.0) <= kEps) break;
  }
  return x / f;
}

// 1 - I_{nu+1}(x) / I_nu(x) without loss of relative accuracy when the ratio is close to one.
double bessel_ratio_complement(double nu, double x) {
  if (std::hypot(nu, x) >= kLargeOrder) return -std::expm1(log_bessel_ratio_uniform(nu, x));
  return 1.0 - bessel_ratio_fraction(nu, x);  // x < 32: the ratio stays below 0.99
}

// Large-argument expansion (Hankel):
//   I_nu(x) ~ exp(x) / sqrt(2 pi x) * sum_k (-1)^k a_k(nu) / x^k,
//   a_k(nu) = prod_{j=1..k} (4 nu^2 - (2j-1)^2) / (k! 8^k).
// Used for x >= max(32, 2 nu^2): successive terms then shrink by at least a
// factor four, and the neglected part is of relative size exp(-2x) < 2e-28.
double log_bessel_i_large_argument(double nu, double x) {
  const double mu = 4.0 * nu * nu;
  double term = 1.0;
  double sum = 0.0;  // terms k >= 1
  for (int k = 1; k <= kHankelMaxTerms; ++k) {
    const double odd = 2.0 * static_cast<double>(k) - 1.0;
    const double next = -term * (mu - odd * odd) / (8.0 * static_cast<double>(k) * x);
    if (std::fabs(next) > std::fabs(term)) break;  // smallest term of the asymptotic series reached
    term = next;
    sum += term;
    if (std::fabs(term) <= 0.25 * kEps) break;
  }
  return x - 0.5 * (kLogTwoPi + std::log(x)) + std::log1p(sum);
}

// Moderate order and argument, where neither expansion has converged: evaluate
// at the order nu + N >= kLargeOrder with the uniform expansion and come down
// with the three-term recurrence written for ratios,
//   I_{mu-1} / I_mu = 2 mu / x + I_{mu+1} / I_mu,
// which is stable in this direction (every quantity is positive).
double log_bessel_i_recurrence(double nu, double x) {
  const int steps = static_cast<int>(std::ceil(kLargeOrder - nu));
  const double top = nu + static_cast<double>(steps);
  double ratio = bessel_ratio(top, x);  // I_{top+1} / I_top
  double product = 1.0;                 // I_nu / I_top; below 3.1^32 because x > 32
  for (int j = steps; j >= 1; --j) {
    const double up = 2.0 * (nu + static_cast<double>(j)) / x + ratio;  // I_{mu-1} / I_mu at mu = nu + j
    product *= up;
    ratio = 1.0 / up;
  }
  return log_bessel_i_uniform(top, x) + std::log(product);
}

bool uses_series(double nu, double x) {
  if (nu >= kLargeOrder) return 0.25 * x * x <= nu + 1.0;  // at most about twenty terms
  return x <= kSeriesMaxArgument;
}

double log_bessel_i_asymptotic(double nu, double x) {
  if (nu >= kLargeOrder) return log_bessel_i_uniform(nu, x);
  if (x >= 2.0 * nu * nu) return log_bessel_i_large_argument(nu, x);
  return log_bessel_i_recurrence(nu, x);
}

// log( Gamma(nu+1) (x/2)^(-nu) I_nu(x) ): zero at x = 0 and increasing in x.
// In the series regime this avoids the cancellation between nu log(x/2) and log I_nu(x).
double log_bessel_i_reduced(double nu, double x) {
  if (uses_series(nu, x)) return log_series_sum(nu, x);
  return log_bessel_i_asymptotic(nu, x) - nu * std::log(0.5 * x) + std::lgamma(nu + 1.0);
}

}  // namespace

double log_bessel_i(double nu, double x) {
  if (std::isnan(nu) || std::isnan(x) || nu < 0.0 || x < 0.0) return kNaN;
  if (x == 0.0) return nu == 0.0 ? 0.0 : -kInf;
  if (std::isinf(x)) return kInf;
  if (uses_series(nu, x)) {
    const double sum = log_series_sum(nu, x);
    if (nu == 0.0) return sum;
    return nu * std::log(0.5 * x) - std::lgamma(nu + 1.0) + sum;
  }
  return log_bessel_i_asymptotic(nu, x);
}

double bessel_ratio(double nu, double x) {
  if (std::isnan(nu) || std::isnan(x) || nu < 0.0 || x < 0.0) return kNaN;
  if (x == 0.0) return 0.0;
  if (std::isinf(x)) return 1.0;
  if (std::hypot(nu, x) >= kLargeOrder) return std::exp(log_bessel_ratio_uniform(nu, x));
  return bessel_ratio_fraction(nu, x);
}

double log_sphere_area(std::size_t m) {
  if (m == 0) return kNaN;
  const double half = 0.5 * static_cast<double>(m);
  return kLogTwo + half * kLogPi - std::lgamma(half);
}

double log_vmf_normalizer(std::size_t m, double kappa) {
  if (m == 0 || std::isnan(kappa) || kappa < 0.0) return kNaN;
  if (m == 1) {
    // S^0 is the pair of points {-1, +1}: C_1 = 1 / (2 cosh kappa).
    return -(kappa + std::log1p(std::exp(-2.0 * kappa)));
  }
  if (std::isinf(kappa)) return kInf;
  // C_m(kappa) = 1 / ( area(S^{m-1}) * Gamma(nu+1) (kappa/2)^(-nu) I_nu(kappa) ),  nu = m/2 - 1.
  const double nu = 0.5 * static_cast<double>(m) - 1.0;
  return -log_sphere_area(m) - log_bessel_i_reduced(nu, kappa);
}

double vmf_mean_resultant_length(std::size_t m, double kappa) {
  if (m == 0 || std::isnan(kappa) || kappa < 0.0) return kNaN;
  if (m == 1) return std::tanh(kappa);
  return bessel_ratio(0.5 * static_cast<double>(m) - 1.0, kappa);
}

double vmf_kappa_mle(std::size_t m, double rbar) {
  if (m == 0 || std::isnan(rbar)) return kNaN;
  if (rbar <= 0.0) return 0.0;
  if (rbar >= 1.0) return kInf;
  if (m == 1) return std::atanh(rbar);

  const double dim = static_cast<double>(m);
  const double nu = 0.5 * dim - 1.0;
  // Residual and slope are formed from 1 - A and 1 - rbar, which keeps their
  // relative accuracy when both are close to one (large kappa).
  const double target = 1.0 - rbar;
  double kappa = rbar * (dim - rbar * rbar) / ((1.0 - rbar) * (1.0 + rbar));
  double previous_step = kInf;
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double complement = bessel_ratio_complement(nu, kappa);  // 1 - A
    const double a = 1.0 - complement;
    // A'(kappa) = 1 - A^2 - (m - 1) A / kappa, positive for kappa > 0.
    const double slope = complement * (1.0 + a) - (dim - 1.0) * a / kappa;
    if (!(slope > 0.0)) break;
    double next = kappa - (target - complement) / slope;
    if (!(next > 0.0)) next = 0.5 * kappa;
    const double step = std::fabs(next - kappa);
    kappa = next;
    if (step <= 1e-15 * kappa) break;
    // Once the residual is at rounding level the steps stop shrinking.
    if (iteration >= 2 && step >= previous_step) break;
    previous_step = step;
  }
  return kappa;
}

}  // namespace prag::kde
