// von Mises-Fisher distribution on the unit sphere S^{m-1} in R^m.
//
//   f(x; mu, kappa) = C_m(kappa) * exp(kappa * <mu, x>),
//   C_m(kappa)      = kappa^{m/2 - 1} / ( (2 pi)^{m/2} * I_{m/2 - 1}(kappa) ),
//
// where I_nu is the modified Bessel function of the first kind. All functions
// work in the log domain so that large kappa and large m do not overflow.
#pragma once

#include <cstddef>

namespace prag::kde {

// log I_nu(x) for nu >= 0 and x >= 0 (returns -infinity at x = 0 when nu > 0).
// Relative error below 1e-9 over nu in [0, 512], x in (0, 1e6].
double log_bessel_i(double nu, double x);

// I_{nu+1}(x) / I_nu(x), in (0, 1), evaluated by a continued fraction while
// sqrt(nu^2 + x^2) < 32 and from the uniform asymptotic expansion beyond (the
// continued fraction needs about 6 sqrt(x) terms and loses accuracy with them).
double bessel_ratio(double nu, double x);

// log C_m(kappa). For kappa -> 0 this tends to -log(area of S^{m-1}).
// m = 1 is the two-point sphere {-1, +1}: C_1 = 1 / (2 cosh kappa).
double log_vmf_normalizer(std::size_t m, double kappa);

// log of the surface area of S^{m-1}: log( 2 pi^{m/2} / Gamma(m/2) ).
double log_sphere_area(std::size_t m);

// A_m(kappa) = I_{m/2}(kappa) / I_{m/2-1}(kappa): mean resultant length of a vMF distribution.
double vmf_mean_resultant_length(std::size_t m, double kappa);

// Maximum-likelihood concentration for mean resultant length rbar in [0, 1):
// the solution of A_m(kappa) = rbar, found by Newton iterations started at
// kappa_0 = rbar (m - rbar^2) / (1 - rbar^2).
double vmf_kappa_mle(std::size_t m, double rbar);

}  // namespace prag::kde
