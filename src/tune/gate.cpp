// Promotion gate: one-sided paired t-test and the other promotion requirements.
#include <cmath>
#include <cstdio>

#include "prag/tune/pipeline.hpp"

namespace prag::tune {
namespace {

// Continued fraction of the incomplete beta function (modified Lentz).
double beta_continued_fraction(double a, double b, double x) {
  constexpr double tiny = 1e-300;
  constexpr double eps = 1e-15;
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < tiny) d = tiny;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= 1000; ++m) {
    const double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1.0) < eps) break;
  }
  return h;
}

bool finite(double v) { return std::isfinite(v); }

std::string format(const char* fmt, double a, double b = 0.0) {
  char buffer[192];
  std::snprintf(buffer, sizeof buffer, fmt, a, b);
  return buffer;
}

}  // namespace

double regularized_incomplete_beta(double a, double b, double x) {
  if (!(x > 0.0)) return 0.0;
  if (!(x < 1.0)) return 1.0;
  const double log_front = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) +
                           b * std::log1p(-x);
  const double front = std::exp(log_front);
  // Use the continued fraction where it converges quickly, the symmetry relation elsewhere.
  if (x < (a + 1.0) / (a + b + 2.0)) return front * beta_continued_fraction(a, b, x) / a;
  return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b;
}

double student_t_cdf(double t, double df) {
  if (std::isnan(t) || !(df > 0.0)) return std::nan("");
  if (std::isinf(t)) return t > 0 ? 1.0 : 0.0;
  // P(|T| > |t|) = I_x(df/2, 1/2) with x = df / (df + t^2).
  const double x = df / (df + t * t);
  const double tail = 0.5 * regularized_incomplete_beta(0.5 * df, 0.5, x);
  return t > 0.0 ? 1.0 - tail : tail;
}

GateResult evaluate_gate(const GateInput& in, const GatePolicy& policy) {
  GateResult r;
  r.general_nll_base = in.general_base_nll;
  r.general_nll_candidate = in.general_candidate_nll;
  r.quantization_kl = in.quantization_kl;

  const std::size_t n = std::min(in.base_nll.size(), in.candidate_nll.size());
  if (n > 0) {
    double sb = 0.0, sc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      sb += in.base_nll[i];
      sc += in.candidate_nll[i];
    }
    r.domain_nll_base = sb / static_cast<double>(n);
    r.domain_nll_candidate = sc / static_cast<double>(n);
  }

  if (!in.candidate_loads) {
    r.reason = "the candidate model does not load or generate";
    return r;
  }
  if (in.base_nll.size() != in.candidate_nll.size()) {
    r.reason = "base and candidate scores cover different examples";
    return r;
  }
  if (n < policy.min_examples || n < 2) {
    r.reason = "too few held-out examples (" + std::to_string(n) + " < " +
               std::to_string(std::max<std::size_t>(policy.min_examples, 2)) + ")";
    return r;
  }

  // Paired differences, positive when the candidate is better.
  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = in.base_nll[i] - in.candidate_nll[i];
    if (!finite(d)) {
      r.reason = "non-finite negative log-likelihood on held-out example " + std::to_string(i);
      return r;
    }
    mean += d;
  }
  mean /= static_cast<double>(n);
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = in.base_nll[i] - in.candidate_nll[i] - mean;
    ss += d * d;
  }
  const double df = static_cast<double>(n - 1);
  const double sd = std::sqrt(ss / df);
  if (sd > 0.0) {
    const double t = mean / (sd / std::sqrt(static_cast<double>(n)));
    r.p_value = 1.0 - student_t_cdf(t, df);
    if (r.p_value < 0.0) r.p_value = 0.0;
  } else {
    r.p_value = mean > 0.0 ? 0.0 : 1.0;
  }

  if (!finite(in.general_base_nll) || !finite(in.general_candidate_nll) || !finite(in.quantization_kl)) {
    r.reason = "non-finite general-probe or quantisation statistic";
    return r;
  }
  if (!(mean > 0.0) || !(r.p_value < policy.alpha)) {
    r.reason = format("held-out domain loss did not improve significantly (mean gain %.4f nats, p = %.4f)", mean,
                      r.p_value);
    return r;
  }
  const double general_loss = in.general_candidate_nll - in.general_base_nll;
  if (general_loss > policy.margin) {
    r.reason = format("general-probe loss worsened by %.4f nats per token (allowed %.4f)", general_loss,
                      policy.margin);
    return r;
  }
  if (in.quantization_kl > policy.max_kl) {
    r.reason = format("quantisation diverges too much: KL %.4f nats per token (allowed %.4f)", in.quantization_kl,
                      policy.max_kl);
    return r;
  }
  r.pass = true;
  return r;
}

}  // namespace prag::tune
