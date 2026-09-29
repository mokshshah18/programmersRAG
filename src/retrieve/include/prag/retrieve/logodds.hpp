// Log-odds arithmetic shared by the calibration and the trace.
//
// Probabilities are combined as logits l = ln(p / (1 - p)). Logits are clamped
// to [-kMaxLogit, kMaxLogit], so p = 0, p = 1 and non-finite inputs never
// produce an infinite or undefined value. sigmoid(kMaxLogit) differs from 1 by
// about 2.3e-16, the resolution of a double near 1, so the clamp loses nothing
// that a probability stored as a double could have represented.
#pragma once

#include <algorithm>
#include <cmath>

namespace prag::retrieve {

inline constexpr double kMaxLogit = 36.0;
// Bound on |ln(f1 / f0)|: the largest shift between two clamped logits.
inline constexpr double kMaxLogRatio = 2.0 * kMaxLogit;

// NaN maps to the lower bound: an undefined value is no evidence of relevance.
inline double clamp_logit(double l) noexcept {
  if (std::isnan(l)) return -kMaxLogit;
  return std::clamp(l, -kMaxLogit, kMaxLogit);
}

// 1 / (1 + exp(-l)), evaluated without overflow for every l.
inline double sigmoid(double l) noexcept {
  if (l >= 0.0) return 1.0 / (1.0 + std::exp(-l));
  const double e = std::exp(l);
  return e / (1.0 + e);
}

// ln(1 + exp(l)); -softplus(l) = ln(1 - sigmoid(l)) and -softplus(-l) = ln sigmoid(l).
inline double softplus(double l) noexcept {
  return l > 0.0 ? l + std::log1p(std::exp(-l)) : std::log1p(std::exp(l));
}

// Clamped logit of a probability; finite for every input.
inline double logit(double p) noexcept {
  if (std::isnan(p) || p <= 0.0) return -kMaxLogit;
  if (p >= 1.0) return kMaxLogit;
  return clamp_logit(std::log(p) - std::log1p(-p));
}

// ln(1 - exp(x)) for x <= 0, accurate at both ends of the range.
inline double log1m_exp(double x) noexcept {
  constexpr double kLn2 = 0.69314718055994530942;
  return x > -kLn2 ? std::log(-std::expm1(x)) : std::log1p(-std::exp(x));
}

// Clamped logit of the probability p given ln(1 - p) <= 0.
inline double logit_from_log_survival(double log_survival) noexcept {
  if (!(log_survival < 0.0)) return -kMaxLogit;  // p = 0 (or an undefined input)
  return clamp_logit(log1m_exp(log_survival) - log_survival);
}

}  // namespace prag::retrieve
