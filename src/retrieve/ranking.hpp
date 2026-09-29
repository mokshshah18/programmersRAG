// Private to the module: standardisation of scores and the order of a ranking,
// shared by the trace and the baselines.
#pragma once

#include <cmath>
#include <limits>

#include "prag/core/types.hpp"
#include "prag/retrieve/calibration.hpp"
#include "prag/retrieve/trace.hpp"

namespace prag::retrieve::detail {

// z = (score - mean) / stddev. Every z is 0 when stddev is not positive and
// finite; a score that is not finite maps to -infinity, the lowest possible.
class Standardiser {
 public:
  Standardiser(double mean, double stddev) noexcept
      : mean_(std::isfinite(mean) ? mean : 0.0),
        inverse_((stddev > 0.0 && std::isfinite(stddev)) ? 1.0 / stddev : 0.0) {}

  double operator()(float score) const noexcept {
    if (!std::isfinite(score)) return -std::numeric_limits<double>::infinity();
    return (static_cast<double>(score) - mean_) * inverse_;
  }

 private:
  double mean_;
  double inverse_;
};

// Larger value first, undefined values last, ties by lower id.
template <class Value>
bool ranks_before(Value value_a, UnitId a, Value value_b, UnitId b) noexcept {
  const bool nan_a = std::isnan(value_a);
  const bool nan_b = std::isnan(value_b);
  if (nan_a != nan_b) return nan_b;
  if (!nan_a && value_a != value_b) return value_a > value_b;
  return a < b;
}

// A unit retrieved on its score alone.
inline TracedUnit scored_unit(UnitId id, float score, const Standardiser& standardise,
                              const Calibration& calibration) {
  const double z = standardise(score);
  TracedUnit unit;
  unit.id = id;
  unit.score = score;
  unit.z = static_cast<float>(z);
  unit.p_score = static_cast<float>(calibration.posterior(z));
  unit.p = unit.p_score;
  return unit;
}

}  // namespace prag::retrieve::detail
