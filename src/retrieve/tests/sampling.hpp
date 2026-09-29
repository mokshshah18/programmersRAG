// Random deviates for the tests of this module.
//
// std::mt19937_64 produces the same sequence everywhere; the distributions of
// <random> do not (their algorithms are left to the library). Tests that state
// a result for one seed therefore draw their samples with the transformations
// below.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace prag::test {

// Uniform on (0, 1): the 53 high bits of one output, centred in their cell.
inline double uniform01(std::mt19937_64& rng) {
  return (static_cast<double>(rng() >> 11) + 0.5) * 0x1.0p-53;
}

inline double uniform(std::mt19937_64& rng, double low, double high) {
  return low + (high - low) * uniform01(rng);
}

// Box-Muller transform; one deviate per pair of uniforms.
inline double normal(std::mt19937_64& rng, double mean = 0.0, double sd = 1.0) {
  const double radius = std::sqrt(-2.0 * std::log(uniform01(rng)));
  const double angle = 6.283185307179586477 * uniform01(rng);
  return mean + sd * radius * std::cos(angle);
}

// Logistic deviate of the given mean and standard deviation.
inline double logistic(std::mt19937_64& rng, double mean = 0.0, double sd = 1.0) {
  const double u = uniform01(rng);
  return mean + sd * (std::sqrt(3.0) / 3.14159265358979323846) * std::log(u / (1.0 - u));
}

}  // namespace prag::test
