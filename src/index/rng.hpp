// Private to the index module: small seeded generators.
//
// The standard distributions are implementation defined, so the same seed
// gives different numbers with different standard libraries. These do not.
#pragma once

#include <cmath>
#include <cstdint>

namespace prag::index {

// splitmix64 (Steele, Lea and Flood 2014): one 64-bit state, period 2^64.
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  std::uint64_t next() noexcept {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  // Uniform on [0, 1) with 53 random bits.
  double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  // Uniform on {0, .., bound - 1}; the bias of the multiply-shift reduction is
  // below bound / 2^64.
  std::uint64_t below(std::uint64_t bound) noexcept {
    const std::uint64_t x = next();
    const std::uint64_t hi = x >> 32;
    const std::uint64_t lo = x & 0xffffffffULL;
    // (x * bound) >> 64 without a 128-bit type; bound < 2^32 in every use.
    return (hi * bound + ((lo * bound) >> 32)) >> 32;
  }

 private:
  std::uint64_t state_;
};

// Standard normal deviates by the Box-Muller transform.
class Gaussian {
 public:
  explicit Gaussian(std::uint64_t seed) noexcept : bits_(seed) {}

  double next() noexcept {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    // u1 in (0, 1] so that the logarithm is finite.
    const double u1 = (static_cast<double>(bits_.next() >> 11) + 1.0) * 0x1.0p-53;
    const double u2 = bits_.uniform();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

 private:
  SplitMix64 bits_;
  double spare_ = 0.0;
  bool has_spare_ = false;
};

}  // namespace prag::index
