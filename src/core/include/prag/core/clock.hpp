// Monotonic timing helpers.
#pragma once

#include <chrono>
#include <cstdint>

namespace prag {

inline std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

class Stopwatch {
 public:
  Stopwatch() noexcept : start_(now_ns()) {}
  void reset() noexcept { start_ = now_ns(); }
  std::uint64_t elapsed_ns() const noexcept { return now_ns() - start_; }
  double elapsed_ms() const noexcept { return static_cast<double>(elapsed_ns()) / 1e6; }
  double elapsed_s() const noexcept { return static_cast<double>(elapsed_ns()) / 1e9; }

 private:
  std::uint64_t start_;
};

}  // namespace prag
