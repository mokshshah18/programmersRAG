// Rows without a direction (private header).
//
// The functions of this module expect unit vectors. Two kinds of rows are not:
// a row of zeros, and a row with a component that is not finite (NaN or
// infinite), which an embedding model can return for a degenerate input. A row
// of zeros is harmless: its inner product with everything is zero. A row that
// is not finite is not: it turns every sum it enters into NaN, and with it the
// result of every other row.
//
// The entry points of the module therefore never let such a row into a sum:
//
//   FiniteRows   replaces the rows that are not finite by rows of zeros, for
//                the functions whose results are indexed by row;
//   RowSubset    leaves rows out, for the functions that reduce the data to a
//                few numbers (the mean and the covariance, a concentration).
//
// Both refer to the caller's storage, without a copy, when there is nothing to
// replace or to leave out.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "kernels.hpp"

namespace prag::kde::detail {

// True when every component is finite. The squares are summed in double
// precision, where the square of a float cannot overflow, so the sum is finite
// exactly when the components are.
inline bool is_finite_row(const float* x, std::size_t m) noexcept { return std::isfinite(dot(x, x, m)); }

class FiniteRows {
 public:
  FiniteRows(std::span<const float> rows, std::size_t m) {
    const std::size_t n = m == 0 ? 0 : rows.size() / m;
    view_ = rows.first(n * m);
    for (std::size_t i = 0; i < n; ++i) {
      if (is_finite_row(rows.data() + i * m, m)) continue;
      if (storage_.empty()) storage_.assign(view_.begin(), view_.end());
      std::fill_n(storage_.begin() + static_cast<std::ptrdiff_t>(i * m), m, 0.0f);
      ++replaced_;
    }
    if (replaced_ != 0) view_ = storage_;
  }
  FiniteRows(const FiniteRows&) = delete;
  FiniteRows& operator=(const FiniteRows&) = delete;

  std::span<const float> rows() const noexcept { return view_; }
  std::size_t replaced() const noexcept { return replaced_; }

 private:
  std::span<const float> view_;
  std::vector<float> storage_;
  std::size_t replaced_ = 0;
};

enum class Keep {
  Finite,    // every row whose components are finite
  Directed,  // of these, the rows that are not zero
};

class RowSubset {
 public:
  RowSubset(std::span<const float> rows, std::size_t m, Keep keep) {
    const std::size_t n = m == 0 ? 0 : rows.size() / m;
    view_ = rows.first(n * m);
    bool copying = false;
    for (std::size_t i = 0; i < n; ++i) {
      const float* x = rows.data() + i * m;
      const double norm2 = dot(x, x, m);
      if (std::isfinite(norm2) && (keep == Keep::Finite || norm2 > 0.0)) {
        if (copying) storage_.insert(storage_.end(), x, x + m);
        continue;
      }
      if (!copying) {
        storage_.reserve((n - 1) * m);
        storage_.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(i * m));
        copying = true;
      }
      ++removed_;
    }
    if (copying) view_ = storage_;
  }
  RowSubset(const RowSubset&) = delete;
  RowSubset& operator=(const RowSubset&) = delete;

  std::span<const float> rows() const noexcept { return view_; }
  std::size_t removed() const noexcept { return removed_; }

 private:
  std::span<const float> view_;
  std::vector<float> storage_;
  std::size_t removed_ = 0;
};

}  // namespace prag::kde::detail
