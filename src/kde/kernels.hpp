// Inner kernels of the kde module (private header).
//
// Every sum is accumulated in a fixed order: eight running sums, element j going
// to sum number j mod 8, combined pairwise at the end. The vector (AVX2) and the
// scalar code paths perform the same additions in the same order, and neither
// uses fused multiply-add, so a result does not depend on how the loop was
// compiled, on the thread that ran it, or on the block it belonged to.
#pragma once

#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#define PRAG_KDE_AVX2 1
#else
#define PRAG_KDE_AVX2 0
#endif

namespace prag::kde::detail {

inline double combine(const double* s) noexcept {
  return ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
}

// sum_j a[j] * b[j] in double precision; a holds float32 data.
inline double dot(const float* a, const double* b, std::size_t n) noexcept {
  double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  for (; j + 8 <= n; j += 8) {
    const __m256d a0 = _mm256_cvtps_pd(_mm_loadu_ps(a + j));
    const __m256d a1 = _mm256_cvtps_pd(_mm_loadu_ps(a + j + 4));
    s0 = _mm256_add_pd(s0, _mm256_mul_pd(a0, _mm256_loadu_pd(b + j)));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a1, _mm256_loadu_pd(b + j + 4)));
  }
  _mm256_storeu_pd(s, s0);
  _mm256_storeu_pd(s + 4, s1);
#else
  for (; j + 8 <= n; j += 8) {
    for (std::size_t l = 0; l < 8; ++l) s[l] += static_cast<double>(a[j + l]) * b[j + l];
  }
#endif
  for (std::size_t l = 0; j < n; ++j, ++l) s[l] += static_cast<double>(a[j]) * b[j];
  return combine(s);
}

// sum_j a[j] * b[j] in double precision; both operands hold float32 data.
inline double dot(const float* a, const float* b, std::size_t n) noexcept {
  double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  for (; j + 8 <= n; j += 8) {
    const __m256d a0 = _mm256_cvtps_pd(_mm_loadu_ps(a + j));
    const __m256d a1 = _mm256_cvtps_pd(_mm_loadu_ps(a + j + 4));
    const __m256d b0 = _mm256_cvtps_pd(_mm_loadu_ps(b + j));
    const __m256d b1 = _mm256_cvtps_pd(_mm_loadu_ps(b + j + 4));
    s0 = _mm256_add_pd(s0, _mm256_mul_pd(a0, b0));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a1, b1));
  }
  _mm256_storeu_pd(s, s0);
  _mm256_storeu_pd(s + 4, s1);
#else
  for (; j + 8 <= n; j += 8) {
    for (std::size_t l = 0; l < 8; ++l) s[l] += static_cast<double>(a[j + l]) * static_cast<double>(b[j + l]);
  }
#endif
  for (std::size_t l = 0; j < n; ++j, ++l) s[l] += static_cast<double>(a[j]) * static_cast<double>(b[j]);
  return combine(s);
}

// sum_j a[j] * b[j]; both operands in double precision.
inline double dot(const double* a, const double* b, std::size_t n) noexcept {
  double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  for (; j + 8 <= n; j += 8) {
    s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_loadu_pd(a + j), _mm256_loadu_pd(b + j)));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(_mm256_loadu_pd(a + j + 4), _mm256_loadu_pd(b + j + 4)));
  }
  _mm256_storeu_pd(s, s0);
  _mm256_storeu_pd(s + 4, s1);
#else
  for (; j + 8 <= n; j += 8) {
    for (std::size_t l = 0; l < 8; ++l) s[l] += a[j + l] * b[j + l];
  }
#endif
  for (std::size_t l = 0; j < n; ++j, ++l) s[l] += a[j] * b[j];
  return combine(s);
}

// y[j] += w * x[j]
inline void axpy(double w, const float* x, double* y, std::size_t n) noexcept {
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  const __m256d weight = _mm256_set1_pd(w);
  for (; j + 4 <= n; j += 4) {
    const __m256d v = _mm256_cvtps_pd(_mm_loadu_ps(x + j));
    _mm256_storeu_pd(y + j, _mm256_add_pd(_mm256_loadu_pd(y + j), _mm256_mul_pd(weight, v)));
  }
#endif
  for (; j < n; ++j) y[j] += w * static_cast<double>(x[j]);
}

inline void axpy(double w, const double* x, double* y, std::size_t n) noexcept {
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  const __m256d weight = _mm256_set1_pd(w);
  for (; j + 4 <= n; j += 4) {
    _mm256_storeu_pd(y + j, _mm256_add_pd(_mm256_loadu_pd(y + j), _mm256_mul_pd(weight, _mm256_loadu_pd(x + j))));
  }
#endif
  for (; j < n; ++j) y[j] += w * x[j];
}

// y[j] = f * y[j] + x[j]
inline void scale_add(double f, const float* x, double* y, std::size_t n) noexcept {
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  const __m256d factor = _mm256_set1_pd(f);
  for (; j + 4 <= n; j += 4) {
    const __m256d v = _mm256_cvtps_pd(_mm_loadu_ps(x + j));
    _mm256_storeu_pd(y + j, _mm256_add_pd(_mm256_mul_pd(factor, _mm256_loadu_pd(y + j)), v));
  }
#endif
  for (; j < n; ++j) y[j] = f * y[j] + static_cast<double>(x[j]);
}

// sum_j a[j] * b[j] in single precision (sixteen running sums), for rankings
// where seven significant digits are enough.
inline float dot_single(const float* a, const float* b, std::size_t n) noexcept {
  float s[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::size_t j = 0;
#if PRAG_KDE_AVX2
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  for (; j + 16 <= n; j += 16) {
    s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a + j + 8), _mm256_loadu_ps(b + j + 8)));
  }
  _mm256_storeu_ps(s, s0);
  _mm256_storeu_ps(s + 8, s1);
#else
  for (; j + 16 <= n; j += 16) {
    for (std::size_t l = 0; l < 16; ++l) s[l] += a[j + l] * b[j + l];
  }
#endif
  for (std::size_t l = 0; j < n; ++j, ++l) s[l] += a[j] * b[j];
  float total[8];
  for (std::size_t l = 0; l < 8; ++l) total[l] = s[l] + s[l + 8];
  return ((total[0] + total[4]) + (total[1] + total[5])) + ((total[2] + total[6]) + (total[3] + total[7]));
}

}  // namespace prag::kde::detail
