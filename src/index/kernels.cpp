// Inner-product kernels and their selection at start-up.
//
// Every kernel exists in a portable form. When the translation unit is compiled
// with AVX2 the vector forms are compiled as well and are selected if the CPU
// and the operating system support AVX2; on a CPU that also has AVX-VNNI the
// int8 kernel uses it. Integer kernels are exact, and the float kernel follows
// the order of operations fixed in vector_store.hpp, so all forms return the
// same bits.
//
// Floating-point contraction must stay off in this file: a fused multiply-add
// rounds once where the definition rounds twice.
#include "kernels.hpp"

#include <bit>
#include <vector>

#include "prag/index/vector_store.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#pragma fp_contract(off)
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace prag::index {

namespace {

// ---- portable ----------------------------------------------------------------

std::int32_t dot_i8_portable(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  std::uint32_t sum = 0;  // unsigned: wraps modulo 2^32 like the vector form
  for (std::size_t i = 0; i < dim; ++i) {
    sum += static_cast<std::uint32_t>(static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]));
  }
  return static_cast<std::int32_t>(sum);
}

float dot_f32_portable(const float* a, const float* b, std::size_t dim) {
  float p[32] = {};
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    for (std::size_t k = 0; k < 32; ++k) p[k] += a[i + k] * b[i + k];
  }
  for (; i + 8 <= dim; i += 8) {
    for (std::size_t k = 0; k < 8; ++k) p[k] += a[i + k] * b[i + k];
  }
  float t[8];
  for (std::size_t k = 0; k < 8; ++k) t[k] = (p[k] + p[k + 8]) + (p[k + 16] + p[k + 24]);
  const float u0 = t[0] + t[4];
  const float u1 = t[1] + t[5];
  const float u2 = t[2] + t[6];
  const float u3 = t[3] + t[7];
  float r = (u0 + u2) + (u1 + u3);
  for (; i < dim; ++i) r += a[i] * b[i];
  return r;
}

std::uint64_t and_popcount_portable(const std::uint64_t* a, const std::uint64_t* b, std::size_t words) {
  std::uint64_t count = 0;
  for (std::size_t w = 0; w < words; ++w) count += static_cast<std::uint64_t>(std::popcount(a[w] & b[w]));
  return count;
}

std::uint32_t popcount_portable(const std::uint64_t* code, std::size_t words) {
  std::uint32_t count = 0;
  for (std::size_t w = 0; w < words; ++w) count += static_cast<std::uint32_t>(std::popcount(code[w]));
  return count;
}

std::uint32_t weighted_popcount_portable(const std::uint64_t* code, const std::uint64_t* planes,
                                         std::size_t words) {
  std::uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (std::size_t w = 0; w < words; ++w) {
    const std::uint64_t c = code[w];
    s0 += static_cast<std::uint32_t>(std::popcount(c & planes[w]));
    s1 += static_cast<std::uint32_t>(std::popcount(c & planes[words + w]));
    s2 += static_cast<std::uint32_t>(std::popcount(c & planes[2 * words + w]));
    s3 += static_cast<std::uint32_t>(std::popcount(c & planes[3 * words + w]));
  }
  return s0 + 2 * s1 + 4 * s2 + 8 * s3;
}

constexpr Kernels kPortable = {"scalar",          dot_i8_portable,   dot_f32_portable,
                               and_popcount_portable, popcount_portable, weighted_popcount_portable};

// ---- AVX2 --------------------------------------------------------------------

#if defined(__AVX2__)

// The hardware population count, without the run-time check std::popcount
// performs on some standard libraries.
inline std::uint64_t popcount64(std::uint64_t x) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
  return __popcnt64(x);
#else
  return static_cast<std::uint64_t>(std::popcount(x));
#endif
}

// Bytes are widened to 16 bits and multiplied pairwise into 32 bits
// (VPMOVSXBW + VPMADDWD). A pair sum is at most 2 * 128^2 = 32768, so nothing
// saturates for any input, -128 included; VPMADDUBSW would saturate.
std::int32_t dot_i8_avx2(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    const __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16)));
    const __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16)));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
  }
  if (i + 16 <= dim) {
    const __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
    i += 16;
  }
  const __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  std::uint32_t sum = static_cast<std::uint32_t>(_mm_cvtsi128_si32(s));
  for (; i < dim; ++i) {
    sum += static_cast<std::uint32_t>(static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]));
  }
  return static_cast<std::int32_t>(sum);
}

float dot_f32_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 p0 = _mm256_setzero_ps();  // partial sums 0..7
  __m256 p1 = _mm256_setzero_ps();  // 8..15
  __m256 p2 = _mm256_setzero_ps();  // 16..23
  __m256 p3 = _mm256_setzero_ps();  // 24..31
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    p0 = _mm256_add_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    p1 = _mm256_add_ps(p1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    p2 = _mm256_add_ps(p2, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
    p3 = _mm256_add_ps(p3, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
  }
  for (; i + 8 <= dim; i += 8) {
    p0 = _mm256_add_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  const __m256 t = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
  const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
  const __m128 v = _mm_add_ps(u, _mm_movehl_ps(u, u));              // (u0 + u2, u1 + u3, ..)
  const __m128 w = _mm_add_ss(v, _mm_shuffle_ps(v, v, 0x55));       // (u0 + u2) + (u1 + u3)
  float r = _mm_cvtss_f32(w);
  for (; i < dim; ++i) r += a[i] * b[i];
  return r;
}

// Population counts of 256 bits at a time: every byte is split into two
// nibbles, the number of ones of a nibble is looked up (VPSHUFB), and the byte
// counts are summed into the four 64-bit lanes (VPSADBW). The scalar
// instruction counts one word per cycle at best; this form is not tied to one
// execution port. Words that do not fill a vector are counted one by one.
inline __m256i ones_per_lane(__m256i bits) noexcept {
  const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                          0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i nibble = _mm256_set1_epi8(0x0f);
  const __m256i low = _mm256_shuffle_epi8(lookup, _mm256_and_si256(bits, nibble));
  const __m256i high = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(bits, 4), nibble));
  return _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256());
}

inline std::uint64_t sum_of_lanes(__m256i lanes) noexcept {
  const __m128i pair = _mm_add_epi64(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
  return static_cast<std::uint64_t>(_mm_cvtsi128_si64(pair)) +
         static_cast<std::uint64_t>(_mm_extract_epi64(pair, 1));
}

inline __m256i load_words(const std::uint64_t* words) noexcept {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(words));
}

std::uint64_t and_popcount_avx2(const std::uint64_t* a, const std::uint64_t* b, std::size_t words) {
  __m256i acc = _mm256_setzero_si256();
  std::size_t w = 0;
  for (; w + 4 <= words; w += 4) {
    acc = _mm256_add_epi64(acc, ones_per_lane(_mm256_and_si256(load_words(a + w), load_words(b + w))));
  }
  std::uint64_t count = sum_of_lanes(acc);
  for (; w < words; ++w) count += popcount64(a[w] & b[w]);
  return count;
}

std::uint32_t popcount_avx2(const std::uint64_t* code, std::size_t words) {
  __m256i acc = _mm256_setzero_si256();
  std::size_t w = 0;
  for (; w + 4 <= words; w += 4) acc = _mm256_add_epi64(acc, ones_per_lane(load_words(code + w)));
  std::uint64_t count = sum_of_lanes(acc);
  for (; w < words; ++w) count += popcount64(code[w]);
  return static_cast<std::uint32_t>(count);
}

std::uint32_t weighted_popcount_avx2(const std::uint64_t* code, const std::uint64_t* planes, std::size_t words) {
  const std::uint64_t* p0 = planes;
  const std::uint64_t* p1 = planes + words;
  const std::uint64_t* p2 = planes + 2 * words;
  const std::uint64_t* p3 = planes + 3 * words;
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  std::size_t w = 0;
  for (; w + 4 <= words; w += 4) {
    const __m256i c = load_words(code + w);
    acc0 = _mm256_add_epi64(acc0, ones_per_lane(_mm256_and_si256(c, load_words(p0 + w))));
    acc1 = _mm256_add_epi64(acc1, ones_per_lane(_mm256_and_si256(c, load_words(p1 + w))));
    acc2 = _mm256_add_epi64(acc2, ones_per_lane(_mm256_and_si256(c, load_words(p2 + w))));
    acc3 = _mm256_add_epi64(acc3, ones_per_lane(_mm256_and_si256(c, load_words(p3 + w))));
  }
  const __m256i weighted =
      _mm256_add_epi64(_mm256_add_epi64(acc0, _mm256_slli_epi64(acc1, 1)),
                       _mm256_add_epi64(_mm256_slli_epi64(acc2, 2), _mm256_slli_epi64(acc3, 3)));
  std::uint64_t sum = sum_of_lanes(weighted);
  for (; w < words; ++w) {
    const std::uint64_t c = code[w];
    sum += popcount64(c & p0[w]) + 2 * popcount64(c & p1[w]) + 4 * popcount64(c & p2[w]) +
           8 * popcount64(c & p3[w]);
  }
  return static_cast<std::uint32_t>(sum);
}

constexpr Kernels kAvx2 = {"avx2",            dot_i8_avx2,   dot_f32_avx2,
                           and_popcount_avx2, popcount_avx2, weighted_popcount_avx2};

// AVX2 and POPCNT present, and the operating system saves the YMM registers.
bool cpu_has_avx2() noexcept {
#if defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuid(regs, 0);
  if (regs[0] < 7) return false;
  __cpuid(regs, 1);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  const bool popcnt = (regs[2] & (1 << 23)) != 0;
  if (!osxsave || !avx || !popcnt) return false;
  if ((_xgetbv(0) & 0x6) != 0x6) return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
#else
  return false;
#endif
}

// ---- AVX-VNNI ----------------------------------------------------------------
//
// The intrinsics need no compiler switch with MSVC; other compilers provide
// them when the translation unit is compiled for AVX-VNNI.
#if defined(_MSC_VER) || defined(__AVXVNNI__)
#define PRAG_HAVE_AVX_VNNI 1

// VPDPBUSD multiplies unsigned bytes by signed bytes and adds the four
// products of a 32-bit lane to it, without saturation in between. The byte a
// with its top bit inverted, read as unsigned, is a + 128, and
//     sum_j a_j b_j  =  sum_j (a_j + 128) b_j  -  128 sum_j b_j,
// where the second sum is the same instruction with the constant 128. Exact
// for every input, -128 included. Four accumulators per sum, because the
// instruction takes several cycles to return its result.
std::int32_t dot_i8_vnni(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  const __m256i top = _mm256_set1_epi8(static_cast<char>(0x80));
  __m256i sum0 = _mm256_setzero_si256();
  __m256i sum1 = _mm256_setzero_si256();
  __m256i sum2 = _mm256_setzero_si256();
  __m256i sum3 = _mm256_setzero_si256();
  __m256i off0 = _mm256_setzero_si256();
  __m256i off1 = _mm256_setzero_si256();
  __m256i off2 = _mm256_setzero_si256();
  __m256i off3 = _mm256_setzero_si256();
  const auto load = [](const std::int8_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  };
  std::size_t i = 0;
  for (; i + 128 <= dim; i += 128) {
    const __m256i b0 = load(b + i);
    const __m256i b1 = load(b + i + 32);
    const __m256i b2 = load(b + i + 64);
    const __m256i b3 = load(b + i + 96);
    sum0 = _mm256_dpbusd_avx_epi32(sum0, _mm256_xor_si256(load(a + i), top), b0);
    sum1 = _mm256_dpbusd_avx_epi32(sum1, _mm256_xor_si256(load(a + i + 32), top), b1);
    sum2 = _mm256_dpbusd_avx_epi32(sum2, _mm256_xor_si256(load(a + i + 64), top), b2);
    sum3 = _mm256_dpbusd_avx_epi32(sum3, _mm256_xor_si256(load(a + i + 96), top), b3);
    off0 = _mm256_dpbusd_avx_epi32(off0, top, b0);
    off1 = _mm256_dpbusd_avx_epi32(off1, top, b1);
    off2 = _mm256_dpbusd_avx_epi32(off2, top, b2);
    off3 = _mm256_dpbusd_avx_epi32(off3, top, b3);
  }
  for (; i + 32 <= dim; i += 32) {
    const __m256i b0 = load(b + i);
    sum0 = _mm256_dpbusd_avx_epi32(sum0, _mm256_xor_si256(load(a + i), top), b0);
    off0 = _mm256_dpbusd_avx_epi32(off0, top, b0);
  }
  const __m256i sums = _mm256_add_epi32(_mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));
  const __m256i offs = _mm256_add_epi32(_mm256_add_epi32(off0, off1), _mm256_add_epi32(off2, off3));
  const __m256i acc = _mm256_sub_epi32(sums, offs);
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  std::uint32_t sum = static_cast<std::uint32_t>(_mm_cvtsi128_si32(s));
  for (; i < dim; ++i) {
    sum += static_cast<std::uint32_t>(static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]));
  }
  return static_cast<std::int32_t>(sum);
}

constexpr Kernels kAvx2Vnni = {"avx2-vnni",       dot_i8_vnni,   dot_f32_avx2,
                               and_popcount_avx2, popcount_avx2, weighted_popcount_avx2};

// CPUID leaf 7, sub-leaf 1, EAX bit 4.
bool cpu_has_avx_vnni() noexcept {
#if defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 7, 0);
  if (regs[0] < 1) return false;
  __cpuidex(regs, 7, 1);
  return (regs[0] & (1 << 4)) != 0;
#else
  return __builtin_cpu_supports("avxvnni");
#endif
}

#endif  // AVX-VNNI

#endif  // __AVX2__

// The tables this CPU can run, the fastest last.
std::vector<const Kernels*> usable() {
  std::vector<const Kernels*> tables = {&kPortable};
#if defined(__AVX2__)
  if (cpu_has_avx2()) {
    tables.push_back(&kAvx2);
#if defined(PRAG_HAVE_AVX_VNNI)
    if (cpu_has_avx_vnni()) tables.push_back(&kAvx2Vnni);
#endif
  }
#endif
  return tables;
}

}  // namespace

const std::vector<const Kernels*>& usable_kernels() {
  static const std::vector<const Kernels*> tables = usable();
  return tables;
}

const Kernels& kernels() noexcept {
  static const Kernels& selected = *usable_kernels().back();
  return selected;
}

std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  return kernels().dot_i8(a, b, dim);
}

float dot_f32(const float* a, const float* b, std::size_t dim) { return kernels().dot_f32(a, b, dim); }

std::uint64_t and_popcount(const std::uint64_t* a, const std::uint64_t* b, std::size_t words) {
  return kernels().and_popcount(a, b, words);
}

const char* simd_kernel() { return kernels().name; }

std::int32_t dot_i8_scalar(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  return dot_i8_portable(a, b, dim);
}

float dot_f32_scalar(const float* a, const float* b, std::size_t dim) { return dot_f32_portable(a, b, dim); }

std::uint64_t and_popcount_scalar(const std::uint64_t* a, const std::uint64_t* b, std::size_t words) {
  return and_popcount_portable(a, b, words);
}

double dot_f64(const double* a, const double* b, std::size_t n) noexcept {
  double p[8] = {};
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (std::size_t k = 0; k < 8; ++k) p[k] += a[i + k] * b[i + k];
  }
  double r = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}

void subtract_scaled(double* v, const double* q, double factor, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) v[i] -= factor * q[i];
}

}  // namespace prag::index
