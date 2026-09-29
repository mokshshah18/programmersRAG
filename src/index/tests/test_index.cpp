// Tests of the index module: kernels, the two codes and their error bounds,
// the block layout, the ordered scan, rescoring, persistence and edge cases.
//
// Reference values are computed here, in double precision and with code that
// shares nothing with the library. The private headers are included to reach
// the representation (rotation, scale, file header) where a test needs it.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "../kernels.hpp"
#include "../persist.hpp"
#include "../rng.hpp"
#include "../store_impl.hpp"
#include "index_testing.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/exec/cache_topology.hpp"

using namespace prag;
using namespace prag::index;
using namespace prag::index::testing;

namespace {

constexpr std::size_t kLengths[] = {0,  1,  7,   8,   9,   15,  16,   17,   31,  32, 33,
                                    63, 64, 65,  100, 127, 128, 129,  255,  256, 257, 1000,
                                    1024, 1027};

// dot_f32 written from its definition in vector_store.hpp, one product at a time.
float dot_f32_by_definition(const float* a, const float* b, std::size_t dim) {
  float p[32] = {};
  const std::size_t m = 32 * (dim / 32);
  const std::size_t groups = (dim - m) / 8;
  for (std::size_t j = 0; j < m; ++j) {
    const float product = a[j] * b[j];
    p[j % 32] = p[j % 32] + product;
  }
  for (std::size_t j = m; j < m + 8 * groups; ++j) {
    const float product = a[j] * b[j];
    p[(j - m) % 8] = p[(j - m) % 8] + product;
  }
  float t[8];
  for (int k = 0; k < 8; ++k) t[k] = (p[k] + p[k + 8]) + (p[k + 16] + p[k + 24]);
  float u[4];
  for (int k = 0; k < 4; ++k) u[k] = t[k] + t[k + 4];
  float r = (u[0] + u[2]) + (u[1] + u[3]);
  for (std::size_t j = m + 8 * groups; j < dim; ++j) {
    const float product = a[j] * b[j];
    r = r + product;
  }
  return r;
}

std::uint32_t bits_of(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// A path that no file has yet, and the removal of the file when the test is
// done with it, however it ends. Every file of the tests gets a name of its
// own: replacing a file that was written a moment ago can fail on Windows
// while a virus scanner or an indexer still holds it open.
struct TempFile {
  std::string path;

  explicit TempFile(const char* name) {
    static const std::uint64_t run = now_ns() % 1000000007ULL;
    static unsigned counter = 0;
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "prag_index_test";
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    path = (directory / (std::to_string(run) + "_" + std::to_string(++counter) + "_" + name)).generic_string();
  }
  ~TempFile() { fs::remove_file(path); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
};

bool write_plain(const std::string& path, const std::string& content) {
  std::ofstream out(std::filesystem::path(path), std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  out.close();
  return !out.fail();
}

// Silences the error log while a test provokes errors on purpose.
struct QuietLog {
  log::Level before = log::level();
  QuietLog() { log::set_level(log::Level::Off); }
  ~QuietLog() { log::set_level(before); }
};

std::vector<float> random_unit_vectors(std::size_t count, std::size_t dim, std::uint64_t seed) {
  std::vector<float> out(count * dim);
  Gaussian normal(seed);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<double> v(dim);
    double norm2 = 0.0;
    for (double& x : v) {
      x = normal.next();
      norm2 += x * x;
    }
    const double inverse = 1.0 / std::sqrt(norm2);
    for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = static_cast<float>(v[j] * inverse);
  }
  return out;
}

// Checks |estimate - exact| <= half-width for every pair; returns the largest ratio.
double check_true_bound(const VectorStore& store, std::span<const float> queries, const char* what) {
  const ErrorStatistics stats = measure_errors(store, queries);
  CHECK(stats.pairs == store.size() * (queries.size() / store.dim()));
  if (!(stats.coverage == 1.0)) {
    std::fprintf(stderr, "  %s: coverage %.6f, largest |error| / half-width %.4f\n", what, stats.coverage,
                 stats.max_ratio);
  }
  CHECK(stats.coverage == 1.0);
  return stats.max_ratio;
}

}  // namespace

// ---- 1. kernels ------------------------------------------------------------------

PRAG_TEST(kernel_selection_is_consistent) {
  const std::vector<const Kernels*>& tables = usable_kernels();
  REQUIRE(!tables.empty());
  CHECK(std::strcmp(tables.front()->name, "scalar") == 0);
  CHECK(tables.back() == &kernels());
  CHECK(std::strcmp(simd_kernel(), kernels().name) == 0);
  std::string names;
  for (const Kernels* table : tables) names += std::string(" ") + table->name;
  std::printf("  kernels this CPU can run:%s; selected: %s\n", names.c_str(), simd_kernel());
#if !defined(__AVX2__)
  CHECK_EQ(tables.size(), std::size_t{1});  // compiled without AVX2: the portable kernels only
#endif
}

PRAG_TEST(dot_i8_equals_the_scalar_reference) {
  SplitMix64 random(101);
  for (const std::size_t length : kLengths) {
    for (int repeat = 0; repeat < 8; ++repeat) {
      // Offsets 0..3 move the data off every alignment the kernels may assume.
      const std::size_t offset = static_cast<std::size_t>(repeat) % 4;
      std::vector<std::int8_t> a(length + 4), b(length + 4);
      for (std::size_t j = 0; j < length; ++j) {
        a[offset + j] = static_cast<std::int8_t>(static_cast<int>(random.below(256)) - 128);
        b[offset + j] = static_cast<std::int8_t>(static_cast<int>(random.below(256)) - 128);
      }
      std::int64_t naive = 0;
      for (std::size_t j = 0; j < length; ++j) {
        naive += static_cast<int>(a[offset + j]) * static_cast<int>(b[offset + j]);
      }
      CHECK_EQ(static_cast<std::int64_t>(dot_i8(a.data() + offset, b.data() + offset, length)), naive);
      CHECK_EQ(static_cast<std::int64_t>(dot_i8_scalar(a.data() + offset, b.data() + offset, length)), naive);
      for (const Kernels* table : usable_kernels()) {
        CHECK_EQ(static_cast<std::int64_t>(table->dot_i8(a.data() + offset, b.data() + offset, length)), naive);
      }
    }
    for (const int x : {-128, -127, 0, 127}) {
      for (const int y : {-128, -127, 127}) {
        const std::vector<std::int8_t> a(length, static_cast<std::int8_t>(x));
        const std::vector<std::int8_t> b(length, static_cast<std::int8_t>(y));
        const std::int64_t expected = static_cast<std::int64_t>(length) * x * y;
        CHECK_EQ(static_cast<std::int64_t>(dot_i8(a.data(), b.data(), length)), expected);
        CHECK_EQ(static_cast<std::int64_t>(dot_i8_scalar(a.data(), b.data(), length)), expected);
        for (const Kernels* table : usable_kernels()) {
          CHECK_EQ(static_cast<std::int64_t>(table->dot_i8(a.data(), b.data(), length)), expected);
          CHECK_EQ(static_cast<std::int64_t>(table->dot_i8(b.data(), a.data(), length)), expected);
        }
      }
    }
  }
  // Beyond 32 bits the sum is taken modulo 2^32 by every kernel.
  const std::size_t length = 140000;
  const std::vector<std::int8_t> a(length, static_cast<std::int8_t>(-128));
  const std::int64_t wrapped = (static_cast<std::int64_t>(length) * 128 * 128) % (std::int64_t{1} << 32);
  for (const Kernels* table : usable_kernels()) {
    const std::uint32_t sum = static_cast<std::uint32_t>(table->dot_i8(a.data(), a.data(), length));
    CHECK_EQ(static_cast<std::int64_t>(sum), wrapped);
  }
}

PRAG_TEST(dot_f32_equals_the_scalar_reference_bit_for_bit) {
  SplitMix64 random(202);
  for (const std::size_t length : kLengths) {
    for (int repeat = 0; repeat < 8; ++repeat) {
      const std::size_t offset = static_cast<std::size_t>(repeat) % 4;
      std::vector<float> a(length + 4), b(length + 4);
      double magnitude = 0.0;
      double reference = 0.0;
      for (std::size_t j = 0; j < length; ++j) {
        // Magnitudes over six decades, so that the order of the additions matters.
        const double scale = std::pow(10.0, -6.0 * random.uniform());
        a[offset + j] = static_cast<float>((2.0 * random.uniform() - 1.0) * scale);
        b[offset + j] = static_cast<float>(2.0 * random.uniform() - 1.0);
        reference += static_cast<double>(a[offset + j]) * static_cast<double>(b[offset + j]);
        magnitude += std::fabs(static_cast<double>(a[offset + j]) * static_cast<double>(b[offset + j]));
      }
      const float selected = dot_f32(a.data() + offset, b.data() + offset, length);
      const float scalar = dot_f32_scalar(a.data() + offset, b.data() + offset, length);
      const float defined = dot_f32_by_definition(a.data() + offset, b.data() + offset, length);
      CHECK_EQ(bits_of(selected), bits_of(scalar));
      CHECK_EQ(bits_of(scalar), bits_of(defined));
      for (const Kernels* table : usable_kernels()) {
        CHECK_EQ(bits_of(table->dot_f32(a.data() + offset, b.data() + offset, length)), bits_of(defined));
      }
      CHECK(std::fabs(static_cast<double>(selected) - reference) <= 1e-5 * magnitude + 1e-30);
    }
    for (const float x : {-127.0f, 127.0f}) {
      for (const float y : {-127.0f, 127.0f}) {
        const std::vector<float> a(length, x);
        const std::vector<float> b(length, y);
        // Exact: an integer below 2^24. The empty sum is +0.
        const float expected = length == 0 ? 0.0f : static_cast<float>(length) * x * y;
        CHECK_EQ(bits_of(dot_f32(a.data(), b.data(), length)), bits_of(expected));
        CHECK_EQ(bits_of(dot_f32_scalar(a.data(), b.data(), length)), bits_of(expected));
        for (const Kernels* table : usable_kernels()) {
          CHECK_EQ(bits_of(table->dot_f32(a.data(), b.data(), length)), bits_of(expected));
        }
      }
    }
  }
}

PRAG_TEST(bit_kernels_equal_the_scalar_reference) {
  SplitMix64 random(303);
  for (const std::size_t words : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
                                  std::size_t{6}, std::size_t{7}, std::size_t{8}, std::size_t{16}, std::size_t{17}}) {
    for (int repeat = 0; repeat < 16; ++repeat) {
      std::vector<std::uint64_t> code(words + 1), other(words + 1), planes(4 * words + 1);
      // All ones in the first round: the largest counts there are.
      for (std::uint64_t& w : code) w = repeat == 0 ? ~std::uint64_t{0} : random.next();
      for (std::uint64_t& w : other) w = repeat == 0 ? ~std::uint64_t{0} : random.next();
      for (std::uint64_t& w : planes) w = repeat == 0 ? ~std::uint64_t{0} : random.next();
      std::uint64_t both = 0;
      std::uint64_t ones = 0;
      std::uint64_t weighted = 0;
      for (std::size_t j = 0; j < 64 * words; ++j) {
        const std::uint64_t bit = (code[j / 64] >> (j % 64)) & 1u;
        ones += bit;
        both += bit & ((other[j / 64] >> (j % 64)) & 1u);
        std::uint64_t level = 0;
        for (unsigned p = 0; p < 4; ++p) level |= ((planes[p * words + j / 64] >> (j % 64)) & 1u) << p;
        weighted += bit * level;
      }
      CHECK_EQ(and_popcount(code.data(), other.data(), words), both);
      CHECK_EQ(and_popcount_scalar(code.data(), other.data(), words), both);
      for (const Kernels* table : usable_kernels()) {
        CHECK_EQ(table->and_popcount(code.data(), other.data(), words), both);
        CHECK_EQ(static_cast<std::uint64_t>(table->popcount(code.data(), words)), ones);
        CHECK_EQ(static_cast<std::uint64_t>(table->weighted_popcount(code.data(), planes.data(), words)), weighted);
      }
    }
  }
}

// ---- rotation ----------------------------------------------------------------------

PRAG_TEST(rotation_is_orthogonal_and_does_not_depend_on_the_thread_count) {
  for (const std::size_t dim :
       {std::size_t{1}, std::size_t{2}, std::size_t{33}, std::size_t{100}, std::size_t{384}}) {
    exec::OrderedExecutor one(1);
    exec::OrderedExecutor many(8);
    // Seeds that no other test uses, so that both calls compute the matrix.
    const auto a = make_rotation(dim, 7001 + dim, one);
    const auto b = make_rotation(dim, 8001 + dim, many);
    const auto c = make_rotation(dim, 8001 + dim, one);  // from the cache
    const auto d = make_rotation(dim, 7001 + dim, many);  // from the cache
    REQUIRE(a && b && c && d);
    REQUIRE(a->size() == dim * dim);
    CHECK(b.get() == c.get());
    CHECK(a.get() == d.get());
    double worst = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = i; j < dim; ++j) {
        double dot = 0.0;
        for (std::size_t t = 0; t < dim; ++t) {
          dot += static_cast<double>((*a)[i * dim + t]) * static_cast<double>((*a)[j * dim + t]);
        }
        worst = std::max(worst, std::fabs(dot - (i == j ? 1.0 : 0.0)));
      }
    }
    CHECK(worst < 2e-6);
  }
  // The same seed with 1 and with 8 threads, neither from the cache: the
  // cache holds two matrices, so the first is evicted by two others.
  exec::OrderedExecutor one(1);
  exec::OrderedExecutor many(8);
  const std::size_t dim = 200;
  const auto first = make_rotation(dim, 9001, one);
  make_rotation(dim, 9002, one);
  make_rotation(dim, 9003, one);
  const auto second = make_rotation(dim, 9001, many);
  REQUIRE(first && second);
  CHECK(first.get() != second.get());
  CHECK(std::memcmp(first->data(), second->data(), dim * dim * sizeof(float)) == 0);
}

// ---- 2. Int8: a true bound ---------------------------------------------------------

PRAG_TEST(int8_error_bound_holds_for_every_pair) {
  for (const std::size_t dim : {std::size_t{1024}, std::size_t{384}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    std::vector<float> queries(c.queries.begin(), c.queries.begin() + static_cast<std::ptrdiff_t>(40 * dim));
    const std::vector<float> random = random_unit_vectors(10, dim, 900 + dim);
    queries.insert(queries.end(), random.begin(), random.end());
    // Queries outside the contract: two large components (clipped whatever the
    // scale), and one that is not normalised. The bound must hold all the same.
    std::vector<float> unusual(dim, 0.0f);
    unusual[0] = 0.8f;
    unusual[dim / 2] = -0.6f;
    for (std::size_t j = 0; j < dim; ++j) unusual.push_back(2.5f * c.queries[j]);

    for (const double quantile : {1.0, 0.999, 0.95}) {
      BuildOptions options;
      options.codes = CodeKind::Int8;
      options.clip_quantile = quantile;
      const VectorStore built = build_store(c, options);
      REQUIRE(built.size() == c.data.n);
      const ErrorStatistics stats = measure_errors(built, queries);
      CHECK_EQ(stats.pairs, c.data.n * 50);
      CHECK(stats.coverage == 1.0);
      CHECK(stats.max_ratio <= 1.0);
      CHECK(check_true_bound(built, unusual, "int8, unusual queries") <= 1.0);
      // The bound is of use: below the spread of the scores.
      if (quantile == 1.0) CHECK(stats.mean_half_width < 0.05);
      std::printf("  int8 dim %4zu quantile %.3f: scale %.4f, largest |error| %.5f, half-width mean %.5f max %.5f, "
                  "largest |error| / half-width %.3f\n",
                  dim, quantile, built.impl().scale, stats.max_abs_error, stats.mean_half_width,
                  stats.max_half_width, stats.max_ratio);
    }
  }
}

PRAG_TEST(int8_codes_and_side_values_are_what_the_header_says) {
  const std::size_t dim = 100;
  const Corpus& c = corpus(dim);
  const VectorStore& s = store(dim, CodeKind::Int8);
  const double scale = s.impl().scale;
  double largest = 0.0;
  for (const float v : c.data.rows) largest = std::max(largest, std::fabs(static_cast<double>(v)));
  CHECK_EQ(scale, largest);  // clip_quantile 1: the largest component
  CHECK_EQ(s.bytes_per_code(), dim + 8);
  const double delta = scale / 127.0;
  for (std::size_t b = 0; b < s.blocks().size(); ++b) {
    const exec::Block& block = s.blocks()[b];
    const std::span<const std::uint8_t> payload = s.block_payload(b);
    const float* side = reinterpret_cast<const float*>(payload.data());
    const std::int8_t* codes = reinterpret_cast<const std::int8_t*>(payload.data() + block.count * 8);
    for (std::size_t v = 0; v < block.count; ++v) {
      const std::span<const float> x = s.vector(s.unit_at(block.first + v));
      double residual2 = 0.0;
      double decoded2 = 0.0;
      for (std::size_t j = 0; j < dim; ++j) {
        const long expected = std::lround(127.0 * static_cast<double>(x[j]) / scale);
        CHECK_EQ(static_cast<long>(codes[v * dim + j]), std::clamp(expected, -127L, 127L));
        const double decoded = delta * codes[v * dim + j];
        residual2 += (x[j] - decoded) * (x[j] - decoded);
        decoded2 += decoded * decoded;
      }
      CHECK(side[2 * v] >= std::sqrt(residual2) * (1.0 - 1e-12));
      CHECK(side[2 * v] <= std::sqrt(residual2) * (1.0 + 1e-6));
      CHECK(side[2 * v + 1] >= std::sqrt(decoded2) * (1.0 - 1e-12));
      CHECK(side[2 * v + 1] <= std::sqrt(decoded2) * (1.0 + 1e-6));
    }
  }
}

// ---- 3. RaBitQ: an unbiased estimate and a probabilistic bound -------------------------

PRAG_TEST(rabitq_codes_and_side_values_follow_the_construction) {
  for (const std::size_t dim : {std::size_t{100}, std::size_t{384}}) {
    const VectorStore& s = store(dim, CodeKind::RaBitQ);
    const VectorStore::Impl& impl = s.impl();
    REQUIRE(impl.rotation != nullptr);
    const float* rotation = impl.rotation->data();
    const std::size_t words = (dim + 63) / 64;
    CHECK_EQ(s.bytes_per_code(), words * 8 + 8);
    const std::span<const float> centroid = s.centroid();
    std::size_t sign_mismatches = 0;
    std::size_t near_zero = 0;
    for (std::size_t b = 0; b < s.blocks().size(); b += 3) {
      const exec::Block& block = s.blocks()[b];
      const std::span<const std::uint8_t> payload = s.block_payload(b);
      const float* side = reinterpret_cast<const float*>(payload.data());
      const std::uint8_t* codes = payload.data() + block.count * 8;
      for (std::size_t v = 0; v < block.count; ++v) {
        const std::span<const float> x = s.vector(s.unit_at(block.first + v));
        std::vector<double> o(dim);
        double norm2 = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
          o[j] = static_cast<double>(x[j]) - static_cast<double>(centroid[j]);
          norm2 += o[j] * o[j];
        }
        const double distance = std::sqrt(norm2);
        for (double& value : o) value /= distance;
        double l1 = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
          double rotated = 0.0;
          for (std::size_t t = 0; t < dim; ++t) rotated += static_cast<double>(rotation[j * dim + t]) * o[t];
          l1 += std::fabs(rotated);
          std::uint64_t word = 0;
          std::memcpy(&word, codes + v * words * 8 + (j / 64) * 8, 8);
          const bool bit = ((word >> (j % 64)) & 1u) != 0;
          // The library rotates in float32: a coordinate within rounding of zero may get either sign.
          if (std::fabs(rotated) < 1e-6) {
            ++near_zero;
          } else if (bit != (rotated > 0.0)) {
            ++sign_mismatches;
          }
        }
        for (std::size_t j = dim; j < 64 * words; ++j) {
          std::uint64_t word = 0;
          std::memcpy(&word, codes + v * words * 8 + (j / 64) * 8, 8);
          CHECK_EQ((word >> (j % 64)) & 1u, std::uint64_t{0});
        }
        CHECK_NEAR(side[2 * v], distance, 1e-6);
        CHECK_NEAR(side[2 * v + 1], l1 / std::sqrt(static_cast<double>(dim)), 1e-5);
      }
    }
    CHECK_EQ(sign_mismatches, std::size_t{0});
    std::printf("  rabitq dim %zu: %zu rotated coordinates within 1e-6 of zero\n", dim, near_zero);
  }
}

namespace {

// Two reference values per pair, both computed here in double precision from
// the stored bits and side values.
//   (1) The estimator with the exact rotated query, <xbar, P^T q'> / a. The
//       scan uses the query rounded at random to 4 bits per coordinate: the
//       difference, in units of the standard deviation that the scan
//       attributes to the rounding, must have mean 0 and variance 1.
//   (2) The estimator with the query rounded as the scan rounds it (same
//       generator, same seed). What remains is the float32 arithmetic of the
//       scan, which must stay within the allowance made for it.
void compare_scan_with_double_precision(std::size_t dim, std::size_t queries, bool check_statistics) {
  const Corpus& c = corpus(dim);
  const VectorStore& s = store(dim, CodeKind::RaBitQ);
  const VectorStore::Impl& impl = s.impl();
  const float* rotation = impl.rotation->data();
  const std::size_t words = impl.words;
  const std::span<const float> centroid = s.centroid();
  double centroid2 = 0.0;
  for (const float v : centroid) centroid2 += static_cast<double>(v) * static_cast<double>(v);

  double z_sum = 0.0;
  double z_sum2 = 0.0;
  double largest_float_error = 0.0;
  std::size_t beyond_allowance = 0;
  std::size_t samples = 0;
  for (std::size_t query = 0; query < queries; ++query) {
    const std::span<const float> q = c.query(query);
    const ScanResult result = scan(s, q, pool());
    std::vector<double> centred(dim);
    double norm2 = 0.0;
    double pivot = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      centred[j] = static_cast<double>(q[j]) - static_cast<double>(centroid[j]);
      norm2 += centred[j] * centred[j];
      pivot += static_cast<double>(q[j]) * static_cast<double>(centroid[j]);
    }
    const double dq = std::sqrt(norm2);
    std::vector<double> rotated(dim);
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -lowest;
    for (std::size_t j = 0; j < dim; ++j) {
      double sum = 0.0;
      for (std::size_t t = 0; t < dim; ++t) sum += static_cast<double>(rotation[j * dim + t]) * centred[t] / dq;
      rotated[j] = sum;
      lowest = std::min(lowest, sum);
      highest = std::max(highest, sum);
    }
    const double step = (highest - lowest) / 15.0;
    double variance = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const double t = (rotated[j] - lowest) / step;
      const double fraction = t - std::floor(t);
      variance += fraction * (1.0 - fraction);
    }
    variance *= step * step / static_cast<double>(dim);

    // The rounding of the scan: see encode_rabitq in scan.cpp.
    SplitMix64 random(hash_combine(impl.options.seed, checksum64(q.data(), dim * sizeof(float), 0x7175657279ULL)));
    std::vector<double> rounded(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      const double t = std::clamp((rotated[j] - lowest) / step, 0.0, 15.0);
      double level = std::floor(t);
      if (random.uniform() < t - level) level += 1.0;
      rounded[j] = lowest + step * std::min(level, 15.0);
    }

    for (std::size_t b = 0; b < s.blocks().size(); ++b) {
      const exec::Block& block = s.blocks()[b];
      const float* side = impl.side(b);
      const std::uint8_t* codes = impl.codes(b);
      for (std::size_t v = 0; v < block.count; ++v) {
        double signed_sum = 0.0;
        double signed_rounded = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
          std::uint64_t word = 0;
          std::memcpy(&word, codes + v * words * 8 + (j / 64) * 8, 8);
          const bool bit = ((word >> (j % 64)) & 1u) != 0;
          signed_sum += bit ? rotated[j] : -rotated[j];
          signed_rounded += bit ? rounded[j] : -rounded[j];
        }
        const double d = side[2 * v];
        const double a = side[2 * v + 1];
        const double root = std::sqrt(static_cast<double>(dim));
        const double base = pivot + 0.5 * (1.0 - centroid2 - d * d);
        const double scanned = result.scores[s.unit_at(block.first + v)];
        const double z = (scanned - (base + d * dq * signed_sum / root / a)) / (d * dq * std::sqrt(variance) / a);
        z_sum += z;
        z_sum2 += z * z;
        const double float_error = std::fabs(scanned - (base + d * dq * signed_rounded / root / a));
        largest_float_error = std::max(largest_float_error, float_error);
        if (float_error > 2e-6 * d * (d + dq) + 1e-7) ++beyond_allowance;
        ++samples;
      }
    }
  }
  const double mean = z_sum / static_cast<double>(samples);
  const double variance = z_sum2 / static_cast<double>(samples) - mean * mean;
  std::printf("  dim %4zu, %zu pairs: rounding of the query, standardised: mean %.4f, variance %.4f; "
              "largest error of the float32 arithmetic %.2e\n",
              dim, samples, mean, variance, largest_float_error);
  CHECK_EQ(samples, queries * s.size());
  CHECK_EQ(beyond_allowance, std::size_t{0});
  CHECK(largest_float_error < 5e-7);
  if (!check_statistics) return;
  // The pairs of one query share the rounding of the query, and the codes of
  // one group are similar, so the averages are taken over fewer independent
  // values than there are pairs; the tolerances allow for about 600.
  CHECK(std::fabs(mean) < 0.1);
  CHECK(variance > 0.85 && variance < 1.15);
}

}  // namespace

PRAG_TEST(rabitq_scan_equals_the_estimator_up_to_the_rounding_of_the_query) {
  compare_scan_with_double_precision(384, 20, true);
  compare_scan_with_double_precision(1024, 6, false);
  compare_scan_with_double_precision(100, 6, false);
}

PRAG_TEST(rabitq_estimate_is_unbiased_and_the_bound_has_the_stated_coverage) {
  // Coverage of a half-width of z standard deviations of a Gaussian error.
  const auto gaussian_coverage = [](double z) { return std::erf(z / std::sqrt(2.0)); };
  for (const std::size_t dim : {std::size_t{1024}, std::size_t{384}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    const std::span<const float> queries(c.queries.data(), 50 * dim);
    std::vector<std::uint32_t> query_group(50);
    for (std::size_t j = 0; j < 50; ++j) query_group[j] = c.data.cluster[c.targets[j]];
    const ErrorStatistics at_19 = measure_errors(store(dim, CodeKind::RaBitQ), queries, c.data.cluster, query_group);
    CHECK(at_19.pairs >= 20000);
    CHECK_EQ(at_19.pairs, std::size_t{50} * c.data.n);

    BuildOptions wide;
    wide.codes = CodeKind::RaBitQ;
    wide.confidence_sigma = 3.0;
    const ErrorStatistics at_30 = measure_errors(build_store(c, wide), queries);

    std::printf("  rabitq dim %4zu, %zu pairs: error mean %+.3e, sd %.5f, standard error %.2e (independent pairs) "
                "%.2e (clustered by group of the unit and of the query)\n",
                dim, at_19.pairs, at_19.mean, at_19.stddev, at_19.naive_se, at_19.clustered_se);
    std::printf("      confidence_sigma 1.9: coverage %.4f (Gaussian %.4f), half-width mean %.5f max %.5f, "
                "largest |error| %.5f, rms error / half-width %.4f\n",
                at_19.coverage, gaussian_coverage(1.9), at_19.mean_half_width, at_19.max_half_width,
                at_19.max_abs_error, at_19.rms_ratio);
    std::printf("      confidence_sigma 3.0: coverage %.4f (Gaussian %.4f), half-width mean %.5f max %.5f\n",
                at_30.coverage, gaussian_coverage(3.0), at_30.mean_half_width, at_30.max_half_width);

    // Unbiased: the mean is within three standard errors of zero.
    CHECK(std::fabs(at_19.mean) <= 3.0 * at_19.clustered_se);
    // The half-width is confidence_sigma standard deviations: the root mean
    // square of error / half-width is 1 / confidence_sigma when the variance
    // is right. It is somewhat smaller, because the bound replaces the factor
    // sqrt(1 - <o, q'>^2) of the standard deviation by 1.
    CHECK(at_19.rms_ratio < 1.03 / 1.9);
    CHECK(at_19.rms_ratio > 0.85 / 1.9);
    // Coverage: not below that of a Gaussian error by more than sampling
    // allows (pairs are correlated, so the allowance is generous).
    CHECK(at_19.coverage >= gaussian_coverage(1.9) - 0.01);
    CHECK(at_30.coverage >= 0.99);
    CHECK(at_30.coverage >= gaussian_coverage(3.0) - 0.003);
  }
}

PRAG_TEST(rabitq_estimate_is_unbiased_over_rotations) {
  // The expectation of the estimate is taken over the rotation. With one
  // matrix per seed, the means of the error over the same 20000 pairs are
  // independent draws; their average must vanish within its standard error.
  const std::size_t dim = 128;
  const Corpus c = make_corpus(500, dim, 40, 4242);
  const int seeds = 40;
  double sum = 0.0;
  double sum2 = 0.0;
  for (int seed = 0; seed < seeds; ++seed) {
    BuildOptions options;
    options.codes = CodeKind::RaBitQ;
    options.seed = 100 + static_cast<std::uint64_t>(seed);
    const ErrorStatistics stats = measure_errors(build_store(c, options), c.queries);
    CHECK_EQ(stats.pairs, std::size_t{20000});
    sum += stats.mean;
    sum2 += stats.mean * stats.mean;
  }
  const double mean = sum / seeds;
  const double variance = (sum2 - seeds * mean * mean) / (seeds - 1);
  const double standard_error = std::sqrt(variance / seeds);
  std::printf("  %d rotations, 20000 pairs each: mean error %+.3e, standard error %.2e, t = %+.2f\n", seeds, mean,
              standard_error, mean / standard_error);
  // Student's t with 39 degrees of freedom exceeds 3.5 in absolute value with probability 0.001.
  CHECK(std::fabs(mean) <= 3.5 * standard_error);
}

// ---- 4. rescoring --------------------------------------------------------------------

namespace {

struct TopTen {
  std::size_t identical = 0;  // queries whose ten best units are the exact ten best
  double recall = 0.0;        // mean fraction of the exact ten best that were found
  double rescored = 0.0;      // mean number of units rescored per query
  bool rescored_exactly = true;
};

// Scans, rescores every unit whose interval reaches the tenth best estimate
// and compares the ten best units with the exact ten best.
TopTen top_ten_after_rescoring(const VectorStore& s, const Corpus& c) {
  TopTen out;
  for (std::size_t query = 0; query < c.query_count; ++query) {
    const std::span<const float> q = c.query(query);
    ScanResult result = scan(s, q, pool());
    const std::vector<float> exact = exact_scores(s, q);
    const std::vector<UnitId> best = top_k(result.scores, 10);
    const float tenth = result.scores[best.back()];
    const std::vector<UnitId> units = reaching(result, tenth);
    rescore(s, q, units, result);
    for (const UnitId unit : units) {
      const bool exactly = bits_of(result.scores[unit]) == bits_of(exact[unit]) && result.error[unit] == 0.0f;
      if (!exactly) out.rescored_exactly = false;
    }
    const std::vector<UnitId> found = top_k(result.scores, 10);
    const std::vector<UnitId> wanted = top_k(exact, 10);
    if (found == wanted) ++out.identical;
    std::size_t hits = 0;
    for (const UnitId unit : wanted) hits += std::count(found.begin(), found.end(), unit) != 0 ? 1 : 0;
    out.recall += static_cast<double>(hits) / 10.0;
    out.rescored += static_cast<double>(units.size());
  }
  out.recall /= static_cast<double>(c.query_count);
  out.rescored /= static_cast<double>(c.query_count);
  return out;
}

}  // namespace

PRAG_TEST(rescoring_recovers_the_exact_top_ten_int8) {
  const Corpus& c = corpus(1024);
  REQUIRE(c.query_count == 100);
  const TopTen result = top_ten_after_rescoring(store(1024, CodeKind::Int8), c);
  std::printf("  int8: exact top ten for %zu of 100 queries, recall %.4f, %.1f of %zu units rescored per query\n",
              result.identical, result.recall, result.rescored, c.data.n);
  CHECK_EQ(result.identical, std::size_t{100});
  CHECK(result.rescored_exactly);
  CHECK(result.rescored < 0.25 * static_cast<double>(c.data.n));
}

PRAG_TEST(rescoring_recovers_the_exact_top_ten_rabitq) {
  // The interval of RaBitQ holds with a probability, so the ten best are
  // recovered with a probability. A unit of the exact top ten is lost when
  // its estimate falls short of the tenth best estimate by more than its
  // half-width, an event of probability below Phi(-confidence_sigma) for each
  // unit near the tenth place. With confidence_sigma 5 that is 3e-7 per unit,
  // and the test must find all ten for all 100 queries; with 1.9 it is up to
  // 3e-2 per unit, and the outcome is reported, with a floor on the recall.
  const Corpus& c = corpus(1024);
  REQUIRE(c.query_count == 100);
  const TopTen narrow = top_ten_after_rescoring(store(1024, CodeKind::RaBitQ), c);
  std::printf("  rabitq, confidence_sigma 1.9: exact top ten for %zu of 100 queries, recall %.4f, "
              "%.1f of %zu units rescored per query\n",
              narrow.identical, narrow.recall, narrow.rescored, c.data.n);
  CHECK(narrow.rescored_exactly);
  CHECK(narrow.recall >= 0.9);

  BuildOptions options;
  options.codes = CodeKind::RaBitQ;
  options.confidence_sigma = 5.0;
  const TopTen wide = top_ten_after_rescoring(build_store(c, options), c);
  std::printf("  rabitq, confidence_sigma 5.0: exact top ten for %zu of 100 queries, recall %.4f, "
              "%.1f of %zu units rescored per query\n",
              wide.identical, wide.recall, wide.rescored, c.data.n);
  CHECK_EQ(wide.identical, std::size_t{100});
  CHECK(wide.rescored_exactly);
  CHECK(wide.rescored < 0.25 * static_cast<double>(c.data.n));
}

// ---- 5. layout -------------------------------------------------------------------------

namespace {

void check_layout(const VectorStore& s, const Corpus& c, bool with_groups) {
  const std::size_t n = c.data.n;
  REQUIRE(s.size() == n);
  const std::size_t record = s.bytes_per_code();
  const std::span<const exec::Block> blocks = s.blocks();
  REQUIRE(!blocks.empty());

  std::vector<int> seen(n, 0);
  std::size_t next = 0;
  const std::uint8_t* previous_end = nullptr;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const exec::Block& block = blocks[b];
    CHECK_EQ(block.seq, static_cast<std::uint64_t>(b));
    CHECK_EQ(block.first, next);  // storage order, no gap
    CHECK(block.count >= 1);
    CHECK_EQ(block.bytes, block.count * record);
    // At most block_bytes, unless a single vector is larger than a block.
    CHECK(block.bytes <= s.block_bytes() || block.count == 1);

    const std::span<const std::uint8_t> payload = s.block_payload(b);
    CHECK_EQ(payload.size(), block.bytes);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(payload.data()) % 64, std::uintptr_t{0});
    if (previous_end != nullptr) CHECK(payload.data() >= previous_end);  // blocks do not overlap, and ascend
    previous_end = payload.data() + payload.size();

    const std::uint32_t group = s.group_of_block(b);
    for (std::size_t v = 0; v < block.count; ++v) {
      const UnitId unit = s.unit_at(block.first + v);
      REQUIRE(unit < n);
      ++seen[unit];
      CHECK_EQ(unit, c.layout[block.first + v]);  // storage order is the layout
      CHECK_EQ(s.storage_position(unit), block.first + v);
      CHECK_EQ(with_groups ? c.data.cluster[unit] : 0u, group);  // one group per block
    }
    next += block.count;
  }
  CHECK_EQ(next, n);
  for (std::size_t unit = 0; unit < n; ++unit) CHECK_EQ(seen[unit], 1);
  CHECK_EQ(s.unit_at(n), kNoUnit);
  CHECK_EQ(s.storage_position(static_cast<UnitId>(n)), n);
  CHECK(s.block_payload(blocks.size()).empty());

  if (with_groups) {
    // A block is cut only where it is full or where the group changes.
    for (std::size_t b = 0; b + 1 < blocks.size(); ++b) {
      const bool full = (blocks[b].count + 1) * record > s.block_bytes();
      CHECK(full || s.group_of_block(b) != s.group_of_block(b + 1));
    }
  }
}

}  // namespace

PRAG_TEST(layout_invariants_hold) {
  for (const std::size_t dim : {std::size_t{1024}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
      for (const std::size_t block_bytes :
           {std::size_t{512}, std::size_t{4096}, std::size_t{16384}, std::size_t{65536}}) {
        // Encoding once per kind is enough: reblock() is checked against a fresh build below.
        VectorStore s;
        {
          BuildOptions options;
          options.codes = kind;
          options.block_bytes = block_bytes;
          if (dim == 100 || block_bytes == 16384) {
            s = build_store(c, options);
          } else {
            options.block_bytes = 16384;
            s = build_store(c, options);
            s.reblock(block_bytes);
          }
        }
        CHECK_EQ(s.block_bytes(), block_bytes);
        CHECK_EQ(s.options().block_bytes, block_bytes);
        CHECK(s.code_kind() == kind);
        check_layout(s, c, true);
      }
    }
  }
  // Without groups and with the identity layout.
  const Corpus& c = corpus(100);
  Corpus plain = c;
  for (std::size_t i = 0; i < plain.layout.size(); ++i) plain.layout[i] = static_cast<std::uint32_t>(i);
  BuildOptions options;
  VectorStore s;
  s.build(plain.data.rows, plain.data.dim, {}, {}, options, pool());
  check_layout(s, plain, false);
  CHECK_EQ(s.blocks().size(), (plain.data.n + 682 - 1) / 682);  // 16384 / (16 + 8) = 682 vectors per block

  // block_bytes 0: from the cache topology.
  options.block_bytes = 0;
  s.build(plain.data.rows, plain.data.dim, {}, {}, options, pool());
  CHECK_EQ(s.block_bytes(), exec::recommended_block_bytes(exec::cache_topology()));
  check_layout(s, plain, false);
}

PRAG_TEST(reblock_moves_codes_without_changing_them) {
  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    const Corpus& c = corpus(384);
    BuildOptions options;
    options.codes = kind;
    options.block_bytes = 4096;
    const VectorStore fresh = build_store(c, options);
    options.block_bytes = 16384;
    VectorStore moved = build_store(c, options);
    const ScanResult before = scan(moved, c.query(3), pool());
    moved.reblock(4096);
    REQUIRE(moved.blocks().size() == fresh.blocks().size());
    for (std::size_t b = 0; b < fresh.blocks().size(); ++b) {
      const auto x = fresh.block_payload(b);
      const auto y = moved.block_payload(b);
      REQUIRE(x.size() == y.size());
      CHECK(std::memcmp(x.data(), y.data(), x.size()) == 0);
    }
    const ScanResult after = scan(moved, c.query(3), pool());
    CHECK(same_bits(before.scores, after.scores));
    CHECK(same_bits(before.error, after.error));
    CHECK_NEAR(before.mean, after.mean, 1e-12);
    CHECK_NEAR(before.stddev, after.stddev, 1e-12);
    CHECK(same_scan(after, scan(fresh, c.query(3), pool())));
  }
}

// ---- 6. the ordered scan ---------------------------------------------------------------

PRAG_TEST(scan_is_bit_identical_for_every_thread_count) {
  for (const std::size_t dim : {std::size_t{1024}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
      BuildOptions options;
      options.codes = kind;
      options.block_bytes = dim == 1024 ? 16384 : 2048;  // many blocks in both cases
      VectorStore built;
      if (dim != 1024) built = build_store(c, options);
      const VectorStore& s = dim == 1024 ? store(dim, kind) : built;
      REQUIRE(s.blocks().size() > 24);

      exec::OrderedExecutor single(1);
      const ScanResult reference = scan(s, c.query(0), single);
      const std::span<const float> batch(c.queries.data(), 16 * dim);
      const std::vector<ScanResult> reference_batch = scan_batch(s, batch, single);
      REQUIRE(reference.scores.size() == s.size());
      REQUIRE(reference_batch.size() == 16);
      CHECK(same_scan(reference, reference_batch[0]));
      CHECK(reference.stddev > 0.0);
      // mean and stddev are those of the scores.
      double sum = 0.0;
      for (const float v : reference.scores) sum += v;
      const double mean = sum / static_cast<double>(s.size());
      double sum2 = 0.0;
      for (const float v : reference.scores) sum2 += (v - mean) * (v - mean);
      CHECK_NEAR(reference.mean, mean, 1e-9);
      CHECK_NEAR(reference.stddev, std::sqrt(sum2 / static_cast<double>(s.size())), 1e-9);

      for (const unsigned threads : {1u, 2u, 8u, 24u}) {
        exec::OrderedExecutor executor(threads);
        for (int repeat = 0; repeat < 3; ++repeat) {
          exec::RunOptions run;
          run.trace = repeat == 0;
          const ScanResult result = scan(s, c.query(0), executor, run);
          CHECK(same_scan(result, reference));
          CHECK_EQ(result.stats.admission_violations, std::uint64_t{0});
          CHECK_EQ(result.stats.commit_violations, std::uint64_t{0});
          CHECK_EQ(result.stats.blocks, s.blocks().size());
          CHECK_EQ(result.stats.threads, threads);
          if (run.trace) CHECK_EQ(result.stats.trace.size(), s.blocks().size());

          const std::vector<ScanResult> results = scan_batch(s, batch, executor);
          REQUIRE(results.size() == 16);
          for (std::size_t i = 0; i < results.size(); ++i) {
            CHECK(same_scan(results[i], reference_batch[i]));
            CHECK_EQ(results[i].stats.admission_violations, std::uint64_t{0});
            CHECK_EQ(results[i].stats.commit_violations, std::uint64_t{0});
          }
        }
        // A narrow reorder window forces workers to wait for commits.
        exec::RunOptions narrow;
        narrow.window = 2;
        CHECK(same_scan(scan(s, c.query(0), executor, narrow), reference));
      }
      // Every query of a batch equals its own scan.
      for (std::size_t i = 0; i < 16; ++i) CHECK(same_scan(scan(s, c.query(i), pool()), reference_batch[i]));
    }
  }
}

PRAG_TEST(build_is_bit_identical_for_every_thread_count) {
  const Corpus& c = corpus(384);
  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    BuildOptions options;
    options.codes = kind;
    options.seed = 77;
    options.clip_quantile = 0.999;
    exec::OrderedExecutor single(1);
    const VectorStore reference = build_store(c, options, single);
    for (const unsigned threads : {2u, 8u, 24u}) {
      exec::OrderedExecutor executor(threads);
      const VectorStore s = build_store(c, options, executor);
      REQUIRE(s.blocks().size() == reference.blocks().size());
      for (std::size_t b = 0; b < s.blocks().size(); ++b) {
        const auto x = reference.block_payload(b);
        const auto y = s.block_payload(b);
        REQUIRE(x.size() == y.size());
        CHECK(std::memcmp(x.data(), y.data(), x.size()) == 0);
      }
      CHECK(std::memcmp(s.centroid().data(), reference.centroid().data(), 384 * sizeof(float)) == 0);
      CHECK(same_bits(s.impl().scale, reference.impl().scale));
      CHECK(same_bits(s.impl().norm_slack, reference.impl().norm_slack));
      CHECK(same_scan(scan(s, c.query(1), executor), scan(reference, c.query(1), single)));
    }
  }
}

PRAG_TEST(exact_scores_and_rescore_agree_with_double_precision) {
  const Corpus& c = corpus(1024);
  const VectorStore& s = store(1024, CodeKind::RaBitQ);
  const std::span<const float> q = c.query(5);
  const std::vector<float> exact = exact_scores(s, q);
  REQUIRE(exact.size() == s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    CHECK_NEAR(exact[i], exact_dot(q, s.vector(static_cast<UnitId>(i))), 2e-6);
  }
  ScanResult result = scan(s, q, pool());
  const ScanResult before = result;
  const std::vector<UnitId> units = {0, 7, 1999, 7, kNoUnit, 2000};  // out of range ids are ignored
  rescore(s, q, units, result);
  for (std::size_t i = 0; i < s.size(); ++i) {
    const bool listed = i == 0 || i == 7 || i == 1999;
    CHECK_EQ(bits_of(result.scores[i]), bits_of(listed ? exact[i] : before.scores[i]));
    CHECK_EQ(result.error[i], listed ? 0.0f : before.error[i]);
  }
  CHECK(same_bits(result.mean, before.mean));
  CHECK(same_bits(result.stddev, before.stddev));
  // The target of the query is its nearest unit.
  CHECK_EQ(top_k(exact, 1).front(), c.targets[5]);
}

// ---- 7. persistence ----------------------------------------------------------------------

namespace {

void store_header(std::string& file, const FileHeader& header) { std::memcpy(file.data(), &header, kHeaderBytes); }

FileHeader header_of(const std::string& file) {
  FileHeader header;
  std::memcpy(&header, file.data(), kHeaderBytes);
  return header;
}

// Makes the checksums agree with the content, so that only the content is wrong.
void seal(std::string& file) {
  FileHeader header = header_of(file);
  header.body_bytes = file.size() - kHeaderBytes;
  header.body_checksum = checksum64(file.data() + kHeaderBytes, file.size() - kHeaderBytes, kBodySeed);
  header.header_checksum = checksum64(&header, kHeaderChecksummedBytes, kHeaderSeed);
  store_header(file, header);
}

bool loads(const std::string& content) {
  const TempFile file("candidate.bin");
  REQUIRE(write_plain(file.path, content));
  VectorStore s;
  const bool ok = s.load(file.path);
  if (!ok) CHECK(s.empty());
  return ok;
}

}  // namespace

PRAG_TEST(save_and_load_round_trip_is_bit_identical) {
  for (const std::size_t dim : {std::size_t{384}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
      const TempFile file("roundtrip.bin");
      const std::string& path = file.path;
      BuildOptions options;
      options.codes = kind;
      options.block_bytes = 4096;
      options.seed = 5;
      options.clip_quantile = 0.999;
      options.confidence_sigma = 2.5;
      const VectorStore original = build_store(c, options);
      REQUIRE(original.save(path));

      VectorStore loaded;
      REQUIRE(loaded.load(path));
      CHECK_EQ(loaded.size(), original.size());
      CHECK_EQ(loaded.dim(), original.dim());
      CHECK(loaded.code_kind() == kind);
      CHECK_EQ(loaded.block_bytes(), std::size_t{4096});
      CHECK_EQ(loaded.options().seed, std::uint64_t{5});
      CHECK_EQ(loaded.options().clip_quantile, 0.999);
      CHECK_EQ(loaded.options().confidence_sigma, 2.5);
      CHECK_EQ(loaded.bytes_per_code(), original.bytes_per_code());
      check_layout(loaded, c, true);
      REQUIRE(loaded.blocks().size() == original.blocks().size());
      for (std::size_t b = 0; b < loaded.blocks().size(); ++b) {
        const auto x = original.block_payload(b);
        const auto y = loaded.block_payload(b);
        REQUIRE(x.size() == y.size());
        CHECK(std::memcmp(x.data(), y.data(), x.size()) == 0);
        CHECK_EQ(loaded.group_of_block(b), original.group_of_block(b));
      }
      CHECK(std::memcmp(loaded.centroid().data(), original.centroid().data(), dim * sizeof(float)) == 0);
      for (const UnitId unit : {UnitId{0}, UnitId{1234}, UnitId{1999}}) {
        CHECK(std::memcmp(loaded.vector(unit).data(), original.vector(unit).data(), dim * sizeof(float)) == 0);
      }
      const std::span<const float> batch(c.queries.data(), 8 * dim);
      const std::vector<ScanResult> expected = scan_batch(original, batch, pool());
      const std::vector<ScanResult> found = scan_batch(loaded, batch, pool());
      REQUIRE(found.size() == expected.size());
      for (std::size_t i = 0; i < found.size(); ++i) CHECK(same_scan(found[i], expected[i]));
      CHECK(same_bits(exact_scores(loaded, c.query(0)), exact_scores(original, c.query(0))));

      // Saving the loaded store gives the same file.
      const TempFile again("again.bin");
      REQUIRE(loaded.save(again.path));
      const auto first = fs::read_file(path);
      const auto second = fs::read_file(again.path);
      REQUIRE(first.has_value() && second.has_value());
      CHECK(*first == *second);
    }
  }
}

PRAG_TEST(load_rejects_damaged_files_and_keeps_the_store) {
  const QuietLog quiet;
  const std::size_t dim = 100;
  const Corpus c = make_corpus(120, dim, 4, 77);  // small files: several hundred are written
  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    BuildOptions options;
    options.codes = kind;
    const VectorStore original = build_store(c, options);
    std::optional<std::string> content;
    {
      const TempFile file("valid.bin");
      REQUIRE(original.save(file.path));
      content = fs::read_file(file.path);
    }
    REQUIRE(content.has_value());
    const std::string& valid = *content;
    REQUIRE(valid.size() > kHeaderBytes);
    CHECK(loads(valid));

    // Truncated, at every kind of place, and extended.
    for (const std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{8}, kHeaderBytes - 1, kHeaderBytes,
                                   kHeaderBytes + 1, valid.size() / 2, valid.size() - 8, valid.size() - 1}) {
      CHECK(!loads(valid.substr(0, size)));
    }
    CHECK(!loads(valid + std::string(8, '\0')));
    CHECK(!loads(valid + "x"));

    // One bit flipped, anywhere: header fields, every section, the last byte.
    SplitMix64 random(404);
    std::vector<std::size_t> places = {0,  7,  8,  12, 16, 24,  25,  26,  27,  32,           40,
                                       48, 56, 64, 72, 80, 88,  96,  104, 112, 120,          127,
                                       kHeaderBytes, valid.size() - 1};
    for (int i = 0; i < 80; ++i) places.push_back(static_cast<std::size_t>(random.below(valid.size())));
    for (const std::size_t place : places) {
      std::string changed = valid;
      changed[place] = static_cast<char>(changed[place] ^ (1 << (place % 8)));
      const bool ok = loads(changed);
      if (ok) std::fprintf(stderr, "  a file with a bit flipped in byte %zu was accepted\n", place);
      CHECK(!ok);
    }

    // A different format version, with a header that is otherwise valid.
    for (const std::uint32_t version : {0u, 2u, 0x01000000u}) {
      std::string changed = valid;
      FileHeader header = header_of(changed);
      header.version = version;
      header.header_checksum = checksum64(&header, kHeaderChecksummedBytes, kHeaderSeed);
      store_header(changed, header);
      CHECK(!loads(changed));
    }
    {
      std::string changed = valid;
      FileHeader header = header_of(changed);
      header.byte_order = 0x0807060504030201ULL;  // the probe as a big-endian host writes it
      header.header_checksum = checksum64(&header, kHeaderChecksummedBytes, kHeaderSeed);
      store_header(changed, header);
      CHECK(!loads(changed));
    }

    // Files whose checksums are right and whose content is not.
    const auto forged = [&](const char* what, const auto& change) {
      std::string changed = valid;
      FileHeader header = header_of(changed);
      change(changed, header);
      store_header(changed, header);
      seal(changed);
      const bool ok = loads(changed);
      if (ok) std::fprintf(stderr, "  accepted a file with %s\n", what);
      CHECK(!ok);
    };
    const std::size_t n = c.data.n;
    const std::size_t rows_at = kHeaderBytes;
    const std::size_t centroid_at = rows_at + n * dim * 4;
    const std::size_t layout_at = centroid_at + round_up(dim * 4, 8);
    const std::size_t groups_at = layout_at + round_up(n * 4, 8);
    const std::size_t rotation_at = groups_at + round_up(n * 4, 8);
    const std::size_t side_at = rotation_at + (kind == CodeKind::RaBitQ ? round_up(dim * dim * 4, 8) : 0);
    const std::size_t codes_at = side_at + n * 8;
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    const float negative = -1.0f;
    const float large = 3.0f;
    REQUIRE(codes_at + n * (original.bytes_per_code() - 8) == valid.size());

    forged("more vectors than the body holds", [&](std::string&, FileHeader& h) { h.n += 1; });
    forged("a huge number of vectors", [&](std::string&, FileHeader& h) { h.n = std::uint64_t{1} << 40; });
    forged("sizes that overflow", [&](std::string&, FileHeader& h) {
      h.n = 0xFFFFFFF0u;
      h.dim = std::uint64_t{1} << 20;
    });
    forged("a different dimension", [&](std::string&, FileHeader& h) { h.dim -= 1; });
    forged("dimension zero", [&](std::string&, FileHeader& h) { h.dim = 0; });
    forged("the other code kind", [&](std::string&, FileHeader& h) { h.code_kind ^= 1; });
    forged("an unknown code kind", [&](std::string&, FileHeader& h) { h.code_kind = 2; });
    forged("groups flag cleared", [&](std::string&, FileHeader& h) { h.has_groups = 0; });
    forged("rotation flag changed", [&](std::string&, FileHeader& h) { h.has_rotation ^= 1; });
    forged("block size zero", [&](std::string&, FileHeader& h) { h.block_bytes = 0; });
    forged("a clip quantile of zero", [&](std::string&, FileHeader& h) { h.clip_quantile = 0.0; });
    forged("a confidence that is not a number", [&](std::string&, FileHeader& h) {
      h.confidence_sigma = std::numeric_limits<double>::quiet_NaN();
    });
    forged("a negative scale", [&](std::string&, FileHeader& h) { h.scale = -1.0; });
    forged("an infinite norm slack", [&](std::string&, FileHeader& h) {
      h.norm_slack = std::numeric_limits<double>::infinity();
    });
    forged("reserved bytes in use", [&](std::string&, FileHeader& h) { h.reserved8[2] = 1; });
    forged("a body shorter by eight bytes", [&](std::string& f, FileHeader&) { f.resize(f.size() - 8); });
    forged("a row that is not a number", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + rows_at + 40, &not_a_number, 4);
    });
    forged("a row that is not a unit vector", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + rows_at + 40, &large, 4);
    });
    forged("a centroid that is not the mean", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + centroid_at + 8, &large, 4);
    });
    forged("a layout that is not a permutation", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + layout_at, f.data() + layout_at + 4, 4);
    });
    forged("a layout that leaves the range", [&](std::string& f, FileHeader&) {
      const std::uint32_t beyond = static_cast<std::uint32_t>(n);
      std::memcpy(f.data() + layout_at + 4 * 17, &beyond, 4);
    });
    forged("a negative side value", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + side_at + 16, &negative, 4);
    });
    forged("a side value that is not a number", [&](std::string& f, FileHeader&) {
      std::memcpy(f.data() + side_at + 20, &not_a_number, 4);
    });
    if (kind == CodeKind::RaBitQ) {
      forged("a rotation that is not orthogonal", [&](std::string& f, FileHeader&) {
        std::memcpy(f.data() + rotation_at + 4 * dim * 3, &large, 4);
      });
      forged("an alignment of zero", [&](std::string& f, FileHeader&) {
        std::memset(f.data() + side_at + 8 * 5 + 4, 0, 4);
      });
      forged("an alignment above one", [&](std::string& f, FileHeader&) {
        std::memcpy(f.data() + side_at + 8 * 5 + 4, &large, 4);
      });
      forged("code bits beyond the dimension", [&](std::string& f, FileHeader&) {
        f[codes_at + 16 * 9 + 15] = static_cast<char>(0x80);  // bit 127 of a 100-bit code
      });
    } else {
      forged("a code of -128", [&](std::string& f, FileHeader&) { f[codes_at + 1000] = static_cast<char>(-128); });
    }

    // A store that holds something keeps it when load() fails.
    VectorStore holder = build_store(c, options);
    const ScanResult before = scan(holder, c.query(2), pool());
    const TempFile damaged("damaged.bin");
    const TempFile missing("does_not_exist.bin");
    REQUIRE(write_plain(damaged.path, valid.substr(0, valid.size() - 3)));
    CHECK(!holder.load(damaged.path));
    CHECK(!holder.load(missing.path));
    CHECK_EQ(holder.size(), c.data.n);
    CHECK(same_scan(scan(holder, c.query(2), pool()), before));
  }
}

// ---- 8. edge cases -----------------------------------------------------------------------

PRAG_TEST(empty_stores_behave) {
  const QuietLog quiet;
  VectorStore s;
  CHECK(s.empty());
  CHECK_EQ(s.size(), std::size_t{0});
  CHECK_EQ(s.dim(), std::size_t{0});
  CHECK_EQ(s.bytes_per_code(), std::size_t{0});
  CHECK(s.blocks().empty());
  CHECK(s.vector(0).empty());
  CHECK(s.centroid().empty());
  CHECK_EQ(s.unit_at(0), kNoUnit);
  CHECK_EQ(s.storage_position(0), std::size_t{0});
  CHECK(s.block_payload(0).empty());
  CHECK_EQ(s.group_of_block(0), 0u);
  s.reblock(4096);
  CHECK(s.empty());
  const std::vector<float> q(16, 0.25f);
  CHECK(scan(s, q, pool()).scores.empty());
  CHECK(scan(s, {}, pool()).scores.empty());
  CHECK(scan_batch(s, q, pool()).empty());
  CHECK(exact_scores(s, q).empty());

  // No rows, but a dimension.
  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    BuildOptions options;
    options.codes = kind;
    VectorStore none;
    none.build({}, 16, {}, {}, options, pool());
    CHECK(none.empty());
    CHECK_EQ(none.dim(), std::size_t{16});
    CHECK(none.blocks().empty());
    const ScanResult result = scan(none, q, pool());
    CHECK(result.scores.empty());
    CHECK(result.error.empty());
    CHECK_EQ(result.mean, 0.0);
    CHECK_EQ(result.stddev, 0.0);
    CHECK_EQ(scan_batch(none, q, pool()).size(), std::size_t{1});

    const TempFile file("empty.bin");
    REQUIRE(none.save(file.path));
    VectorStore loaded = build_store(corpus(100), options);
    REQUIRE(loaded.load(file.path));
    CHECK(loaded.empty());
    CHECK_EQ(loaded.dim(), std::size_t{16});
    const TempFile other("default.bin");
    REQUIRE(s.save(other.path));  // the default store
    REQUIRE(loaded.load(other.path));
    CHECK_EQ(loaded.dim(), std::size_t{0});
  }

  // Moving leaves an empty store behind.
  VectorStore a = build_store(corpus(100), BuildOptions{});
  VectorStore b = std::move(a);
  CHECK_EQ(b.size(), std::size_t{2000});
  CHECK(a.empty());  // NOLINT: the moved-from state is specified
  CHECK_EQ(a.dim(), std::size_t{0});
  CHECK(scan(a, q, pool()).scores.empty());
  a = std::move(b);
  CHECK_EQ(a.size(), std::size_t{2000});
  CHECK(b.empty());  // NOLINT
}

PRAG_TEST(invalid_arguments_leave_the_store_empty) {
  const QuietLog quiet;
  const Corpus& c = corpus(100);
  const std::size_t dim = 100;
  const std::span<const float> rows(c.data.rows.data(), 50 * dim);
  const auto rejected = [&](std::span<const float> r, std::size_t d, std::span<const std::uint32_t> layout,
                            std::span<const std::uint32_t> groups, const BuildOptions& options) {
    VectorStore s = build_store(c, BuildOptions{});  // holds something before
    s.build(r, d, layout, groups, options, pool());
    return s.empty() && s.blocks().empty();
  };
  std::vector<std::uint32_t> layout(50);
  for (std::size_t i = 0; i < 50; ++i) layout[i] = static_cast<std::uint32_t>(49 - i);
  const std::vector<std::uint32_t> groups(50, 3);
  BuildOptions options;
  {
    VectorStore s;
    s.build(rows, dim, layout, groups, options, pool());
    CHECK_EQ(s.size(), std::size_t{50});
    CHECK_EQ(s.unit_at(0), 49u);
    CHECK_EQ(s.group_of_block(0), 3u);
  }
  CHECK(rejected(rows, 0, {}, {}, options));
  CHECK(rejected(rows.subspan(0, 50 * dim - 1), dim, {}, {}, options));
  CHECK(rejected(rows, dim, std::span<const std::uint32_t>(layout).subspan(0, 49), {}, options));
  CHECK(rejected(rows, dim, {}, std::span<const std::uint32_t>(groups).subspan(0, 49), options));
  std::vector<std::uint32_t> twice = layout;
  twice[3] = twice[4];
  CHECK(rejected(rows, dim, twice, {}, options));
  std::vector<std::uint32_t> beyond = layout;
  beyond[3] = 50;
  CHECK(rejected(rows, dim, beyond, {}, options));
  for (const double quantile : {0.0, -0.5, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
    BuildOptions bad;
    bad.clip_quantile = quantile;
    CHECK(rejected(rows, dim, {}, {}, bad));
  }
  for (const double sigma :
       {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    BuildOptions bad;
    bad.confidence_sigma = sigma;
    CHECK(rejected(rows, dim, {}, {}, bad));
  }
  std::vector<float> poisoned(rows.begin(), rows.end());
  poisoned[1234] = std::numeric_limits<float>::infinity();
  CHECK(rejected(poisoned, dim, {}, {}, options));
  poisoned[1234] = std::numeric_limits<float>::quiet_NaN();
  CHECK(rejected(poisoned, dim, {}, {}, options));

  // Queries that cannot be scored give empty results.
  const VectorStore& s = store(dim, CodeKind::RaBitQ);
  CHECK(scan(s, c.query(0).subspan(1), pool()).scores.empty());
  CHECK(scan(s, {}, pool()).scores.empty());
  CHECK(scan_batch(s, std::span<const float>(c.queries.data(), 2 * dim + 1), pool()).empty());
  std::vector<float> bad(c.query(0).begin(), c.query(0).end());
  bad[5] = std::numeric_limits<float>::quiet_NaN();
  CHECK(scan(s, bad, pool()).scores.empty());
  CHECK(exact_scores(s, c.query(0).subspan(1)).empty());
  ScanResult result = scan(s, c.query(0), pool());
  const ScanResult before = result;
  rescore(s, c.query(0).subspan(1), std::vector<UnitId>{1, 2}, result);
  CHECK(same_scan(result, before));
}

PRAG_TEST(one_vector_and_identical_vectors) {
  for (const std::size_t dim : {std::size_t{1}, std::size_t{100}, std::size_t{1024}}) {
    const std::vector<float> row = random_unit_vectors(1, dim, 50 + dim);
    const std::vector<float> queries = random_unit_vectors(4, dim, 60 + dim);
    for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
      for (const std::size_t copies : {std::size_t{1}, std::size_t{37}}) {
        std::vector<float> rows;
        for (std::size_t i = 0; i < copies; ++i) rows.insert(rows.end(), row.begin(), row.end());
        BuildOptions options;
        options.codes = kind;
        VectorStore s;
        s.build(rows, dim, {}, {}, options, pool());
        REQUIRE(s.size() == copies);
        CHECK(std::memcmp(s.centroid().data(), row.data(), dim * sizeof(float)) == 0);
        const double bound = check_true_bound(s, queries, "identical vectors");
        CHECK(bound <= 1.0);
        for (std::size_t j = 0; j < 4; ++j) {
          const std::span<const float> q(queries.data() + j * dim, dim);
          const ScanResult result = scan(s, q, pool());
          REQUIRE(result.scores.size() == copies);
          for (std::size_t i = 0; i < copies; ++i) {
            CHECK_EQ(bits_of(result.scores[i]), bits_of(result.scores[0]));
            CHECK(std::isfinite(result.error[i]));
            // A vector at the centroid is known exactly to RaBitQ.
            if (kind == CodeKind::RaBitQ) CHECK(result.error[i] < 1e-4f);
          }
          CHECK_NEAR(result.mean, result.scores[0], 1e-7);
          CHECK(result.stddev < 1e-7);
        }
      }
    }
  }
}

PRAG_TEST(dimensions_that_are_not_multiples_of_the_word_size) {
  for (const std::size_t dim : {std::size_t{2}, std::size_t{3}, std::size_t{63}, std::size_t{65}, std::size_t{127},
                                std::size_t{129}, std::size_t{200}}) {
    const Corpus c = make_corpus(300, dim, 20, 31 + dim);
    BuildOptions options;
    options.codes = CodeKind::Int8;
    const VectorStore int8 = build_store(c, options);
    CHECK_EQ(int8.bytes_per_code(), dim + 8);
    CHECK(check_true_bound(int8, c.queries, "int8") <= 1.0);
    check_layout(int8, c, true);

    options.codes = CodeKind::RaBitQ;
    options.confidence_sigma = 3.0;
    const VectorStore rabitq = build_store(c, options);
    CHECK_EQ(rabitq.bytes_per_code(), 8 * ((dim + 63) / 64) + 8);
    check_layout(rabitq, c, true);
    const ErrorStatistics stats = measure_errors(rabitq, c.queries);
    CHECK_EQ(stats.pairs, std::size_t{6000});
    std::printf("  rabitq dim %3zu, confidence_sigma 3: coverage %.4f, error mean %+.2e, half-width mean %.4f\n", dim,
                stats.coverage, stats.mean, stats.mean_half_width);
    // In low dimension the error is not Gaussian but bounded, so three
    // standard deviations cover at least as much as for a Gaussian.
    CHECK(stats.coverage >= 0.985);
  }
}

PRAG_TEST(queries_at_the_centroid_and_at_zero) {
  for (const std::size_t dim : {std::size_t{1024}, std::size_t{100}}) {
    const Corpus& c = corpus(dim);
    for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
      const VectorStore& s = store(dim, kind);
      const std::span<const float> centroid = s.centroid();
      double norm2 = 0.0;
      for (const float v : centroid) norm2 += static_cast<double>(v) * static_cast<double>(v);
      REQUIRE(norm2 > 0.1);  // an anisotropic corpus
      std::vector<float> direction(dim);
      for (std::size_t j = 0; j < dim; ++j) direction[j] = static_cast<float>(centroid[j] / std::sqrt(norm2));
      const std::vector<float> at_centroid(centroid.begin(), centroid.end());
      const std::vector<float> zero(dim, 0.0f);

      const std::vector<float>* const special[] = {&direction, &at_centroid};
      for (const std::vector<float>* q : special) {
        const ScanResult result = scan(s, *q, pool());
        REQUIRE(result.scores.size() == s.size());
        std::size_t covered = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
          CHECK(std::isfinite(result.scores[i]));
          CHECK(std::isfinite(result.error[i]));
          const double exact = exact_dot(*q, s.vector(static_cast<UnitId>(i)));
          if (std::fabs(result.scores[i] - exact) <= result.error[i]) ++covered;
        }
        if (kind == CodeKind::Int8) {
          CHECK_EQ(covered, s.size());
        } else {
          CHECK(static_cast<double>(covered) >= 0.9 * static_cast<double>(s.size()));
        }
      }
      // With the centroid itself as query the centred query vanishes and
      // RaBitQ is exact: <c, x> = (1 + ||c||^2 - ||x - c||^2) / 2.
      if (kind == CodeKind::RaBitQ) {
        const ScanResult result = scan(s, at_centroid, pool());
        for (std::size_t i = 0; i < s.size(); ++i) {
          CHECK_NEAR(result.scores[i], exact_dot(at_centroid, s.vector(static_cast<UnitId>(i))), 1e-6);
          CHECK(result.error[i] < 1e-4f);
        }
      }

      const ScanResult nothing = scan(s, zero, pool());
      REQUIRE(nothing.scores.size() == s.size());
      for (std::size_t i = 0; i < s.size(); ++i) {
        CHECK_EQ(nothing.scores[i], 0.0f);
        CHECK_EQ(nothing.error[i], 0.0f);
      }
      CHECK_EQ(nothing.mean, 0.0);
      CHECK_EQ(nothing.stddev, 0.0);
      // In a batch, next to queries that are not zero.
      std::vector<float> batch(c.queries.begin(), c.queries.begin() + static_cast<std::ptrdiff_t>(dim));
      batch.insert(batch.end(), zero.begin(), zero.end());
      batch.insert(batch.end(), c.queries.begin() + static_cast<std::ptrdiff_t>(dim),
                   c.queries.begin() + static_cast<std::ptrdiff_t>(2 * dim));
      const std::vector<ScanResult> results = scan_batch(s, batch, pool());
      REQUIRE(results.size() == 3);
      CHECK(same_scan(results[0], scan(s, c.query(0), pool())));
      CHECK(same_scan(results[1], nothing));
      CHECK(same_scan(results[2], scan(s, c.query(1), pool())));
    }
  }
}

PRAG_TEST(rows_that_are_not_unit_vectors_widen_the_bound) {
  // Outside the contract of build(), but it must neither crash nor lie: the
  // largest deviation from unit length enters every half-width.
  const std::size_t dim = 64;
  Corpus c = make_corpus(200, dim, 10, 99);
  for (std::size_t i = 0; i < c.data.n; ++i) {
    const float factor = 0.9f + 0.2f * static_cast<float>(i % 11) / 10.0f;  // lengths from 0.9 to 1.1
    for (std::size_t j = 0; j < dim; ++j) c.data.rows[i * dim + j] *= factor;
  }
  // Three rows of which the third is the centroid of all: e, -e and 0.
  std::vector<float> three(3 * dim, 0.0f);
  three[0] = 1.0f;
  three[dim] = -1.0f;
  const std::vector<float> queries = random_unit_vectors(8, dim, 5);
  for (const CodeKind kind : {CodeKind::Int8, CodeKind::RaBitQ}) {
    BuildOptions options;
    options.codes = kind;
    options.confidence_sigma = 4.0;
    const VectorStore s = build_store(c, options);
    REQUIRE(s.size() == c.data.n);
    CHECK(s.impl().norm_slack > 0.09 && s.impl().norm_slack < 0.12);
    const ErrorStatistics stats = measure_errors(s, c.queries);
    if (kind == CodeKind::Int8) {
      CHECK(stats.coverage == 1.0);
    } else {
      CHECK(stats.coverage >= 0.999);
    }

    VectorStore small;
    small.build(three, dim, {}, {}, options, pool());
    REQUIRE(small.size() == 3);
    for (const float v : small.centroid()) CHECK_EQ(v, 0.0f);
    const ErrorStatistics few = measure_errors(small, queries);
    CHECK(std::isfinite(few.max_half_width));
    CHECK(std::isfinite(few.max_abs_error));
    CHECK(few.coverage == 1.0);  // the slack of 1/2 covers everything here
  }
}
