// Vector storage laid out in cache-sized blocks, and the ordered parallel scan.
//
// Each embedding is kept twice: as float32 (exact rescoring, density estimation)
// and as a compact code (the scan index). Codes are stored in `layout` order,
// which the engine sets to density-mode order so that similar functions are
// adjacent, and are cut into blocks of at most block_bytes so that one block
// plus the encoded query fits in the L1 data cache of the worker scanning it.
//
// Two codes are available.
//
//   Int8    symmetric uniform scalar quantisation with one global scale s,
//           the clip_quantile quantile of |x_j| over the corpus:
//               c_j = clamp(round(127 x_j / s), -127, 127),
//               <q, x> ~ (s / 127)^2 sum_j cq_j cx_j.
//           D bytes per vector plus two floats (the norm of the quantisation
//           residual x - (s / 127) c and the norm of the decoded vector
//           (s / 127) c), from which the error bound is computed.
//
//   RaBitQ  one bit per dimension with an unbiased estimator and a probabilistic
//           error bound (Gao and Long, 2024). Vectors are centred on the corpus
//           centroid c, normalised, rotated by a fixed random orthogonal matrix
//           P and reduced to their signs. With o = (x - c) / ||x - c|| and
//           obar = P sign(P^T o) / sqrt(D), the inner product with the
//           normalised centred query q' is estimated by <obar, q'> / <obar, o>.
//           D / 8 bytes per vector (rounded up to whole 64-bit words) plus two
//           floats (||x - c|| and <obar, o>). The rotated query is rounded to
//           4 bits per dimension, at random and without bias, so that the
//           scan consists of AND and population count.
//
// In both cases `scan` returns an estimate of the cosine <q, x> for every unit
// together with a half-width e such that the exact cosine lies in
// [estimate - e, estimate + e]: always for Int8, with a probability for
// RaBitQ. Units whose interval contains a decision threshold are rescored
// exactly from the float32 rows.
//
// Meaning of the Int8 half-width. With xh and qh the decoded row and query,
// |<q, x> - <qh, xh>| <= ||q|| ||x - xh|| + ||q - qh|| ||xh||, which contains
// the rounding and the clipping of both operands. It is a bound, not an
// estimate of the error: in the tests at D = 1024 with clip_quantile 1 the
// largest error is a tenth of its half-width.
//
// Meaning of the RaBitQ half-width. The error of the estimate of <o, q'> is
// sqrt(1 - <o, q'>^2) * sqrt(1 - a^2) / a * u with a = <obar, o> and u one
// coordinate of a point drawn uniformly from the unit sphere of R^(D-1), so its
// standard deviation is at most sigma = sqrt((1 - a^2) / a^2) / sqrt(D - 1).
// The published bound is confidence_sigma * sigma, and the half-width is that
// bound with the variance of the rounding of the query added to sigma^2 (it
// widens the interval by a few per cent), scaled by ||x - c|| ||q - c||. It is
// an interval of confidence_sigma standard deviations of an almost Gaussian
// error, so for unrelated pairs it holds with probability
// erf(confidence_sigma / sqrt(2)): 94.3 % for 1.9, 99 % for 2.58, 99.7 % for 3,
// and more often for pairs with a large centred cosine. The tests measure
// 94.3 % and 99.7 % for 1.9 and 3 at D = 1024.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/types.hpp"
#include "prag/exec/block.hpp"
#include "prag/exec/ordered_executor.hpp"

namespace prag::index {

enum class CodeKind : std::uint8_t { Int8 = 0, RaBitQ = 1 };

const char* to_string(CodeKind kind);

// Inner-product kernels. `simd_kernel` names the implementation selected for
// this CPU at start-up: "scalar", "avx2", or "avx2-vnni" (AVX2 with the int8
// kernel on AVX-VNNI). All kernels return exactly the same integers and the
// same float bits, so results do not depend on the kernel.
//
// dot_i8 is exact for every input as long as the sum fits 32 bits
// (dim < 131072); beyond that it is the sum modulo 2^32 in every kernel.
//
// dot_f32 is defined with a fixed order of operations, which every kernel
// reproduces bit for bit. Each product is rounded to float32 before it is
// added (no fused multiply-add). With m = 32 * floor(dim / 32), product j < m
// is added to the partial sum p[j mod 32]; the products of the following whole
// groups of eight go to p[0..7]; then
//     t[k] = (p[k] + p[k + 8]) + (p[k + 16] + p[k + 24])      k = 0..7
//     u[k] = t[k] + t[k + 4]                                  k = 0..3
//     r    = (u[0] + u[2]) + (u[1] + u[3])
// and the remaining dim mod 8 products are added to r one by one.
std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::size_t dim);
float dot_f32(const float* a, const float* b, std::size_t dim);
const char* simd_kernel();

// Number of positions at which both bit strings hold a one (64-bit popcount).
std::uint64_t and_popcount(const std::uint64_t* a, const std::uint64_t* b, std::size_t words);

// The portable implementations, whatever kernel is selected (reference for tests).
std::int32_t dot_i8_scalar(const std::int8_t* a, const std::int8_t* b, std::size_t dim);
float dot_f32_scalar(const float* a, const float* b, std::size_t dim);
std::uint64_t and_popcount_scalar(const std::uint64_t* a, const std::uint64_t* b, std::size_t words);

struct BuildOptions {
  CodeKind codes = CodeKind::RaBitQ;
  std::size_t block_bytes = 16 * 1024;  // 0 = exec::recommended_block_bytes(exec::cache_topology())
  std::uint64_t seed = 1;          // seed of the random rotation and of the rounding of queries (RaBitQ)
  double clip_quantile = 1.0;      // Int8: quantile of |x_j| used as the scale s, in (0, 1]
  double confidence_sigma = 1.9;   // RaBitQ: width of the error bound in standard deviations, >= 0
};

class VectorStore {
 public:
  VectorStore();
  ~VectorStore();
  VectorStore(VectorStore&&) noexcept;
  VectorStore& operator=(VectorStore&&) noexcept;
  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;

  // rows: n * dim L2-normalised floats in unit-id order.
  // layout: storage order as a permutation of 0..n-1; empty means identity.
  // group_of_unit: optional group (density mode) per unit; a block never spans
  // two groups, so whole groups can be skipped or scanned together.
  // Invalid arguments (sizes that do not match, a layout that is not a
  // permutation, rows that are not finite, options out of range) are logged
  // and leave the store empty. Rows that are not of unit length are accepted;
  // the largest deviation widens every half-width.
  // The result does not depend on the number of threads of the executor.
  // The first form runs on exec::OrderedExecutor::shared().
  void build(std::span<const float> rows, std::size_t dim, std::span<const std::uint32_t> layout,
             std::span<const std::uint32_t> group_of_unit, const BuildOptions& options);
  void build(std::span<const float> rows, std::size_t dim, std::span<const std::uint32_t> layout,
             std::span<const std::uint32_t> group_of_unit, const BuildOptions& options,
             exec::OrderedExecutor& executor);

  // Cuts the stored codes into blocks of a different size without encoding
  // them again (0 = from the cache topology). Estimates and half-widths do not
  // depend on the block size; `mean` and `stddev` of a scan may change in the
  // last bits, because partial sums are formed per block.
  void reblock(std::size_t block_bytes);

  std::size_t size() const noexcept;
  std::size_t dim() const noexcept;
  bool empty() const noexcept { return size() == 0; }
  CodeKind code_kind() const noexcept;
  std::size_t block_bytes() const noexcept;
  std::size_t bytes_per_code() const noexcept;  // code bytes per vector, side values included
  const BuildOptions& options() const noexcept; // as built or loaded, block_bytes resolved

  std::span<const float> vector(UnitId unit) const;  // float32 original; empty when out of range
  std::span<const float> centroid() const;           // mean of the rows

  // Blocks in storage order. Block::first and Block::count index storage
  // positions, Block::bytes is the size of the payload. A block holds as many
  // vectors of one group as fit into block_bytes, and at least one: only when
  // a single code with its side values is larger than block_bytes does a
  // payload exceed block_bytes.
  std::span<const exec::Block> blocks() const noexcept;
  UnitId unit_at(std::size_t storage_position) const;      // kNoUnit when out of range
  std::size_t storage_position(UnitId unit) const;          // size() when out of range

  // Payload of one block: the side values of its vectors (two floats each)
  // followed by their codes, contiguous, starting on a 64-byte boundary.
  std::span<const std::uint8_t> block_payload(std::size_t block) const;
  // Group shared by every unit of the block (0 when no groups were given).
  std::uint32_t group_of_block(std::size_t block) const;

  // A single little-endian binary file with a magic number, a format version,
  // the sizes and two checksums; it is written through a temporary file and
  // renamed. `load` validates the whole file, header, sizes, checksums and
  // content, and leaves the store unchanged when it returns false. A loaded
  // store scans to the same bits as the store that was saved.
  bool save(std::string_view path) const;
  bool load(std::string_view path);

  struct Impl;
  const Impl& impl() const noexcept { return *impl_; }

 private:
  Impl* impl_;
};

struct ScanResult {
  std::vector<float> scores;  // estimated cosine per unit id
  std::vector<float> error;   // half-width of the interval containing the exact cosine
  double mean = 0.0;          // mean of `scores` over the corpus
  double stddev = 0.0;        // population standard deviation of `scores` (may be zero)
  exec::RunStats stats;
};

// Scores every unit against `query` (dim floats, L2-normalised). Blocks are
// admitted and committed in storage order, so the result is bit-identical for
// every thread count. A query of the wrong length, or with a value that is
// not finite, gives an empty result. The zero vector gives scores and
// half-widths of zero.
ScanResult scan(const VectorStore& store, std::span<const float> query, exec::OrderedExecutor& executor,
                const exec::RunOptions& options = {});

// The same for a batch of queries (queries.size() / dim rows) in blocked form:
// each block is scored against every query of the batch while it is in the
// cache of the worker, before the next block is touched. Result i is
// bit-identical to scan() of query i. All results carry the statistics of the
// one run.
std::vector<ScanResult> scan_batch(const VectorStore& store, std::span<const float> queries,
                                   exec::OrderedExecutor& executor, const exec::RunOptions& options = {});

// Replaces the estimate of each listed unit by its exact float32 cosine and
// sets its error to zero. `mean` and `stddev` are left unchanged.
void rescore(const VectorStore& store, std::span<const float> query, std::span<const UnitId> units,
             ScanResult& result);

// Exact float32 cosine for every unit (reference for tests and experiments).
std::vector<float> exact_scores(const VectorStore& store, std::span<const float> query);

}  // namespace prag::index
