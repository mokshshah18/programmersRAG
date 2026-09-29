// Private to the index module: the representation of a VectorStore.
//
// Storage of a block (its payload), starting on a 64-byte boundary of `arena`:
//
//     side values   count * 2 floats      two per vector, in storage order
//     codes         count * code_bytes    one code per vector, in storage order
//
// Side values: Int8    [0] ||x - (s/127) c||   rounded up
//                      [1] ||(s/127) c||       rounded up
//              RaBitQ  [0] ||x - centroid||
//                      [1] <obar, o>
// Codes:       Int8    dim signed bytes in [-127, 127]
//              RaBitQ  `words` 64-bit words; bit j of the code is bit (j mod 64)
//                      of word (j / 64); bits beyond dim are zero
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "aligned.hpp"
#include "prag/exec/block.hpp"
#include "prag/exec/ordered_executor.hpp"
#include "prag/index/vector_store.hpp"

namespace prag::index {

inline constexpr std::size_t kSideValues = 2;
inline constexpr std::size_t kSideBytes = kSideValues * sizeof(float);
inline constexpr unsigned kQueryBits = 4;  // bits per dimension of the quantised RaBitQ query

using Rotation = AlignedArray<float>;  // dim x dim, orthonormal rows; row j is column j of P

struct VectorStore::Impl {
  BuildOptions options;         // block_bytes resolved
  std::size_t n = 0;
  std::size_t dim = 0;
  std::size_t words = 0;        // RaBitQ: 64-bit words per code
  std::size_t code_bytes = 0;   // bytes of one code, side values not included
  double scale = 1.0;           // Int8: s
  double norm_slack = 0.0;      // max_i | ||x_i||^2 - 1 | / 2, the error of taking rows as unit vectors
  double centroid_norm2 = 0.0;  // ||centroid||^2

  AlignedArray<float> rows;             // n * dim, unit-id order
  std::vector<float> centroid;          // dim
  std::vector<std::uint32_t> layout;    // storage position -> unit
  std::vector<std::uint32_t> position;  // unit -> storage position
  std::vector<std::uint32_t> groups;    // group per unit; empty when none were given
  std::shared_ptr<const Rotation> rotation;  // RaBitQ only

  std::vector<exec::Block> blocks;      // storage order
  std::vector<std::size_t> offsets;     // byte offset of each payload in `arena`
  AlignedArray<std::uint8_t> arena;
  std::size_t max_block_count = 0;

  std::size_t record_bytes() const noexcept { return code_bytes + kSideBytes; }

  const std::uint8_t* payload(std::size_t block) const noexcept { return arena.data() + offsets[block]; }
  const float* side(std::size_t block) const noexcept {
    return reinterpret_cast<const float*>(payload(block));
  }
  const std::uint8_t* codes(std::size_t block) const noexcept {
    return payload(block) + blocks[block].count * kSideBytes;
  }
  float* side(std::size_t block) noexcept { return reinterpret_cast<float*>(arena.data() + offsets[block]); }
  std::uint8_t* codes(std::size_t block) noexcept {
    return arena.data() + offsets[block] + blocks[block].count * kSideBytes;
  }

  std::uint32_t group_of(std::uint32_t unit) const noexcept { return groups.empty() ? 0u : groups[unit]; }
};

// The representation shared by every empty store: default constructed and
// moved-from stores point to it, so they need no allocation and behave as
// empty stores. It is never modified and never deleted.
VectorStore::Impl* empty_impl() noexcept;

// Bytes of one code for the given kind and dimension.
std::size_t code_bytes_for(CodeKind kind, std::size_t dim) noexcept;
std::size_t resolve_block_bytes(std::size_t requested) noexcept;

// Sets words and code_bytes from options.codes and dim.
void set_code_geometry(VectorStore::Impl& s) noexcept;
// Sets `position` from `layout`, and centroid_norm2 from `centroid`.
void finish_derived(VectorStore::Impl& s);
// Cuts the storage order into blocks that never span two groups, and allocates
// the (zeroed) arena. Needs n, layout, groups, code_bytes and options.block_bytes.
void layout_blocks(VectorStore::Impl& s);

// Where a storage position lives.
struct Slot {
  std::size_t block = 0;
  std::size_t index = 0;  // within the block
};
Slot locate(const VectorStore::Impl& s, std::size_t storage_position) noexcept;

// Random orthogonal matrix from the QR factorisation of a Gaussian matrix.
// Deterministic in (dim, seed); independent of the executor's thread count.
std::shared_ptr<const Rotation> make_rotation(std::size_t dim, std::uint64_t seed, exec::OrderedExecutor& executor);

// The reorder window the executor will use for these options.
std::size_t resolved_window(const exec::RunOptions& options, const exec::OrderedExecutor& executor) noexcept;

// Buffers for results that are finished but not yet committed. The executor
// admits block i only after block i - window has been committed, so the blocks
// in flight have distinct sequence numbers modulo the window and slot
// seq mod window is free when block seq is admitted. Allocated once per run;
// slots are whole cache lines, so two workers never write to the same line.
class ResultRing {
 public:
  ResultRing(std::size_t window, std::size_t blocks, std::size_t slot_bytes)
      : window_(window == 0 ? 1 : window),
        slot_bytes_(round_up(slot_bytes == 0 ? 1 : slot_bytes, kCacheLine)),
        storage_((blocks < window_ ? blocks : window_) * slot_bytes_) {}

  std::uint8_t* slot(std::uint64_t seq) noexcept {
    return storage_.data() + static_cast<std::size_t>(seq % window_) * slot_bytes_;
  }
  std::size_t window() const noexcept { return window_; }

 private:
  std::size_t window_;
  std::size_t slot_bytes_;
  AlignedArray<std::uint8_t> storage_;
};

// Rounds a non-negative double to float without ever rounding down, so that a
// bound stays a bound.
float round_up_to_float(double value) noexcept;

// 64-bit checksum over 8-byte little-endian words (see persist.cpp).
std::uint64_t checksum64(const void* data, std::size_t size, std::uint64_t seed) noexcept;

}  // namespace prag::index
