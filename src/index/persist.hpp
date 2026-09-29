// Private to the index module: the layout of a persisted VectorStore.
//
// The file is a header of kHeaderBytes followed by the body. Every integer and
// every floating-point number is little-endian. The body consists of the
// sections below, in this order, each padded with zero bytes to a multiple of
// eight bytes:
//
//     rows          n * dim float32      unit-id order
//     centroid      dim float32
//     layout        n uint32             storage position -> unit
//     groups        n uint32             only when has_groups
//     rotation      dim * dim float32    only when has_rotation (row j = column j of P)
//     side values   n * 2 float32        storage order
//     codes         n * code_bytes       storage order
//
// Blocks are not stored: they follow from the layout, the groups and
// block_bytes, so a loaded store has the blocks of the store that was saved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace prag::index {

inline constexpr char kFileMagic[8] = {'P', 'R', 'A', 'G', 'V', 'E', 'C', '\n'};
inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint64_t kByteOrderProbe = 0x0102030405060708ULL;
inline constexpr std::uint64_t kHeaderSeed = 0x686561646572ULL;  // checksum seeds
inline constexpr std::uint64_t kBodySeed = 0x626f6479ULL;

struct FileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;      // sizeof(FileHeader)
  std::uint64_t byte_order;        // kByteOrderProbe as the writer stores it
  std::uint8_t code_kind;          // CodeKind
  std::uint8_t has_groups;         // 0 or 1
  std::uint8_t has_rotation;       // 0 or 1
  std::uint8_t reserved8[5];       // zero
  std::uint64_t n;
  std::uint64_t dim;
  std::uint64_t block_bytes;
  std::uint64_t seed;
  double clip_quantile;
  double confidence_sigma;
  double scale;
  double norm_slack;
  std::uint64_t body_bytes;
  std::uint64_t body_checksum;     // checksum64(body, kBodySeed)
  std::uint64_t reserved64;        // zero
  std::uint64_t header_checksum;   // checksum64 of the bytes before this field, kHeaderSeed
};

inline constexpr std::size_t kHeaderBytes = 128;
inline constexpr std::size_t kHeaderChecksummedBytes = kHeaderBytes - sizeof(std::uint64_t);

static_assert(sizeof(FileHeader) == kHeaderBytes);
static_assert(std::is_trivially_copyable_v<FileHeader>);

}  // namespace prag::index
