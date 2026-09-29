// Persistence of a VectorStore: a single binary file (layout in persist.hpp).
//
// Little-endian only. Numbers are written in the byte order of the host, and
// save() and load() refuse to run on a host that is not little-endian; no
// conversion is implemented.
//
// load() trusts nothing. In this order it checks: the size of the file, the
// magic number, the format version, the byte-order probe, the checksum of the
// header, every header field, the size of the body against the sizes that
// follow from n, dim and the flags (computed without overflow), the checksum
// of the body, and then the content: finite rows of unit length within the
// recorded slack, a centroid that is the mean of the rows, a layout that is a
// permutation, a rotation with rows of unit length, side values and codes in
// their ranges. Only then is the store replaced. What is not checked: that the
// codes are the codes of the rows (that would mean encoding again), and that
// the rows of the rotation are orthogonal to each other.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "persist.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/index/vector_store.hpp"
#include "store_impl.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)  // results must not depend on the instruction set the compiler may use
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace prag::index {

// One multiply-rotate-multiply step per 8-byte word. For a fixed state the
// step is injective in the word, and for a fixed word it is a bijection of the
// state, so a change confined to one word always changes the result.
std::uint64_t checksum64(const void* data, std::size_t size, std::uint64_t seed) noexcept {
  constexpr std::uint64_t kMul1 = 0x9e3779b185ebca87ULL;
  constexpr std::uint64_t kMul2 = 0xc2b2ae3d27d4eb4fULL;
  constexpr std::uint64_t kAdd = 0x165667b19e3779f9ULL;
  const unsigned char* p = static_cast<const unsigned char*>(data);
  std::uint64_t h = mix64(seed) ^ kAdd;
  const std::size_t words = size / 8;
  for (std::size_t i = 0; i < words; ++i) {
    std::uint64_t w = 0;
    std::memcpy(&w, p + 8 * i, 8);
    h = std::rotl(h ^ (w * kMul1), 27) * kMul2 + kAdd;
  }
  const std::size_t tail = size % 8;
  if (tail != 0) {
    std::uint64_t w = 0;
    std::memcpy(&w, p + 8 * words, tail);
    h = std::rotl(h ^ (w * kMul1), 27) * kMul2 + kAdd;
  }
  return mix64(h ^ static_cast<std::uint64_t>(size));
}

namespace {

using Impl = VectorStore::Impl;

constexpr std::size_t kMaxDim = std::size_t{1} << 20;
constexpr std::uint64_t kMaxBlockBytes = std::uint64_t{1} << 40;
constexpr double kUnitNormTolerance = 1e-3;  // rows of the rotation
constexpr double kCentroidTolerance = 1e-5;  // per component
constexpr float kLargestAlignment = 1.0f + 1e-3f;

std::size_t padded(std::size_t bytes) noexcept { return round_up(bytes, 8); }

// a * b, or false when the product does not fit.
bool multiply(std::size_t a, std::size_t b, std::size_t& out) noexcept {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

struct Sections {
  std::size_t rows = 0;  // sizes in bytes, without padding
  std::size_t centroid = 0;
  std::size_t layout = 0;
  std::size_t groups = 0;
  std::size_t rotation = 0;
  std::size_t side = 0;
  std::size_t codes = 0;
  std::size_t total = 0;  // with padding
};

bool section_sizes(std::size_t n, std::size_t dim, CodeKind kind, bool has_groups, bool has_rotation,
                   Sections& out) noexcept {
  std::size_t cells = 0;
  std::size_t square = 0;
  if (!multiply(n, dim, cells) || !multiply(cells, sizeof(float), out.rows)) return false;
  if (!multiply(dim, sizeof(float), out.centroid)) return false;
  if (!multiply(n, sizeof(std::uint32_t), out.layout)) return false;
  out.groups = has_groups ? out.layout : 0;
  out.rotation = 0;
  if (has_rotation && (!multiply(dim, dim, square) || !multiply(square, sizeof(float), out.rotation))) {
    return false;
  }
  if (!multiply(n, kSideBytes, out.side)) return false;
  if (!multiply(n, code_bytes_for(kind, dim), out.codes)) return false;
  std::size_t total = 0;
  for (const std::size_t bytes :
       {out.rows, out.centroid, out.layout, out.groups, out.rotation, out.side, out.codes}) {
    if (bytes > std::numeric_limits<std::size_t>::max() - 8) return false;
    const std::size_t with_padding = padded(bytes);
    if (with_padding > std::numeric_limits<std::size_t>::max() - total) return false;
    total += with_padding;
  }
  out.total = total;
  return true;
}

void append(std::string& out, const void* data, std::size_t bytes) {
  if (bytes != 0) out.append(static_cast<const char*>(data), bytes);
  out.append(padded(bytes) - bytes, '\0');
}

bool all_finite(const float* values, std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

bool reject(std::string_view path, const char* reason) {
  PRAG_ERROR("index: %.*s is not a valid vector store: %s", static_cast<int>(path.size()), path.data(), reason);
  return false;
}

// Copies `count` elements of a section to `out` and moves `cursor` past the
// section and its padding. The caller has checked the sizes.
template <class T>
void take(const unsigned char*& cursor, T* out, std::size_t count) noexcept {
  const std::size_t bytes = count * sizeof(T);
  if (bytes != 0) std::memcpy(static_cast<void*>(out), cursor, bytes);
  cursor += padded(bytes);
}

// The header fields on their own; sizes and content are checked by the caller.
const char* header_defect(const FileHeader& header) noexcept {
  if (header.header_bytes != kHeaderBytes) return "unexpected header size";
  if (header.header_checksum != checksum64(&header, kHeaderChecksummedBytes, kHeaderSeed)) {
    return "the header checksum does not match";
  }
  if (header.code_kind > static_cast<std::uint8_t>(CodeKind::RaBitQ)) return "unknown code kind";
  if (header.has_groups > 1 || header.has_rotation > 1) return "invalid flags";
  for (const std::uint8_t byte : header.reserved8) {
    if (byte != 0) return "reserved bytes are not zero";
  }
  if (header.reserved64 != 0) return "reserved bytes are not zero";
  if (header.n >= static_cast<std::uint64_t>(kNoUnit)) return "too many vectors";
  if (header.dim > kMaxDim) return "the dimension is too large";
  if (header.dim == 0 && header.n != 0) return "vectors of dimension zero";
  if (header.block_bytes == 0 || header.block_bytes > kMaxBlockBytes) return "invalid block size";
  if (!(header.clip_quantile > 0.0 && header.clip_quantile <= 1.0)) return "invalid clip quantile";
  if (!(header.confidence_sigma >= 0.0) || !std::isfinite(header.confidence_sigma)) return "invalid confidence";
  if (!(header.scale > 0.0) || !std::isfinite(header.scale)) return "invalid scale";
  if (!(header.norm_slack >= 0.0) || !std::isfinite(header.norm_slack)) return "invalid norm slack";
  const bool rabitq = header.code_kind == static_cast<std::uint8_t>(CodeKind::RaBitQ);
  if ((header.has_rotation != 0) != (rabitq && header.n != 0)) return "the rotation flag does not fit the codes";
  if (header.has_groups != 0 && header.n == 0) return "groups without vectors";
  return nullptr;
}

// Rows, centroid, layout and rotation of a store whose sections were copied.
const char* content_defect(const Impl& s) {
  const std::size_t n = s.n;
  const std::size_t dim = s.dim;
  if (!all_finite(s.rows.data(), n * dim)) return "a row is not finite";
  if (!all_finite(s.centroid.data(), dim)) return "the centroid is not finite";

  std::vector<double> sum(dim, 0.0);
  double max_dev = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = s.rows.data() + i * dim;
    double norm2 = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const double v = x[j];
      sum[j] += v;
      norm2 += v * v;
    }
    max_dev = std::max(max_dev, std::fabs(norm2 - 1.0));
  }
  // The slack was computed with the same operations, row by row.
  if (0.5 * max_dev > s.norm_slack + 1e-12) return "a row is longer or shorter than recorded";
  for (std::size_t j = 0; j < dim && n != 0; ++j) {
    const double mean = sum[j] / static_cast<double>(n);
    if (std::fabs(mean - static_cast<double>(s.centroid[j])) > kCentroidTolerance) {
      return "the centroid is not the mean of the rows";
    }
  }

  std::vector<std::uint8_t> seen(n, 0);
  for (const std::uint32_t unit : s.layout) {
    if (unit >= n || seen[unit] != 0) return "the layout is not a permutation";
    seen[unit] = 1;
  }

  if (s.rotation != nullptr) {
    const float* rotation = s.rotation->data();
    if (!all_finite(rotation, dim * dim)) return "the rotation is not finite";
    for (std::size_t j = 0; j < dim; ++j) {
      double norm2 = 0.0;
      for (std::size_t i = 0; i < dim; ++i) {
        const double v = rotation[j * dim + i];
        norm2 += v * v;
      }
      if (std::fabs(norm2 - 1.0) > kUnitNormTolerance) return "the rotation is not orthogonal";
    }
  }
  return nullptr;
}

// Side values and codes, once they are in their blocks.
const char* code_defect(const Impl& s) noexcept {
  const bool rabitq = s.options.codes == CodeKind::RaBitQ;
  const std::uint64_t unused_bits = rabitq ? s.words * 64 - s.dim : 0;  // of the last word of a code
  const std::uint64_t unused_mask = unused_bits == 0 ? 0 : ~std::uint64_t{0} << (64 - unused_bits);
  for (std::size_t b = 0; b < s.blocks.size(); ++b) {
    const float* side = s.side(b);
    const std::size_t count = s.blocks[b].count;
    if (!all_finite(side, count * kSideValues)) return "a side value is not finite";
    for (std::size_t v = 0; v < count; ++v) {
      if (side[2 * v] < 0.0f || side[2 * v + 1] < 0.0f) return "a side value is negative";
      if (rabitq && !(side[2 * v + 1] > 0.0f && side[2 * v + 1] <= kLargestAlignment)) {
        return "an alignment lies outside (0, 1]";
      }
    }
    if (!rabitq) {
      const std::int8_t* codes = reinterpret_cast<const std::int8_t*>(s.codes(b));
      for (std::size_t i = 0; i < count * s.dim; ++i) {
        if (codes[i] == std::numeric_limits<std::int8_t>::min()) return "a code lies outside [-127, 127]";
      }
    } else if (unused_mask != 0) {
      const std::uint8_t* codes = s.codes(b);
      for (std::size_t v = 0; v < count; ++v) {
        std::uint64_t last = 0;
        std::memcpy(&last, codes + (v * s.words + s.words - 1) * sizeof(std::uint64_t), sizeof(last));
        if ((last & unused_mask) != 0) return "a code has bits beyond the dimension";
      }
    }
  }
  return nullptr;
}

}  // namespace

bool VectorStore::save(std::string_view path) const {
  if (std::endian::native != std::endian::little) {
    PRAG_ERROR("index: vector stores are written on little-endian hosts only");
    return false;
  }
  const Impl& s = *impl_;
  const bool has_groups = !s.groups.empty();
  const bool has_rotation = s.rotation != nullptr;
  Sections sizes;
  if (!section_sizes(s.n, s.dim, s.options.codes, has_groups, has_rotation, sizes)) {
    PRAG_ERROR("index: the store is too large to be written");
    return false;
  }
  if (s.centroid.size() != s.dim || s.layout.size() != s.n || (has_groups && s.groups.size() != s.n)) {
    PRAG_ERROR("index: internal error, the store does not have the size that follows from its dimensions");
    return false;
  }

  std::string file;
  file.reserve(kHeaderBytes + sizes.total);
  file.assign(kHeaderBytes, '\0');
  append(file, s.rows.data(), sizes.rows);
  append(file, s.centroid.data(), sizes.centroid);
  append(file, s.layout.data(), sizes.layout);
  append(file, s.groups.data(), sizes.groups);
  append(file, has_rotation ? s.rotation->data() : nullptr, sizes.rotation);
  for (std::size_t b = 0; b < s.blocks.size(); ++b) {
    file.append(reinterpret_cast<const char*>(s.side(b)), s.blocks[b].count * kSideBytes);
  }
  file.append(padded(sizes.side) - sizes.side, '\0');
  for (std::size_t b = 0; b < s.blocks.size(); ++b) {
    file.append(reinterpret_cast<const char*>(s.codes(b)), s.blocks[b].count * s.code_bytes);
  }
  file.append(padded(sizes.codes) - sizes.codes, '\0');
  if (file.size() != kHeaderBytes + sizes.total) {
    PRAG_ERROR("index: internal error, the blocks do not hold every vector once");
    return false;
  }

  FileHeader header{};
  std::memcpy(header.magic, kFileMagic, sizeof(header.magic));
  header.version = kFormatVersion;
  header.header_bytes = static_cast<std::uint32_t>(kHeaderBytes);
  header.byte_order = kByteOrderProbe;
  header.code_kind = static_cast<std::uint8_t>(s.options.codes);
  header.has_groups = has_groups ? 1 : 0;
  header.has_rotation = has_rotation ? 1 : 0;
  header.n = s.n;
  header.dim = s.dim;
  header.block_bytes = s.options.block_bytes;
  header.seed = s.options.seed;
  header.clip_quantile = s.options.clip_quantile;
  header.confidence_sigma = s.options.confidence_sigma;
  header.scale = s.scale;
  header.norm_slack = s.norm_slack;
  header.body_bytes = sizes.total;
  header.body_checksum = checksum64(file.data() + kHeaderBytes, sizes.total, kBodySeed);
  header.header_checksum = checksum64(&header, kHeaderChecksummedBytes, kHeaderSeed);
  std::memcpy(file.data(), &header, kHeaderBytes);

  if (!fs::write_file_atomic(path, file)) {
    PRAG_ERROR("index: could not write %.*s", static_cast<int>(path.size()), path.data());
    return false;
  }
  return true;
}

bool VectorStore::load(std::string_view path) {
  if (std::endian::native != std::endian::little) {
    PRAG_ERROR("index: vector stores are read on little-endian hosts only");
    return false;
  }
  const std::optional<std::string> content = fs::read_file(path);
  if (!content) {
    PRAG_ERROR("index: could not read %.*s", static_cast<int>(path.size()), path.data());
    return false;
  }
  const std::string& file = *content;

  // ---- header -------------------------------------------------------------------
  // Magic number, version and byte order come first: a file of another version
  // may have another header, and its checksum means nothing to this reader.
  if (file.size() < kHeaderBytes) return reject(path, "the file is shorter than the header");
  FileHeader header;
  std::memcpy(&header, file.data(), kHeaderBytes);
  if (std::memcmp(header.magic, kFileMagic, sizeof(header.magic)) != 0) return reject(path, "wrong magic number");
  if (header.version != kFormatVersion) return reject(path, "unsupported format version");
  if (header.byte_order != kByteOrderProbe) return reject(path, "written with a different byte order");
  if (const char* defect = header_defect(header)) return reject(path, defect);

  const std::size_t n = static_cast<std::size_t>(header.n);
  const std::size_t dim = static_cast<std::size_t>(header.dim);
  const CodeKind kind = static_cast<CodeKind>(header.code_kind);
  const bool has_groups = header.has_groups != 0;
  const bool has_rotation = header.has_rotation != 0;

  // ---- sizes and body checksum -----------------------------------------------------
  Sections sizes;
  if (!section_sizes(n, dim, kind, has_groups, has_rotation, sizes)) return reject(path, "the sizes overflow");
  if (header.body_bytes != sizes.total) return reject(path, "the body size does not follow from the header");
  if (file.size() - kHeaderBytes != sizes.total) return reject(path, "the file is truncated or has trailing bytes");
  const unsigned char* body = reinterpret_cast<const unsigned char*>(file.data()) + kHeaderBytes;
  if (header.body_checksum != checksum64(body, sizes.total, kBodySeed)) {
    return reject(path, "the body checksum does not match");
  }

  // ---- content ---------------------------------------------------------------------
  auto fresh = std::make_unique<Impl>();
  Impl& s = *fresh;
  s.options.codes = kind;
  s.options.block_bytes = static_cast<std::size_t>(header.block_bytes);
  s.options.seed = header.seed;
  s.options.clip_quantile = header.clip_quantile;
  s.options.confidence_sigma = header.confidence_sigma;
  s.n = n;
  s.dim = dim;
  s.scale = header.scale;
  s.norm_slack = header.norm_slack;
  set_code_geometry(s);

  const unsigned char* cursor = body;
  s.rows.reset(n * dim);
  take(cursor, s.rows.data(), n * dim);
  s.centroid.assign(dim, 0.0f);
  take(cursor, s.centroid.data(), dim);
  s.layout.assign(n, 0);
  take(cursor, s.layout.data(), n);
  if (has_groups) {
    s.groups.assign(n, 0);
    take(cursor, s.groups.data(), n);
  }
  if (has_rotation) {
    auto rotation = std::make_shared<Rotation>(dim * dim);
    take(cursor, rotation->data(), dim * dim);
    s.rotation = std::move(rotation);
  }
  if (const char* defect = content_defect(s)) return reject(path, defect);

  // The layout is a permutation: blocks can be cut and filled.
  finish_derived(s);
  layout_blocks(s);
  for (std::size_t b = 0; b < s.blocks.size(); ++b) {
    const std::size_t bytes = s.blocks[b].count * kSideBytes;
    std::memcpy(static_cast<void*>(s.side(b)), cursor, bytes);
    cursor += bytes;
  }
  cursor += padded(sizes.side) - sizes.side;
  for (std::size_t b = 0; b < s.blocks.size(); ++b) {
    const std::size_t bytes = s.blocks[b].count * s.code_bytes;
    std::memcpy(s.codes(b), cursor, bytes);
    cursor += bytes;
  }
  if (const char* defect = code_defect(s)) return reject(path, defect);

  if (impl_ != empty_impl()) delete impl_;
  impl_ = fresh.release();
  return true;
}

}  // namespace prag::index
