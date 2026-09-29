// Construction of a VectorStore: statistics of the rows, block layout, codes.
//
// Every pass over the rows runs through the ordered executor. The passes that
// encode write each code to its final place; the places of two blocks are
// disjoint, and a code is a function of its row alone, so nothing depends on
// the order in which the workers run. The statistics (centroid, largest
// component) are sums and maxima: partial results are formed per block and
// combined in `commit`, in sequence order.
#include "prag/index/vector_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "kernels.hpp"
#include "prag/core/log.hpp"
#include "prag/exec/cache_topology.hpp"
#include "store_impl.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)  // results must not depend on the instruction set the compiler may use
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace prag::index {

const char* to_string(CodeKind kind) {
  switch (kind) {
    case CodeKind::Int8:
      return "int8";
    case CodeKind::RaBitQ:
      return "rabitq";
  }
  return "unknown";
}

// ---- helpers shared with scan.cpp and persist.cpp ------------------------------

std::size_t code_bytes_for(CodeKind kind, std::size_t dim) noexcept {
  return kind == CodeKind::Int8 ? dim : sizeof(std::uint64_t) * ((dim + 63) / 64);
}

std::size_t resolve_block_bytes(std::size_t requested) noexcept {
  return requested != 0 ? requested : exec::recommended_block_bytes(exec::cache_topology());
}

void set_code_geometry(VectorStore::Impl& s) noexcept {
  s.code_bytes = code_bytes_for(s.options.codes, s.dim);
  s.words = s.options.codes == CodeKind::RaBitQ ? s.code_bytes / sizeof(std::uint64_t) : 0;
}

void finish_derived(VectorStore::Impl& s) {
  s.position.assign(s.n, 0);
  for (std::size_t p = 0; p < s.n; ++p) s.position[s.layout[p]] = static_cast<std::uint32_t>(p);
  double norm2 = 0.0;
  for (const float c : s.centroid) norm2 += static_cast<double>(c) * static_cast<double>(c);
  s.centroid_norm2 = norm2;
}

void layout_blocks(VectorStore::Impl& s) {
  s.blocks.clear();
  s.offsets.clear();
  s.max_block_count = 0;
  const std::size_t record = s.record_bytes();
  for (std::size_t run = 0; run < s.n;) {
    // A run is a maximal stretch of the storage order inside one group.
    const std::uint32_t group = s.group_of(s.layout[run]);
    std::size_t end = run + 1;
    while (end < s.n && s.group_of(s.layout[end]) == group) ++end;
    for (const exec::Block& part : exec::partition_fixed(end - run, record, s.options.block_bytes)) {
      s.blocks.push_back(exec::Block{s.blocks.size(), run + part.first, part.count, part.bytes});
    }
    run = end;
  }
  std::size_t offset = 0;
  s.offsets.reserve(s.blocks.size());
  for (const exec::Block& b : s.blocks) {
    s.offsets.push_back(offset);
    offset = round_up(offset + b.bytes, kCacheLine);
    s.max_block_count = std::max(s.max_block_count, b.count);
  }
  s.arena.reset(offset);
}

Slot locate(const VectorStore::Impl& s, std::size_t storage_position) noexcept {
  const auto after = std::upper_bound(s.blocks.begin(), s.blocks.end(), storage_position,
                                      [](std::size_t p, const exec::Block& b) { return p < b.first; });
  const std::size_t block = static_cast<std::size_t>(after - s.blocks.begin()) - 1;
  return Slot{block, storage_position - s.blocks[block].first};
}

std::size_t resolved_window(const exec::RunOptions& options, const exec::OrderedExecutor& executor) noexcept {
  return options.window != 0 ? options.window : static_cast<std::size_t>(executor.threads()) * 4;
}

float round_up_to_float(double value) noexcept {
  if (!(value <= static_cast<double>(std::numeric_limits<float>::max()))) {
    return std::numeric_limits<float>::infinity();  // too large, or not a number
  }
  float f = static_cast<float>(value);
  if (static_cast<double>(f) < value) f = std::nextafter(f, std::numeric_limits<float>::infinity());
  return f;
}

VectorStore::Impl* empty_impl() noexcept {
  static VectorStore::Impl instance;
  return &instance;
}

// ---- build -------------------------------------------------------------------

namespace {

using Impl = VectorStore::Impl;

bool is_permutation_of_range(std::span<const std::uint32_t> layout, std::size_t n) {
  if (layout.size() != n) return false;
  std::vector<std::uint8_t> seen(n, 0);
  for (const std::uint32_t unit : layout) {
    if (unit >= n || seen[unit] != 0) return false;
    seen[unit] = 1;
  }
  return true;
}

struct RowSummary {
  double max_abs = 0.0;   // largest |x_j|
  double max_dev = 0.0;   // largest | ||x||^2 - 1 |
  bool finite = true;
};

// Centroid, largest component and largest deviation from unit norm. Partial
// sums are formed per block of rows and added in sequence order.
RowSummary summarise_rows(Impl& s, exec::OrderedExecutor& executor) {
  const std::size_t dim = s.dim;
  const auto blocks = exec::partition_fixed(s.n, dim * sizeof(float), s.options.block_bytes);
  exec::RunOptions run;
  run.window = resolved_window(run, executor);
  ResultRing ring(run.window, blocks.size(), (dim + 3) * sizeof(double));

  std::vector<double> sum(dim, 0.0);
  RowSummary summary;
  executor.run(
      blocks,
      [&](const exec::Block& b, unsigned) {
        double* out = reinterpret_cast<double*>(ring.slot(b.seq));
        std::fill(out, out + dim + 3, 0.0);
        double max_abs = 0.0;
        double max_dev = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < b.count; ++i) {
          const float* x = s.rows.data() + (b.first + i) * dim;
          double norm2 = 0.0;
          for (std::size_t j = 0; j < dim; ++j) {
            const double v = x[j];
            out[j] += v;
            norm2 += v * v;
            const double a = std::fabs(v);
            if (!(a <= static_cast<double>(std::numeric_limits<float>::max()))) finite = false;
            if (a > max_abs) max_abs = a;
          }
          max_dev = std::max(max_dev, std::fabs(norm2 - 1.0));
        }
        out[dim] = max_abs;
        out[dim + 1] = max_dev;
        out[dim + 2] = finite ? 1.0 : 0.0;
        return out;
      },
      [&](const exec::Block&, double*&& out) {
        for (std::size_t j = 0; j < dim; ++j) sum[j] += out[j];
        summary.max_abs = std::max(summary.max_abs, out[dim]);
        summary.max_dev = std::max(summary.max_dev, out[dim + 1]);
        if (out[dim + 2] == 0.0) summary.finite = false;
      },
      run);

  s.centroid.assign(dim, 0.0f);
  if (summary.finite) {
    for (std::size_t j = 0; j < dim; ++j) s.centroid[j] = static_cast<float>(sum[j] / static_cast<double>(s.n));
  }
  return summary;
}

// The clip_quantile quantile of |x_j| over all components of all rows. Large
// inputs are sampled at a fixed stride, which keeps the selection within
// kMaxSample values and is deterministic.
double component_quantile(const Impl& s, double quantile) {
  constexpr std::size_t kMaxSample = std::size_t{1} << 24;
  const std::size_t total = s.n * s.dim;
  const std::size_t stride = (total + kMaxSample - 1) / kMaxSample;
  std::vector<float> sample;
  sample.reserve(total / stride + 1);
  for (std::size_t i = 0; i < total; i += stride) sample.push_back(std::fabs(s.rows[i]));
  const double wanted = std::ceil(quantile * static_cast<double>(sample.size()));
  std::size_t k = wanted >= 1.0 ? static_cast<std::size_t>(wanted) - 1 : 0;
  k = std::min(k, sample.size() - 1);
  std::nth_element(sample.begin(), sample.begin() + static_cast<std::ptrdiff_t>(k), sample.end());
  return static_cast<double>(sample[k]);
}

void encode_int8(Impl& s, exec::OrderedExecutor& executor) {
  const std::size_t dim = s.dim;
  const double delta = s.scale / 127.0;
  const double inverse = 127.0 / s.scale;
  const auto blocks = exec::partition_fixed(s.n, dim * sizeof(float), s.options.block_bytes);
  executor.run(
      blocks,
      [&](const exec::Block& b, unsigned) {
        for (std::size_t i = 0; i < b.count; ++i) {
          const std::size_t p = b.first + i;
          const float* x = s.rows.data() + static_cast<std::size_t>(s.layout[p]) * dim;
          const Slot slot = locate(s, p);
          std::int8_t* code = reinterpret_cast<std::int8_t*>(s.codes(slot.block)) + slot.index * dim;
          double residual2 = 0.0;
          std::int64_t code2 = 0;
          for (std::size_t j = 0; j < dim; ++j) {
            const double v = x[j];
            double level = std::floor(v * inverse + 0.5);
            level = std::min(127.0, std::max(-127.0, level));
            const int c = static_cast<int>(level);
            code[j] = static_cast<std::int8_t>(c);
            const double r = v - level * delta;  // rounding error, or clipping error when |x_j| > s
            residual2 += r * r;
            code2 += static_cast<std::int64_t>(c) * c;
          }
          float* side = s.side(slot.block) + kSideValues * slot.index;
          side[0] = round_up_to_float(std::sqrt(residual2));
          side[1] = round_up_to_float(delta * std::sqrt(static_cast<double>(code2)));
        }
        return 0;
      },
      [](const exec::Block&, int&&) {});
}

struct alignas(kCacheLine) RotateScratch {
  AlignedArray<float> vectors;       // count * dim: centred, normalised rows of the block
  AlignedArray<std::uint64_t> bits;  // count * words
  AlignedArray<double> l1;           // sum_j |o'_j|
  AlignedArray<double> distance;     // ||x - c||
};

// Blocked rotation: the vectors of a block (block_bytes of floats) stay in the
// cache of the worker while the rows of the rotation stream past them; each
// row is used for every vector of the block before the next row is fetched.
// Only the sign and the magnitude of each rotated coordinate are kept.
void encode_rabitq(Impl& s, exec::OrderedExecutor& executor) {
  const std::size_t dim = s.dim;
  const std::size_t words = s.words;
  const Kernels& k = kernels();
  const float* rotation = s.rotation->data();
  const auto blocks = exec::partition_fixed(s.n, dim * sizeof(float), s.options.block_bytes);
  std::size_t per_block = 0;
  for (const exec::Block& b : blocks) per_block = std::max(per_block, b.count);

  std::vector<RotateScratch> scratch(executor.threads());
  for (RotateScratch& w : scratch) {
    w.vectors.reset(per_block * dim);
    w.bits.reset(per_block * words);
    w.l1.reset(per_block);
    w.distance.reset(per_block);
  }
  const double inverse_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(dim));

  executor.run(
      blocks,
      [&](const exec::Block& b, unsigned worker) {
        RotateScratch& w = scratch[worker];
        for (std::size_t i = 0; i < b.count; ++i) {
          const float* x = s.rows.data() + static_cast<std::size_t>(s.layout[b.first + i]) * dim;
          float* o = w.vectors.data() + i * dim;
          double norm2 = 0.0;
          for (std::size_t j = 0; j < dim; ++j) {
            const float diff = x[j] - s.centroid[j];
            o[j] = diff;
            norm2 += static_cast<double>(diff) * static_cast<double>(diff);
          }
          const double distance = std::sqrt(norm2);
          if (distance > 0.0) {
            for (std::size_t j = 0; j < dim; ++j) o[j] = static_cast<float>(static_cast<double>(o[j]) / distance);
          }
          w.distance[i] = distance;
          w.l1[i] = 0.0;
        }
        std::fill(w.bits.data(), w.bits.data() + b.count * words, std::uint64_t{0});

        for (std::size_t j = 0; j < dim; ++j) {
          const float* row = rotation + j * dim;
          const std::size_t word = j / 64;
          const std::uint64_t bit = std::uint64_t{1} << (j % 64);
          for (std::size_t i = 0; i < b.count; ++i) {
            const float t = k.dot_f32(w.vectors.data() + i * dim, row, dim);
            w.l1[i] += std::fabs(static_cast<double>(t));
            if (t >= 0.0f) w.bits[i * words + word] |= bit;
          }
        }

        for (std::size_t i = 0; i < b.count; ++i) {
          const Slot slot = locate(s, b.first + i);
          float* side = s.side(slot.block) + kSideValues * slot.index;
          std::uint8_t* code = s.codes(slot.block) + slot.index * s.code_bytes;
          if (w.distance[i] > 0.0) {
            // <obar, o> = ||P^T o||_1 / sqrt(D), at least 1 / sqrt(D) for a unit vector.
            const double alignment = std::max(w.l1[i] * inverse_sqrt_dim, 1e-6);
            side[0] = static_cast<float>(w.distance[i]);
            side[1] = static_cast<float>(alignment);
            std::memcpy(code, w.bits.data() + i * words, s.code_bytes);
          } else {
            // The row is the centroid: no direction to encode. With distance
            // zero the estimate reduces to the exact value and the half-width
            // to the rounding slack; the alignment only has to be non-zero.
            side[0] = 0.0f;
            side[1] = 1.0f;
            std::memset(code, 0, s.code_bytes);
          }
        }
        return 0;
      },
      [](const exec::Block&, int&&) {});
}

std::unique_ptr<Impl> build_store(std::span<const float> rows, std::size_t dim,
                                  std::span<const std::uint32_t> layout,
                                  std::span<const std::uint32_t> group_of_unit, const BuildOptions& options,
                                  exec::OrderedExecutor& executor) {
  if (dim == 0 || rows.size() % dim != 0) {
    PRAG_ERROR("index: %zu floats are not a whole number of rows of dimension %zu", rows.size(), dim);
    return nullptr;
  }
  const std::size_t n = rows.size() / dim;
  if (n >= static_cast<std::size_t>(kNoUnit)) {
    PRAG_ERROR("index: %zu rows exceed the range of unit ids", n);
    return nullptr;
  }
  if (!layout.empty() && !is_permutation_of_range(layout, n)) {
    PRAG_ERROR("index: the layout is not a permutation of the %zu units", n);
    return nullptr;
  }
  if (!group_of_unit.empty() && group_of_unit.size() != n) {
    PRAG_ERROR("index: %zu groups were given for %zu units", group_of_unit.size(), n);
    return nullptr;
  }
  if (options.codes != CodeKind::Int8 && options.codes != CodeKind::RaBitQ) {
    PRAG_ERROR("index: unknown code kind %d", static_cast<int>(options.codes));
    return nullptr;
  }
  if (!(options.clip_quantile > 0.0 && options.clip_quantile <= 1.0) || !(options.confidence_sigma >= 0.0) ||
      !std::isfinite(options.confidence_sigma)) {
    PRAG_ERROR("index: clip_quantile must lie in (0, 1] and confidence_sigma must be finite and not negative");
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  Impl& s = *impl;
  s.options = options;
  s.options.block_bytes = resolve_block_bytes(options.block_bytes);
  s.n = n;
  s.dim = dim;
  set_code_geometry(s);

  s.rows.reset(n * dim);
  if (n != 0) std::memcpy(s.rows.data(), rows.data(), n * dim * sizeof(float));
  s.layout.resize(n);
  if (layout.empty()) {
    for (std::size_t p = 0; p < n; ++p) s.layout[p] = static_cast<std::uint32_t>(p);
  } else {
    std::copy(layout.begin(), layout.end(), s.layout.begin());
  }
  s.groups.assign(group_of_unit.begin(), group_of_unit.end());

  const RowSummary summary = n != 0 ? summarise_rows(s, executor) : RowSummary{};
  if (!summary.finite) {
    PRAG_ERROR("index: the rows contain values that are not finite");
    return nullptr;
  }
  if (n == 0) s.centroid.assign(dim, 0.0f);
  s.norm_slack = 0.5 * summary.max_dev;
  finish_derived(s);
  layout_blocks(s);
  if (n == 0) return impl;

  if (s.options.codes == CodeKind::Int8) {
    double scale = s.options.clip_quantile >= 1.0 ? summary.max_abs : component_quantile(s, s.options.clip_quantile);
    if (!(scale > 0.0)) scale = summary.max_abs;  // the quantile fell on a zero component
    if (!(scale > 0.0)) scale = 1.0;              // every row is the zero vector
    s.scale = scale;
    encode_int8(s, executor);
  } else {
    s.rotation = make_rotation(dim, s.options.seed, executor);
    encode_rabitq(s, executor);
  }
  return impl;
}

}  // namespace

// ---- VectorStore ---------------------------------------------------------------

VectorStore::VectorStore() : impl_(empty_impl()) {}

VectorStore::~VectorStore() {
  if (impl_ != empty_impl()) delete impl_;
}

VectorStore::VectorStore(VectorStore&& other) noexcept : impl_(std::exchange(other.impl_, empty_impl())) {}

VectorStore& VectorStore::operator=(VectorStore&& other) noexcept {
  if (this != &other) {
    if (impl_ != empty_impl()) delete impl_;
    impl_ = std::exchange(other.impl_, empty_impl());
  }
  return *this;
}

void VectorStore::build(std::span<const float> rows, std::size_t dim, std::span<const std::uint32_t> layout,
                        std::span<const std::uint32_t> group_of_unit, const BuildOptions& options) {
  build(rows, dim, layout, group_of_unit, options, exec::OrderedExecutor::shared());
}

void VectorStore::build(std::span<const float> rows, std::size_t dim, std::span<const std::uint32_t> layout,
                        std::span<const std::uint32_t> group_of_unit, const BuildOptions& options,
                        exec::OrderedExecutor& executor) {
  std::unique_ptr<Impl> fresh = build_store(rows, dim, layout, group_of_unit, options, executor);
  if (impl_ != empty_impl()) delete impl_;
  impl_ = fresh ? fresh.release() : empty_impl();
}

void VectorStore::reblock(std::size_t block_bytes) {
  if (impl_ == empty_impl()) return;
  Impl& s = *impl_;
  const std::vector<exec::Block> old_blocks = std::move(s.blocks);
  const std::vector<std::size_t> old_offsets = std::move(s.offsets);
  const AlignedArray<std::uint8_t> old_arena = std::move(s.arena);
  s.options.block_bytes = resolve_block_bytes(block_bytes);
  layout_blocks(s);

  // Both partitions list the storage positions in order: walk them together.
  std::size_t from = 0;
  std::size_t from_index = 0;
  for (std::size_t to = 0; to < s.blocks.size(); ++to) {
    for (std::size_t index = 0; index < s.blocks[to].count; ++index) {
      while (from_index == old_blocks[from].count) {
        ++from;
        from_index = 0;
      }
      const std::uint8_t* source = old_arena.data() + old_offsets[from];
      std::memcpy(s.side(to) + kSideValues * index, source + from_index * kSideBytes, kSideBytes);
      std::memcpy(s.codes(to) + index * s.code_bytes,
                  source + old_blocks[from].count * kSideBytes + from_index * s.code_bytes, s.code_bytes);
      ++from_index;
    }
  }
}

std::size_t VectorStore::size() const noexcept { return impl_->n; }
std::size_t VectorStore::dim() const noexcept { return impl_->dim; }
CodeKind VectorStore::code_kind() const noexcept { return impl_->options.codes; }
std::size_t VectorStore::block_bytes() const noexcept { return impl_->options.block_bytes; }
std::size_t VectorStore::bytes_per_code() const noexcept { return impl_->dim == 0 ? 0 : impl_->record_bytes(); }
const BuildOptions& VectorStore::options() const noexcept { return impl_->options; }

std::span<const float> VectorStore::vector(UnitId unit) const {
  if (unit >= impl_->n) return {};
  return {impl_->rows.data() + static_cast<std::size_t>(unit) * impl_->dim, impl_->dim};
}

std::span<const float> VectorStore::centroid() const { return impl_->centroid; }

std::span<const exec::Block> VectorStore::blocks() const noexcept { return impl_->blocks; }

UnitId VectorStore::unit_at(std::size_t storage_position) const {
  return storage_position < impl_->n ? impl_->layout[storage_position] : kNoUnit;
}

std::size_t VectorStore::storage_position(UnitId unit) const {
  return unit < impl_->n ? impl_->position[unit] : impl_->n;
}

std::span<const std::uint8_t> VectorStore::block_payload(std::size_t block) const {
  if (block >= impl_->blocks.size()) return {};
  return {impl_->payload(block), impl_->blocks[block].bytes};
}

std::uint32_t VectorStore::group_of_block(std::size_t block) const {
  if (block >= impl_->blocks.size()) return 0;
  return impl_->group_of(impl_->layout[impl_->blocks[block].first]);
}

}  // namespace prag::index
