// Cache blocking shared by the kernel sums of the kde module (private header).
//
// A pairwise computation between `items` (queries or mean-shift seeds) and the
// data rows is cut in two directions. The data are cut into tiles of at most
// half a block; the items into batches whose state fits in the other half. A
// worker owns one batch at a time and sweeps the tiles with it, so that its
// working set is one tile plus one batch: one block.
//
// Batches are the blocks handed to the ordered executor. Inside a batch every
// item meets the data rows in index order, tile after tile, which makes the
// result of an item independent of the tiling and of the batch it belongs to.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "prag/exec/block.hpp"
#include "prag/exec/cache_topology.hpp"

namespace prag::kde::detail {

struct Layout {
  std::vector<exec::Block> tiles;    // over data rows
  std::vector<exec::Block> batches;  // over items
  std::size_t max_batch = 0;         // largest number of items in a batch
};

inline std::size_t resolve_block_bytes(std::size_t block_bytes) {
  return block_bytes != 0 ? block_bytes : exec::recommended_block_bytes(exec::cache_topology());
}

inline Layout make_layout(std::size_t data_rows, std::size_t row_bytes, std::size_t items, std::size_t item_bytes,
                          std::size_t block_bytes) {
  const std::size_t block = resolve_block_bytes(block_bytes);
  const std::size_t tile_bytes = block / 2;
  Layout layout;
  layout.tiles = exec::partition_fixed(data_rows, row_bytes, tile_bytes);
  layout.batches = exec::partition_fixed(items, item_bytes, block - tile_bytes);
  for (const exec::Block& b : layout.batches) layout.max_batch = std::max(layout.max_batch, b.count);
  return layout;
}

}  // namespace prag::kde::detail
