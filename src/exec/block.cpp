#include "prag/exec/block.hpp"

#include <algorithm>

namespace prag::exec {

std::vector<Block> partition_fixed(std::size_t n_items, std::size_t item_bytes, std::size_t block_bytes) {
  std::vector<Block> blocks;
  if (n_items == 0) return blocks;
  const std::size_t per_block = item_bytes == 0 ? n_items : std::max<std::size_t>(1, block_bytes / item_bytes);
  blocks.reserve((n_items + per_block - 1) / per_block);
  for (std::size_t first = 0; first < n_items; first += per_block) {
    const std::size_t count = std::min(per_block, n_items - first);
    blocks.push_back(Block{blocks.size(), first, count, count * item_bytes});
  }
  return blocks;
}

std::vector<Block> partition_by_bytes(std::span<const std::uint32_t> item_bytes, std::size_t block_bytes) {
  std::vector<Block> blocks;
  std::size_t first = 0;
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < item_bytes.size(); ++i) {
    const std::size_t size = item_bytes[i];
    if (i > first && bytes + size > block_bytes) {
      blocks.push_back(Block{blocks.size(), first, i - first, bytes});
      first = i;
      bytes = 0;
    }
    bytes += size;
  }
  if (first < item_bytes.size()) {
    blocks.push_back(Block{blocks.size(), first, item_bytes.size() - first, bytes});
  }
  return blocks;
}

std::vector<Block> partition_text(std::string_view text, std::size_t block_bytes) {
  std::vector<Block> blocks;
  if (text.empty()) return blocks;
  if (block_bytes == 0) block_bytes = 1;
  std::size_t first = 0;
  while (first < text.size()) {
    std::size_t end = std::min(text.size(), first + block_bytes);
    if (end < text.size()) {
      // Prefer the last newline inside the window so that lines stay whole.
      const std::size_t newline = text.rfind('\n', end - 1);
      if (newline != std::string_view::npos && newline >= first) {
        end = newline + 1;
      } else {
        // One line longer than a block: cut, but never inside a UTF-8 sequence.
        std::size_t cut = end;
        while (cut > first + 1 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
        end = cut;
      }
    }
    blocks.push_back(Block{blocks.size(), first, end - first, end - first});
    first = end;
  }
  return blocks;
}

}  // namespace prag::exec
