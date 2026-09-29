// Partitioning of work into cache-sized blocks.
//
// A block is a contiguous run of items whose total payload does not exceed the
// block size, so that one worker can hold the whole block in its L1 data cache.
// Blocks carry consecutive sequence numbers starting at 0; the sequence number
// defines the order in which blocks are admitted and committed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace prag::exec {

struct Block {
  std::uint64_t seq = 0;   // position in the total order
  std::size_t first = 0;   // index of the first item (or first byte for text blocks)
  std::size_t count = 0;   // number of items (or bytes for text blocks)
  std::size_t bytes = 0;   // payload size in bytes
};

// Fixed-size items: floor(block_bytes / item_bytes) items per block, at least one.
std::vector<Block> partition_fixed(std::size_t n_items, std::size_t item_bytes, std::size_t block_bytes);

// Variable-size items: greedy packing in item order. An item larger than
// block_bytes forms a block of its own.
std::vector<Block> partition_by_bytes(std::span<const std::uint32_t> item_bytes, std::size_t block_bytes);

// Text: blocks are byte ranges that end at a newline whenever a newline occurs
// within block_bytes; a single line longer than block_bytes is cut at block_bytes,
// never inside a UTF-8 sequence. Concatenating the blocks reproduces `text`.
std::vector<Block> partition_text(std::string_view text, std::size_t block_bytes);

}  // namespace prag::exec
