#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "prag/exec/block.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/exec/ordered_executor.hpp"

using namespace prag::exec;

PRAG_TEST(topology_is_sane) {
  const CacheTopology& t = cache_topology();
  CHECK(t.logical_cpus >= 1u);
  CHECK(t.line_bytes >= 16u);
  CHECK(t.l1d_min_bytes >= 4u * 1024u);
  CHECK(t.l1d_per_thread_min_bytes <= t.l1d_min_bytes);
  const std::size_t block = recommended_block_bytes(t);
  CHECK(block >= 4u * 1024u);
  CHECK(block <= 64u * 1024u);
  CHECK_EQ(block % 1024u, 0u);
  CHECK(block <= t.l1d_per_thread_min_bytes);
  CHECK(!describe(t).empty());
}

PRAG_TEST(block_rule_reproduces_the_reference_machine) {
  // 8 P-cores: 48 KiB L1d shared by 2 hyper-threads; 8 E-cores: 32 KiB unshared.
  CacheTopology t;
  t.l1d_min_bytes = 32 * 1024;
  t.l1d_per_thread_min_bytes = 24 * 1024;
  CHECK_EQ(recommended_block_bytes(t), std::size_t{16 * 1024});
  // With the whole L1d allowed, the hyper-thread share (24 KiB) becomes the bound.
  CHECK_EQ(recommended_block_bytes(t, 1.0), std::size_t{24 * 1024});
  // A 32 KiB L1d shared by two hyper-threads: 16 KiB either way.
  t.l1d_per_thread_min_bytes = 16 * 1024;
  CHECK_EQ(recommended_block_bytes(t), std::size_t{16 * 1024});
  t.l1d_min_bytes = 2 * 1024;
  t.l1d_per_thread_min_bytes = 1024;
  CHECK_EQ(recommended_block_bytes(t), std::size_t{4 * 1024});
  t.l1d_min_bytes = 1024 * 1024;
  t.l1d_per_thread_min_bytes = 1024 * 1024;
  CHECK_EQ(recommended_block_bytes(t), std::size_t{64 * 1024});
}

PRAG_TEST(partition_fixed_covers_every_item_once) {
  const auto blocks = partition_fixed(1000, 256, 16 * 1024);
  REQUIRE(!blocks.empty());
  std::size_t next = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    CHECK_EQ(blocks[i].seq, static_cast<std::uint64_t>(i));
    CHECK_EQ(blocks[i].first, next);
    CHECK(blocks[i].bytes <= 16u * 1024u);
    CHECK_EQ(blocks[i].bytes, blocks[i].count * 256u);
    next += blocks[i].count;
  }
  CHECK_EQ(next, std::size_t{1000});
  CHECK_EQ(blocks.front().count, std::size_t{64});
  CHECK(partition_fixed(0, 256, 16 * 1024).empty());
  // An item larger than the block still gets a block of its own.
  CHECK_EQ(partition_fixed(3, 100000, 16 * 1024).size(), std::size_t{3});
}

PRAG_TEST(partition_by_bytes_packs_greedily) {
  const std::vector<std::uint32_t> sizes = {6000, 6000, 6000, 20000, 100, 100, 16384, 1};
  const auto blocks = partition_by_bytes(sizes, 16384);
  REQUIRE(blocks.size() == 6u);
  CHECK_EQ(blocks[0].count, std::size_t{2});   // 12000
  CHECK_EQ(blocks[1].count, std::size_t{1});   // 6000 (next item does not fit)
  CHECK_EQ(blocks[2].count, std::size_t{1});   // 20000 oversized, alone
  CHECK_EQ(blocks[3].count, std::size_t{2});   // 200
  CHECK_EQ(blocks[4].count, std::size_t{1});   // 16384 exactly
  CHECK_EQ(blocks[5].count, std::size_t{1});   // 1
  std::size_t total = 0;
  for (const Block& b : blocks) total += b.count;
  CHECK_EQ(total, sizes.size());
}

PRAG_TEST(partition_text_is_lossless_and_line_aligned) {
  std::string text;
  std::mt19937 rng(7);
  for (int line = 0; line < 4000; ++line) {
    text.append(static_cast<std::size_t>(rng() % 120), 'a' + static_cast<char>(line % 26));
    text.push_back('\n');
  }
  text += "tail without newline";
  const auto blocks = partition_text(text, 16 * 1024);
  REQUIRE(blocks.size() > 1u);
  std::string rebuilt;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    CHECK(blocks[i].count <= 16u * 1024u);
    CHECK(blocks[i].count > 0u);
    const std::string_view piece = std::string_view(text).substr(blocks[i].first, blocks[i].count);
    if (i + 1 < blocks.size()) CHECK(piece.back() == '\n');
    rebuilt.append(piece);
  }
  CHECK(rebuilt == text);
}

PRAG_TEST(partition_text_never_splits_utf8) {
  std::string text;
  for (int i = 0; i < 3000; ++i) text += "\xE2\x82\xAC";  // one 9000-byte line of euro signs
  const auto blocks = partition_text(text, 1000);
  std::string rebuilt;
  for (const Block& b : blocks) {
    const std::string_view piece = std::string_view(text).substr(b.first, b.count);
    CHECK_EQ(piece.size() % 3u, 0u);
    CHECK((static_cast<unsigned char>(piece.front()) & 0xC0) != 0x80);
    rebuilt.append(piece);
  }
  CHECK(rebuilt == text);
}

PRAG_TEST(commits_arrive_in_sequence_order) {
  for (unsigned threads : {1u, 2u, 3u, 8u, 24u}) {
    OrderedExecutor executor(threads);
    const auto blocks = partition_fixed(5000, 64, 4096);
    std::vector<std::uint64_t> order;
    std::atomic<int> in_commit{0};
    int overlaps = 0;
    RunOptions options;
    options.trace = true;
    const RunStats stats = executor.run(
        blocks,
        [](const Block& b, unsigned) {
          // Uneven work so that later blocks often finish first.
          volatile std::uint64_t sink = 0;
          const std::uint64_t spin = (b.seq % 7 == 0) ? 40000 : 200;
          for (std::uint64_t k = 0; k < spin; ++k) sink = sink + k;
          return b.seq * 3;
        },
        [&](const Block& b, std::uint64_t&& value) {
          if (in_commit.fetch_add(1) != 0) ++overlaps;
          CHECK_EQ(value, b.seq * 3);
          order.push_back(b.seq);
          in_commit.fetch_sub(1);
        },
        options);
    REQUIRE(order.size() == blocks.size());
    for (std::size_t i = 0; i < order.size(); ++i) CHECK_EQ(order[i], static_cast<std::uint64_t>(i));
    CHECK_EQ(overlaps, 0);
    CHECK_EQ(stats.admission_violations, std::uint64_t{0});
    CHECK_EQ(stats.commit_violations, std::uint64_t{0});
    CHECK_EQ(stats.trace.size(), blocks.size());
    CHECK(stats.max_buffered <= static_cast<std::size_t>(threads) * 4);
  }
}

PRAG_TEST(admissions_are_totally_ordered) {
  OrderedExecutor executor(16);
  const auto blocks = partition_fixed(20000, 16, 256);
  std::atomic<std::uint64_t> last_admitted{0};
  std::atomic<std::uint64_t> violations{0};
  std::atomic<int> inside_gate{0};
  const RunStats stats = executor.run_with_admit(
      blocks,
      [&](const Block& b, unsigned) {
        // The admit callback runs inside the gate: never concurrently, always in order.
        if (inside_gate.fetch_add(1) != 0) violations.fetch_add(1);
        const std::uint64_t expected = b.seq;
        if (last_admitted.load() != expected) violations.fetch_add(1);
        last_admitted.store(expected + 1);
        inside_gate.fetch_sub(1);
      },
      [](const Block& b, unsigned) { return static_cast<int>(b.count); },
      [](const Block&, int&&) {});
  CHECK_EQ(violations.load(), std::uint64_t{0});
  CHECK_EQ(stats.admission_violations, std::uint64_t{0});
  CHECK_EQ(last_admitted.load(), static_cast<std::uint64_t>(blocks.size()));
}

PRAG_TEST(floating_point_reduction_is_bit_identical_across_thread_counts) {
  std::vector<float> data(200000);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1e3f, 1e3f);
  for (float& v : data) v = dist(rng) * (rng() % 1000 == 0 ? 1e6f : 1.0f);
  const auto blocks = partition_fixed(data.size(), sizeof(float), 16 * 1024);

  const auto reduce = [&](unsigned threads) {
    OrderedExecutor executor(threads);
    float total = 0.0f;
    executor.run(
        blocks,
        [&](const Block& b, unsigned) {
          float partial = 0.0f;
          for (std::size_t i = 0; i < b.count; ++i) partial += data[b.first + i];
          return partial;
        },
        [&](const Block&, float&& partial) { total += partial; });
    std::uint32_t bits = 0;
    std::memcpy(&bits, &total, sizeof(bits));
    return bits;
  };

  const std::uint32_t reference = reduce(1);
  for (unsigned threads : {2u, 5u, 16u, 24u}) {
    for (int repeat = 0; repeat < 5; ++repeat) CHECK_EQ(reduce(threads), reference);
  }
}

PRAG_TEST(exception_stops_the_run_and_nothing_after_it_commits) {
  OrderedExecutor executor(8);
  const auto blocks = partition_fixed(400, 1, 1);
  std::vector<std::uint64_t> committed;
  bool thrown = false;
  try {
    executor.run(
        blocks,
        [](const Block& b, unsigned) {
          if (b.seq == 137) throw std::runtime_error("boom");
          return 0;
        },
        [&](const Block& b, int&&) { committed.push_back(b.seq); });
  } catch (const std::runtime_error& e) {
    thrown = std::string(e.what()) == "boom";
  }
  CHECK(thrown);
  CHECK(committed.size() <= 137u);
  for (std::size_t i = 0; i < committed.size(); ++i) CHECK_EQ(committed[i], static_cast<std::uint64_t>(i));

  // The executor is reusable after a failure.
  std::size_t count = 0;
  executor.run(blocks, [](const Block&, unsigned) { return 1; }, [&](const Block&, int&& v) { count += v; });
  CHECK_EQ(count, blocks.size());
}

PRAG_TEST(window_bounds_the_reorder_buffer) {
  OrderedExecutor executor(12);
  const auto blocks = partition_fixed(3000, 1, 1);
  RunOptions options;
  options.window = 5;
  const RunStats stats = executor.run(
      blocks,
      [](const Block& b, unsigned) {
        if (b.seq % 50 == 0) std::this_thread::sleep_for(std::chrono::microseconds(300));
        return 0;
      },
      [](const Block&, int&&) {},
      options);
  CHECK(stats.max_buffered <= 5u);
  CHECK_EQ(stats.commit_violations, std::uint64_t{0});
}

PRAG_TEST(empty_and_single_block_runs) {
  OrderedExecutor executor(4);
  int commits = 0;
  const RunStats none = executor.run(
      std::span<const Block>{}, [](const Block&, unsigned) { return 0; }, [&](const Block&, int&&) { ++commits; });
  CHECK_EQ(none.blocks, std::size_t{0});
  CHECK_EQ(commits, 0);
  const auto one = partition_fixed(1, 8, 64);
  executor.run(one, [](const Block&, unsigned) { return 5; }, [&](const Block&, int&& v) { commits += v; });
  CHECK_EQ(commits, 5);
}

PRAG_TEST(many_short_runs_do_not_lose_or_duplicate_work) {
  OrderedExecutor executor(8);
  for (int round = 0; round < 300; ++round) {
    const auto blocks = partition_fixed(static_cast<std::size_t>(1 + round % 17), 1, 1);
    std::uint64_t sum = 0;
    executor.run(
        blocks, [](const Block& b, unsigned) { return b.seq + 1; },
        [&](const Block&, std::uint64_t&& v) { sum += v; });
    const std::uint64_t n = blocks.size();
    CHECK_EQ(sum, n * (n + 1) / 2);
  }
}
