// Ordered, cache-blocked parallel execution.
//
// OrderedExecutor runs `work` on every block using a fixed pool of worker
// threads and delivers the results to `commit`, with these guarantees:
//
//   (G1) Admission order. Block i is admitted before block i+1 for every i.
//        Admission is the moment a worker, running on a CPU, takes ownership of
//        the block and performs its first read of the block payload (the
//        `admit` callback). Admissions are serialised by a FIFO ticket gate, so
//        they form a total order identical to the sequence order.
//   (G2) Commit order. commit(i) completes before commit(i+1) starts, and two
//        commits never overlap, whatever the number of threads.
//   (G3) Exactly once. work and commit run exactly once per block unless work
//        throws; the first exception stops admission of further blocks, already
//        admitted blocks finish, no block after the failed one is committed,
//        and the exception is rethrown from run().
//   (G4) Bounded reordering. Block i is not admitted until block i - window has
//        been committed, so at most `window` results are buffered.
//
// Because commits happen in sequence order, any reduction performed in `commit`
// (floating-point sums, top-k merges with index tie-breaks) produces bit-identical
// results for every thread count and every scheduling.
//
// What is not guaranteed: the operating system may preempt a worker between
// admission and the rest of its work. The order of admissions and commits is
// fixed; the interleaving of the work bodies is not, and need not be.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "prag/exec/block.hpp"

namespace prag::exec {

struct BlockTrace {
  std::uint64_t seq = 0;
  std::uint32_t worker = 0;
  std::uint64_t admit_tick = 0;   // value of the global logical clock at admission
  std::uint64_t commit_tick = 0;  // value of the global logical clock at commit
  std::uint64_t admit_ns = 0;     // monotonic time stamps relative to the start of run()
  std::uint64_t done_ns = 0;
  std::uint64_t commit_ns = 0;
};

struct RunStats {
  std::size_t blocks = 0;
  unsigned threads = 0;
  std::uint64_t wall_ns = 0;
  std::uint64_t work_ns = 0;        // summed over workers
  std::uint64_t gate_wait_ns = 0;   // time workers spent waiting at the admission gate
  std::uint64_t admission_violations = 0;  // blocks whose admit_tick order disagrees with seq order
  std::uint64_t commit_violations = 0;     // blocks whose commit_tick order disagrees with seq order
  std::size_t max_buffered = 0;     // high-water mark of the reorder buffer
  std::vector<BlockTrace> trace;    // filled only when tracing is enabled
};

struct RunOptions {
  bool trace = false;       // record a BlockTrace per block
  std::size_t window = 0;   // reorder window; 0 = 4 * threads
};

namespace detail {

// Type-erased engine behind OrderedExecutor::run.
class Engine {
 public:
  using Task = std::function<void(const Block&, unsigned worker)>;
  using Finish = std::function<void(const Block&)>;
  struct State;

  explicit Engine(unsigned threads);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  unsigned threads() const noexcept { return threads_; }

  // `admit` and `work` run on a worker; `commit` runs on whichever worker
  // completes the in-order prefix, under the commit lock.
  RunStats run(std::span<const Block> blocks, const Task& admit, const Task& work, const Finish& commit,
               const RunOptions& options);

 private:
  void worker_loop(unsigned worker);

  unsigned threads_;
  std::unique_ptr<State> state_;
  std::vector<std::thread> pool_;
};

}  // namespace detail

class OrderedExecutor {
 public:
  // threads == 0 selects std::thread::hardware_concurrency().
  explicit OrderedExecutor(unsigned threads = 0) : engine_(resolve(threads)) {}

  unsigned threads() const noexcept { return engine_.threads(); }

  // work:   R(const Block&, unsigned worker)      may run concurrently
  // commit: void(const Block&, R&&)               runs in sequence order, never concurrently
  // admit:  void(const Block&, unsigned worker)   optional first touch, runs inside the gate
  template <class Work, class Commit>
  RunStats run(std::span<const Block> blocks, Work&& work, Commit&& commit, const RunOptions& options = {}) {
    return run_with_admit(blocks, [](const Block&, unsigned) {}, std::forward<Work>(work),
                          std::forward<Commit>(commit), options);
  }

  template <class Admit, class Work, class Commit>
  RunStats run_with_admit(std::span<const Block> blocks, Admit&& admit, Work&& work, Commit&& commit,
                          const RunOptions& options = {}) {
    using R = std::invoke_result_t<Work&, const Block&, unsigned>;
    static_assert(!std::is_void_v<R>, "work must return a value; return a dummy int when there is no result");
    std::vector<std::optional<R>> slots(blocks.size());
    const Block* base = blocks.data();
    return engine_.run(
        blocks,
        [&](const Block& b, unsigned worker) { admit(b, worker); },
        [&](const Block& b, unsigned worker) { slots[static_cast<std::size_t>(&b - base)].emplace(work(b, worker)); },
        [&](const Block& b) {
          auto& slot = slots[static_cast<std::size_t>(&b - base)];
          commit(b, std::move(*slot));
          slot.reset();
        },
        options);
  }

  // Process-wide executor sized to the machine.
  static OrderedExecutor& shared();

 private:
  static unsigned resolve(unsigned threads) {
    if (threads != 0) return threads;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
  }

  detail::Engine engine_;
};

}  // namespace prag::exec
