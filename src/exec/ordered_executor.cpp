#include "prag/exec/ordered_executor.hpp"

#include <algorithm>
#include <new>

#include "prag/core/clock.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define PRAG_CPU_RELAX() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define PRAG_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define PRAG_CPU_RELAX() __asm__ __volatile__("yield")
#else
#define PRAG_CPU_RELAX() ((void)0)
#endif

namespace prag::exec {

namespace detail {

namespace {

constexpr std::size_t kCacheLine = 64;

struct alignas(kCacheLine) PaddedCounter {
  std::atomic<std::uint64_t> value{0};
};

struct alignas(kCacheLine) WorkerStats {
  std::uint64_t work_ns = 0;
  std::uint64_t gate_wait_ns = 0;
};

// Per-block bookkeeping. One entry is written by exactly one worker (the owner
// of the block) and then by the committer, never concurrently.
struct BlockRecord {
  std::atomic<std::uint8_t> done{0};
  std::uint32_t worker = 0;
  std::uint64_t admit_tick = 0;
  std::uint64_t commit_tick = 0;
  std::uint64_t admit_ns = 0;
  std::uint64_t done_ns = 0;
  std::uint64_t commit_ns = 0;
};

}  // namespace

struct Engine::State {
  // Pool control.
  std::mutex run_mutex;  // serialises run() calls
  std::mutex mutex;      // protects everything below up to `active`
  std::condition_variable wake;
  std::condition_variable idle;
  bool shutdown = false;
  bool job_open = false;
  std::uint64_t generation = 0;
  unsigned wanted = 0;  // pool workers that should join the current job
  unsigned joined = 0;
  unsigned active = 0;

  // Current job (valid while job_open or active != 0).
  const Block* blocks = nullptr;
  std::size_t n = 0;
  std::size_t window = 0;
  const Task* admit = nullptr;
  const Task* work = nullptr;
  const Finish* commit = nullptr;
  std::uint64_t start_ns = 0;
  std::unique_ptr<BlockRecord[]> records;
  std::vector<WorkerStats> worker_stats;

  PaddedCounter next_ticket;  // next sequence number to hand out
  PaddedCounter admitted;     // number of blocks admitted so far
  PaddedCounter committed;    // number of blocks committed so far
  PaddedCounter clock;        // logical clock shared by admissions and commits
  PaddedCounter buffered_max;
  std::atomic<bool> failed{false};

  std::mutex commit_mutex;
  std::mutex error_mutex;
  std::exception_ptr error;
  std::uint64_t error_seq = 0;
};

Engine::Engine(unsigned threads) : threads_(std::max(1u, threads)), state_(std::make_unique<State>()) {
  pool_.reserve(threads_ - 1);
  // The calling thread is worker 0; the pool supplies workers 1 .. threads-1.
  for (unsigned w = 1; w < threads_; ++w) {
    pool_.emplace_back([this, w] { worker_loop(w); });
  }
}

Engine::~Engine() {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->shutdown = true;
  }
  state_->wake.notify_all();
  for (std::thread& t : pool_) {
    if (t.joinable()) t.join();
  }
}

namespace {

// Runs blocks until the tickets are exhausted or the job has failed.
void drive(Engine::State& s, unsigned worker);

}  // namespace

void Engine::worker_loop(unsigned worker) {
  State& s = *state_;
  std::uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(s.mutex);
      s.wake.wait(lock, [&] { return s.shutdown || (s.generation != seen && s.job_open && s.joined < s.wanted); });
      if (s.shutdown) return;
      seen = s.generation;
      ++s.joined;
      ++s.active;
    }
    drive(s, worker);
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      --s.active;
    }
    s.idle.notify_all();
  }
}

namespace {

void drain_commits(Engine::State& s) {
  std::lock_guard<std::mutex> lock(s.commit_mutex);
  for (;;) {
    const std::uint64_t c = s.committed.value.load(std::memory_order_relaxed);
    if (c >= s.n) return;
    BlockRecord& record = s.records[c];
    if (record.done.load(std::memory_order_acquire) == 0) return;
    // A failed block never sets `done`, so nothing at or after it is committed.
    try {
      (*s.commit)(s.blocks[c]);
    } catch (...) {
      {
        std::lock_guard<std::mutex> error_lock(s.error_mutex);
        if (!s.error || c < s.error_seq) {
          s.error = std::current_exception();
          s.error_seq = c;
        }
      }
      s.failed.store(true, std::memory_order_release);
      return;
    }
    record.commit_tick = s.clock.value.fetch_add(1, std::memory_order_relaxed);
    record.commit_ns = now_ns() - s.start_ns;
    s.committed.value.store(c + 1, std::memory_order_release);
  }
}

void drive(Engine::State& s, unsigned worker) {
  WorkerStats& stats = s.worker_stats[worker];
  for (;;) {
    if (s.failed.load(std::memory_order_acquire)) return;
    const std::uint64_t i = s.next_ticket.value.fetch_add(1, std::memory_order_relaxed);
    if (i >= s.n) return;

    // ---- admission gate (FIFO by ticket) ------------------------------------
    const std::uint64_t gate_start = now_ns();
    unsigned spins = 0;
    for (;;) {
      if (s.failed.load(std::memory_order_acquire)) return;
      const bool my_turn = s.admitted.value.load(std::memory_order_acquire) == i;
      if (my_turn && i < s.committed.value.load(std::memory_order_acquire) + s.window) break;
      if (++spins < 64) {
        PRAG_CPU_RELAX();
      } else {
        std::this_thread::yield();
      }
    }
    BlockRecord& record = s.records[i];
    record.worker = worker;
    record.admit_tick = s.clock.value.fetch_add(1, std::memory_order_relaxed);
    record.admit_ns = now_ns() - s.start_ns;
    stats.gate_wait_ns += record.admit_ns + s.start_ns - gate_start;

    bool ok = true;
    try {
      (*s.admit)(s.blocks[i], worker);  // first touch of the payload, inside the gate
    } catch (...) {
      ok = false;
      std::lock_guard<std::mutex> error_lock(s.error_mutex);
      if (!s.error || i < s.error_seq) {
        s.error = std::current_exception();
        s.error_seq = i;
      }
    }
    if (!ok) {
      s.failed.store(true, std::memory_order_release);
      return;
    }
    const std::uint64_t in_flight = i + 1 - s.committed.value.load(std::memory_order_relaxed);
    std::uint64_t seen_max = s.buffered_max.value.load(std::memory_order_relaxed);
    while (in_flight > seen_max &&
           !s.buffered_max.value.compare_exchange_weak(seen_max, in_flight, std::memory_order_relaxed)) {
    }
    s.admitted.value.store(i + 1, std::memory_order_release);  // opens the gate for block i+1

    // ---- work (concurrent) --------------------------------------------------
    const std::uint64_t work_start = now_ns();
    try {
      (*s.work)(s.blocks[i], worker);
    } catch (...) {
      ok = false;
      std::lock_guard<std::mutex> error_lock(s.error_mutex);
      if (!s.error || i < s.error_seq) {
        s.error = std::current_exception();
        s.error_seq = i;
      }
    }
    const std::uint64_t work_end = now_ns();
    stats.work_ns += work_end - work_start;
    if (!ok) {
      s.failed.store(true, std::memory_order_release);
      return;
    }
    record.done_ns = work_end - s.start_ns;
    record.done.store(1, std::memory_order_release);

    // ---- ordered commit -----------------------------------------------------
    drain_commits(s);
  }
}

}  // namespace

RunStats Engine::run(std::span<const Block> blocks, const Task& admit, const Task& work, const Finish& commit,
                     const RunOptions& options) {
  State& s = *state_;
  std::lock_guard<std::mutex> run_lock(s.run_mutex);

  RunStats out;
  out.blocks = blocks.size();
  out.threads = threads_;
  if (blocks.empty()) return out;

  const unsigned helpers = static_cast<unsigned>(std::min<std::size_t>(threads_ - 1, blocks.size() - 1));
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.blocks = blocks.data();
    s.n = blocks.size();
    s.window = options.window != 0 ? options.window : static_cast<std::size_t>(threads_) * 4;
    s.admit = &admit;
    s.work = &work;
    s.commit = &commit;
    s.records = std::make_unique<BlockRecord[]>(blocks.size());
    s.worker_stats.assign(threads_, WorkerStats{});
    s.next_ticket.value.store(0, std::memory_order_relaxed);
    s.admitted.value.store(0, std::memory_order_relaxed);
    s.committed.value.store(0, std::memory_order_relaxed);
    s.clock.value.store(0, std::memory_order_relaxed);
    s.buffered_max.value.store(0, std::memory_order_relaxed);
    s.failed.store(false, std::memory_order_relaxed);
    s.error = nullptr;
    s.error_seq = 0;
    s.wanted = helpers;
    s.joined = 0;
    s.start_ns = now_ns();
    ++s.generation;
    s.job_open = true;
  }
  if (helpers != 0) s.wake.notify_all();

  drive(s, 0);

  {
    std::unique_lock<std::mutex> lock(s.mutex);
    s.job_open = false;  // workers that have not joined yet never will
    s.idle.wait(lock, [&] { return s.active == 0; });
  }
  // Every worker has left, so a final drain on this thread commits whatever a
  // worker finished after the last in-order drain (nothing, unless the job failed).
  if (!s.failed.load(std::memory_order_acquire)) drain_commits(s);

  out.wall_ns = now_ns() - s.start_ns;
  out.max_buffered = static_cast<std::size_t>(s.buffered_max.value.load(std::memory_order_relaxed));
  for (const WorkerStats& w : s.worker_stats) {
    out.work_ns += w.work_ns;
    out.gate_wait_ns += w.gate_wait_ns;
  }

  const std::size_t committed = static_cast<std::size_t>(s.committed.value.load(std::memory_order_acquire));
  const std::size_t admitted = static_cast<std::size_t>(s.admitted.value.load(std::memory_order_acquire));
  for (std::size_t i = 1; i < admitted; ++i) {
    if (s.records[i].admit_tick <= s.records[i - 1].admit_tick) ++out.admission_violations;
  }
  for (std::size_t i = 1; i < committed; ++i) {
    if (s.records[i].commit_tick <= s.records[i - 1].commit_tick) ++out.commit_violations;
  }
  if (options.trace) {
    out.trace.reserve(admitted);
    for (std::size_t i = 0; i < admitted; ++i) {
      const BlockRecord& r = s.records[i];
      BlockTrace t;
      t.seq = blocks[i].seq;
      t.worker = r.worker;
      t.admit_tick = r.admit_tick;
      t.commit_tick = i < committed ? r.commit_tick : 0;
      t.admit_ns = r.admit_ns;
      t.done_ns = r.done_ns;
      t.commit_ns = i < committed ? r.commit_ns : 0;
      out.trace.push_back(t);
    }
  }

  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(s.error_mutex);
    error = s.error;
    s.error = nullptr;
  }
  s.records.reset();
  if (error) std::rethrow_exception(error);
  return out;
}

}  // namespace detail

OrderedExecutor& OrderedExecutor::shared() {
  static OrderedExecutor executor(0);
  return executor;
}

}  // namespace prag::exec
