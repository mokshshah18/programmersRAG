// E5: execution. Block-size and thread sweeps of the ordered scan, of a batch
// scan and of mean shift; ordered against unordered execution; determinism.
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "common.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/hash.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/exec/ordered_executor.hpp"
#include "prag/index/synthetic.hpp"
#include "prag/index/vector_store.hpp"
#include "prag/kde/directional.hpp"

namespace bench {

namespace {

namespace idx = prag::index;
namespace ex = prag::exec;

// A persistent pool running an unordered parallel loop: blocks are handed out
// by one atomic counter, nothing is ordered or committed.
class Pool {
 public:
  explicit Pool(unsigned threads) : n_(threads) {
    for (unsigned w = 0; w < threads; ++w) workers_.emplace_back([this, w] { loop(w); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  void run(std::size_t blocks, const std::function<void(std::size_t, unsigned)>& fn) {
    {
      std::lock_guard<std::mutex> lock(m_);
      fn_ = &fn;
      blocks_ = blocks;
      next_.store(0);
      pending_ = n_;
      ++generation_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(m_);
    done_.wait(lock, [&] { return pending_ == 0; });
  }

 private:
  void loop(unsigned w) {
    std::size_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(m_);
      cv_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      const auto* fn = fn_;
      const std::size_t blocks = blocks_;
      lock.unlock();
      for (;;) {
        const std::size_t b = next_.fetch_add(1);
        if (b >= blocks) break;
        (*fn)(b, w);
      }
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }

  unsigned n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(std::size_t, unsigned)>* fn_ = nullptr;
  std::size_t blocks_ = 0;
  std::atomic<std::size_t> next_{0};
  std::size_t pending_ = 0;
  std::size_t generation_ = 0;
  bool stop_ = false;
};

double iteration_mean(const prag::kde::Modes& m) {
  if (m.iterations.empty()) return kNaN;
  double sum = 0.0;
  for (const auto v : m.iterations) sum += static_cast<double>(v);
  return sum / D(m.iterations.size());
}

std::string hex(std::uint64_t h) {
  char buffer[24];
  std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(h));
  return buffer;
}

template <class T>
std::uint64_t hash_of(const std::vector<T>& v, std::uint64_t seed = 0) {
  return prag::hash64(std::string_view(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T)), seed);
}

struct Tally {
  std::uint64_t admission_violations = 0, commit_violations = 0, runs = 0, unordered_mismatches = 0;
  // hashes[kind] -> key (block or "all") -> set of hashes
  std::map<std::string, std::map<std::string, std::set<std::string>>> hashes;
  void add(const ex::RunStats& s) {
    admission_violations += s.admission_violations;
    commit_violations += s.commit_violations;
    ++runs;
  }
  void hash(const std::string& kind, const std::string& scope, std::uint64_t h) { hashes[kind][scope].insert(hex(h)); }
};

struct Settings {
  std::vector<std::size_t> blocks_kib, threads;
  std::size_t reps = 7;
};

// Times `f` (microseconds) `reps` times after one warm-up run.
template <class F>
std::vector<double> time_us(std::size_t reps, F&& f) {
  f();
  std::vector<double> samples;
  for (std::size_t r = 0; r < reps; ++r) {
    prag::Stopwatch watch;
    f();
    samples.push_back(static_cast<double>(watch.elapsed_ns()) / 1e3);
  }
  return samples;
}

void block_dots(const idx::VectorStore& store, const ex::Block& b, const float* q, std::size_t dim, float* out) {
  for (std::size_t i = 0; i < b.count; ++i)
    out[i] = idx::dot_f32(q, store.vector(store.unit_at(b.first + i)).data(), dim);
}

struct Bed {
  std::string name;
  idx::VectorStore* store;
  std::vector<float> query;   // one query
  std::vector<float> batch;   // 64 queries
  std::size_t batch_count = 64;
};

// One sweep over block sizes and thread counts for a bed.
json sweep_bed(Bed& bed, const Settings& cfg, std::vector<std::unique_ptr<ex::OrderedExecutor>>& execs,
               std::vector<std::unique_ptr<Pool>>& pools, Tally& tally, unsigned logical_cpus) {
  const std::size_t dim = bed.store->dim(), n = bed.store->size();
  json rows = json::array();
  for (const std::size_t kib : cfg.blocks_kib) {
    bed.store->reblock(kib * 1024);
    const std::size_t blocks = bed.store->blocks().size();
    for (std::size_t ti = 0; ti < cfg.threads.size(); ++ti) {
      const unsigned threads = static_cast<unsigned>(cfg.threads[ti]);
      ex::OrderedExecutor& executor = *execs[ti];
      Pool& pool = *pools[ti];
      json row{{"bed", bed.name},         {"block_kib", kib},      {"threads", threads}, {"blocks", blocks},
               {"oversubscribed", threads > logical_cpus}, {"vectors", n}};
      const std::string scope = bed.name + "/" + std::to_string(kib);

      // (a)/(b): single-query scan with the codes, through the ordered executor.
      {
        std::vector<double> wall, gate_share, gate_ns, work_ns;
        idx::ScanResult last;
        const auto samples = time_us(cfg.reps, [&] {
          last = idx::scan(*bed.store, bed.query, executor);
          tally.add(last.stats);
          wall.push_back(static_cast<double>(last.stats.wall_ns) / 1e3);
          const double total = static_cast<double>(last.stats.wall_ns) * static_cast<double>(last.stats.threads);
          gate_share.push_back(total > 0.0 ? static_cast<double>(last.stats.gate_wait_ns) / total : kNaN);
          gate_ns.push_back(static_cast<double>(last.stats.gate_wait_ns));
          work_ns.push_back(static_cast<double>(last.stats.work_ns));
        });
        tally.hash(bed.name + ":scan_scores", "all", hash_of(last.scores));
        tally.hash(bed.name + ":scan_scores", scope, hash_of(last.scores));
        tally.hash(bed.name + ":scan_error", "all", hash_of(last.error));
        const double stats[2] = {last.mean, last.stddev};
        tally.hash(bed.name + ":scan_mean_stddev", scope, prag::hash64(std::string_view(reinterpret_cast<const char*>(stats), sizeof stats)));
        row["scan_us"] = timing_json(samples);
        row["scan_wall_in_executor_us"] = timing_json(wall);
        row["gate_wait_share_of_worker_time"] = timing_json(gate_share);
        row["gate_wait_ns"] = timing_json(gate_ns);
        row["work_ns"] = timing_json(work_ns);
      }

      // Ordered against unordered execution of the same work (exact dot products of every block).
      {
        std::vector<float> result(n), q(bed.query);
        std::vector<double> gate;
        const auto ordered = time_us(cfg.reps, [&] {
          const ex::RunStats s = executor.run(
              bed.store->blocks(),
              [&](const ex::Block& b, unsigned) {
                std::vector<float> v(b.count);
                block_dots(*bed.store, b, q.data(), dim, v.data());
                return v;
              },
              [&](const ex::Block& b, std::vector<float>&& v) {
                for (std::size_t i = 0; i < b.count; ++i) result[bed.store->unit_at(b.first + i)] = v[i];
              });
          tally.add(s);
        });
        tally.hash(bed.name + ":dot_ordered", "all", hash_of(result));
        const std::uint64_t reference = hash_of(result);
        std::fill(result.begin(), result.end(), 0.0f);
        const auto unordered = time_us(cfg.reps, [&] {
          pool.run(bed.store->blocks().size(), [&](std::size_t bi, unsigned) {
            const ex::Block& b = bed.store->blocks()[bi];
            std::vector<float> v(b.count);
            block_dots(*bed.store, b, q.data(), dim, v.data());
            for (std::size_t i = 0; i < b.count; ++i) result[bed.store->unit_at(b.first + i)] = v[i];
          });
        });
        if (hash_of(result) != reference) ++tally.unordered_mismatches;
        row["dot_ordered_us"] = timing_json(ordered);
        row["dot_unordered_us"] = timing_json(unordered);
        row["ordered_over_unordered_median"] = median(ordered) / median(unordered);
      }

      // (c): a batch of queries, block by block.
      {
        std::vector<idx::ScanResult> results;
        const auto samples = time_us(std::max<std::size_t>(3, cfg.reps / 2 + 1), [&] {
          results = idx::scan_batch(*bed.store, bed.batch, executor);
          if (!results.empty()) tally.add(results.front().stats);
        });
        std::uint64_t h = 0;
        for (const auto& r : results) h = prag::hash_combine(h, hash_of(r.scores));
        tally.hash(bed.name + ":batch_scores", "all", h);
        row["batch_us"] = timing_json(samples);
        row["batch_queries"] = bed.batch_count;
      }
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

}  // namespace

int run_e5(Context& ctx) {
  const std::string started = iso_now();
  const auto& c = *ctx.corpus;
  const std::size_t dim = ctx.dim();
  const prag::exec::CacheTopology& topology = prag::exec::cache_topology();
  Settings cfg;
  cfg.blocks_kib = ctx.quick() ? std::vector<std::size_t>{4, 16, 64} : std::vector<std::size_t>{1, 2, 4, 8, 16, 32, 64, 256, 1024};
  cfg.threads = ctx.quick() ? std::vector<std::size_t>{1, 2, 4} : std::vector<std::size_t>{1, 2, 4, 8, 16, 24};
  cfg.reps = ctx.quick() ? 3 : 7;
  const std::size_t synthetic_n = ctx.quick() ? 100000 : 1000000;

  std::vector<std::unique_ptr<ex::OrderedExecutor>> execs;
  std::vector<std::unique_ptr<Pool>> pools;
  for (const std::size_t t : cfg.threads) {
    execs.push_back(std::make_unique<ex::OrderedExecutor>(static_cast<unsigned>(t)));
    pools.push_back(std::make_unique<Pool>(static_cast<unsigned>(t)));
  }

  Tally tally;
  json body = json::object();

  body["cache_topology"] = json{{"description", prag::exec::describe(topology)},
                                {"detected", topology.detected},
                                {"logical_cpus", topology.logical_cpus},
                                {"l1d_min_bytes", topology.l1d_min_bytes},
                                {"l1d_per_thread_min_bytes", topology.l1d_per_thread_min_bytes},
                                {"recommended_block_bytes", prag::exec::recommended_block_bytes(topology)},
                                {"engine_block_bytes", ctx.engine->block_bytes()},
                                {"simd_kernel", idx::simd_kernel()},
                                {"hardware_counters", "none read; cache behaviour is inferred from timing only"}};
  json classes = json::array();
  for (const auto& k : topology.classes)
    classes.push_back(json{{"level", k.level},
                           {"type", k.type == ex::CacheType::Data ? "data" : k.type == ex::CacheType::Instruction ? "instruction" : "unified"},
                           {"size_bytes", k.size_bytes},
                           {"sharing_threads", k.sharing_threads},
                           {"instances", k.instances}});
  body["cache_topology"]["classes"] = classes;

  // Test beds.
  idx::VectorStore corpus_store;
  {
    idx::BuildOptions options = c.store.options();
    options.block_bytes = 16 * 1024;
    corpus_store.build(corpus_rows(ctx), dim, prag::kde::layout_by_mode(c.modes), c.modes.assignment, options,
                       ctx.engine->executor());
  }
  Bed corpus_bed;
  corpus_bed.name = "corpus";
  corpus_bed.store = &corpus_store;
  {
    QuerySets& sets = query_sets(ctx);
    std::vector<std::string> texts;
    for (std::size_t i = 0; i < 64; ++i) {
      if (!sets.doc.queries.empty()) texts.push_back(sets.doc.queries[i % sets.doc.queries.size()].question);
      else texts.push_back("how does the allocator release a block of memory");
    }
    corpus_bed.batch.resize(64 * dim);
    const auto emb = ctx.engine->embedder();
    for (std::size_t i = 0; i < 64; ++i) {
      const std::string_view one[1] = {texts[i]};
      emb->embed(one, prag::embed::Role::Query, &corpus_bed.batch[i * dim]);
    }
    corpus_bed.query.assign(corpus_bed.batch.begin(), corpus_bed.batch.begin() + static_cast<std::ptrdiff_t>(dim));
  }

  progress("E5: synthetic index of %zu vectors of dimension 64 ...", synthetic_n);
  prag::Stopwatch build_watch;
  idx::SyntheticOptions so;
  so.n = synthetic_n;
  so.dim = 64;
  so.clusters = 32;
  so.seed = ctx.seed + 51;
  idx::SyntheticData data = idx::synthetic_embeddings(so);
  const double generate_s = build_watch.elapsed_s();
  idx::VectorStore synthetic_store;
  build_watch.reset();
  {
    idx::BuildOptions options;
    options.codes = idx::CodeKind::RaBitQ;
    options.block_bytes = 16 * 1024;
    options.seed = 1;
    synthetic_store.build(data.rows, so.dim, idx::layout_by_cluster(data), data.cluster, options, ctx.engine->executor());
  }
  const double store_s = build_watch.elapsed_s();
  Bed synthetic_bed;
  synthetic_bed.name = "synthetic";
  synthetic_bed.store = &synthetic_store;
  synthetic_bed.batch = idx::synthetic_queries(data, 64, 0.3, ctx.seed + 52);
  synthetic_bed.query.assign(synthetic_bed.batch.begin(), synthetic_bed.batch.begin() + static_cast<std::ptrdiff_t>(so.dim));
  body["synthetic_index"] = json{{"vectors", so.n},          {"dim", so.dim},
                                 {"clusters", so.clusters},  {"seed", so.seed},
                                 {"codes", "rabitq"},        {"bytes_per_code", synthetic_store.bytes_per_code()},
                                 {"generate_seconds", generate_s}, {"build_seconds", store_s}};
  body["corpus_index"] = json{{"vectors", corpus_store.size()}, {"dim", dim}, {"codes", idx::to_string(corpus_store.code_kind())},
                              {"bytes_per_code", corpus_store.bytes_per_code()}};

  body["scan_corpus"] = guarded("scan corpus", [&] { return sweep_bed(corpus_bed, cfg, execs, pools, tally, topology.logical_cpus); });
  body["scan_synthetic"] = guarded("scan synthetic", [&] { return sweep_bed(synthetic_bed, cfg, execs, pools, tally, topology.logical_cpus); });

  // Batch of queries against the same number of single scans, at the default block size.
  body["batch_vs_single"] = guarded("batch vs single", [&] {
    json rows = json::array();
    for (Bed* bed : {&corpus_bed, &synthetic_bed}) {
      bed->store->reblock(16 * 1024);
      for (std::size_t ti = 0; ti < cfg.threads.size(); ++ti) {
        ex::OrderedExecutor& executor = *execs[ti];
        const auto single = time_us(std::max<std::size_t>(3, cfg.reps / 2 + 1), [&] {
          for (std::size_t i = 0; i < bed->batch_count; ++i) {
            const idx::ScanResult r = idx::scan(*bed->store, std::span<const float>(&bed->batch[i * bed->store->dim()], bed->store->dim()), executor);
            tally.add(r.stats);
          }
        });
        const auto batch = time_us(std::max<std::size_t>(3, cfg.reps / 2 + 1), [&] {
          const auto r = idx::scan_batch(*bed->store, bed->batch, executor);
          if (!r.empty()) tally.add(r.front().stats);
        });
        rows.push_back(json{{"bed", bed->name}, {"threads", cfg.threads[ti]}, {"block_kib", 16},
                            {"queries", bed->batch_count}, {"sequential_scans_us", timing_json(single)},
                            {"batch_us", timing_json(batch)}});
      }
    }
    return rows;
  });

  // Bit-identity of the batch with single scans.
  body["batch_equals_single"] = guarded("batch equals single", [&] {
    json out = json::object();
    for (Bed* bed : {&corpus_bed, &synthetic_bed}) {
      bed->store->reblock(16 * 1024);
      const auto r = idx::scan_batch(*bed->store, bed->batch, *execs.back());
      bool equal = r.size() == bed->batch_count;
      for (std::size_t i = 0; equal && i < std::min<std::size_t>(r.size(), 8); ++i) {
        const auto s = idx::scan(*bed->store, std::span<const float>(&bed->batch[i * bed->store->dim()], bed->store->dim()), *execs.front());
        equal = hash_of(s.scores) == hash_of(r[i].scores) && hash_of(s.error) == hash_of(r[i].error);
      }
      out[bed->name] = json{{"first_8_queries_bit_identical", equal}};
    }
    return out;
  });

  // (d) mean shift on the corpus.
  body["mean_shift"] = guarded("mean shift", [&] {
    json rows = json::array();
    const std::size_t m = c.reduction.out_dim;
    for (const std::size_t kib : cfg.blocks_kib) {
      for (std::size_t ti = 0; ti < cfg.threads.size(); ++ti) {
        const unsigned threads = static_cast<unsigned>(cfg.threads[ti]);
        prag::kde::MeanShiftOptions mo;
        mo.kappa = c.modes.kappa;
        mo.min_group = ctx.engine->settings().get<std::size_t>("kde.min_group", 3);
        mo.block_bytes = kib * 1024;
        prag::kde::Modes modes;
        const auto samples = time_us(std::max<std::size_t>(3, cfg.reps), [&] { modes = prag::kde::find_modes(c.reduced, m, mo, *execs[ti]); });
        const std::uint64_t ha = hash_of(modes.assignment), hi = hash_of(modes.iterations), hc = hash_of(modes.centers);
        tally.hash("mean_shift_assignment", "all", ha);
        tally.hash("mean_shift_iterations", "all", hi);
        tally.hash("mean_shift_centers", "all", hc);
        tally.hash("mean_shift_centers", "block " + std::to_string(kib), hc);
        rows.push_back(json{{"block_kib", kib},
                            {"threads", threads},
                            {"oversubscribed", threads > topology.logical_cpus},
                            {"seconds", timing_json([&] { std::vector<double> s; for (double v : samples) s.push_back(v / 1e6); return s; }())},
                            {"modes", modes.count()},
                            {"mean_iterations", iteration_mean(modes)}});
      }
    }
    return json{{"points", c.reduced.size() / m}, {"dim", m}, {"kappa", c.modes.kappa}, {"rows", rows}};
  });

  // Determinism and ordering summary.
  json det = json::object();
  for (const auto& [kind, scopes] : tally.hashes) {
    json entry = json::object();
    for (const auto& [scope, set] : scopes) {
      json hs = json::array();
      for (const auto& h : set) hs.push_back(h);
      entry[scope] = json{{"distinct", set.size()}, {"values", hs}};
    }
    det[kind] = entry;
  }
  body["determinism"] = json{{"hashes", det},
                             {"note", "scan scores, errors and batch scores must give one hash across all block sizes and thread counts; "
                                      "mean and standard deviation give one hash per block size; mean-shift assignment and iterations one hash overall"}};
  body["ordering"] = json{{"runs_checked", tally.runs},
                          {"unordered_loop_result_differs_from_ordered", tally.unordered_mismatches},
                          {"admission_violations", tally.admission_violations},
                          {"commit_violations", tally.commit_violations}};

  write_result(ctx, "e5_execution", started,
               json{{"blocks_kib", cfg.blocks_kib},
                    {"threads", cfg.threads},
                    {"repetitions", cfg.reps},
                    {"warmup_runs", 1},
                    {"statistic", "median with inter-quartile range, microseconds (seconds for mean shift)"},
                    {"synthetic_vectors", synthetic_n},
                    {"batch_queries", 64},
                    {"unordered_loop", "persistent thread pool, one atomic block counter, same dot-product kernel and block partition"},
                    {"hardware_counters", "none"}},
               std::move(body));
  return 0;
}

}  // namespace bench
