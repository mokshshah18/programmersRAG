// Index build (DESIGN.md section 6.13) and score calibration.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>

#include "engine_impl.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/core/text.hpp"
#include "prag/embed/hashing.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/parse/parser.hpp"
#include "prag/retrieve/baselines.hpp"

namespace prag::engine {

using detail::count_words;

namespace {

struct BuildingFlag {
  std::atomic<bool>& flag;
  explicit BuildingFlag(std::atomic<bool>& f) : flag(f) { flag = true; }
  ~BuildingFlag() { flag = false; }
};

struct ParsedItem {
  parse::ParsedFile file;
  std::uint64_t hash = 0;
  bool ok = false;
};

// For every unit, its `k` nearest units by cosine inside the same mode, nearest first.
graph::NeighborGraph nearest_in_mode(std::span<const float> rows, std::size_t dim, const kde::Modes& modes,
                                     std::size_t k, std::size_t block_bytes, exec::OrderedExecutor& executor) {
  const std::size_t n = modes.assignment.size();
  std::vector<std::vector<UnitId>> members(modes.count());
  for (std::size_t i = 0; i < n; ++i) members[modes.assignment[i]].push_back(static_cast<UnitId>(i));
  std::vector<std::uint32_t> offsets(1, 0);
  std::vector<UnitId> targets;
  const auto blocks = exec::partition_fixed(n, dim * sizeof(float), block_bytes);
  executor.run(
      blocks,
      [&](const exec::Block& b, unsigned) {
        std::vector<std::vector<UnitId>> out(b.count);
        std::vector<std::pair<float, UnitId>> cand;
        for (std::size_t t = 0; t < b.count; ++t) {
          const std::size_t i = b.first + t;
          cand.clear();
          for (const UnitId j : members[modes.assignment[i]]) {
            if (j == i) continue;
            cand.emplace_back(index::dot_f32(&rows[i * dim], &rows[static_cast<std::size_t>(j) * dim], dim), j);
          }
          const std::size_t take = std::min(k, cand.size());
          std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(take), cand.end(),
                            [](const auto& a, const auto& c) { return a.first > c.first || (a.first == c.first && a.second < c.second); });
          for (std::size_t q = 0; q < take; ++q) out[t].push_back(cand[q].second);
        }
        return out;
      },
      [&](const exec::Block&, std::vector<std::vector<UnitId>>&& out) {
        for (const auto& list : out) {
          targets.insert(targets.end(), list.begin(), list.end());
          offsets.push_back(static_cast<std::uint32_t>(targets.size()));
        }
      });
  return graph::NeighborGraph::from_csr(std::move(offsets), std::move(targets));
}

double to_seconds(const Stopwatch& sw) { return sw.elapsed_s(); }

}  // namespace

bool Engine::Impl::report_progress(const IndexProgressCallback& callback, const std::string& stage,
                                   std::size_t done, std::size_t total, const std::string& message) {
  IndexProgress p{stage, done, total, message};
  {
    std::lock_guard<std::mutex> lock(mu);
    progress = p;
  }
  return callback ? callback(p) : true;
}

IndexReport Engine::Impl::build(const IndexProgressCallback& callback) {
  std::lock_guard<std::mutex> build_lock(build_mu);
  BuildingFlag flag(building);
  {
    std::lock_guard<std::mutex> lock(mu);
    progress_running = true;
  }
  Stopwatch total_sw;
  Stopwatch sw;
  IndexReport rep;
  auto finish = [&](const std::string& error) {
    rep.error = error;
    rep.seconds["total"] = to_seconds(total_sw);
    const IndexProgress last{error.empty() ? "done" : "failed", rep.units, rep.units,
                             error.empty() ? "index built" : error};
    {
      std::lock_guard<std::mutex> lock(mu);
      progress = last;
      progress_running = false;
      last_report = rep;
    }
    if (!error.empty()) PRAG_WARN("index build failed: %s", error.c_str());
    if (callback) callback(last);
    return rep;
  };
  auto stage_done = [&](const char* name) {
    rep.seconds[name] = to_seconds(sw);
    sw.reset();
  };
  const char* cancelled = "cancelled";

  // ---- discover
  if (!report_progress(callback, "discover", 0, 0, "listing files")) return finish(cancelled);
  const std::vector<std::string> roots = detail::corpus_roots(settings, base_dir);
  if (roots.empty()) return finish("no corpus roots configured (setting corpus.roots)");
  fs::WalkOptions wo;
  wo.extensions = settings.get<std::vector<std::string>>(
      "corpus.extensions", {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".py"});
  for (auto& e : wo.extensions) e = text::to_lower(e);
  wo.exclude_substrings = settings.get<std::vector<std::string>>(
      "corpus.exclude", {"/.git/", "/build/", "/out/", "/.deps/", "/node_modules/"});
  wo.max_file_bytes = settings.get<std::uint64_t>("corpus.max_file_bytes", 1048576);
  std::vector<std::string> files;
  for (const std::string& root : roots) {
    if (!fs::exists(root)) return finish("corpus root does not exist: " + root);
    for (std::string& f : fs::walk(root, wo)) files.push_back(std::move(f));
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  if (files.empty()) return finish("no source files found under the corpus roots");
  stage_done("discover");

  exec::OrderedExecutor& ex = *executor;

  // ---- parse
  std::vector<ParsedItem> parsed(files.size());
  std::atomic<bool> cancel{false};
  {
    const auto blocks = exec::partition_fixed(files.size(), 1, 1);
    std::size_t done = 0;
    const std::size_t step = std::max<std::size_t>(1, files.size() / 50);
    try {
      ex.run(
          blocks,
          [&](const exec::Block& b, unsigned) {
            ParsedItem item;
            if (cancel) return item;
            const auto content = fs::read_file(files[b.first]);
            if (!content) return item;
            item.hash = hash64(*content);
            item.file = parse::parse_source(files[b.first], *content);
            item.ok = true;
            return item;
          },
          [&](const exec::Block& b, ParsedItem&& item) {
            parsed[b.first] = std::move(item);
            ++done;
            if (!cancel && (done % step == 0 || done == files.size()) &&
                !report_progress(callback, "parse", done, files.size(), "parsing files"))
              cancel = true;
          });
    } catch (const std::exception& e) {
      return finish(std::string("parse failed: ") + e.what());
    }
  }
  if (cancel) return finish(cancelled);

  auto corpus_ptr = std::make_shared<Corpus>();
  Corpus& c = *corpus_ptr;
  c.roots = roots;
  for (std::size_t f = 0; f < files.size(); ++f) {
    if (!parsed[f].ok) {
      PRAG_WARN("cannot read %s", files[f].c_str());
      continue;
    }
    const auto file_index = static_cast<std::uint32_t>(c.files.size());
    c.files.push_back(parsed[f].file.path.empty() ? files[f] : parsed[f].file.path);
    c.file_hashes.push_back(parsed[f].hash);
    for (SourceUnit& u : parsed[f].file.units) {
      u.id = static_cast<UnitId>(c.units.size());
      if (u.kind == UnitKind::Function) ++rep.functions;
      c.units.push_back(std::move(u));
      c.file_of_unit.push_back(file_index);
    }
  }
  parsed.clear();
  rep.files = c.files.size();
  rep.units = c.units.size();
  const std::size_t n = c.units.size();
  if (n < 2) return finish("the corpus has fewer than two units");
  stage_done("parse");

  // ---- graph
  if (!report_progress(callback, "graph", 0, n, "building the call graph")) return finish(cancelled);
  c.calls = parse::build_call_graph(c.units);
  rep.call_edges = c.calls.edges();
  stage_done("graph");

  // ---- embed
  const auto emb = ensure_embedder();
  const std::size_t dim = emb->dim();
  c.embedder_id = emb->id();
  const std::string cache_path = index_dir() + "/embeddings.bin";
  if (!report_progress(callback, "embed", 0, n, "embedding units")) return finish(cancelled);
  if (emb_cache.embedder_id != c.embedder_id) {
    detail::EmbeddingCache loaded;
    if (detail::load_embedding_cache(cache_path, &loaded) && loaded.embedder_id == c.embedder_id &&
        loaded.dim == dim)
      emb_cache = std::move(loaded);
    else
      emb_cache = detail::EmbeddingCache{c.embedder_id, dim, {}};
  }
  std::vector<float> rows(n * dim);
  {
    const std::size_t budget = emb->max_tokens();
    std::vector<std::string> rendered(n);
    std::vector<std::uint64_t> keys(n);
    std::vector<std::size_t> missing;
    for (std::size_t i = 0; i < n; ++i) {
      rendered[i] = embed::render_for_embedding(c.units[i], budget);
      keys[i] = hash_combine(c.units[i].content_hash, hash64(rendered[i]));
      const auto it = emb_cache.rows.find(keys[i]);
      if (it != emb_cache.rows.end()) {
        std::copy(it->second.begin(), it->second.end(), rows.begin() + static_cast<std::ptrdiff_t>(i * dim));
        ++rep.reused;
      } else {
        missing.push_back(i);
      }
    }
    const std::size_t batch = 16;
    for (std::size_t s = 0; s < missing.size(); s += batch) {
      if (!report_progress(callback, "embed", s, missing.size(), "embedding units")) return finish(cancelled);
      const std::size_t e = std::min(missing.size(), s + batch);
      std::vector<std::string_view> texts;
      for (std::size_t q = s; q < e; ++q) texts.push_back(rendered[missing[q]]);
      const std::vector<float> out = embed_texts(*emb, texts, embed::Role::Document);
      for (std::size_t q = s; q < e; ++q) {
        std::copy(out.begin() + static_cast<std::ptrdiff_t>((q - s) * dim),
                  out.begin() + static_cast<std::ptrdiff_t>((q - s + 1) * dim),
                  rows.begin() + static_cast<std::ptrdiff_t>(missing[q] * dim));
      }
    }
    rep.embedded = missing.size();
    // New cache: the rows of this build first, then older rows up to twice the corpus.
    detail::EmbeddingCache next{c.embedder_id, dim, {}};
    for (std::size_t i = 0; i < n; ++i)
      next.rows.emplace(keys[i], std::vector<float>(rows.begin() + static_cast<std::ptrdiff_t>(i * dim),
                                                    rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim)));
    std::vector<std::uint64_t> old_keys;
    for (const auto& kv : emb_cache.rows)
      if (!next.rows.count(kv.first)) old_keys.push_back(kv.first);
    std::sort(old_keys.begin(), old_keys.end());
    for (const std::uint64_t k : old_keys) {
      if (next.rows.size() >= 2 * n + 1024) break;
      next.rows.emplace(k, std::move(emb_cache.rows[k]));
    }
    emb_cache = std::move(next);
  }
  stage_done("embed");

  // ---- density
  const std::size_t bb = block_bytes();
  if (!report_progress(callback, "density", 0, n, "reducing and finding density modes")) return finish(cancelled);
  try {
    c.reduction = kde::fit_reduction(rows, dim, settings.get<std::size_t>("kde.dim", 32),
                                     settings.get<std::size_t>("kde.drop_top", 0), 1);
    c.reduced = kde::reduce(c.reduction, rows);
    const std::size_t m = c.reduction.out_dim;
    const kde::KappaSelection ks = kde::select_kappa(c.reduced, m, ex, bb);
    kde::MeanShiftOptions mo;
    mo.min_group = settings.get<std::size_t>("kde.min_group", 3);
    mo.block_bytes = bb;
    const double kappa_setting = settings.get<double>("kde.kappa", 0.0);
    if (kappa_setting > 0.0) {
      mo.kappa = kappa_setting;
    } else {
      const double nk = kde::neighbourhood_kappa(c.reduced, m, ex, bb);
      double k = ks.kappa;
      if (nk > 0.0 && std::isfinite(nk)) k = (k > 0.0 && std::isfinite(k)) ? std::min(k, nk) : nk;
      mo.kappa = (k > 0.0 && std::isfinite(k)) ? k : 0.0;
    }
    c.modes = kde::find_modes(c.reduced, m, mo, ex);
    rep.kappa = ks.kappa;
    c.anisotropy_before = kde::mean_pairwise_cosine(rows, dim);
    c.anisotropy_after = kde::mean_pairwise_cosine(c.reduced, m);
    if (c.modes.assignment.size() != n) return finish("mode finding failed");
    c.siblings = nearest_in_mode(rows, dim, c.modes, 8, bb, ex);
  } catch (const std::exception& e) {
    return finish(std::string("density failed: ") + e.what());
  }
  rep.groups = c.modes.count();
  rep.reduced_dim = c.reduction.out_dim;
  rep.explained_variance = c.reduction.explained_variance_ratio();
  rep.anisotropy_before = c.anisotropy_before;
  rep.anisotropy_after = c.anisotropy_after;
  stage_done("density");

  // ---- codes
  if (!report_progress(callback, "codes", 0, n, "encoding vectors")) return finish(cancelled);
  {
    index::BuildOptions bo;
    bo.codes = settings.get<std::string>("index.codes", "rabitq") == "int8" ? index::CodeKind::Int8
                                                                             : index::CodeKind::RaBitQ;
    bo.block_bytes = bb;
    const std::vector<std::uint32_t> layout = kde::layout_by_mode(c.modes);
    try {
      c.store.build(rows, dim, layout, c.modes.assignment, bo, ex);
    } catch (const std::exception& e) {
      return finish(std::string("encoding failed: ") + e.what());
    }
    if (c.store.size() != n) return finish("encoding the vectors failed");
  }
  rep.blocks = c.store.blocks().size();
  rep.block_bytes = c.store.block_bytes();
  stage_done("codes");

  // ---- calibrate
  c.lifts = default_lifts();
  c.calibration = retrieve::Calibration::fallback();
  {
    std::string error;
    if (!calibrate_corpus(c, CalibrationOptions{}, callback, emb, chat_model(), &c.calibration, &c.lifts,
                          &c.calibration_info, &error)) {
      if (error == cancelled) return finish(cancelled);
      PRAG_WARN("calibration failed: %s", error.c_str());
    }
  }
  stage_done("calibrate");

  // ---- save
  if (!report_progress(callback, "save", 0, 1, "writing the index")) return finish(cancelled);
  c.built_at = detail::iso_now();
  const std::uint64_t settings_hash = detail::index_settings_hash(settings);
  rep.seconds["save"] = 0.0;
  rep.seconds["total"] = to_seconds(total_sw);
  {
    std::string error;
    if (!detail::save_corpus(c, rep.kappa, settings_hash, rep, index_dir(), &error)) return finish(error);
    if (!detail::save_embedding_cache(cache_path, emb_cache)) PRAG_WARN("cannot write %s", cache_path.c_str());
  }
  stage_done("save");

  {
    std::lock_guard<std::mutex> lock(mu);
    corpus = corpus_ptr;
    kappa_cv = rep.kappa;
    stale = false;
  }
  PRAG_INFO("index built: %zu files, %zu units, %zu groups", rep.files, rep.units, rep.groups);
  return finish("");
}

// ---- calibration -----------------------------------------------------------

namespace {

struct QueryScores {
  bool ok = false;
  double z_relevant = 0.0;
  std::vector<double> z_negative;
};

std::string generate_question(llm::LanguageModel& chat, const SourceUnit& unit) {
  std::string body = unit.text;
  if (body.size() > 6000) body.resize(6000);
  const std::vector<ChatMessage> messages = {
      {"system", "You write questions about source code."},
      {"user",
       "Write one question a developer would ask that this function answers. Do not use the function name. "
       "Reply with the question only.\n\n```\n" +
           body + "\n```"}};
  llm::GenerateOptions go;
  go.max_tokens = 64;
  go.temperature = 0.0f;
  std::string out;
  chat.generate(messages, go, [&](std::string_view piece) {
    out += piece;
    return true;
  });
  const auto lines = text::split_lines(out);
  for (const std::string_view line : lines) {
    const std::string_view t = text::trim(line);
    if (!t.empty()) return std::string(t);
  }
  return {};
}

}  // namespace

bool Engine::Impl::calibrate_corpus(const Corpus& c, const CalibrationOptions& o,
                                    const IndexProgressCallback& callback,
                                    const std::shared_ptr<embed::Embedder>& emb,
                                    const std::shared_ptr<llm::LanguageModel>& lm, retrieve::Calibration* cal_out,
                                    retrieve::EdgeLift* lifts_out, CalibrationInfo* info_out, std::string* error) {
  auto fail = [&](const std::string& e) {
    if (error) *error = e;
    return false;
  };
  const std::size_t n = c.units.size();
  const std::size_t dim = c.store.dim();
  if (!emb || emb->id() != c.embedder_id) return fail("the embedder does not match the index");
  if (!report_progress(callback, "calibrate", 0, o.max_queries, "calibrating scores")) return fail("cancelled");

  CalibrationInfo info;
  const bool generate = lm && o.generate_questions;
  info.source = generate ? "generated" : "documentation";

  std::vector<UnitId> pool;
  for (const SourceUnit& u : c.units) {
    if (u.kind != UnitKind::Function || u.name.empty()) continue;
    if (generate ? u.tokens >= 16 : count_words(u.doc) >= 6) pool.push_back(u.id);
  }
  detail::shuffle_ids(pool, o.seed);

  std::vector<UnitId> qunits;
  std::vector<std::string> questions;
  for (const UnitId id : pool) {
    if (questions.size() >= o.max_queries) break;
    const SourceUnit& u = c.units[id];
    std::string q;
    if (generate) {
      q = generate_question(*lm, u);
      if (q.empty() || text::contains_icase(q, u.name)) continue;
    } else {
      q = std::string(text::trim(u.doc));
    }
    qunits.push_back(id);
    questions.push_back(std::move(q));
    if (generate && !report_progress(callback, "calibrate", questions.size(), o.max_queries, "generating questions"))
      return fail("cancelled");
  }

  const retrieve::EdgeLift defaults = default_lifts();
  if (questions.size() < 20) {
    *cal_out = retrieve::Calibration::fallback();
    *lifts_out = defaults;
    info.queries = questions.size();
    *info_out = info;
    return true;
  }

  // Question embeddings and embeddings of the relevant units without their documentation.
  const std::size_t nq = questions.size();
  std::vector<std::string_view> qtexts(questions.begin(), questions.end());
  const std::vector<float> qv = embed_texts(*emb, qtexts, embed::Role::Query);
  std::vector<float> rv(nq * dim);
  if (generate) {
    for (std::size_t i = 0; i < nq; ++i) {
      const auto v = c.store.vector(qunits[i]);
      std::copy(v.begin(), v.end(), rv.begin() + static_cast<std::ptrdiff_t>(i * dim));
    }
  } else {
    std::vector<std::string> rendered(nq);
    for (std::size_t i = 0; i < nq; ++i) {
      SourceUnit u = c.units[qunits[i]];
      u.doc.clear();
      u.text = detail::strip_leading_comment(u.text, u.language);
      rendered[i] = embed::render_for_embedding(u, emb->max_tokens());
    }
    std::vector<std::string_view> rtexts(rendered.begin(), rendered.end());
    rv = embed_texts(*emb, rtexts, embed::Role::Document);
  }

  std::vector<QueryScores> scores(nq);
  const auto blocks = exec::partition_fixed(nq, 1, 1);
  executor->run(
      blocks,
      [&](const exec::Block& b, unsigned) {
        const std::size_t i = b.first;
        QueryScores r;
        const float* q = &qv[i * dim];
        std::vector<double> s(n);
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          s[j] = index::dot_f32(q, c.store.vector(static_cast<UnitId>(j)).data(), dim);
          sum += s[j];
        }
        const double mean = sum / static_cast<double>(n);
        double var = 0.0;
        for (std::size_t j = 0; j < n; ++j) var += (s[j] - mean) * (s[j] - mean);
        const double sd = std::sqrt(var / static_cast<double>(n));
        if (!(sd > 0.0) || !std::isfinite(sd)) return r;
        r.z_relevant = (static_cast<double>(index::dot_f32(q, &rv[i * dim], dim)) - mean) / sd;
        std::uint64_t state = hash_combine(o.seed, i + 1);
        const std::size_t want = std::min(o.negatives_per_query, n - 1);
        std::vector<char> taken(n, 0);
        taken[qunits[i]] = 1;
        while (r.z_negative.size() < want) {
          const std::size_t j = static_cast<std::size_t>(detail::next_random(state) % n);
          if (taken[j]) continue;
          taken[j] = 1;
          r.z_negative.push_back((s[j] - mean) / sd);
        }
        r.ok = true;
        return r;
      },
      [&](const exec::Block& b, QueryScores&& r) { scores[b.first] = std::move(r); });

  const std::size_t heldout = static_cast<std::size_t>(std::llround(o.heldout_fraction * static_cast<double>(nq)));
  const std::size_t n_fit = nq - std::min(heldout, nq);
  std::vector<double> zr_fit, zn_fit, zr_held;
  std::vector<retrieve::LabeledScore> held;
  std::size_t fit_queries = 0, held_queries = 0;
  for (std::size_t i = 0; i < nq; ++i) {
    if (!scores[i].ok) continue;
    if (i < n_fit) {
      ++fit_queries;
      zr_fit.push_back(scores[i].z_relevant);
      zn_fit.insert(zn_fit.end(), scores[i].z_negative.begin(), scores[i].z_negative.end());
    } else {
      ++held_queries;
      zr_held.push_back(scores[i].z_relevant);
      held.push_back({scores[i].z_relevant, true});
      for (const double z : scores[i].z_negative) held.push_back({z, false});
    }
  }
  if (fit_queries < 20) {
    *cal_out = retrieve::Calibration::fallback();
    *lifts_out = defaults;
    info.queries = fit_queries;
    *info_out = info;
    return true;
  }
  *cal_out = retrieve::Calibration::fit(zr_fit, zn_fit, 1.0 / static_cast<double>(n));
  info.fitted = cal_out->fitted();
  info.queries = fit_queries;
  info.heldout_queries = held_queries;
  if (!held.empty()) info.heldout = retrieve::reliability(*cal_out, held);
  info.conformal_z = retrieve::conformal_threshold(zr_held.empty() ? zr_fit : zr_held, info.conformal_alpha);
  if (!std::isfinite(info.conformal_z)) info.conformal_z = 0.0;

  // Edge lifts from utility labels (docs/EXPERIMENTS.md), with the chat model only.
  *lifts_out = defaults;
  if (lm && o.estimate_lifts) {
    std::size_t useful[4] = {0, 0, 0, 0};
    std::size_t useless[4] = {0, 0, 0, 0};
    const std::size_t lq = std::min(o.lift_queries, n_fit);
    std::uint64_t state = hash_combine(o.seed, 0x11f7);
    for (std::size_t i = 0; i < lq; ++i) {
      if (!report_progress(callback, "calibrate", i, lq, "estimating edge lifts")) return fail("cancelled");
      const UnitId u = qunits[i];
      const UnitId oracle[1] = {u};
      const auto base_messages = detail::prompt_for(c, questions[i], oracle, false);
      llm::GenerateOptions go;
      go.max_tokens = 128;
      go.temperature = 0.0f;
      std::string answer;
      lm->generate(base_messages, go, [&](std::string_view p) {
        answer += p;
        return true;
      });
      if (text::trim(answer).empty()) continue;
      const llm::Likelihood base = lm->score(base_messages, answer);
      if (base.tokens == 0) continue;
      const double base_nll = base.nll / base.tokens;
      std::vector<UnitId> kinds[4];
      const std::size_t k = o.lift_candidates_per_kind;
      for (const UnitId v : c.calls.callees(u))
        if (kinds[0].size() < k && v != u) kinds[0].push_back(v);
      for (const UnitId v : c.calls.callers(u))
        if (kinds[1].size() < k && v != u) kinds[1].push_back(v);
      for (const UnitId v : c.siblings.of(u))
        if (kinds[2].size() < k && v != u) kinds[2].push_back(v);
      const std::size_t n_random = kinds[0].size() + kinds[1].size() + kinds[2].size();
      for (std::size_t t = 0; t < n_random * 4 && kinds[3].size() < n_random; ++t) {
        const auto v = static_cast<UnitId>(detail::next_random(state) % n);
        if (v != u) kinds[3].push_back(v);
      }
      for (int kind = 0; kind < 4; ++kind) {
        for (const UnitId v : kinds[kind]) {
          const UnitId ctx[2] = {u, v};
          const llm::Likelihood with = lm->score(detail::prompt_for(c, questions[i], ctx, false), answer);
          if (with.tokens == 0) continue;
          if (base_nll - with.nll / with.tokens >= o.lift_min_gain)
            ++useful[kind];
          else
            ++useless[kind];
        }
      }
    }
    auto odds = [&](int kind) {
      return (static_cast<double>(useful[kind]) + 1.0) / (static_cast<double>(useless[kind]) + 1.0);
    };
    auto lift = [&](int kind) { return std::clamp(odds(kind) / odds(3), 1.0, 16.0); };
    lifts_out->callee = lift(0);
    lifts_out->caller = lift(1);
    lifts_out->sibling = lift(2);
    info.lifts_estimated = true;
  }
  *info_out = info;
  report_progress(callback, "calibrate", o.max_queries, o.max_queries, "calibration done");
  return true;
}

}  // namespace prag::engine
