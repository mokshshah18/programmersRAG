// Query sets, corpus embeddings and exact scoring shared by the experiments.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

#include "common.hpp"
#include "prag/context/prompt.hpp"
#include "prag/core/hash.hpp"
#include "prag/embed/hashing.hpp"
#include "prag/index/vector_store.hpp"

namespace fsys = std::filesystem;

namespace bench {

std::size_t count_words(std::string_view s) {
  std::size_t n = 0;
  bool in_word = false;
  for (const char c : s) {
    const bool blank = c == ' ' || c == '\t' || c == '\n' || c == '\r';
    if (!blank && !in_word) ++n;
    in_word = !blank;
  }
  return n;
}

std::string strip_leading_comment(std::string_view text, std::string_view language) {
  std::size_t i = 0;
  const bool python = language == "python";
  for (;;) {
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
    if (!python && text.substr(i, 2) == "//") {
      const std::size_t e = text.find('\n', i);
      i = e == std::string_view::npos ? text.size() : e + 1;
    } else if (!python && text.substr(i, 2) == "/*") {
      const std::size_t e = text.find("*/", i + 2);
      i = e == std::string_view::npos ? text.size() : e + 2;
    } else if (python && i < text.size() && text[i] == '#') {
      const std::size_t e = text.find('\n', i);
      i = e == std::string_view::npos ? text.size() : e + 1;
    } else {
      break;
    }
  }
  return std::string(text.substr(i));
}

std::string corpus_fingerprint(const Context& ctx) {
  std::uint64_t h = prag::hash64(ctx.corpus->embedder_id);
  for (const auto& u : ctx.corpus->units) h = prag::hash_combine(h, u.content_hash);
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(h));
  return buffer;
}

namespace {

const char* split_name(int s) { return s == kCalibration ? "calibration" : s == kHeldout ? "heldout" : "test"; }
int split_from_name(const std::string& s) { return s == "calibration" ? kCalibration : s == "heldout" ? kHeldout : kTest; }

json set_json(const QuerySet& set) {
  if (!set.ok()) return skipped(set.skipped);
  json queries = json::array();
  std::size_t by_split[3] = {0, 0, 0};
  for (const Query& q : set.queries) {
    queries.push_back(json{{"id", q.id},
                           {"unit", q.unit},
                           {"question", q.question},
                           {"split", split_name(q.split)},
                           {"reference", q.reference}});
    ++by_split[q.split];
  }
  return json{{"count", set.queries.size()},
              {"by_split", json{{"calibration", by_split[0]}, {"heldout", by_split[1]}, {"test", by_split[2]}}},
              {"queries", queries}};
}

void read_set(const json& j, QuerySet* set) {
  set->queries.clear();
  set->skipped.clear();
  if (!j.is_object()) {
    set->skipped = "missing";
    return;
  }
  if (j.contains("skipped")) {
    set->skipped = j["skipped"].get<std::string>();
    return;
  }
  for (const json& q : j.value("queries", json::array())) {
    Query query;
    query.id = q.value("id", std::string());
    query.unit = q.value("unit", prag::kNoUnit);
    query.question = q.value("question", std::string());
    query.split = split_from_name(q.value("split", std::string("test")));
    query.reference = q.value("reference", std::string());
    set->queries.push_back(std::move(query));
  }
}

}  // namespace

json query_sets_json(Context& ctx, const QuerySets& sets) {
  (void)ctx;
  return json{{"fingerprint", sets.fingerprint}, {"info", sets.info},          {"q_doc", set_json(sets.doc)},
              {"q_gen", set_json(sets.gen)},     {"utility", sets.utility}};
}

QuerySets& query_sets(Context& ctx) {
  if (ctx.sets) return *ctx.sets;
  auto sets = std::make_unique<QuerySets>();
  sets->doc.name = "q_doc";
  sets->gen.name = "q_gen";
  sets->fingerprint = corpus_fingerprint(ctx);
  const std::string path = ctx.out_dir + "/queries.json";
  bool loaded = false;
  if (fsys::exists(path)) {
    std::ifstream f(path, std::ios::binary);
    json j = json::parse(f, nullptr, false);
    if (j.is_object() && j.value("fingerprint", std::string()) == sets->fingerprint &&
        j.value("quick", false) == ctx.quick()) {
      read_set(j.value("q_doc", json()), &sets->doc);
      read_set(j.value("q_gen", json()), &sets->gen);
      sets->utility = j.value("utility", skipped("not stored"));
      sets->info = j.value("info", json::object());
      loaded = true;
      progress("query sets loaded from %s", path.c_str());
    }
  }
  if (!loaded) {
    progress("query sets built in memory (no matching %s)", path.c_str());
    build_query_sets(ctx, sets.get());
  }
  ctx.sets = std::move(sets);
  return *ctx.sets;
}

// ---- splits ----------------------------------------------------------------

const std::vector<int>& unit_splits(Context& ctx) {
  static std::vector<int> splits;
  static const Context* owner = nullptr;
  if (owner == &ctx && splits.size() == ctx.n()) return splits;
  splits.assign(ctx.n(), -1);
  std::vector<UnitId> ids;
  for (const auto& u : ctx.corpus->units)
    if (u.kind == prag::UnitKind::Function && !u.name.empty()) ids.push_back(u.id);
  shuffle(ids, ctx.seed);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const double f = D(i) / D(ids.size());
    splits[ids[i]] = f < 0.5 ? kCalibration : f < 0.75 ? kHeldout : kTest;
  }
  owner = &ctx;
  return splits;
}

// ---- embeddings ------------------------------------------------------------

const std::vector<float>& corpus_rows(Context& ctx) {
  if (ctx.rows_cache.size() == ctx.n() * ctx.dim()) return ctx.rows_cache;
  const std::size_t dim = ctx.dim();
  ctx.rows_cache.assign(ctx.n() * dim, 0.0f);
  for (std::size_t i = 0; i < ctx.n(); ++i) {
    const auto v = ctx.corpus->store.vector(static_cast<UnitId>(i));
    std::copy(v.begin(), v.end(), ctx.rows_cache.begin() + static_cast<std::ptrdiff_t>(i * dim));
  }
  return ctx.rows_cache;
}

const std::vector<float>& docfree_rows(Context& ctx) {
  const std::size_t n = ctx.n(), dim = ctx.dim();
  if (ctx.docfree_cache.size() == n * dim) return ctx.docfree_cache;
  const auto emb = ctx.engine->embedder();
  if (!emb || emb->id() != ctx.corpus->embedder_id) throw std::runtime_error("embedder does not match the index");
  const std::uint64_t key = prag::hash_combine(prag::hash64(corpus_fingerprint(ctx)), prag::hash64("docfree-v1"));
  const std::string path = ctx.data_dir + "/docfree.bin";
  {
    std::ifstream f(path, std::ios::binary);
    char magic[8] = {0};
    std::uint64_t k = 0, nn = 0, dd = 0;
    if (f && f.read(magic, 8) && f.read(reinterpret_cast<char*>(&k), 8) && f.read(reinterpret_cast<char*>(&nn), 8) &&
        f.read(reinterpret_cast<char*>(&dd), 8) && std::memcmp(magic, "PRAGDOCF", 8) == 0 && k == key && nn == n &&
        dd == dim) {
      ctx.docfree_cache.resize(n * dim);
      f.read(reinterpret_cast<char*>(ctx.docfree_cache.data()), static_cast<std::streamsize>(n * dim * sizeof(float)));
      if (f) {
        progress("documentation-free embeddings loaded from cache");
        return ctx.docfree_cache;
      }
    }
  }
  const std::vector<float>& base = corpus_rows(ctx);
  ctx.docfree_cache = base;
  std::vector<std::size_t> changed;
  std::vector<std::string> rendered;
  for (std::size_t i = 0; i < n; ++i) {
    prag::SourceUnit u = ctx.corpus->units[i];
    const std::string stripped = strip_leading_comment(u.text, u.language);
    if (stripped == u.text) continue;
    u.doc.clear();
    u.text = stripped;
    changed.push_back(i);
    rendered.push_back(prag::embed::render_for_embedding(u, emb->max_tokens()));
  }
  progress("embedding %zu units without their leading comment ...", changed.size());
  const std::size_t batch = 16;
  std::vector<float> out(batch * dim);
  for (std::size_t start = 0; start < changed.size(); start += batch) {
    const std::size_t count = std::min(batch, changed.size() - start);
    std::vector<std::string_view> texts;
    for (std::size_t j = 0; j < count; ++j) texts.emplace_back(rendered[start + j]);
    emb->embed(texts, prag::embed::Role::Document, out.data());
    for (std::size_t j = 0; j < count; ++j)
      std::copy(out.begin() + static_cast<std::ptrdiff_t>(j * dim), out.begin() + static_cast<std::ptrdiff_t>((j + 1) * dim),
                ctx.docfree_cache.begin() + static_cast<std::ptrdiff_t>(changed[start + j] * dim));
    if ((start / batch) % 25 == 24) progress("  %zu / %zu", start + count, changed.size());
  }
  {
    std::ofstream f(path, std::ios::binary);
    const std::uint64_t nn = n, dd = dim;
    f.write("PRAGDOCF", 8);
    f.write(reinterpret_cast<const char*>(&key), 8);
    f.write(reinterpret_cast<const char*>(&nn), 8);
    f.write(reinterpret_cast<const char*>(&dd), 8);
    f.write(reinterpret_cast<const char*>(ctx.docfree_cache.data()), static_cast<std::streamsize>(n * dim * sizeof(float)));
  }
  return ctx.docfree_cache;
}

const std::vector<float>& rows_for(Context& ctx, const QuerySet& set) {
  return set.name == "q_doc" ? docfree_rows(ctx) : corpus_rows(ctx);
}

// ---- scoring ---------------------------------------------------------------

std::vector<Scored> score_queries(Context& ctx, const QuerySet& set) {
  std::vector<Scored> out(set.queries.size());
  if (set.queries.empty()) return out;
  const auto emb = ctx.engine->embedder();
  const std::size_t n = ctx.n(), dim = ctx.dim();
  const std::vector<float>& rows = rows_for(ctx, set);
  const std::size_t batch = 16;
  std::vector<float> qv(batch * dim);
  for (std::size_t start = 0; start < set.queries.size(); start += batch) {
    const std::size_t count = std::min(batch, set.queries.size() - start);
    std::vector<std::string_view> texts;
    for (std::size_t j = 0; j < count; ++j) texts.emplace_back(set.queries[start + j].question);
    emb->embed(texts, prag::embed::Role::Query, qv.data());
    for (std::size_t j = 0; j < count; ++j) {
      Scored& s = out[start + j];
      s.q.assign(qv.begin() + static_cast<std::ptrdiff_t>(j * dim), qv.begin() + static_cast<std::ptrdiff_t>((j + 1) * dim));
      s.score.resize(n);
      double sum = 0.0;
      for (std::size_t u = 0; u < n; ++u) {
        s.score[u] = prag::index::dot_f32(s.q.data(), &rows[u * dim], dim);
        sum += static_cast<double>(s.score[u]);
      }
      s.mean = sum / D(n);
      double var = 0.0;
      for (std::size_t u = 0; u < n; ++u) {
        const double d = static_cast<double>(s.score[u]) - s.mean;
        var += d * d;
      }
      s.sd = std::sqrt(var / D(n));
      s.ok = s.sd > 0.0 && std::isfinite(s.sd);
    }
  }
  return out;
}

Labeled collect_labeled(const QuerySet& set, const std::vector<Scored>& scored, int split,
                        std::size_t negatives_per_query, std::uint64_t seed) {
  Labeled out;
  for (std::size_t i = 0; i < set.queries.size(); ++i) {
    const Query& q = set.queries[i];
    const Scored& s = scored[i];
    if (q.split != split || !s.ok) continue;
    const std::size_t n = s.score.size();
    ++out.queries;
    const double zr = s.z(q.unit);
    out.z_relevant.push_back(zr);
    out.samples.push_back({zr, true});
    if (negatives_per_query == 0 || negatives_per_query + 1 >= n) {
      for (std::size_t j = 0; j < n; ++j) {
        if (j == q.unit) continue;
        const double z = s.z(j);
        out.z_nonrelevant.push_back(z);
        out.samples.push_back({z, false});
      }
    } else {
      std::uint64_t state = prag::hash_combine(seed, i + 1);
      std::unordered_set<std::size_t> taken{q.unit};
      while (taken.size() < negatives_per_query + 1) {
        const std::size_t j = static_cast<std::size_t>(rng_below(state, n));
        if (!taken.insert(j).second) continue;
        const double z = s.z(j);
        out.z_nonrelevant.push_back(z);
        out.samples.push_back({z, false});
      }
    }
  }
  return out;
}

// ---- prompts ---------------------------------------------------------------

std::vector<prag::ChatMessage> messages_for(const Context& ctx, const std::string& question,
                                            const std::vector<UnitId>& units, bool edit_mode) {
  const prag::engine::Corpus& c = *ctx.corpus;
  prag::context::Selection selection;
  for (const UnitId id : units) {
    if (id >= c.units.size()) continue;
    prag::context::Entry e;
    e.id = id;
    e.group = c.file_of_unit[id];
    e.form = prag::context::Form::Full;
    e.p = 1.0;
    e.quality = 1.0;
    e.tokens = c.units[id].tokens;
    selection.entries.push_back(e);
  }
  prag::context::PromptOptions po;
  po.edit_mode = edit_mode;
  po.root = c.roots.empty() ? std::string() : c.roots.front();
  return prag::context::build_messages(question, c.units, selection, {}, {}, po);
}

double answer_loss(prag::llm::LanguageModel& model, const std::vector<prag::ChatMessage>& messages,
                   const std::string& answer, bool* truncated) {
  const prag::llm::Likelihood l = model.score(messages, answer);
  if (truncated) *truncated = l.truncated;
  return l.tokens > 0 ? l.nll / static_cast<double>(l.tokens) : kNaN;
}

}  // namespace bench
