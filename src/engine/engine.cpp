// Engine: settings, models, queries, history, edits and tuning (DESIGN.md 6.13).
#include "prag/engine/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "engine_impl.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/core/text.hpp"
#include "prag/embed/hashing.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/parse/parser.hpp"

namespace prag::engine {

using nlohmann::json;

// ---- Impl helpers ----------------------------------------------------------

std::string Engine::Impl::data_dir() const {
  return detail::resolve_path(base_dir, settings.get<std::string>("paths.data", "data"));
}

std::size_t Engine::Impl::block_bytes() const {
  const auto b = settings.get<std::size_t>("exec.block_bytes", 0);
  return b != 0 ? b : exec::recommended_block_bytes(exec::cache_topology());
}

std::shared_ptr<embed::Embedder> Engine::Impl::ensure_embedder() {
  {
    std::lock_guard<std::mutex> lock(mu);
    if (embedder) return embedder;
  }
  std::shared_ptr<embed::Embedder> e;
  bool is_model = false;
  const std::string path = settings.get<std::string>("models.embedding", "");
  if (!path.empty()) {
    llm::EmbedderOptions eo;
    eo.path = detail::resolve_path(base_dir, path);
    eo.context_tokens = settings.get<int>("models.embedding_context", 2048);
    eo.threads = settings.get<int>("models.threads", 0);
    eo.query_prefix = settings.get<std::string>(
        "models.embedding_query_prefix",
        "Instruct: Given a question about a C code base, retrieve the code that answers it\nQuery: ");
    std::string error;
    e = llm::load_embedder(eo, &error);
    if (e)
      is_model = true;
    else
      PRAG_WARN("embedding model not loaded (%s); using the hashing embedder", error.c_str());
  }
  if (!e) e = embed::make_hashing_embedder();
  std::lock_guard<std::mutex> lock(mu);
  if (!embedder) {
    embedder = std::move(e);
    embedder_is_model = is_model;
  }
  return embedder;
}

std::shared_ptr<llm::LanguageModel> Engine::Impl::chat_model() const {
  std::lock_guard<std::mutex> lock(mu);
  return chat;
}

std::shared_ptr<const Corpus> Engine::Impl::snapshot() const {
  std::lock_guard<std::mutex> lock(mu);
  return corpus;
}

std::vector<float> Engine::Impl::embed_texts(embed::Embedder& e, std::span<const std::string_view> texts,
                                             embed::Role role) {
  std::vector<float> out(texts.size() * e.dim());
  if (texts.empty()) return out;
  std::lock_guard<std::mutex> lock(embed_mu);
  e.embed(texts, role, out.data());
  return out;
}

std::vector<float> Engine::Impl::embed_question(embed::Embedder& e, const std::string& question) {
  std::lock_guard<std::mutex> lock(embed_mu);
  if (question_cache_id != e.id()) {
    question_cache.clear();
    question_cache_id = e.id();
  }
  const auto it = question_cache.find(question);
  if (it != question_cache.end()) return it->second;
  std::vector<float> v = e.embed_one(question, embed::Role::Query);
  if (question_cache.size() > 4096) question_cache.clear();
  question_cache.emplace(question, v);
  return v;
}

retrieve::EdgeLift Engine::Impl::default_lifts() const {
  retrieve::EdgeLift l;
  l.callee = settings.get<double>("retrieve.lift.callee", l.callee);
  l.caller = settings.get<double>("retrieve.lift.caller", l.caller);
  l.sibling = settings.get<double>("retrieve.lift.sibling", l.sibling);
  return l;
}

edit::ApplyOptions Engine::Impl::apply_options(const std::vector<std::string>& roots) const {
  edit::ApplyOptions ao;
  ao.allowed_roots = roots;
  ao.match.allow_fuzzy = settings.get<bool>("edit.allow_fuzzy", false);
  ao.backup_dir = edits_dir();
  if (settings.get<bool>("edit.validate_syntax", true)) {
    ao.syntax_check = [](std::string_view path, std::string_view text) {
      return parse::count_syntax_errors(parse::detect_language(path, text), text);
    };
  }
  ao.unit_locator = [](std::string_view path, std::string_view text, std::string_view name, std::size_t* b,
                       std::size_t* e) { return parse::find_unit(path, text, name, b, e); };
  return ao;
}

static json history_entry_json(const HistoryEntry& h) {
  return json{{"id", h.id},       {"time", h.time},   {"question", h.question},
              {"answer", h.answer}, {"units", h.units}, {"rating", h.rating}};
}

void Engine::Impl::load_history() {
  std::lock_guard<std::mutex> lock(mu);
  history.clear();
  const auto text = fs::read_file(history_path());
  if (!text) return;
  for (const std::string_view line : text::split_lines(*text)) {
    if (line.empty()) continue;
    const json j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.is_object()) continue;
    HistoryEntry h;
    h.id = j.value("id", "");
    h.time = j.value("time", "");
    h.question = j.value("question", "");
    h.answer = j.value("answer", "");
    h.units = j.value("units", std::vector<UnitId>{});
    h.rating = j.value("rating", 0);
    unsigned long long number = 0;
    if (std::sscanf(h.id.c_str(), "q-%llu", &number) == 1 && number >= next_query) next_query = number + 1;
    history.push_back(std::move(h));
  }
}

bool Engine::Impl::save_history_locked() const {
  std::string out;
  for (const HistoryEntry& h : history) {
    out += history_entry_json(h).dump(-1, ' ', false, json::error_handler_t::replace);
    out += '\n';
  }
  fs::create_directories(data_dir());
  return fs::write_file_atomic(history_path(), out);
}

void Engine::Impl::make_tuning() {
  tune::Hooks hooks;
  hooks.corpus = [this](tune::CorpusView* view) {
    auto snap = snapshot();
    if (!snap) return false;
    {
      std::lock_guard<std::mutex> lock(mu);
      tune_snapshot = snap;  // keeps the spans of the view alive
    }
    const Corpus* c = snap.get();
    view->units = c->units;
    view->distractors = [c](UnitId unit, std::size_t count) {
      std::vector<UnitId> out;
      for (const UnitId v : c->siblings.of(unit)) {
        if (out.size() >= count) break;
        out.push_back(v);
      }
      return out;
    };
    view->prompt = [c](std::string_view question, std::span<const UnitId> ids) {
      return detail::prompt_for(*c, question, ids, false);
    };
    return true;
  };
  hooks.serving_model = [this] {
    std::lock_guard<std::mutex> lock(mu);
    return chat_path;
  };
  hooks.score_examples = [this](const std::string& model_path, std::span<const tune::Example> examples,
                                std::vector<double>* nll, std::string* error) {
    auto snap = snapshot();
    if (!snap) {
      if (error) *error = "no index";
      return false;
    }
    std::shared_ptr<llm::LanguageModel> model;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (model_path == chat_path) model = chat;
    }
    if (!model) {
      llm::ModelOptions mo;
      mo.path = model_path;
      mo.context_tokens = settings.get<int>("models.chat_context", 4096);
      mo.threads = settings.get<int>("models.threads", 0);
      model = llm::load_model(mo, error);
      if (!model) return false;
    }
    nll->clear();
    for (const tune::Example& ex : examples) {
      const auto messages = detail::prompt_for(*snap, ex.question, ex.context, false);
      const llm::Likelihood l = model->score(messages, ex.answer);
      nll->push_back(l.tokens > 0 ? l.nll / l.tokens : std::nan(""));
    }
    return true;
  };
  hooks.promote = [this](const std::string& model_path, std::string* error) {
    llm::ModelOptions mo;
    mo.path = model_path;
    mo.context_tokens = settings.get<int>("models.chat_context", 4096);
    mo.threads = settings.get<int>("models.threads", 0);
    std::shared_ptr<llm::LanguageModel> m = llm::load_model(mo, error);
    if (!m) return false;
    std::lock_guard<std::mutex> lock(mu);
    chat = std::move(m);
    chat_path = model_path;
    return true;
  };
  hooks.feedback_examples = [this] {
    std::vector<tune::Example> out;
    std::lock_guard<std::mutex> lock(mu);
    for (const HistoryEntry& h : history) {
      if (h.rating <= 0 || h.units.empty() || h.answer.empty()) continue;
      tune::Example ex;
      ex.id = "fb-" + h.id;
      ex.question = h.question;
      ex.context = h.units;
      ex.oracle = h.units.front();
      ex.answer = h.answer;
      ex.source = "feedback";
      out.push_back(std::move(ex));
    }
    return out;
  };
  tuning = std::make_unique<tune::Pipeline>(settings, data_dir() + "/tune", std::move(hooks));
}

// ---- Engine ----------------------------------------------------------------

Engine::Engine(std::string settings_path) : impl_(std::make_unique<Impl>()) {
  impl_->settings_path = fs::normalize(settings_path);
  const std::size_t slash = impl_->settings_path.find_last_of('/');
  impl_->base_dir = slash == std::string::npos ? std::string(".") : impl_->settings_path.substr(0, slash);
  if (!impl_->settings.load(impl_->settings_path))
    PRAG_WARN("settings file %s is not valid JSON; using defaults", impl_->settings_path.c_str());
  impl_->executor = std::make_unique<exec::OrderedExecutor>(impl_->settings.get<unsigned>("exec.threads", 0));
  impl_->make_tuning();
}

Engine::~Engine() {
  if (impl_->tuning) impl_->tuning->stop();
  // Wait for a build running on another thread.
  std::lock_guard<std::mutex> lock(impl_->build_mu);
}

Settings& Engine::settings() { return impl_->settings; }
const Settings& Engine::settings() const { return impl_->settings; }

bool Engine::save_settings() const { return impl_->settings.save(impl_->settings_path); }

void Engine::update_settings(const nlohmann::json& patch) {
  const std::uint64_t before = detail::index_settings_hash(impl_->settings);
  const std::string embedding_before = impl_->settings.get<std::string>("models.embedding", "");
  impl_->settings.merge(patch);
  save_settings();
  const bool changed = detail::index_settings_hash(impl_->settings) != before;
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (changed) impl_->stale = true;
  if (impl_->settings.get<std::string>("models.embedding", "") != embedding_before) impl_->embedder.reset();
}

void Engine::open() {
  Impl& d = *impl_;
  d.load_history();
  edit::recover(d.edits_dir());
  const auto embedder = d.ensure_embedder();
  const std::string chat = d.settings.get<std::string>("models.chat", "");
  if (!chat.empty() && !d.chat_model()) {
    std::string error;
    if (!load_chat_model(chat, &error)) PRAG_WARN("chat model not loaded: %s", error.c_str());
  }
  detail::Persisted p;
  std::string error;
  if (detail::load_corpus(d.index_dir(), embedder->id(), &p, &error)) {
    std::lock_guard<std::mutex> lock(d.mu);
    d.corpus = p.corpus;
    d.kappa_cv = p.kappa_cv;
    d.stale = p.settings_hash != detail::index_settings_hash(d.settings);
    if (p.report && !d.last_report) d.last_report = p.report;
  } else {
    PRAG_INFO("index not loaded: %s", error.c_str());
  }
  d.tuning->start();
}

exec::OrderedExecutor& Engine::executor() { return *impl_->executor; }
std::size_t Engine::block_bytes() const { return impl_->block_bytes(); }

std::shared_ptr<embed::Embedder> Engine::embedder() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->embedder;
}

std::shared_ptr<llm::LanguageModel> Engine::chat_model() const { return impl_->chat_model(); }

bool Engine::load_chat_model(const std::string& path, std::string* error) {
  llm::ModelOptions mo;
  mo.path = detail::resolve_path(impl_->base_dir, path);
  mo.context_tokens = impl_->settings.get<int>("models.chat_context", 4096);
  mo.threads = impl_->settings.get<int>("models.threads", 0);
  std::shared_ptr<llm::LanguageModel> m = llm::load_model(mo, error);
  if (!m) return false;
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->chat = std::move(m);
  impl_->chat_path = mo.path;
  return true;
}

std::shared_ptr<const Corpus> Engine::corpus() const { return impl_->snapshot(); }

bool Engine::index_ready() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->corpus && !impl_->stale;
}

bool Engine::index_building() const { return impl_->building.load(); }

IndexReport Engine::build_index(const IndexProgressCallback& progress) {
  impl_->tuning->notify_activity();
  return impl_->build(progress);
}

IndexProgress Engine::index_progress() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->progress;
}

std::optional<IndexReport> Engine::last_index_report() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->last_report;
}

bool Engine::calibrate(const CalibrationOptions& options, const IndexProgressCallback& progress,
                       std::string* error) {
  Impl& d = *impl_;
  d.tuning->notify_activity();
  std::lock_guard<std::mutex> build_lock(d.build_mu);
  const auto snap = d.snapshot();
  if (!snap) {
    if (error) *error = "no index";
    return false;
  }
  retrieve::Calibration cal;
  retrieve::EdgeLift lifts = d.default_lifts();
  CalibrationInfo info;
  if (!d.calibrate_corpus(*snap, options, progress, d.ensure_embedder(), d.chat_model(), &cal, &lifts, &info,
                          error))
    return false;
  if (!detail::save_calibration(d.index_dir(), cal, lifts, info)) {
    if (error) *error = "cannot write calibration.json";
    return false;
  }
  detail::Persisted p;
  if (!detail::load_corpus(d.index_dir(), snap->embedder_id, &p, error)) return false;
  std::lock_guard<std::mutex> lock(d.mu);
  if (d.corpus == snap) d.corpus = p.corpus;
  return true;
}

// ---- query -----------------------------------------------------------------

namespace {

std::string extractive_answer(const Corpus& c, const context::Selection& sel) {
  std::vector<const context::Entry*> entries;
  for (const auto& e : sel.entries) entries.push_back(&e);
  std::stable_sort(entries.begin(), entries.end(), [](const auto* a, const auto* b) { return a->p > b->p; });
  std::string out = "No language model is loaded; these are the most relevant parts of the code:\n";
  std::size_t k = 0;
  for (const context::Entry* e : entries) {
    const SourceUnit& u = c.units[e->id];
    char head[96];
    std::snprintf(head, sizeof head, "\n%zu. ", ++k);
    out += head;
    out += "`" + (u.name.empty() ? std::string(to_string(u.kind)) : u.name) + "` (" + to_string(u.kind) + ") at " +
           display_path(u.path, c.roots) + ":" + std::to_string(u.start_line) + "-" + std::to_string(u.end_line);
    char p[48];
    std::snprintf(p, sizeof p, ", p = %.2f\n", e->p);
    out += p;
    if (!u.signature.empty()) out += "   " + std::string(text::trim(u.signature)) + "\n";
    std::size_t lines = 0;
    for (const std::string_view line : text::split_lines(u.doc)) {
      if (text::trim(line).empty()) continue;
      if (++lines > 4) break;
      out += "   " + std::string(text::trim(line)) + "\n";
    }
  }
  return text::sanitize_utf8(out);
}

}  // namespace

QueryResult Engine::query(std::string_view question, const QueryOptions& options,
                          const RetrievalCallback& on_retrieval, const llm::PieceCallback& on_piece) {
  Impl& d = *impl_;
  Stopwatch total;
  d.tuning->notify_activity();
  QueryResult r;
  r.question = std::string(question);
  {
    std::lock_guard<std::mutex> lock(d.mu);
    char id[32];
    std::snprintf(id, sizeof id, "q-%06llu", static_cast<unsigned long long>(d.next_query++));
    r.id = id;
  }
  const std::shared_ptr<const Corpus> snap = d.snapshot();
  if (!snap) {
    r.error = "no index: build the index first";
    return r;
  }
  const Corpus& c = *snap;
  const auto embedder = d.ensure_embedder();
  if (embedder->id() != c.embedder_id) {
    r.error = "the index was built with a different embedder; rebuild the index";
    return r;
  }
  const std::size_t n = c.units.size();
  const std::size_t dim = c.store.dim();
  const Settings& s = d.settings;
  r.tau = options.tau > 0.0 ? options.tau : s.get<double>("retrieve.tau", 0.25);
  const auto chat = d.chat_model();

  // 1. embed
  Stopwatch sw;
  const std::vector<float> q = d.embed_question(*embedder, r.question);
  r.embed_ms = sw.elapsed_ms();
  sw.reset();

  // 2. scan
  index::ScanResult sr = index::scan(c.store, q, *d.executor);
  if (sr.scores.size() != n) {
    r.error = "scan failed";
    return r;
  }
  const double mean = sr.mean;
  const double sd = sr.stddev;
  const retrieve::Calibration& cal = c.calibration;
  std::vector<char> exact(n, 0);
  std::size_t rescored = 0;
  auto rescore = [&](std::vector<UnitId>& ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](UnitId v) { return v >= n || exact[v]; }), ids.end());
    if (ids.empty()) return false;
    for (const UnitId v : ids) exact[v] = 1;
    index::rescore(c.store, q, ids, sr);
    rescored += ids.size();
    return true;
  };

  // 3. rescore every unit whose interval reaches the score at which the posterior is tau
  {
    std::vector<UnitId> ids;
    const double zt = cal.threshold(r.tau);
    if (sd > 0.0 && std::isfinite(sd) && zt != std::numeric_limits<double>::infinity()) {
      const double st = std::isfinite(zt) ? mean + zt * sd : -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < n; ++j)
        if (static_cast<double>(sr.scores[j]) + static_cast<double>(sr.error[j]) >= st) ids.push_back(static_cast<UnitId>(j));
    }
    rescore(ids);
  }

  // 4. trace, rescoring every unit the trace looked at until all decisions rest on exact scores
  retrieve::TraceOptions to;
  to.tau = r.tau;
  to.max_units = s.get<std::size_t>("retrieve.max_units", 64);
  to.max_hops = s.get<std::size_t>("retrieve.max_hops", 6);
  to.lift = c.lifts;
  retrieve::TraceResult tr;
  for (int round = 0; round < 64; ++round) {
    tr = retrieve::trace(sr.scores, mean, sd, cal, c.calls, c.siblings, to);
    std::vector<UnitId> ids;
    for (const retrieve::TracedUnit& t : tr.units) {
      ids.push_back(t.id);
      for (const UnitId v : c.calls.callees(t.id)) ids.push_back(v);
      for (const UnitId v : c.calls.callers(t.id)) ids.push_back(v);
      std::size_t k = 0;
      for (const UnitId v : c.siblings.of(t.id)) {
        if (k++ >= to.sibling_fanout) break;
        ids.push_back(v);
      }
    }
    if (tr.low_confidence && !tr.units.empty()) {
      // The best unit must be the best by exact score.
      const float best = sr.scores[tr.units.front().id];
      for (std::size_t j = 0; j < n; ++j)
        if (!exact[j] && sr.scores[j] + sr.error[j] >= best) ids.push_back(static_cast<UnitId>(j));
    }
    if (!rescore(ids)) break;
  }

  // 5. context selection
  std::unordered_map<UnitId, std::size_t> traced_index;
  for (std::size_t i = 0; i < tr.units.size(); ++i) traced_index.emplace(tr.units[i].id, i);
  auto count_tokens = [&](const std::string& t) -> std::uint32_t {
    return chat ? static_cast<std::uint32_t>(std::max(0, chat->count_tokens(t))) : text::estimate_tokens(t);
  };
  std::vector<context::Candidate> candidates;
  std::unordered_set<UnitId> in_candidates;
  auto add_candidate = [&](UnitId id, double p) {
    if (!in_candidates.insert(id).second) return;
    const SourceUnit& u = c.units[id];
    context::Candidate cand;
    cand.id = id;
    cand.group = c.file_of_unit[id];
    cand.p = p;
    cand.tokens_full = chat ? count_tokens(u.text) : u.tokens;
    cand.tokens_skeleton = count_tokens(context::skeleton_of(u));
    cand.embedding = c.store.vector(id).data();
    const auto callees = c.calls.callees(id);
    cand.requires_units.assign(callees.begin(), callees.end());
    candidates.push_back(std::move(cand));
  };
  for (const retrieve::TracedUnit& t : tr.units) add_candidate(t.id, t.p);
  std::unordered_map<UnitId, UnitId> callee_parent;
  for (const retrieve::TracedUnit& t : tr.units) {
    for (const UnitId v : c.calls.callees(t.id)) {
      if (traced_index.count(v) || in_candidates.count(v)) continue;
      const double z = sd > 0.0 ? (static_cast<double>(sr.scores[v]) - mean) / sd : 0.0;
      callee_parent.emplace(v, t.id);
      add_candidate(v, cal.posterior(z));
    }
  }
  context::Options co;
  co.theta = s.get<double>("context.theta", 0.25);
  co.token_budget = s.get<std::uint32_t>("context.token_budget", 2400);
  co.full_text_share = s.get<double>("context.full_text_share", 0.5);
  co.unit_overhead = s.get<std::uint32_t>("context.unit_overhead", 25);
  co.group_overhead = s.get<std::uint32_t>("context.group_overhead", 8);
  co.dim = dim;
  r.selection = context::select(candidates, co);

  // 6. history gate and messages
  std::vector<HistoryEntry> past;
  if (options.use_history) {
    std::lock_guard<std::mutex> lock(d.mu);
    past = d.history;
  }
  std::vector<context::Turn> turns;
  std::vector<double> posterior;
  for (const HistoryEntry& h : past) {
    if (h.answer.empty()) continue;
    context::Turn t;
    t.question = h.question;
    t.answer = h.answer;
    turns.push_back(std::move(t));
    const std::vector<float> pv = d.embed_question(*embedder, h.question);
    const double cosine = index::dot_f32(q.data(), pv.data(), dim);
    posterior.push_back(cal.posterior(sd > 0.0 ? (cosine - mean) / sd : 0.0));
  }
  context::HistoryOptions ho;
  ho.theta = co.theta;
  ho.token_budget = s.get<std::uint32_t>("context.history_budget", 400);
  const std::vector<std::size_t> kept = context::gate_history(turns, posterior, ho);
  r.history_kept = kept.size();
  r.history_total = turns.size();
  context::PromptOptions po;
  po.edit_mode = options.edit;
  po.root = c.roots.empty() ? std::string() : c.roots.front();
  r.messages = context::build_messages(r.question, c.units, r.selection, turns, kept, po);
  const int max_tokens = s.get<int>("generate.max_tokens", 512);
  if (chat && options.generate) {
    for (int attempt = 0; attempt < 4; ++attempt) {
      long long need = max_tokens;
      for (const ChatMessage& m : r.messages) need += chat->count_tokens(m.content) + 8;
      const long long excess = need - chat->context_tokens();
      if (excess <= 0 || co.token_budget <= 128) break;
      co.token_budget = static_cast<std::uint32_t>(std::max<long long>(128, co.token_budget - excess - 16));
      r.selection = context::select(candidates, co);
      r.messages = context::build_messages(r.question, c.units, r.selection, turns, kept, po);
    }
  }

  // Retrieved units: traced first, then the other entries of the selection.
  std::unordered_map<UnitId, const context::Entry*> entry_of;
  for (const context::Entry& e : r.selection.entries) entry_of.emplace(e.id, &e);
  auto fill = [&](RetrievedUnit& ru) {
    ru.group = c.modes.assignment[ru.traced.id];
    const auto it = entry_of.find(ru.traced.id);
    if (it != entry_of.end()) {
      ru.in_context = true;
      ru.form = it->second->form;
      ru.support = it->second->support;
      ru.redundancy = it->second->redundancy;
      ru.quality = it->second->quality;
    }
  };
  for (const retrieve::TracedUnit& t : tr.units) {
    RetrievedUnit ru;
    ru.traced = t;
    fill(ru);
    r.units.push_back(ru);
  }
  for (const context::Entry& e : r.selection.entries) {
    if (traced_index.count(e.id)) continue;
    RetrievedUnit ru;
    ru.traced.id = e.id;
    ru.traced.score = sr.scores[e.id];
    ru.traced.z = static_cast<float>(sd > 0.0 ? (static_cast<double>(sr.scores[e.id]) - mean) / sd : 0.0);
    ru.traced.p_score = static_cast<float>(cal.posterior(ru.traced.z));
    ru.traced.p = static_cast<float>(e.p);
    const auto parent = callee_parent.find(e.id);
    if (parent != callee_parent.end()) {
      ru.traced.parent = parent->second;
      ru.traced.via = EdgeKind::Callee;
      ru.traced.hops = static_cast<std::uint16_t>(tr.units[traced_index[parent->second]].hops + 1);
    }
    fill(ru);
    r.units.push_back(ru);
  }
  r.trace = tr;
  r.scan.blocks = sr.stats.blocks;
  r.scan.block_bytes = c.store.block_bytes();
  r.scan.threads = sr.stats.threads;
  r.scan.wall_us = sr.stats.wall_ns / 1000;
  r.scan.work_us = sr.stats.work_ns / 1000;
  r.scan.gate_wait_us = sr.stats.gate_wait_ns / 1000;
  r.scan.admission_violations = sr.stats.admission_violations;
  r.scan.commit_violations = sr.stats.commit_violations;
  r.scan.rescored = rescored;
  r.scan.codes = index::to_string(c.store.code_kind());
  r.scan.simd = index::simd_kernel();
  r.retrieve_ms = sw.elapsed_ms();
  if (on_retrieval) on_retrieval(r);

  // 7. generate
  if (options.generate) {
    if (chat) {
      llm::GenerateOptions go;
      go.max_tokens = max_tokens;
      go.temperature = s.get<float>("generate.temperature", 0.2f);
      if (options.edit && s.get<bool>("edit.use_grammar", false)) go.grammar = std::string(edit::edit_grammar());
      r.generation = chat->generate(r.messages, go, [&](std::string_view piece) {
        r.answer += piece;
        return on_piece ? on_piece(piece) : true;
      });
      r.model = chat->id();
      if (r.generation.stop_reason == "error") r.error = r.generation.error;
    } else {
      r.answer = extractive_answer(c, r.selection);
      r.extractive = true;
      r.model = "extractive";
      r.generation.stop_reason = "eos";
      if (on_piece) on_piece(r.answer);
    }
  }

  // 8. edit blocks
  if (options.edit && !r.answer.empty()) {
    const edit::ParseResult pr = edit::parse_edit_blocks(r.answer);
    r.edit_warnings = pr.warnings;
    if (!pr.blocks.empty()) {
      edit::Transaction tx = edit::prepare(pr.blocks, d.apply_options(c.roots));
      std::lock_guard<std::mutex> lock(d.mu);
      d.transactions[tx.id] = tx;
      r.transaction = std::move(tx);
    }
  }

  // 9. history
  if (options.record) {
    HistoryEntry h;
    h.id = r.id;
    h.time = detail::iso_now();
    h.question = r.question;
    h.answer = r.answer;
    for (const retrieve::TracedUnit& t : tr.units) h.units.push_back(t.id);
    std::lock_guard<std::mutex> lock(d.mu);
    d.history.push_back(std::move(h));
    if (!d.save_history_locked()) PRAG_WARN("cannot write %s", d.history_path().c_str());
  }
  r.total_ms = total.elapsed_ms();
  return r;
}

// ---- history ---------------------------------------------------------------

std::vector<HistoryEntry> Engine::history() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->history;
}

bool Engine::clear_history() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->history.clear();
  return impl_->save_history_locked();
}

bool Engine::rate(const std::string& id, int rating) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  for (HistoryEntry& h : impl_->history) {
    if (h.id != id) continue;
    h.rating = rating > 0 ? 1 : (rating < 0 ? -1 : 0);
    return impl_->save_history_locked();
  }
  return false;
}

// ---- edits -----------------------------------------------------------------

std::vector<std::string> Engine::Impl::current_roots() const {
  const auto snap = snapshot();
  return snap ? snap->roots : detail::corpus_roots(settings, base_dir);
}

edit::Transaction Engine::preview_edit(std::string_view model_output, std::vector<std::string>* warnings) {
  const edit::ParseResult pr = edit::parse_edit_blocks(model_output);
  if (warnings) *warnings = pr.warnings;
  return preview_edit(pr.blocks);
}

edit::Transaction Engine::preview_edit(const std::vector<edit::EditBlock>& blocks) {
  impl_->tuning->notify_activity();
  edit::Transaction tx;
  if (blocks.empty()) {
    tx.status = edit::Status::NotFound;
    tx.message = "no edit blocks";
    return tx;
  }
  tx = edit::prepare(blocks, impl_->apply_options(impl_->current_roots()));
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->transactions[tx.id] = tx;
  return tx;
}

edit::Status Engine::apply_edit(const std::string& id, edit::Transaction* result) {
  Impl& d = *impl_;
  std::lock_guard<std::mutex> edit_lock(d.edit_mu);
  d.tuning->notify_activity();
  edit::Transaction tx;
  {
    std::lock_guard<std::mutex> lock(d.mu);
    const auto it = d.transactions.find(id);
    if (it == d.transactions.end()) return edit::Status::NotFound;
    if (d.committed.count(id)) {
      if (result) *result = it->second;
      return edit::Status::Ok;
    }
    tx = it->second;
  }
  if (tx.status != edit::Status::Ok) {
    if (result) *result = tx;
    return tx.status;
  }
  fs::create_directories(d.edits_dir());
  const edit::Status st = edit::commit(tx, d.apply_options(d.current_roots()));
  {
    std::lock_guard<std::mutex> lock(d.mu);
    d.transactions[id] = tx;
    if (st == edit::Status::Ok) d.committed.insert(id);
  }
  if (result) *result = tx;
  if (st == edit::Status::Ok) d.build({});
  return st;
}

edit::Status Engine::revert_edit(const std::string& id, edit::Transaction* result) {
  Impl& d = *impl_;
  std::lock_guard<std::mutex> edit_lock(d.edit_mu);
  d.tuning->notify_activity();
  edit::Transaction tx;
  {
    std::lock_guard<std::mutex> lock(d.mu);
    const auto it = d.transactions.find(id);
    if (it == d.transactions.end()) return edit::Status::NotFound;
    tx = it->second;
    if (!d.committed.count(id)) {
      if (result) *result = tx;
      return edit::Status::Ok;
    }
  }
  const edit::Status st = edit::revert(tx);
  if (st == edit::Status::Ok) {
    std::lock_guard<std::mutex> lock(d.mu);
    d.committed.erase(id);
  }
  if (result) *result = tx;
  if (st == edit::Status::Ok) d.build({});
  return st;
}

bool Engine::transaction_committed(const std::string& id) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->committed.count(id) != 0;
}

tune::Pipeline& Engine::tuning() { return *impl_->tuning; }

// ---- display path ----------------------------------------------------------

std::string display_path(std::string_view path, const std::vector<std::string>& roots) {
  const std::string p = fs::normalize(path);
  for (const std::string& root : roots) {
    const std::string r = fs::normalize(root);
    if (!fs::is_within(p, r)) continue;
    if (p.size() == r.size()) {
      const std::size_t slash = p.find_last_of('/');
      return slash == std::string::npos ? p : p.substr(slash + 1);
    }
    std::size_t cut = r.size();
    if (cut < p.size() && p[cut] == '/') ++cut;
    else if (!r.empty() && r.back() != '/') continue;
    return p.substr(cut);
  }
  return std::string(path);
}

}  // namespace prag::engine
