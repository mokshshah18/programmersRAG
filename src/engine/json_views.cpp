// JSON views with the field names of DESIGN.md section 9.
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "engine_impl.hpp"
#include "prag/core/text.hpp"
#include "prag/exec/cache_topology.hpp"

namespace prag::engine {

using nlohmann::json;

namespace {

double finite(double x) { return std::isfinite(x) ? x : 0.0; }

const char* cache_type(exec::CacheType t) {
  switch (t) {
    case exec::CacheType::Data: return "data";
    case exec::CacheType::Instruction: return "instruction";
    case exec::CacheType::Unified: return "unified";
  }
  return "unified";
}

std::string file_stem(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  const std::size_t dot = name.rfind(".gguf");
  if (dot != std::string::npos) name.resize(dot);
  return name;
}

std::string preview_of(const SourceUnit& u) {
  std::string out;
  std::size_t lines = 0;
  for (const std::string_view line : text::split_lines(u.text)) {
    if (lines == 0 && text::trim(line).empty()) continue;
    if (++lines > 6 || out.size() + line.size() > 400) break;
    out.append(line);
    out += '\n';
  }
  return text::sanitize_utf8(out);
}

}  // namespace

json report_json(const IndexReport& r) {
  json seconds = json::object();
  for (const auto& [k, v] : r.seconds) seconds[k] = finite(v);
  return json{{"files", r.files},
              {"units", r.units},
              {"functions", r.functions},
              {"call_edges", r.call_edges},
              {"embedded", r.embedded},
              {"reused", r.reused},
              {"groups", r.groups},
              {"kappa", finite(r.kappa)},
              {"reduced_dim", r.reduced_dim},
              {"explained_variance", finite(r.explained_variance)},
              {"anisotropy_before", finite(r.anisotropy_before)},
              {"anisotropy_after", finite(r.anisotropy_after)},
              {"blocks", r.blocks},
              {"block_bytes", r.block_bytes},
              {"seconds", seconds},
              {"error", r.error}};
}

json Engine::status_json() const {
  const Impl& d = *impl_;
  std::shared_ptr<const Corpus> c;
  double kappa = 0.0;
  bool stale = false;
  std::string embedder_id;
  bool embedding_loaded = false;
  std::shared_ptr<llm::LanguageModel> chat;
  {
    std::lock_guard<std::mutex> lock(d.mu);
    c = d.corpus;
    kappa = d.kappa_cv;
    stale = d.stale;
    if (d.embedder) embedder_id = d.embedder->id();
    embedding_loaded = d.embedder_is_model;
    chat = d.chat;
  }
  json index{{"ready", c && !stale}, {"building", d.building.load()}};
  std::size_t functions = 0;
  if (c) {
    for (const SourceUnit& u : c->units) functions += u.kind == UnitKind::Function;
    index.update(json{{"files", c->files.size()},
                      {"units", c->units.size()},
                      {"functions", functions},
                      {"call_edges", c->calls.edges()},
                      {"groups", c->modes.count()},
                      {"dim", c->store.dim()},
                      {"reduced_dim", c->reduction.out_dim},
                      {"kappa", finite(kappa)},
                      {"codes", index::to_string(c->store.code_kind())},
                      {"blocks", c->store.blocks().size()},
                      {"block_bytes", c->store.block_bytes()},
                      {"embedder", c->embedder_id},
                      {"built_at", c->built_at},
                      {"calibrated", c->calibration.fitted()}});
  } else {
    index.update(json{{"files", 0},  {"units", 0},       {"functions", 0}, {"call_edges", 0}, {"groups", 0},
                      {"dim", 0},    {"reduced_dim", 0}, {"kappa", 0.0},   {"codes", d.settings.get<std::string>("index.codes", "rabitq")},
                      {"blocks", 0}, {"block_bytes", block_bytes()},       {"embedder", embedder_id},
                      {"built_at", ""}, {"calibrated", false}});
  }
  const exec::CacheTopology& t = exec::cache_topology();
  json classes = json::array();
  for (const exec::CacheClass& k : t.classes) {
    classes.push_back(json{{"level", k.level},
                           {"type", cache_type(k.type)},
                           {"size_bytes", k.size_bytes},
                           {"sharing_threads", k.sharing_threads},
                           {"instances", k.instances}});
  }
  return json{{"version", "2.0.0"},
              {"index", index},
              {"models",
               {{"llama", llm::available()},
                {"chat", chat ? file_stem(chat->id()) : std::string()},
                {"chat_loaded", chat != nullptr},
                {"embedding_loaded", embedding_loaded},
                {"backend", llm::backend_info()}}},
              {"cache",
               {{"detected", t.detected},
                {"logical_cpus", t.logical_cpus},
                {"l1d_min_bytes", t.l1d_min_bytes},
                {"l1d_per_thread_min_bytes", t.l1d_per_thread_min_bytes},
                {"line_bytes", t.line_bytes},
                {"block_bytes", block_bytes()},
                {"threads", d.executor->threads()},
                {"simd", index::simd_kernel()},
                {"classes", classes}}},
              {"tune", tune::to_json(d.tuning->status())},
              {"roots", c ? c->roots : d.current_roots()}};
}

json Engine::index_progress_json() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  const IndexProgress& p = impl_->progress;
  return json{{"running", impl_->building.load()},
              {"stage", p.stage},
              {"done", p.done},
              {"total", p.total},
              {"message", p.message},
              {"report", impl_->last_report ? report_json(*impl_->last_report) : json(nullptr)}};
}

json Engine::groups_json() const {
  const auto c = corpus();
  double kappa = 0.0;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    kappa = impl_->kappa_cv;
  }
  json out{{"kappa", finite(kappa)}, {"reduced_dim", c ? c->reduction.out_dim : 0}, {"groups", json::array()},
           {"units", json::array()}};
  if (!c || c->modes.count() == 0) return out;
  const std::size_t m = c->reduction.out_dim;
  const std::size_t n = c->units.size();
  double scale = 0.0;
  auto coord = [&](const float* x, std::size_t k) { return m > k ? static_cast<double>(x[k]) : 0.0; };
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < 2; ++k) scale = std::max(scale, std::fabs(coord(&c->reduced[i * m], k)));
  for (std::size_t g = 0; g < c->modes.count(); ++g)
    for (std::size_t k = 0; k < 2; ++k)
      scale = std::max(scale, std::fabs(coord(&c->modes.centers[g * c->modes.m], k)));
  if (!(scale > 0.0)) scale = 1.0;

  // Labels: the names of the densest functions of each group.
  std::vector<std::vector<UnitId>> members(c->modes.count());
  for (std::size_t i = 0; i < n; ++i)
    if (!c->units[i].name.empty()) members[c->modes.assignment[i]].push_back(static_cast<UnitId>(i));
  for (std::size_t g = 0; g < c->modes.count(); ++g) {
    auto& list = members[g];
    std::stable_sort(list.begin(), list.end(), [&](UnitId a, UnitId b) {
      const bool fa = c->units[a].kind == UnitKind::Function, fb = c->units[b].kind == UnitKind::Function;
      if (fa != fb) return fa;
      return c->modes.log_density[a] > c->modes.log_density[b];
    });
    std::string label;
    for (std::size_t k = 0; k < list.size() && k < 3; ++k) label += (k ? ", " : "") + c->units[list[k]].name;
    const float* ctr = &c->modes.centers[g * c->modes.m];
    out["groups"].push_back(json{{"id", g},
                                 {"size", c->modes.sizes[g]},
                                 {"log_density", finite(c->modes.center_log_density[g])},
                                 {"x", coord(ctr, 0) / scale},
                                 {"y", coord(ctr, 1) / scale},
                                 {"label", text::sanitize_utf8(label)}});
  }
  for (std::size_t i = 0; i < n; ++i) {
    const SourceUnit& u = c->units[i];
    out["units"].push_back(json{{"id", i},
                                {"group", c->modes.assignment[i]},
                                {"x", coord(&c->reduced[i * m], 0) / scale},
                                {"y", coord(&c->reduced[i * m], 1) / scale},
                                {"name", text::sanitize_utf8(u.name)},
                                {"path", display_path(u.path, c->roots)},
                                {"kind", to_string(u.kind)},
                                {"log_density", finite(c->modes.log_density[i])}});
  }
  return out;
}

json Engine::unit_json(UnitId id) const {
  const auto c = corpus();
  if (!c || id >= c->units.size()) return json{{"error", "no such unit"}};
  const SourceUnit& u = c->units[id];
  const auto callees = c->calls.callees(id);
  const auto callers = c->calls.callers(id);
  return json{{"id", id},
              {"path", display_path(u.path, c->roots)},
              {"name", text::sanitize_utf8(u.name)},
              {"kind", to_string(u.kind)},
              {"start_line", u.start_line},
              {"end_line", u.end_line},
              {"signature", text::sanitize_utf8(u.signature)},
              {"doc", text::sanitize_utf8(u.doc)},
              {"text", text::sanitize_utf8(u.text)},
              {"callees", std::vector<UnitId>(callees.begin(), callees.end())},
              {"callers", std::vector<UnitId>(callers.begin(), callers.end())},
              {"group", c->modes.assignment[id]},
              {"tokens", u.tokens}};
}

json Engine::history_json() const {
  json out = json::array();
  for (const HistoryEntry& h : history()) {
    out.push_back(json{{"id", h.id},
                       {"time", h.time},
                       {"question", text::sanitize_utf8(h.question)},
                       {"answer", text::sanitize_utf8(h.answer)},
                       {"units", h.units},
                       {"rating", h.rating}});
  }
  return out;
}

json retrieval_json(const QueryResult& r, const Corpus& c) {
  json units = json::array();
  for (const RetrievedUnit& ru : r.units) {
    const retrieve::TracedUnit& t = ru.traced;
    if (t.id >= c.units.size()) continue;
    const SourceUnit& u = c.units[t.id];
    units.push_back(json{{"id", t.id},
                         {"path", display_path(u.path, c.roots)},
                         {"name", text::sanitize_utf8(u.name)},
                         {"kind", to_string(u.kind)},
                         {"start_line", u.start_line},
                         {"end_line", u.end_line},
                         {"group", ru.group},
                         {"tokens", u.tokens},
                         {"score", finite(t.score)},
                         {"z", finite(t.z)},
                         {"p_score", finite(t.p_score)},
                         {"p", finite(t.p)},
                         {"hops", t.hops},
                         {"parent", t.parent == kNoUnit ? json(nullptr) : json(t.parent)},
                         {"via", to_string(t.via)},
                         {"in_context", ru.in_context},
                         {"form", ru.in_context ? context::to_string(ru.form) : "none"},
                         {"support", ru.support},
                         {"redundancy", finite(ru.redundancy)},
                         {"quality", finite(ru.quality)},
                         {"preview", preview_of(u)}});
  }
  const context::Selection& s = r.selection;
  json groups = json::array();
  for (const context::NodeQuality& g : s.groups) {
    groups.push_back(json{{"path", g.group < c.files.size() ? display_path(c.files[g.group], c.roots) : ""},
                          {"tokens", g.tokens},
                          {"quality", finite(g.quality)},
                          {"units", g.units}});
  }
  return json{{"id", r.id},
              {"tau", r.tau},
              {"z_threshold", finite(r.trace.z_threshold)},
              {"low_confidence", r.trace.low_confidence},
              {"truncated", r.trace.truncated},
              {"units", units},
              {"context",
               {{"tokens_before", s.tokens_before},
                {"tokens_after", s.tokens_after},
                {"quality_before", finite(s.quality_before)},
                {"quality_after", finite(s.quality_after)},
                {"rot_before", finite(s.rot_before)},
                {"rot_after", finite(s.rot_after)},
                {"compression", finite(s.compression)},
                {"passes", s.passes},
                {"dropped_low_quality", s.dropped_low_quality},
                {"dropped_redundant", s.dropped_redundant},
                {"dropped_budget", s.dropped_budget},
                {"skeletonised", s.skeletonised},
                {"support_added", s.support_added},
                {"history_kept", r.history_kept},
                {"history_total", r.history_total},
                {"groups", groups}}},
              {"scan",
               {{"blocks", r.scan.blocks},
                {"block_bytes", r.scan.block_bytes},
                {"threads", r.scan.threads},
                {"wall_us", r.scan.wall_us},
                {"work_us", r.scan.work_us},
                {"gate_wait_us", r.scan.gate_wait_us},
                {"admission_violations", r.scan.admission_violations},
                {"commit_violations", r.scan.commit_violations},
                {"rescored", r.scan.rescored},
                {"codes", r.scan.codes},
                {"simd", r.scan.simd}}},
              {"trace", {{"seeds", r.trace.seeds}, {"evaluated", r.trace.evaluated}, {"expansions", r.trace.expansions}}}};
}

json done_json(const QueryResult& r) {
  return json{{"id", r.id},
              {"answer", text::sanitize_utf8(r.answer)},
              {"prompt_tokens", r.generation.prompt_tokens},
              {"completion_tokens", r.generation.completion_tokens},
              {"prompt_ms", r.generation.prompt_ms},
              {"completion_ms", r.generation.completion_ms},
              {"total_ms", r.total_ms},
              {"stop_reason", r.generation.stop_reason},
              {"model", r.extractive ? r.model : file_stem(r.model)},
              {"extractive", r.extractive}};
}

json transaction_json(const edit::Transaction& tx, bool committed, const std::vector<std::string>& roots,
                      const std::vector<std::string>& warnings) {
  json files = json::array();
  for (const edit::FileProposal& f : tx.files) {
    json matches = json::array();
    for (const edit::Match& m : f.matches) {
      matches.push_back(json{{"kind", edit::to_string(m.kind)},
                             {"first_line", m.first_line},
                             {"last_line", m.last_line},
                             {"distance", m.distance},
                             {"candidates", m.candidates}});
    }
    files.push_back(json{{"path", f.path},
                         {"display_path", f.display_path.empty() ? display_path(f.path, roots) : f.display_path},
                         {"status", edit::to_string(f.status)},
                         {"creates_file", f.creates_file},
                         {"diff", text::sanitize_utf8(f.diff)},
                         {"matches", matches},
                         {"syntax_errors_before", f.syntax_errors_before},
                         {"syntax_errors_after", f.syntax_errors_after},
                         {"message", text::sanitize_utf8(f.message)}});
  }
  return json{{"id", tx.id},
              {"status", edit::to_string(tx.status)},
              {"committed", committed},
              {"files", files},
              {"warnings", warnings}};
}

}  // namespace prag::engine
