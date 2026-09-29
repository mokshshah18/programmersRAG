// Helpers and the on-disk form of a corpus snapshot (DESIGN.md section 5).
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <span>

#include "engine_impl.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/core/text.hpp"
#include "prag/parse/parser.hpp"

namespace prag::engine::detail {

using nlohmann::json;

std::string iso_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

static bool is_absolute(const std::string& p) {
  if (p.empty()) return false;
  if (p[0] == '/' || p[0] == '\\') return true;
  return p.size() > 1 && p[1] == ':';
}

std::string resolve_path(const std::string& base_dir, const std::string& p) {
  if (p.empty()) return p;
  return fs::normalize(is_absolute(p) ? p : base_dir + "/" + p);
}

std::vector<std::string> corpus_roots(const Settings& settings, const std::string& base_dir) {
  std::vector<std::string> roots;
  for (const std::string& r : settings.get<std::vector<std::string>>("corpus.roots", {})) {
    if (!r.empty()) roots.push_back(resolve_path(base_dir, r));
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

std::uint64_t index_settings_hash(const Settings& settings) {
  const json doc = settings.snapshot();
  json key = json::object();
  for (const char* section : {"corpus", "kde", "index"}) {
    if (doc.contains(section)) key[section] = doc[section];
  }
  key["embedding"] = settings.get<std::string>("models.embedding", "");
  key["embedding_query_prefix"] = settings.get<std::string>("models.embedding_query_prefix", "");
  return hash64(key.dump());
}

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

std::uint64_t next_random(std::uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  return mix64(state);
}

void shuffle_ids(std::vector<UnitId>& ids, std::uint64_t seed) {
  std::uint64_t state = mix64(seed);
  for (std::size_t i = ids.size(); i > 1; --i) {
    const std::size_t j = static_cast<std::size_t>(next_random(state) % i);
    std::swap(ids[i - 1], ids[j]);
  }
}

// ---- binary helpers --------------------------------------------------------

namespace {

template <class T>
void put(std::string& out, const T& v) {
  out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

struct Reader {
  std::string_view data;
  std::size_t pos = 0;
  template <class T>
  bool get(T* v) {
    if (data.size() - pos < sizeof(T)) return false;
    std::memcpy(v, data.data() + pos, sizeof(T));
    pos += sizeof(T);
    return true;
  }
  bool get_bytes(void* dst, std::size_t n) {
    if (data.size() - pos < n) return false;
    std::memcpy(dst, data.data() + pos, n);
    pos += n;
    return true;
  }
};

constexpr char kCacheMagic[8] = {'P', 'R', 'A', 'G', 'E', 'M', 'B', '1'};
constexpr char kGraphMagic[8] = {'P', 'R', 'A', 'G', 'G', 'R', 'F', '1'};

void put_csr(std::string& out, const graph::NeighborGraph& g) {
  put<std::uint64_t>(out, g.offsets().size());
  put<std::uint64_t>(out, g.targets().size());
  out.append(reinterpret_cast<const char*>(g.offsets().data()), g.offsets().size() * sizeof(std::uint32_t));
  out.append(reinterpret_cast<const char*>(g.targets().data()), g.targets().size() * sizeof(UnitId));
}

bool get_csr(Reader& r, std::vector<std::uint32_t>* offsets, std::vector<UnitId>* targets) {
  std::uint64_t no = 0, nt = 0;
  if (!r.get(&no) || !r.get(&nt) || no > (1ull << 32) || nt > (1ull << 34)) return false;
  offsets->resize(static_cast<std::size_t>(no));
  targets->resize(static_cast<std::size_t>(nt));
  return r.get_bytes(offsets->data(), offsets->size() * sizeof(std::uint32_t)) &&
         r.get_bytes(targets->data(), targets->size() * sizeof(UnitId));
}

template <class T>
json array_of(const std::vector<T>& v) {
  json a = json::array();
  for (const T& x : v) a.push_back(x);
  return a;
}

template <class T>
bool read_array(const json& j, const char* key, std::vector<T>* out) {
  if (!j.contains(key) || !j[key].is_array()) return false;
  out->clear();
  out->reserve(j[key].size());
  for (const json& x : j[key]) {
    if (!x.is_number()) return false;
    out->push_back(x.get<T>());
  }
  return true;
}

bool write_json(const std::string& path, const json& j) { return fs::write_file_atomic(path, j.dump()); }

std::optional<json> read_json(const std::string& path) {
  const auto text = fs::read_file(path);
  if (!text) return std::nullopt;
  json j = json::parse(*text, nullptr, false);
  if (j.is_discarded()) return std::nullopt;
  return j;
}

}  // namespace

bool load_embedding_cache(const std::string& path, EmbeddingCache* cache) {
  const auto data = fs::read_file(path);
  if (!data) return false;
  Reader r{*data};
  char magic[8];
  std::uint32_t id_len = 0;
  std::uint64_t dim = 0, count = 0;
  if (!r.get_bytes(magic, 8) || std::memcmp(magic, kCacheMagic, 8) != 0 || !r.get(&id_len) || id_len > 4096)
    return false;
  std::string id(id_len, '\0');
  if (!r.get_bytes(id.data(), id_len) || !r.get(&dim) || !r.get(&count) || dim == 0 || dim > 65536) return false;
  if ((data->size() - r.pos) != count * (8 + dim * sizeof(float))) return false;
  EmbeddingCache c;
  c.embedder_id = id;
  c.dim = static_cast<std::size_t>(dim);
  c.rows.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    std::uint64_t key = 0;
    std::vector<float> row(c.dim);
    r.get(&key);
    r.get_bytes(row.data(), c.dim * sizeof(float));
    c.rows.emplace(key, std::move(row));
  }
  *cache = std::move(c);
  return true;
}

bool save_embedding_cache(const std::string& path, const EmbeddingCache& cache) {
  std::vector<std::uint64_t> keys;
  keys.reserve(cache.rows.size());
  for (const auto& kv : cache.rows) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  std::string out(kCacheMagic, 8);
  put<std::uint32_t>(out, static_cast<std::uint32_t>(cache.embedder_id.size()));
  out += cache.embedder_id;
  put<std::uint64_t>(out, cache.dim);
  put<std::uint64_t>(out, keys.size());
  for (const std::uint64_t k : keys) {
    put(out, k);
    const auto& row = cache.rows.at(k);
    out.append(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
  }
  return fs::write_file_atomic(path, out);
}

// ---- JSON forms ------------------------------------------------------------

json unit_to_json(const SourceUnit& u) {
  return json{{"id", u.id},
              {"path", u.path},
              {"language", u.language},
              {"kind", static_cast<int>(u.kind)},
              {"name", u.name},
              {"signature", u.signature},
              {"doc", u.doc},
              {"text", u.text},
              {"start_line", u.start_line},
              {"end_line", u.end_line},
              {"start_byte", u.start_byte},
              {"end_byte", u.end_byte},
              {"callees", u.callees},
              {"content_hash", u.content_hash},
              {"tokens", u.tokens}};
}

SourceUnit unit_from_json(const json& j) {
  SourceUnit u;
  u.id = j.value("id", kNoUnit);
  u.path = j.value("path", "");
  u.language = j.value("language", "");
  u.kind = static_cast<UnitKind>(j.value("kind", 3));
  u.name = j.value("name", "");
  u.signature = j.value("signature", "");
  u.doc = j.value("doc", "");
  u.text = j.value("text", "");
  u.start_line = j.value("start_line", 0u);
  u.end_line = j.value("end_line", 0u);
  u.start_byte = j.value("start_byte", 0u);
  u.end_byte = j.value("end_byte", 0u);
  u.callees = j.value("callees", std::vector<std::string>{});
  u.content_hash = j.value("content_hash", std::uint64_t{0});
  u.tokens = j.value("tokens", 0u);
  return u;
}

IndexReport report_from_json(const json& j) {
  IndexReport r;
  r.files = j.value("files", std::size_t{0});
  r.units = j.value("units", std::size_t{0});
  r.functions = j.value("functions", std::size_t{0});
  r.call_edges = j.value("call_edges", std::size_t{0});
  r.embedded = j.value("embedded", std::size_t{0});
  r.reused = j.value("reused", std::size_t{0});
  r.groups = j.value("groups", std::size_t{0});
  r.kappa = j.value("kappa", 0.0);
  r.reduced_dim = j.value("reduced_dim", std::size_t{0});
  r.explained_variance = j.value("explained_variance", 0.0);
  r.anisotropy_before = j.value("anisotropy_before", 0.0);
  r.anisotropy_after = j.value("anisotropy_after", 0.0);
  r.blocks = j.value("blocks", std::size_t{0});
  r.block_bytes = j.value("block_bytes", std::size_t{0});
  if (j.contains("seconds") && j["seconds"].is_object()) {
    for (auto it = j["seconds"].begin(); it != j["seconds"].end(); ++it) {
      if (it.value().is_number()) r.seconds[it.key()] = it.value().get<double>();
    }
  }
  r.error = j.value("error", "");
  return r;
}

static json calibration_info_json(const CalibrationInfo& c) {
  return json{{"fitted", c.fitted},
              {"source", c.source},
              {"queries", c.queries},
              {"heldout_queries", c.heldout_queries},
              {"ece", c.heldout.expected_calibration_error},
              {"brier", c.heldout.brier},
              {"log_loss", c.heldout.log_loss},
              {"bin_confidence", c.heldout.bin_confidence},
              {"bin_accuracy", c.heldout.bin_accuracy},
              {"bin_count", c.heldout.bin_count},
              {"conformal_z", c.conformal_z},
              {"conformal_alpha", c.conformal_alpha},
              {"lifts_estimated", c.lifts_estimated}};
}

static CalibrationInfo calibration_info_from(const json& j) {
  CalibrationInfo c;
  c.fitted = j.value("fitted", false);
  c.source = j.value("source", "");
  c.queries = j.value("queries", std::size_t{0});
  c.heldout_queries = j.value("heldout_queries", std::size_t{0});
  c.heldout.expected_calibration_error = j.value("ece", 0.0);
  c.heldout.brier = j.value("brier", 0.0);
  c.heldout.log_loss = j.value("log_loss", 0.0);
  c.heldout.bin_confidence = j.value("bin_confidence", std::vector<double>{});
  c.heldout.bin_accuracy = j.value("bin_accuracy", std::vector<double>{});
  c.heldout.bin_count = j.value("bin_count", std::vector<std::size_t>{});
  c.conformal_z = j.value("conformal_z", 0.0);
  c.conformal_alpha = j.value("conformal_alpha", 0.1);
  c.lifts_estimated = j.value("lifts_estimated", false);
  return c;
}

// ---- save / load -----------------------------------------------------------

bool save_calibration(const std::string& dir, const retrieve::Calibration& calibration,
                      const retrieve::EdgeLift& lifts, const CalibrationInfo& info) {
  const json cal{{"calibration", calibration.to_json()},
                 {"lifts", {{"callee", lifts.callee}, {"caller", lifts.caller}, {"sibling", lifts.sibling}}},
                 {"info", calibration_info_json(info)}};
  return write_json(dir + "/calibration.json", cal);
}

bool save_corpus(const Corpus& c, double kappa_cv, std::uint64_t settings_hash, const IndexReport& report,
                 const std::string& dir, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = "cannot write " + what;
    return false;
  };
  if (!fs::create_directories(dir)) return fail(dir);

  std::string units;
  for (const SourceUnit& u : c.units) {
    units += unit_to_json(u).dump(-1, ' ', false, json::error_handler_t::replace);
    units += '\n';
  }
  if (!fs::write_file_atomic(dir + "/units.jsonl", units)) return fail("units.jsonl");
  if (!c.store.save(dir + "/vectors.bin")) return fail("vectors.bin");

  std::string g(kGraphMagic, 8);
  put_csr(g, c.calls.forward());
  put_csr(g, c.siblings);
  if (!fs::write_file_atomic(dir + "/graph.bin", g)) return fail("graph.bin");

  const kde::Reduction& r = c.reduction;
  const kde::Modes& m = c.modes;
  json density{{"reduction",
                {{"in_dim", r.in_dim},
                 {"out_dim", r.out_dim},
                 {"dropped", r.dropped},
                 {"mean", array_of(r.mean)},
                 {"basis", array_of(r.basis)},
                 {"variance", array_of(r.variance)},
                 {"total_variance", r.total_variance}}},
               {"kappa", kappa_cv},
               {"mode_kappa", m.kappa},
               {"anisotropy_before", c.anisotropy_before},
               {"anisotropy_after", c.anisotropy_after},
               {"modes",
                {{"m", m.m},
                 {"centers", array_of(m.centers)},
                 {"assignment", array_of(m.assignment)},
                 {"sizes", array_of(m.sizes)},
                 {"log_density", array_of(m.log_density)},
                 {"center_log_density", array_of(m.center_log_density)},
                 {"not_converged", m.not_converged},
                 {"density_decreases", m.density_decreases}}}};
  if (!write_json(dir + "/density.json", density)) return fail("density.json");

  if (!save_calibration(dir, c.calibration, c.lifts, c.calibration_info)) return fail("calibration.json");

  char hash_hex[24];
  std::snprintf(hash_hex, sizeof hash_hex, "%016llx", static_cast<unsigned long long>(settings_hash));
  json manifest{{"format", 1},
                {"embedder_id", c.embedder_id},
                {"dim", c.store.dim()},
                {"settings_hash", hash_hex},
                {"roots", c.roots},
                {"files", c.files},
                {"file_hashes", c.file_hashes},
                {"file_of_unit", c.file_of_unit},
                {"units", c.units.size()},
                {"built_at", c.built_at},
                {"report", report_json(report)}};
  if (!write_json(dir + "/manifest.json", manifest)) return fail("manifest.json");
  return true;
}

bool load_corpus(const std::string& dir, const std::string& embedder_id, Persisted* out, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };
  const auto manifest = read_json(dir + "/manifest.json");
  if (!manifest) return fail("no index");
  if (manifest->value("embedder_id", "") != embedder_id)
    return fail("the index was built with embedder " + manifest->value("embedder_id", "") + ", not " + embedder_id);

  auto c = std::make_shared<Corpus>();
  c->embedder_id = embedder_id;
  c->roots = manifest->value("roots", std::vector<std::string>{});
  c->files = manifest->value("files", std::vector<std::string>{});
  c->file_hashes = manifest->value("file_hashes", std::vector<std::uint64_t>{});
  c->file_of_unit = manifest->value("file_of_unit", std::vector<std::uint32_t>{});
  c->built_at = manifest->value("built_at", "");
  const std::size_t n = manifest->value("units", std::size_t{0});

  const auto units_text = fs::read_file(dir + "/units.jsonl");
  if (!units_text) return fail("units.jsonl missing");
  for (const std::string_view line : text::split_lines(*units_text)) {
    if (line.empty()) continue;
    const json j = json::parse(line, nullptr, false);
    if (j.is_discarded()) return fail("units.jsonl is corrupt");
    c->units.push_back(unit_from_json(j));
  }
  if (c->units.size() != n || c->file_of_unit.size() != n) return fail("units.jsonl does not match the manifest");
  for (std::size_t i = 0; i < n; ++i) {
    if (c->units[i].id != i || c->file_of_unit[i] >= c->files.size()) return fail("units.jsonl is inconsistent");
  }

  if (!c->store.load(dir + "/vectors.bin") || c->store.size() != n) return fail("vectors.bin missing or corrupt");

  const auto g = fs::read_file(dir + "/graph.bin");
  if (!g || g->size() < 8 || std::memcmp(g->data(), kGraphMagic, 8) != 0) return fail("graph.bin missing");
  Reader r{*g, 8};
  std::vector<std::uint32_t> co, so;
  std::vector<UnitId> ct, st;
  if (!get_csr(r, &co, &ct) || !get_csr(r, &so, &st) || so.size() != n + 1) return fail("graph.bin is corrupt");
  c->siblings = graph::NeighborGraph::from_csr(std::move(so), std::move(st));
  c->calls = parse::build_call_graph(c->units);
  if (c->calls.edges() != ct.size()) return fail("graph.bin does not match the units");

  const auto density = read_json(dir + "/density.json");
  if (!density) return fail("density.json missing");
  try {
    const json& jr = density->at("reduction");
    kde::Reduction& red = c->reduction;
    red.in_dim = jr.at("in_dim").get<std::size_t>();
    red.out_dim = jr.at("out_dim").get<std::size_t>();
    red.dropped = jr.at("dropped").get<std::size_t>();
    red.total_variance = jr.at("total_variance").get<double>();
    if (!read_array(jr, "mean", &red.mean) || !read_array(jr, "basis", &red.basis) ||
        !read_array(jr, "variance", &red.variance))
      return fail("density.json is corrupt");
    const json& jm = density->at("modes");
    kde::Modes& m = c->modes;
    m.m = jm.at("m").get<std::size_t>();
    m.kappa = density->value("mode_kappa", 0.0);
    m.not_converged = jm.value("not_converged", std::size_t{0});
    m.density_decreases = jm.value("density_decreases", std::size_t{0});
    if (!read_array(jm, "centers", &m.centers) || !read_array(jm, "assignment", &m.assignment) ||
        !read_array(jm, "sizes", &m.sizes) || !read_array(jm, "log_density", &m.log_density) ||
        !read_array(jm, "center_log_density", &m.center_log_density))
      return fail("density.json is corrupt");
    c->anisotropy_before = density->value("anisotropy_before", 0.0);
    c->anisotropy_after = density->value("anisotropy_after", 0.0);
    out->kappa_cv = density->value("kappa", 0.0);
  } catch (const json::exception&) {
    return fail("density.json is corrupt");
  }
  if (c->modes.assignment.size() != n || c->reduction.in_dim != c->store.dim())
    return fail("density.json does not match the units");

  std::vector<float> rows(n * c->store.dim());
  for (std::size_t i = 0; i < n; ++i) {
    const auto v = c->store.vector(static_cast<UnitId>(i));
    std::copy(v.begin(), v.end(), rows.begin() + static_cast<std::ptrdiff_t>(i * c->store.dim()));
  }
  c->reduced = kde::reduce(c->reduction, rows);

  const auto cal = read_json(dir + "/calibration.json");
  if (cal) {
    c->calibration = retrieve::Calibration::from_json(cal->value("calibration", json::object()));
    const json l = cal->value("lifts", json::object());
    c->lifts.callee = l.value("callee", c->lifts.callee);
    c->lifts.caller = l.value("caller", c->lifts.caller);
    c->lifts.sibling = l.value("sibling", c->lifts.sibling);
    c->calibration_info = calibration_info_from(cal->value("info", json::object()));
  } else {
    c->calibration = retrieve::Calibration::fallback();
  }

  const std::string hash_hex = manifest->value("settings_hash", "");
  out->settings_hash = std::strtoull(hash_hex.c_str(), nullptr, 16);
  if (manifest->contains("report") && (*manifest)["report"].is_object())
    out->report = report_from_json((*manifest)["report"]);
  out->corpus = std::move(c);
  return true;
}

std::vector<ChatMessage> prompt_for(const Corpus& corpus, std::string_view question, std::span<const UnitId> ids,
                                    bool edit_mode) {
  context::Selection sel;
  for (const UnitId id : ids) {
    if (id >= corpus.units.size()) continue;
    context::Entry e;
    e.id = id;
    e.group = corpus.file_of_unit[id];
    e.form = context::Form::Full;
    e.p = 1.0;
    e.quality = 1.0;
    e.tokens = corpus.units[id].tokens;
    sel.entries.push_back(e);
  }
  context::PromptOptions po;
  po.edit_mode = edit_mode;
  po.root = corpus.roots.empty() ? std::string() : corpus.roots.front();
  return context::build_messages(question, corpus.units, sel, {}, {}, po);
}

}  // namespace prag::engine::detail
