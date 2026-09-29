// Private state of prag::engine::Engine, shared by the implementation files.
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "prag/engine/engine.hpp"

namespace prag::engine {

namespace detail {

std::string iso_now();
// `p` when absolute, otherwise base_dir/p; normalised.
std::string resolve_path(const std::string& base_dir, const std::string& p);
std::vector<std::string> corpus_roots(const Settings& settings, const std::string& base_dir);
// Hash of the settings that shape the index (corpus.*, models.embedding, kde.*, index.*).
std::uint64_t index_settings_hash(const Settings& settings);
std::size_t count_words(std::string_view s);
// Unit text without its leading comment block.
std::string strip_leading_comment(std::string_view text, std::string_view language);
// Deterministic Fisher-Yates shuffle (independent of the standard library).
void shuffle_ids(std::vector<UnitId>& ids, std::uint64_t seed);
std::uint64_t next_random(std::uint64_t& state);

struct EmbeddingCache {
  std::string embedder_id;
  std::size_t dim = 0;
  std::unordered_map<std::uint64_t, std::vector<float>> rows;
};
bool load_embedding_cache(const std::string& path, EmbeddingCache* cache);
bool save_embedding_cache(const std::string& path, const EmbeddingCache& cache);

// Persistence of a corpus snapshot (DESIGN.md section 5).
struct Persisted {
  std::shared_ptr<Corpus> corpus;
  double kappa_cv = 0.0;
  std::uint64_t settings_hash = 0;
  std::optional<IndexReport> report;
};
bool save_corpus(const Corpus& corpus, double kappa_cv, std::uint64_t settings_hash, const IndexReport& report,
                 const std::string& index_dir, std::string* error);
bool save_calibration(const std::string& index_dir, const retrieve::Calibration& calibration,
                      const retrieve::EdgeLift& lifts, const CalibrationInfo& info);
bool load_corpus(const std::string& index_dir, const std::string& embedder_id, Persisted* out, std::string* error);

nlohmann::json unit_to_json(const SourceUnit& unit);
SourceUnit unit_from_json(const nlohmann::json& j);
IndexReport report_from_json(const nlohmann::json& j);

// Messages for `question` over the given context units, in the serving format.
std::vector<ChatMessage> prompt_for(const Corpus& corpus, std::string_view question, std::span<const UnitId> ids,
                                    bool edit_mode);

}  // namespace detail

struct Engine::Impl {
  std::string settings_path;
  std::string base_dir;
  Settings settings;
  std::unique_ptr<exec::OrderedExecutor> executor;

  mutable std::mutex mu;  // guards everything down to `tune_snapshot`
  std::shared_ptr<const Corpus> corpus;
  double kappa_cv = 0.0;
  bool stale = false;
  std::shared_ptr<embed::Embedder> embedder;
  bool embedder_is_model = false;
  std::shared_ptr<llm::LanguageModel> chat;
  std::string chat_path;
  IndexProgress progress{"done", 0, 0, ""};
  bool progress_running = false;
  std::optional<IndexReport> last_report;
  std::vector<HistoryEntry> history;
  std::uint64_t next_query = 1;
  std::map<std::string, edit::Transaction> transactions;
  std::set<std::string> committed;
  std::shared_ptr<const Corpus> tune_snapshot;

  std::mutex embed_mu;  // serialises embedder calls; guards the question cache
  std::string question_cache_id;
  std::unordered_map<std::string, std::vector<float>> question_cache;

  std::mutex build_mu;  // one build or calibration at a time; guards emb_cache
  std::atomic<bool> building{false};
  detail::EmbeddingCache emb_cache;

  std::unique_ptr<tune::Pipeline> tuning;

  std::string data_dir() const;
  std::string index_dir() const { return data_dir() + "/index"; }
  std::string edits_dir() const { return data_dir() + "/edits"; }
  std::string history_path() const { return data_dir() + "/history.jsonl"; }
  std::size_t block_bytes() const;

  std::shared_ptr<embed::Embedder> ensure_embedder();
  std::shared_ptr<llm::LanguageModel> chat_model() const;
  std::shared_ptr<const Corpus> snapshot() const;
  // Serialised call of the embedder: texts.size() * dim floats.
  std::vector<float> embed_texts(embed::Embedder& embedder, std::span<const std::string_view> texts,
                                 embed::Role role);
  std::vector<float> embed_question(embed::Embedder& embedder, const std::string& question);

  bool report_progress(const IndexProgressCallback& callback, const std::string& stage, std::size_t done,
                       std::size_t total, const std::string& message);
  IndexReport build(const IndexProgressCallback& callback);
  bool calibrate_corpus(const Corpus& corpus, const CalibrationOptions& options,
                        const IndexProgressCallback& callback, const std::shared_ptr<embed::Embedder>& embedder,
                        const std::shared_ptr<llm::LanguageModel>& chat, retrieve::Calibration* calibration,
                        retrieve::EdgeLift* lifts, CalibrationInfo* info, std::string* error);

  std::vector<std::string> current_roots() const;
  edit::ApplyOptions apply_options(const std::vector<std::string>& roots) const;
  void load_history();
  bool save_history_locked() const;  // caller holds mu
  retrieve::EdgeLift default_lifts() const;
  void make_tuning();
};

}  // namespace prag::engine
