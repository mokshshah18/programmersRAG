// The engine: corpus, index, models, retrieval, generation, edits and tuning
// behind one thread-safe object. The server, the command line and the
// experiment drivers talk only to this class (DESIGN.md section 6.13).
//
// Concurrency model. Everything derived from the corpus lives in an immutable
// Corpus snapshot held by shared_ptr. A query takes the current snapshot and
// works on it without locks; a rebuild constructs a new snapshot and swaps the
// pointer. Models are shared objects that serialise their own calls.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "prag/context/prompt.hpp"
#include "prag/context/rot.hpp"
#include "prag/core/embedder.hpp"
#include "prag/core/settings.hpp"
#include "prag/core/types.hpp"
#include "prag/edit/edit.hpp"
#include "prag/exec/ordered_executor.hpp"
#include "prag/graph/call_graph.hpp"
#include "prag/index/vector_store.hpp"
#include "prag/kde/directional.hpp"
#include "prag/llm/model.hpp"
#include "prag/retrieve/calibration.hpp"
#include "prag/retrieve/trace.hpp"
#include "prag/tune/pipeline.hpp"

namespace prag::engine {

// ---- corpus snapshot -------------------------------------------------------

struct CalibrationInfo {
  bool fitted = false;
  std::string source;               // "documentation" or "generated"
  std::size_t queries = 0;          // calibration queries used for fitting
  std::size_t heldout_queries = 0;
  retrieve::Reliability heldout;    // reliability on the held-out queries
  double conformal_z = 0.0;         // split-conformal threshold for alpha = conformal_alpha
  double conformal_alpha = 0.1;
  bool lifts_estimated = false;     // false: EdgeLift holds the defaults
};

struct Corpus {
  std::vector<std::string> roots;
  std::vector<std::string> files;           // normalised paths, sorted
  std::vector<std::uint64_t> file_hashes;   // hash64 of each file's content
  std::vector<SourceUnit> units;            // units[i].id == i
  std::vector<std::uint32_t> file_of_unit;  // index into files
  graph::CallGraph calls;
  graph::NeighborGraph siblings;            // nearest same-mode units
  std::string embedder_id;
  index::VectorStore store;
  kde::Reduction reduction;
  kde::Modes modes;                         // modes.assignment[i] is the group of unit i
  std::vector<float> reduced;               // units * reduction.out_dim
  retrieve::Calibration calibration;
  retrieve::EdgeLift lifts;
  CalibrationInfo calibration_info;
  double anisotropy_before = 0.0;           // mean pairwise cosine of the raw embeddings
  double anisotropy_after = 0.0;            // after centring and reduction
  std::string built_at;                     // ISO 8601, UTC
};

// ---- indexing --------------------------------------------------------------

struct IndexProgress {
  std::string stage;  // discover, parse, graph, embed, density, codes, calibrate, save, done, failed
  std::size_t done = 0;
  std::size_t total = 0;
  std::string message;
};

// Return false to cancel the build.
using IndexProgressCallback = std::function<bool(const IndexProgress&)>;

struct IndexReport {
  std::size_t files = 0;
  std::size_t units = 0;
  std::size_t functions = 0;
  std::size_t call_edges = 0;
  std::size_t embedded = 0;  // units embedded in this build
  std::size_t reused = 0;    // units whose embedding was taken from the previous index
  std::size_t groups = 0;
  double kappa = 0.0;
  std::size_t reduced_dim = 0;
  double explained_variance = 0.0;
  double anisotropy_before = 0.0;
  double anisotropy_after = 0.0;
  std::size_t blocks = 0;
  std::size_t block_bytes = 0;
  std::map<std::string, double> seconds;  // per stage, plus "total"
  std::string error;                      // empty on success
};

// ---- queries ---------------------------------------------------------------

struct QueryOptions {
  bool edit = false;      // ask the model for edit blocks and prepare a transaction
  double tau = 0.0;       // 0 = setting retrieve.tau
  bool generate = true;   // false: retrieval and context selection only
  bool record = true;     // append the exchange to the history
  bool use_history = true;
};

struct RetrievedUnit {
  retrieve::TracedUnit traced;
  std::uint32_t group = 0;
  bool in_context = false;
  context::Form form = context::Form::Full;
  bool support = false;
  double redundancy = 0.0;
  double quality = 0.0;
};

struct ScanInfo {
  std::size_t blocks = 0;
  std::size_t block_bytes = 0;
  unsigned threads = 0;
  std::uint64_t wall_us = 0;
  std::uint64_t work_us = 0;
  std::uint64_t gate_wait_us = 0;
  std::uint64_t admission_violations = 0;
  std::uint64_t commit_violations = 0;
  std::size_t rescored = 0;
  std::string codes;
  std::string simd;
};

struct QueryResult {
  std::string id;
  std::string question;
  double tau = 0.25;
  std::vector<RetrievedUnit> units;  // traced units first (acceptance order), then support units
  retrieve::TraceResult trace;       // `units` inside may be empty; counters are filled
  context::Selection selection;
  ScanInfo scan;
  std::size_t history_kept = 0;
  std::size_t history_total = 0;
  std::vector<ChatMessage> messages; // the prompt given to the model
  std::string answer;
  llm::GenerateStats generation;
  bool extractive = false;           // answered without a language model
  std::string model;
  double embed_ms = 0.0;
  double retrieve_ms = 0.0;          // scan, rescoring, trace and selection
  double total_ms = 0.0;
  std::optional<edit::Transaction> transaction;  // edit mode
  std::vector<std::string> edit_warnings;
  std::string error;                 // empty on success
};

// Called once, when retrieval and context selection are complete and before
// generation starts. `partial.answer` is empty at that point.
using RetrievalCallback = std::function<void(const QueryResult& partial)>;

struct HistoryEntry {
  std::string id;
  std::string time;  // ISO 8601, UTC
  std::string question;
  std::string answer;
  std::vector<UnitId> units;
  int rating = 0;    // -1, 0 or +1
};

struct CalibrationOptions {
  std::size_t max_queries = 300;
  std::size_t negatives_per_query = 64;
  double heldout_fraction = 0.3;
  bool generate_questions = true;   // use the chat model when loaded; otherwise documentation
  bool estimate_lifts = true;       // needs the chat model; otherwise the defaults stay
  std::size_t lift_queries = 60;
  std::size_t lift_candidates_per_kind = 3;
  double lift_min_gain = 0.02;      // nats per token by which a unit must lower the answer loss
  std::uint64_t seed = 11;
};

// ---- engine ----------------------------------------------------------------

class Engine {
 public:
  // Loads the settings file (created on first save when missing).
  explicit Engine(std::string settings_path);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  Settings& settings();
  const Settings& settings() const;
  bool save_settings() const;
  // Merges `patch`, saves, and marks the index stale when an index-relevant key changed.
  void update_settings(const nlohmann::json& patch);

  // Loads the persisted index and the configured models, starts the tuning
  // pipeline thread. Missing pieces are reported by the status, not by failure.
  void open();

  exec::OrderedExecutor& executor();
  std::size_t block_bytes() const;

  // Models. May be null.
  std::shared_ptr<embed::Embedder> embedder() const;
  std::shared_ptr<llm::LanguageModel> chat_model() const;
  bool load_chat_model(const std::string& path, std::string* error);

  // Index.
  std::shared_ptr<const Corpus> corpus() const;  // null before the first build
  bool index_ready() const;                      // a corpus exists and is not stale
  bool index_building() const;
  IndexReport build_index(const IndexProgressCallback& progress = {});
  IndexProgress index_progress() const;
  std::optional<IndexReport> last_index_report() const;
  // Re-estimates the calibration of the current corpus and saves it.
  bool calibrate(const CalibrationOptions& options, const IndexProgressCallback& progress = {},
                 std::string* error = nullptr);

  // Queries.
  QueryResult query(std::string_view question, const QueryOptions& options,
                    const RetrievalCallback& on_retrieval = {}, const llm::PieceCallback& on_piece = {});

  // History and feedback.
  std::vector<HistoryEntry> history() const;
  bool clear_history();
  bool rate(const std::string& id, int rating);

  // Edits. Transactions are kept in memory by id until the engine is destroyed.
  edit::Transaction preview_edit(std::string_view model_output, std::vector<std::string>* warnings = nullptr);
  edit::Transaction preview_edit(const std::vector<edit::EditBlock>& blocks);
  // After a successful commit the touched files are re-indexed.
  edit::Status apply_edit(const std::string& id, edit::Transaction* result = nullptr);
  edit::Status revert_edit(const std::string& id, edit::Transaction* result = nullptr);
  bool transaction_committed(const std::string& id) const;

  tune::Pipeline& tuning();

  // JSON views with the field names of DESIGN.md section 9.
  nlohmann::json status_json() const;
  nlohmann::json index_progress_json() const;
  nlohmann::json groups_json() const;
  nlohmann::json unit_json(UnitId id) const;
  nlohmann::json history_json() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// JSON for the query stream events and for transactions (DESIGN.md section 9).
nlohmann::json retrieval_json(const QueryResult& result, const Corpus& corpus);
nlohmann::json done_json(const QueryResult& result);
nlohmann::json transaction_json(const edit::Transaction& transaction, bool committed,
                                const std::vector<std::string>& roots,
                                const std::vector<std::string>& warnings = {});
nlohmann::json report_json(const IndexReport& report);

// Path shown to the user: relative to the root that contains it, otherwise unchanged.
std::string display_path(std::string_view path, const std::vector<std::string>& roots);

}  // namespace prag::engine
