// Opportunistic fine-tuning pipeline (DESIGN.md section 6.12).
//
// The pipeline is switched on and off by the user. While on, it waits until the
// machine is idle, then builds a training set from the indexed repository and
// the recorded feedback, fine-tunes the float32 base model, calibrates and
// quantises the result with repository text, evaluates it against the serving
// model and promotes it only when it passes the gate. Any user activity cancels
// the running stage at its next safe point; a later idle period resumes from the
// last finished stage.
//
// The module does not know the engine: everything it needs from the outside
// arrives through Hooks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "prag/core/settings.hpp"
#include "prag/core/types.hpp"

namespace prag::tune {

// ---- training examples -----------------------------------------------------

struct Example {
  std::string id;                  // stable, derived from the content
  std::string question;
  std::vector<UnitId> context;     // units shown to the model, in prompt order
  UnitId oracle = kNoUnit;         // the unit that answers the question
  bool oracle_present = true;      // false: the context holds distractors only
  std::string answer;              // reference answer
  std::string source;              // "documentation", "generated", "feedback" or "edit"
};

struct ExampleSet {
  std::vector<Example> train;
  std::vector<Example> holdout;    // never trained on; used by the promotion gate
};

struct CorpusView {
  std::span<const SourceUnit> units;
  // Units that are hard negatives for `unit`: same density mode, or relevance
  // probability closest to the retrieval threshold. At most `count` ids.
  std::function<std::vector<UnitId>(UnitId unit, std::size_t count)> distractors;
  // Chat messages for a question over the given context units, in the exact
  // format used when serving.
  std::function<std::vector<ChatMessage>(std::string_view question, std::span<const UnitId> context)> prompt;
};

struct ExampleOptions {
  std::size_t max_examples = 400;
  std::size_t distractors = 4;
  double oracle_fraction = 0.8;    // share of examples whose context contains the oracle
  double holdout_fraction = 0.15;  // by unit: a unit is either in train or in holdout
  std::size_t min_doc_words = 6;   // units with shorter documentation yield no example
  std::uint64_t seed = 7;
};

// Deterministic for a given corpus, options and seed. Examples whose oracle is
// absent carry the reference answer "The supplied code does not contain the answer."
ExampleSet build_examples(const CorpusView& corpus, const ExampleOptions& options);

// Training text: every example rendered with `chat_template` applied to the
// prompt plus the reference answer as the assistant turn, concatenated.
std::string render_training_text(const ExampleSet& examples, const CorpusView& corpus,
                                 const std::function<std::string(std::span<const ChatMessage>)>& chat_template);

// ---- idle detection --------------------------------------------------------

struct IdleState {
  bool idle = false;
  double seconds_since_activity = 0.0;
  double cpu_load = 0.0;       // share of CPU time used by other processes, 0..1
  bool on_ac_power = true;
  std::string reason;          // why the machine is or is not considered idle
};

struct IdlePolicy {
  double idle_seconds = 300.0;
  double max_cpu_load = 0.25;
  bool require_ac_power = true;
};

// Samples CPU load over about 200 ms. `seconds_since_activity` is supplied by the caller.
IdleState probe_idle(double seconds_since_activity, const IdlePolicy& policy);

// ---- promotion gate --------------------------------------------------------

struct GateInput {
  std::vector<double> base_nll;       // per held-out example, nats per token, serving model
  std::vector<double> candidate_nll;  // same examples, candidate model
  double general_base_nll = 0.0;      // nats per token on the general probe text
  double general_candidate_nll = 0.0;
  double quantization_kl = 0.0;       // mean KL per token, candidate before vs after quantisation
  bool candidate_loads = true;
};

struct GatePolicy {
  double alpha = 0.05;   // significance level of the one-sided paired test
  double margin = 0.02;  // allowed loss on the general probe, nats per token
  double max_kl = 0.05;  // allowed divergence caused by quantisation, nats per token
  std::size_t min_examples = 20;
};

struct GateResult {
  bool pass = false;
  double domain_nll_base = 0.0;       // means over the held-out examples
  double domain_nll_candidate = 0.0;
  double p_value = 1.0;               // one-sided paired t-test, H1: candidate < base
  double general_nll_base = 0.0;
  double general_nll_candidate = 0.0;
  double quantization_kl = 0.0;
  std::string reason;                 // first failed requirement; empty when pass
};

GateResult evaluate_gate(const GateInput& input, const GatePolicy& policy);

// ---- pipeline --------------------------------------------------------------

enum class Stage {
  Disabled, Waiting, Collecting, Training, Calibrating, Quantizing, Evaluating, Promoting, Failed
};

const char* to_string(Stage stage);

struct RunRecord {
  std::string id;
  std::string started;   // ISO 8601, UTC
  std::string finished;
  Stage stage = Stage::Waiting;  // last stage reached
  bool promoted = false;
  std::size_t examples = 0;
  int epochs = 0;
  std::vector<double> train_loss;
  std::vector<double> validation_loss;
  GateResult gate;
  std::string model;     // path of the quantised candidate
  double seconds = 0.0;
  std::string error;
};

struct Status {
  bool available = false;
  bool enabled = false;
  Stage stage = Stage::Disabled;
  double progress = 0.0;
  std::string message;
  IdleState idle;
  std::size_t pending_examples = 0;
  std::string serving_model;
  std::string base_model;
  std::vector<RunRecord> runs;  // oldest first
};

nlohmann::json to_json(const Status& status);  // field names as in DESIGN.md section 9

struct Hooks {
  // Snapshot of the corpus; returns false when no index is available.
  std::function<bool(CorpusView* view)> corpus;
  // Path of the model that currently serves requests.
  std::function<std::string()> serving_model;
  // Negative log-likelihood per token of the reference answers of `examples`
  // under the model at `model_path` (loaded for the purpose unless it is the
  // serving model). One value per example.
  std::function<bool(const std::string& model_path, std::span<const Example> examples,
                     std::vector<double>* nll, std::string* error)> score_examples;
  // Replaces the serving model. Returns false and leaves the old model in place on failure.
  std::function<bool(const std::string& model_path, std::string* error)> promote;
  // Examples recorded from user feedback since the last run.
  std::function<std::vector<Example>()> feedback_examples;
};

class Pipeline {
 public:
  // `settings` supplies the tune.* and models.* keys; `data_dir` is where runs are stored.
  Pipeline(const Settings& settings, std::string data_dir, Hooks hooks);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  void start();  // starts the background thread; idempotent
  void stop();   // cancels and joins

  void set_enabled(bool enabled);
  bool enabled() const;

  // To be called on every user request. Cancels a running stage.
  void notify_activity();

  // Runs the pipeline once now, without waiting for idleness. Returns false
  // when a run is already in progress or tuning is unavailable.
  bool run_now(std::string* error = nullptr);

  Status status() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace prag::tune
