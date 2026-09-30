// Fine-tuning pipeline: state machine, persistence and JSON (DESIGN.md section 6.12).
#include "prag/tune/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/model.hpp"

namespace prag::tune {
namespace {

namespace stdfs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr const char* kStageNames[] = {"disabled",    "waiting",   "collecting", "training", "calibrating",
                                       "quantizing", "evaluating", "promoting",  "failed"};

std::string iso_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

double finite_or(double v, double fallback) { return std::isfinite(v) ? v : fallback; }

stdfs::path path_of(const std::string& s) { return stdfs::path(std::u8string(s.begin(), s.end())); }

void remove_quietly(const std::string& path) {
  std::error_code ec;
  stdfs::remove(path_of(path), ec);
}

}  // namespace

const char* to_string(Stage stage) {
  const auto i = static_cast<std::size_t>(stage);
  return i < std::size(kStageNames) ? kStageNames[i] : "failed";
}

Stage stage_from_string(std::string_view name, Stage fallback) {
  for (std::size_t i = 0; i < std::size(kStageNames); ++i)
    if (name == kStageNames[i]) return static_cast<Stage>(i);
  return fallback;
}

// ---- JSON -------------------------------------------------------------------

nlohmann::json to_json(const IdleState& idle) {
  return {{"idle", idle.idle},
          {"seconds_since_activity", finite_or(idle.seconds_since_activity, 0.0)},
          {"cpu_load", finite_or(idle.cpu_load, 0.0)},
          {"on_ac_power", idle.on_ac_power},
          {"reason", idle.reason}};
}

nlohmann::json to_json(const GateResult& g) {
  return {{"pass", g.pass},
          {"domain_nll_base", finite_or(g.domain_nll_base, 0.0)},
          {"domain_nll_candidate", finite_or(g.domain_nll_candidate, 0.0)},
          {"p_value", finite_or(g.p_value, 1.0)},
          {"general_nll_base", finite_or(g.general_nll_base, 0.0)},
          {"general_nll_candidate", finite_or(g.general_nll_candidate, 0.0)},
          {"quantization_kl", finite_or(g.quantization_kl, 0.0)},
          {"reason", g.reason}};
}

nlohmann::json to_json(const RunRecord& r) {
  return {{"id", r.id},
          {"started", r.started},
          {"finished", r.finished},
          {"stage", to_string(r.stage)},
          {"promoted", r.promoted},
          {"examples", r.examples},
          {"epochs", r.epochs},
          {"train_loss", r.train_loss},
          {"validation_loss", r.validation_loss},
          {"gate", to_json(r.gate)},
          {"model", r.model},
          {"seconds", finite_or(r.seconds, 0.0)},
          {"error", r.error}};
}

bool run_from_json(const nlohmann::json& j, RunRecord* r) {
  if (!j.is_object() || !j.contains("id")) return false;
  try {
    RunRecord out;
    out.id = j.at("id").get<std::string>();
    out.started = j.value("started", std::string());
    out.finished = j.value("finished", std::string());
    out.stage = stage_from_string(j.value("stage", std::string("waiting")));
    out.promoted = j.value("promoted", false);
    out.examples = j.value("examples", std::size_t{0});
    out.epochs = j.value("epochs", 0);
    out.train_loss = j.value("train_loss", std::vector<double>());
    out.validation_loss = j.value("validation_loss", std::vector<double>());
    if (j.contains("gate") && j["gate"].is_object()) {
      const auto& g = j["gate"];
      out.gate.pass = g.value("pass", false);
      out.gate.domain_nll_base = g.value("domain_nll_base", 0.0);
      out.gate.domain_nll_candidate = g.value("domain_nll_candidate", 0.0);
      out.gate.p_value = g.value("p_value", 1.0);
      out.gate.general_nll_base = g.value("general_nll_base", 0.0);
      out.gate.general_nll_candidate = g.value("general_nll_candidate", 0.0);
      out.gate.quantization_kl = g.value("quantization_kl", 0.0);
      out.gate.reason = g.value("reason", std::string());
    }
    out.model = j.value("model", std::string());
    out.seconds = j.value("seconds", 0.0);
    out.error = j.value("error", std::string());
    *r = std::move(out);
    return true;
  } catch (const nlohmann::json::exception&) {
    return false;
  }
}

nlohmann::json to_json(const Status& s) {
  nlohmann::json runs = nlohmann::json::array();
  for (const RunRecord& r : s.runs) runs.push_back(to_json(r));
  return {{"available", s.available},
          {"enabled", s.enabled},
          {"stage", to_string(s.stage)},
          {"progress", finite_or(s.progress, 0.0)},
          {"message", s.message},
          {"idle", to_json(s.idle)},
          {"pending_examples", s.pending_examples},
          {"serving_model", s.serving_model},
          {"base_model", s.base_model},
          {"runs", std::move(runs)}};
}

// ---- pipeline ---------------------------------------------------------------

struct Pipeline::Impl {
  Settings settings;
  std::string data_dir;
  Hooks hooks;
  Steps steps;
  bool available = false;
  std::string unavailable_reason;

  // Shared state, guarded by `mutex`. It is never held while a hook or step runs,
  // so status() cannot block on a stage.
  mutable std::mutex mutex;
  std::condition_variable cv;
  bool enabled = false;
  bool running = false;        // a run has been claimed or is executing
  bool run_requested = false;  // manual request waiting for the worker
  bool stopping = false;
  Stage stage = Stage::Disabled;
  double progress = 0.0;
  std::string message;
  IdleState idle;
  std::size_t pending = 0;
  std::string serving_model;
  std::vector<RunRecord> runs;
  int next_run_number = 1;
  Clock::time_point last_activity = Clock::now();
  Clock::time_point cooldown_until = Clock::time_point::min();

  std::atomic<bool> cancel{false};
  std::thread worker;
  bool started = false;

  // State of an unfinished run; touched by the worker thread only.
  struct Active {
    bool on = false;
    RunRecord record;
    std::string dir, f32, tuned, quantized;
    ExampleSet examples;
    std::string text;
    llm::ImportanceMatrix importance;
    int done = 0;  // 1 collected, 2 converted, 3 trained, 4 calibrated, 5 quantised, 6 evaluated
    double seconds = 0.0;
  } run;

  Impl(const Settings& s, std::string dir, Hooks h, Steps st)
      : settings(s), data_dir(std::move(dir)), hooks(std::move(h)), steps(std::move(st)) {
    const std::string base = settings.get<std::string>("models.tuning_base", "");
    if (steps.needs_runtime && !llm::available())
      unavailable_reason = "this build has no llama.cpp";
    else if (base.empty())
      unavailable_reason = "models.tuning_base is not set";
    available = unavailable_reason.empty();

    enabled = settings.get<bool>("tune.enabled", false);
    if (hooks.serving_model) serving_model = hooks.serving_model();
    load_runs();
    apply_enabled_stage();
  }

  std::string runs_path() const { return (path_of(data_dir) / "runs.jsonl").generic_string(); }

  void load_runs() {
    std::ifstream in(path_of(runs_path()));
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      RunRecord r;
      const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
      if (j.is_discarded() || !run_from_json(j, &r)) continue;
      const auto dash = r.id.rfind('-');
      if (dash != std::string::npos) {
        const int n = std::atoi(r.id.c_str() + dash + 1);
        next_run_number = std::max(next_run_number, n + 1);
      }
      runs.push_back(std::move(r));
    }
  }

  // Caller holds the mutex.
  void apply_enabled_stage() {
    if (!available) {
      stage = Stage::Disabled;
      message = enabled ? "Tuning is unavailable: " + unavailable_reason : std::string();
    } else if (enabled) {
      stage = Stage::Waiting;
      message = "Waiting for the machine to be idle";
    } else {
      stage = Stage::Disabled;
      message.clear();
    }
  }

  void set_stage(Stage s, std::string msg, double p) {
    std::lock_guard<std::mutex> lock(mutex);
    stage = s;
    message = std::move(msg);
    progress = p;
    run.record.stage = s;
  }

  void set_progress(double p, const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex);
    progress = p;
    if (!msg.empty()) message = msg;
  }

  llm::ProgressCallback progress_callback(double lo, double hi) {
    return [this, lo, hi](double fraction, std::string_view msg) {
      set_progress(lo + (hi - lo) * std::clamp(fraction, 0.0, 1.0), std::string(msg));
      return !cancel.load();
    };
  }

  // ---- worker -------------------------------------------------------------

  void loop() {
    const double poll_seconds = settings.get<double>("tune.poll_seconds", 1.0);
    while (true) {
      bool manual = false;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::duration<double>(poll_seconds), [&] { return stopping || run_requested; });
        if (stopping) return;
        manual = run_requested;
        run_requested = false;
        if (!manual && (!enabled || !available || running)) continue;
      }
      if (!manual && !auto_ready()) continue;
      execute();
    }
  }

  // Probes the machine; claims the run when it is idle.
  bool auto_ready() {
    IdlePolicy policy;
    policy.idle_seconds = settings.get<double>("tune.idle_seconds", 300.0);
    policy.max_cpu_load = settings.get<double>("tune.max_cpu_load", 0.25);
    policy.require_ac_power = settings.get<bool>("tune.require_ac_power", true);

    std::string serving;
    if (hooks.serving_model) serving = hooks.serving_model();
    std::size_t pending_feedback = 0;
    if (hooks.feedback_examples) pending_feedback = hooks.feedback_examples().size();

    double since;
    {
      std::lock_guard<std::mutex> lock(mutex);
      since = std::chrono::duration<double>(Clock::now() - last_activity).count();
    }
    const IdleState state = probe_idle(since, policy);

    std::lock_guard<std::mutex> lock(mutex);
    idle = state;
    serving_model = serving;
    if (!run.on) pending = pending_feedback;
    if (!enabled || running || stopping) return false;
    if (!state.idle || Clock::now() < cooldown_until) return false;
    // Activity that arrived while probing means the machine is no longer idle.
    if (std::chrono::duration<double>(Clock::now() - last_activity).count() < policy.idle_seconds) return false;
    running = true;
    return true;
  }

  enum class Outcome { Finished, Paused, Failed };

  void execute() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      running = true;
    }
    cancel.store(false);
    Stopwatch watch;
    const Outcome outcome = run_stages();
    run.seconds += watch.elapsed_s();

    std::lock_guard<std::mutex> lock(mutex);
    running = false;
    if (outcome == Outcome::Paused) {
      apply_enabled_stage();
      message = "Paused by user activity; the run resumes when the machine is idle";
      return;
    }
    const double cooldown = settings.get<double>("tune.cooldown_seconds", 3600.0);
    cooldown_until = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(cooldown));
    if (outcome == Outcome::Failed) {
      apply_enabled_stage();
      stage = Stage::Failed;
      message = "Run " + run.record.id + " failed: " + run.record.error;
    } else {
      apply_enabled_stage();
      message = run.record.promoted ? "Run " + run.record.id + " promoted"
                                    : "Run " + run.record.id + " not promoted: " + run.record.gate.reason;
    }
    progress = outcome == Outcome::Finished ? 1.0 : 0.0;
    run = Active{};
  }

  // Ends the run, records it and removes the intermediate files.
  Outcome finish(bool failed) {
    RunRecord rec = run.record;
    rec.finished = iso_now();
    rec.seconds = run.seconds;
    remove_quietly(run.f32);
    remove_quietly(run.tuned);
    if (!rec.promoted) remove_quietly(run.quantized);
    {
      std::ofstream out(path_of(runs_path()), std::ios::binary | std::ios::app);
      if (out) out << to_json(rec).dump() << '\n';
      else PRAG_WARN("tune: cannot append to %s", runs_path().c_str());
    }
    std::lock_guard<std::mutex> lock(mutex);
    runs.push_back(rec);
    run.record = rec;
    return failed ? Outcome::Failed : Outcome::Finished;
  }

  Outcome fail(const std::string& error) {
    PRAG_WARN("tune: %s", error.c_str());
    run.record.error = error;
    return finish(true);
  }

  Outcome run_stages() {
    const std::string base = settings.get<std::string>("models.tuning_base", "");
    if (!available) return Outcome::Paused;  // run_now refuses this case; defensive

    if (!run.on) {
      run = Active{};
      run.on = true;
      std::lock_guard<std::mutex> lock(mutex);
      char id[32];
      std::snprintf(id, sizeof id, "run-%04d", next_run_number++);
      run.record.id = id;
      run.record.started = iso_now();
      run.dir = (path_of(data_dir) / id).generic_string();
      run.f32 = run.dir + "/base-f32.gguf";
      run.tuned = run.dir + "/tuned-f32.gguf";
    }
    std::error_code ec;
    stdfs::create_directories(path_of(run.dir), ec);
    RunRecord& rec = run.record;

    // 1. Collecting ---------------------------------------------------------
    if (run.done < 1) {
      set_stage(Stage::Collecting, "Building training examples", 0.0);
      CorpusView view;
      if (!hooks.corpus || !hooks.corpus(&view)) return fail("no index is available");
      ExampleOptions eo;
      eo.max_examples = settings.get<std::size_t>("tune.max_examples", eo.max_examples);
      eo.distractors = settings.get<std::size_t>("tune.distractors", eo.distractors);
      eo.oracle_fraction = settings.get<double>("tune.oracle_fraction", eo.oracle_fraction);
      eo.holdout_fraction = settings.get<double>("tune.holdout_fraction", eo.holdout_fraction);
      run.examples = build_examples(view, eo);
      if (hooks.feedback_examples) {
        std::vector<std::string> known;
        for (const Example& e : run.examples.train) known.push_back(e.id);
        std::sort(known.begin(), known.end());
        for (Example& e : hooks.feedback_examples()) {
          if (std::binary_search(known.begin(), known.end(), e.id)) continue;
          run.examples.train.push_back(std::move(e));
        }
      }
      const std::size_t minimum = settings.get<std::size_t>("tune.min_examples", 64);
      if (run.examples.train.size() < minimum)
        return fail("too few training examples (" + std::to_string(run.examples.train.size()) + " < " +
                    std::to_string(minimum) + ")");
      run.text = render_training_text(run.examples, view, hooks.chat_template);
      rec.examples = run.examples.train.size();
      run.done = 1;
      {
        std::lock_guard<std::mutex> lock(mutex);
        pending = run.examples.train.size();
      }
      if (cancel.load()) return Outcome::Paused;
    }

    // 2. Training -----------------------------------------------------------
    if (run.done < 2) {
      set_stage(Stage::Training, "Converting the base model to float32", 0.05);
      std::string error;
      if (!steps.convert_f32(base, run.f32, &error)) return fail("float32 conversion failed: " + error);
      run.done = 2;
      if (cancel.load()) return Outcome::Paused;
    }
    if (run.done < 3) {
      set_stage(Stage::Training, "Fine-tuning", 0.15);
      llm::FineTuneOptions fo;
      fo.input = run.f32;
      fo.output = run.tuned;
      fo.epochs = settings.get<int>("tune.epochs", fo.epochs);
      fo.trainable_blocks = settings.get<int>("tune.trainable_blocks", fo.trainable_blocks);
      fo.threads = settings.get<int>("models.threads", 0);
      const llm::FineTuneReport report = steps.fine_tune(fo, run.text, progress_callback(0.15, 0.6));
      rec.epochs = report.epochs_run;
      rec.train_loss = report.train_loss;
      rec.validation_loss = report.validation_loss;
      if (report.cancelled) return Outcome::Paused;
      if (!report.ok) return fail("fine-tuning failed: " + report.error);
      run.done = 3;
      if (cancel.load()) return Outcome::Paused;
    }

    // 3. Calibrating --------------------------------------------------------
    if (run.done < 4) {
      set_stage(Stage::Calibrating, "Collecting the importance matrix", 0.6);
      llm::ImportanceOptions io;
      io.model = run.tuned;
      io.threads = settings.get<int>("models.threads", 0);
      std::string error;
      if (!steps.compute_importance(io, run.text, &run.importance, progress_callback(0.6, 0.75), &error)) {
        if (cancel.load()) return Outcome::Paused;
        return fail("calibration failed: " + error);
      }
      run.done = 4;
      if (cancel.load()) return Outcome::Paused;
    }

    // 4. Quantizing ---------------------------------------------------------
    if (run.done < 5) {
      set_stage(Stage::Quantizing, "Quantizing", 0.75);
      llm::QuantizeOptions qo;
      qo.type = settings.get<std::string>("tune.quantization", "Q4_K_M");
      qo.input = run.tuned;
      qo.output = run.dir + "/model-" + qo.type + ".gguf";
      qo.threads = settings.get<int>("models.threads", 0);
      qo.importance = run.importance.empty() ? nullptr : &run.importance;
      std::string error;
      if (!steps.quantize(qo, &error)) return fail("quantization failed: " + error);
      run.quantized = qo.output;
      rec.model = qo.output;
      run.done = 5;
      if (cancel.load()) return Outcome::Paused;
    }

    // 5. Evaluating ---------------------------------------------------------
    if (run.done < 6) {
      set_stage(Stage::Evaluating, "Evaluating the candidate", 0.9);
      GateInput gi;
      std::string error;
      const std::string serving = hooks.serving_model ? hooks.serving_model() : std::string();
      if (!hooks.score_examples) return fail("no scoring hook");
      if (!hooks.score_examples(serving, run.examples.holdout, &gi.base_nll, &error))
        return fail("scoring the serving model failed: " + error);
      if (cancel.load()) return Outcome::Paused;
      if (!hooks.score_examples(run.quantized, run.examples.holdout, &gi.candidate_nll, &error)) {
        gi.candidate_loads = false;
        PRAG_WARN("tune: candidate scoring failed: %s", error.c_str());
      }
      if (cancel.load()) return Outcome::Paused;
      if (gi.candidate_loads && hooks.general_nll) {
        if (!hooks.general_nll(serving, &gi.general_base_nll, &error) ||
            !hooks.general_nll(run.quantized, &gi.general_candidate_nll, &error))
          return fail("general probe failed: " + error);
        if (cancel.load()) return Outcome::Paused;
      }
      if (gi.candidate_loads && hooks.quantization_kl) {
        if (!hooks.quantization_kl(run.tuned, run.quantized, &gi.quantization_kl, &error))
          return fail("quantization KL failed: " + error);
      }
      GatePolicy gp;
      gp.alpha = settings.get<double>("tune.gate_alpha", gp.alpha);
      gp.margin = settings.get<double>("tune.gate_margin", gp.margin);
      gp.max_kl = settings.get<double>("tune.gate_kl", gp.max_kl);
      gp.min_examples = settings.get<std::size_t>("tune.gate_min_examples", gp.min_examples);
      rec.gate = evaluate_gate(gi, gp);
      run.done = 6;
    }

    // 6. Promoting ----------------------------------------------------------
    if (rec.gate.pass) {
      set_stage(Stage::Promoting, "Promoting the candidate", 0.98);
      std::string error;
      if (!hooks.promote || !hooks.promote(run.quantized, &error)) {
        rec.error = "promotion failed: " + (hooks.promote ? error : std::string("no promote hook"));
        return finish(true);
      }
      rec.promoted = true;
      std::lock_guard<std::mutex> lock(mutex);
      serving_model = run.quantized;
    } else {
      rec.stage = Stage::Evaluating;
    }
    return finish(false);
  }
};

Pipeline::Pipeline(const Settings& settings, std::string data_dir, Hooks hooks)
    : Pipeline(settings, std::move(data_dir), std::move(hooks), Steps{}) {}

Pipeline::Pipeline(const Settings& settings, std::string data_dir, Hooks hooks, Steps steps)
    : impl_(std::make_unique<Impl>(settings, std::move(data_dir), std::move(hooks), std::move(steps))) {}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->started) return;
  impl_->started = true;
  impl_->stopping = false;
  impl_->worker = std::thread([this] { impl_->loop(); });
}

void Pipeline::stop() {
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->started) return;
    impl_->stopping = true;
    impl_->cancel.store(true);
  }
  impl_->cv.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->started = false;
}

void Pipeline::set_enabled(bool enabled) {
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->enabled = enabled;
    if (!impl_->running) impl_->apply_enabled_stage();
    if (!enabled && impl_->running) impl_->cancel.store(true);
  }
  impl_->cv.notify_all();
  if (impl_->hooks.enabled_changed) impl_->hooks.enabled_changed(enabled);
}

bool Pipeline::enabled() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->enabled;
}

void Pipeline::notify_activity() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->last_activity = Clock::now();
  if (impl_->running) impl_->cancel.store(true);
}

bool Pipeline::run_now(std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->available) {
    if (error) *error = "tuning is unavailable: " + impl_->unavailable_reason;
    return false;
  }
  if (impl_->running || impl_->run_requested) {
    if (error) *error = "a tuning run is already in progress";
    return false;
  }
  impl_->running = true;
  impl_->run_requested = true;
  impl_->cancel.store(false);
  impl_->cv.notify_all();
  return true;
}

Status Pipeline::status() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Status s;
  s.available = impl_->available;
  s.enabled = impl_->enabled;
  s.stage = impl_->stage;
  s.progress = impl_->progress;
  s.message = impl_->message;
  s.idle = impl_->idle;
  s.pending_examples = impl_->pending;
  s.serving_model = impl_->serving_model;
  s.base_model = impl_->settings.get<std::string>("models.tuning_base", "");
  s.runs = impl_->runs;
  return s;
}

}  // namespace prag::tune
