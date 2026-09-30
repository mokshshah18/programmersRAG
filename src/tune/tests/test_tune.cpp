// Tests of the tune module. No model is needed: the heavy steps and the engine hooks are fakes.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "prag/core/hash.hpp"
#include "prag/llm/model.hpp"
#include "prag/tune/pipeline.hpp"

using namespace prag;
using namespace prag::tune;
namespace stdfs = std::filesystem;

namespace {

// ---- corpus -------------------------------------------------------------------

std::vector<SourceUnit> make_units(std::size_t n) {
  std::vector<SourceUnit> units(n);
  for (std::size_t i = 0; i < n; ++i) {
    SourceUnit& u = units[i];
    u.id = static_cast<UnitId>(i);
    u.path = "/repo/file" + std::to_string(i % 7) + ".c";
    u.language = "c";
    u.kind = UnitKind::Function;
    u.name = "func_" + std::to_string(i);
    u.signature = "int func_" + std::to_string(i) + "(int x)";
    u.doc = i % 10 == 9 ? "short doc" : "Computes the value number " + std::to_string(i) + " for the caller. More text follows here.";
    u.text = u.doc + "\n" + u.signature + " { return x + " + std::to_string(i) + "; }";
    u.content_hash = hash64(u.text);
  }
  return units;
}

CorpusView make_view(const std::vector<SourceUnit>& units) {
  CorpusView view;
  view.units = units;
  const std::size_t n = units.size();
  view.distractors = [n](UnitId unit, std::size_t count) {
    std::vector<UnitId> ids;
    for (std::size_t k = 1; k <= count; ++k) ids.push_back(static_cast<UnitId>((unit + k * 3) % n));
    return ids;
  };
  view.prompt = [](std::string_view question, std::span<const UnitId> context) {
    std::string text;
    for (UnitId id : context) text += "unit " + std::to_string(id) + "\n";
    return std::vector<ChatMessage>{{"system", "You answer questions about code."},
                                    {"user", text + std::string(question)}};
  };
  return view;
}

bool wait_for(const std::function<bool()>& pred, double seconds = 15.0) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < end) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return pred();
}

// ---- pipeline fixture -----------------------------------------------------------

struct Fixture {
  std::vector<SourceUnit> units = make_units(300);
  std::string dir;
  Settings settings;
  Hooks hooks;
  Steps steps;

  std::atomic<int> converts{0}, fine_tunes{0}, importances{0}, quantizes{0}, promotes{0};
  std::atomic<bool> candidate_better{true};
  std::atomic<bool> block_fine_tune{false};     // fine_tune waits until released or cancelled
  std::atomic<bool> in_fine_tune{false};
  std::atomic<bool> cancel_first_fine_tune{false};
  std::string serving = "serving.gguf";
  std::mutex serving_mutex;
  std::vector<bool> enabled_events;

  explicit Fixture(const char* name) {
    dir = (stdfs::temp_directory_path() / (std::string("prag_t_") + name)).string();
    std::error_code ec;
    stdfs::remove_all(dir, ec);
    stdfs::create_directories(dir);
    settings.set("models.tuning_base", std::string("base.gguf"));
    settings.set("tune.min_examples", 10);
    settings.set("tune.gate_min_examples", 10);
    settings.set("tune.poll_seconds", 0.05);

    hooks.corpus = [this](CorpusView* v) {
      *v = make_view(units);
      return true;
    };
    hooks.serving_model = [this] {
      std::lock_guard<std::mutex> lock(serving_mutex);
      return serving;
    };
    hooks.score_examples = [this](const std::string& model, std::span<const Example> ex, std::vector<double>* nll,
                                  std::string*) {
      nll->clear();
      const bool candidate = model != "serving.gguf" && model.find("base") == std::string::npos &&
                             model.find("model-") != std::string::npos;
      for (std::size_t i = 0; i < ex.size(); ++i) {
        const double noise = 0.01 * static_cast<double>(i % 5);
        nll->push_back(2.0 + noise - (candidate ? (candidate_better ? 0.2 : -0.2) : 0.0));
      }
      return true;
    };
    hooks.promote = [this](const std::string& path, std::string*) {
      ++promotes;
      std::lock_guard<std::mutex> lock(serving_mutex);
      serving = path;
      return true;
    };
    hooks.general_nll = [](const std::string&, double* nll, std::string*) {
      *nll = 2.4;
      return true;
    };
    hooks.quantization_kl = [](const std::string&, const std::string&, double* kl, std::string*) {
      *kl = 0.01;
      return true;
    };
    hooks.enabled_changed = [this](bool e) { enabled_events.push_back(e); };

    steps.needs_runtime = false;
    steps.convert_f32 = [this](const std::string&, const std::string&, std::string*) {
      ++converts;
      return true;
    };
    steps.fine_tune = [this](const llm::FineTuneOptions&, std::string_view text, const llm::ProgressCallback& progress) {
      const int call = ++fine_tunes;
      llm::FineTuneReport report;
      if (text.empty()) {
        report.error = "empty training text";
        return report;
      }
      const bool cancel_here = cancel_first_fine_tune && call == 1;
      if (cancel_here || block_fine_tune) {
        in_fine_tune = true;
        while (true) {
          if (!progress(0.5, "training")) {
            report.cancelled = true;
            in_fine_tune = false;
            return report;
          }
          if (!cancel_here && !block_fine_tune) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        in_fine_tune = false;
      }
      progress(1.0, "done");
      report.ok = true;
      report.epochs_run = 2;
      report.train_loss = {1.9, 1.6};
      report.validation_loss = {1.8, 1.7};
      return report;
    };
    steps.compute_importance = [this](const llm::ImportanceOptions&, std::string_view, llm::ImportanceMatrix* out,
                                      const llm::ProgressCallback& progress, std::string*) {
      ++importances;
      out->names = {"w"};
      out->values = {{1.0f}};
      return progress(1.0, "done");
    };
    steps.quantize = [this](const llm::QuantizeOptions&, std::string*) {
      ++quantizes;
      return true;
    };
  }

  ~Fixture() {
    std::error_code ec;
    stdfs::remove_all(dir, ec);
  }

  std::unique_ptr<Pipeline> make() { return std::make_unique<Pipeline>(settings, dir, hooks, steps); }
};

bool run_when_possible(Pipeline& p) {
  return wait_for([&] { return p.run_now(); });
}

}  // namespace

// ---- examples -------------------------------------------------------------------

PRAG_TEST(examples_deterministic_and_well_formed) {
  const auto units = make_units(300);
  const CorpusView view = make_view(units);
  ExampleOptions options;
  const ExampleSet a = build_examples(view, options);
  const ExampleSet b = build_examples(view, options);
  REQUIRE(!a.train.empty());
  REQUIRE(a.train.size() == b.train.size() && a.holdout.size() == b.holdout.size());
  for (std::size_t i = 0; i < a.train.size(); ++i) {
    CHECK_EQ(a.train[i].id, b.train[i].id);
    CHECK_EQ(a.train[i].answer, b.train[i].answer);
    CHECK(a.train[i].context == b.train[i].context);
  }
  ExampleOptions other = options;
  other.seed = 99;
  CHECK(build_examples(view, other).train.front().id != a.train.front().id);

  // Units with short documentation yield nothing: 30 of 300 are short.
  CHECK_EQ(a.train.size() + a.holdout.size(), std::size_t{270});
  std::set<std::string> ids;
  for (const Example& e : a.train) ids.insert(e.id);
  for (const Example& e : a.holdout) ids.insert(e.id);
  CHECK_EQ(ids.size(), std::size_t{270});
  for (const Example& e : a.train) {
    CHECK(e.oracle % 10 != 9);
    CHECK(!e.answer.empty());
    const bool in_context = std::find(e.context.begin(), e.context.end(), e.oracle) != e.context.end();
    CHECK_EQ(in_context, e.oracle_present);
    if (!e.oracle_present) CHECK_EQ(e.answer, std::string("The supplied code does not contain the answer."));
    else CHECK(e.answer.find(units[e.oracle].name) != std::string::npos);
    CHECK(e.context.size() <= options.distractors + 1);
  }
}

PRAG_TEST(examples_oracle_fraction_and_holdout_split) {
  const auto units = make_units(300);
  const CorpusView view = make_view(units);
  for (double fraction : {0.5, 0.8, 1.0}) {
    ExampleOptions options;
    options.oracle_fraction = fraction;
    const ExampleSet set = build_examples(view, options);
    for (const auto* part : {&set.train, &set.holdout}) {
      std::size_t present = 0;
      for (const Example& e : *part) present += e.oracle_present ? 1 : 0;
      CHECK_NEAR(static_cast<double>(present) / static_cast<double>(part->size()), fraction,
                 1.5 / static_cast<double>(part->size()));
    }
  }

  ExampleOptions options;
  options.holdout_fraction = 0.15;
  const ExampleSet set = build_examples(view, options);
  CHECK_NEAR(static_cast<double>(set.holdout.size()) / static_cast<double>(set.train.size() + set.holdout.size()),
             0.15, 0.01);
  std::set<UnitId> train_oracles, holdout_oracles;
  for (const Example& e : set.train) train_oracles.insert(e.oracle);
  for (const Example& e : set.holdout) holdout_oracles.insert(e.oracle);
  for (UnitId id : holdout_oracles) CHECK(!train_oracles.count(id));
  // Held-out units never appear anywhere in a training prompt.
  for (const Example& e : set.train)
    for (UnitId id : e.context) CHECK(!holdout_oracles.count(id));

  // max_examples is respected.
  options.max_examples = 50;
  const ExampleSet small = build_examples(view, options);
  CHECK(small.train.size() + small.holdout.size() <= 50);
  CHECK(!small.holdout.empty());
}

PRAG_TEST(training_text_renders_train_examples_only) {
  const auto units = make_units(60);
  const CorpusView view = make_view(units);
  const ExampleSet set = build_examples(view, ExampleOptions{});
  const std::string text = render_training_text(
      set, view, [](std::span<const ChatMessage> m) {
        std::string s;
        for (const ChatMessage& c : m) s += "[" + c.role + "]" + c.content + "\n";
        return s;
      });
  for (const Example& e : set.train) CHECK(text.find(e.answer) != std::string::npos);
  for (const Example& e : set.holdout)
    if (e.oracle_present) CHECK(text.find(e.answer) == std::string::npos);
  CHECK(text.find("[assistant]") != std::string::npos);
  CHECK(!render_training_text(set, view, nullptr).empty());
}

// ---- numerics and gate ----------------------------------------------------------

PRAG_TEST(student_t_known_values) {
  CHECK_NEAR(student_t_cdf(0.0, 7), 0.5, 1e-12);
  CHECK_NEAR(student_t_cdf(1.0, 1), 0.75, 1e-9);               // Cauchy
  CHECK_NEAR(student_t_cdf(-1.0, 1), 0.25, 1e-9);
  CHECK_NEAR(student_t_cdf(1.0, 2), 0.5 + 1.0 / (2.0 * std::sqrt(3.0)), 1e-9);
  CHECK_NEAR(student_t_cdf(12.7062047, 1), 0.975, 1e-6);
  CHECK_NEAR(student_t_cdf(2.2281389, 10), 0.975, 1e-6);
  CHECK_NEAR(student_t_cdf(1.8124611, 10), 0.95, 1e-6);
  CHECK_NEAR(student_t_cdf(2.2621572, 9), 0.975, 1e-6);
  CHECK_NEAR(student_t_cdf(-2.0, 10), 1.0 - student_t_cdf(2.0, 10), 1e-12);
  CHECK_NEAR(1.0 - student_t_cdf(2.0, 10), 0.036694, 1e-5);
  CHECK_NEAR(student_t_cdf(1.959964, 100000), 0.975, 1e-4);  // approaches the normal law
  CHECK_NEAR(regularized_incomplete_beta(1, 1, 0.37), 0.37, 1e-12);
  CHECK_NEAR(regularized_incomplete_beta(2, 2, 0.5), 0.5, 1e-12);
  CHECK_NEAR(regularized_incomplete_beta(2, 3, 0.3), 6 * 0.09 - 8 * 0.027 + 3 * 0.0081, 1e-12);
  CHECK_NEAR(regularized_incomplete_beta(0.5, 0.5, 0.5), 0.5, 1e-12);
}

namespace {
GateInput gate_input(const std::vector<double>& diff, double base = 2.0) {
  GateInput in;
  for (double d : diff) {
    in.base_nll.push_back(base);
    in.candidate_nll.push_back(base - d);
  }
  in.general_base_nll = 2.4;
  in.general_candidate_nll = 2.4;
  in.quantization_kl = 0.01;
  return in;
}
GatePolicy small_policy() {
  GatePolicy p;
  p.min_examples = 2;
  return p;
}
}  // namespace

PRAG_TEST(gate_p_value_matches_paired_t_test) {
  // differences 1..5: mean 3, sd sqrt(2.5), t = 4.2426, df 4.
  GateResult r = evaluate_gate(gate_input({1, 2, 3, 4, 5}), small_policy());
  CHECK_NEAR(r.p_value, 1.0 - student_t_cdf(3.0 / (std::sqrt(2.5) / std::sqrt(5.0)), 4), 1e-12);
  CHECK(r.p_value > 0.005 && r.p_value < 0.01);
  CHECK(r.pass);
  CHECK(r.reason.empty());
  CHECK_NEAR(r.domain_nll_base, 2.0, 1e-12);
  CHECK_NEAR(r.domain_nll_candidate, -1.0, 1e-12);

  // differences 1,-1,2,0,3: mean 1, sd sqrt(2.5), t = sqrt(2), df 4, one-sided p about 0.115.
  r = evaluate_gate(gate_input({1, -1, 2, 0, 3}), small_policy());
  CHECK_NEAR(r.p_value, 0.1151, 0.001);
  CHECK(!r.pass);
  CHECK(!r.reason.empty());

  // A worse candidate has a p-value above one half.
  r = evaluate_gate(gate_input({-1, -2, -1, -3}), small_policy());
  CHECK(r.p_value > 0.5);
  CHECK(!r.pass);

  // Constant improvement: zero variance.
  r = evaluate_gate(gate_input({0.1, 0.1, 0.1, 0.1}), small_policy());
  CHECK_EQ(r.p_value, 0.0);
  CHECK(r.pass);
  r = evaluate_gate(gate_input({0.0, 0.0, 0.0}), small_policy());
  CHECK_EQ(r.p_value, 1.0);
  CHECK(!r.pass);

  // The test is one-sided at the requested level.
  GatePolicy strict = small_policy();
  strict.alpha = 0.005;
  CHECK(!evaluate_gate(gate_input({1, 2, 3, 4, 5}), strict).pass);
}

PRAG_TEST(gate_failure_reasons) {
  const std::vector<double> good = {0.3, 0.2, 0.25, 0.3, 0.22, 0.28};
  GatePolicy policy = small_policy();
  CHECK(evaluate_gate(gate_input(good), policy).pass);

  GateInput in = gate_input(good);
  in.candidate_loads = false;
  GateResult r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("load") != std::string::npos);

  in = gate_input(good);
  in.candidate_nll.pop_back();
  r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("different examples") != std::string::npos);

  policy.min_examples = 20;
  r = evaluate_gate(gate_input(good), policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("too few") != std::string::npos);
  policy.min_examples = 2;

  in = gate_input({0.0, 0.01, -0.01, 0.0, 0.01, 0.0});
  r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("domain loss") != std::string::npos);

  in = gate_input(good);
  in.general_candidate_nll = 2.4 + 0.05;
  r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("general") != std::string::npos);
  in.general_candidate_nll = 2.4 + 0.019;  // within the margin
  CHECK(evaluate_gate(in, policy).pass);

  in = gate_input(good);
  in.quantization_kl = 0.2;
  r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("quantisation") != std::string::npos);
  CHECK_NEAR(r.quantization_kl, 0.2, 1e-12);

  in = gate_input(good);
  in.candidate_nll[0] = std::nan("");
  r = evaluate_gate(in, policy);
  CHECK(!r.pass);
  CHECK(r.reason.find("non-finite") != std::string::npos);
}

PRAG_TEST(probe_idle_reports_reasons) {
  IdlePolicy policy;
  policy.idle_seconds = 300;
  IdleState s = probe_idle(5.0, policy);
  CHECK(!s.idle);
  CHECK(s.reason.find("user active") != std::string::npos);
  CHECK(s.cpu_load >= 0.0 && s.cpu_load <= 1.0);
  policy.max_cpu_load = 1.0;  // never exceeded
  policy.require_ac_power = false;
  s = probe_idle(1000.0, policy);
  CHECK(s.idle);
  CHECK(s.reason.find("idle for") != std::string::npos);
}

// ---- pipeline -------------------------------------------------------------------

PRAG_TEST(state_machine_promotes_and_records_run) {
  Fixture f("promote");
  auto p = f.make();
  p->start();
  REQUIRE(p->status().available);
  CHECK_EQ(p->status().serving_model, std::string("serving.gguf"));
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));

  const Status s = p->status();
  CHECK_EQ(s.runs.size(), std::size_t{1});
  const RunRecord& r = s.runs[0];
  CHECK_EQ(r.id, std::string("run-0001"));
  CHECK(r.promoted);
  CHECK(r.error.empty());
  CHECK(r.stage == Stage::Promoting);
  CHECK_EQ(r.epochs, 2);
  CHECK_EQ(r.train_loss.size(), std::size_t{2});
  CHECK(r.gate.pass);
  CHECK(r.gate.p_value < 0.05);
  CHECK(r.examples >= 10);
  CHECK(!r.started.empty() && !r.finished.empty());
  CHECK(r.model.find("model-Q4_K_M.gguf") != std::string::npos);
  CHECK_EQ(f.converts.load(), 1);
  CHECK_EQ(f.fine_tunes.load(), 1);
  CHECK_EQ(f.importances.load(), 1);
  CHECK_EQ(f.quantizes.load(), 1);
  CHECK_EQ(f.promotes.load(), 1);
  CHECK(wait_for([&] { return p->status().serving_model == r.model; }));

  // Persistence: runs.jsonl holds the record and a new pipeline numbers on from it.
  std::ifstream in(stdfs::path(f.dir) / "runs.jsonl");
  std::string line;
  REQUIRE(static_cast<bool>(std::getline(in, line)));
  CHECK(nlohmann::json::parse(line)["promoted"].get<bool>());
  p->stop();
  auto again = f.make();
  CHECK_EQ(again->status().runs.size(), std::size_t{1});
  const Status loaded = again->status();
  CHECK_EQ(loaded.runs[0].id, std::string("run-0001"));
  again->start();
  REQUIRE(run_when_possible(*again));
  REQUIRE(wait_for([&] { return again->status().runs.size() == 2; }));
  const Status after = again->status();
  CHECK_EQ(after.runs[1].id, std::string("run-0002"));
}

PRAG_TEST(failing_gate_leaves_serving_model_unchanged) {
  Fixture f("gate");
  f.candidate_better = false;
  auto p = f.make();
  p->start();
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));
  const Status s = p->status();
  const RunRecord& r = s.runs[0];
  CHECK(!r.promoted);
  CHECK(!r.gate.pass);
  CHECK(!r.gate.reason.empty());
  CHECK(r.error.empty());
  CHECK(r.stage == Stage::Evaluating);
  CHECK_EQ(f.promotes.load(), 0);
  CHECK(wait_for([&] { return p->status().serving_model == "serving.gguf" && p->status().stage != Stage::Evaluating; }));
  CHECK_EQ(f.hooks.serving_model(), std::string("serving.gguf"));
}

PRAG_TEST(failed_stage_is_recorded_with_error) {
  Fixture f("fail");
  f.settings.set("tune.min_examples", 100000);
  auto p = f.make();
  p->start();
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));
  Status s = p->status();
  CHECK(!s.runs[0].promoted);
  CHECK(s.runs[0].error.find("too few") != std::string::npos);
  CHECK(wait_for([&] { return p->status().stage == Stage::Failed; }));
  CHECK_EQ(f.fine_tunes.load(), 0);
}

PRAG_TEST(activity_cancels_training_and_later_run_resumes) {
  Fixture f("cancel");
  f.cancel_first_fine_tune = true;
  auto p = f.make();
  p->start();
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return f.in_fine_tune.load(); }));
  CHECK(p->status().stage == Stage::Training);
  p->notify_activity();
  // Refused until the cancelled run has wound down, then accepted.
  REQUIRE(run_when_possible(*p));
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));

  const Status s = p->status();
  CHECK_EQ(s.runs.size(), std::size_t{1});  // the cancelled attempt was not recorded separately
  CHECK_EQ(s.runs[0].id, std::string("run-0001"));
  CHECK(s.runs[0].promoted);
  CHECK_EQ(f.fine_tunes.load(), 2);
  CHECK_EQ(f.converts.load(), 1);  // the conversion was not repeated
  CHECK_EQ(f.quantizes.load(), 1);
}

PRAG_TEST(activity_is_ignored_when_nothing_runs_and_run_now_refused_while_running) {
  Fixture f("busy");
  f.block_fine_tune = true;
  auto p = f.make();
  p->start();
  p->notify_activity();  // nothing running: must not poison the next run
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return f.in_fine_tune.load(); }));

  std::string error;
  CHECK(!p->run_now(&error));
  CHECK(!error.empty());

  // status() does not wait for the running stage.
  const auto t0 = std::chrono::steady_clock::now();
  const Status s = p->status();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CHECK(ms < 100.0);
  CHECK(s.stage == Stage::Training);
  CHECK(s.progress > 0.0 && s.progress < 1.0);

  f.block_fine_tune = false;
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));
  CHECK(p->status().runs[0].promoted);
}

PRAG_TEST(unavailable_without_tuning_base_or_runtime) {
  Fixture f("unavail");
  f.settings.set("models.tuning_base", std::string());
  auto p = f.make();
  Status s = p->status();
  CHECK(!s.available);
  CHECK(s.stage == Stage::Disabled);
  std::string error;
  CHECK(!p->run_now(&error));
  CHECK(!error.empty());

  p->set_enabled(true);
  s = p->status();
  CHECK(s.enabled);  // the wish is kept, and reported to the owner
  CHECK(s.stage == Stage::Disabled);
  CHECK(s.message.find("tuning_base") != std::string::npos);
  REQUIRE(!f.enabled_events.empty());
  CHECK(f.enabled_events.back());

  // Default steps need llama.cpp; a build without it reports why.
  Settings with_base;
  with_base.set("models.tuning_base", std::string("base.gguf"));
  Pipeline plain(with_base, f.dir, Hooks{});
  CHECK_EQ(plain.status().available, llm::available());
}

PRAG_TEST(enable_disable_moves_between_waiting_and_disabled) {
  Fixture f("toggle");
  auto p = f.make();
  CHECK(!p->enabled());
  CHECK(p->status().stage == Stage::Disabled);
  p->set_enabled(true);
  CHECK(p->enabled());
  CHECK(p->status().stage == Stage::Waiting);
  p->set_enabled(false);
  CHECK(p->status().stage == Stage::Disabled);
  CHECK_EQ(f.enabled_events.size(), std::size_t{2});
}

PRAG_TEST(status_json_has_the_specified_fields) {
  Fixture f("json");
  auto p = f.make();
  p->start();
  REQUIRE(p->run_now());
  REQUIRE(wait_for([&] { return !p->status().runs.empty(); }));
  const nlohmann::json j = to_json(p->status());
  for (const char* key : {"available", "enabled", "stage", "progress", "message", "idle", "pending_examples",
                          "serving_model", "base_model", "runs"})
    CHECK(j.contains(key));
  for (const char* key : {"idle", "seconds_since_activity", "cpu_load", "on_ac_power", "reason"})
    CHECK(std::string(key) == "idle" ? j["idle"].contains("idle") : j["idle"].contains(key));
  CHECK(j["stage"].is_string());
  REQUIRE(j["runs"].is_array() && j["runs"].size() == 1);
  const nlohmann::json& run = j["runs"][0];
  for (const char* key : {"id", "started", "finished", "stage", "promoted", "examples", "epochs", "train_loss",
                          "validation_loss", "gate", "model", "seconds", "error"})
    CHECK(run.contains(key));
  for (const char* key : {"pass", "domain_nll_base", "domain_nll_candidate", "p_value", "general_nll_base",
                          "general_nll_candidate", "quantization_kl", "reason"})
    CHECK(run["gate"].contains(key));
  CHECK_EQ(run["stage"].get<std::string>(), std::string("promoting"));
  CHECK(run["train_loss"].is_array());

  // Round trip through run_from_json.
  RunRecord back;
  REQUIRE(run_from_json(run, &back));
  CHECK_EQ(to_json(back).dump(), run.dump());
  CHECK_EQ(std::string(to_string(Stage::Calibrating)), std::string("calibrating"));
}
