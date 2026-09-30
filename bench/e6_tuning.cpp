// E6: fine-tuning and quantisation of the serving model. Needs llama.cpp, the
// float16 tuning base and the serving model; otherwise it writes {"skipped": ...}.
#include "common.hpp"
#include "prag/llm/model.hpp"

#if defined(PRAG_HAVE_LLAMA)
#include <filesystem>

#include "llama.h"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/llm/tuning.hpp"
#include "prag/tune/pipeline.hpp"
#endif

namespace bench {

#if !defined(PRAG_HAVE_LLAMA)

int run_e6(Context& ctx) {
  const std::string started = iso_now();
  write_result(ctx, "e6_tuning", started, json{{"requires", "llama.cpp, tuning base model, serving model"}},
               skipped("prag-bench was built without llama.cpp (PRAG_WITH_LLAMA=OFF)"));
  return 0;
}

#else

namespace {

namespace fsys = std::filesystem;
namespace tn = prag::tune;
namespace lm = prag::llm;

// A fixed passage that does not come from the corpus: prose and two short programs.
const char* kGeneralProbe =
    "The committee met on Tuesday to discuss the budget for the coming year. After a long debate about "
    "transportation, libraries and the upkeep of public parks, the members agreed to postpone the final vote "
    "until more data on tax revenue became available. Several residents spoke during the public comment "
    "period, and most of them asked for better bus service in the eastern neighbourhoods. The chairwoman "
    "thanked everyone for their patience and promised a written summary within two weeks.\n\n"
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
    "Photosynthesis is the process by which green plants use sunlight to turn water and carbon dioxide into "
    "sugar and oxygen. It takes place mostly in the leaves, inside small structures called chloroplasts, "
    "which contain the pigment chlorophyll. The oxygen that is released into the air is what most animals "
    "need to breathe.\n\n"
    "SELECT name, COUNT(*) AS orders FROM customers JOIN orders ON orders.customer_id = customers.id "
    "GROUP BY name ORDER BY orders DESC LIMIT 10;\n";

std::string chatml(std::span<const prag::ChatMessage> messages) {
  std::string out;
  for (const auto& m : messages) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
  return out;
}

struct Loss {
  std::vector<double> per_example;
  double mean = kNaN;
  std::string error;
};

std::unique_ptr<lm::LanguageModel> load(Context& ctx, const std::string& path, std::string* error) {
  lm::ModelOptions mo;
  mo.path = path;
  mo.context_tokens = ctx.engine->settings().get<int>("models.chat_context", 4096);
  mo.threads = ctx.engine->settings().get<int>("models.threads", 0);
  return lm::load_model(mo, error);
}

Loss domain_loss(lm::LanguageModel& model, const std::vector<tn::Example>& examples, const tn::CorpusView& view) {
  Loss out;
  for (const tn::Example& ex : examples) {
    const auto messages = view.prompt(ex.question, ex.context);
    const double l = answer_loss(model, messages, ex.answer);
    out.per_example.push_back(l);
  }
  out.mean = mean_of(out.per_example);
  return out;
}

double general_loss(lm::LanguageModel& model) {
  const lm::Likelihood l = model.score_text(kGeneralProbe, 256);
  return l.tokens > 0 ? l.nll / l.tokens : kNaN;
}

// ---- KL divergence and top-1 agreement between two models ------------------

struct Session {
  llama_model* model = nullptr;
  llama_context* context = nullptr;
  const llama_vocab* vocab = nullptr;
  int n_vocab = 0;
  ~Session() {
    if (context) llama_free(context);
    if (model) llama_model_free(model);
  }
  bool open(const std::string& path, int window, int threads, std::string* error) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
      *error = "cannot load " + path;
      return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<std::uint32_t>(window);
    cp.n_batch = static_cast<std::uint32_t>(window);
    cp.n_ubatch = static_cast<std::uint32_t>(window);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    context = llama_init_from_model(model, cp);
    if (!context) {
      *error = "cannot create a context for " + path;
      return false;
    }
    vocab = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);
    return true;
  }
  std::vector<llama_token> tokenize(const std::string& text, int limit) const {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<std::int32_t>(text.size()), tokens.data(),
                           static_cast<std::int32_t>(tokens.size()), true, true);
    if (n < 0) {
      tokens.resize(static_cast<std::size_t>(-n));
      n = llama_tokenize(vocab, text.c_str(), static_cast<std::int32_t>(text.size()), tokens.data(),
                         static_cast<std::int32_t>(tokens.size()), true, true);
    }
    tokens.resize(static_cast<std::size_t>(std::max(0, n)));
    if (static_cast<int>(tokens.size()) > limit) tokens.resize(static_cast<std::size_t>(limit));
    return tokens;
  }
  // Logits of every position (n_tokens * n_vocab); empty on failure.
  std::vector<float> logits(const std::vector<llama_token>& tokens) {
    llama_memory_clear(llama_get_memory(context), true);
    const int n = static_cast<int>(tokens.size());
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
      batch.token[i] = tokens[static_cast<std::size_t>(i)];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    std::vector<float> out;
    if (llama_decode(context, batch) == 0) {
      out.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n_vocab));
      for (int i = 0; i < n; ++i) {
        const float* row = llama_get_logits_ith(context, i);
        std::copy(row, row + n_vocab, out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i) * static_cast<std::size_t>(n_vocab)));
      }
    }
    llama_batch_free(batch);
    return out;
  }
};

struct KlResult {
  double mean_kl = kNaN, agreement = kNaN;
  std::size_t tokens = 0;
  std::string error;
};

KlResult kl_between(const std::string& reference, const std::string& other, const std::vector<std::string>& texts,
                    int window, int threads) {
  KlResult out;
  Session a, b;
  if (!a.open(reference, window, threads, &out.error) || !b.open(other, window, threads, &out.error)) return out;
  if (a.n_vocab != b.n_vocab) {
    out.error = "the vocabularies differ";
    return out;
  }
  double kl_sum = 0.0;
  std::size_t agree = 0, count = 0;
  const std::size_t v = static_cast<std::size_t>(a.n_vocab);
  std::vector<double> logp(v), logq(v);
  for (const std::string& text : texts) {
    const auto tokens = a.tokenize(text, window);
    if (tokens.size() < 2) continue;
    const auto la = a.logits(tokens), lb = b.logits(tokens);
    if (la.empty() || lb.empty()) {
      out.error = "decoding failed";
      return out;
    }
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      const float* x = &la[i * v];
      const float* y = &lb[i * v];
      double mx = -1e300, my = -1e300;
      std::size_t ax = 0, ay = 0;
      for (std::size_t j = 0; j < v; ++j) {
        if (x[j] > mx) { mx = x[j]; ax = j; }
        if (y[j] > my) { my = y[j]; ay = j; }
      }
      double sx = 0.0, sy = 0.0;
      for (std::size_t j = 0; j < v; ++j) {
        sx += std::exp(static_cast<double>(x[j]) - mx);
        sy += std::exp(static_cast<double>(y[j]) - my);
      }
      const double lx = mx + std::log(sx), ly = my + std::log(sy);
      double kl = 0.0;
      for (std::size_t j = 0; j < v; ++j) {
        const double lp = static_cast<double>(x[j]) - lx, lq = static_cast<double>(y[j]) - ly;
        kl += std::exp(lp) * (lp - lq);
      }
      kl_sum += kl;
      agree += ax == ay ? 1 : 0;
      ++count;
    }
  }
  if (count) {
    out.mean_kl = kl_sum / D(count);
    out.agreement = D(agree) / D(count);
  }
  out.tokens = count;
  return out;
}

json speed_of(lm::LanguageModel& model, const std::vector<std::vector<prag::ChatMessage>>& prompts) {
  double prompt_tokens = 0, prompt_ms = 0, completion_tokens = 0, completion_ms = 0;
  for (const auto& messages : prompts) {
    lm::GenerateOptions go;
    go.max_tokens = 48;
    go.temperature = 0.0f;
    const lm::GenerateStats g = model.generate(messages, go, [](std::string_view) { return true; });
    prompt_tokens += g.prompt_tokens;
    prompt_ms += g.prompt_ms;
    completion_tokens += g.completion_tokens;
    completion_ms += g.completion_ms;
  }
  return json{{"prompt_tokens_per_second", prompt_ms > 0 ? prompt_tokens / prompt_ms * 1000.0 : kNaN},
              {"generation_tokens_per_second", completion_ms > 0 ? completion_tokens / completion_ms * 1000.0 : kNaN},
              {"prompts", prompts.size()}};
}

}  // namespace

int run_e6(Context& ctx) {
  const std::string started = iso_now();
  const json settings{{"training", "examples of the calibration and held-out functions; test functions never appear"},
                      {"general_probe", "fixed passage of prose and short programs, 256-token windows"},
                      {"kl", "mean KL(reference || quantised) per token over the rendered held-out examples, 256-token windows"},
                      {"peak_memory", "process-wide peak working set, monotone over the run"}};
  if (!lm::available()) {
    write_result(ctx, "e6_tuning", started, settings, skipped("llama.cpp is not available"));
    return 0;
  }
  const auto& s = ctx.engine->settings();
  const std::string base_f16 = s.get<std::string>("models.tuning_base", "");
  const std::string serving = s.get<std::string>("models.chat", "");
  if (base_f16.empty() || !fsys::exists(base_f16) || serving.empty() || !fsys::exists(serving)) {
    write_result(ctx, "e6_tuning", started, settings,
                 skipped("models.tuning_base and models.chat must name existing GGUF files"));
    return 0;
  }
  const auto& c = *ctx.corpus;
  const std::string work = ctx.repo_dir + "/build/bench_tune";
  fsys::create_directories(work);
  json body = json::object();
  const int threads = s.get<int>("models.threads", 0) > 0 ? s.get<int>("models.threads", 0)
                                                           : static_cast<int>(std::max(1u, std::min(8u, ctx.engine->executor().threads())));

  // Examples.
  tn::CorpusView view;
  view.units = c.units;
  view.distractors = [&c](UnitId unit, std::size_t count) {
    std::vector<UnitId> out;
    for (const UnitId v : c.siblings.of(unit)) {
      if (out.size() >= count) break;
      out.push_back(v);
    }
    return out;
  };
  view.prompt = [&ctx](std::string_view question, std::span<const UnitId> ids) {
    return messages_for(ctx, std::string(question), std::vector<UnitId>(ids.begin(), ids.end()), false);
  };
  tn::ExampleOptions eo;
  eo.max_examples = ctx.quick() ? 24 : 400;
  eo.distractors = s.get<std::size_t>("tune.distractors", 4);
  eo.oracle_fraction = s.get<double>("tune.oracle_fraction", 0.8);
  eo.seed = ctx.seed + 61;
  tn::ExampleSet set = tn::build_examples(view, eo);
  const auto& splits = unit_splits(ctx);
  std::size_t removed = 0;
  {
    std::vector<tn::Example> kept;
    for (auto& ex : set.train) {
      if (ex.oracle != prag::kNoUnit && ex.oracle < splits.size() && splits[ex.oracle] == kTest) {
        ++removed;
        continue;
      }
      kept.push_back(std::move(ex));
    }
    set.train = std::move(kept);
  }
  const std::size_t holdout_cap = ctx.quick() ? 8 : 100;
  if (set.holdout.size() > holdout_cap) set.holdout.resize(holdout_cap);
  body["examples"] = json{{"train", set.train.size()}, {"holdout", set.holdout.size()}, {"train_removed_test_oracle", removed}};
  if (set.train.size() < 4 || set.holdout.empty()) {
    write_result(ctx, "e6_tuning", started, settings, skipped("too few training or held-out examples"));
    return 0;
  }
  const std::string training_text = tn::render_training_text(set, view, chatml);
  std::vector<std::string> kl_texts;
  for (const tn::Example& ex : set.holdout) {
    auto messages = view.prompt(ex.question, ex.context);
    messages.push_back({"assistant", ex.answer});
    kl_texts.push_back(chatml(messages));
    if (kl_texts.size() >= (ctx.quick() ? 4u : 16u)) break;
  }
  std::vector<std::vector<prag::ChatMessage>> speed_prompts;
  for (std::size_t i = 0; i < std::min<std::size_t>(3, set.holdout.size()); ++i)
    speed_prompts.push_back(view.prompt(set.holdout[i].question, set.holdout[i].context));

  std::string error;
  prag::Stopwatch watch;

  // Float32 base.
  const std::string base_f32 = work + "/base_f32.gguf";
  if (!fsys::exists(base_f32)) {
    progress("E6: converting the tuning base to float32 ...");
    lm::QuantizeOptions qo;
    qo.input = base_f16;
    qo.output = base_f32;
    qo.type = "F32";
    qo.threads = threads;
    if (!lm::quantize(qo, &error)) {
      write_result(ctx, "e6_tuning", started, settings, skipped("conversion to float32 failed: " + error));
      return 0;
    }
  }
  body["peak_working_set_bytes_after_conversion"] = peak_working_set_bytes();

  // Base losses.
  Loss base_domain, serving_domain;
  double base_general = kNaN, serving_general = kNaN;
  json serving_speed, base_speed;
  {
    auto m = load(ctx, base_f32, &error);
    if (!m) { write_result(ctx, "e6_tuning", started, settings, skipped("cannot load the float32 base: " + error)); return 0; }
    progress("E6: base losses ...");
    base_domain = domain_loss(*m, set.holdout, view);
    base_general = general_loss(*m);
    base_speed = speed_of(*m, speed_prompts);
  }
  {
    auto m = load(ctx, serving, &error);
    if (m) {
      serving_domain = domain_loss(*m, set.holdout, view);
      serving_general = general_loss(*m);
      serving_speed = speed_of(*m, speed_prompts);
    } else {
      serving_domain.error = error;
    }
  }
  body["base"] = json{{"model", base_f32}, {"domain_loss", base_domain.mean}, {"general_loss", base_general}, {"speed", base_speed}};
  body["serving"] = json{{"model", serving}, {"domain_loss", serving_domain.mean}, {"general_loss", serving_general},
                         {"speed", serving_speed}, {"error", serving_domain.error}};

  // Fine-tuning.
  const std::string tuned = work + "/ft_f32.gguf";
  lm::FineTuneOptions fo;
  fo.input = base_f32;
  fo.output = tuned;
  fo.epochs = ctx.quick() ? 1 : s.get<int>("tune.epochs", 2);
  fo.trainable_blocks = s.get<int>("tune.trainable_blocks", 2);
  fo.threads = threads;
  fo.window = 256;
  watch.reset();
  double last_print = -1.0;
  const lm::FineTuneReport report = lm::fine_tune(fo, training_text, [&](double f, std::string_view) {
    if (f - last_print >= 0.1) {
      last_print = f;
      progress("E6: fine-tuning %.0f %%", f * 100.0);
    }
    return true;
  });
  const double tune_seconds = watch.elapsed_s();
  body["fine_tune"] = json{{"ok", report.ok},
                           {"error", report.error},
                           {"epochs", fo.epochs},
                           {"trainable_blocks", fo.trainable_blocks},
                           {"window", fo.window},
                           {"learning_rate", fo.learning_rate},
                           {"train_loss_per_epoch", report.train_loss},
                           {"validation_loss_per_epoch", report.validation_loss},
                           {"train_windows", report.train_windows},
                           {"validation_windows", report.validation_windows},
                           {"trainable_parameters", report.trainable_parameters},
                           {"total_parameters", report.total_parameters},
                           {"seconds_reported", report.seconds},
                           {"wall_seconds", tune_seconds},
                           {"peak_working_set_bytes_after_training", peak_working_set_bytes()}};
  if (!report.ok) {
    write_result(ctx, "e6_tuning", started, settings, std::move(body));
    return 1;
  }
  Loss tuned_domain;
  double tuned_general = kNaN;
  {
    auto m = load(ctx, tuned, &error);
    if (!m) { body["fine_tuned_error"] = error; write_result(ctx, "e6_tuning", started, settings, std::move(body)); return 1; }
    tuned_domain = domain_loss(*m, set.holdout, view);
    tuned_general = general_loss(*m);
    body["fine_tuned_f32"] = json{{"domain_loss", tuned_domain.mean}, {"general_loss", tuned_general}, {"speed", speed_of(*m, speed_prompts)}};
  }

  // Importance matrix from repository text in the serving prompt format.
  lm::ImportanceMatrix matrix;
  lm::ImportanceOptions io;
  io.model = tuned;
  io.max_windows = ctx.quick() ? 8 : 64;
  io.threads = threads;
  watch.reset();
  const bool have_matrix = lm::compute_importance(io, training_text, &matrix, nullptr, &error);
  body["importance"] = json{{"ok", have_matrix}, {"error", error}, {"tokens", matrix.tokens}, {"matrices", matrix.names.size()},
                            {"seconds", watch.elapsed_s()}};

  // Quantisation.
  json quantised = json::array();
  struct Candidate { std::string type, path; bool imp; std::vector<double> per_example; double general; double kl; };
  Candidate gate_candidate{"", "", false, {}, kNaN, kNaN};
  for (const char* type : {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}) {
    for (const bool imp : {false, true}) {
      if (imp && !have_matrix) continue;
      const std::string out = work + "/ft_" + type + (imp ? "_imp" : "") + ".gguf";
      lm::QuantizeOptions qo;
      qo.input = tuned;
      qo.output = out;
      qo.type = type;
      qo.threads = threads;
      qo.importance = imp ? &matrix : nullptr;
      json row{{"type", type}, {"importance_matrix", imp}};
      watch.reset();
      std::string qerror;
      if (!lm::quantize(qo, &qerror)) {
        row["error"] = qerror;
        quantised.push_back(row);
        continue;
      }
      row["quantize_seconds"] = watch.elapsed_s();
      row["file_bytes"] = fsys::file_size(out);
      auto m = load(ctx, out, &qerror);
      if (!m) {
        row["error"] = qerror;
        quantised.push_back(row);
        continue;
      }
      const Loss l = domain_loss(*m, set.holdout, view);
      row["domain_loss"] = l.mean;
      row["general_loss"] = general_loss(*m);
      row["speed"] = speed_of(*m, speed_prompts);
      m.reset();
      const KlResult kl = kl_between(tuned, out, kl_texts, 256, threads);
      row["mean_kl_per_token"] = kl.mean_kl;
      row["top1_agreement"] = kl.agreement;
      row["kl_tokens"] = kl.tokens;
      row["kl_error"] = kl.error;
      if (std::string(type) == "Q4_K_M" && imp) gate_candidate = {type, out, imp, l.per_example, row["general_loss"].get<double>(), kl.mean_kl};
      if (std::string(type) == "Q4_K_M" && !imp && gate_candidate.path.empty()) gate_candidate = {type, out, imp, l.per_example, row["general_loss"].get<double>(), kl.mean_kl};
      quantised.push_back(std::move(row));
      progress("E6: %s%s done", type, imp ? " + importance" : "");
    }
  }
  body["quantised"] = quantised;
  body["peak_working_set_bytes_final"] = peak_working_set_bytes();

  // Promotion gate: the serving model against the quantised fine-tuned candidate.
  if (!gate_candidate.path.empty() && !serving_domain.per_example.empty()) {
    tn::GateInput gi;
    gi.base_nll = serving_domain.per_example;
    gi.candidate_nll = gate_candidate.per_example;
    gi.general_base_nll = serving_general;
    gi.general_candidate_nll = gate_candidate.general;
    gi.quantization_kl = gate_candidate.kl;
    tn::GatePolicy policy;
    policy.alpha = s.get<double>("tune.gate_alpha", 0.05);
    policy.margin = s.get<double>("tune.gate_margin", 0.02);
    policy.max_kl = s.get<double>("tune.gate_kl", 0.05);
    const tn::GateResult gate = tn::evaluate_gate(gi, policy);
    body["gate"] = json{{"candidate", gate_candidate.path},
                        {"candidate_type", gate_candidate.type},
                        {"candidate_importance_matrix", gate_candidate.imp},
                        {"base_model", serving},
                        {"examples", gi.base_nll.size()},
                        {"policy", json{{"alpha", policy.alpha}, {"margin", policy.margin}, {"max_kl", policy.max_kl}, {"min_examples", policy.min_examples}}},
                        {"result", tn::to_json(gate)}};
  } else {
    body["gate"] = skipped("no quantised candidate or no serving-model losses");
  }
  write_result(ctx, "e6_tuning", started,
               json{{"settings", settings},
                    {"epochs", fo.epochs},
                    {"trainable_blocks", fo.trainable_blocks},
                    {"max_examples", eo.max_examples},
                    {"holdout_cap", holdout_cap},
                    {"base_model", base_f16},
                    {"serving_model", serving},
                    {"chat_template_for_training", "ChatML rendering by the driver"}},
               std::move(body));
  return 0;
}

#endif

}  // namespace bench
