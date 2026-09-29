// Slow tests of the llm module: precision conversion, quantisation, importance
// calibration, fine-tuning and throughput. They run only when the environment
// variable PRAG_SLOW_TESTS is 1, and they skip when a model file is missing.
//
// Every case prints its measurements on lines that start with "MEASURE".
// Peak memory is that of the process so far; run one case per process (pass
// its name as the argument) to measure a single step. PRAG_LLM_THREADS
// overrides the 8 threads of the two throughput cases.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/clock.hpp"
#include "prag/llm/model.hpp"
#include "prag/llm/tuning.hpp"

#ifdef PRAG_HAVE_LLAMA
#include "gguf.h"  // to inspect the files the module writes
#endif

#ifdef _WIN32
#include <windows.h>

#include <psapi.h>
#endif

namespace llm = prag::llm;
using prag::ChatMessage;
using prag::Stopwatch;

#ifdef PRAG_HAVE_LLAMA

namespace {

constexpr std::string_view kFallbackCode = R"(bool _mi_strlcpy(char* dest, const char* src, size_t dest_size) {
  if (dest==NULL || src==NULL || dest_size == 0) return (src==NULL || *src==0);
  // copy until end of src, or when dest is (almost) full
  while (*src != 0 && dest_size > 1) {
    *dest++ = *src++;
    dest_size--;
  }
  // always zero terminate
  *dest = 0;
  return (*src == 0);
}

static void chacha_split(mi_random_ctx_t* ctx, uint64_t nonce, mi_random_ctx_t* ctx_new) {
  _mi_memzero(ctx_new, sizeof(*ctx_new));
  ctx_new->weak = ctx->weak;
  _mi_memcpy(ctx_new->input, ctx->input, sizeof(ctx_new->input));
  ctx_new->input[12] = 0;
  ctx_new->input[13] = 0;
  ctx_new->input[14] = (uint32_t)nonce;
  ctx_new->input[15] = (uint32_t)(nonce >> 32);
  chacha_block(ctx_new);
}
)";

int benchmark_threads() {
  const char* value = std::getenv("PRAG_LLM_THREADS");
  const int threads = value ? std::atoi(value) : 0;
  return threads > 0 ? threads : 8;
}

bool enabled() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  return value && std::string_view(value) == "1";
}

void skip(const char* reason) { std::printf("  skipped: %s\n", reason); }

// Read access to the metadata and the tensor data of a GGUF file.
struct GgufFile {
  gguf_context* gguf = nullptr;

  explicit GgufFile(const std::string& path) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf = gguf_init_from_file(path.c_str(), params);
  }
  ~GgufFile() {
    if (gguf) gguf_free(gguf);
  }
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  std::string text(const char* key) const {
    const std::int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return {};
    return gguf_get_val_str(gguf, id);
  }

  std::int64_t number(const char* key) const {
    const std::int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return -1;
    switch (gguf_get_kv_type(gguf, id)) {
      case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
      case GGUF_TYPE_INT32: return gguf_get_val_i32(gguf, id);
      case GGUF_TYPE_UINT64: return static_cast<std::int64_t>(gguf_get_val_u64(gguf, id));
      case GGUF_TYPE_INT64: return gguf_get_val_i64(gguf, id);
      default: return -1;
    }
  }

  std::vector<char> tensor(std::ifstream& file, std::int64_t id) const {
    std::vector<char> bytes(gguf_get_tensor_size(gguf, id));
    file.clear();
    file.seekg(static_cast<std::streamoff>(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id)));
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) bytes.clear();
    return bytes;
  }
};

// True when the model id names the quantisation type, in any letter case.
bool names(const std::string& id, std::string_view type) {
  std::string upper = id;
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return upper.find(type) != std::string::npos;
}

bool file_exists(const std::string& path) {
  std::error_code code;
  return std::filesystem::exists(std::filesystem::path(path), code);
}

double megabytes(const std::string& path) {
  std::error_code code;
  const auto size = std::filesystem::file_size(std::filesystem::path(path), code);
  return code ? 0.0 : static_cast<double>(size) / 1048576.0;
}

double peak_memory_megabytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters;
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<double>(counters.PeakWorkingSetSize) / 1048576.0;
  }
#endif
  return 0.0;
}

std::string model_path(const char* file) { return std::string(PRAG_SOURCE_DIR) + "/models/" + file; }

std::string work_path(const char* file) {
#ifdef PRAG_LLM_TEST_DIR
  const std::filesystem::path directory(PRAG_LLM_TEST_DIR);
#else
  const std::filesystem::path directory = std::filesystem::temp_directory_path() / "prag-llm";
#endif
  std::error_code code;
  std::filesystem::create_directories(directory, code);
  return (directory / file).generic_string();
}

void remove(const std::string& path) {
  std::error_code code;
  std::filesystem::remove(std::filesystem::path(path), code);
}

// C source text: the allocator of the demo repository, else the built-in snippets.
std::string c_text(std::size_t bytes) {
  std::string text;
  for (const char* file : {"alloc.c", "free.c", "page.c", "heap.c"}) {
    std::ifstream in(std::filesystem::path(std::string(PRAG_DEMO_REPO) + "/src/" + file), std::ios::binary);
    if (!in) continue;
    std::ostringstream content;
    content << in.rdbuf();
    text += content.str();
    if (text.size() >= bytes) break;
  }
  while (text.size() < bytes) text += kFallbackCode;
  text.resize(bytes);
  return text;
}

const std::vector<ChatMessage>& question() {
  static const std::vector<ChatMessage> messages = {
      {"system", "You are a concise assistant for C programmers."},
      {"user", "In one sentence, what does the C function malloc do?"},
  };
  return messages;
}

struct Probe {
  bool loaded = false;
  std::string id;
  std::string answer;
  double nats_per_token = 0.0;  // on C text
  double load_seconds = 0.0;
};

// Loads a model, lets it answer and measures how well it predicts C text.
Probe probe(const std::string& path) {
  Probe out;
  llm::ModelOptions options;
  options.path = path;
  options.context_tokens = 512;
  options.threads = 8;
  std::string error;
  const Stopwatch clock;
  const std::unique_ptr<llm::LanguageModel> model = llm::load_model(options, &error);
  out.load_seconds = clock.elapsed_s();
  if (!model) {
    std::fprintf(stderr, "  could not load %s: %s\n", path.c_str(), error.c_str());
    return out;
  }
  out.loaded = true;
  out.id = model->id();
  llm::GenerateOptions generate;
  generate.max_tokens = 24;
  generate.temperature = 0.0f;
  model->generate(question(), generate, [&](std::string_view piece) {
    out.answer += piece;
    return true;
  });
  const llm::Likelihood likelihood = model->score_text(c_text(6000), 256);
  out.nats_per_token = likelihood.tokens > 0 ? likelihood.nll / likelihood.tokens : std::nan("");
  return out;
}

const std::string& fp16_path() {
  static const std::string path = model_path("qwen2.5-coder-0.5b-instruct-fp16.gguf");
  return path;
}

// The float32 base is needed by most cases; it is made once and kept.
bool ensure_f32(std::string* path) {
  *path = work_path("base-f32.gguf");
  if (file_exists(*path)) return true;
  llm::QuantizeOptions options;
  options.input = fp16_path();
  options.output = *path;
  options.type = "F32";
  options.threads = 8;
  std::string error;
  if (!llm::quantize(options, &error)) {
    std::fprintf(stderr, "  conversion to F32 failed: %s\n", error.c_str());
    return false;
  }
  return true;
}

const llm::ImportanceMatrix* importance_matrix(const std::string& model) {
  static llm::ImportanceMatrix matrix;
  static bool tried = false;
  if (!tried) {
    tried = true;
    llm::ImportanceOptions options;
    options.model = model;
    options.window = 128;
    options.max_windows = 4;
    options.threads = 8;
    std::string error;
    if (!llm::compute_importance(options, c_text(16000), &matrix, {}, &error)) {
      std::fprintf(stderr, "  importance calibration failed: %s\n", error.c_str());
    }
  }
  return matrix.empty() ? nullptr : &matrix;
}

void quantize_and_probe(const char* type, bool with_importance) {
  std::string f32;
  REQUIRE(ensure_f32(&f32));
  const llm::ImportanceMatrix* matrix = with_importance ? importance_matrix(f32) : nullptr;
  if (with_importance) REQUIRE(matrix != nullptr);

  const std::string name = std::string("model-") + type + (with_importance ? "-imatrix" : "") + ".gguf";
  llm::QuantizeOptions options;
  options.input = f32;
  options.output = work_path(name.c_str());
  options.type = type;
  options.threads = 8;
  options.importance = matrix;
  remove(options.output);

  std::string error;
  const Stopwatch clock;
  const bool ok = llm::quantize(options, &error);
  const double seconds = clock.elapsed_s();
  if (!ok) std::fprintf(stderr, "  %s\n", error.c_str());
  REQUIRE(ok);
  CHECK(file_exists(options.output));
  CHECK(!file_exists(options.output + ".partial"));
  CHECK(megabytes(options.output) < 0.6 * megabytes(f32));

  const Probe result = probe(options.output);
  CHECK(result.loaded);
  CHECK(!result.answer.empty());
  CHECK(std::isfinite(result.nats_per_token));
  CHECK(result.nats_per_token < 3.0);  // still a language model
  CHECK(names(result.id, type));
  std::printf("MEASURE quantize %s%s: %.1f s, %.1f MiB -> %.1f MiB, %.4f nats/token on C text, peak memory %.0f MiB\n",
              type, with_importance ? " with importance matrix" : "", seconds, megabytes(f32),
              megabytes(options.output), result.nats_per_token, peak_memory_megabytes());
  std::printf("  answer: %s\n", result.answer.c_str());
  remove(options.output);
}

}  // namespace

PRAG_TEST(convert_fp16_to_f32) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("models/qwen2.5-coder-0.5b-instruct-fp16.gguf is missing");

  const std::string output = work_path("base-f32.gguf");
  remove(output);
  llm::QuantizeOptions options;
  options.input = fp16_path();
  options.output = output;
  options.type = "F32";
  options.threads = 8;
  std::string error;
  const Stopwatch clock;
  const bool ok = llm::quantize(options, &error);
  const double seconds = clock.elapsed_s();
  if (!ok) std::fprintf(stderr, "  %s\n", error.c_str());
  REQUIRE(ok);
  CHECK(megabytes(output) > 1.9 * megabytes(fp16_path()));
  CHECK(megabytes(output) < 2.1 * megabytes(fp16_path()));
  std::printf("MEASURE convert F16 -> F32: %.1f s, %.1f MiB -> %.1f MiB, peak memory %.0f MiB\n", seconds,
              megabytes(fp16_path()), megabytes(output), peak_memory_megabytes());

  // Half precision converts to single precision without loss.
  const Probe half = probe(fp16_path());
  const Probe single = probe(output);
  REQUIRE(half.loaded);
  REQUIRE(single.loaded);
  CHECK(names(single.id, "F32"));
  CHECK(!single.answer.empty());
  CHECK_EQ(single.answer, half.answer);
  CHECK_NEAR(single.nats_per_token, half.nats_per_token, 1e-3);
  std::printf("MEASURE float models: F16 %.4f and F32 %.4f nats/token on C text; load %.2f s and %.2f s\n",
              half.nats_per_token, single.nats_per_token, half.load_seconds, single.load_seconds);
  std::printf("  answer: %s\n", single.answer.c_str());
}

PRAG_TEST(quantize_q8_0) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  quantize_and_probe("Q8_0", false);
}

PRAG_TEST(quantize_q4_k_m) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  quantize_and_probe("Q4_K_M", false);
}

PRAG_TEST(importance_matrix_on_c_text) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  std::string f32;
  REQUIRE(ensure_f32(&f32));

  llm::ImportanceOptions options;
  options.model = f32;
  options.window = 128;
  options.max_windows = 4;
  options.threads = 8;

  int calls = 0;
  double last = -1.0;
  bool monotone = true;
  llm::ImportanceMatrix matrix;
  std::string error;
  const Stopwatch clock;
  const bool ok = llm::compute_importance(
      options, c_text(16000), &matrix,
      [&](double fraction, std::string_view) {
        ++calls;
        monotone = monotone && fraction >= last && fraction >= 0.0 && fraction <= 1.0;
        last = fraction;
        return true;
      },
      &error);
  const double seconds = clock.elapsed_s();
  if (!ok) std::fprintf(stderr, "  %s\n", error.c_str());
  REQUIRE(ok);

  CHECK_EQ(matrix.tokens, std::uint64_t{4 * 128});
  CHECK_EQ(matrix.names.size(), matrix.values.size());
  CHECK(calls >= 4);
  CHECK(monotone);
  CHECK_NEAR(last, 1.0, 1e-12);
  // 24 blocks, each with four attention and three feed-forward matrices.
  CHECK_EQ(matrix.names.size(), std::size_t{24 * 7});
  CHECK(std::is_sorted(matrix.names.begin(), matrix.names.end()));
  std::size_t values = 0;
  bool valid = true;
  double largest = 0.0;
  for (std::size_t i = 0; i < matrix.names.size(); ++i) {
    valid = valid && matrix.names[i].rfind("blk.", 0) == 0 && !matrix.values[i].empty();
    for (const float value : matrix.values[i]) {
      valid = valid && std::isfinite(value) && value >= 0.0f;
      largest = std::max(largest, static_cast<double>(value));
    }
    values += matrix.values[i].size();
  }
  CHECK(valid);
  CHECK(largest > 0.0);
  std::printf("MEASURE importance matrix: %.1f s for %llu tokens (%.0f tokens/s), %zu tensors, %zu values, "
              "peak memory %.0f MiB\n",
              seconds, static_cast<unsigned long long>(matrix.tokens), static_cast<double>(matrix.tokens) / seconds,
              matrix.names.size(), values, peak_memory_megabytes());

  // The same text gives the same matrix.
  llm::ImportanceMatrix again;
  REQUIRE(llm::compute_importance(options, c_text(16000), &again, {}, &error));
  CHECK(again.names == matrix.names);
  CHECK(again.values == matrix.values);

  // Cancelling leaves nothing behind.
  llm::ImportanceMatrix cancelled;
  int seen = 0;
  CHECK(!llm::compute_importance(
      options, c_text(16000), &cancelled, [&](double, std::string_view) { return ++seen < 2; }, &error));
  CHECK_EQ(error, std::string("cancelled"));
  CHECK(cancelled.empty());

  // Too little text is an error, not a crash.
  CHECK(!llm::compute_importance(options, "int x;", &cancelled, {}, &error));
  CHECK(error.find("tokens") != std::string::npos);
}

PRAG_TEST(quantize_q8_0_with_importance) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  quantize_and_probe("Q8_0", true);
}

PRAG_TEST(quantize_q4_k_m_with_importance) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  quantize_and_probe("Q4_K_M", true);
}

PRAG_TEST(a_wrong_importance_matrix_is_rejected) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  std::string f32;
  REQUIRE(ensure_f32(&f32));

  llm::ImportanceMatrix matrix;
  matrix.names.push_back("blk.0.attn_q.weight");
  matrix.values.push_back(std::vector<float>(17, 1.0f));  // the tensor has 896 input columns
  llm::QuantizeOptions options;
  options.input = f32;
  options.output = work_path("model-wrong.gguf");
  options.type = "Q4_K_M";
  options.threads = 8;
  options.importance = &matrix;
  std::string error;
  CHECK(!llm::quantize(options, &error));
  CHECK(error.find("imatrix") != std::string::npos || error.find("importance") != std::string::npos);
  CHECK(!file_exists(options.output));
  CHECK(!file_exists(options.output + ".partial"));
  std::printf("  message: %s\n", error.c_str());
}

PRAG_TEST(fine_tune_with_tiny_settings) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  std::string f32;
  REQUIRE(ensure_f32(&f32));

  // About 40 windows of 64 tokens.
  const std::string text = c_text(9000);
  llm::FineTuneOptions options;
  options.input = f32;
  options.output = work_path("tuned-f32.gguf");
  options.window = 64;
  options.epochs = 1;
  options.validation_fraction = 0.1;
  options.learning_rate = 1e-5;
  options.min_learning_rate = 1e-6;
  options.trainable_blocks = 1;
  options.threads = 8;
  options.parse_special = false;
  remove(options.output);

  int calls = 0;
  double last = -1.0;
  bool monotone = true;
  const llm::FineTuneReport report = llm::fine_tune(options, text, [&](double fraction, std::string_view) {
    ++calls;
    monotone = monotone && fraction >= last && fraction >= 0.0 && fraction <= 1.0;
    last = fraction;
    return true;
  });
  if (!report.ok) std::fprintf(stderr, "  %s\n", report.error.c_str());
  REQUIRE(report.ok);
  CHECK(!report.cancelled);
  CHECK(report.error.empty());
  CHECK_EQ(report.epochs_run, 1);
  CHECK(report.train_windows >= 24);
  CHECK(report.validation_windows >= 1);
  CHECK_EQ(report.train_loss.size(), std::size_t{1});
  CHECK_EQ(report.validation_loss.size(), std::size_t{1});
  CHECK(std::isfinite(report.train_loss[0]));
  CHECK(std::isfinite(report.validation_loss[0]));
  CHECK(report.train_loss[0] > 0.0);
  CHECK(report.train_loss[0] < 5.0);
  CHECK(report.trainable_parameters > 0);
  CHECK(report.trainable_parameters < report.total_parameters / 10);
  CHECK(calls >= static_cast<int>(report.train_windows + report.validation_windows));
  CHECK(monotone);
  CHECK(file_exists(options.output));
  CHECK(!file_exists(options.output + ".raw"));
  CHECK(!file_exists(options.output + ".partial"));
  CHECK_NEAR(megabytes(options.output), megabytes(f32), 1.0);
  std::printf("MEASURE fine-tune: %.1f s, %lld training and %lld validation windows of %d tokens (%.0f tokens/s), "
              "training loss %.4f, validation loss %.4f, %llu of %llu parameters, output %.1f MiB, "
              "peak memory %.0f MiB\n",
              report.seconds, static_cast<long long>(report.train_windows),
              static_cast<long long>(report.validation_windows), options.window,
              static_cast<double>((report.train_windows + report.validation_windows) * options.window) / report.seconds,
              report.train_loss[0], report.validation_loss[0],
              static_cast<unsigned long long>(report.trainable_parameters),
              static_cast<unsigned long long>(report.total_parameters), megabytes(options.output),
              peak_memory_megabytes());

  // The result is a complete model: it loads, keeps its chat template and its
  // context length, answers, and has moved towards the training text.
  const Probe base = probe(f32);
  const Probe tuned = probe(options.output);
  REQUIRE(base.loaded);
  REQUIRE(tuned.loaded);
  CHECK(!tuned.answer.empty());
  CHECK(names(tuned.id, "F32"));
  CHECK(std::isfinite(tuned.nats_per_token));
  CHECK(tuned.nats_per_token != base.nats_per_token);
  CHECK(tuned.nats_per_token < base.nats_per_token);
  std::printf("MEASURE fine-tune effect: %.4f -> %.4f nats/token on the training text\n", base.nats_per_token,
              tuned.nats_per_token);
  std::printf("  answer: %s\n", tuned.answer.c_str());

  // File against file: the metadata is that of the base model, and the only
  // tensors that changed are those of the last block.
  {
    const GgufFile before(f32);
    const GgufFile after(options.output);
    REQUIRE(before.gguf != nullptr);
    REQUIRE(after.gguf != nullptr);
    CHECK_EQ(gguf_get_n_kv(after.gguf), gguf_get_n_kv(before.gguf));
    CHECK_EQ(gguf_get_n_tensors(after.gguf), gguf_get_n_tensors(before.gguf));
    CHECK(!after.text("tokenizer.chat_template").empty());
    CHECK_EQ(after.text("tokenizer.chat_template"), before.text("tokenizer.chat_template"));
    CHECK_EQ(after.text("general.architecture"), before.text("general.architecture"));
    const std::string context_key = before.text("general.architecture") + ".context_length";
    CHECK(before.number(context_key.c_str()) >= 2048);  // not the training window
    CHECK_EQ(after.number(context_key.c_str()), before.number(context_key.c_str()));
    CHECK_EQ(after.number("general.file_type"), before.number("general.file_type"));

    std::ifstream before_data(std::filesystem::path(f32), std::ios::binary);
    std::ifstream after_data(std::filesystem::path(options.output), std::ios::binary);
    std::vector<std::string> changed;
    bool only_last_block = true;
    for (std::int64_t i = 0; i < gguf_get_n_tensors(before.gguf); ++i) {
      const char* name = gguf_get_tensor_name(before.gguf, i);
      const std::int64_t j = gguf_find_tensor(after.gguf, name);
      REQUIRE(j >= 0);
      const std::vector<char> a = before.tensor(before_data, i);
      const std::vector<char> b = after.tensor(after_data, j);
      REQUIRE(!a.empty());
      REQUIRE(a.size() == b.size());
      if (a == b) continue;
      changed.emplace_back(name);
      only_last_block = only_last_block && changed.back().rfind("blk.23.", 0) == 0;
    }
    CHECK(only_last_block);
    // A block has twelve tensors. Eight receive a gradient: the key and value
    // projections do not, because llama.cpp lets attention read keys and
    // values from its cache buffer, which the backward pass does not enter.
    CHECK(changed.size() >= 6);
    CHECK(changed.size() <= 12);
    const auto was_trained = [&](const char* name) {
      return std::find(changed.begin(), changed.end(), name) != changed.end();
    };
    CHECK(was_trained("blk.23.attn_q.weight"));
    CHECK(was_trained("blk.23.attn_output.weight"));
    CHECK(was_trained("blk.23.ffn_up.weight"));
    CHECK(was_trained("blk.23.ffn_down.weight"));
    std::printf("MEASURE fine-tune changed %zu of %lld tensors:", changed.size(),
                static_cast<long long>(gguf_get_n_tensors(before.gguf)));
    for (const std::string& name : changed) std::printf(" %s", name.c_str());
    std::printf("\n");
  }

  // The tuned model can be quantised like any other.
  llm::QuantizeOptions quantize;
  quantize.input = options.output;
  quantize.output = work_path("tuned-Q4_K_M.gguf");
  quantize.type = "Q4_K_M";
  quantize.threads = 8;
  std::string error;
  CHECK(llm::quantize(quantize, &error));
  const Probe served = probe(quantize.output);
  CHECK(served.loaded);
  CHECK(!served.answer.empty());
  remove(quantize.output);
  remove(options.output);
}

PRAG_TEST(fine_tuning_can_be_cancelled) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  std::string f32;
  REQUIRE(ensure_f32(&f32));

  llm::FineTuneOptions options;
  options.input = f32;
  options.output = work_path("cancelled-f32.gguf");
  options.window = 64;
  options.epochs = 2;
  options.trainable_blocks = 1;
  options.threads = 8;
  options.parse_special = false;
  remove(options.output);

  // Let three windows pass, then cancel: the rest of the epoch must not be trained.
  int batches = 0;
  const Stopwatch clock;
  const llm::FineTuneReport report =
      llm::fine_tune(options, c_text(9000), [&](double, std::string_view message) {
        if (message.find("trained on window") != std::string_view::npos) ++batches;
        return batches < 3;
      });
  CHECK(!report.ok);
  CHECK(report.cancelled);
  CHECK_EQ(report.error, std::string("cancelled"));
  CHECK_EQ(batches, 3);
  CHECK_EQ(report.epochs_run, 0);
  CHECK(!file_exists(options.output));
  CHECK(!file_exists(options.output + ".raw"));
  std::printf("MEASURE fine-tune cancelled after 3 windows: returned after %.1f s\n", clock.elapsed_s());

  // Too little text is reported before any work is done.
  const llm::FineTuneReport tiny = llm::fine_tune(options, "int x;");
  CHECK(!tiny.ok);
  CHECK(tiny.error.find("tokens") != std::string::npos);

  options.train_embeddings = true;
  const llm::FineTuneReport embeddings = llm::fine_tune(options, c_text(9000));
  CHECK(!embeddings.ok);
  CHECK(embeddings.error.find("train_embeddings") != std::string::npos);
}

PRAG_TEST(too_many_trainable_blocks_are_an_error_not_a_crash) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  if (!file_exists(fp16_path())) return skip("float model missing");
  std::string f32;
  REQUIRE(ensure_f32(&f32));

  // llama.cpp b11257 fails an internal check when the backward pass of 19 or
  // more of the 24 blocks is added to its compute graph. Should a later
  // version accept the graph, the run is cancelled after its first window.
  llm::FineTuneOptions options;
  options.input = f32;
  options.output = work_path("many-blocks-f32.gguf");
  options.window = 64;
  options.epochs = 1;
  options.trainable_blocks = 20;
  options.threads = 8;
  options.parse_special = false;
  remove(options.output);

  const auto until_first_window = [](double, std::string_view message) {
    return message.find("trained on window") == std::string_view::npos;
  };
  const llm::FineTuneReport report = llm::fine_tune(options, c_text(3000), until_first_window);
  CHECK(!report.ok);
  CHECK(report.cancelled || report.error.find("trainable_blocks") != std::string::npos);
  CHECK(!file_exists(options.output));
  std::printf("  message: %s\n", report.error.c_str());
  std::printf("MEASURE refused training graph: %.1f s, peak memory %.0f MiB\n", report.seconds,
              peak_memory_megabytes());

  // The module is intact afterwards: the same call with one block trains.
  options.trainable_blocks = 1;
  const llm::FineTuneReport next = llm::fine_tune(options, c_text(3000), until_first_window);
  CHECK(next.cancelled);
  CHECK(next.trainable_parameters > 0);
  CHECK(next.trainable_parameters < report.trainable_parameters);
  CHECK(!file_exists(options.output));
  std::printf("MEASURE after a second run in the same process: peak memory %.0f MiB\n", peak_memory_megabytes());
}

PRAG_TEST(throughput_of_the_serving_model) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  const std::string path = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
  if (!file_exists(path)) return skip("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf is missing");

  llm::ModelOptions options;
  options.path = path;
  options.context_tokens = 4096;
  options.threads = benchmark_threads();
  std::string error;
  const Stopwatch load_clock;
  const std::unique_ptr<llm::LanguageModel> model = llm::load_model(options, &error);
  const double load_seconds = load_clock.elapsed_s();
  REQUIRE(model != nullptr);

  const std::vector<ChatMessage> messages = {
      {"system", "You answer questions about the C code you are given."},
      {"user", c_text(7000) + "\n\nExplain what the code above does."},
  };
  llm::GenerateOptions generate;
  generate.max_tokens = 128;
  generate.temperature = 0.0f;
  // The first run pays for page faults of the memory-mapped file; report the second.
  model->generate(messages, generate, {});
  const llm::GenerateStats stats = model->generate(messages, generate, {});
  CHECK(stats.error.empty());
  CHECK(stats.prompt_tokens > 1000);
  CHECK(stats.completion_tokens > 0);
  std::printf("MEASURE serving model (Q4_K_M, %d threads): load %.2f s, prompt %d tokens in %.0f ms = %.0f tokens/s, "
              "generation %d tokens in %.0f ms = %.0f tokens/s, peak memory %.0f MiB\n",
              options.threads, load_seconds, stats.prompt_tokens, stats.prompt_ms, 1000.0 * stats.prompt_tokens / stats.prompt_ms,
              stats.completion_tokens, stats.completion_ms, 1000.0 * stats.completion_tokens / stats.completion_ms,
              peak_memory_megabytes());
}

PRAG_TEST(long_texts_are_embedded_in_pieces) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  const std::string path = model_path("Qwen3-Embedding-0.6B-Q8_0.gguf");
  if (!file_exists(path)) return skip("models/Qwen3-Embedding-0.6B-Q8_0.gguf is missing");

  llm::EmbedderOptions options;
  options.path = path;
  options.context_tokens = 2048;  // four times what one call of the model evaluates
  options.threads = 8;
  options.append_eos = true;
  std::string error;
  const std::unique_ptr<prag::embed::Embedder> embedder = llm::load_embedder(options, &error);
  REQUIRE(embedder != nullptr);
  CHECK(embedder->max_tokens() > 2000);

  const auto embed = [&](std::string_view text) { return embedder->embed_one(text, prag::embed::Role::Document); };
  const auto cosine = [](const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return sum;
  };

  // About 1000 tokens: two pieces. The tail is part of the embedding.
  const std::string text = c_text(3600);
  const Stopwatch clock;
  const std::vector<float> whole = embed(text);
  const double seconds = clock.elapsed_s();
  const std::vector<float> again = embed(text);
  const std::vector<float> head = embed(std::string_view(text).substr(0, 1500));
  const std::vector<float> changed = embed(text.substr(0, 3000) + "int unrelated_function(void) { return 42; }\n");
  CHECK_NEAR(cosine(whole, whole), 1.0, 1e-4);
  CHECK(whole == again);
  CHECK(cosine(whole, head) < 0.999);
  CHECK(cosine(whole, head) > 0.5);
  CHECK(cosine(whole, changed) < 0.9999);

  // In a batch a long text is evaluated on its own and the short ones together.
  const std::vector<std::string_view> texts = {"short text", text, "another short text"};
  std::vector<float> rows(texts.size() * embedder->dim());
  embedder->embed(texts, prag::embed::Role::Document, rows.data());
  const std::vector<float> middle(rows.begin() + static_cast<std::ptrdiff_t>(embedder->dim()),
                                  rows.begin() + static_cast<std::ptrdiff_t>(2 * embedder->dim()));
  CHECK(middle == whole);
  const std::vector<float> first(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(embedder->dim()));
  CHECK(first == embed("short text"));
  std::printf("MEASURE long text in pieces: %.2f s, cosine to its first 1500 bytes %.4f, peak memory %.0f MiB\n",
              seconds, cosine(whole, head), peak_memory_megabytes());
}

PRAG_TEST(throughput_of_the_embedder) {
  if (!enabled()) return skip("set PRAG_SLOW_TESTS=1 to run");
  const std::string path = model_path("Qwen3-Embedding-0.6B-Q8_0.gguf");
  if (!file_exists(path)) return skip("models/Qwen3-Embedding-0.6B-Q8_0.gguf is missing");

  llm::EmbedderOptions options;
  options.path = path;
  options.context_tokens = 2048;
  options.threads = benchmark_threads();
  options.append_eos = true;
  std::string error;
  const std::unique_ptr<prag::embed::Embedder> embedder = llm::load_embedder(options, &error);
  REQUIRE(embedder != nullptr);

  // Units of a code base: a few hundred tokens each. The token count comes
  // from the chat model, which shares the vocabulary of the embedding model.
  const std::string chat_path = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
  if (!file_exists(chat_path)) return skip("the chat model is needed to count tokens");
  llm::ModelOptions chat_options;
  chat_options.path = chat_path;
  chat_options.context_tokens = 256;
  const std::unique_ptr<llm::LanguageModel> counter = llm::load_model(chat_options, &error);
  REQUIRE(counter != nullptr);

  const std::string corpus = c_text(48000);
  std::vector<std::string> units;
  for (std::size_t at = 0; at + 1500 <= corpus.size(); at += 1500) units.push_back(corpus.substr(at, 1500));
  std::vector<std::string_view> views(units.begin(), units.end());
  std::size_t tokens = 0;
  for (const std::string& unit : units) tokens += static_cast<std::size_t>(counter->count_tokens(unit)) + 1;

  std::vector<float> out(views.size() * embedder->dim());
  embedder->embed(std::span<const std::string_view>(views).first(2), prag::embed::Role::Document, out.data());  // warm up
  const Stopwatch batch_clock;
  embedder->embed(views, prag::embed::Role::Document, out.data());
  const double batch_seconds = batch_clock.elapsed_s();

  const Stopwatch single_clock;
  std::vector<float> row(embedder->dim());
  for (const std::string_view view : views) {
    const std::string_view one[1] = {view};
    embedder->embed(one, prag::embed::Role::Document, row.data());
  }
  const double single_seconds = single_clock.elapsed_s();

  std::printf("MEASURE embedder (Q8_0, %d threads): %zu texts, %zu tokens; batched %.2f s = %.0f tokens/s, "
              "one text per call %.2f s = %.0f tokens/s, peak memory %.0f MiB\n",
              options.threads, views.size(), tokens, batch_seconds, static_cast<double>(tokens) / batch_seconds, single_seconds,
              static_cast<double>(tokens) / single_seconds, peak_memory_megabytes());
  CHECK(batch_seconds > 0.0);
}

#else  // !PRAG_HAVE_LLAMA

PRAG_TEST(slow_tests_need_llama) { std::printf("  skipped: built without llama.cpp\n"); }

#endif
