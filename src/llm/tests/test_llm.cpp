// Fast tests of the llm module. Cases that need a model file skip when the
// file is missing; without llama.cpp only the stub behaviour is checked.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../detail/util.hpp"
#include "prag/llm/model.hpp"
#include "prag/llm/tuning.hpp"

#ifdef PRAG_HAVE_LLAMA
#include <stdexcept>

#include "../detail/runtime.hpp"
#endif

namespace llm = prag::llm;
using prag::ChatMessage;
using prag::embed::Role;

namespace {

// ---- real code used as embedding input (mimalloc, MIT licence) --------------

constexpr std::string_view kHeapMalloc = R"(// Allocate `size` bytes in a specific heap.
mi_decl_nodiscard mi_decl_restrict void* mi_heap_malloc(mi_heap_t* heap, size_t size) mi_attr_noexcept {
  mi_theap_t* const theap = _mi_heap_theap_cached(heap);
  if mi_likely(theap!=NULL) { return mi_theap_nonnull_malloc(theap, size); }
                       else { return mi_heap_init_malloc(heap, size); }
}
)";

constexpr std::string_view kStringCopy = R"(bool _mi_strlcpy(char* dest, const char* src, size_t dest_size) {
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
)";

constexpr std::string_view kChachaSplit =
    R"(static void chacha_split(mi_random_ctx_t* ctx, uint64_t nonce, mi_random_ctx_t* ctx_new) {
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

constexpr const char* kQueryPrefix =
    "Instruct: Given a question about a C code base, retrieve the code that answers it\nQuery: ";

// ---- helpers ----------------------------------------------------------------

std::string model_path(const char* file) { return std::string(PRAG_SOURCE_DIR) + "/models/" + file; }

bool file_exists(const std::string& path) {
  std::error_code code;
  return std::filesystem::exists(std::filesystem::path(path), code);
}

void skip(const char* reason) { std::printf("  skipped: %s\n", reason); }

std::unique_ptr<llm::LanguageModel> load_chat_model(int context_tokens) {
  const std::string path = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
  if (!llm::available() || !file_exists(path)) return nullptr;
  llm::ModelOptions options;
  options.path = path;
  options.context_tokens = context_tokens;
  std::string error;
  auto model = llm::load_model(options, &error);
  if (!model) std::fprintf(stderr, "  could not load the chat model: %s\n", error.c_str());
  return model;
}

// Loaded once and shared: loading dominates the run time of these tests.
llm::LanguageModel* chat_model() {
  static std::unique_ptr<llm::LanguageModel> model;
  static bool tried = false;
  if (!tried) {
    tried = true;
    model = load_chat_model(512);
  }
  return model.get();
}

std::unique_ptr<prag::embed::Embedder> load_embedder(std::size_t truncate_dim, const char* query_prefix,
                                                     int context_tokens = 512, bool append_eos = true) {
  const std::string path = model_path("Qwen3-Embedding-0.6B-Q8_0.gguf");
  if (!llm::available() || !file_exists(path)) return nullptr;
  llm::EmbedderOptions options;
  options.path = path;
  options.context_tokens = context_tokens;
  options.truncate_dim = truncate_dim;
  options.query_prefix = query_prefix;
  options.append_eos = append_eos;  // Qwen3-Embedding pools the end-of-text position
  std::string error;
  auto embedder = llm::load_embedder(options, &error);
  if (!embedder) std::fprintf(stderr, "  could not load the embedder: %s\n", error.c_str());
  return embedder;
}

prag::embed::Embedder* embedder() {
  static std::unique_ptr<prag::embed::Embedder> instance;
  static bool tried = false;
  if (!tried) {
    tried = true;
    instance = load_embedder(0, kQueryPrefix);
  }
  return instance.get();
}

double dot(const float* a, const float* b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return sum;
}

double norm(const std::vector<float>& v) { return std::sqrt(dot(v.data(), v.data(), v.size())); }

bool is_valid_utf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    const auto b = static_cast<unsigned char>(s[i]);
    std::size_t length = 0;
    if (b < 0x80) {
      length = 1;
    } else if ((b & 0xE0) == 0xC0) {
      length = 2;
    } else if ((b & 0xF0) == 0xE0) {
      length = 3;
    } else if ((b & 0xF8) == 0xF0) {
      length = 4;
    } else {
      return false;
    }
    if (i + length > s.size()) return false;
    for (std::size_t k = 1; k < length; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    }
    i += length;
  }
  return true;
}

struct Generated {
  std::string text;
  int pieces = 0;
  bool all_pieces_valid = true;
  llm::GenerateStats stats;
};

Generated generate(llm::LanguageModel& model, const std::vector<ChatMessage>& messages,
                   const llm::GenerateOptions& options, int stop_after_pieces = 0) {
  Generated out;
  out.stats = model.generate(messages, options, [&](std::string_view piece) {
    out.text += piece;
    ++out.pieces;
    if (piece.empty() || !is_valid_utf8(piece)) out.all_pieces_valid = false;
    return stop_after_pieces == 0 || out.pieces < stop_after_pieces;
  });
  return out;
}

llm::GenerateOptions greedy(int max_tokens) {
  llm::GenerateOptions options;
  options.max_tokens = max_tokens;
  options.temperature = 0.0f;
  return options;
}

}  // namespace

// ---- independent of llama.cpp ------------------------------------------------

PRAG_TEST(utf8_prefix_holds_back_incomplete_characters) {
  using llm::detail::utf8_complete_prefix;
  CHECK_EQ(utf8_complete_prefix(""), std::size_t{0});
  CHECK_EQ(utf8_complete_prefix("abc"), std::size_t{3});
  // U+00E9 (2 bytes), U+20AC (3 bytes), U+1F600 (4 bytes)
  const std::string e_acute = "\xC3\xA9";
  const std::string euro = "\xE2\x82\xAC";
  const std::string smile = "\xF0\x9F\x98\x80";
  CHECK_EQ(utf8_complete_prefix("a" + e_acute), std::size_t{3});
  CHECK_EQ(utf8_complete_prefix("a" + e_acute.substr(0, 1)), std::size_t{1});
  CHECK_EQ(utf8_complete_prefix(euro), std::size_t{3});
  CHECK_EQ(utf8_complete_prefix(euro.substr(0, 1)), std::size_t{0});
  CHECK_EQ(utf8_complete_prefix(euro.substr(0, 2)), std::size_t{0});
  CHECK_EQ(utf8_complete_prefix("xy" + smile), std::size_t{6});
  for (std::size_t cut = 1; cut < 4; ++cut) CHECK_EQ(utf8_complete_prefix("xy" + smile.substr(0, cut)), std::size_t{2});
  // Bytes that can never be completed are passed through, not held back forever.
  CHECK_EQ(utf8_complete_prefix("a\xFF"), std::size_t{2});
  CHECK_EQ(utf8_complete_prefix("\x80\x80\x80\x80"), std::size_t{4});
}

PRAG_TEST(utf8_trim_removes_partial_characters_at_both_ends) {
  using llm::detail::utf8_trim_partial;
  const std::string euro = "\xE2\x82\xAC";
  const std::string text = euro + "abc" + euro;
  CHECK_EQ(std::string(utf8_trim_partial(text)), text);
  CHECK_EQ(std::string(utf8_trim_partial(std::string_view(text).substr(1))), "abc" + euro);
  CHECK_EQ(std::string(utf8_trim_partial(std::string_view(text).substr(0, text.size() - 1))), euro + "abc");
  CHECK_EQ(std::string(utf8_trim_partial(std::string_view(text).substr(2, text.size() - 4))), std::string("abc"));
}

PRAG_TEST(thread_count_defaults_to_physical_cores_at_most_eight) {
  using llm::detail::physical_cores;
  using llm::detail::resolve_threads;
  CHECK(physical_cores() >= 1);
  CHECK_EQ(resolve_threads(3), 3);
  CHECK_EQ(resolve_threads(12), 12);
  const int automatic = resolve_threads(0);
  CHECK(automatic >= 1);
  CHECK(automatic <= 8);
  CHECK_EQ(automatic, std::min(physical_cores(), 8));
}

PRAG_TEST(file_stem_and_case_helpers) {
  using namespace llm::detail;
  CHECK_EQ(file_stem("C:/models/qwen-q4_k_m.gguf"), std::string("qwen-q4_k_m"));
  CHECK_EQ(file_stem("C:\\models\\a.b.gguf"), std::string("a.b"));
  CHECK_EQ(file_stem("plain"), std::string("plain"));
  CHECK_EQ(to_upper("q4_k_m"), std::string("Q4_K_M"));
  CHECK_EQ(to_lower("AdamW"), std::string("adamw"));
}

PRAG_TEST(perplexity_is_the_exponential_of_the_mean) {
  llm::Likelihood likelihood;
  CHECK(std::isnan(likelihood.perplexity()));
  likelihood.nll = 6.0;
  likelihood.tokens = 3;
  CHECK_NEAR(likelihood.perplexity(), std::exp(2.0), 1e-12);
}

PRAG_TEST(availability_matches_the_build) {
#ifdef PRAG_HAVE_LLAMA
  CHECK(llm::available());
  const std::string info = llm::backend_info();
  CHECK(info.find("llama.cpp") != std::string::npos);
  std::printf("  backend: %s\n", info.c_str());
#else
  CHECK(!llm::available());
  CHECK(!llm::backend_info().empty());
#endif
}

#ifndef PRAG_HAVE_LLAMA

PRAG_TEST(stubs_fail_with_a_clear_message) {
  const auto explains = [](const std::string& error) { return error.find("PRAG_WITH_LLAMA") != std::string::npos; };

  std::string error;
  llm::ModelOptions model;
  model.path = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
  CHECK(llm::load_model(model, &error) == nullptr);
  CHECK(explains(error));
  CHECK(llm::load_model(model) == nullptr);  // the error argument is optional

  error.clear();
  llm::EmbedderOptions embedding;
  embedding.path = model_path("Qwen3-Embedding-0.6B-Q8_0.gguf");
  CHECK(llm::load_embedder(embedding, &error) == nullptr);
  CHECK(explains(error));

  error.clear();
  llm::QuantizeOptions quantize;
  quantize.input = "in.gguf";
  quantize.output = "out.gguf";
  CHECK(!llm::quantize(quantize, &error));
  CHECK(explains(error));

  error.clear();
  llm::ImportanceMatrix matrix;
  matrix.names.push_back("stale");
  matrix.values.push_back({1.0f});
  CHECK(!llm::compute_importance(llm::ImportanceOptions{}, "int main(void) { return 0; }", &matrix, {}, &error));
  CHECK(explains(error));
  CHECK(matrix.empty());

  const llm::FineTuneReport report = llm::fine_tune(llm::FineTuneOptions{}, "text");
  CHECK(!report.ok);
  CHECK(!report.cancelled);
  CHECK(explains(report.error));
}

#else  // PRAG_HAVE_LLAMA

// ---- helpers of the llama.cpp implementation ----------------------------------

PRAG_TEST(internal_failures_of_llama_become_errors) {
  using llm::detail::guarded;
  std::string error = "stale";
  CHECK(guarded([] {}, &error));
  CHECK_EQ(error, std::string("stale"));  // untouched on success

  // What a failed GGML_ASSERT does: without the guard this ends the process.
  CHECK(!guarded([] { GGML_ABORT("deliberate failure %d", 42); }, &error));
  CHECK(error.find("deliberate failure 42") != std::string::npos);
  CHECK(error.find("test_llm.cpp") != std::string::npos);  // file and line, without the directory
  CHECK(error.find(PRAG_SOURCE_DIR) == std::string::npos);

  CHECK(!guarded([] { throw std::runtime_error("boom"); }, &error));
  CHECK(error.find("boom") != std::string::npos);
  CHECK(!guarded([] { throw 7; }, nullptr));

  // Guards nest, and the guard is in force again afterwards.
  bool inner = true;
  CHECK(guarded([&] { inner = guarded([] { GGML_ABORT("inner"); }, nullptr); }, &error));
  CHECK(!inner);
  CHECK(!guarded([] { GGML_ABORT("again"); }, &error));
  CHECK(error.find("again") != std::string::npos);
}

PRAG_TEST(quantisation_types_have_names) {
  using namespace llm::detail;
  llama_ftype type = LLAMA_FTYPE_ALL_F32;
  CHECK(parse_ftype("q4_k_m", &type));
  CHECK_EQ(static_cast<int>(type), static_cast<int>(LLAMA_FTYPE_MOSTLY_Q4_K_M));
  CHECK(parse_ftype("Q4_K", &type));  // alias of the quantize tool
  CHECK_EQ(static_cast<int>(type), static_cast<int>(LLAMA_FTYPE_MOSTLY_Q4_K_M));
  CHECK(parse_ftype("F32", &type));
  CHECK_EQ(static_cast<int>(type), static_cast<int>(LLAMA_FTYPE_ALL_F32));
  CHECK(parse_ftype("f16", &type));
  CHECK_EQ(static_cast<int>(type), static_cast<int>(LLAMA_FTYPE_MOSTLY_F16));
  CHECK(!parse_ftype("", &type));
  CHECK(!parse_ftype("Q4", &type));
  CHECK_EQ(static_cast<int>(type), static_cast<int>(LLAMA_FTYPE_MOSTLY_F16));  // untouched on failure

  CHECK_EQ(std::string(ftype_name(LLAMA_FTYPE_MOSTLY_Q4_K_M)), std::string("Q4_K_M"));
  CHECK_EQ(std::string(ftype_name(LLAMA_FTYPE_MOSTLY_Q8_0)), std::string("Q8_0"));
  CHECK_EQ(std::string(ftype_name(LLAMA_FTYPE_ALL_F32)), std::string("F32"));
  // A type guessed from the tensors carries a flag bit.
  const auto guessed = static_cast<llama_ftype>(LLAMA_FTYPE_MOSTLY_Q6_K | LLAMA_FTYPE_GUESSED);
  CHECK_EQ(std::string(ftype_name(guessed)), std::string("Q6_K"));
  CHECK_EQ(std::string(ftype_name(static_cast<llama_ftype>(999))), std::string("unknown"));
  CHECK(known_ftypes().find("Q5_K_M") != std::string::npos);
}

PRAG_TEST(log_probabilities_are_exact_for_any_scale) {
  using llm::detail::negative_log_probability;
  const float uniform[4] = {3.0f, 3.0f, 3.0f, 3.0f};
  CHECK_NEAR(negative_log_probability(uniform, 4, 2), std::log(4.0), 1e-12);

  const float mixed[3] = {1.0f, 2.0f, 3.0f};
  const double z = std::exp(1.0) + std::exp(2.0) + std::exp(3.0);
  double total = 0.0;
  for (int i = 0; i < 3; ++i) {
    CHECK_NEAR(negative_log_probability(mixed, 3, i), std::log(z) - (i + 1), 1e-12);
    total += std::exp(-negative_log_probability(mixed, 3, i));
  }
  CHECK_NEAR(total, 1.0, 1e-12);

  // exp(1000) overflows; the shifted form does not.
  const float large[2] = {1000.0f, 0.0f};
  CHECK_NEAR(negative_log_probability(large, 2, 0), 0.0, 1e-12);
  CHECK_NEAR(negative_log_probability(large, 2, 1), 1000.0, 1e-9);
  const float negative[2] = {-1000.0f, -1001.0f};
  CHECK_NEAR(negative_log_probability(negative, 2, 0), std::log1p(std::exp(-1.0)), 1e-12);

  CHECK(std::isnan(negative_log_probability(mixed, 3, 3)));
  CHECK(std::isnan(negative_log_probability(mixed, 3, -1)));
  CHECK(std::isnan(negative_log_probability(nullptr, 3, 0)));
}

// ---- errors that need no model -----------------------------------------------

PRAG_TEST(loaders_report_missing_files) {
  std::string error;
  llm::ModelOptions model;
  model.path = model_path("no-such-model.gguf");
  CHECK(llm::load_model(model, &error) == nullptr);
  CHECK(error.find("not found") != std::string::npos);
  CHECK(error.find("no-such-model.gguf") != std::string::npos);

  error.clear();
  llm::EmbedderOptions embedding;
  embedding.path = model_path("no-such-model.gguf");
  CHECK(llm::load_embedder(embedding, &error) == nullptr);
  CHECK(error.find("not found") != std::string::npos);

  error.clear();
  CHECK(llm::load_model(llm::ModelOptions{}, &error) == nullptr);
  CHECK(!error.empty());
}

PRAG_TEST(a_file_that_is_no_model_is_rejected) {
  // The test harness header is certainly not a GGUF file.
  std::string error;
  llm::ModelOptions model;
  model.path = std::string(PRAG_SOURCE_DIR) + "/tests/prag_test.hpp";
  CHECK(llm::load_model(model, &error) == nullptr);
  CHECK(!error.empty());
  std::printf("  message: %s\n", error.c_str());
}

PRAG_TEST(tuning_primitives_validate_their_arguments) {
  std::string error;
  llm::QuantizeOptions quantize;
  quantize.input = model_path("no-such-model.gguf");
  quantize.output = model_path("out.gguf");
  CHECK(!llm::quantize(quantize, &error));
  CHECK(error.find("not found") != std::string::npos);

  const std::string chat = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
  if (file_exists(chat)) {
    error.clear();
    quantize.input = chat;
    quantize.output = chat;
    CHECK(!llm::quantize(quantize, &error));
    CHECK(error.find("same file") != std::string::npos);

    error.clear();
    quantize.output = model_path("out.gguf");
    quantize.type = "Q9_X";
    CHECK(!llm::quantize(quantize, &error));
    CHECK(error.find("Q9_X") != std::string::npos);
    CHECK(error.find("Q4_K_M") != std::string::npos);  // the message lists what is known
    CHECK(!file_exists(quantize.output));

    // Quantised weights cannot be trained; the message says what to do.
    llm::FineTuneOptions tune;
    tune.input = chat;
    tune.output = model_path("tuned.gguf");
    const llm::FineTuneReport report = llm::fine_tune(tune, "some text");
    CHECK(!report.ok);
    CHECK(report.error.find("float32") != std::string::npos);
    CHECK(report.error.find("F32") != std::string::npos);
    CHECK(!file_exists(tune.output));
    std::printf("  message: %s\n", report.error.c_str());
  }

  llm::FineTuneOptions tune;
  tune.input = model_path("no-such-model.gguf");
  tune.output = model_path("tuned.gguf");
  CHECK(llm::fine_tune(tune, "text").error.find("not found") != std::string::npos);

  error.clear();
  llm::ImportanceMatrix matrix;
  llm::ImportanceOptions importance;
  importance.model = model_path("no-such-model.gguf");
  CHECK(!llm::compute_importance(importance, "text", &matrix, {}, &error));
  CHECK(error.find("not found") != std::string::npos);
  CHECK(matrix.empty());
}

// ---- chat model --------------------------------------------------------------

PRAG_TEST(chat_model_loads_and_describes_itself) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf is missing");
  CHECK_EQ(model->context_tokens(), 512);
  CHECK(model->id().find("qwen2.5-coder-0.5b-instruct-q4_k_m") != std::string::npos);
  std::printf("  id: %s\n", model->id().c_str());
}

PRAG_TEST(tokens_are_counted) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  CHECK_EQ(model->count_tokens(""), 0);
  const int one = model->count_tokens("hello");
  CHECK(one >= 1);
  CHECK(one <= 2);
  const int code = model->count_tokens(kHeapMalloc);
  CHECK(code > 40);
  CHECK(code < static_cast<int>(kHeapMalloc.size()));
  // Counting is additive up to the merge at the seam.
  const std::string twice = std::string(kHeapMalloc) + std::string(kHeapMalloc);
  CHECK(std::abs(model->count_tokens(twice) - 2 * code) <= 2);
}

PRAG_TEST(greedy_generation_is_deterministic) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::vector<ChatMessage> messages = {
      {"system", "You are a concise assistant for C programmers."},
      {"user", "In one sentence, what does the C function malloc do?"},
  };
  const Generated first = generate(*model, messages, greedy(16));
  const Generated second = generate(*model, messages, greedy(16));
  CHECK(first.stats.error.empty());
  CHECK(!first.text.empty());
  CHECK_EQ(first.text, second.text);
  CHECK(first.all_pieces_valid);
  CHECK(first.stats.prompt_tokens > 10);
  CHECK(first.stats.completion_tokens > 0);
  CHECK(first.stats.completion_tokens <= 16);
  CHECK_EQ(first.stats.completion_tokens, second.stats.completion_tokens);
  CHECK(first.stats.prompt_ms > 0.0);
  CHECK(first.stats.completion_ms > 0.0);
  CHECK(!first.stats.truncated);
  CHECK(first.stats.stop_reason == "eos" || first.stats.stop_reason == "length");
  std::printf("  answer: %s\n", first.text.c_str());
}

PRAG_TEST(seeded_sampling_is_reproducible) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::vector<ChatMessage> messages = {{"user", "Name three sorting algorithms."}};
  llm::GenerateOptions options;
  options.max_tokens = 8;
  options.temperature = 0.9f;
  options.seed = 7;
  const Generated first = generate(*model, messages, options);
  const Generated second = generate(*model, messages, options);
  CHECK(!first.text.empty());
  CHECK_EQ(first.text, second.text);
}

PRAG_TEST(max_tokens_is_honoured) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::vector<ChatMessage> messages = {{"user", "Explain in detail how a hash table works."}};
  const Generated out = generate(*model, messages, greedy(5));
  CHECK_EQ(out.stats.completion_tokens, 5);
  CHECK_EQ(out.stats.stop_reason, std::string("length"));

  const Generated none = generate(*model, messages, greedy(0));
  CHECK_EQ(none.stats.completion_tokens, 0);
  CHECK(none.text.empty());
  CHECK_EQ(none.stats.stop_reason, std::string("length"));
  CHECK(none.stats.prompt_tokens > 0);
}

PRAG_TEST(the_callback_can_stop_generation) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::vector<ChatMessage> messages = {{"user", "Explain in detail how a hash table works."}};
  const Generated out = generate(*model, messages, greedy(32), 3);
  CHECK_EQ(out.pieces, 3);
  CHECK_EQ(out.stats.stop_reason, std::string("cancelled"));
  CHECK(out.stats.completion_tokens >= 3);
  CHECK(out.stats.completion_tokens < 32);
  CHECK(out.stats.error.empty());

  // A missing callback is allowed: the statistics are still produced.
  const llm::GenerateStats silent = model->generate(messages, greedy(4), {});
  CHECK_EQ(silent.completion_tokens, 4);
}

PRAG_TEST(streamed_pieces_are_whole_characters) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  // Chinese text takes three bytes per character and several characters share
  // no token boundary with their bytes.
  const std::vector<ChatMessage> messages = {
      {"user", "\xE8\xAF\xB7\xE7\x94\xA8\xE4\xB8\xAD\xE6\x96\x87\xE5\x9B\x9E\xE7\xAD\x94\xEF\xBC\x9A"
               "\xE4\xBB\x80\xE4\xB9\x88\xE6\x98\xAF\xE6\x8C\x87\xE9\x92\x88\xEF\xBC\x9F"}};
  const Generated out = generate(*model, messages, greedy(12));
  CHECK(!out.text.empty());
  CHECK(out.all_pieces_valid);
  CHECK(is_valid_utf8(out.text));
}

PRAG_TEST(an_over_long_prompt_loses_its_middle) {
  // The smallest window llama.cpp offers, to keep the evaluation short.
  const std::unique_ptr<llm::LanguageModel> model = load_chat_model(256);
  if (!model) return skip("chat model missing");
  CHECK_EQ(model->context_tokens(), 256);
  std::string filler;
  for (int i = 0; i < 60; ++i) filler += std::string(kStringCopy);
  const std::string content =
      "Remember the code word at the start: ALPHA.\n" + filler + "\nWhat is 2 + 2? Answer with the number only.";
  REQUIRE(model->count_tokens(content) > 4 * model->context_tokens());

  const std::vector<ChatMessage> messages = {{"system", "You are a calculator."}, {"user", content}};
  const Generated out = generate(*model, messages, greedy(8));
  CHECK(out.stats.error.empty());
  CHECK(out.stats.truncated);
  CHECK(out.stats.prompt_tokens <= model->context_tokens() - 8);
  CHECK(out.stats.prompt_tokens > model->context_tokens() / 2);
  CHECK(!out.text.empty());
  // The end of the user content survived the cut: the model answers the question.
  CHECK(out.text.find('4') != std::string::npos);
  std::printf("  prompt tokens: %d, answer: %s\n", out.stats.prompt_tokens, out.text.c_str());

  const llm::Likelihood likelihood = model->score(messages, "4");
  CHECK(likelihood.error.empty());
  CHECK(likelihood.truncated);
  CHECK(likelihood.tokens >= 1);

  // When the question is short and the system prompt is what does not fit,
  // the system prompt is cut and the question survives.
  const std::vector<ChatMessage> long_system = {
      {"system", "You are a calculator. Reference material follows.\n" + filler},
      {"user", "What is 3 + 4? Answer with the number only."},
  };
  const Generated other = generate(*model, long_system, greedy(8));
  CHECK(other.stats.error.empty());
  CHECK(other.stats.truncated);
  CHECK(other.stats.prompt_tokens <= model->context_tokens() - 8);
  CHECK(other.text.find('7') != std::string::npos);
  std::printf("  prompt tokens: %d, answer: %s\n", other.stats.prompt_tokens, other.text.c_str());
}

PRAG_TEST(score_prefers_a_sensible_continuation) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::vector<ChatMessage> messages = {{"user", "In one sentence, what does the C function malloc do?"}};
  const std::string sensible = "It allocates a block of memory on the heap and returns a pointer to it.";
  const std::string scrambled = "pointer heap it a of to memory returns block and allocates the on It a.";

  const llm::Likelihood good = model->score(messages, sensible);
  const llm::Likelihood bad = model->score(messages, scrambled);
  CHECK(good.error.empty());
  CHECK(bad.error.empty());
  CHECK_EQ(good.tokens, model->count_tokens(sensible));
  CHECK_EQ(bad.tokens, model->count_tokens(scrambled));
  CHECK(std::isfinite(good.nll));
  CHECK(std::isfinite(bad.nll));
  CHECK(good.nll > 0.0);
  CHECK(!good.truncated);
  CHECK(good.nll / good.tokens < bad.nll / bad.tokens);
  CHECK(good.nll < bad.nll);
  std::printf("  nats per token: sensible %.3f, scrambled %.3f\n", good.nll / good.tokens, bad.nll / bad.tokens);

  // The same call gives the same number, and the conditioning matters.
  const llm::Likelihood again = model->score(messages, sensible);
  CHECK_NEAR(again.nll, good.nll, 1e-9);
  const std::vector<ChatMessage> unrelated = {{"user", "What is the capital of France?"}};
  CHECK(model->score(unrelated, sensible).nll > good.nll);

  const llm::Likelihood empty = model->score(messages, "");
  CHECK_EQ(empty.tokens, 0);
  CHECK(!empty.error.empty());
  CHECK(std::isnan(empty.perplexity()));
}

PRAG_TEST(score_agrees_with_greedy_generation) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  // What greedy decoding produced is, token for token, the most probable
  // continuation, so every token has probability above 1 / vocabulary size and
  // the mean is far below the uniform value ln(151936) = 11.9.
  const std::vector<ChatMessage> messages = {{"user", "In one sentence, what does the C function malloc do?"}};
  llm::GenerateOptions options = greedy(10);
  options.repeat_penalty = 1.0f;  // pure argmax
  const Generated out = generate(*model, messages, options);
  REQUIRE(!out.text.empty());
  const llm::Likelihood likelihood = model->score(messages, out.text);
  CHECK(likelihood.error.empty());
  CHECK(likelihood.tokens > 0);
  CHECK(likelihood.nll / likelihood.tokens < 2.0);
  std::printf("  nats per token of the greedy answer: %.3f\n", likelihood.nll / likelihood.tokens);
}

PRAG_TEST(raw_text_is_scored_in_windows) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  const std::string text = std::string(kStringCopy) + std::string(kHeapMalloc);
  const int tokens = model->count_tokens(text);
  REQUIRE(tokens > 128);
  REQUIRE(tokens < 512);

  const llm::Likelihood whole = model->score_text(text, 512);
  CHECK(whole.error.empty());
  CHECK_EQ(whole.tokens, tokens - 1);  // one window: every token but the first
  CHECK(std::isfinite(whole.nll));
  CHECK(whole.perplexity() > 1.0);
  CHECK(whole.perplexity() < 100.0);  // code is predictable for a code model

  const int window = 64;
  const llm::Likelihood split = model->score_text(text, window);
  const int full = tokens / window;
  const int rest = tokens % window;
  CHECK_EQ(split.tokens, full * (window - 1) + (rest >= 2 ? rest - 1 : 0));
  // Less context per token can only make the text harder to predict.
  CHECK(split.nll / split.tokens > whole.nll / whole.tokens);
  std::printf("  perplexity: %.2f in one window, %.2f in windows of %d tokens\n", whole.perplexity(),
              split.perplexity(), window);

  CHECK(!model->score_text("", 64).error.empty());
}

PRAG_TEST(a_grammar_restricts_the_output) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  llm::GenerateOptions options = greedy(8);
  options.grammar = "root ::= \"yes\" | \"no\"";

  const std::vector<ChatMessage> yes = {{"user", "Is C a programming language? Answer yes or no."}};
  const Generated first = generate(*model, yes, options);
  CHECK(first.stats.error.empty());
  CHECK(first.text == "yes" || first.text == "no");
  CHECK_EQ(first.stats.stop_reason, std::string("eos"));
  CHECK_EQ(first.text, std::string("yes"));

  // Even a question that invites prose gets one of the two words.
  const std::vector<ChatMessage> prose = {{"user", "Write a poem about the sea."}};
  const Generated second = generate(*model, prose, options);
  CHECK(second.text == "yes" || second.text == "no");

  // The same with sampling instead of greedy decoding.
  options.temperature = 1.0f;
  options.seed = 3;
  const Generated third = generate(*model, prose, options);
  CHECK(third.text == "yes" || third.text == "no");
}

PRAG_TEST(an_invalid_grammar_is_an_error) {
  llm::LanguageModel* model = chat_model();
  if (!model) return skip("chat model missing");
  llm::GenerateOptions options = greedy(8);
  options.grammar = "root ::= (";
  const std::vector<ChatMessage> messages = {{"user", "Say hello."}};
  const Generated out = generate(*model, messages, options);
  CHECK_EQ(out.stats.stop_reason, std::string("error"));
  CHECK(out.stats.error.find("grammar") != std::string::npos);
  CHECK(out.text.empty());

  // The model is still usable afterwards.
  const Generated next = generate(*model, messages, greedy(4));
  CHECK(next.stats.error.empty());
  CHECK(!next.text.empty());

  CHECK_EQ(model->generate({}, greedy(4), {}).stop_reason, std::string("error"));
}

// ---- embedder ----------------------------------------------------------------

PRAG_TEST(embedder_loads_with_the_native_dimension) {
  prag::embed::Embedder* model = embedder();
  if (!model) return skip("models/Qwen3-Embedding-0.6B-Q8_0.gguf is missing");
  CHECK_EQ(model->dim(), std::size_t{1024});
  CHECK(model->max_tokens() > 400);
  CHECK(model->max_tokens() < 512);
  CHECK(model->id().find("Qwen3-Embedding-0.6B-Q8_0") != std::string::npos);
  CHECK(model->id().find("pool=last") != std::string::npos);
  std::printf("  id: %s\n", model->id().c_str());
}

PRAG_TEST(embeddings_have_unit_norm) {
  prag::embed::Embedder* model = embedder();
  if (!model) return skip("embedding model missing");
  for (const std::string_view text : {kHeapMalloc, std::string_view("x"), std::string_view("")}) {
    const std::vector<float> vector = model->embed_one(text, Role::Document);
    CHECK_EQ(vector.size(), model->dim());
    CHECK_NEAR(norm(vector), 1.0, 1e-4);
    for (const float value : vector) CHECK(std::isfinite(value));
  }
}

PRAG_TEST(a_question_about_allocation_finds_the_allocator) {
  prag::embed::Embedder* model = embedder();
  if (!model) return skip("embedding model missing");
  const std::vector<float> query =
      model->embed_one("How is memory allocated from a specific heap?", Role::Query);
  const std::vector<float> allocator = model->embed_one(kHeapMalloc, Role::Document);
  const std::vector<float> copy = model->embed_one(kStringCopy, Role::Document);
  const std::vector<float> random = model->embed_one(kChachaSplit, Role::Document);

  const double to_allocator = dot(query.data(), allocator.data(), query.size());
  const double to_copy = dot(query.data(), copy.data(), query.size());
  const double to_random = dot(query.data(), random.data(), query.size());
  std::printf("  cosine: mi_heap_malloc %.3f, _mi_strlcpy %.3f, chacha_split %.3f\n", to_allocator, to_copy,
              to_random);
  CHECK(to_allocator > to_copy + 0.05);
  CHECK(to_allocator > to_random + 0.05);

  // And the other way round.
  const std::vector<float> other = model->embed_one("How is a string copied into a bounded buffer?", Role::Query);
  CHECK(dot(other.data(), copy.data(), other.size()) > dot(other.data(), allocator.data(), other.size()) + 0.05);
}

PRAG_TEST(a_batch_gives_the_vectors_of_single_calls) {
  prag::embed::Embedder* model = embedder();
  if (!model) return skip("embedding model missing");
  const std::vector<std::string_view> texts = {kHeapMalloc, "short", kStringCopy};
  const std::size_t dim = model->dim();
  std::vector<float> batch(texts.size() * dim);
  model->embed(texts, Role::Document, batch.data());

  // Equal up to rounding. The first text of a batch is reproduced bit for bit;
  // the others sit at other offsets of the key-value buffer, where the
  // vectorised attention kernels of llama.cpp add up in another order, and the
  // 8-bit activations of a quantised model amplify that to the third decimal.
  for (std::size_t i = 0; i < texts.size(); ++i) {
    const std::vector<float> single = model->embed_one(texts[i], Role::Document);
    const double cosine = dot(single.data(), batch.data() + i * dim, dim);
    double largest = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
      largest = std::max(largest, std::fabs(static_cast<double>(single[k]) - static_cast<double>(batch[i * dim + k])));
    }
    std::printf("  text %zu: cosine %.6f, largest difference %.2e\n", i, cosine, largest);
    CHECK(cosine > 0.999);
    CHECK(largest < 1e-2);
    if (i == 0) CHECK_EQ(largest, 0.0);
  }
  // The rows are in the order of the texts, not mixed up.
  CHECK(dot(batch.data(), batch.data() + dim, dim) < 0.9);
  CHECK(dot(batch.data(), batch.data() + 2 * dim, dim) < 0.9);

  // The same call twice gives the same bits.
  std::vector<float> again(texts.size() * dim);
  model->embed(texts, Role::Document, again.data());
  CHECK(batch == again);
}

PRAG_TEST(the_query_prefix_applies_to_queries_only) {
  prag::embed::Embedder* model = embedder();
  if (!model) return skip("embedding model missing");
  const std::string_view text = "free a block of memory";
  const std::vector<float> as_query = model->embed_one(text, Role::Query);
  const std::vector<float> as_document = model->embed_one(text, Role::Document);
  const std::string prefixed = std::string(kQueryPrefix) + std::string(text);
  const std::vector<float> manual = model->embed_one(prefixed, Role::Document);

  CHECK(dot(as_query.data(), as_document.data(), as_query.size()) < 0.999);
  CHECK(as_query == manual);  // a query is the prefix followed by the text, nothing else
}

PRAG_TEST(matryoshka_truncation_and_the_context_limit) {
  prag::embed::Embedder* full = embedder();
  if (!full) return skip("embedding model missing");
  // A second instance: 256 leading dimensions, and the smallest window
  // llama.cpp offers so that the over-long text below is cheap.
  const std::unique_ptr<prag::embed::Embedder> small = load_embedder(256, kQueryPrefix, 256);
  REQUIRE(small != nullptr);
  CHECK_EQ(small->dim(), std::size_t{256});
  CHECK(small->id() != full->id());
  CHECK(small->max_tokens() < 256);
  CHECK(small->max_tokens() > 200);

  const std::vector<float> reference = full->embed_one(kHeapMalloc, Role::Document);
  const std::vector<float> truncated = small->embed_one(kHeapMalloc, Role::Document);
  CHECK_EQ(truncated.size(), std::size_t{256});
  CHECK_NEAR(norm(truncated), 1.0, 1e-4);
  std::vector<float> expected(reference.begin(), reference.begin() + 256);
  prag::embed::normalize(expected.data(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_NEAR(truncated[i], expected[i], 1e-5);

  // A text beyond the window is cut: what lies beyond does not matter.
  std::string text;
  for (int i = 0; i < 12; ++i) text += std::string(kStringCopy);
  const std::vector<float> cut = small->embed_one(text, Role::Document);
  const std::vector<float> longer = small->embed_one(text + std::string(kChachaSplit), Role::Document);
  CHECK_NEAR(norm(cut), 1.0, 1e-4);
  CHECK(cut == longer);

  // The tokeniser of this model ends every text with the end-of-text token,
  // which is the position that gets pooled. Without the option the token is
  // still there, and it is still there after a text has been cut.
  const std::unique_ptr<prag::embed::Embedder> implicit = load_embedder(256, kQueryPrefix, 256, false);
  REQUIRE(implicit != nullptr);
  CHECK_EQ(implicit->max_tokens(), small->max_tokens());
  CHECK(implicit->embed_one(kHeapMalloc, Role::Document) == truncated);
  CHECK(implicit->embed_one("free a block of memory", Role::Query) ==
        small->embed_one("free a block of memory", Role::Query));
  CHECK(implicit->embed_one(text, Role::Document) == cut);
}

#endif  // PRAG_HAVE_LLAMA
