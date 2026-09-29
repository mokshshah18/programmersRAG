// Backend start-up, log routing and the helpers shared by the llama.cpp
// implementation files.
#include "detail/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

#include "detail/util.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/model.hpp"

namespace prag::llm {

namespace detail {

namespace {

// ---- log routing ------------------------------------------------------------

thread_local std::string t_log_line;
thread_local ggml_log_level t_log_level = GGML_LOG_LEVEL_INFO;
thread_local ErrorCapture* t_capture = nullptr;

// Warnings that llama.cpp prints for every healthy model of the families this
// project uses. They describe the model file, not a problem of this run.
bool is_routine_warning(std::string_view line) {
  static constexpr std::string_view kRoutine[] = {
      "control-looking token",
      "special_eos_id is not in special_eog_ids",
      "setting new yarn_attn_factor",
      "the full capacity of the model will not be utilized",
      "n_ctx is not divisible by n_seq_max",
      "embeddings required but some input tokens were not marked as outputs",
      "possible training context overflow",
      // K-quantisation needs rows of a multiple of 256 values; other tensors get
      // the closest legacy type. One line per tensor, plus a summary.
      "falling back to",
      "required fallback quantization",
  };
  for (const std::string_view pattern : kRoutine) {
    if (line.find(pattern) != std::string_view::npos) return true;
  }
  return false;
}

void emit_log_line(ggml_log_level level, std::string_view line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
  if (line.find_first_not_of(" .\t") == std::string_view::npos) return;  // blank lines and progress dots

  const int length = static_cast<int>(std::min<std::size_t>(line.size(), 1800));
  if (level == GGML_LOG_LEVEL_ERROR) {
    capture_error_line(line);
    PRAG_WARN("llama.cpp: %.*s", length, line.data());
  } else if (level == GGML_LOG_LEVEL_WARN && !is_routine_warning(line)) {
    PRAG_WARN("llama.cpp: %.*s", length, line.data());
  } else if (log::level() <= log::Level::Debug) {
    PRAG_DEBUG("llama.cpp: %.*s", length, line.data());
  }
}

// llama.cpp logs in fragments: a line may arrive in several calls, the later
// ones with the level CONT. Lines are assembled per thread.
void log_callback(ggml_log_level level, const char* text, void* /*user_data*/) {
  if (!text) return;
  if (level != GGML_LOG_LEVEL_CONT) {
    if (!t_log_line.empty() && level != t_log_level) {
      emit_log_line(t_log_level, t_log_line);
      t_log_line.clear();
    }
    t_log_level = level;
  }
  t_log_line.append(text);
  std::size_t start = 0;
  for (std::size_t end = t_log_line.find('\n'); end != std::string::npos; end = t_log_line.find('\n', start)) {
    emit_log_line(t_log_level, std::string_view(t_log_line).substr(start, end - start));
    start = end + 1;
  }
  t_log_line.erase(0, start);
  // A fragment that never gets its newline must not grow without bound.
  if (t_log_line.size() > 4096) {
    emit_log_line(t_log_level, t_log_line);
    t_log_line.clear();
  }
}

// ---- abort guard ------------------------------------------------------------

struct AbortSignal {
  std::string message;
};

thread_local int t_guard_depth = 0;

// Installed with ggml_set_abort_callback. Not declared extern "C" on purpose:
// the function leaves by throwing when a guard is active on this thread.
void abort_callback(const char* message) {
  const char* text = message ? message : "unknown failure";
  if (t_guard_depth > 0) throw AbortSignal{text};
  PRAG_ERROR("llama.cpp failed an internal check and ends the process: %s", text);
}

std::string strip_source_path(std::string message) {
  // "C:/.../ggml/src/ggml.c:1234: GGML_ASSERT(...) failed" -> "ggml.c:1234: ..."
  const std::size_t marker = message.find(": ");
  if (marker == std::string::npos) return message;
  const std::size_t slash = message.find_last_of("/\\", marker);
  if (slash != std::string::npos) message.erase(0, slash + 1);
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();
  return message;
}

struct FtypeEntry {
  const char* name;
  llama_ftype ftype;
  bool alias;
};

// Order matters for ftype_name: the first entry of a type is its name.
constexpr FtypeEntry kFtypes[] = {
    {"F32", LLAMA_FTYPE_ALL_F32, false},
    {"F16", LLAMA_FTYPE_MOSTLY_F16, false},
    {"BF16", LLAMA_FTYPE_MOSTLY_BF16, false},
    {"Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0, false},
    {"Q6_K", LLAMA_FTYPE_MOSTLY_Q6_K, false},
    {"Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M, false},
    {"Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S, false},
    {"Q5_K", LLAMA_FTYPE_MOSTLY_Q5_K_M, true},
    {"Q5_1", LLAMA_FTYPE_MOSTLY_Q5_1, false},
    {"Q5_0", LLAMA_FTYPE_MOSTLY_Q5_0, false},
    {"Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, false},
    {"Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S, false},
    {"Q4_K", LLAMA_FTYPE_MOSTLY_Q4_K_M, true},
    {"Q4_1", LLAMA_FTYPE_MOSTLY_Q4_1, false},
    {"Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0, false},
    {"IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL, false},
    {"IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS, false},
    {"Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L, false},
    {"Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M, false},
    {"Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S, false},
    {"Q3_K", LLAMA_FTYPE_MOSTLY_Q3_K_M, true},
    {"IQ3_M", LLAMA_FTYPE_MOSTLY_IQ3_M, false},
    {"IQ3_S", LLAMA_FTYPE_MOSTLY_IQ3_S, false},
    {"IQ3_XS", LLAMA_FTYPE_MOSTLY_IQ3_XS, false},
    {"IQ3_XXS", LLAMA_FTYPE_MOSTLY_IQ3_XXS, false},
    {"Q2_K", LLAMA_FTYPE_MOSTLY_Q2_K, false},
    {"Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S, false},
    {"IQ2_M", LLAMA_FTYPE_MOSTLY_IQ2_M, false},
    {"IQ2_S", LLAMA_FTYPE_MOSTLY_IQ2_S, false},
    {"IQ2_XS", LLAMA_FTYPE_MOSTLY_IQ2_XS, false},
    {"IQ2_XXS", LLAMA_FTYPE_MOSTLY_IQ2_XXS, false},
    {"IQ1_M", LLAMA_FTYPE_MOSTLY_IQ1_M, false},
    {"IQ1_S", LLAMA_FTYPE_MOSTLY_IQ1_S, false},
    {"TQ2_0", LLAMA_FTYPE_MOSTLY_TQ2_0, false},
    {"TQ1_0", LLAMA_FTYPE_MOSTLY_TQ1_0, false},
};

}  // namespace

void capture_error_line(std::string_view line) {
  if (!t_capture) return;
  // Keep the message, drop the "function_name: " prefix repeated on every line.
  if (t_capture->lines_.size() < 8) t_capture->lines_.emplace_back(line);
}

void ensure_backend() {
  static std::once_flag once;
  std::call_once(once, [] {
    llama_log_set(log_callback, nullptr);
    ggml_set_abort_callback(abort_callback);
    llama_backend_init();
  });
}

// ---- thread pool ------------------------------------------------------------

void ThreadpoolDeleter::operator()(ggml_threadpool* pool) const noexcept { ggml_threadpool_free(pool); }

ThreadpoolPtr attach_threadpool(llama_context* context, int threads) {
  if (!context || threads < 2) return nullptr;
  struct ggml_threadpool_params params = ggml_threadpool_params_default(threads);
  ThreadpoolPtr pool(ggml_threadpool_new(&params));
  if (!pool) {
    PRAG_WARN("llm: could not create a pool of %d threads; evaluation starts its threads per call", threads);
    return nullptr;
  }
  // One pool serves both prompt evaluation and generation: they never overlap.
  llama_attach_threadpool(context, pool.get(), pool.get());
  return pool;
}

// ---- Batch ------------------------------------------------------------------

Batch::Batch(int capacity) : batch_(llama_batch_init(std::max(1, capacity), 0, 1)), capacity_(std::max(1, capacity)) {
  batch_.n_tokens = 0;
}

Batch::~Batch() { llama_batch_free(batch_); }

void Batch::add(llama_token token, llama_pos position, llama_seq_id sequence, bool output) noexcept {
  if (batch_.n_tokens >= capacity_) return;  // callers size the batch; never write past it
  const int i = batch_.n_tokens;
  batch_.token[i] = token;
  batch_.pos[i] = position;
  batch_.n_seq_id[i] = 1;
  batch_.seq_id[i][0] = sequence;
  batch_.logits[i] = output ? 1 : 0;
  ++batch_.n_tokens;
}

// ---- text <-> tokens --------------------------------------------------------

std::vector<llama_token> tokenize(const llama_vocab* vocab, std::string_view text, bool add_special,
                                  bool parse_special) {
  std::vector<llama_token> tokens;
  if (text.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) return tokens;
  const auto length = static_cast<std::int32_t>(text.size());
  // A token covers at least one byte, so this is an upper bound in all but
  // degenerate cases; the second call handles those.
  tokens.resize(text.size() + (add_special ? 4 : 0) + 1);
  std::int32_t n = llama_tokenize(vocab, text.data(), length, tokens.data(), static_cast<std::int32_t>(tokens.size()),
                                  add_special, parse_special);
  if (n == std::numeric_limits<std::int32_t>::min()) return {};
  if (n < 0) {
    tokens.resize(static_cast<std::size_t>(-static_cast<std::int64_t>(n)));
    n = llama_tokenize(vocab, text.data(), length, tokens.data(), static_cast<std::int32_t>(tokens.size()),
                       add_special, parse_special);
    if (n < 0) return {};
  }
  tokens.resize(static_cast<std::size_t>(n));
  return tokens;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token, bool render_special) {
  char stack[64];
  const std::int32_t n = llama_token_to_piece(vocab, token, stack, sizeof(stack), 0, render_special);
  if (n >= 0) return std::string(stack, static_cast<std::size_t>(n));
  std::string piece(static_cast<std::size_t>(-static_cast<std::int64_t>(n)), '\0');
  const std::int32_t m =
      llama_token_to_piece(vocab, token, piece.data(), static_cast<std::int32_t>(piece.size()), 0, render_special);
  piece.resize(m > 0 ? static_cast<std::size_t>(m) : 0);
  return piece;
}

std::string detokenize(const llama_vocab* vocab, std::span<const llama_token> tokens, bool render_special) {
  std::string text;
  if (tokens.empty()) return text;
  text.resize(tokens.size() * 4 + 16);
  std::int32_t n = llama_detokenize(vocab, tokens.data(), static_cast<std::int32_t>(tokens.size()), text.data(),
                                    static_cast<std::int32_t>(text.size()), false, render_special);
  if (n < 0) {
    text.resize(static_cast<std::size_t>(-static_cast<std::int64_t>(n)));
    n = llama_detokenize(vocab, tokens.data(), static_cast<std::int32_t>(tokens.size()), text.data(),
                         static_cast<std::int32_t>(text.size()), false, render_special);
  }
  text.resize(n > 0 ? static_cast<std::size_t>(n) : 0);
  return text;
}

// ---- model description ------------------------------------------------------

const char* ftype_name(llama_ftype ftype) {
  const int plain = static_cast<int>(ftype) & ~static_cast<int>(LLAMA_FTYPE_GUESSED);
  for (const FtypeEntry& entry : kFtypes) {
    if (!entry.alias && static_cast<int>(entry.ftype) == plain) return entry.name;
  }
  return "unknown";
}

bool parse_ftype(std::string_view name, llama_ftype* ftype) {
  const std::string upper = to_upper(name);
  for (const FtypeEntry& entry : kFtypes) {
    if (upper == entry.name) {
      if (ftype) *ftype = entry.ftype;
      return true;
    }
  }
  return false;
}

std::string known_ftypes() {
  std::string out;
  for (const FtypeEntry& entry : kFtypes) {
    if (entry.alias) continue;
    if (!out.empty()) out += ", ";
    out += entry.name;
  }
  return out;
}

std::string ftype_label(const llama_model* model) { return ftype_name(llama_model_ftype(model)); }

std::string meta_string(const llama_model* model, const char* key) {
  char stack[256];
  const std::int32_t n = llama_model_meta_val_str(model, key, stack, sizeof(stack));
  if (n < 0) return {};
  if (static_cast<std::size_t>(n) < sizeof(stack)) return std::string(stack, static_cast<std::size_t>(n));
  std::string value(static_cast<std::size_t>(n) + 1, '\0');
  const std::int32_t m = llama_model_meta_val_str(model, key, value.data(), value.size());
  value.resize(m > 0 ? static_cast<std::size_t>(m) : 0);
  return value;
}

std::string model_id(std::string_view path, const llama_model* model) {
  std::string id = file_stem(path);
  const std::string label = ftype_label(model);
  if (label != "unknown" && to_upper(id).find(label) == std::string::npos) id += " (" + label + ")";
  return id;
}

// ---- likelihood -------------------------------------------------------------

double negative_log_probability(const float* logits, int n_vocab, llama_token target) {
  if (!logits || target < 0 || target >= n_vocab) return std::numeric_limits<double>::quiet_NaN();
  float top = logits[0];
  for (int i = 1; i < n_vocab; ++i) top = std::max(top, logits[i]);
  const double shift = static_cast<double>(top);
  double sum = 0.0;
  for (int i = 0; i < n_vocab; ++i) sum += std::exp(static_cast<double>(logits[i]) - shift);
  return std::log(sum) + shift - static_cast<double>(logits[target]);
}

// ---- failure handling -------------------------------------------------------

ErrorCapture::ErrorCapture() : previous_(t_capture) { t_capture = this; }

ErrorCapture::~ErrorCapture() { t_capture = previous_; }

std::string ErrorCapture::message(std::string_view fallback) const {
  if (lines_.empty()) return std::string(fallback);
  std::string out;
  for (const std::string& line : lines_) {
    if (!out.empty()) out += "; ";
    out += line;
  }
  return out;
}

bool guarded(const std::function<void()>& body, std::string* error) {
  ensure_backend();
  std::string message;
  bool ok = false;
  ++t_guard_depth;
  try {
    body();
    ok = true;
  } catch (const AbortSignal& signal) {
    message = "llama.cpp failed an internal check: " + strip_source_path(signal.message);
  } catch (const std::bad_alloc&) {
    message = "out of memory inside llama.cpp";
  } catch (const std::exception& e) {
    message = std::string("llama.cpp raised an exception: ") + e.what();
  } catch (...) {
    message = "llama.cpp raised an unknown exception";
  }
  --t_guard_depth;
  if (!ok) {
    PRAG_WARN("%s", message.c_str());
    if (error) *error = std::move(message);
  }
  return ok;
}

std::mutex& heavy_operation_mutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace detail

// ---- public: availability ---------------------------------------------------

bool available() { return true; }

std::string backend_info() {
  detail::ensure_backend();
  std::string info = "llama.cpp ";
#ifdef PRAG_LLAMA_TAG
  info += PRAG_LLAMA_TAG;
#else
  info += llama_version();
#endif

  // llama_print_system_info: "CPU : SSE3 = 1 | AVX = 1 | AVX2 = 1 | ... | ".
  // Keep the names of the enabled instruction sets.
  static constexpr std::string_view kInstructionSets[] = {
      "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16", "AVX2", "AVX_VNNI", "AVX",  "FMA",  "F16C",
      "BMI2",   "SSE4_2",      "SSSE3",       "SSE3",        "NEON", "ARM_FMA",  "SVE",  "SVE2", "DOTPROD",
      "MATMUL_INT8", "RISCV_V", "VSX", "WASM_SIMD", "CUDA", "VULKAN", "METAL",
  };
  const std::string_view system = llama_print_system_info();
  std::vector<std::string_view> enabled;
  for (std::size_t at = 0; at < system.size();) {
    std::size_t end = system.find('|', at);
    if (end == std::string_view::npos) end = system.size();
    std::string_view field = system.substr(at, end - at);
    at = end + 1;
    const std::size_t colon = field.find(" : ");
    if (colon != std::string_view::npos) field.remove_prefix(colon + 3);
    const std::size_t equals = field.find('=');
    if (equals == std::string_view::npos) continue;
    std::string_view name = field.substr(0, equals);
    std::string_view value = field.substr(equals + 1);
    while (!name.empty() && name.front() == ' ') name.remove_prefix(1);
    while (!name.empty() && name.back() == ' ') name.remove_suffix(1);
    while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
    while (!value.empty() && value.back() == ' ') value.remove_suffix(1);
    if (value != "1") continue;
    if (std::find(std::begin(kInstructionSets), std::end(kInstructionSets), name) == std::end(kInstructionSets)) continue;
    if (std::find(enabled.begin(), enabled.end(), name) == enabled.end()) enabled.push_back(name);
  }
  // Most capable first, in the order of the table.
  std::string sets;
  for (const std::string_view candidate : kInstructionSets) {
    if (std::find(enabled.begin(), enabled.end(), candidate) == enabled.end()) continue;
    if (!sets.empty()) sets += ' ';
    sets += candidate;
  }
  if (sets.empty()) sets = "no SIMD";
  return info + ", " + sets;
}

}  // namespace prag::llm
