// Shared llama.cpp plumbing: backend start-up, log routing, owning handles,
// tokenisation helpers and the abort guard. Private to the llm module.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"

namespace prag::llm::detail {

// Initialises the llama.cpp backend and installs the log router. Idempotent
// and thread-safe; every entry point of the module calls it first.
void ensure_backend();

// ---- owning handles ---------------------------------------------------------

struct ModelDeleter {
  void operator()(llama_model* model) const noexcept { llama_model_free(model); }
};
struct ContextDeleter {
  void operator()(llama_context* context) const noexcept { llama_free(context); }
};
struct SamplerDeleter {
  void operator()(llama_sampler* sampler) const noexcept { llama_sampler_free(sampler); }
};
struct GgufDeleter {
  void operator()(gguf_context* context) const noexcept { gguf_free(context); }
};
struct GgmlDeleter {
  void operator()(ggml_context* context) const noexcept { ggml_free(context); }
};

struct ThreadpoolDeleter {
  void operator()(ggml_threadpool* pool) const noexcept;
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;
using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;
using GgmlPtr = std::unique_ptr<ggml_context, GgmlDeleter>;
using ThreadpoolPtr = std::unique_ptr<ggml_threadpool, ThreadpoolDeleter>;

// Worker threads that live as long as the context they are attached to.
// Without them llama.cpp starts and joins its workers in every evaluation,
// which costs most when one token is evaluated at a time. The pool must
// outlive the context. Returns nullptr when the pool cannot be created or
// `threads` is below 2; the context then works as it does without one.
ThreadpoolPtr attach_threadpool(llama_context* context, int threads);

// A llama_batch with room for `capacity` tokens, each in one sequence.
class Batch {
 public:
  explicit Batch(int capacity);
  ~Batch();
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  void clear() noexcept { batch_.n_tokens = 0; }
  int size() const noexcept { return batch_.n_tokens; }
  int capacity() const noexcept { return capacity_; }
  // `output` requests logits (or the embedding) at this position.
  void add(llama_token token, llama_pos position, llama_seq_id sequence, bool output) noexcept;
  const llama_batch& get() const noexcept { return batch_; }

 private:
  llama_batch batch_;
  int capacity_;
};

// ---- text <-> tokens --------------------------------------------------------

// Never fails for text below 2 GiB; returns an empty vector for larger input.
std::vector<llama_token> tokenize(const llama_vocab* vocab, std::string_view text, bool add_special,
                                  bool parse_special);

// Bytes of one token; may be a fragment of a multi-byte character.
std::string token_to_piece(const llama_vocab* vocab, llama_token token, bool render_special);

std::string detokenize(const llama_vocab* vocab, std::span<const llama_token> tokens, bool render_special);

// ---- model description ------------------------------------------------------

// "Q4_K_M", "F16", "F32", ... ; "unknown" for a type this module has no name for.
const char* ftype_name(llama_ftype ftype);

// Case-insensitive inverse of ftype_name; also accepts the aliases of the
// llama.cpp quantize tool ("Q4_K" = "Q4_K_M", ...).
bool parse_ftype(std::string_view name, llama_ftype* ftype);

// Names accepted by parse_ftype, separated by ", " (for error messages).
std::string known_ftypes();

// ftype_name of the file type stored in (or guessed for) the model.
std::string ftype_label(const llama_model* model);

// Value of a metadata key, or an empty string.
std::string meta_string(const llama_model* model, const char* key);

// File stem, followed by the quantisation type unless the stem already names it.
std::string model_id(std::string_view path, const llama_model* model);

// ---- likelihood -------------------------------------------------------------

// -ln softmax(logits)[target], evaluated in double precision with the usual
// max-shift, so it is exact to rounding for any logit scale.
double negative_log_probability(const float* logits, int n_vocab, llama_token target);

// ---- failure handling -------------------------------------------------------

// Called by the log router for every error-level line.
void capture_error_line(std::string_view line);

// While an ErrorCapture is alive, error-level lines that llama.cpp logs on the
// same thread are collected, so that a failed call can report its cause.
class ErrorCapture {
 public:
  ErrorCapture();
  ~ErrorCapture();
  ErrorCapture(const ErrorCapture&) = delete;
  ErrorCapture& operator=(const ErrorCapture&) = delete;

  // The collected lines joined by "; ", or `fallback` when there are none.
  std::string message(std::string_view fallback) const;

 private:
  std::vector<std::string> lines_;
  ErrorCapture* previous_;
  friend void capture_error_line(std::string_view line);
};

// Runs `body`. llama.cpp reports broken invariants through GGML_ASSERT, which
// ends the process; while `body` runs on this thread such a failure is turned
// into a return value instead: guarded() returns false and *error holds the
// assertion text. C++ exceptions thrown by `body` are reported the same way.
//
// After a false return the llama.cpp objects used by `body` are in an unknown
// state. They must not be used again and must not be freed (release() the
// owning handles): leaking them is the only safe option.
//
// Failures on worker threads of llama.cpp cannot be intercepted and still end
// the process.
bool guarded(const std::function<void()>& body, std::string* error);

// Serialises the operations that load a second copy of a model and use every
// core: quantisation, importance calibration and fine-tuning.
std::mutex& heavy_operation_mutex();

}  // namespace prag::llm::detail
