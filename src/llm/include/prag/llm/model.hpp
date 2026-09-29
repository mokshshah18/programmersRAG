// Local language model runtime (llama.cpp) behind a small interface.
//
// When the project is built without llama.cpp (PRAG_WITH_LLAMA=OFF) the loaders
// return nullptr with an explanatory error and available() is false; the engine
// then answers extractively and embeds with the hashing embedder.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/embedder.hpp"
#include "prag/core/types.hpp"

namespace prag::llm {

bool available();            // compiled with llama.cpp
std::string backend_info();  // llama.cpp build and the instruction sets in use

struct ModelOptions {
  std::string path;          // GGUF file
  int context_tokens = 4096;
  int threads = 0;           // 0 = number of physical cores, at most 8
  int gpu_layers = 0;
  std::string adapter_path;  // optional LoRA adapter (GGUF)
};

struct GenerateOptions {
  int max_tokens = 512;
  float temperature = 0.2f;
  float top_p = 0.9f;
  int top_k = 40;
  float repeat_penalty = 1.05f;
  std::uint32_t seed = 1;
  std::string grammar;       // GBNF; empty = unconstrained
};

struct GenerateStats {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double prompt_ms = 0.0;      // time to evaluate the prompt
  double completion_ms = 0.0;  // time to generate the completion
  bool truncated = false;      // the prompt was cut to fit the context window
  std::string stop_reason;     // "eos", "length", "cancelled" or "error"
  std::string error;           // set when stop_reason is "error"
};

struct Likelihood {
  double nll = 0.0;   // sum over continuation tokens of -ln p(token | everything before it)
  int tokens = 0;     // continuation tokens scored
  bool truncated = false;  // the prompt was cut to fit the context window
  std::string error;       // set when nothing could be scored; tokens is then 0
  double perplexity() const;  // exp(nll / tokens); NaN when tokens is 0
};

// Called with each decoded piece of text; return false to stop generation.
using PieceCallback = std::function<bool(std::string_view piece)>;

class LanguageModel {
 public:
  virtual ~LanguageModel() = default;

  virtual std::string id() const = 0;   // file name and quantisation type
  virtual int context_tokens() const = 0;
  virtual int count_tokens(std::string_view text) const = 0;

  // Thread-safe: calls are serialised internally.
  virtual GenerateStats generate(std::span<const ChatMessage> messages, const GenerateOptions& options,
                                 const PieceCallback& on_piece) = 0;

  // Negative log-likelihood of `continuation` as the assistant reply to `messages`.
  virtual Likelihood score(std::span<const ChatMessage> messages, std::string_view continuation) = 0;

  // Negative log-likelihood of raw text, in windows of `window` tokens (held-out perplexity).
  // Windows do not overlap and are evaluated independently; every token that has
  // at least one token of context inside its window is scored.
  virtual Likelihood score_text(std::string_view text, int window = 512) = 0;
};

// Returns nullptr and sets *error when the model cannot be loaded.
std::unique_ptr<LanguageModel> load_model(const ModelOptions& options, std::string* error = nullptr);

struct EmbedderOptions {
  std::string path;            // GGUF embedding model
  int context_tokens = 2048;
  int threads = 0;               // 0 = number of physical cores, at most 8
  int gpu_layers = 0;
  std::size_t truncate_dim = 0;  // Matryoshka truncation; 0 = native dimension
  std::string query_prefix;      // prepended to queries, e.g. an instruction
  std::string document_prefix;
  // Some embedding models expect an explicit end-of-text token. A model whose
  // tokeniser adds the token by itself gets it with either value, also when
  // the text had to be truncated.
  bool append_eos = false;
};

// Returns nullptr and sets *error when the model cannot be loaded.
//
// embed() evaluates several texts per call of the model where they fit. The
// vectors of texts embedded together agree with those of single calls up to
// rounding (cosine above 0.999 for a quantised model), not bit for bit;
// embed one text per call where identical bits matter. The same call with the
// same texts always returns the same bits.
std::unique_ptr<embed::Embedder> load_embedder(const EmbedderOptions& options, std::string* error = nullptr);

}  // namespace prag::llm
