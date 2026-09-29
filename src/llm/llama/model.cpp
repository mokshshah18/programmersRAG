// LanguageModel on top of llama.cpp: chat formatting, prompt truncation,
// sampling, UTF-8 safe streaming and exact likelihoods.
#include "prag/llm/model.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <utility>

#include "detail/runtime.hpp"
#include "detail/util.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"

namespace prag::llm {

namespace {

using detail::Batch;
using detail::ContextPtr;
using detail::ModelPtr;
using detail::SamplerPtr;
using detail::ThreadpoolPtr;

// Logit rows requested per decode call while scoring. One row is n_vocab
// floats (0.6 MiB for a 150k vocabulary), so this bounds the output buffer.
constexpr int kScoreOutputsPerBatch = 64;
constexpr int kPenaltyLastN = 64;
constexpr std::string_view kCutMarker = "\n[...]\n";

struct LoraDeleter {
  void operator()(llama_adapter_lora* adapter) const noexcept { llama_adapter_lora_free(adapter); }
};
using LoraPtr = std::unique_ptr<llama_adapter_lora, LoraDeleter>;

class LlamaModel final : public LanguageModel {
 public:
  LlamaModel(ModelPtr model, LoraPtr adapter, ThreadpoolPtr pool, ContextPtr context, std::string id)
      : model_(std::move(model)),
        adapter_(std::move(adapter)),
        pool_(std::move(pool)),
        context_(std::move(context)),
        vocab_(llama_model_get_vocab(model_.get())),
        id_(std::move(id)),
        n_ctx_(static_cast<int>(llama_n_ctx(context_.get()))),
        n_batch_(static_cast<int>(llama_n_batch(context_.get()))),
        n_vocab_(llama_vocab_n_tokens(vocab_)) {
    if (const char* stored = llama_model_chat_template(model_.get(), nullptr)) template_ = stored;
  }

  ~LlamaModel() override {
    if (poisoned_) {
      // See detail::guarded: the objects are in an unknown state, freeing them could crash.
      (void)context_.release();
      (void)pool_.release();
      (void)adapter_.release();
      (void)model_.release();
    } else {
      context_.reset();  // before the thread pool, the adapter and the model it refers to
    }
  }

  std::string id() const override { return id_; }
  int context_tokens() const override { return n_ctx_; }

  int count_tokens(std::string_view text) const override {
    // llama_tokenize is thread-safe and does not touch the context.
    return static_cast<int>(detail::tokenize(vocab_, text, false, false).size());
  }

  GenerateStats generate(std::span<const ChatMessage> messages, const GenerateOptions& options,
                         const PieceCallback& on_piece) override {
    std::lock_guard<std::mutex> lock(mutex_);
    GenerateStats stats;
    if (poisoned_) {
      stats.stop_reason = "error";
      stats.error = kPoisoned;
      return stats;
    }
    std::string failure;
    if (!detail::guarded([&] { generate_locked(messages, options, on_piece, &stats); }, &failure)) {
      poisoned_ = true;
      stats.stop_reason = "error";
      stats.error = failure;
    }
    return stats;
  }

  Likelihood score(std::span<const ChatMessage> messages, std::string_view continuation) override {
    std::lock_guard<std::mutex> lock(mutex_);
    Likelihood result;
    if (poisoned_) {
      result.error = kPoisoned;
      return result;
    }
    std::string failure;
    if (!detail::guarded([&] { score_locked(messages, continuation, &result); }, &failure)) {
      poisoned_ = true;
      result = Likelihood{};
      result.error = failure;
    }
    return result;
  }

  Likelihood score_text(std::string_view text, int window) override {
    std::lock_guard<std::mutex> lock(mutex_);
    Likelihood result;
    if (poisoned_) {
      result.error = kPoisoned;
      return result;
    }
    std::string failure;
    if (!detail::guarded([&] { score_text_locked(text, window, &result); }, &failure)) {
      poisoned_ = true;
      result = Likelihood{};
      result.error = failure;
    }
    return result;
  }

 private:
  static constexpr const char* kPoisoned =
      "the model is unusable after an internal failure of llama.cpp; load it again";

  // ---- chat formatting ------------------------------------------------------

  bool format(std::span<const ChatMessage> messages, std::string* out, std::string* error) {
    if (messages.empty()) {
      *error = "no messages to format";
      return false;
    }
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    std::size_t bytes = 0;
    for (const ChatMessage& message : messages) {
      chat.push_back({message.role.c_str(), message.content.c_str()});
      bytes += message.role.size() + message.content.size() + 64;
    }

    const auto apply = [&](const char* tmpl) -> std::int32_t {
      out->resize(2 * bytes + 256);
      std::int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, out->data(),
                                                 static_cast<std::int32_t>(out->size()));
      if (n > static_cast<std::int32_t>(out->size())) {
        out->resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, out->data(),
                                      static_cast<std::int32_t>(out->size()));
      }
      return n;
    };

    // nullptr selects ChatML inside llama.cpp.
    std::int32_t n = apply(template_.empty() ? nullptr : template_.c_str());
    if (n < 0 && !template_.empty()) {
      if (!warned_template_) {
        PRAG_WARN("llm: the chat template of %s is not supported by llama.cpp; using ChatML", id_.c_str());
        warned_template_ = true;
      }
      n = apply(nullptr);
    }
    if (n < 0) {
      out->clear();
      *error = "the chat template could not be applied";
      return false;
    }
    out->resize(static_cast<std::size_t>(n));
    return true;
  }

  bool format_and_tokenize(std::span<const ChatMessage> messages, std::vector<llama_token>* tokens,
                           std::string* error) {
    std::string text;
    if (!format(messages, &text, error)) return false;
    *tokens = detail::tokenize(vocab_, text, true, true);
    if (tokens->empty()) {
      *error = "the prompt could not be tokenised";
      return false;
    }
    return true;
  }

  // Formats and tokenises `messages` so that the result has at most `budget`
  // tokens. An over-long prompt loses the middle of its user content: the
  // beginning (instructions) and the end (the question) survive.
  bool build_prompt(std::span<const ChatMessage> messages, int budget, std::vector<llama_token>* tokens,
                    bool* truncated, std::string* error) {
    *truncated = false;
    if (budget < 1) {
      *error = "the context window leaves no room for the prompt";
      return false;
    }
    if (!format_and_tokenize(messages, tokens, error)) return false;
    if (static_cast<int>(tokens->size()) <= budget) return true;
    *truncated = true;

    // The message to cut: the last user message when it can absorb the excess,
    // else the longest message (a long system prompt, a long earlier turn).
    const std::size_t excess_at_start = tokens->size() - static_cast<std::size_t>(budget);
    std::size_t target = messages.size();
    std::vector<llama_token> content;
    for (std::size_t i = messages.size(); i-- > 0;) {
      if (messages[i].role != "user" || messages[i].content.empty()) continue;
      content = detail::tokenize(vocab_, messages[i].content, false, false);
      if (content.size() >= excess_at_start + 16) target = i;
      break;
    }
    if (target == messages.size()) {
      target = 0;
      for (std::size_t i = 1; i < messages.size(); ++i) {
        if (messages[i].content.size() > messages[target].content.size()) target = i;
      }
      content = detail::tokenize(vocab_, messages[target].content, false, false);
    }

    std::vector<ChatMessage> edited(messages.begin(), messages.end());
    std::size_t remove = 0;
    for (int attempt = 0; attempt < 6 && remove < content.size(); ++attempt) {
      // The marker and the re-tokenisation at the two cut points cost a few tokens.
      const std::size_t excess = tokens->size() - static_cast<std::size_t>(budget);
      remove = std::min(content.size(), remove + excess + 8);
      const std::size_t keep = content.size() - remove;
      const std::size_t head = (keep + 1) / 2;
      const std::size_t tail = keep - head;
      const std::span<const llama_token> all(content);
      const std::string head_text = detail::detokenize(vocab_, all.first(head), false);
      const std::string tail_text = detail::detokenize(vocab_, all.last(tail), false);
      std::string cut(detail::utf8_trim_partial(head_text));
      cut += kCutMarker;
      cut += detail::utf8_trim_partial(tail_text);
      edited[target].content = std::move(cut);
      if (!format_and_tokenize(edited, tokens, error)) return false;
      if (static_cast<int>(tokens->size()) <= budget) return true;
    }

    // The other messages alone exceed the budget: cut the token sequence itself.
    // The tail is kept because it holds the marker that opens the assistant turn.
    const std::size_t head = static_cast<std::size_t>(budget) / 2;
    const std::size_t tail = static_cast<std::size_t>(budget) - head;
    std::vector<llama_token> cut(tokens->begin(), tokens->begin() + static_cast<std::ptrdiff_t>(head));
    cut.insert(cut.end(), tokens->end() - static_cast<std::ptrdiff_t>(tail), tokens->end());
    *tokens = std::move(cut);
    return true;
  }

  // ---- evaluation -----------------------------------------------------------

  void clear_memory() { llama_memory_clear(llama_get_memory(context_.get()), true); }

  // Decodes `tokens` at positions first_position.. of sequence 0, requesting
  // logits for the last token only.
  bool decode_prompt(std::span<const llama_token> tokens, int first_position, std::string* error) {
    Batch batch(n_batch_);
    for (std::size_t at = 0; at < tokens.size();) {
      batch.clear();
      while (at < tokens.size() && batch.size() < n_batch_) {
        batch.add(tokens[at], first_position + static_cast<int>(at), 0, at + 1 == tokens.size());
        ++at;
      }
      if (!decode(batch, error)) return false;
    }
    return true;
  }

  bool decode(const Batch& batch, std::string* error) {
    detail::ErrorCapture capture;
    const std::int32_t status = llama_decode(context_.get(), batch.get());
    if (status == 0) return true;
    const char* meaning = status == 1   ? "no room left in the context window"
                          : status == 2 ? "aborted"
                          : status == -1 ? "invalid batch"
                                         : "evaluation failed";
    *error = capture.message(meaning);
    if (error->find(meaning) == std::string::npos) *error = std::string(meaning) + ": " + *error;
    return false;
  }

  // Evaluates `tokens` from an empty memory and accumulates -ln p(tokens[i] | tokens[0..i))
  // for every i >= first_scored (first_scored >= 1).
  bool accumulate_nll(std::span<const llama_token> tokens, std::size_t first_scored, double* nll, int* scored,
                      std::string* error) {
    if (first_scored < 1) first_scored = 1;
    if (tokens.size() <= first_scored) return true;  // nothing to score
    clear_memory();

    struct Row {
      int batch_index;
      std::size_t target;
    };
    Batch batch(n_batch_);
    std::vector<Row> rows;
    rows.reserve(kScoreOutputsPerBatch);

    // The last token is only ever a target: its own logits predict nothing we score.
    const std::size_t end = tokens.size() - 1;
    for (std::size_t at = 0; at < end;) {
      batch.clear();
      rows.clear();
      while (at < end && batch.size() < n_batch_ && static_cast<int>(rows.size()) < kScoreOutputsPerBatch) {
        const bool output = at + 1 >= first_scored;
        if (output) rows.push_back({batch.size(), at + 1});
        batch.add(tokens[at], static_cast<llama_pos>(at), 0, output);
        ++at;
      }
      if (!decode(batch, error)) return false;
      for (const Row& row : rows) {
        const float* logits = llama_get_logits_ith(context_.get(), row.batch_index);
        const double value = detail::negative_log_probability(logits, n_vocab_, tokens[row.target]);
        if (!std::isfinite(value)) {
          *error = "the model produced a non-finite probability";
          return false;
        }
        *nll += value;
        ++*scored;
      }
    }
    return true;
  }

  // ---- sampling -------------------------------------------------------------

  SamplerPtr make_sampler(const GenerateOptions& options, std::string* error) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = true;
    SamplerPtr chain(llama_sampler_chain_init(params));
    if (!chain) {
      *error = "could not create the sampler";
      return nullptr;
    }

    // The grammar comes first: every later stage then only sees tokens that
    // keep the output inside the language.
    if (!options.grammar.empty()) {
      detail::ErrorCapture capture;
      llama_sampler* grammar = llama_sampler_init_grammar(vocab_, options.grammar.c_str(), "root");
      if (!grammar) {
        *error = "invalid grammar: " + capture.message("the GBNF text could not be parsed (is there a root rule?)");
        return nullptr;
      }
      llama_sampler_chain_add(chain.get(), grammar);
    }
    if (options.repeat_penalty > 0.0f && options.repeat_penalty != 1.0f) {
      llama_sampler_chain_add(chain.get(),
                              llama_sampler_init_penalties(n_vocab_, kPenaltyLastN, options.repeat_penalty, 0.0f, 0.0f));
    }
    if (options.temperature <= 0.0f) {
      llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
      return chain;
    }
    if (options.top_k > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(options.top_k));
    if (options.top_p > 0.0f && options.top_p < 1.0f) {
      llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(options.top_p, 1));
    }
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(options.seed));
    return chain;
  }

  // ---- the three operations ---------------------------------------------------

  void generate_locked(std::span<const ChatMessage> messages, const GenerateOptions& options,
                       const PieceCallback& on_piece, GenerateStats* stats) {
    const auto fail = [&](std::string message) {
      stats->stop_reason = "error";
      stats->error = std::move(message);
    };

    // Room kept free for the answer; a prompt that would leave less is cut.
    const int wanted = std::max(0, options.max_tokens);
    const int reserve = std::clamp(wanted, 1, std::max(1, n_ctx_ / 2));
    std::vector<llama_token> prompt;
    std::string error;
    if (!build_prompt(messages, n_ctx_ - reserve, &prompt, &stats->truncated, &error)) return fail(error);
    stats->prompt_tokens = static_cast<int>(prompt.size());

    SamplerPtr sampler = make_sampler(options, &error);
    if (!sampler) return fail(error);

    clear_memory();
    const Stopwatch prompt_clock;
    if (!decode_prompt(prompt, 0, &error)) return fail("prompt evaluation failed: " + error);
    stats->prompt_ms = prompt_clock.elapsed_ms();

    const Stopwatch completion_clock;
    int position = static_cast<int>(prompt.size());
    std::string pending;  // bytes not yet delivered: at most one incomplete character
    Batch next(1);
    stats->stop_reason = "length";
    while (stats->completion_tokens < wanted) {
      const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
      if (token == LLAMA_TOKEN_NULL) {
        fail("the sampler returned no token");
        break;
      }
      if (llama_vocab_is_eog(vocab_, token)) {
        stats->stop_reason = "eos";
        break;
      }
      ++stats->completion_tokens;

      pending += detail::token_to_piece(vocab_, token, false);
      const std::size_t ready = detail::utf8_complete_prefix(pending);
      if (ready > 0) {
        bool keep_going = true;
        if (on_piece) {
          try {
            keep_going = on_piece(std::string_view(pending).substr(0, ready));
          } catch (...) {
            fail("the piece callback threw an exception");
            break;
          }
        }
        pending.erase(0, ready);
        if (!keep_going) {
          stats->stop_reason = "cancelled";
          break;
        }
      }

      if (stats->completion_tokens >= wanted || position >= n_ctx_) break;  // "length"
      next.clear();
      next.add(token, position, 0, true);
      if (!decode(next, &error)) {
        fail("evaluation failed: " + error);
        break;
      }
      ++position;
    }
    // Whatever is left in `pending` is an incomplete character and is dropped.
    stats->completion_ms = completion_clock.elapsed_ms();
  }

  void score_locked(std::span<const ChatMessage> messages, std::string_view continuation, Likelihood* result) {
    std::vector<llama_token> reply = detail::tokenize(vocab_, continuation, false, false);
    if (reply.empty()) {
      result->error = continuation.empty() ? "the continuation is empty" : "the continuation could not be tokenised";
      return;
    }
    // The continuation may take at most three quarters of the window, so that
    // the prompt it is conditioned on keeps a meaningful share.
    const std::size_t reply_limit = static_cast<std::size_t>(std::max(1, n_ctx_ - std::max(1, n_ctx_ / 4)));
    if (reply.size() > reply_limit) {
      reply.resize(reply_limit);
      result->truncated = true;
    }

    std::vector<llama_token> tokens;
    bool prompt_cut = false;
    if (!build_prompt(messages, n_ctx_ - static_cast<int>(reply.size()), &tokens, &prompt_cut, &result->error)) return;
    result->truncated = result->truncated || prompt_cut;
    const std::size_t first_scored = tokens.size();
    tokens.insert(tokens.end(), reply.begin(), reply.end());

    double nll = 0.0;
    int scored = 0;
    if (!accumulate_nll(tokens, first_scored, &nll, &scored, &result->error)) return;
    result->nll = nll;
    result->tokens = scored;
  }

  void score_text_locked(std::string_view text, int window, Likelihood* result) {
    std::vector<llama_token> tokens = detail::tokenize(vocab_, text, false, false);
    if (tokens.size() < 2) {
      result->error = "the text has fewer than two tokens";
      return;
    }
    const std::size_t size = static_cast<std::size_t>(std::clamp(window, 2, n_ctx_));
    const bool add_bos = llama_vocab_get_add_bos(vocab_) && llama_vocab_bos(vocab_) != LLAMA_TOKEN_NULL;

    double nll = 0.0;
    int scored = 0;
    std::vector<llama_token> chunk;
    for (std::size_t start = 0; start < tokens.size(); start += size) {
      const std::size_t length = std::min(size, tokens.size() - start);
      if (length < 2) break;  // a lone trailing token has no context inside its window
      chunk.assign(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                   tokens.begin() + static_cast<std::ptrdiff_t>(start + length));
      // Models trained with a beginning-of-text token expect one at the start
      // of every window; it takes the place of the token that is not scored.
      if (add_bos) chunk[0] = llama_vocab_bos(vocab_);
      std::string error;
      if (!accumulate_nll(chunk, 1, &nll, &scored, &error)) {
        *result = Likelihood{};
        result->error = error;
        return;
      }
    }
    result->nll = nll;
    result->tokens = scored;
  }

  std::mutex mutex_;
  ModelPtr model_;
  LoraPtr adapter_;
  ThreadpoolPtr pool_;
  ContextPtr context_;
  const llama_vocab* vocab_;
  std::string id_;
  std::string template_;
  int n_ctx_;
  int n_batch_;
  int n_vocab_;
  bool warned_template_ = false;
  bool poisoned_ = false;
};

}  // namespace

std::unique_ptr<LanguageModel> load_model(const ModelOptions& options, std::string* error) {
  const auto fail = [&](std::string message) -> std::unique_ptr<LanguageModel> {
    if (error) *error = std::move(message);
    return nullptr;
  };
  if (error) error->clear();
  if (options.path.empty()) return fail("no model file given");
  if (!fs::exists(options.path)) return fail("model file not found: " + options.path);
  if (options.context_tokens < 16) return fail("context_tokens must be at least 16");
  if (!options.adapter_path.empty() && !fs::exists(options.adapter_path)) {
    return fail("adapter file not found: " + options.adapter_path);
  }

  detail::ensure_backend();
  std::unique_ptr<LanguageModel> result;
  std::string message;
  std::string failure;
  // Raw pointers inside the guarded region: on an internal failure they are leaked on purpose.
  llama_model* model = nullptr;
  llama_adapter_lora* adapter = nullptr;
  llama_context* context = nullptr;
  ggml_threadpool* pool = nullptr;
  const bool completed = detail::guarded(
      [&] {
        detail::ErrorCapture capture;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        model = llama_model_load_from_file(options.path.c_str(), model_params);
        if (!model) {
          message = "could not load " + options.path + ": " + capture.message("not a usable GGUF model");
          return;
        }
        if (!llama_model_has_decoder(model)) {
          message = options.path + " is not a generative model";
          return;
        }
        if (!options.adapter_path.empty()) {
          adapter = llama_adapter_lora_init(model, options.adapter_path.c_str());
          if (!adapter) {
            message = "could not load the adapter " + options.adapter_path + ": " +
                      capture.message("not a LoRA adapter for this model");
            return;
          }
        }

        const int threads = detail::resolve_threads(options.threads);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(options.context_tokens);
        context_params.n_batch = static_cast<std::uint32_t>(std::min(options.context_tokens, 2048));
        context_params.n_ubatch = static_cast<std::uint32_t>(std::min(options.context_tokens, 512));
        context_params.n_outputs_max = kScoreOutputsPerBatch;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.no_perf = true;
        context = llama_init_from_model(model, context_params);
        if (!context) {
          message = "could not create a context of " + std::to_string(options.context_tokens) + " tokens for " +
                    options.path + ": " + capture.message("not enough memory?");
          return;
        }
        pool = detail::attach_threadpool(context, threads).release();
        if (adapter) {
          float scale = 1.0f;
          if (llama_set_adapters_lora(context, &adapter, 1, &scale) != 0) {
            message = "could not apply the adapter " + options.adapter_path;
            return;
          }
        }
      },
      &failure);

  if (!completed) return fail("loading " + options.path + " failed: " + failure);  // handles leaked, see guarded()
  if (!message.empty()) {
    if (context) llama_free(context);
    if (pool) ggml_threadpool_free(pool);
    if (adapter) llama_adapter_lora_free(adapter);
    if (model) llama_model_free(model);
    return fail(std::move(message));
  }

  std::string id = detail::model_id(options.path, model);
  if (adapter) id += " + " + detail::file_stem(options.adapter_path);
  PRAG_INFO("llm: loaded %s (context %u tokens, %d threads)", id.c_str(), llama_n_ctx(context),
            llama_n_threads(context));
  return std::make_unique<LlamaModel>(ModelPtr(model), LoraPtr(adapter), ThreadpoolPtr(pool), ContextPtr(context),
                                      std::move(id));
}

}  // namespace prag::llm
