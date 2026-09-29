// embed::Embedder on top of a llama.cpp embedding model.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>

#include "detail/runtime.hpp"
#include "detail/util.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/model.hpp"

namespace prag::llm {

namespace {

using detail::Batch;
using detail::ContextPtr;
using detail::ModelPtr;
using detail::ThreadpoolPtr;

// Texts evaluated by one decode call. The sequences share one key-value
// buffer, so the limit that matters in practice is the token count.
constexpr int kMaxSequences = 32;

// Tokens per decode call when a text may be evaluated in pieces. llama.cpp
// sizes its compute and output buffers by this number (the output buffer holds
// a row of vocabulary size per token even when only embeddings are read), so
// it bounds the memory of the embedder: about 1.5 GiB at 512 for a 0.6B model,
// against 2.6 GiB when a window of 2048 tokens is evaluated in one call.
constexpr int kPieceTokens = 512;

const char* pooling_name(enum llama_pooling_type type) {
  switch (type) {
    case LLAMA_POOLING_TYPE_MEAN: return "mean";
    case LLAMA_POOLING_TYPE_CLS: return "cls";
    case LLAMA_POOLING_TYPE_LAST: return "last";
    case LLAMA_POOLING_TYPE_RANK: return "rank";
    case LLAMA_POOLING_TYPE_NONE: return "none";
    case LLAMA_POOLING_TYPE_UNSPECIFIED: break;
  }
  return "unspecified";
}

// With last-token pooling and causal attention the embedding is the state of
// the final position, which sees the earlier positions through the key-value
// buffer: such a text can be evaluated in several calls. Every other pooling
// needs the whole text in one call.
bool pools_in_pieces(const llama_context* context) {
  return llama_pooling_type(context) == LLAMA_POOLING_TYPE_LAST && llama_get_causal_attn(context);
}

class LlamaEmbedder final : public embed::Embedder {
 public:
  LlamaEmbedder(ModelPtr model, ThreadpoolPtr pool, ContextPtr context, const EmbedderOptions& options,
                std::string model_id)
      : model_(std::move(model)),
        pool_(std::move(pool)),
        context_(std::move(context)),
        vocab_(llama_model_get_vocab(model_.get())),
        query_prefix_(options.query_prefix),
        document_prefix_(options.document_prefix),
        n_ctx_(static_cast<int>(llama_n_ctx(context_.get()))),
        n_batch_(static_cast<int>(llama_n_batch(context_.get()))) {
    const auto native = static_cast<std::size_t>(llama_model_n_embd_out(model_.get()));
    dim_ = options.truncate_dim > 0 && options.truncate_dim < native ? options.truncate_dim : native;
    window_ = pools_in_pieces(context_.get()) ? n_ctx_ : std::min(n_ctx_, n_batch_);

    // A tokeniser that ends every text with an end-of-text token on its own
    // (Qwen3-Embedding) needs the same care as an explicit request: the token
    // is what gets pooled, so it has to survive the truncation of a long text.
    if (options.append_eos || llama_vocab_get_add_eos(vocab_)) {
      end_token_ = llama_vocab_eos(vocab_);
      if (end_token_ == LLAMA_TOKEN_NULL) end_token_ = llama_vocab_sep(vocab_);
      if (end_token_ == LLAMA_TOKEN_NULL && options.append_eos) {
        PRAG_WARN("llm: %s defines no end-of-text token; append_eos is ignored", model_id.c_str());
      }
    }

    // A text may use the window minus the longer prefix, the tokens the
    // tokeniser adds by itself and the end token.
    const auto overhead_of = [&](const std::string& prefix) {
      std::vector<llama_token> tokens = detail::tokenize(vocab_, prefix, true, false);
      const bool ends = end_token_ != LLAMA_TOKEN_NULL;
      if (ends && !tokens.empty() && tokens.back() == end_token_) tokens.pop_back();
      return tokens.size() + (ends ? 1 : 0);
    };
    const std::size_t overhead = std::max(overhead_of(query_prefix_), overhead_of(document_prefix_));
    const auto window = static_cast<std::size_t>(window_);
    max_tokens_ = window > overhead + 1 ? window - overhead : 1;

    char settings[160];
    std::snprintf(settings, sizeof(settings), "|pool=%s|dim=%zu|ctx=%d|eos=%d|prefix=%016llx",
                  pooling_name(llama_pooling_type(context_.get())), dim_, window_, end_token_ != LLAMA_TOKEN_NULL ? 1 : 0,
                  static_cast<unsigned long long>(hash_combine(hash64(query_prefix_), hash64(document_prefix_))));
    id_ = std::move(model_id) + settings;
  }

  ~LlamaEmbedder() override {
    if (poisoned_) {
      // See detail::guarded: the objects are in an unknown state, freeing them could crash.
      (void)context_.release();
      (void)pool_.release();
      (void)model_.release();
    } else {
      context_.reset();  // before the thread pool and the model it refers to
    }
  }

  std::size_t dim() const override { return dim_; }
  std::string id() const override { return id_; }
  std::size_t max_tokens() const override { return max_tokens_; }

  void embed(std::span<const std::string_view> texts, embed::Role role, float* out) override {
    if (texts.empty() || !out) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(out, out + texts.size() * dim_, 0.0f);  // rows that fail stay zero
    if (poisoned_) {
      PRAG_ERROR("llm: %s is unusable after an internal failure of llama.cpp", id_.c_str());
      return;
    }
    std::string failure;
    if (!detail::guarded([&] { embed_locked(texts, role, out); }, &failure)) {
      poisoned_ = true;
      PRAG_ERROR("llm: embedding failed: %s", failure.c_str());
    }
  }

 private:
  std::vector<llama_token> prepare(std::string_view text, embed::Role role) const {
    const std::string& prefix = role == embed::Role::Query ? query_prefix_ : document_prefix_;
    std::string input;
    input.reserve(prefix.size() + text.size());
    input += prefix;
    input += text;
    // The text is data: markers that look like control tokens stay plain text.
    std::vector<llama_token> tokens = detail::tokenize(vocab_, input, true, false);

    const bool ends = end_token_ != LLAMA_TOKEN_NULL;
    if (ends && !tokens.empty() && tokens.back() == end_token_) tokens.pop_back();  // added by the tokeniser
    const std::size_t limit = static_cast<std::size_t>(window_) - (ends ? 1 : 0);
    if (tokens.size() > limit) tokens.resize(limit);
    if (ends) tokens.push_back(end_token_);
    if (tokens.empty()) {
      // An empty text still needs one position to pool over.
      llama_token filler = llama_vocab_bos(vocab_);
      if (filler == LLAMA_TOKEN_NULL) filler = llama_vocab_eos(vocab_);
      if (filler == LLAMA_TOKEN_NULL) filler = 0;
      tokens.push_back(filler);
    }
    return tokens;
  }

  bool decode(const Batch& batch, std::size_t texts) {
    detail::ErrorCapture capture;
    const std::int32_t status = llama_decode(context_.get(), batch.get());
    if (status == 0) return true;
    PRAG_ERROR("llm: embedding %zu text(s) failed: %s", texts,
               capture.message("llama_decode returned " + std::to_string(status)).c_str());
    return false;
  }

  // Copies the pooled state of `sequence` into `row`.
  void store(llama_seq_id sequence, float* row) {
    const float* pooled = llama_get_embeddings_seq(context_.get(), sequence);
    if (!pooled) {
      PRAG_ERROR("llm: the model returned no embedding for a text");
      return;
    }
    // Matryoshka models order their dimensions by importance: keep the
    // leading ones, then restore unit length.
    std::memcpy(row, pooled, dim_ * sizeof(float));
    embed::normalize(row, dim_);
  }

  void embed_locked(std::span<const std::string_view> texts, embed::Role role, float* out) {
    Batch batch(n_batch_);
    std::vector<std::size_t> rows;  // text index of every sequence in the batch

    const auto flush = [&] {
      if (rows.empty()) return;
      llama_memory_clear(llama_get_memory(context_.get()), true);
      if (decode(batch, rows.size())) {
        for (std::size_t sequence = 0; sequence < rows.size(); ++sequence) {
          store(static_cast<llama_seq_id>(sequence), out + rows[sequence] * dim_);
        }
      }
      batch.clear();
      rows.clear();
    };

    for (std::size_t i = 0; i < texts.size(); ++i) {
      const std::vector<llama_token> tokens = prepare(texts[i], role);

      if (static_cast<int>(tokens.size()) > n_batch_) {
        // Longer than one call: evaluated alone, piece by piece. The state
        // after the last piece is the embedding.
        flush();
        llama_memory_clear(llama_get_memory(context_.get()), true);
        bool ok = true;
        for (std::size_t at = 0; ok && at < tokens.size();) {
          batch.clear();
          while (at < tokens.size() && batch.size() < n_batch_) {
            batch.add(tokens[at], static_cast<llama_pos>(at), 0, true);
            ++at;
          }
          ok = decode(batch, 1);
        }
        if (ok) store(0, out + i * dim_);
        batch.clear();
        continue;
      }

      if (batch.size() + static_cast<int>(tokens.size()) > n_batch_ || static_cast<int>(rows.size()) >= kMaxSequences) {
        flush();
      }
      const auto sequence = static_cast<llama_seq_id>(rows.size());
      for (std::size_t position = 0; position < tokens.size(); ++position) {
        batch.add(tokens[position], static_cast<llama_pos>(position), sequence, true);
      }
      rows.push_back(i);
    }
    flush();
  }

  std::mutex mutex_;
  ModelPtr model_;
  ThreadpoolPtr pool_;
  ContextPtr context_;
  const llama_vocab* vocab_;
  std::string query_prefix_;
  std::string document_prefix_;
  std::string id_;
  int n_ctx_;
  int n_batch_;
  int window_ = 0;  // longest token sequence that is embedded
  std::size_t dim_ = 0;
  std::size_t max_tokens_ = 0;
  llama_token end_token_ = LLAMA_TOKEN_NULL;
  bool poisoned_ = false;
};

}  // namespace

std::unique_ptr<embed::Embedder> load_embedder(const EmbedderOptions& options, std::string* error) {
  const auto fail = [&](std::string message) -> std::unique_ptr<embed::Embedder> {
    if (error) *error = std::move(message);
    return nullptr;
  };
  if (error) error->clear();
  if (options.path.empty()) return fail("no embedding model file given");
  if (!fs::exists(options.path)) return fail("embedding model file not found: " + options.path);
  if (options.context_tokens < 16) return fail("context_tokens must be at least 16");

  detail::ensure_backend();
  std::string message;
  std::string failure;
  llama_model* model = nullptr;
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
        if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
          message = options.path + " is an encoder-decoder model; it cannot be used for embeddings";
          return;
        }

        const int threads = detail::resolve_threads(options.threads);
        const auto create = [&](enum llama_pooling_type pooling, int batch) {
          llama_context_params params = llama_context_default_params();
          params.n_ctx = static_cast<std::uint32_t>(options.context_tokens);
          params.n_batch = static_cast<std::uint32_t>(batch);
          params.n_ubatch = static_cast<std::uint32_t>(batch);
          params.n_seq_max = kMaxSequences;
          params.kv_unified = true;
          params.embeddings = true;
          params.pooling_type = pooling;
          params.n_threads = threads;
          params.n_threads_batch = threads;
          params.no_perf = true;
          return llama_init_from_model(model, params);
        };
        // Pooling as stored in the model. Whether a text may be evaluated in
        // pieces is known once a context exists, so the first one is small.
        const int pieces = std::min(options.context_tokens, kPieceTokens);
        context = create(LLAMA_POOLING_TYPE_UNSPECIFIED, pieces);
        if (context && llama_pooling_type(context) == LLAMA_POOLING_TYPE_NONE) {
          PRAG_WARN("llm: %s stores no pooling type; using mean pooling", options.path.c_str());
          llama_free(context);
          context = create(LLAMA_POOLING_TYPE_MEAN, options.context_tokens);
        } else if (context && pieces < options.context_tokens && !pools_in_pieces(context)) {
          llama_free(context);
          context = create(LLAMA_POOLING_TYPE_UNSPECIFIED, options.context_tokens);
        }
        if (!context) {
          message = "could not create an embedding context of " + std::to_string(options.context_tokens) +
                    " tokens for " + options.path + ": " + capture.message("not enough memory?");
          return;
        }
        if (llama_pooling_type(context) == LLAMA_POOLING_TYPE_RANK) {
          message = options.path + " is a reranking model, not an embedding model";
          return;
        }
        pool = detail::attach_threadpool(context, threads).release();
      },
      &failure);

  if (!completed) return fail("loading " + options.path + " failed: " + failure);  // handles leaked, see guarded()
  if (!message.empty()) {
    if (context) llama_free(context);
    if (pool) ggml_threadpool_free(pool);
    if (model) llama_model_free(model);
    return fail(std::move(message));
  }

  std::string id = detail::model_id(options.path, model);
  auto embedder = std::make_unique<LlamaEmbedder>(ModelPtr(model), ThreadpoolPtr(pool), ContextPtr(context), options,
                                                  std::move(id));
  PRAG_INFO("llm: loaded embedder %s (dimension %zu)", embedder->id().c_str(), embedder->dim());
  return embedder;
}

}  // namespace prag::llm
