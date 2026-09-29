// Importance matrix: E[x_j^2] at the input of every weight matrix, collected
// through the evaluation callback of the llama.cpp scheduler. The selection of
// nodes and the accumulation follow tools/imatrix/imatrix.cpp, so that the
// result means to llama_model_quantize what an imatrix file would.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "detail/runtime.hpp"
#include "detail/util.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/tuning.hpp"

namespace prag::llm {

namespace {

// llama.cpp ignores matrix products over fewer rows than this, and so do we:
// the statistics of a handful of tokens are dominated by the first positions.
constexpr std::int64_t kMinRows = 16;
constexpr int kMaxPhysicalBatch = 512;

// "CUDA0#blk.0.attn_k.weight#0" -> "blk.0.attn_k.weight"
std::string weight_name(const char* name) {
  const char* first = std::strchr(name, '#');
  if (!first) return name;
  ++first;
  const char* second = std::strchr(first, '#');
  return second ? std::string(first, static_cast<std::size_t>(second - first)) : std::string(first);
}

class Collector {
 public:
  static bool callback(struct ggml_tensor* node, bool ask, void* self) {
    return static_cast<Collector*>(self)->observe(node, ask);
  }

  const std::string& error() const { return error_; }

  // Mean squared activations, tensors in name order.
  void finish(ImportanceMatrix* out) const {
    for (const auto& [name, entry] : entries_) {
      const std::size_t columns = entry.sums.size() / entry.counts.size();
      std::vector<float> values(entry.sums.size());
      bool any = false;
      for (std::size_t part = 0; part < entry.counts.size(); ++part) {
        const std::int64_t count = entry.counts[part];
        any = any || count > 0;
        for (std::size_t j = 0; j < columns; ++j) {
          // A part that was never exercised (an unused expert) gets the neutral weight 1.
          values[part * columns + j] =
              count > 0 ? static_cast<float>(entry.sums[part * columns + j] / static_cast<double>(count)) : 1.0f;
        }
      }
      if (!any) continue;
      out->names.push_back(name);
      out->values.push_back(std::move(values));
    }
  }

 private:
  struct Entry {
    std::vector<double> sums;          // per part (expert) and input column: sum of x^2
    std::vector<std::int64_t> counts;  // per part: rows accumulated
  };

  bool observe(struct ggml_tensor* node, bool ask) {
    const struct ggml_tensor* weight = node->src[0];
    const struct ggml_tensor* input = node->src[1];

    if (ask) {
      if (!error_.empty()) return false;
      if (node->op == GGML_OP_MUL_MAT_ID) return true;
      if (node->op != GGML_OP_MUL_MAT) return false;
      if (!weight || !input) return false;
      if (input->ne[1] < kMinRows || input->type != GGML_TYPE_F32) return false;
      return std::strncmp(weight_name(weight->name).c_str(), "blk.", 4) == 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_.empty()) return false;
    if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float)) {
      error_ = std::string("unexpected activation layout at ") + weight->name;
      return false;
    }

    const char* data = static_cast<const char*>(input->data);
    if (!ggml_backend_buffer_is_host(input->buffer)) {
      scratch_.resize(ggml_nbytes(input));
      ggml_backend_tensor_get(input, scratch_.data(), 0, scratch_.size());
      data = scratch_.data();
    }

    const std::string name = weight_name(weight->name);
    const std::int64_t columns = input->ne[0];
    Entry& entry = entries_[name];

    if (node->op == GGML_OP_MUL_MAT_ID) {
      // Mixture of experts: the weights are [columns, rows, experts], the input
      // is [columns, experts used, tokens] and ids holds the experts chosen per token.
      const struct ggml_tensor* ids = node->src[2];
      const std::int64_t experts = weight->ne[2];
      const std::int64_t used = ids->ne[0];
      const std::int64_t tokens = input->ne[2];
      if (ids->ne[1] != tokens || ggml_nrows(input) != input->ne[1] * input->ne[2]) {
        error_ = "unsupported expert routing layout at " + name;
        return false;
      }
      if (!prepare(entry, name, columns, experts)) return false;
      ids_.resize(ggml_nbytes(ids));
      ggml_backend_tensor_get(ids, ids_.data(), 0, ids_.size());
      for (std::int64_t slot = 0; slot < used; ++slot) {
        for (std::int64_t row = 0; row < tokens; ++row) {
          std::int32_t expert = 0;
          std::memcpy(&expert, ids_.data() + row * ids->nb[1] + slot * ids->nb[0], sizeof(expert));
          if (expert < 0 || expert >= experts) {
            error_ = "expert index out of range at " + name;
            return false;
          }
          const std::int64_t column_block = slot % input->ne[1];
          accumulate(entry.sums.data() + expert * columns, data + column_block * input->nb[1] + row * input->nb[2],
                     columns);
          ++entry.counts[static_cast<std::size_t>(expert)];
        }
      }
    } else {
      const std::int64_t parts = weight->ne[2] * weight->ne[3];
      if (!prepare(entry, name, columns, parts)) return false;
      for (std::int64_t i3 = 0; i3 < input->ne[3]; ++i3) {
        for (std::int64_t i2 = 0; i2 < input->ne[2]; ++i2) {
          const std::int64_t part = (i3 % weight->ne[3]) * weight->ne[2] + (i2 % weight->ne[2]);
          double* sums = entry.sums.data() + part * columns;
          for (std::int64_t row = 0; row < input->ne[1]; ++row) {
            accumulate(sums, data + row * input->nb[1] + i2 * input->nb[2] + i3 * input->nb[3], columns);
          }
        }
      }
      const std::int64_t rows = ggml_nrows(input) / parts;
      for (std::int64_t& count : entry.counts) count += rows;
    }

    for (const double sum : entry.sums) {
      if (!std::isfinite(sum)) {
        error_ = "non-finite activations at " + name;
        return false;
      }
    }
    return true;
  }

  bool prepare(Entry& entry, const std::string& name, std::int64_t columns, std::int64_t parts) {
    const auto size = static_cast<std::size_t>(columns * parts);
    if (entry.sums.empty()) {
      entry.sums.assign(size, 0.0);
      entry.counts.assign(static_cast<std::size_t>(parts), 0);
      return true;
    }
    if (entry.sums.size() != size || entry.counts.size() != static_cast<std::size_t>(parts)) {
      error_ = "inconsistent activation size at " + name;
      return false;
    }
    return true;
  }

  static void accumulate(double* sums, const char* row, std::int64_t columns) {
    for (std::int64_t j = 0; j < columns; ++j) {
      float x;
      std::memcpy(&x, row + j * static_cast<std::int64_t>(sizeof(float)), sizeof(float));
      sums[j] += static_cast<double>(x) * static_cast<double>(x);
    }
  }

  std::mutex mutex_;
  std::map<std::string, Entry> entries_;
  std::vector<char> scratch_;
  std::vector<char> ids_;
  std::string error_;
};

}  // namespace

bool compute_importance(const ImportanceOptions& options, std::string_view calibration_text, ImportanceMatrix* out,
                        const ProgressCallback& progress, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (error) error->clear();
  if (!out) return fail("no output given");
  *out = ImportanceMatrix{};
  if (options.model.empty()) return fail("no model given");
  if (!fs::exists(options.model)) return fail("model file not found: " + options.model);
  if (options.window < kMinRows) return fail("window must be at least " + std::to_string(kMinRows) + " tokens");
  if (options.max_windows < 1) return fail("max_windows must be at least 1");

  detail::ensure_backend();
  std::lock_guard<std::mutex> heavy(detail::heavy_operation_mutex());

  Collector collector;
  std::string message;
  std::string failure;
  bool cancelled = false;
  std::uint64_t evaluated = 0;
  llama_model* model = nullptr;
  llama_context* context = nullptr;
  ggml_threadpool* pool = nullptr;
  const Stopwatch clock;

  const bool completed = detail::guarded(
      [&] {
        detail::ErrorCapture capture;
        model = llama_model_load_from_file(options.model.c_str(), llama_model_default_params());
        if (!model) {
          message = "could not load " + options.model + ": " + capture.message("not a usable GGUF model");
          return;
        }
        const llama_vocab* vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tokens = detail::tokenize(vocab, calibration_text, true, options.parse_special);
        if (static_cast<std::int64_t>(tokens.size()) < kMinRows) {
          message = "the calibration text has " + std::to_string(tokens.size()) + " tokens; at least " +
                    std::to_string(kMinRows) + " are needed";
          return;
        }
        const int window = static_cast<int>(std::min<std::size_t>(tokens.size(), static_cast<std::size_t>(options.window)));
        const int windows = static_cast<int>(
            std::min<std::size_t>(tokens.size() / static_cast<std::size_t>(window), static_cast<std::size_t>(options.max_windows)));

        const int threads = detail::resolve_threads(options.threads);
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(window);
        params.n_batch = static_cast<std::uint32_t>(window);
        params.n_ubatch = static_cast<std::uint32_t>(std::min(window, kMaxPhysicalBatch));
        params.n_threads = threads;
        params.n_threads_batch = threads;
        params.no_perf = true;
        params.cb_eval = Collector::callback;
        params.cb_eval_user_data = &collector;
        context = llama_init_from_model(model, params);
        if (!context) {
          message = "could not create a context of " + std::to_string(window) + " tokens: " +
                    capture.message("not enough memory?");
          return;
        }
        // The callback makes the scheduler evaluate the graph in many small
        // pieces; without a pool every piece would start its own threads.
        pool = detail::attach_threadpool(context, threads).release();

        const bool add_bos = llama_vocab_get_add_bos(vocab) && llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL;
        detail::Batch batch(window);
        for (int w = 0; w < windows; ++w) {
          if (progress && !progress(static_cast<double>(w) / windows,
                                    "calibrating: window " + std::to_string(w + 1) + " of " + std::to_string(windows))) {
            cancelled = true;
            return;
          }
          llama_memory_clear(llama_get_memory(context), true);
          const llama_token* first = tokens.data() + static_cast<std::ptrdiff_t>(w) * window;
          batch.clear();
          for (int k = 0; k < window; ++k) {
            // Every position is an output: otherwise the last block would only
            // be evaluated for the final token and its statistics would be lost.
            batch.add(k == 0 && add_bos ? llama_vocab_bos(vocab) : first[k], k, 0, true);
          }
          const std::int32_t status = llama_decode(context, batch.get());
          if (!collector.error().empty()) {
            message = collector.error();
            return;
          }
          if (status != 0) {
            message = "evaluation of window " + std::to_string(w + 1) + " failed: " +
                      capture.message("llama_decode returned " + std::to_string(status));
            return;
          }
          evaluated += static_cast<std::uint64_t>(window);
        }
      },
      &failure);

  if (!completed) return fail("importance calibration failed: " + failure);  // handles leaked, see guarded()
  if (context) llama_free(context);
  if (pool) ggml_threadpool_free(pool);
  if (model) llama_model_free(model);
  if (cancelled) return fail("cancelled");
  if (!message.empty()) return fail(std::move(message));

  collector.finish(out);
  out->tokens = evaluated;
  if (out->empty()) {
    *out = ImportanceMatrix{};
    return fail("no matrix multiplication was observed; the model has no tensors named blk.*");
  }
  if (progress) progress(1.0, "calibrated " + std::to_string(out->names.size()) + " tensors");
  PRAG_INFO("llm: importance matrix over %llu tokens, %zu tensors, %.1f s", static_cast<unsigned long long>(evaluated),
            out->names.size(), clock.elapsed_s());
  return true;
}

}  // namespace prag::llm
