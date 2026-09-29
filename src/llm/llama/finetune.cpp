// Fine-tuning of the last transformer blocks with the training support of
// llama.cpp (llama_opt_init / llama_opt_epoch), which upstream still marks as
// work in progress. What this file adds around it:
//
//   * windows of any length: llama.cpp rounds the context up to a multiple of
//     256 tokens and reads one context-sized record per window, so a record is
//     the window followed by padding that is never evaluated;
//   * a seeded train/validation split and a per-step cosine learning rate;
//   * cancellation: the epoch callback cannot stop llama_opt_epoch, so a
//     cancelled run makes every remaining evaluation abort at its first node;
//   * the metadata of the base model: llama_model_save_to_file writes the
//     tensors but neither the chat template nor the file type, and it stores
//     the training window as the context length of the model.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numbers>
#include <unordered_set>
#include <vector>

#include "detail/runtime.hpp"
#include "detail/util.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/tuning.hpp"

namespace prag::llm {

namespace {

using detail::GgmlPtr;
using detail::GgufPtr;

constexpr int kMinWindow = 8;

std::filesystem::path to_path(const std::string& utf8) {
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

// ---- GGUF files -------------------------------------------------------------

struct GgufFile {
  GgufPtr gguf;
  GgmlPtr tensors;  // tensor descriptions without data
};

bool open_gguf(const std::string& path, GgufFile* file, std::string* error) {
  detail::ErrorCapture capture;
  struct ggml_context* tensors = nullptr;
  struct gguf_init_params params;
  params.no_alloc = true;
  params.ctx = &tensors;
  file->gguf.reset(gguf_init_from_file(path.c_str(), params));
  file->tensors.reset(tensors);
  if (!file->gguf) {
    *error = "could not read " + path + ": " + capture.message("not a GGUF file");
    return false;
  }
  return true;
}

// Empty when every tensor is float32, else the reason.
std::string check_float32(const std::string& path) {
  GgufFile file;
  std::string error;
  if (!open_gguf(path, &file, &error)) return error;
  const std::int64_t n = gguf_get_n_tensors(file.gguf.get());
  if (n == 0) return path + " contains no tensors";
  for (std::int64_t i = 0; i < n; ++i) {
    const enum ggml_type type = gguf_get_tensor_type(file.gguf.get(), i);
    if (type == GGML_TYPE_F32) continue;
    return "fine-tuning needs float32 weights, but " + std::string(gguf_get_tensor_name(file.gguf.get(), i)) +
           " in " + path + " is stored as " + ggml_type_name(type) +
           "; convert the model first with quantize(type = \"F32\")";
  }
  return {};
}

bool copy_bytes(std::ifstream& in, std::uint64_t offset, std::uint64_t size, std::ofstream& out,
                std::vector<char>& buffer) {
  in.clear();
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  while (size > 0 && in && out) {
    const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(size, buffer.size()));
    in.read(buffer.data(), chunk);
    if (in.gcount() != chunk) return false;
    out.write(buffer.data(), chunk);
    size -= static_cast<std::uint64_t>(chunk);
  }
  return size == 0 && static_cast<bool>(out);
}

// Writes `output` with the metadata and the tensor order of `base` and the
// tensor data of `tuned`. A tensor that `tuned` lacks is taken from `base`.
bool write_with_base_metadata(const std::string& base_path, const std::string& tuned_path, const std::string& output,
                              std::string* error) {
  GgufFile base;
  GgufFile tuned;
  if (!open_gguf(base_path, &base, error) || !open_gguf(tuned_path, &tuned, error)) return false;

  GgufPtr out(gguf_init_empty());
  gguf_set_kv(out.get(), base.gguf.get());
  if (gguf_get_alignment(out.get()) != gguf_get_alignment(base.gguf.get())) {
    gguf_remove_key(out.get(), "general.alignment");  // offsets below follow the alignment of `out`
  }

  struct Source {
    bool from_tuned;
    std::uint64_t offset;
    std::uint64_t size;
  };
  std::vector<Source> sources;
  const std::int64_t n = gguf_get_n_tensors(base.gguf.get());
  sources.reserve(static_cast<std::size_t>(n));
  std::int64_t missing = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    const char* name = gguf_get_tensor_name(base.gguf.get(), i);
    const std::int64_t j = gguf_find_tensor(tuned.gguf.get(), name);
    if (j < 0) {
      ++missing;
      gguf_add_tensor(out.get(), ggml_get_tensor(base.tensors.get(), name));
      sources.push_back({false, gguf_get_data_offset(base.gguf.get()) + gguf_get_tensor_offset(base.gguf.get(), i),
                         gguf_get_tensor_size(base.gguf.get(), i)});
      continue;
    }
    if (gguf_get_tensor_type(tuned.gguf.get(), j) != gguf_get_tensor_type(base.gguf.get(), i) ||
        gguf_get_tensor_size(tuned.gguf.get(), j) != gguf_get_tensor_size(base.gguf.get(), i)) {
      *error = std::string("the trained tensor ") + name + " differs in type or size from the base model";
      return false;
    }
    gguf_add_tensor(out.get(), ggml_get_tensor(tuned.tensors.get(), name));
    sources.push_back({true, gguf_get_data_offset(tuned.gguf.get()) + gguf_get_tensor_offset(tuned.gguf.get(), j),
                       gguf_get_tensor_size(tuned.gguf.get(), j)});
  }
  if (missing > 0) PRAG_DEBUG("llm: %lld tensor(s) were copied from the base model", static_cast<long long>(missing));

  std::ifstream base_in(to_path(base_path), std::ios::binary);
  std::ifstream tuned_in(to_path(tuned_path), std::ios::binary);
  std::ofstream file(to_path(output), std::ios::binary | std::ios::trunc);
  if (!base_in || !tuned_in || !file) {
    *error = "could not open the files needed to write " + output;
    return false;
  }

  std::vector<char> header(gguf_get_meta_size(out.get()));
  gguf_get_meta_data(out.get(), header.data());
  file.write(header.data(), static_cast<std::streamsize>(header.size()));

  const std::uint64_t alignment = gguf_get_alignment(out.get());
  std::vector<char> buffer(std::size_t{8} << 20);
  const std::vector<char> zeros(static_cast<std::size_t>(alignment), '\0');
  std::uint64_t written = 0;  // bytes of the data section
  for (std::int64_t i = 0; i < n; ++i) {
    const Source& source = sources[static_cast<std::size_t>(i)];
    const std::uint64_t offset = gguf_get_tensor_offset(out.get(), i);
    if (offset < written || offset - written > alignment) {
      *error = "unexpected tensor layout while writing " + output;
      return false;
    }
    file.write(zeros.data(), static_cast<std::streamsize>(offset - written));
    if (!copy_bytes(source.from_tuned ? tuned_in : base_in, source.offset, source.size, file, buffer)) {
      *error = "could not copy tensor data into " + output + " (disk full?)";
      return false;
    }
    written = offset + source.size;
  }
  const std::uint64_t padded = (written + alignment - 1) / alignment * alignment;
  file.write(zeros.data(), static_cast<std::streamsize>(padded - written));
  file.flush();
  if (!file) {
    *error = "could not write " + output + " (disk full?)";
    return false;
  }
  return true;
}

// ---- state shared with the callbacks ---------------------------------------

struct Trainer {
  // parameter selection
  bool all_parameters = false;
  int first_block = 0;  // blocks with this index or a higher one are trained
  std::unordered_set<const struct ggml_tensor*> selected;
  std::uint64_t trainable_parameters = 0;

  // optimiser
  double learning_rate = 1e-5;
  double min_learning_rate = 1e-5;
  double weight_decay = 0.0;
  std::int64_t total_steps = 1;
  std::int64_t step = 0;

  // progress
  const ProgressCallback* progress = nullptr;
  int epoch = 0;
  int epochs = 1;
  std::int64_t train_windows = 0;
  std::int64_t validation_windows = 0;
  std::int64_t done_in_epoch = 0;
  std::atomic<bool> cancel{false};

  double rate_at(std::int64_t at) const {
    if (total_steps <= 1 || min_learning_rate >= learning_rate) return learning_rate;
    const double t = static_cast<double>(std::min(at, total_steps - 1)) / static_cast<double>(total_steps - 1);
    return min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1.0 + std::cos(std::numbers::pi * t));
  }

  bool report(double fraction, const std::string& message) {
    if (!progress || !*progress) return true;
    try {
      return (*progress)(std::clamp(fraction, 0.0, 1.0), message);
    } catch (...) {
      return false;  // a throwing callback cancels; nothing may propagate through llama.cpp
    }
  }
};

// The epoch callback of llama.cpp has no user data argument.
thread_local Trainer* t_trainer = nullptr;

// "blk.17.attn_q.weight" -> 17; -1 for tensors outside the blocks.
int block_index(const char* name) {
  const char* at = std::strstr(name, "blk.");
  if (!at) return -1;
  at += 4;
  if (*at < '0' || *at > '9') return -1;
  int index = 0;
  while (*at >= '0' && *at <= '9') {
    index = index * 10 + (*at - '0');
    ++at;
  }
  return *at == '.' ? index : -1;
}

// Attention reads keys and values from the cache buffer of llama.cpp, into
// which they are copied; the backward pass does not follow the copy. The key
// and value projections therefore never receive a gradient, and selecting
// them would only expose them to weight decay.
bool reaches_the_loss(const char* name) {
  return std::strstr(name, ".attn_k.") == nullptr && std::strstr(name, ".attn_v.") == nullptr;
}

bool select_parameter(const struct ggml_tensor* tensor, void* self) {
  auto& trainer = *static_cast<Trainer*>(self);
  // llama.cpp refuses these two after asking; answer no, so that the count is right.
  if (std::strcmp(tensor->name, "token_embd.weight") == 0 || std::strcmp(tensor->name, "rope_freqs.weight") == 0) {
    return false;
  }
  if (!reaches_the_loss(tensor->name)) return false;
  if (!trainer.all_parameters) {
    const int block = block_index(tensor->name);
    if (block < 0 || block < trainer.first_block) return false;
  }
  if (trainer.selected.insert(tensor).second) {
    trainer.trainable_parameters += static_cast<std::uint64_t>(ggml_nelements(tensor));
  }
  return true;
}

struct ggml_opt_optimizer_params optimizer_parameters(void* self) {
  auto& trainer = *static_cast<Trainer*>(self);
  struct ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
  const auto rate = static_cast<float>(trainer.rate_at(trainer.step));
  ++trainer.step;
  params.adamw.alpha = rate;
  params.sgd.alpha = rate;
  params.adamw.wd = static_cast<float>(trainer.weight_decay);
  params.sgd.wd = static_cast<float>(trainer.weight_decay);
  return params;
}

void epoch_callback(bool train, ggml_opt_context_t /*opt*/, ggml_opt_dataset_t /*dataset*/, ggml_opt_result_t result,
                    std::int64_t /*ibatch*/, std::int64_t /*ibatch_max*/, std::int64_t /*t_start_us*/) {
  Trainer* trainer = t_trainer;
  if (!trainer || trainer->cancel.load(std::memory_order_relaxed)) return;
  ++trainer->done_in_epoch;

  double loss = 0.0;
  ggml_opt_result_loss(result, &loss, nullptr);
  const std::int64_t per_epoch = trainer->train_windows + trainer->validation_windows;
  const double fraction = (static_cast<double>(trainer->epoch) * static_cast<double>(per_epoch) +
                           static_cast<double>(trainer->done_in_epoch)) /
                          (static_cast<double>(trainer->epochs) * static_cast<double>(per_epoch));
  char message[160];
  if (train) {
    std::snprintf(message, sizeof(message), "epoch %d of %d: trained on window %lld of %lld, loss %.4f",
                  trainer->epoch + 1, trainer->epochs, static_cast<long long>(trainer->done_in_epoch),
                  static_cast<long long>(trainer->train_windows), loss);
  } else {
    std::snprintf(message, sizeof(message), "epoch %d of %d: validated window %lld of %lld, loss %.4f",
                  trainer->epoch + 1, trainer->epochs,
                  static_cast<long long>(trainer->done_in_epoch - trainer->train_windows),
                  static_cast<long long>(trainer->validation_windows), loss);
  }
  if (!trainer->report(fraction, message)) trainer->cancel.store(true, std::memory_order_relaxed);
}

bool abort_requested(void* self) { return static_cast<Trainer*>(self)->cancel.load(std::memory_order_relaxed); }

// Fisher-Yates with a fixed generator: the same order on every platform.
void shuffle(std::vector<std::int64_t>& values, std::size_t first, std::size_t last, std::uint64_t& state) {
  for (std::size_t i = last; i > first + 1; --i) {
    state = mix64(state + 0x9e3779b97f4a7c15ULL);
    const std::size_t j = first + static_cast<std::size_t>(state % (i - first));
    std::swap(values[i - 1], values[j]);
  }
}

std::string validate(const FineTuneOptions& options) {
  if (options.input.empty()) return "no input model given";
  if (options.output.empty()) return "no output file given";
  if (!fs::exists(options.input)) return "input model not found: " + options.input;
  if (fs::normalize(options.input) == fs::normalize(options.output)) return "input and output are the same file";
  if (options.window < kMinWindow) return "window must be at least " + std::to_string(kMinWindow) + " tokens";
  if (options.epochs < 1) return "epochs must be at least 1";
  if (!(options.validation_fraction >= 0.0 && options.validation_fraction < 1.0)) {
    return "validation_fraction must be in [0, 1)";
  }
  if (!(options.learning_rate > 0.0) || !std::isfinite(options.learning_rate)) return "learning_rate must be positive";
  if (!(options.weight_decay >= 0.0 && options.weight_decay <= 1.0)) return "weight_decay must be in [0, 1]";
  if (options.trainable_blocks < 0) return "trainable_blocks must not be negative";
  if (options.stride < 0) return "stride must not be negative";
  const std::string optimizer = detail::to_lower(options.optimizer);
  if (optimizer != "adamw" && optimizer != "sgd") {
    return "unknown optimizer \"" + options.optimizer + "\"; use \"adamw\" or \"sgd\"";
  }
  if (options.train_embeddings) {
    return "train_embeddings is not supported: llama.cpp excludes token_embd.weight from training";
  }
  return {};
}

}  // namespace

FineTuneReport fine_tune(const FineTuneOptions& options, std::string_view training_text,
                         const ProgressCallback& progress) {
  FineTuneReport report;
  const Stopwatch clock;
  const auto finish = [&](std::string message) {
    report.ok = false;
    report.error = std::move(message);
    report.seconds = clock.elapsed_s();
    return report;
  };

  if (std::string problem = validate(options); !problem.empty()) return finish(std::move(problem));
  detail::ensure_backend();
  if (std::string problem = check_float32(options.input); !problem.empty()) return finish(std::move(problem));

  std::lock_guard<std::mutex> heavy(detail::heavy_operation_mutex());
  const std::string parent = to_path(options.output).parent_path().generic_string();
  if (!parent.empty() && !fs::create_directories(parent)) return finish("could not create the directory " + parent);
  const std::string raw_path = options.output + ".raw";          // as written by llama_model_save_to_file
  const std::string partial_path = options.output + ".partial";  // with the metadata of the base model
  fs::remove_file(raw_path);
  fs::remove_file(partial_path);

  Trainer trainer;
  trainer.progress = &progress;
  trainer.epochs = options.epochs;
  trainer.learning_rate = options.learning_rate;
  // Without a usable lower end the rate stays constant.
  trainer.min_learning_rate = options.min_learning_rate > 0.0
                                  ? std::min(options.min_learning_rate, options.learning_rate)
                                  : options.learning_rate;
  trainer.weight_decay = options.weight_decay;

  std::string message;
  std::string failure;
  llama_model* model = nullptr;
  llama_context* context = nullptr;
  int blocks = 0;
  ggml_opt_dataset_t dataset = nullptr;
  ggml_opt_result_t result_train = nullptr;
  ggml_opt_result_t result_validation = nullptr;
  bool saved = false;

  const bool completed = detail::guarded(
      [&] {
        detail::ErrorCapture capture;
        if (!trainer.report(0.0, "loading " + detail::file_stem(options.input))) {
          report.cancelled = true;
          return;
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;  // the optimiser writes into the weights
        model = llama_model_load_from_file(options.input.c_str(), model_params);
        if (!model) {
          message = "could not load " + options.input + ": " + capture.message("not a usable GGUF model");
          return;
        }
        const llama_vocab* vocab = llama_model_get_vocab(model);
        blocks = llama_model_n_layer(model);
        report.total_parameters = llama_model_n_params(model);
        trainer.all_parameters = options.trainable_blocks == 0 || options.trainable_blocks >= blocks;
        trainer.first_block = std::max(0, blocks - options.trainable_blocks);
        if (options.window > llama_model_n_ctx_train(model)) {
          message = "window (" + std::to_string(options.window) + ") exceeds the context length of the model (" +
                    std::to_string(llama_model_n_ctx_train(model)) + ")";
          return;
        }

        // ---- windows ---------------------------------------------------------
        const std::vector<llama_token> tokens = detail::tokenize(vocab, training_text, true, options.parse_special);
        const auto window = static_cast<std::size_t>(options.window);
        const auto stride = static_cast<std::size_t>(options.stride > 0 ? options.stride : options.window);
        // A window needs one more token: the label of its last position.
        if (tokens.size() < window + 1) {
          message = "the training text has " + std::to_string(tokens.size()) + " tokens; one window of " +
                    std::to_string(window) + " tokens needs " + std::to_string(window + 1);
          return;
        }
        const std::int64_t windows = static_cast<std::int64_t>((tokens.size() - window - 1) / stride) + 1;
        std::int64_t validation = static_cast<std::int64_t>(
            std::llround(static_cast<double>(windows) * options.validation_fraction));
        if (options.validation_fraction > 0.0 && validation == 0 && windows >= 2) validation = 1;
        validation = std::min(validation, windows - 1);
        report.train_windows = windows - validation;
        report.validation_windows = validation;
        trainer.train_windows = report.train_windows;
        trainer.validation_windows = report.validation_windows;
        trainer.total_steps = report.train_windows * options.epochs;

        std::vector<std::int64_t> order(static_cast<std::size_t>(windows));
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int64_t>(i);
        std::uint64_t random = hash_combine(0x70726167ULL, options.seed);
        shuffle(order, 0, order.size(), random);  // the first train_windows entries are the training set

        // ---- context ---------------------------------------------------------
        const int threads = detail::resolve_threads(options.threads);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(options.window);
        context_params.n_batch = static_cast<std::uint32_t>(options.window);
        context_params.n_ubatch = static_cast<std::uint32_t>(options.window);  // one optimiser step per window
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // it has no backward pass
        context_params.type_k = GGML_TYPE_F32;  // the backward pass needs float32 keys and values
        context_params.type_v = GGML_TYPE_F32;
        context_params.no_perf = true;
        context_params.abort_callback = abort_requested;
        context_params.abort_callback_data = &trainer;
        context = llama_init_from_model(model, context_params);
        if (!context) {
          message = "could not create a training context of " + std::to_string(options.window) +
                    " tokens: " + capture.message("not enough memory?");
          return;
        }
        // The training loop computes through the scheduler directly and would
        // run with the default thread count of the backend; one ordinary
        // evaluation makes the context install its own.
        {
          detail::Batch warm_up(1);
          warm_up.add(tokens[0], 0, 0, true);
          if (llama_decode(context, warm_up.get()) != 0) {
            message = "the model could not be evaluated: " + capture.message("llama_decode reported an error");
            return;
          }
          llama_memory_clear(llama_get_memory(context), true);
        }

        // ---- data set --------------------------------------------------------
        // llama.cpp reads one record of llama_n_ctx tokens per window and
        // evaluates its first `window` tokens.
        const auto record = static_cast<std::size_t>(llama_n_ctx(context));
        dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, static_cast<std::int64_t>(record),
                                        static_cast<std::int64_t>(record), windows, 1);
        auto* data = static_cast<llama_token*>(ggml_opt_dataset_data(dataset)->data);
        auto* labels = static_cast<llama_token*>(ggml_opt_dataset_labels(dataset)->data);
        std::fill(data, data + record * order.size(), 0);
        std::fill(labels, labels + record * order.size(), 0);
        const auto fill = [&] {
          for (std::size_t slot = 0; slot < order.size(); ++slot) {
            const std::size_t start = static_cast<std::size_t>(order[slot]) * stride;
            std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(start), window, data + slot * record);
            std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(start + 1), window, labels + slot * record);
          }
        };

        // ---- optimiser -------------------------------------------------------
        struct llama_opt_params opt_params;
        opt_params.n_ctx_train = static_cast<std::uint32_t>(options.window);
        opt_params.param_filter = select_parameter;
        opt_params.param_filter_ud = &trainer;
        opt_params.get_opt_pars = optimizer_parameters;
        opt_params.get_opt_pars_ud = &trainer;
        opt_params.optimizer_type = detail::to_lower(options.optimizer) == "sgd" ? GGML_OPT_OPTIMIZER_TYPE_SGD
                                                                                   : GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        llama_opt_init(context, model, opt_params);
        report.trainable_parameters = trainer.trainable_parameters;
        if (trainer.trainable_parameters == 0) {
          message = "no trainable parameters were selected (the model has " + std::to_string(blocks) + " blocks)";
          return;
        }
        PRAG_INFO("llm: fine-tuning %llu of %llu parameters (blocks %d..%d) on %lld windows of %d tokens, "
                  "%lld held out, %d threads",
                  static_cast<unsigned long long>(report.trainable_parameters),
                  static_cast<unsigned long long>(report.total_parameters), trainer.all_parameters ? 0 : trainer.first_block,
                  blocks - 1, static_cast<long long>(report.train_windows), options.window,
                  static_cast<long long>(report.validation_windows), threads);

        // ---- epochs ----------------------------------------------------------
        result_train = ggml_opt_result_init();
        result_validation = ggml_opt_result_init();
        t_trainer = &trainer;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
          trainer.epoch = epoch;
          trainer.done_in_epoch = 0;
          if (!trainer.report(static_cast<double>(epoch) / options.epochs,
                              "epoch " + std::to_string(epoch + 1) + " of " + std::to_string(options.epochs))) {
            trainer.cancel.store(true);
            break;
          }
          if (epoch > 0) shuffle(order, 0, static_cast<std::size_t>(report.train_windows), random);
          fill();
          ggml_opt_result_reset(result_train);
          ggml_opt_result_reset(result_validation);
          llama_opt_epoch(context, dataset, result_train, result_validation, report.train_windows, epoch_callback,
                          epoch_callback);
          if (trainer.cancel.load()) break;  // the results of an aborted epoch mean nothing

          double train_loss = 0.0;
          double validation_loss = 0.0;
          ggml_opt_result_loss(result_train, &train_loss, nullptr);
          if (report.validation_windows > 0) ggml_opt_result_loss(result_validation, &validation_loss, nullptr);
          if (!std::isfinite(train_loss) || !std::isfinite(validation_loss)) {
            message = "training diverged in epoch " + std::to_string(epoch + 1) +
                      ": the loss is not finite; lower the learning rate";
            break;
          }
          report.train_loss.push_back(train_loss);
          if (report.validation_windows > 0) report.validation_loss.push_back(validation_loss);
          report.epochs_run = epoch + 1;
          PRAG_INFO("llm: epoch %d of %d: training loss %.4f, validation loss %.4f", epoch + 1, options.epochs,
                    train_loss, report.validation_windows > 0 ? validation_loss : std::nan(""));
        }
        t_trainer = nullptr;
        if (trainer.cancel.load()) {
          report.cancelled = true;
          return;
        }
        if (!message.empty()) return;

        trainer.report(1.0, "writing " + detail::file_stem(options.output));
        llama_model_save_to_file(model, raw_path.c_str());
        saved = fs::file_size(raw_path) > 0;
        if (!saved) message = "llama.cpp could not write " + raw_path + ": " + capture.message("disk full?");
      },
      &failure);
  t_trainer = nullptr;

  if (!completed) {
    fs::remove_file(raw_path);
    // llama.cpp sizes its compute graph for inference. With many trainable
    // blocks the backward pass does not fit, and an assertion fails while the
    // first training graph is built: nothing has been computed or modified
    // yet, so in this one case the objects are intact and are released below.
    if (failure.find("n_nodes < cgraph->size") == std::string::npos) {
      // Model, context and data set are leaked on purpose, see detail::guarded.
      return finish("fine-tuning failed: " + failure);
    }
    const int trained = trainer.all_parameters ? blocks : blocks - trainer.first_block;
    message = "llama.cpp cannot build the training graph for " + std::to_string(trained) + " of the " +
              std::to_string(blocks) + " blocks of this model: the backward pass needs more graph nodes than "
              "llama.cpp reserves; lower trainable_blocks";
  }
  detail::guarded(
      [&] {
        if (result_train) ggml_opt_result_free(result_train);
        if (result_validation) ggml_opt_result_free(result_validation);
        if (dataset) ggml_opt_dataset_free(dataset);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
      },
      nullptr);

  if (report.cancelled) {
    fs::remove_file(raw_path);
    return finish("cancelled");
  }
  if (!message.empty()) {
    fs::remove_file(raw_path);
    return finish(std::move(message));
  }

  std::string error;
  bool written = false;
  const bool rewritten = detail::guarded(
      [&] { written = write_with_base_metadata(options.input, raw_path, partial_path, &error); }, &failure);
  fs::remove_file(raw_path);
  if (!rewritten || !written) {
    fs::remove_file(partial_path);
    return finish("could not write the trained model: " + (rewritten ? error : failure));
  }
  std::error_code code;
  std::filesystem::rename(to_path(partial_path), to_path(options.output), code);
  if (code) {
    fs::remove_file(options.output);
    code.clear();
    std::filesystem::rename(to_path(partial_path), to_path(options.output), code);
  }
  if (code) {
    fs::remove_file(partial_path);
    return finish("could not write " + options.output + ": " + code.message());
  }

  report.ok = true;
  report.seconds = clock.elapsed_s();
  PRAG_INFO("llm: fine-tuned model written to %s (%.1f s)", options.output.c_str(), report.seconds);
  return report;
}

}  // namespace prag::llm
