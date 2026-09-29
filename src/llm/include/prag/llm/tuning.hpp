// Fine-tuning, importance-matrix calibration and quantisation primitives.
// All of them run in-process on top of llama.cpp and report progress through
// a callback that can also cancel the operation.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace prag::llm {

// fraction in [0, 1]; return false to cancel at the next safe point.
using ProgressCallback = std::function<bool(double fraction, std::string_view message)>;

// ---- precision conversion and quantisation ---------------------------------

struct ImportanceMatrix {
  // For every weight matrix W (named as in the GGUF), the mean squared
  // activation of each input column over the calibration text: E[x_j^2].
  std::vector<std::string> names;
  std::vector<std::vector<float>> values;
  std::uint64_t tokens = 0;  // calibration tokens seen
  bool empty() const { return names.empty(); }
};

struct QuantizeOptions {
  std::string input;          // GGUF
  std::string output;         // GGUF
  std::string type = "Q4_K_M";  // "F32", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", ...
  int threads = 0;
  bool allow_requantize = true;
  const ImportanceMatrix* importance = nullptr;
};

bool quantize(const QuantizeOptions& options, std::string* error = nullptr);

// Collects E[x_j^2] at the input of every matrix multiplication while the model
// reads `calibration_text` in windows of `window` tokens.
struct ImportanceOptions {
  std::string model;   // GGUF, any precision
  int window = 512;
  int max_windows = 64;
  int threads = 0;
  bool parse_special = true;  // chat-template markers in the text become special tokens
};

bool compute_importance(const ImportanceOptions& options, std::string_view calibration_text,
                        ImportanceMatrix* out, const ProgressCallback& progress = {}, std::string* error = nullptr);

// ---- fine-tuning -----------------------------------------------------------

struct FineTuneOptions {
  std::string input;    // GGUF with float32 weights; see quantize(type = "F32")
  std::string output;   // GGUF with float32 weights and the metadata of the input
  int window = 256;     // training context length in tokens, at least 8
  int epochs = 2;
  double validation_fraction = 0.1;
  // The rate falls from learning_rate to min_learning_rate along a half cosine
  // over all optimiser steps (one step per window). A min_learning_rate of 0
  // or less keeps the rate constant.
  double learning_rate = 1e-5;
  double min_learning_rate = 1e-6;
  double weight_decay = 0.0;
  // Only the last `trainable_blocks` transformer blocks are updated (0 = all
  // parameters). Freezing the lower blocks bounds both memory and the drift
  // from the base model.
  //
  // llama.cpp b11257 sizes its compute graph for inference: the backward pass
  // fits for up to about three quarters of the blocks (18 of the 24 blocks of
  // Qwen2.5-Coder-0.5B). Beyond that, and for 0, fine_tune fails with an error
  // that says so. It also computes no gradient for the key and value
  // projections of attention (attn_k, attn_v), which stay as they are; all
  // other tensors of the selected blocks are updated.
  int trainable_blocks = 2;
  // Not supported by llama.cpp b11257, which excludes the token embeddings
  // from training; fine_tune fails with an error when this is set.
  bool train_embeddings = false;
  int threads = 0;
  std::string optimizer = "adamw";  // "adamw" or "sgd"
  int stride = 0;             // tokens between the starts of two windows; 0 = window (no overlap)
  bool parse_special = true;  // chat-template markers in the text become special tokens
  std::uint32_t seed = 1;     // assignment of windows to training and validation, and their order
};

struct FineTuneReport {
  bool ok = false;
  bool cancelled = false;
  int epochs_run = 0;
  std::int64_t train_windows = 0;
  std::int64_t validation_windows = 0;
  std::vector<double> train_loss;       // per epoch, mean cross-entropy in nats per token
  std::vector<double> validation_loss;  // per epoch
  std::uint64_t trainable_parameters = 0;  // parameters the optimiser updates
  std::uint64_t total_parameters = 0;      // parameters of the model
  double seconds = 0.0;
  std::string error;
};

FineTuneReport fine_tune(const FineTuneOptions& options, std::string_view training_text,
                         const ProgressCallback& progress = {});

}  // namespace prag::llm
