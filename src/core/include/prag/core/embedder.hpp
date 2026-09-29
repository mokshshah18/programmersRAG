// Text embedding interface shared by every embedder.
//
// Every embedder returns L2-normalised vectors, so the inner product of two
// embeddings is their cosine similarity. Implementations: the hashing embedder
// (embed module) and the llama.cpp embedder (llm module).
#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace prag::embed {

enum class Role { Query, Document };

class Embedder {
 public:
  virtual ~Embedder() = default;

  virtual std::size_t dim() const = 0;
  // Identifies the model and every setting that changes its output. Stored in
  // the index; a mismatch forces re-embedding.
  virtual std::string id() const = 0;
  // Longest input, in model tokens, that is embedded without truncation.
  virtual std::size_t max_tokens() const = 0;

  // Writes texts.size() * dim() floats to `out`, one L2-normalised row per text.
  // Callers serialise access: one call at a time per embedder.
  virtual void embed(std::span<const std::string_view> texts, Role role, float* out) = 0;

  std::vector<float> embed_one(std::string_view text, Role role) {
    std::vector<float> out(dim());
    const std::string_view texts[1] = {text};
    embed(texts, role, out.data());
    return out;
  }
};

// In-place L2 normalisation; a zero vector is left unchanged. Returns the original norm.
inline float normalize(float* x, std::size_t dim) {
  double sum = 0.0;
  for (std::size_t i = 0; i < dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  const double norm = std::sqrt(sum);
  if (norm > 0.0) {
    const float inverse = static_cast<float>(1.0 / norm);
    for (std::size_t i = 0; i < dim; ++i) x[i] *= inverse;
  }
  return static_cast<float>(norm);
}

}  // namespace prag::embed
