// Built instead of llama/*.cpp when the project is configured with
// PRAG_WITH_LLAMA=OFF: every loader and operation fails with the same message.
#include "prag/llm/model.hpp"
#include "prag/llm/tuning.hpp"

namespace prag::llm {

namespace {

constexpr const char* kUnavailable =
    "this build has no language model runtime: configure with -DPRAG_WITH_LLAMA=ON to link llama.cpp";

void explain(std::string* error) {
  if (error) *error = kUnavailable;
}

}  // namespace

bool available() { return false; }

std::string backend_info() { return "llama.cpp not available (built with PRAG_WITH_LLAMA=OFF)"; }

std::unique_ptr<LanguageModel> load_model(const ModelOptions&, std::string* error) {
  explain(error);
  return nullptr;
}

std::unique_ptr<embed::Embedder> load_embedder(const EmbedderOptions&, std::string* error) {
  explain(error);
  return nullptr;
}

bool quantize(const QuantizeOptions&, std::string* error) {
  explain(error);
  return false;
}

bool compute_importance(const ImportanceOptions&, std::string_view, ImportanceMatrix* out, const ProgressCallback&,
                        std::string* error) {
  if (out) *out = ImportanceMatrix{};
  explain(error);
  return false;
}

FineTuneReport fine_tune(const FineTuneOptions&, std::string_view, const ProgressCallback&) {
  FineTuneReport report;
  report.error = kUnavailable;
  return report;
}

}  // namespace prag::llm
