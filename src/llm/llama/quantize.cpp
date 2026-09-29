// Precision conversion and quantisation through llama_model_quantize.
#include <cmath>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <vector>

#include "detail/runtime.hpp"
#include "detail/util.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"
#include "prag/llm/tuning.hpp"

namespace prag::llm {

namespace {

std::filesystem::path to_path(const std::string& utf8) {
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

bool same_file(const std::string& a, const std::string& b) {
  std::error_code code;
  if (std::filesystem::equivalent(to_path(a), to_path(b), code)) return true;
  return fs::normalize(a) == fs::normalize(b);
}

}  // namespace

bool quantize(const QuantizeOptions& options, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (error) error->clear();
  if (options.input.empty()) return fail("no input model given");
  if (options.output.empty()) return fail("no output file given");
  if (!fs::exists(options.input)) return fail("input model not found: " + options.input);
  if (same_file(options.input, options.output)) return fail("input and output are the same file: " + options.input);

  llama_ftype ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
  if (!detail::parse_ftype(options.type, &ftype)) {
    return fail("unknown quantisation type \"" + options.type + "\"; known types: " + detail::known_ftypes());
  }
  const bool is_float =
      ftype == LLAMA_FTYPE_ALL_F32 || ftype == LLAMA_FTYPE_MOSTLY_F16 || ftype == LLAMA_FTYPE_MOSTLY_BF16;

  // The importance matrix as llama.cpp wants it: an array ended by a null name.
  std::vector<llama_model_imatrix_data> importance;
  if (options.importance && !options.importance->empty() && !is_float) {
    const ImportanceMatrix& matrix = *options.importance;
    if (matrix.names.size() != matrix.values.size()) {
      return fail("the importance matrix has " + std::to_string(matrix.names.size()) + " names for " +
                  std::to_string(matrix.values.size()) + " value rows");
    }
    importance.reserve(matrix.names.size() + 1);
    for (std::size_t i = 0; i < matrix.names.size(); ++i) {
      if (matrix.names[i].empty() || matrix.values[i].empty()) continue;
      for (const float value : matrix.values[i]) {
        if (!std::isfinite(value) || value < 0.0f) {
          return fail("the importance matrix holds an invalid value for " + matrix.names[i]);
        }
      }
      importance.push_back({matrix.names[i].c_str(), matrix.values[i].data(), matrix.values[i].size()});
    }
    if (!importance.empty()) importance.push_back({nullptr, nullptr, 0});
  }

  detail::ensure_backend();
  std::lock_guard<std::mutex> heavy(detail::heavy_operation_mutex());

  const std::string parent = to_path(options.output).parent_path().generic_string();
  if (!parent.empty() && !fs::create_directories(parent)) return fail("could not create the directory " + parent);

  // Written next to the target and renamed at the end, so that a reader never
  // sees a partial model under the final name.
  const std::string partial = options.output + ".partial";
  fs::remove_file(partial);

  llama_model_quantize_params params = llama_model_quantize_default_params();
  params.nthread = detail::resolve_threads(options.threads);
  params.ftype = ftype;
  params.allow_requantize = options.allow_requantize;
  params.imatrix = importance.empty() ? nullptr : importance.data();

  const Stopwatch clock;
  std::string failure;
  std::string message;
  const bool completed = detail::guarded(
      [&] {
        detail::ErrorCapture capture;
        if (llama_model_quantize(options.input.c_str(), partial.c_str(), &params) != 0) {
          message = capture.message("llama_model_quantize reported an error");
        }
      },
      &failure);
  if (!completed) message = failure;
  if (!message.empty()) {
    fs::remove_file(partial);
    return fail("converting " + options.input + " to " + options.type + " failed: " + message);
  }

  std::error_code code;
  std::filesystem::rename(to_path(partial), to_path(options.output), code);
  if (code) {
    // rename does not replace an existing file on every platform
    fs::remove_file(options.output);
    code.clear();
    std::filesystem::rename(to_path(partial), to_path(options.output), code);
  }
  if (code) {
    fs::remove_file(partial);
    return fail("could not write " + options.output + ": " + code.message());
  }

  PRAG_INFO("llm: wrote %s as %s (%.1f MiB -> %.1f MiB, %.1f s%s)", options.output.c_str(),
            detail::ftype_name(ftype), static_cast<double>(fs::file_size(options.input)) / 1048576.0,
            static_cast<double>(fs::file_size(options.output)) / 1048576.0, clock.elapsed_s(),
            importance.empty() ? "" : ", with importance matrix");
  return true;
}

}  // namespace prag::llm
