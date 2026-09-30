// Stand-in for prag_tune while that module has no implementation. Active only
// when CMake defines PRAG_ENGINE_TUNE_STUB (prag_tune is header-only); include
// it in exactly one translation unit per test executable.
#pragma once
#if defined(PRAG_ENGINE_TUNE_STUB)
#include "prag/tune/pipeline.hpp"
namespace prag::tune {
struct Pipeline::Impl {};
Pipeline::Pipeline(const Settings&, std::string, Hooks) {}
Pipeline::~Pipeline() = default;
void Pipeline::start() {}
void Pipeline::stop() {}
void Pipeline::set_enabled(bool) {}
bool Pipeline::enabled() const { return false; }
void Pipeline::notify_activity() {}
bool Pipeline::run_now(std::string* error) {
  if (error) *error = "tuning is not available in this build";
  return false;
}
Status Pipeline::status() const { return {}; }
const char* to_string(Stage) { return "disabled"; }
nlohmann::json to_json(const Status&) {
  return nlohmann::json{{"available", false}, {"enabled", false}, {"stage", "disabled"}, {"progress", 0.0},
                        {"message", ""}, {"runs", nlohmann::json::array()}};
}
}  // namespace prag::tune
#endif
