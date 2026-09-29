// TEMPORARY: stands in for prag_tune until that module is implemented. Remove.
#pragma once
#include "prag/tune/pipeline.hpp"
namespace prag::tune {
struct Pipeline::Impl {};
Pipeline::Pipeline(const Settings&, std::string, Hooks) {}
Pipeline::~Pipeline() = default;
void Pipeline::start() {}
void Pipeline::stop() {}
void Pipeline::notify_activity() {}
Status Pipeline::status() const { return {}; }
nlohmann::json to_json(const Status&) { return nlohmann::json::object(); }
}  // namespace prag::tune
