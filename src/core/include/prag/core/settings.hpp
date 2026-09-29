// Runtime settings: one JSON document addressed by dotted keys, e.g. "retrieve.tau".
// Every module reads its own keys and supplies its own defaults, so adding an
// option never requires touching this header. The documented keys are listed in
// docs/DESIGN.md.
#pragma once

#include <mutex>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace prag {

class Settings {
 public:
  Settings() = default;
  Settings(const Settings& other);
  Settings& operator=(const Settings& other);

  // Loads `path`; a missing file yields empty settings and returns true.
  // Returns false (and keeps the previous content) when the file is not valid JSON.
  bool load(std::string_view path);
  bool save(std::string_view path) const;

  // Value at `dotted_key`, or `fallback` when the key is absent or has the wrong type.
  template <class T>
  T get(std::string_view dotted_key, T fallback) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const nlohmann::json* node = find(dotted_key);
    if (!node || node->is_null()) return fallback;
    try {
      return node->get<T>();
    } catch (const nlohmann::json::exception&) {
      return fallback;
    }
  }

  template <class T>
  void set(std::string_view dotted_key, const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    slot(dotted_key) = value;
  }

  bool has(std::string_view dotted_key) const;
  nlohmann::json snapshot() const;
  // Deep-merges `patch` into the document (objects merge, everything else replaces).
  void merge(const nlohmann::json& patch);

 private:
  const nlohmann::json* find(std::string_view dotted_key) const;
  nlohmann::json& slot(std::string_view dotted_key);

  mutable std::mutex mutex_;
  nlohmann::json doc_ = nlohmann::json::object();
};

}  // namespace prag
