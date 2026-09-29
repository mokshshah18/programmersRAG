#include "prag/core/settings.hpp"

#include "prag/core/fs.hpp"

namespace prag {

namespace {

// Splits "a.b.c" into successive keys.
template <class Visit>
void for_each_key(std::string_view dotted_key, Visit&& visit) {
  std::size_t begin = 0;
  while (begin <= dotted_key.size()) {
    std::size_t end = dotted_key.find('.', begin);
    if (end == std::string_view::npos) end = dotted_key.size();
    visit(std::string(dotted_key.substr(begin, end - begin)));
    if (end == dotted_key.size()) break;
    begin = end + 1;
  }
}

void deep_merge(nlohmann::json& target, const nlohmann::json& patch) {
  if (!patch.is_object() || !target.is_object()) {
    target = patch;
    return;
  }
  for (auto it = patch.begin(); it != patch.end(); ++it) {
    if (it->is_object() && target.contains(it.key()) && target[it.key()].is_object()) {
      deep_merge(target[it.key()], *it);
    } else {
      target[it.key()] = *it;
    }
  }
}

}  // namespace

Settings::Settings(const Settings& other) {
  std::lock_guard<std::mutex> lock(other.mutex_);
  doc_ = other.doc_;
}

Settings& Settings::operator=(const Settings& other) {
  if (this == &other) return *this;
  nlohmann::json copy;
  {
    std::lock_guard<std::mutex> lock(other.mutex_);
    copy = other.doc_;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  doc_ = std::move(copy);
  return *this;
}

bool Settings::load(std::string_view path) {
  if (!fs::exists(path)) {
    std::lock_guard<std::mutex> lock(mutex_);
    doc_ = nlohmann::json::object();
    return true;
  }
  const auto content = fs::read_file(path);
  if (!content) return false;
  nlohmann::json parsed = nlohmann::json::parse(*content, nullptr, /*allow_exceptions=*/false, /*ignore_comments=*/true);
  if (parsed.is_discarded() || !parsed.is_object()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  doc_ = std::move(parsed);
  return true;
}

bool Settings::save(std::string_view path) const {
  std::string serialised;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    serialised = doc_.dump(2);
  }
  serialised.push_back('\n');
  return fs::write_file_atomic(path, serialised);
}

bool Settings::has(std::string_view dotted_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const nlohmann::json* node = find(dotted_key);
  return node != nullptr && !node->is_null();
}

nlohmann::json Settings::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return doc_;
}

void Settings::merge(const nlohmann::json& patch) {
  std::lock_guard<std::mutex> lock(mutex_);
  deep_merge(doc_, patch);
}

const nlohmann::json* Settings::find(std::string_view dotted_key) const {
  const nlohmann::json* node = &doc_;
  bool missing = false;
  for_each_key(dotted_key, [&](const std::string& key) {
    if (missing) return;
    if (!node->is_object()) {
      missing = true;
      return;
    }
    const auto it = node->find(key);
    if (it == node->end()) {
      missing = true;
      return;
    }
    node = &*it;
  });
  return missing ? nullptr : node;
}

nlohmann::json& Settings::slot(std::string_view dotted_key) {
  nlohmann::json* node = &doc_;
  for_each_key(dotted_key, [&](const std::string& key) {
    if (!node->is_object()) *node = nlohmann::json::object();
    node = &(*node)[key];
  });
  return *node;
}

}  // namespace prag
