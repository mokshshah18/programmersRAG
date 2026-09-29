// Helpers shared by the tests of the edit module.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "prag/core/fs.hpp"
#include "prag/edit/edit.hpp"

namespace edit_test {

// Windows refuses to touch a file for a moment while a virus scanner or an
// indexer looks at it; fixtures are therefore written with a few attempts.
template <class Operation>
bool persistently(Operation&& operation) {
  for (const int delay_ms : {0, 1, 2, 5, 10, 20, 50, 100, 200}) {
    if (delay_ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    if (operation()) return true;
  }
  return false;
}

// Empty directory below the temporary directory of the system.
inline std::string temp_dir(const char* name) {
  const auto dir = std::filesystem::temp_directory_path() / "prag-edit-tests" / name;
  persistently([&] {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (ec) return false;
    std::filesystem::create_directories(dir, ec);
    return !ec;
  });
  return prag::fs::normalize(dir.string());
}

// Plain write without flushing to disk: test fixtures need no durability.
inline bool write_file(const std::string& path, std::string_view content) {
  std::error_code ec;
  const std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
  return persistently([&] {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
    return static_cast<bool>(out);
  });
}

inline std::string read_file(const std::string& path) {
  std::string content = "<unreadable>";
  persistently([&] {
    auto read = prag::fs::read_file(path);
    if (read) content = std::move(*read);
    return read.has_value() || !prag::fs::exists(path);
  });
  return content;
}

inline std::size_t count_entries(const std::string& directory) {
  std::size_t n = 0;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) ++n;
  return n;
}

inline prag::edit::EditBlock search_replace(std::string path, std::string search, std::string replace,
                                            std::string scope = {}) {
  prag::edit::EditBlock block;
  block.kind = prag::edit::EditKind::SearchReplace;
  block.path = std::move(path);
  block.scope = std::move(scope);
  block.search = std::move(search);
  block.replace = std::move(replace);
  return block;
}

inline prag::edit::EditBlock unit_rewrite(std::string path, std::string unit, std::string text) {
  prag::edit::EditBlock block;
  block.kind = prag::edit::EditKind::UnitRewrite;
  block.path = std::move(path);
  block.scope = std::move(unit);
  block.replace = std::move(text);
  return block;
}

inline prag::edit::ApplyOptions options_for(const std::string& root) {
  prag::edit::ApplyOptions options;
  options.allowed_roots = {root};
  options.backup_dir = root + "/.edits";
  return options;
}

// Deterministic generator (splitmix64); the same on every platform, unlike the
// distributions of the standard library.
class Random {
 public:
  explicit Random(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // Uniform in [0, n); n must be positive.
  std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % static_cast<std::uint64_t>(n)); }
  // Uniform in [low, high].
  std::size_t between(std::size_t low, std::size_t high) { return low + below(high - low + 1); }
  bool chance(double p) { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0) < p; }

 private:
  std::uint64_t state_;
};

}  // namespace edit_test
