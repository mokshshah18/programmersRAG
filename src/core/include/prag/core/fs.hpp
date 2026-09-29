// File-system helpers. All paths returned by this module use forward slashes.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace prag::fs {

// Absolute, lexically normalised, forward-slash form of `path`.
std::string normalize(std::string_view path);

// True when `path` is `root` itself or lies inside it (both are normalised first;
// the comparison is case-insensitive on Windows).
bool is_within(std::string_view path, std::string_view root);

bool exists(std::string_view path);
bool is_directory(std::string_view path);
std::uint64_t file_size(std::string_view path);  // 0 when the file does not exist

// Whole-file read in binary mode; nullopt on failure.
std::optional<std::string> read_file(std::string_view path);

// Writes to a temporary file in the same directory, flushes it to disk, then
// renames it over `path`, so readers observe either the old or the new content.
// Returns false and leaves `path` untouched on failure.
bool write_file_atomic(std::string_view path, std::string_view content);

bool copy_file(std::string_view from, std::string_view to);
bool remove_file(std::string_view path);
bool create_directories(std::string_view path);

struct WalkOptions {
  std::vector<std::string> extensions;          // lower-case, leading dot; empty = every file
  std::vector<std::string> exclude_substrings;  // matched case-insensitively against the normalised path
  std::uint64_t max_file_bytes = 0;             // 0 = no limit
};

// Recursively lists regular files under `root` (or `root` itself when it is a
// file) that pass the filters. The result is sorted, so it is deterministic.
std::vector<std::string> walk(std::string_view root, const WalkOptions& options);

}  // namespace prag::fs
