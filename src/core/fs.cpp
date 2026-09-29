#include "prag/core/fs.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace prag::fs {

namespace stdfs = std::filesystem;

namespace {

stdfs::path to_path(std::string_view s) { return stdfs::path(std::u8string(s.begin(), s.end())); }

std::string to_generic(const stdfs::path& p) {
  const auto u8 = p.generic_u8string();
  return std::string(u8.begin(), u8.end());
}

#ifdef _WIN32
constexpr bool kCaseInsensitivePaths = true;
#else
constexpr bool kCaseInsensitivePaths = false;
#endif

std::string comparable(std::string_view path) {
  std::string n = normalize(path);
  while (n.size() > 1 && n.back() == '/') n.pop_back();
  if (kCaseInsensitivePaths) n = text::to_lower(n);
  return n;
}

}  // namespace

std::string normalize(std::string_view path) {
  std::error_code ec;
  stdfs::path p = to_path(path);
  stdfs::path abs = stdfs::absolute(p, ec);
  if (ec) abs = p;
  return to_generic(abs.lexically_normal());
}

bool is_within(std::string_view path, std::string_view root) {
  const std::string p = comparable(path);
  const std::string r = comparable(root);
  if (r.empty()) return false;
  if (p == r) return true;
  if (p.size() <= r.size()) return false;
  if (p.compare(0, r.size(), r) != 0) return false;
  return r.back() == '/' || p[r.size()] == '/';
}

bool exists(std::string_view path) {
  std::error_code ec;
  return stdfs::exists(to_path(path), ec);
}

bool is_directory(std::string_view path) {
  std::error_code ec;
  return stdfs::is_directory(to_path(path), ec);
}

std::uint64_t file_size(std::string_view path) {
  std::error_code ec;
  const auto size = stdfs::file_size(to_path(path), ec);
  return ec ? 0 : static_cast<std::uint64_t>(size);
}

std::optional<std::string> read_file(std::string_view path) {
  std::ifstream in(to_path(path), std::ios::binary);
  if (!in) return std::nullopt;
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) return std::nullopt;
  in.seekg(0, std::ios::beg);
  std::string content(static_cast<std::size_t>(size), '\0');
  if (size > 0 && !in.read(content.data(), size)) return std::nullopt;
  return content;
}

bool write_file_atomic(std::string_view path, std::string_view content) {
  static std::atomic<std::uint64_t> counter{0};
  const stdfs::path target = to_path(path);
  std::error_code ec;
  if (target.has_parent_path()) stdfs::create_directories(target.parent_path(), ec);

  const std::uint64_t nonce = hash_combine(hash64(path), counter.fetch_add(1) + 1);
  stdfs::path temp = target;
  temp += ".tmp" + std::to_string(nonce % 1000000007ULL);

#ifdef _WIN32
  HANDLE file = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  bool ok = true;
  std::size_t written_total = 0;
  while (ok && written_total < content.size()) {
    const std::size_t remaining = content.size() - written_total;
    const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(remaining, 1u << 30));
    DWORD written = 0;
    ok = WriteFile(file, content.data() + written_total, chunk, &written, nullptr) != 0 && written == chunk;
    written_total += written;
  }
  ok = ok && FlushFileBuffers(file) != 0;
  CloseHandle(file);
  if (ok) ok = MoveFileExW(temp.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
  if (!ok) DeleteFileW(temp.c_str());
  return ok;
#else
  const std::string temp_native = temp.string();
  const int fd = ::open(temp_native.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  bool ok = true;
  std::size_t written_total = 0;
  while (ok && written_total < content.size()) {
    const ssize_t written = ::write(fd, content.data() + written_total, content.size() - written_total);
    if (written <= 0) {
      ok = false;
    } else {
      written_total += static_cast<std::size_t>(written);
    }
  }
  ok = ok && ::fsync(fd) == 0;
  ::close(fd);
  if (ok) ok = ::rename(temp_native.c_str(), target.string().c_str()) == 0;
  if (!ok) ::unlink(temp_native.c_str());
  return ok;
#endif
}

bool copy_file(std::string_view from, std::string_view to) {
  std::error_code ec;
  const stdfs::path target = to_path(to);
  if (target.has_parent_path()) stdfs::create_directories(target.parent_path(), ec);
  ec.clear();
  stdfs::copy_file(to_path(from), target, stdfs::copy_options::overwrite_existing, ec);
  return !ec;
}

bool remove_file(std::string_view path) {
  std::error_code ec;
  return stdfs::remove(to_path(path), ec) && !ec;
}

bool create_directories(std::string_view path) {
  std::error_code ec;
  stdfs::create_directories(to_path(path), ec);
  return !ec && stdfs::is_directory(to_path(path), ec);
}

std::vector<std::string> walk(std::string_view root, const WalkOptions& options) {
  std::vector<std::string> extensions;
  extensions.reserve(options.extensions.size());
  for (const std::string& e : options.extensions) extensions.push_back(text::to_lower(e));

  const auto accept = [&](const stdfs::path& p, std::uint64_t size) {
    if (options.max_file_bytes != 0 && size > options.max_file_bytes) return false;
    if (!extensions.empty()) {
      const std::string ext = text::to_lower(to_generic(p.extension()));
      if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) return false;
    }
    return true;
  };
  const auto excluded = [&](const std::string& generic) {
    for (const std::string& needle : options.exclude_substrings) {
      if (!needle.empty() && text::contains_icase(generic, needle)) return true;
    }
    return false;
  };

  std::vector<std::string> out;
  std::error_code ec;
  const stdfs::path start = to_path(normalize(root));
  if (stdfs::is_regular_file(start, ec)) {
    const std::string generic = to_generic(start);
    if (!excluded(generic) && accept(start, static_cast<std::uint64_t>(stdfs::file_size(start, ec)))) {
      out.push_back(generic);
    }
    return out;
  }
  if (!stdfs::is_directory(start, ec)) return out;

  stdfs::recursive_directory_iterator it(start, stdfs::directory_options::skip_permission_denied, ec);
  const stdfs::recursive_directory_iterator end;
  while (!ec && it != end) {
    const stdfs::directory_entry& entry = *it;
    const std::string generic = to_generic(entry.path());
    std::error_code entry_ec;
    if (entry.is_directory(entry_ec)) {
      // The trailing slash lets patterns such as "/build/" match the directory itself.
      if (excluded(generic + "/")) it.disable_recursion_pending();
    } else if (entry.is_regular_file(entry_ec)) {
      const std::uint64_t size = static_cast<std::uint64_t>(entry.file_size(entry_ec));
      if (!entry_ec && !excluded(generic) && accept(entry.path(), size)) out.push_back(generic);
    }
    it.increment(ec);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace prag::fs
