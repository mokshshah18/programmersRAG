// Resolution of paths written by a model against the directories that are open
// for edits.
#include <algorithm>
#include <filesystem>
#include <system_error>

#include "internal.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/text.hpp"

namespace prag::edit::detail {

namespace stdfs = std::filesystem;

namespace {

#ifdef _WIN32
constexpr bool kWindows = true;
#else
constexpr bool kWindows = false;
#endif

stdfs::path to_path(std::string_view s) { return stdfs::path(std::u8string(s.begin(), s.end())); }

std::string to_generic(const stdfs::path& p) {
  const auto u8 = p.generic_u8string();
  return std::string(u8.begin(), u8.end());
}

// "C:/dir/" -> "C:/dir"; "C:/" and "/" stay as they are.
std::string without_trailing_slash(std::string path) {
  while (path.size() > 1 && path.back() == '/' && path[path.size() - 2] != ':') path.pop_back();
  return path;
}

// Canonical form of the absolute path `path`: links are followed and names are
// spelled as on disk as far as the path exists; what does not exist cannot be a
// link and is kept as written. std::filesystem::weakly_canonical gives the same
// result, but probes the path from its root downwards with one system call for
// every directory; here the search goes upwards from the file, which takes one
// call for a file that exists and two for a new file in a directory that does.
// False when the path cannot be examined.
bool canonical_form(const stdfs::path& path, stdfs::path* canonical, bool* exists) {
  std::error_code ec;
  stdfs::path head = path;
  std::vector<stdfs::path> missing;  // names below `head`, innermost first
  for (;;) {
    stdfs::path resolved = stdfs::canonical(head, ec);
    if (!ec) {
      for (auto name = missing.rbegin(); name != missing.rend(); ++name) resolved /= *name;
      *canonical = std::move(resolved);
      *exists = missing.empty();
      return true;
    }
    // Only what does not exist may be skipped; anything else (no permission, a
    // device that does not answer) leaves the question of containment open.
    const stdfs::file_status status = stdfs::status(head, ec);
    if (status.type() != stdfs::file_type::not_found) return false;
    const stdfs::path parent = head.parent_path();
    if (parent.empty() || parent == head || !head.has_filename()) return false;
    missing.push_back(head.filename());
    head = parent;
  }
}

struct Candidate {
  std::string path;
  std::string root;
  std::string display;
  std::string key;
  bool exists = false;
};

// Places `written` below `root`. False when it does not lie inside the root,
// lexically or after links have been followed.
bool place(std::string_view written, bool anchored, std::string_view root, Candidate* out) {
  const std::string root_n = without_trailing_slash(fs::normalize(root));
  if (root_n.empty()) return false;

  std::string lexical;
  if (anchored) {
    lexical = fs::normalize(written);
  } else {
    // A root that is a single file stands for itself; names are relative to its directory.
    std::string base = root_n;
    if (!fs::is_directory(root_n)) {
      const std::size_t slash = root_n.rfind('/');
      if (slash == std::string::npos) return false;
      base = root_n.substr(0, slash);
    }
    lexical = fs::normalize(base + "/" + std::string(written));
  }
  lexical = without_trailing_slash(lexical);
  if (!fs::is_within(lexical, root_n)) return false;

  // The same test on canonical paths: a link inside the root must not lead out of it.
  stdfs::path root_canonical;
  stdfs::path file_canonical;
  bool root_exists = false;
  bool file_exists = false;
  if (!canonical_form(to_path(root_n), &root_canonical, &root_exists)) return false;
  if (!canonical_form(to_path(lexical), &file_canonical, &file_exists)) return false;
  const std::string root_c = without_trailing_slash(to_generic(root_canonical));
  const std::string file_c = without_trailing_slash(to_generic(file_canonical));
  if (!fs::is_within(file_c, root_c)) return false;

  // Below the root the canonical spelling is used, which is the spelling on
  // disk; the root keeps the spelling it was configured with.
  std::string relative = file_c.size() > root_c.size() ? file_c.substr(root_c.size()) : std::string();
  while (!relative.empty() && relative.front() == '/') relative.erase(relative.begin());

  out->root = root_n;
  if (relative.empty()) {
    out->path = root_n;
    const std::size_t slash = root_n.rfind('/');
    out->display = slash == std::string::npos ? root_n : root_n.substr(slash + 1);
  } else {
    out->path = root_n.back() == '/' ? root_n + relative : root_n + "/" + relative;
    out->display = relative;
  }
  out->key = kWindows ? text::to_lower(file_c) : file_c;
  out->exists = file_exists;
  return true;
}

bool is_sentence_mark(char c) { return c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?'; }

}  // namespace

ResolvedPath resolve_path(std::string_view written, const std::vector<std::string>& roots) {
  ResolvedPath out;
  std::string raw(text::trim(written));
  std::replace(raw.begin(), raw.end(), '\\', '/');
  out.path = raw;
  out.display = raw;
  if (raw.empty()) {
    out.status = Status::FileMissing;
    out.message = "the block names no file";
    return out;
  }
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const bool control = static_cast<unsigned char>(raw[i]) < 0x20;
    // On Windows a colon after the drive letter would address a stream of a file.
    const bool stream = kWindows && raw[i] == ':' && i != 1;
    if (control || stream) {
      out.message = "the path contains characters that a file name cannot have";
      return out;
    }
  }
  if (roots.empty()) {
    out.message = "no directory is open for edits";
    return out;
  }

  // The path as written; and, when it contains blanks and names nothing that
  // exists, its last word ("the change goes into src/alloc.c").
  std::vector<std::string> spellings{raw};
  const std::size_t blank = raw.find_last_of(" \t");
  if (blank != std::string::npos) {
    std::string word = raw.substr(blank + 1);
    while (!word.empty() && is_sentence_mark(word.back())) word.pop_back();
    if (!word.empty()) spellings.push_back(std::move(word));
  }

  Candidate fresh;
  bool have_fresh = false;
  for (std::size_t s = 0; s < spellings.size(); ++s) {
    const stdfs::path as_path = to_path(spellings[s]);
    const bool anchored = as_path.has_root_directory() || as_path.has_root_name();
    std::vector<Candidate> existing;
    for (const std::string& root : roots) {
      Candidate candidate;
      if (!place(spellings[s], anchored, root, &candidate)) continue;
      if (candidate.exists) {
        const bool known = std::any_of(existing.begin(), existing.end(),
                                       [&](const Candidate& c) { return c.key == candidate.key; });
        if (!known) existing.push_back(std::move(candidate));
      } else if (s == 0 && !have_fresh) {
        fresh = std::move(candidate);
        have_fresh = true;
      }
    }
    if (existing.size() > 1) {
      out.status = Status::Ambiguous;
      out.message = "the path names a file under several roots: " + existing[0].path + " and " + existing[1].path;
      return out;
    }
    if (existing.size() == 1) {
      fresh = std::move(existing[0]);
      have_fresh = true;
      break;
    }
  }
  if (!have_fresh) {
    out.message = "the path lies outside every directory that is open for edits";
    return out;
  }
  out.status = Status::Ok;
  out.path = std::move(fresh.path);
  out.root = std::move(fresh.root);
  out.display = std::move(fresh.display);
  out.key = std::move(fresh.key);
  return out;
}

}  // namespace prag::edit::detail
