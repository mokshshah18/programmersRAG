// Transactions: computing the new content of every file (prepare), writing it
// with a journal (commit), undoing it (revert) and rolling back what a crash
// left behind (recover).
//
// Layout of the backup directory:
//
//   <backup_dir>/<id>.journal     JSON, present while the transaction is in flight
//   <backup_dir>/<id>/<n>.bak     pre-edit bytes of file n of the transaction
//
// commit() writes the backups, then the journal, then replaces the targets one
// by one, then removes the journal. Every write is atomic (write to a temporary
// file, flush, rename), so at any moment a target holds either its old or its
// new content, and the journal is either complete or absent:
//
//   no journal                 nothing was replaced, or everything was and the
//                              transaction is committed
//   journal                    some targets may be replaced; recover() puts the
//                              old content back from the backups
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>

#include <nlohmann/json.hpp>

#include "internal.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"

namespace prag::edit {

using detail::Line;

namespace detail {

namespace {

std::atomic<Fault> g_fault{Fault::None};

struct SourceSlot {
  std::mutex mutex;
  SnapshotSource source;
  std::atomic<bool> set{false};
};

SourceSlot& source_slot() {
  static SourceSlot slot;
  return slot;
}

// True when a snapshot source is set and knows `path`.
bool from_snapshot_source(const std::string& path, std::string* text) {
  SourceSlot& slot = source_slot();
  if (!slot.set.load(std::memory_order_acquire)) return false;
  const std::lock_guard<std::mutex> lock(slot.mutex);
  return slot.source && slot.source(path, text);
}

}  // namespace

void set_fault(Fault fault) noexcept { g_fault.store(fault); }
Fault fault() noexcept { return g_fault.load(); }

void set_snapshot_source(SnapshotSource source) {
  SourceSlot& slot = source_slot();
  const std::lock_guard<std::mutex> lock(slot.mutex);
  slot.source = std::move(source);
  slot.set.store(static_cast<bool>(slot.source), std::memory_order_release);
}

std::string journal_name(std::string_view id) { return std::string(id) + ".journal"; }

}  // namespace detail

namespace {

constexpr std::size_t kNoPosition = static_cast<std::size_t>(-1);

std::string hex64(std::uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(value));
  return buffer;
}

bool parse_hex64(std::string_view s, std::uint64_t* value) {
  if (s.empty() || s.size() > 16) return false;
  std::uint64_t v = 0;
  for (const char c : s) {
    v <<= 4;
    if (c >= '0' && c <= '9') {
      v |= static_cast<std::uint64_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      v |= static_cast<std::uint64_t>(c - 'a' + 10);
    } else {
      return false;
    }
  }
  *value = v;
  return true;
}

// Identifiers become file names, so nothing but "tx-" and hexadecimal digits is accepted.
bool valid_id(std::string_view id) {
  if (id.size() < 4 || id.size() > 35 || id.compare(0, 3, "tx-") != 0) return false;
  return std::all_of(id.begin() + 3, id.end(),
                     [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::string in_quotes(std::string_view s) { return "'" + std::string(s) + "'"; }

// ---- file operations ----------------------------------------------------------------
//
// Windows does not replace, remove or open a file while another process holds
// it without sharing; virus scanners and indexers do that for a moment after
// every write. Such a failure passes, so an operation that failed is tried
// again a few times before it is reported. Elsewhere failures do not pass by
// themselves and are reported at once.

template <class Operation>
bool persistently(const char* what, const std::string& path, Operation&& operation) {
  if (operation()) return true;
#ifdef _WIN32
  for (const int delay_ms : {1, 2, 5, 10, 20, 50, 100, 200}) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    if (operation()) {
      PRAG_DEBUG("edit: %s %s succeeded after waiting", what, path.c_str());
      return true;
    }
  }
#endif
  PRAG_DEBUG("edit: %s %s failed", what, path.c_str());
  return false;
}

bool write_file(const std::string& path, std::string_view content) {
  return persistently("writing", path, [&] { return fs::write_file_atomic(path, content); });
}

bool remove_file(const std::string& path) {
  return persistently("removing", path, [&] { return !fs::exists(path) || fs::remove_file(path); });
}

// nullopt when the file does not exist or cannot be read.
std::optional<std::string> read_file(const std::string& path) {
  std::optional<std::string> content;
  persistently("reading", path, [&] {
    content = fs::read_file(path);
    return content.has_value() || !fs::exists(path);
  });
  return content;
}

void remove_tree(const std::string& directory) {
  const std::filesystem::path path(std::u8string(directory.begin(), directory.end()));
  persistently("removing", directory, [&] {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    return !ec;
  });
}

// ---- prepare -------------------------------------------------------------------

// One replacement of the bytes [begin, end) of the snapshot.
struct Splice {
  std::size_t begin = 0;
  std::size_t end = 0;
  std::string text;
  bool deletes_lines = false;  // whole lines are removed and nothing is put in their place
};

struct Resolution {
  Status status = Status::Ok;
  Match match;
  Splice splice;
  std::string message;
};

// Lines joined with `inner`; the last line ends in `last`, which is the
// terminator of the last line that is replaced (empty at the end of a file
// without final newline).
std::string join(const std::vector<std::string>& lines, std::string_view inner, std::string_view last) {
  std::string out;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    out.append(lines[i]);
    out.append(i + 1 < lines.size() ? inner : last);
  }
  return out;
}

std::vector<std::string> owned_lines(std::string_view text) {
  std::vector<std::string> out;
  for (const std::string_view line : detail::line_contents(text)) out.emplace_back(line);
  return out;
}

std::uint32_t line_number_at(std::string_view text, std::size_t offset) {
  const auto end = text.begin() + static_cast<std::ptrdiff_t>(std::min(offset, text.size()));
  return static_cast<std::uint32_t>(std::count(text.begin(), end, '\n') + 1);
}

Resolution failed(Status status, std::string message, Match match = {}) {
  Resolution r;
  r.status = status;
  r.match = std::move(match);
  r.match.status = status;
  r.message = std::move(message);
  return r;
}

Resolution rewrite_unit(const EditBlock& block, const FileProposal& proposal, std::size_t unit_begin,
                        std::size_t unit_end, char style) {
  const std::string_view original = proposal.original;
  const std::string_view unit = original.substr(unit_begin, unit_end - unit_begin);
  const std::vector<Line> lines = detail::split_lines(unit);

  // Blank lines around the unit separate it from its neighbours and stay in place.
  std::size_t first = 0;
  while (first < lines.size() && detail::is_blank_line(detail::content_of(unit, lines[first]))) ++first;
  std::size_t last = lines.size();
  while (last > first && detail::is_blank_line(detail::content_of(unit, lines[last - 1]))) --last;

  std::vector<std::string> fresh = owned_lines(block.replace);
  while (!fresh.empty() && detail::is_blank_line(fresh.back())) fresh.pop_back();
  std::size_t leading = 0;
  while (leading < fresh.size() && detail::is_blank_line(fresh[leading])) ++leading;
  fresh.erase(fresh.begin(), fresh.begin() + static_cast<std::ptrdiff_t>(leading));
  if (fresh.empty()) {
    return failed(Status::NotFound, "the new text of unit " + in_quotes(block.scope) + " is empty");
  }

  Resolution r;
  r.match.kind = MatchKind::Unit;
  r.match.status = Status::Ok;
  r.match.candidates = 1;
  std::string_view terminator;
  if (first == last) {
    // The unit holds nothing but blank lines; the whole range is replaced.
    r.splice.begin = unit_begin;
    r.splice.end = unit_end;
    terminator = unit.empty() || unit.back() != '\n' ? std::string_view() : std::string_view(proposal.line_ending);
  } else {
    r.splice.begin = unit_begin + lines[first].begin;
    r.splice.end = unit_begin + lines[last - 1].end;
    terminator = detail::terminator_of(unit, lines[last - 1]);

    // A unit that is nested in the file is often written without its base indentation.
    const std::string_view old_first = detail::content_of(unit, lines[first]);
    if (detail::indentation_of(old_first) != detail::indentation_of(fresh.front())) {
      r.match.indent_from = std::string(detail::indentation_of(fresh.front()));
      r.match.indent_to = std::string(detail::indentation_of(old_first));
      std::string why;
      const std::vector<std::string_view> from{std::string_view(fresh.front())};
      const std::vector<std::string_view> to{old_first};
      if (!detail::reindent_replacement(from, to, &fresh, &why)) {
        return failed(Status::NotFound, "unit " + in_quotes(block.scope) + ": " + why, r.match);
      }
    }
  }
  r.splice.text = join(fresh, proposal.line_ending, terminator);
  r.match.begin = r.splice.begin;
  r.match.end = r.splice.end;
  r.match.first_line = line_number_at(original, r.splice.begin);
  r.match.last_line = r.match.first_line + static_cast<std::uint32_t>(last > first ? last - first - 1 : 0);

  std::vector<std::string_view> replaced;
  for (std::size_t i = first; i < last; ++i) replaced.push_back(detail::content_of(unit, lines[i]));
  std::string why;
  if (!detail::keeps_indentation_style(replaced, fresh, style, &why)) {
    return failed(Status::NotFound, "unit " + in_quotes(block.scope) + ": " + why, r.match);
  }
  return r;
}

// `style` is the indentation character of the file, 0 when it is not to be guarded.
Resolution resolve_block(const EditBlock& block, const FileProposal& proposal, const std::vector<Line>& lines,
                         const ApplyOptions& options, char style) {
  const std::string_view original = proposal.original;
  std::size_t scope_begin = 0;
  std::size_t scope_end = kNoPosition;
  if (block.kind == EditKind::UnitRewrite && block.scope.empty()) {
    return failed(Status::ScopeNotFound, "a unit rewrite must name the unit");
  }
  if (!block.scope.empty()) {
    const bool found = options.unit_locator &&
                       options.unit_locator(proposal.path, original, block.scope, &scope_begin, &scope_end) &&
                       scope_begin <= scope_end && scope_end <= original.size();
    if (!found) return failed(Status::ScopeNotFound, "the file has no unit called " + in_quotes(block.scope));
  }
  if (block.kind == EditKind::UnitRewrite) return rewrite_unit(block, proposal, scope_begin, scope_end, style);

  if (detail::line_contents(block.search).empty()) {
    if (!lines.empty()) {
      return failed(Status::NotFound, "the SEARCH section is empty, but the file exists and is not empty");
    }
    // An empty file is filled like a new one.
    Resolution r;
    r.match.status = Status::Ok;
    r.match.candidates = 1;
    r.match.begin = r.match.end = original.size();
    r.splice.begin = r.splice.end = original.size();
    r.splice.text = block.replace;
    return r;
  }

  Resolution r;
  r.match = locate(original, block.search, options.match, scope_begin, scope_end);
  if (r.match.status == Status::Ambiguous) {
    return failed(Status::Ambiguous,
                  "the SEARCH text matches " + std::to_string(r.match.candidates) + " places (" +
                      to_string(r.match.kind) + " comparison); add lines or a scope to make it unique",
                  r.match);
  }
  if (r.match.status != Status::Ok) {
    return failed(r.match.status, "the SEARCH text was not found" +
                                      (block.scope.empty() ? std::string() : " in unit " + in_quotes(block.scope)),
                  r.match);
  }

  const std::size_t first = r.match.first_line - 1;
  const std::size_t last = r.match.last_line - 1;
  std::vector<std::string_view> found;  // the lines that are replaced
  for (std::size_t i = first; i <= last; ++i) found.push_back(detail::content_of(original, lines[i]));
  std::vector<std::string> replacement = owned_lines(block.replace);
  const auto refused = [&](const char* how, const std::string& why) {
    const Match located = r.match;
    return failed(Status::NotFound,
                  "the SEARCH text matches lines " + std::to_string(located.first_line) + "-" +
                      std::to_string(located.last_line) + how + ", but " + why,
                  located);
  };
  std::string why;
  if (r.match.kind == MatchKind::Indentation &&
      !detail::reindent_replacement(detail::line_contents(block.search), found, &replacement, &why)) {
    return refused(" up to indentation", why);
  }
  if (!detail::keeps_indentation_style(found, replacement, style, &why)) return refused("", why);
  r.splice.begin = lines[first].begin;
  r.splice.end = lines[last].end;
  r.splice.deletes_lines = replacement.empty();
  r.splice.text = join(replacement, proposal.line_ending, detail::terminator_of(original, lines[last]));
  return r;
}

void reject(FileProposal* proposal, Status status, std::string message) {
  proposal->status = status;
  proposal->message = std::move(message);
  proposal->updated = proposal->original;
}

// Runs the syntax hook and the diff; the last steps of every successful proposal.
void finish(FileProposal* proposal, const ApplyOptions& options) {
  const std::string new_label = "b/" + proposal->display_path;
  const std::string old_label = proposal->creates_file ? std::string("/dev/null") : "a/" + proposal->display_path;
  proposal->diff = unified_diff(proposal->original, proposal->updated, old_label, new_label);
  proposal->status = Status::Ok;
  if (!options.syntax_check) return;
  proposal->syntax_errors_before =
      proposal->creates_file ? 0 : options.syntax_check(proposal->path, proposal->original);
  proposal->syntax_errors_after = options.syntax_check(proposal->path, proposal->updated);
  if (proposal->syntax_errors_after > proposal->syntax_errors_before) {
    // The diff stays, so that the refused change can be shown.
    reject(proposal, Status::SyntaxRegression,
           "the edit raises the number of syntax errors from " + std::to_string(proposal->syntax_errors_before) +
               " to " + std::to_string(proposal->syntax_errors_after));
  }
}

FileProposal propose(const detail::ResolvedPath& target, const std::vector<const EditBlock*>& blocks,
                     const ApplyOptions& options) {
  FileProposal proposal;
  proposal.path = target.path;
  proposal.display_path = target.display;
  proposal.line_ending = "\n";
  proposal.original_hash = hash64(std::string_view());
  if (target.status != Status::Ok) {
    proposal.matches.resize(blocks.size());
    reject(&proposal, target.status, target.message);
    return proposal;
  }
  std::optional<std::string> content;
  std::string supplied;
  if (detail::from_snapshot_source(proposal.path, &supplied)) content = std::move(supplied);

  if (!content && fs::is_directory(proposal.path)) {
    proposal.matches.resize(blocks.size());
    reject(&proposal, Status::IoError, "the path names a directory");
    return proposal;
  }

  if (!content && !fs::exists(proposal.path)) {
    proposal.matches.resize(blocks.size());
    const bool all_create = std::all_of(blocks.begin(), blocks.end(), [](const EditBlock* b) {
      return b->kind == EditKind::SearchReplace && detail::line_contents(b->search).empty();
    });
    if (!all_create) {
      reject(&proposal, Status::FileMissing, "the file does not exist");
    } else if (blocks.size() > 1) {
      reject(&proposal, Status::Overlap, "several blocks create the same file");
    } else {
      proposal.creates_file = true;
      proposal.updated = blocks.front()->replace;
      proposal.matches.front().status = Status::Ok;
      proposal.matches.front().candidates = 1;
      finish(&proposal, options);
    }
    return proposal;
  }

  if (!content) content = read_file(proposal.path);
  if (!content) {
    proposal.matches.resize(blocks.size());
    reject(&proposal, Status::IoError, "the file cannot be read");
    return proposal;
  }
  proposal.original = std::move(*content);
  proposal.original_hash = hash64(proposal.original);
  if (proposal.original.find('\0') != std::string::npos) {
    proposal.matches.resize(blocks.size());
    reject(&proposal, Status::IoError, "the file is not a text file");
    return proposal;
  }
  const std::vector<Line> lines = detail::split_lines(proposal.original);
  proposal.line_ending = detail::dominant_line_ending(lines);
  const char style = options.keep_indent_style ? detail::indentation_style(proposal.original, lines) : '\0';

  // Every block is resolved against the same snapshot.
  std::vector<Splice> splices;
  Status status = Status::Ok;
  std::string message;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    Resolution r = resolve_block(*blocks[i], proposal, lines, options, style);
    proposal.matches.push_back(std::move(r.match));
    if (r.status != Status::Ok && status == Status::Ok) {
      status = r.status;
      message = blocks.size() > 1 ? "block " + std::to_string(i + 1) + ": " + r.message : r.message;
    }
    splices.push_back(std::move(r.splice));
  }
  if (status != Status::Ok) {
    reject(&proposal, status, std::move(message));
    return proposal;
  }

  std::vector<std::size_t> order(splices.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return splices[a].begin != splices[b].begin ? splices[a].begin < splices[b].begin : a < b;
  });
  for (std::size_t k = 1; k < order.size(); ++k) {
    const Splice& before = splices[order[k - 1]];
    const Splice& after = splices[order[k]];
    if (after.begin < before.end || after.begin == before.begin) {
      reject(&proposal, Status::Overlap,
             "blocks " + std::to_string(std::min(order[k - 1], order[k]) + 1) + " and " +
                 std::to_string(std::max(order[k - 1], order[k]) + 1) + " change the same lines");
      return proposal;
    }
  }

  // From the last to the first, so that the offsets of the snapshot stay valid.
  proposal.updated = proposal.original;
  for (std::size_t k = order.size(); k-- > 0;) {
    const Splice& splice = splices[order[k]];
    proposal.updated.replace(splice.begin, splice.end - splice.begin, splice.text);
  }
  // Removing the last lines of a file without final newline must not give it one.
  const Splice& final_splice = splices[order.back()];
  const bool had_final_newline = proposal.original.empty() || proposal.original.back() == '\n';
  if (!had_final_newline && final_splice.deletes_lines && final_splice.end == proposal.original.size() &&
      !proposal.updated.empty() && proposal.updated.back() == '\n') {
    proposal.updated.pop_back();
    if (!proposal.updated.empty() && proposal.updated.back() == '\r') proposal.updated.pop_back();
  }

  finish(&proposal, options);
  return proposal;
}

std::string make_id(const Transaction& transaction) {
  std::uint64_t h = hash64("prag::edit::Transaction");
  for (const FileProposal& file : transaction.files) {
    h = hash_combine(h, hash64(file.path));
    h = hash_combine(h, file.original_hash);
    h = hash_combine(h, hash64(file.updated));
    h = hash_combine(h, static_cast<std::uint64_t>(file.status) * 2 + (file.creates_file ? 1 : 0));
  }
  return "tx-" + hex64(h);
}

// ---- commit, revert, recover -----------------------------------------------------

using Json = nlohmann::json;

struct JournalEntry {
  std::string path;
  std::string backup;
  bool created = false;
  std::uint64_t original_hash = 0;
  std::uint64_t updated_hash = 0;
};

struct Journal {
  std::string id;
  std::vector<JournalEntry> files;
};

std::string write_journal(const Journal& journal) {
  Json doc;
  doc["format"] = 1;
  doc["id"] = journal.id;
  doc["files"] = Json::array();
  for (const JournalEntry& entry : journal.files) {
    Json file;
    file["path"] = entry.path;
    file["backup"] = entry.backup;
    file["created"] = entry.created;
    file["original_hash"] = hex64(entry.original_hash);
    file["updated_hash"] = hex64(entry.updated_hash);
    doc["files"].push_back(std::move(file));
  }
  return doc.dump(2, ' ', false, Json::error_handler_t::replace) + "\n";
}

bool read_journal(std::string_view text, Journal* journal) {
  try {
    const Json doc = Json::parse(text.begin(), text.end(), nullptr, false);
    if (!doc.is_object() || doc.value("format", 0) != 1) return false;
    journal->id = doc.value("id", std::string());
    if (!valid_id(journal->id)) return false;
    const auto files = doc.find("files");
    if (files == doc.end() || !files->is_array() || files->empty()) return false;
    for (const Json& file : *files) {
      JournalEntry entry;
      entry.path = file.value("path", std::string());
      entry.backup = file.value("backup", std::string());
      entry.created = file.value("created", false);
      if (entry.path.empty()) return false;
      if (!parse_hex64(file.value("original_hash", std::string()), &entry.original_hash)) return false;
      if (!parse_hex64(file.value("updated_hash", std::string()), &entry.updated_hash)) return false;
      journal->files.push_back(std::move(entry));
    }
    return true;
  } catch (const Json::exception&) {
    return false;
  }
}

// Puts back the first `count` files of the transaction. `before` holds the
// bytes that were on disk when commit() started.
bool roll_back(const Transaction& transaction, const std::vector<std::string>& before, std::size_t count) {
  bool ok = true;
  for (std::size_t i = count; i-- > 0;) {
    const FileProposal& file = transaction.files[i];
    const bool restored = file.creates_file ? remove_file(file.path) : write_file(file.path, before[i]);
    if (!restored) PRAG_ERROR("edit: could not restore %s", file.path.c_str());
    ok = ok && restored;
  }
  return ok;
}

}  // namespace

Transaction prepare(const std::vector<EditBlock>& blocks, const ApplyOptions& options) {
  Transaction transaction;
  if (blocks.empty()) {
    transaction.message = "there are no edit blocks";
    transaction.id = make_id(transaction);
    return transaction;
  }

  // Blocks are grouped by the file they address, whatever its spelling.
  struct Group {
    detail::ResolvedPath target;
    std::string key;
    std::vector<const EditBlock*> blocks;
  };
  std::vector<Group> groups;
  for (const EditBlock& block : blocks) {
    detail::ResolvedPath target = detail::resolve_path(block.path, options.allowed_roots);
    std::string key = target.status == Status::Ok ? target.key : "?" + target.path;
    const auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) { return g.key == key; });
    if (it != groups.end()) {
      it->blocks.push_back(&block);
    } else {
      groups.push_back({std::move(target), std::move(key), {&block}});
    }
  }

  transaction.status = Status::Ok;
  for (const Group& group : groups) {
    transaction.files.push_back(propose(group.target, group.blocks, options));
    const FileProposal& file = transaction.files.back();
    if (file.status != Status::Ok && transaction.status == Status::Ok) {
      transaction.status = file.status;
      transaction.message = file.display_path + ": " + file.message;
    }
  }
  transaction.id = make_id(transaction);
  return transaction;
}

Status commit(Transaction& transaction, const ApplyOptions& options) {
  const auto refuse = [&](Status status, std::string message) {
    transaction.message = std::move(message);
    return status;
  };
  if (transaction.files.empty()) return refuse(Status::NotFound, "the transaction has no files");
  if (transaction.status != Status::Ok) return transaction.status;
  for (const FileProposal& file : transaction.files) {
    if (file.status != Status::Ok) return refuse(file.status, file.display_path + ": " + file.message);
  }
  if (!valid_id(transaction.id)) return refuse(Status::IoError, "the transaction has no valid identifier");
  transaction.message.clear();

  // The proposal may have been handed around since prepare(); check again
  // where it points and what is on disk.
  std::vector<std::string> before(transaction.files.size());
  for (std::size_t i = 0; i < transaction.files.size(); ++i) {
    const FileProposal& file = transaction.files[i];
    const bool inside = std::any_of(options.allowed_roots.begin(), options.allowed_roots.end(),
                                    [&](const std::string& root) { return fs::is_within(file.path, root); });
    if (!inside) return refuse(Status::OutsideRoots, file.path + " lies outside the allowed roots");
    for (std::size_t k = 0; k < i; ++k) {
      if (fs::is_within(file.path, transaction.files[k].path) && fs::is_within(transaction.files[k].path, file.path)) {
        return refuse(Status::Overlap, file.path + " occurs twice in the transaction");
      }
    }
    const bool present = fs::exists(file.path);
    if (file.creates_file) {
      if (present) return refuse(Status::Conflict, file.path + " did not exist when the edit was prepared");
      continue;
    }
    if (!present || fs::is_directory(file.path)) {
      return refuse(Status::Conflict, file.path + " no longer exists");
    }
    std::optional<std::string> content = read_file(file.path);
    if (!content) return refuse(Status::IoError, file.path + " cannot be read");
    const bool snapshot_kept = hash64(file.original) == file.original_hash;
    if (hash64(*content) != file.original_hash || (snapshot_kept && *content != file.original)) {
      return refuse(Status::Conflict, file.path + " changed after the edit was prepared");
    }
    before[i] = std::move(*content);
  }
  if (options.dry_run) return Status::Ok;
  if (options.backup_dir.empty()) return refuse(Status::IoError, "no backup directory is configured");

  const std::string directory = fs::normalize(options.backup_dir);
  const std::string backups = directory + "/" + transaction.id;
  const std::string journal_path = directory + "/" + detail::journal_name(transaction.id);
  if (!fs::create_directories(backups)) return refuse(Status::IoError, "cannot create " + backups);

  Journal journal;
  journal.id = transaction.id;
  for (std::size_t i = 0; i < transaction.files.size(); ++i) {
    const FileProposal& file = transaction.files[i];
    JournalEntry entry;
    entry.path = file.path;
    entry.created = file.creates_file;
    entry.original_hash = file.original_hash;
    entry.updated_hash = hash64(file.updated);
    if (!file.creates_file) {
      entry.backup = backups + "/" + std::to_string(i) + ".bak";
      if (!write_file(entry.backup, before[i])) {
        remove_tree(backups);
        return refuse(Status::IoError, "cannot write the backup of " + file.path);
      }
    }
    journal.files.push_back(std::move(entry));
  }
  if (!write_file(journal_path, write_journal(journal))) {
    remove_tree(backups);
    return refuse(Status::IoError, "cannot write the journal " + journal_path);
  }
  transaction.journal_path = journal_path;
  const detail::Fault fault = detail::fault();
  if (fault == detail::Fault::CrashAfterJournal) return refuse(Status::IoError, "simulated crash");

  for (std::size_t i = 0; i < transaction.files.size(); ++i) {
    const FileProposal& file = transaction.files[i];
    const bool simulated_failure = fault == detail::Fault::FailSecondWrite && i == 1;
    if (simulated_failure || !write_file(file.path, file.updated)) {
      // Undo what was replaced. The journal goes only when that worked;
      // otherwise recover() finds it and tries again.
      if (roll_back(transaction, before, i)) {
        remove_file(journal_path);
        remove_tree(backups);
        transaction.journal_path.clear();
        return refuse(Status::IoError, "cannot write " + file.path + "; no file was changed");
      }
      return refuse(Status::IoError,
                    "cannot write " + file.path + " and cannot restore the files replaced before it; run recover()");
    }
    if (fault == detail::Fault::CrashAfterFirstWrite && i == 0) return refuse(Status::IoError, "simulated crash");
  }
  if (fault == detail::Fault::CrashBeforeCommitPoint) return refuse(Status::IoError, "simulated crash");

  // The commit point.
  if (!remove_file(journal_path)) {
    if (roll_back(transaction, before, transaction.files.size())) {
      remove_tree(backups);
      transaction.journal_path.clear();
    }
    return refuse(Status::IoError, "cannot remove the journal " + journal_path);
  }
  for (std::size_t i = 0; i < transaction.files.size(); ++i) {
    transaction.files[i].backup_path = options.keep_backup ? journal.files[i].backup : std::string();
  }
  if (!options.keep_backup) remove_tree(backups);
  return Status::Ok;
}

Status revert(const Transaction& transaction) {
  if (transaction.files.empty()) return Status::NotFound;
  if (transaction.status != Status::Ok) return transaction.status;

  // First find out what has to be done, then do it: a conflict in any file
  // leaves every file as it is.
  enum class Action { None, Restore, Remove };
  std::vector<Action> actions(transaction.files.size(), Action::None);
  std::vector<std::string> pre_edit(transaction.files.size());
  for (std::size_t i = 0; i < transaction.files.size(); ++i) {
    const FileProposal& file = transaction.files[i];
    const bool present = fs::exists(file.path);
    if (file.creates_file) {
      if (!present) continue;  // already gone
      const std::optional<std::string> content = read_file(file.path);
      if (!content) return Status::IoError;
      if (*content != file.updated) return Status::Conflict;
      actions[i] = Action::Remove;
      continue;
    }
    if (!present) return Status::Conflict;
    std::optional<std::string> content = read_file(file.path);
    if (!content) return Status::IoError;

    // The backup is the record of what was on disk; the snapshot in memory
    // serves when no backup was kept.
    std::optional<std::string> backup;
    if (!file.backup_path.empty()) backup = read_file(file.backup_path);
    if (backup && hash64(*backup) == file.original_hash) {
      pre_edit[i] = std::move(*backup);
    } else if (hash64(file.original) == file.original_hash) {
      pre_edit[i] = file.original;
    } else {
      return Status::IoError;
    }
    if (*content == pre_edit[i]) continue;  // already reverted
    if (*content != file.updated) return Status::Conflict;
    actions[i] = Action::Restore;
  }

  for (std::size_t i = transaction.files.size(); i-- > 0;) {
    const FileProposal& file = transaction.files[i];
    bool ok = true;
    if (actions[i] == Action::Restore) ok = write_file(file.path, pre_edit[i]);
    if (actions[i] == Action::Remove) ok = remove_file(file.path);
    if (ok) continue;
    // Back to the committed state.
    for (std::size_t k = i + 1; k < transaction.files.size(); ++k) {
      if (actions[k] == Action::None) continue;
      if (!write_file(transaction.files[k].path, transaction.files[k].updated)) {
        PRAG_ERROR("edit: could not write %s again", transaction.files[k].path.c_str());
      }
    }
    return Status::IoError;
  }
  return Status::Ok;
}

std::size_t recover(std::string_view journal_dir) {
  const std::string directory = fs::normalize(journal_dir);
  if (!fs::is_directory(directory)) return 0;

  std::vector<std::string> journals;
  std::error_code ec;
  const std::filesystem::path root(std::u8string(directory.begin(), directory.end()));
  for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code entry_ec;
    if (!it->is_regular_file(entry_ec) || it->path().extension() != ".journal") continue;
    const auto name = it->path().generic_u8string();
    journals.emplace_back(name.begin(), name.end());
  }
  std::sort(journals.begin(), journals.end());

  std::size_t restored = 0;
  for (const std::string& journal_path : journals) {
    const std::optional<std::string> text = read_file(journal_path);
    Journal journal;
    if (!text || !read_journal(*text, &journal)) {
      PRAG_WARN("edit: %s is not a valid journal; left in place", journal_path.c_str());
      continue;
    }

    // What has to be done for each file, decided before anything is written.
    enum class Action { None, Restore, Remove };
    std::vector<Action> actions(journal.files.size(), Action::None);
    std::vector<std::string> pre_edit(journal.files.size());
    bool resolvable = true;
    for (std::size_t i = 0; i < journal.files.size() && resolvable; ++i) {
      const JournalEntry& entry = journal.files[i];
      const bool present = fs::exists(entry.path);
      std::optional<std::string> content;
      if (present) content = read_file(entry.path);
      if (present && !content) {
        resolvable = false;
      } else if (entry.created) {
        if (!present) continue;
        // Anything but the content of the transaction was put there by someone else.
        if (hash64(*content) == entry.updated_hash) {
          actions[i] = Action::Remove;
        } else {
          resolvable = false;
        }
      } else {
        if (present && hash64(*content) == entry.original_hash) continue;
        if (present && hash64(*content) != entry.updated_hash) {
          resolvable = false;
          continue;
        }
        std::optional<std::string> backup;
        if (fs::is_within(entry.backup, directory)) backup = read_file(entry.backup);
        if (backup && hash64(*backup) == entry.original_hash) {
          pre_edit[i] = std::move(*backup);
          actions[i] = Action::Restore;
        } else {
          resolvable = false;
        }
      }
    }
    for (std::size_t i = journal.files.size(); resolvable && i-- > 0;) {
      const JournalEntry& entry = journal.files[i];
      if (actions[i] == Action::Restore) resolvable = write_file(entry.path, pre_edit[i]);
      if (actions[i] == Action::Remove) resolvable = remove_file(entry.path);
    }
    if (!resolvable) {
      // Kept under another name, so that the next start does not try again
      // and the backups can still be found by hand.
      PRAG_ERROR("edit: transaction %s could not be rolled back; see %s.unresolved", journal.id.c_str(),
                 journal_path.c_str());
      std::error_code rename_ec;
      const std::filesystem::path from(std::u8string(journal_path.begin(), journal_path.end()));
      std::filesystem::path to = from;
      to += ".unresolved";
      std::filesystem::rename(from, to, rename_ec);
      continue;
    }
    remove_file(journal_path);
    remove_tree(directory + "/" + journal.id);
    ++restored;
  }
  return restored;
}

}  // namespace prag::edit
