// Inline code edits: parsing model output, locating the target text, and
// applying the change as an atomic, reversible transaction.
//
// The model may answer in either of two content-anchored forms (no line numbers):
//
//   Unit rewrite (default for small local models): the complete new text of one
//   function or top-level declaration, identified by its name.
//
//       path/to/file.c
//       <<<<<<< UNIT mi_heap_malloc
//       ...complete new text of the unit...
//       >>>>>>> UNIT
//
//   Search/replace: used when the changed span is much shorter than the unit.
//   The optional scope line restricts the search to one unit.
//
//       path/to/file.c
//       @@ mi_heap_malloc
//       <<<<<<< SEARCH
//       exact lines to find
//       =======
//       replacement lines
//       >>>>>>> REPLACE
//
//   An empty SEARCH section on a path that does not exist creates the file.
//
// Locating SEARCH text walks a ladder of progressively looser comparisons and
// stops at the first rung on which anything matches:
//   1. Exact               byte-for-byte.
//   2. TrailingSpace       ignoring blanks at the end of each line.
//   3. Indentation         ignoring leading and trailing blanks; the replacement
//                          is re-indented by the indentation difference.
//   4. Fuzzy (opt-in)      the window of lines with the smallest edit distance d
//                          to the SEARCH text, accepted only when
//                          d <= floor(max_distance * |search|) and no other
//                          non-overlapping window is within that bound.
// On every rung the match must be unique inside the scope; several matches
// yield Ambiguous and nothing is applied. Uniqueness is tested inside the scope
// unit first and in the whole file only when the edit names no scope.
//
// Comparisons are made line by line on whole lines. The line terminator is not
// part of a line, so SEARCH text written with LF matches a CRLF file on the
// exact rung; the replacement is written with the line ending of the file.
// A match on rungs 1 to 3 is accepted only when no second location matches
// under the comparison of rung 3 (MatchOptions::unique_across_rungs): SEARCH
// text that was taken from an older state of the file cannot tell apart two
// places that differ in white space only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace prag::edit {

enum class EditKind { SearchReplace, UnitRewrite };

struct EditBlock {
  EditKind kind = EditKind::SearchReplace;
  std::string path;     // as written by the model
  std::string scope;    // unit name; required for UnitRewrite, optional otherwise
  std::string search;   // SearchReplace only; may be empty (file creation)
  std::string replace;  // replacement text, or the new unit text
};

struct ParseResult {
  std::vector<EditBlock> blocks;
  std::vector<std::string> warnings;  // malformed blocks that were skipped
};

// Tolerates surrounding prose, markdown code fences, CRLF line endings and
// between 5 and 9 marker characters.
ParseResult parse_edit_blocks(std::string_view model_output);

// GBNF grammar (llama.cpp) that constrains the envelope of a response to the
// two forms above while leaving the code payload free.
std::string_view edit_grammar();

enum class MatchKind { None, Exact, TrailingSpace, Indentation, Fuzzy, Unit };
enum class Status {
  Ok,
  NotFound,          // nothing matched on any enabled rung
  Ambiguous,         // more than one location matched on the first matching rung
  ScopeNotFound,     // the named unit does not exist in the file
  OutsideRoots,      // path resolves outside every allowed root
  FileMissing,
  Overlap,           // two blocks of one transaction touch the same bytes
  SyntaxRegression,  // the edit would add syntax errors
  IoError,
  Conflict,          // the file changed on disk after the proposal was prepared
};

const char* to_string(EditKind kind);
const char* to_string(MatchKind kind);
const char* to_string(Status status);

struct MatchOptions {
  bool allow_fuzzy = false;       // rung 4
  double max_distance = 0.10;     // rung 4 bound, as a fraction of the SEARCH length
  std::size_t min_fuzzy_lines = 4;    // shorter SEARCH texts never match fuzzily
  std::size_t max_fuzzy_lines = 400;  // longer SEARCH texts never match fuzzily
  // A single match on rung 1 or 2 is Ambiguous when rung 3 finds further
  // locations. false gives the plain ladder: the first rung that matches decides alone.
  bool unique_across_rungs = true;
};

struct Match {
  MatchKind kind = MatchKind::None;
  Status status = Status::NotFound;
  std::size_t begin = 0;  // byte offset of the matched range in the file
  std::size_t end = 0;    // one past the last matched byte
  std::uint32_t first_line = 0;  // 1-based
  std::uint32_t last_line = 0;   // 1-based, inclusive
  std::size_t distance = 0;      // edit distance in characters; 0 unless Fuzzy
  std::size_t candidates = 0;    // locations that matched on the deciding rung
  std::string indent_from;       // Indentation rung: indentation of the SEARCH text
  std::string indent_to;         // Indentation rung: indentation found in the file
};

// Byte range [scope_begin, scope_end) limits the search; pass 0 and npos for the whole file.
Match locate(std::string_view file_text, std::string_view search, const MatchOptions& options = {},
             std::size_t scope_begin = 0, std::size_t scope_end = static_cast<std::size_t>(-1));

// Levenshtein distance with early exit: returns `bound + 1` as soon as the distance exceeds `bound`.
std::size_t edit_distance(std::string_view a, std::string_view b, std::size_t bound);

// Hooks supplied by the caller, so that this module does not depend on a parser.
// SyntaxCheck: number of syntax errors in `text`, read as the language of `path`.
using SyntaxCheck = std::function<std::size_t(std::string_view path, std::string_view text)>;
// UnitLocator: byte range of the unit called `name`; false when it does not exist.
using UnitLocator = std::function<bool(std::string_view path, std::string_view text, std::string_view name,
                                       std::size_t* begin, std::size_t* end)>;

struct ApplyOptions {
  std::vector<std::string> allowed_roots;  // edits outside these directories are refused
  MatchOptions match;
  bool keep_backup = true;    // keep the pre-edit file for undo
  std::string backup_dir;     // where backups and journals live; required by commit()
  SyntaxCheck syntax_check;   // when set, edits that increase the error count are refused
  UnitLocator unit_locator;   // when set, resolves scope names; otherwise a scope is ScopeNotFound
  bool dry_run = false;       // compute everything, write nothing
  // Refuse a block whose replacement brings lines indented with tabs into a
  // file that is indented with spaces, or the reverse (NotFound, with the reason
  // in the message). Indentation that occurs on the replaced lines is accepted.
  bool keep_indent_style = true;
};

// Result for one file: every block addressed to the file, resolved against the
// same snapshot and applied together or not at all.
struct FileProposal {
  std::string path;                // normalised absolute path
  Status status = Status::NotFound;
  std::vector<Match> matches;      // one per block, in block order
  std::string original;            // snapshot the proposal was computed from
  std::string updated;             // content after all blocks
  std::string diff;                // unified diff, 3 lines of context
  std::uint64_t original_hash = 0;
  std::size_t syntax_errors_before = 0;
  std::size_t syntax_errors_after = 0;
  bool creates_file = false;
  std::string line_ending;         // "\n" or "\r\n", preserved from the original
  std::string message;             // explanation when status != Ok
  std::string backup_path;         // set by commit()
  std::string display_path;        // path relative to its root, as used in the labels of the diff
};

struct Transaction {
  std::string id;                  // stable identifier derived from the content
  Status status = Status::NotFound;  // Ok only when every file proposal is Ok
  std::vector<FileProposal> files;
  std::string journal_path;        // set by commit()
  std::string message;             // explanation when prepare() or commit() did not return Ok
};

// Resolves paths against the allowed roots, locates every block and computes the
// new content of each file. Writes nothing.
Transaction prepare(const std::vector<EditBlock>& blocks, const ApplyOptions& options);

// Writes all files of the transaction. A journal listing every target, its
// backup and its pre-edit hash is flushed before the first replacement and
// removed after the last; removing it is the commit point. Any failure restores
// the files already replaced. Fails with Conflict when a file no longer hashes
// to the snapshot.
Status commit(Transaction& transaction, const ApplyOptions& options);

// Restores the pre-edit content of every file of a committed transaction.
Status revert(const Transaction& transaction);

// Rolls back transactions whose journal survived a crash. Returns the number restored.
std::size_t recover(std::string_view journal_dir);

// Unified diff of two texts (Myers O(ND) on lines), with `context` lines around each hunk.
std::string unified_diff(std::string_view old_text, std::string_view new_text, std::string_view old_label,
                         std::string_view new_label, std::size_t context = 3);

}  // namespace prag::edit
