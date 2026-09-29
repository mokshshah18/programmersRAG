// Helpers shared by the implementation files of the edit module and by its
// tests. Not part of the public interface; other modules include edit.hpp only.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "prag/edit/edit.hpp"

namespace prag::edit::detail {

// ---- lines -------------------------------------------------------------------

// One line of a text. The terminator ("\n" or "\r\n") is not part of the
// content, so a comparison of contents is independent of the line-ending
// convention of the file.
struct Line {
  std::size_t begin = 0;        // offset of the first byte
  std::size_t content_end = 0;  // one past the last byte before the terminator
  std::size_t end = 0;          // one past the terminator; equals content_end on a last line without one
};

// Splits at '\n'; a '\r' directly before it belongs to the terminator. A UTF-8
// byte order mark at the start of the text is skipped, so that it is never
// part of a matched range. Empty text has no lines; "a\n" has one.
std::vector<Line> split_lines(std::string_view text);

inline std::string_view content_of(std::string_view text, const Line& line) {
  return text.substr(line.begin, line.content_end - line.begin);
}
inline std::string_view terminator_of(std::string_view text, const Line& line) {
  return text.substr(line.content_end, line.end - line.content_end);
}

// Contents of the lines of `text`, without terminators.
std::vector<std::string_view> line_contents(std::string_view text);

inline bool is_blank(char c) noexcept { return c == ' ' || c == '\t'; }
std::string_view strip_right(std::string_view line) noexcept;
std::string_view strip_both(std::string_view line) noexcept;
std::string_view indentation_of(std::string_view line) noexcept;  // the leading blanks
inline bool is_blank_line(std::string_view line) noexcept { return strip_right(line).empty(); }

// "\r\n" when more lines end in CRLF than in a bare LF, "\n" otherwise.
std::string dominant_line_ending(const std::vector<Line>& lines);

// ---- indentation ---------------------------------------------------------------

// Longest common prefix of the indentation of the non-blank lines; empty when
// every line is blank.
std::string common_indentation(const std::vector<std::string_view>& lines);

// Re-indents `replacement` (line contents) for a match found on the
// indentation rung, where search line k corresponds to file line k.
//
// A replacement line whose indentation also occurs in the SEARCH text receives
// the indentation that the file has on the corresponding line. Any other
// indentation is extrapolated, and only when that is unambiguous: the
// difference between SEARCH and file must be the same on every line, it must
// be a pure insertion or a pure removal of a prefix, and the result must not
// mix tabs and spaces in a way the file does not show. Blank lines are left
// alone. Returns false, with an explanation in `why`, when no safe
// re-indentation exists; `replacement` is then unchanged.
bool reindent_replacement(const std::vector<std::string_view>& search_lines,
                          const std::vector<std::string_view>& file_lines, std::vector<std::string>* replacement,
                          std::string* why);

// The character the file is indented with: '\t' or ' ' when at least nine of
// ten indented lines begin with it, 0 when the file has no indented line or
// uses both. Continuation lines of block comments (" * text") are not counted;
// files that are indented with tabs have them as well.
char indentation_style(std::string_view text, const std::vector<Line>& lines);

// Guards the indentation that an edit brings into a file. Every non-blank line
// of `replacement` must be indented with the character of the file (`style`),
// unless it is not indented at all, its indentation occurs on one of the lines
// that are replaced (`replaced`), or it continues a block comment (" * text").
// Nothing is refused when `style` is 0. Returns false with an explanation in
// `why` otherwise: text that a model wrote with four spaces for every tab would
// be found by its content and written into the file as it is.
bool keeps_indentation_style(const std::vector<std::string_view>& replaced,
                             const std::vector<std::string>& replacement, char style, std::string* why);

// ---- paths ---------------------------------------------------------------------

struct ResolvedPath {
  Status status = Status::OutsideRoots;
  std::string path;     // normalised absolute path, spelled as on disk below the root
  std::string root;     // normalised root the path was resolved against
  std::string display;  // path relative to the root
  std::string key;      // identity of the file; equal for every spelling of the same file
  std::string message;
};

// Resolves a path written by the model against the allowed roots (DESIGN 6.10).
// Containment is checked twice: lexically, and on the canonical path, so that a
// link inside a root cannot lead out of it.
ResolvedPath resolve_path(std::string_view written, const std::vector<std::string>& roots);

// ---- fault injection (tests only) ------------------------------------------------

// Simulated failures inside commit(). A crash returns from commit() at once and
// leaves the disk as a killed process would; a failed write takes the ordinary
// error path. The setting is process-wide and is not meant for production code.
enum class Fault {
  None,
  CrashAfterJournal,       // journal and backups written, no target replaced
  CrashAfterFirstWrite,    // one target replaced
  CrashBeforeCommitPoint,  // every target replaced, journal still present
  FailSecondWrite,         // replacing the second target reports an I/O error
};

void set_fault(Fault fault) noexcept;
Fault fault() noexcept;

// ---- snapshot source (tests only) ----------------------------------------------------
//
// Stands in for the disk as the source of file contents in prepare(). While a
// source is set, prepare() asks it for every path it has resolved: true means
// that the file exists and has the content stored in `text`, false that the
// disk is to be read as usual. Paths are still resolved against the file
// system, and commit(), revert() and recover() always work on the disk. This
// lets a test run thousands of cases through prepare() without writing a file
// for each of them. The setting is process-wide; an empty function removes it.
using SnapshotSource = std::function<bool(const std::string& path, std::string* text)>;

void set_snapshot_source(SnapshotSource source);

// File name of the journal of transaction `id` inside the backup directory.
std::string journal_name(std::string_view id);

}  // namespace prag::edit::detail
