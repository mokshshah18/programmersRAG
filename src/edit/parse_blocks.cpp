// Parsing edit blocks out of model output, the grammar that constrains the
// output to that form, and the names of the enumerations.
#include <algorithm>
#include <string>

#include "internal.hpp"
#include "prag/core/text.hpp"

namespace prag::edit {

namespace {

enum class Marker { None, SearchBegin, UnitBegin, Divider, ReplaceEnd, UnitEnd };

struct Classified {
  Marker marker = Marker::None;
  std::string_view name;  // UnitBegin: the unit name
};

bool equals_icase(std::string_view a, std::string_view b) {
  return a.size() == b.size() && text::to_lower(a) == text::to_lower(b);
}

// A marker line starts in the first column with 5 to 9 equal characters out of
// '<', '=' and '>'. Anything else is payload or prose.
Classified classify(std::string_view line) {
  Classified out;
  line = detail::strip_right(line);
  if (line.empty()) return out;
  const char c = line[0];
  if (c != '<' && c != '=' && c != '>') return out;
  std::size_t run = 0;
  while (run < line.size() && line[run] == c) ++run;
  if (run < 5 || run > 9) return out;
  const std::string_view rest = line.substr(run);
  if (c == '=') {
    if (rest.empty()) out.marker = Marker::Divider;
    return out;
  }
  if (rest.empty() || !detail::is_blank(rest[0])) return out;
  const std::string_view words = detail::strip_both(rest);
  std::size_t split = 0;
  while (split < words.size() && !detail::is_blank(words[split])) ++split;
  const std::string_view keyword = words.substr(0, split);
  const std::string_view argument = detail::strip_both(words.substr(split));
  if (c == '<') {
    if (equals_icase(keyword, "SEARCH") && argument.empty()) out.marker = Marker::SearchBegin;
    if (equals_icase(keyword, "UNIT")) {
      out.marker = Marker::UnitBegin;
      out.name = argument;
    }
  } else {
    if (equals_icase(keyword, "REPLACE") && argument.empty()) out.marker = Marker::ReplaceEnd;
    if (equals_icase(keyword, "UNIT") && argument.empty()) out.marker = Marker::UnitEnd;
  }
  return out;
}

bool is_fence(std::string_view line) {
  const std::string_view s = detail::strip_both(line);
  return s.size() >= 3 && (s.compare(0, 3, "```") == 0 || s.compare(0, 3, "~~~") == 0);
}

bool starts_with_icase(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && equals_icase(s.substr(0, prefix.size()), prefix);
}

// Removes what models put around a file name: list bullets, heading marks,
// "File:", emphasis, quotes and a trailing colon.
std::string_view undecorate(std::string_view line) {
  const auto is_wrapping = [](char c) { return c == '`' || c == '*' || c == '"' || c == '\''; };
  std::string_view s = text::trim(line);
  while (!s.empty() && s.front() == '#') s.remove_prefix(1);
  s = text::trim(s);
  if (s.size() >= 3 && s.compare(0, 3, "// ") == 0) s = text::trim(s.substr(3));
  if (s.size() >= 2 && (s[0] == '-' || s[0] == '*' || s[0] == '+') && s[1] == ' ') s = text::trim(s.substr(2));
  // None of the characters removed here can begin or end a file name.
  for (bool changed = true; changed && !s.empty();) {
    const std::size_t before = s.size();
    while (!s.empty() && is_wrapping(s.front())) s.remove_prefix(1);
    while (!s.empty() && (is_wrapping(s.back()) || s.back() == ':')) s.remove_suffix(1);
    for (const std::string_view label : {std::string_view("file:"), std::string_view("path:")}) {
      if (starts_with_icase(s, label)) s.remove_prefix(label.size());
    }
    s = text::trim(s);
    changed = s.size() != before;
  }
  return s;
}

bool is_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '+' ||
         c == '-';
}

// A line is taken for a file name when it has a directory separator or ends in
// an extension, and contains nothing a path cannot contain. A sentence that
// merely mentions a file is not a file name.
bool looks_like_path(std::string_view s) {
  if (s.empty() || s.size() > 512) return false;
  bool separator = false;
  bool space = false;
  for (const char c : s) {
    const auto u = static_cast<unsigned char>(c);
    if (u < 0x20 || c == '<' || c == '>' || c == '|' || c == '"' || c == '?' || c == '*' || c == '`') return false;
    if (c == '/' || c == '\\') separator = true;
    if (c == ' ') space = true;
  }
  if (separator) return true;
  if (space) return false;
  const std::size_t dot = s.rfind('.');
  if (dot == std::string_view::npos || dot == 0 || dot + 1 == s.size() || s.size() - dot > 9) return false;
  return std::all_of(s.begin() + static_cast<std::ptrdiff_t>(dot) + 1, s.end(), is_name_char);
}

std::string_view clean_unit_name(std::string_view name) {
  name = undecorate(name);
  // "name()" and "name(args)" mean the function called name.
  const std::size_t paren = name.find('(');
  if (paren != std::string_view::npos) name = text::trim(name.substr(0, paren));
  return name;
}

struct Header {
  std::string path;
  std::string scope;
};

// Reads the lines above a begin marker: an optional "@@ unit" line and, above
// it, the file name. Fences and blank lines in between are skipped; the search
// never goes above `floor`, the first line after the previous block.
Header read_header(const std::vector<std::string_view>& lines, std::size_t marker_line, std::size_t floor) {
  Header header;
  constexpr std::size_t kMaxSkipped = 3;
  std::size_t skipped = 0;
  bool scope_seen = false;
  for (std::size_t i = marker_line; i-- > floor;) {
    const std::string_view line = lines[i];
    if (detail::is_blank_line(line) || is_fence(line)) {
      if (++skipped > kMaxSkipped) break;
      continue;
    }
    const std::string_view stripped = detail::strip_both(line);
    if (!scope_seen && stripped.size() >= 2 && stripped.compare(0, 2, "@@") == 0) {
      std::string_view name = detail::strip_both(stripped.substr(2));
      if (name.size() >= 2 && name.compare(name.size() - 2, 2, "@@") == 0) {
        name = detail::strip_both(name.substr(0, name.size() - 2));
      }
      header.scope = std::string(clean_unit_name(name));
      scope_seen = true;
      continue;
    }
    const std::string_view candidate = undecorate(line);
    if (looks_like_path(candidate)) header.path = std::string(candidate);
    break;
  }
  return header;
}

std::string join_payload(const std::vector<std::string_view>& lines, std::size_t begin, std::size_t end) {
  std::string out;
  for (std::size_t i = begin; i < end; ++i) {
    out.append(lines[i]);
    out.push_back('\n');
  }
  return out;
}

std::string at_line(std::size_t index) { return "line " + std::to_string(index + 1); }

}  // namespace

ParseResult parse_edit_blocks(std::string_view model_output) {
  ParseResult result;
  // Contents without terminators; CRLF output is read like LF output.
  const std::vector<std::string_view> lines = detail::line_contents(model_output);

  enum class State { Outside, Search, Replace, Unit, Dropped };
  State state = State::Outside;
  std::size_t block_line = 0;    // line of the begin marker
  std::size_t section_line = 0;  // first payload line of the current section
  std::size_t floor = 0;         // first line after the previous block
  Header header;
  std::string unit_name;
  std::string search;
  std::string last_path;

  const auto begin_block = [&](std::size_t i, const Classified& c) {
    block_line = i;
    section_line = i + 1;
    header = read_header(lines, i, floor);
    unit_name = std::string(clean_unit_name(c.name));
    search.clear();
    state = c.marker == Marker::UnitBegin ? State::Unit : State::Search;
  };
  const auto drop = [&](std::string message, State next) {
    result.warnings.push_back(std::move(message));
    state = next;
  };
  const auto finish = [&](std::size_t i, EditKind kind) {
    EditBlock block;
    block.kind = kind;
    block.path = header.path.empty() ? last_path : header.path;
    block.replace = join_payload(lines, section_line, i);
    if (kind == EditKind::UnitRewrite) {
      block.scope = unit_name;
    } else {
      block.scope = header.scope;
      block.search = search;
    }
    state = State::Outside;
    floor = i + 1;
    if (block.path.empty()) {
      result.warnings.push_back("block at " + at_line(block_line) + " skipped: no file name above it");
      return;
    }
    if (kind == EditKind::UnitRewrite && block.scope.empty()) {
      result.warnings.push_back("block at " + at_line(block_line) + " skipped: UNIT without a name");
      return;
    }
    last_path = block.path;
    result.blocks.push_back(std::move(block));
  };

  for (std::size_t i = 0; i < lines.size(); ++i) {
    const Classified c = classify(lines[i]);
    const bool begins = c.marker == Marker::SearchBegin || c.marker == Marker::UnitBegin;
    if (begins && state != State::Outside && state != State::Dropped) {
      result.warnings.push_back("block at " + at_line(block_line) + " skipped: not terminated before " + at_line(i));
      // The file name of the new block may be the last line of what was read as payload.
      floor = block_line + 1;
    }
    if (begins) {
      begin_block(i, c);
      continue;
    }
    switch (state) {
      case State::Outside:
        // A row of '=' under a heading is ordinary prose; an end marker is not.
        if (c.marker == Marker::ReplaceEnd || c.marker == Marker::UnitEnd) {
          result.warnings.push_back("end marker at " + at_line(i) + " ignored: no block is open");
          floor = i + 1;
        }
        break;
      case State::Search:
        if (c.marker == Marker::Divider) {
          search = join_payload(lines, section_line, i);
          section_line = i + 1;
          state = State::Replace;
        } else if (c.marker != Marker::None) {
          drop("block at " + at_line(block_line) + " skipped: end marker at " + at_line(i) + " before the divider",
               State::Outside);
          floor = i + 1;
        }
        break;
      case State::Replace:
        if (c.marker == Marker::ReplaceEnd) {
          finish(i, EditKind::SearchReplace);
        } else if (c.marker == Marker::Divider) {
          // Two dividers: which one separates SEARCH from REPLACE cannot be known.
          drop("block at " + at_line(block_line) + " skipped: second divider at " + at_line(i), State::Dropped);
        } else if (c.marker == Marker::UnitEnd) {
          drop("block at " + at_line(block_line) + " skipped: closed by a UNIT marker at " + at_line(i),
               State::Outside);
          floor = i + 1;
        }
        break;
      case State::Unit:
        // A row of '=' is payload here, the unit form has no divider.
        if (c.marker == Marker::UnitEnd) {
          finish(i, EditKind::UnitRewrite);
        } else if (c.marker == Marker::ReplaceEnd) {
          drop("block at " + at_line(block_line) + " skipped: closed by a REPLACE marker at " + at_line(i),
               State::Outside);
          floor = i + 1;
        }
        break;
      case State::Dropped:
        if (c.marker == Marker::ReplaceEnd || c.marker == Marker::UnitEnd) {
          state = State::Outside;
          floor = i + 1;
        }
        break;
    }
  }
  if (state == State::Search || state == State::Replace || state == State::Unit) {
    result.warnings.push_back("block at " + at_line(block_line) + " skipped: not terminated");
  }
  return result;
}

// The envelope only. A payload line is any line that does not begin with five
// or more characters out of '<', '=' and '>', which is what keeps the markers
// unambiguous; everything else about the code is left to the model.
//
//   root    one optional line of prose, then one or more blocks
//   prose   at most 160 characters, not starting like a marker, a scope line or a fence
//   block   file name on its own line, then either form; blank lines may precede it
//   unit    complete new text of the named unit
//   patch   SEARCH and REPLACE sections, optionally restricted to a unit by "@@ name"
//   line    empty, or starting with anything but a marker character, or with at
//           most four marker characters followed by something else
//
// Rule names use letters and dashes only, literals use the escapes \n and \\,
// and no rule is left-recursive, as llama.cpp requires.
std::string_view edit_grammar() {
  static constexpr std::string_view kGrammar =
      "root ::= prose? block+\n"
      "prose ::= [^\\n<=>@`~] [^\\n]{0,159} \"\\n\"\n"
      "block ::= \"\\n\"* path \"\\n\" (unit | patch)\n"
      "path ::= [A-Za-z0-9_.] [A-Za-z0-9_./+-]*\n"
      "unit ::= \"<<<<<<< UNIT \" name \"\\n\" line* \">>>>>>> UNIT\\n\"\n"
      "patch ::= (\"@@ \" name \"\\n\")? \"<<<<<<< SEARCH\\n\" line* \"=======\\n\" line* \">>>>>>> REPLACE\\n\"\n"
      "name ::= [A-Za-z_~] [A-Za-z0-9_:~]*\n"
      "line ::= \"\\n\" | [^\\n<=>] [^\\n]* \"\\n\" | [<=>]{1,4} (\"\\n\" | [^\\n<=>] [^\\n]* \"\\n\")\n";
  return kGrammar;
}

const char* to_string(EditKind kind) {
  switch (kind) {
    case EditKind::SearchReplace:
      return "search_replace";
    case EditKind::UnitRewrite:
      return "unit_rewrite";
  }
  return "unknown";
}

const char* to_string(MatchKind kind) {
  switch (kind) {
    case MatchKind::None:
      return "none";
    case MatchKind::Exact:
      return "exact";
    case MatchKind::TrailingSpace:
      return "trailing_space";
    case MatchKind::Indentation:
      return "indentation";
    case MatchKind::Fuzzy:
      return "fuzzy";
    case MatchKind::Unit:
      return "unit";
  }
  return "unknown";
}

const char* to_string(Status status) {
  switch (status) {
    case Status::Ok:
      return "ok";
    case Status::NotFound:
      return "not_found";
    case Status::Ambiguous:
      return "ambiguous";
    case Status::ScopeNotFound:
      return "scope_not_found";
    case Status::OutsideRoots:
      return "outside_roots";
    case Status::FileMissing:
      return "file_missing";
    case Status::Overlap:
      return "overlap";
    case Status::SyntaxRegression:
      return "syntax_regression";
    case Status::IoError:
      return "io_error";
    case Status::Conflict:
      return "conflict";
  }
  return "unknown";
}

}  // namespace prag::edit
