// Checks shared by the parse tests. Nothing here uses the code under test: the
// function counter in particular has a scanner of its own, so that it can serve
// as an independent reference.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"
#include "prag/parse/parser.hpp"
#include "prag_test.hpp"

namespace parse_checks {

inline void fail(const char* file, int line, const std::string& label, const std::string& message) {
  ::prag::test::report(file, line, label + ": " + message);
}

#define PARSE_EXPECT(cond, label, message)                                   \
  do {                                                                       \
    if (!(cond)) ::parse_checks::fail(__FILE__, __LINE__, (label), (message)); \
  } while (0)

// The partition property of parser.hpp and the fields derived from the text.
// Reports every violation; returns true when there is none.
inline bool check_partition(std::string_view text, const prag::parse::ParsedFile& file, const std::string& label,
                            const prag::parse::ParseOptions& options = {}) {
  const int before = ::prag::test::failures();
  const auto& units = file.units;
  PARSE_EXPECT(text.empty() == units.empty(), label, "units exist exactly when the text is not empty");
  std::uint32_t expected_begin = 0;
  std::uint32_t line = 1;  // line of the byte at expected_begin
  std::string joined;
  joined.reserve(text.size());
  for (std::size_t i = 0; i < units.size(); ++i) {
    const prag::SourceUnit& u = units[i];
    const std::string where = "unit " + std::to_string(i) + " '" + u.name + "'";
    PARSE_EXPECT(u.id == i, label, where + ": id is not the position in the file");
    PARSE_EXPECT(u.start_byte == expected_begin, label, where + ": does not start where the previous unit ends");
    PARSE_EXPECT(u.end_byte > u.start_byte, label, where + ": is empty");
    PARSE_EXPECT(u.end_byte <= text.size(), label, where + ": ends beyond the text");
    if (u.end_byte <= u.start_byte || u.end_byte > text.size() || u.start_byte != expected_begin) return false;
    PARSE_EXPECT(u.text == text.substr(u.start_byte, u.end_byte - u.start_byte), label,
                 where + ": text differs from the bytes of its range");
    PARSE_EXPECT(u.start_line == line, label,
                 where + ": start_line " + std::to_string(u.start_line) + ", expected " + std::to_string(line));
    const std::uint32_t lines = prag::text::count_lines(u.text);
    PARSE_EXPECT(u.end_line == u.start_line + lines - 1, label,
                 where + ": end_line " + std::to_string(u.end_line) + " does not match the line count");
    PARSE_EXPECT(u.content_hash == prag::hash64(u.text), label, where + ": content hash");
    PARSE_EXPECT(u.tokens == prag::text::estimate_tokens(u.text), label, where + ": token estimate");
    PARSE_EXPECT(u.path == file.path, label, where + ": path");
    PARSE_EXPECT(u.language == file.language, label, where + ": language");
    // residual regions have no name; a region with a name is a part of a unit
    const bool residual = u.kind == prag::UnitKind::Region && u.name.empty();
    const std::size_t limit = residual ? options.max_region_bytes : options.max_unit_bytes;
    if (u.kind == prag::UnitKind::Region && !u.name.empty()) {
      PARSE_EXPECT(u.name.find('#') != std::string::npos, label, where + ": a named region is a part, 'name#n'");
    }
    PARSE_EXPECT(u.text.size() <= limit, label,
                 where + ": " + std::to_string(u.text.size()) + " bytes, limit " + std::to_string(limit));
    if (u.kind != prag::UnitKind::Region) PARSE_EXPECT(!u.name.empty(), label, where + ": has no name");
    for (std::size_t a = 0; a < u.callees.size(); ++a) {
      PARSE_EXPECT(!u.callees[a].empty(), label, where + ": empty callee");
      for (std::size_t b = 0; b < a; ++b) {
        PARSE_EXPECT(u.callees[a] != u.callees[b], label, where + ": callee listed twice: " + u.callees[a]);
      }
    }
    joined += u.text;
    line += static_cast<std::uint32_t>(std::count(u.text.begin(), u.text.end(), '\n'));
    expected_begin = u.end_byte;
  }
  PARSE_EXPECT(expected_begin == text.size(), label, "the last unit does not end at the end of the file");
  PARSE_EXPECT(joined == text, label, "the unit texts do not reproduce the file");
  return ::prag::test::failures() == before;
}

inline bool is_identifier(std::string_view s) {
  if (s.empty() || (s[0] >= '0' && s[0] <= '9')) return false;
  for (const char c : s) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    if (!ok) return false;
  }
  static const char* const keywords[] = {"if",     "else",   "while",  "for",   "do",     "switch", "case",
                                         "return", "sizeof", "struct", "union", "enum",   "typedef", "static",
                                         "extern", "inline", "const",  "void",  "int",    "char",   "goto"};
  for (const char* keyword : keywords) {
    if (s == keyword) return false;
  }
  return true;
}

// "operator" followed by new, delete, a type or operator symbols.
inline bool is_operator_name(std::string_view s) {
  if (s.substr(0, 8) != "operator" || s.size() <= 8) return false;
  std::string_view rest = s.substr(8);
  if (rest[0] == ' ') {
    // new, delete, or the type of a conversion, which may be qualified
    rest.remove_prefix(1);
    rest = rest.substr(0, rest.find_first_of("[]*& "));
    if (rest == "new" || rest == "delete") return true;
    for (std::size_t begin = 0;;) {
      const std::size_t end = rest.find("::", begin);
      const std::string_view part = rest.substr(begin, end == std::string_view::npos ? end : end - begin);
      if (!is_identifier(part) && part != "int" && part != "char" && part != "void" && part != "const") return false;
      if (end == std::string_view::npos) return true;
      begin = end + 2;
    }
  }
  return rest.find_first_not_of("+-*/%^&|~!=<>,()[]") == std::string_view::npos;
}

// A C identifier, or a C++ name: identifiers joined by "::", where the last
// component may be a destructor or an operator.
inline bool is_function_name(std::string_view s, std::string_view separator = "::") {
  std::size_t begin = 0;
  for (;;) {
    const std::size_t end = s.substr(begin, 8) == "operator" ? std::string_view::npos : s.find(separator, begin);
    std::string_view part = s.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
    const bool last = end == std::string_view::npos;
    if (last && is_operator_name(part)) return true;
    if (last && !part.empty() && part[0] == '~') part.remove_prefix(1);
    if (!is_identifier(part)) return false;
    if (last) return true;
    begin = end + separator.size();
  }
}

// ---- independent reference: top-level function bodies ---------------------------------

// `text` with comments, literals and preprocessor lines replaced by spaces.
// Line breaks and quotes stay where they are.
inline std::string code_only(std::string_view text) {
  std::string out(text);
  const std::size_t n = text.size();
  std::size_t i = 0;
  bool line_start = true;  // only white space so far on this line
  const auto blank = [&](std::size_t from, std::size_t to) {
    for (std::size_t k = from; k < to && k < n; ++k) {
      if (out[k] != '\n') out[k] = ' ';
    }
  };
  while (i < n) {
    const char c = text[i];
    if (c == '\n') {
      line_start = true;
      ++i;
    } else if (c == ' ' || c == '\t' || c == '\r') {
      ++i;
    } else if (c == '/' && i + 1 < n && text[i + 1] == '/') {
      std::size_t j = i;
      while (j < n && text[j] != '\n') j += (text[j] == '\\' && j + 1 < n) ? 2 : 1;
      blank(i, j);
      i = std::min(j, n);
    } else if (c == '/' && i + 1 < n && text[i + 1] == '*') {
      std::size_t j = text.find("*/", i + 2);
      j = j == std::string_view::npos ? n : j + 2;
      blank(i, j);
      i = j;
    } else if (c == '#' && line_start) {
      // a directive with its continuation lines; comments inside it may span lines
      std::size_t j = i;
      while (j < n && text[j] != '\n') {
        if (text[j] == '\\' && j + 1 < n) {
          j += 2;
        } else if (text[j] == '/' && j + 1 < n && text[j + 1] == '*') {
          const std::size_t close = text.find("*/", j + 2);
          j = close == std::string_view::npos ? n : close + 2;
        } else if (text[j] == '/' && j + 1 < n && text[j + 1] == '/') {
          while (j < n && text[j] != '\n') j += (text[j] == '\\' && j + 1 < n) ? 2 : 1;
        } else {
          ++j;
        }
      }
      blank(i, j);
      i = std::min(j, n);
    } else if (c == '"' || c == '\'') {
      std::size_t j = i + 1;
      while (j < n && text[j] != c && text[j] != '\n') j += (text[j] == '\\' && j + 1 < n) ? 2 : 1;
      blank(i + 1, j);
      i = j < n && text[j] == c ? j + 1 : j;
      line_start = false;
    } else {
      line_start = false;
      ++i;
    }
  }
  return out;
}

// Offsets of the opening braces of the function bodies at the top level of a C
// file, found by counting braces. A block at depth zero is a function body when
// the text since the previous block or semicolon has a parenthesised list and
// is neither an initialiser (no '=') nor a type definition; `extern "C" {` does
// not count as a level.
inline std::vector<std::size_t> top_level_function_bodies(std::string_view text) {
  const std::string code = code_only(text);
  std::vector<std::size_t> bodies;
  const auto starts_with_word = [](std::string_view s, std::string_view word) {
    return s.substr(0, word.size()) == word &&
           (s.size() == word.size() || !(std::isalnum(static_cast<unsigned char>(s[word.size()])) ||
                                         s[word.size()] == '_'));
  };
  int depth = 0;
  int linkage = 0;  // open `extern "C" {` blocks
  std::size_t header = 0;
  for (std::size_t i = 0; i < code.size(); ++i) {
    const char c = code[i];
    if (c == '{') {
      if (depth == 0) {
        const std::string_view h = prag::text::trim(std::string_view(code).substr(header, i - header));
        if (starts_with_word(h, "extern") && !h.empty() && h.back() == '"' && h.find('(') == std::string_view::npos) {
          ++linkage;
          header = i + 1;
          continue;
        }
        const bool list = h.find('(') != std::string_view::npos && h.find(')') != std::string_view::npos;
        const bool type = starts_with_word(h, "typedef") ||
                          ((starts_with_word(h, "struct") || starts_with_word(h, "union") ||
                            starts_with_word(h, "enum")) &&
                           h.back() != ')');
        if (list && !type && h.find('=') == std::string_view::npos) bodies.push_back(i);
      }
      ++depth;
    } else if (c == '}') {
      if (depth > 0) {
        --depth;
      } else if (linkage > 0) {
        --linkage;
      }
      if (depth == 0) header = i + 1;
    } else if (c == ';' && depth == 0) {
      header = i + 1;
    }
  }
  return bodies;
}

inline const prag::SourceUnit* find_unit_named(const prag::parse::ParsedFile& file, std::string_view name,
                                               prag::UnitKind kind) {
  for (const prag::SourceUnit& unit : file.units) {
    if (unit.kind == kind && unit.name == name) return &unit;
  }
  return nullptr;
}

inline bool calls(const prag::SourceUnit& unit, std::string_view callee) {
  return std::find(unit.callees.begin(), unit.callees.end(), callee) != unit.callees.end();
}

inline bool same_units(const prag::parse::ParsedFile& a, const prag::parse::ParsedFile& b) {
  if (a.path != b.path || a.language != b.language || a.had_syntax_errors != b.had_syntax_errors ||
      a.units.size() != b.units.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.units.size(); ++i) {
    const prag::SourceUnit& x = a.units[i];
    const prag::SourceUnit& y = b.units[i];
    if (x.id != y.id || x.path != y.path || x.language != y.language || x.kind != y.kind || x.name != y.name ||
        x.signature != y.signature || x.doc != y.doc || x.text != y.text || x.start_line != y.start_line ||
        x.end_line != y.end_line || x.start_byte != y.start_byte || x.end_byte != y.end_byte ||
        x.callees != y.callees || x.content_hash != y.content_hash || x.tokens != y.tokens) {
      return false;
    }
  }
  return true;
}

}  // namespace parse_checks
