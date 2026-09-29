#include "prag/embed/hashing.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/text.hpp"

namespace prag::embed {

namespace {

constexpr std::string_view kWhitespace = " \t\r\n\f\v";

inline bool is_separator(char c) { return c == '/' || c == '\\'; }
inline bool is_space(char c) { return kWhitespace.find(c) != std::string_view::npos; }
inline bool is_word_byte(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c >= 0x80;
}

std::string path_tail(std::string_view path) {
  while (!path.empty() && is_separator(path.back())) path.remove_suffix(1);
  std::size_t begin = path.size();
  int separators = 0;
  while (begin > 0) {
    if (is_separator(path[begin - 1]) && ++separators == 2) break;
    --begin;
  }
  std::string tail(path.substr(begin));
  std::replace(tail.begin(), tail.end(), '\\', '/');
  return tail;
}

// Runs of whitespace become one space, so a declaration spread over several
// lines takes one line.
std::string on_one_line(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  bool pending_space = false;
  for (const char c : text::trim(s)) {
    if (is_space(c)) {
      pending_space = true;
      continue;
    }
    if (pending_space) out.push_back(' ');
    pending_space = false;
    out.push_back(c);
  }
  return out;
}

std::string without_blank_lines(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (const std::string_view line : text::split_lines(s)) {
    const std::string_view content = text::trim(line);
    if (content.empty()) continue;
    if (!out.empty()) out.push_back('\n');
    out.append(content);
  }
  return out;
}

// The text behind the comment block (and the blank lines) it starts with.
// An unterminated block comment is left in place.
std::string_view skip_leading_comment(std::string_view s, std::string_view language) {
  const bool hash_comments = language == "python";
  std::size_t i = 0;
  for (;;) {
    while (i < s.size() && is_space(s[i])) ++i;
    const std::string_view rest = s.substr(i);
    if (rest.starts_with("/*")) {
      const std::size_t close = rest.find("*/", 2);
      if (close == std::string_view::npos) break;
      i += close + 2;
    } else if (rest.starts_with("//") || (hash_comments && rest.starts_with("#"))) {
      const std::size_t line_end = rest.find('\n');
      if (line_end == std::string_view::npos) return {};
      i += line_end + 1;
    } else {
      break;
    }
  }
  return s.substr(i);
}

// Removes leading blank lines (the indentation of the first line stays) and
// trailing whitespace.
std::string_view strip_blank_edges(std::string_view s) {
  const std::size_t first = s.find_first_not_of(kWhitespace);
  if (first == std::string_view::npos) return {};
  const std::size_t newline = s.rfind('\n', first);
  s.remove_prefix(newline == std::string_view::npos ? first : newline + 1);
  const std::size_t last = s.find_last_not_of(kWhitespace);
  return s.substr(0, last + 1);
}

// Walks `s` in the normal form in which parse hands over a declaration: every
// run of white space, line splices and (with `c_comments`) comments is one
// blank, and there is none at the ends. Calls emit(character, end) for every
// character of that form, where `end` is the offset in `s` behind the
// character (for a blank: the offset at which the next character starts), and
// stops when emit returns false.
template <class Emit>
void walk_normal_form(std::string_view s, bool c_comments, Emit&& emit) {
  bool filler = false;  // something that counts as white space was passed
  bool first = true;
  std::size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c_comments && c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      const std::size_t close = s.find("*/", i + 2);
      i = close == std::string_view::npos ? s.size() : close + 2;
      filler = true;
    } else if (c_comments && c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
      const std::size_t line_end = s.find('\n', i);
      i = line_end == std::string_view::npos ? s.size() : line_end;
      filler = true;
    } else if (c == '\\' && i + 1 < s.size() && (s[i + 1] == '\n' || s[i + 1] == '\r')) {
      ++i;
      filler = true;
    } else if (is_space(c)) {
      ++i;
      filler = true;
    } else {
      if (filler && !first && !emit(' ', i)) return;
      if (!emit(c, i + 1)) return;
      filler = false;
      first = false;
      ++i;
    }
  }
}

std::string normal_form(std::string_view s, bool c_comments) {
  std::string out;
  out.reserve(s.size());
  walk_normal_form(s, c_comments, [&out](char c, std::size_t) {
    out.push_back(c);
    return true;
  });
  return out;
}

// Offset behind the first occurrence of `signature` in `text`, or npos. The
// literal text is looked for first; then the two are compared in their normal
// form, because the producer of the signature may have laid it out anew.
std::size_t end_of_signature(std::string_view text, std::string_view signature, bool c_comments) {
  const std::size_t literal = text.find(signature);
  if (literal != std::string_view::npos) return literal + signature.size();

  const std::string wanted = normal_form(signature, c_comments);
  if (wanted.empty()) return std::string_view::npos;
  const std::size_t at = normal_form(text, c_comments).find(wanted);
  if (at == std::string_view::npos) return std::string_view::npos;
  // Back from the normal form to the text: the offset behind character number
  // at + wanted.size() of the normal form.
  std::size_t remaining = at + wanted.size();
  std::size_t end = std::string_view::npos;
  walk_normal_form(text, c_comments, [&](char, std::size_t offset) {
    end = offset;
    return --remaining > 0;
  });
  return end;
}

inline bool is_string_prefix(char c) {
  return c == 'r' || c == 'R' || c == 'b' || c == 'B' || c == 'u' || c == 'U' || c == 'f' || c == 'F';
}

// Python: the text behind the documentation string that `body` starts with.
// parse puts that string into `doc`, so it would be rendered twice. `body` is
// returned as it is unless it starts with a string literal that stands alone
// on its line(s) and whose first line of text is found in `doc`.
std::string_view skip_documentation_string(std::string_view body, std::string_view doc) {
  std::size_t open = 0;
  while (open < body.size() && is_space(body[open])) ++open;
  for (int k = 0; k < 2 && open < body.size() && is_string_prefix(body[open]); ++k) ++open;
  if (open >= body.size() || (body[open] != '"' && body[open] != '\'')) return body;
  const bool triple = open + 2 < body.size() && body[open + 1] == body[open] && body[open + 2] == body[open];
  const std::string_view quotes = body.substr(open, triple ? 3 : 1);

  std::size_t close = open + quotes.size();
  for (;; ++close) {
    if (close >= body.size()) return body;  // not closed
    if (body[close] == '\\') {
      ++close;  // the character behind a backslash never closes the literal
    } else if (body[close] == '\n' && !triple) {
      return body;
    } else if (body.substr(close, quotes.size()) == quotes) {
      break;
    }
  }
  const std::size_t after = close + quotes.size();

  // "abc".join(parts) and "a" + b are expressions, not documentation.
  std::size_t rest = after;
  while (rest < body.size() && (body[rest] == ' ' || body[rest] == '\t' || body[rest] == '\r')) ++rest;
  if (rest < body.size() && body[rest] != '\n' && body[rest] != '#') return body;

  const std::string_view content = body.substr(open + quotes.size(), close - open - quotes.size());
  for (const std::string_view line : text::split_lines(content)) {
    const std::string_view words = text::trim(line);
    if (words.empty()) continue;
    if (doc.find(words) == std::string_view::npos) return body;
    break;
  }
  return body.substr(after);
}

std::string_view body_of(const SourceUnit& unit) {
  const bool python = unit.language == "python";
  const bool documented = !text::trim(unit.doc).empty();
  std::string_view body = unit.text;
  if (documented) body = skip_leading_comment(body, unit.language);
  const std::string_view signature = text::trim(unit.signature);
  if (!signature.empty()) {
    const std::size_t end = end_of_signature(body, signature, !python);
    if (end != std::string_view::npos) body.remove_prefix(end);
  }
  if (python && documented) body = skip_documentation_string(body, unit.doc);
  return strip_blank_edges(body);
}

// Appends `part` to `out` on a line of its own, or the longest prefix of it
// that keeps estimate_tokens(out) within `budget`; returns true when the part
// was appended completely. The estimate is evaluated on the assembled text,
// because it is not additive over lines (white space is counted per run).
bool append_within(std::string& out, std::string_view part, std::uint32_t budget) {
  if (part.empty()) return true;
  const std::size_t base = out.size() + (out.empty() ? 0 : 1);
  if (!out.empty()) out.push_back('\n');
  const auto fits = [&](std::size_t length) {
    out.resize(base);
    out.append(part.substr(0, length));
    return text::estimate_tokens(out) <= budget;
  };
  if (fits(part.size())) return true;

  // Ends of the lines of `part`, line terminators excluded.
  std::vector<std::size_t> ends;
  for (std::size_t at = part.find('\n'); at != std::string_view::npos; at = part.find('\n', at + 1)) {
    ends.push_back(at > 0 && part[at - 1] == '\r' ? at - 1 : at);
  }
  ends.push_back(part.size());

  // Largest number of whole lines that fit. The estimate grows with the
  // length of the text, so the fitting prefixes come first.
  std::size_t low = 0;  // lines known to fit
  std::size_t high = ends.size();  // lines known not to fit
  while (high - low > 1) {
    const std::size_t middle = low + (high - low) / 2;
    if (fits(ends[middle - 1])) {
      low = middle;
    } else {
      high = middle;
    }
  }
  std::size_t length = low == 0 ? 0 : ends[low - 1];

  if (low == 0) {
    // The first line alone is too long: cut inside it, in front of a word and
    // never inside a UTF-8 sequence.
    std::size_t fit = 0;
    std::size_t over = ends[0];
    while (over - fit > 1) {
      const std::size_t middle = fit + (over - fit) / 2;
      if (fits(middle)) {
        fit = middle;
      } else {
        over = middle;
      }
    }
    length = fit;
    while (length > 0 && is_word_byte(static_cast<unsigned char>(part[length])) &&
           is_word_byte(static_cast<unsigned char>(part[length - 1]))) {
      --length;
    }
    std::size_t content = length;  // without the blanks in front of the word
    while (content > 0 && is_space(part[content - 1])) --content;
    if (content == 0) {
      // One word longer than the budget, possibly behind its indentation.
      length = fit;
      while (length > 0 && (static_cast<unsigned char>(part[length]) & 0xC0) == 0x80) --length;
    }
  }

  while (length > 0 && is_space(part[length - 1])) --length;
  if (length == 0 || !fits(length)) {
    out.resize(base == 0 ? 0 : base - 1);  // nothing of the part fits: take the line break back
  }
  return false;
}

}  // namespace

std::string render_for_embedding(const SourceUnit& unit, std::size_t max_tokens) {
  const auto budget =
      static_cast<std::uint32_t>(std::min<std::size_t>(max_tokens, std::numeric_limits<std::uint32_t>::max()));

  std::string out;
  const auto add_line = [&out](std::string_view line) {
    if (line.empty()) return;
    if (!out.empty()) out.push_back('\n');
    out.append(line);
  };
  add_line(path_tail(unit.path));
  std::string heading = to_string(unit.kind);
  const std::string_view name = text::trim(unit.name);
  if (!name.empty()) {
    heading.push_back(' ');
    heading.append(name);
  }
  add_line(heading);
  add_line(on_one_line(unit.signature));

  // The body follows only a complete documentation; otherwise the budget is used up.
  if (append_within(out, without_blank_lines(unit.doc), budget)) append_within(out, body_of(unit), budget);
  return out;
}

}  // namespace prag::embed
