#include "lexer.hpp"

#include <algorithm>
#include <string>

namespace prag::parse::detail {

namespace {

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }
inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_ident_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$' ||
         static_cast<unsigned char>(c) >= 0x80;
}
inline bool is_ident_char(char c) { return is_ident_start(c) || is_digit(c); }

// Length of a backslash-newline splice at `i`, 0 when there is none.
inline std::size_t splice_at(std::string_view s, std::size_t i) {
  if (s[i] != '\\' || i + 1 >= s.size()) return 0;
  if (s[i + 1] == '\n') return 2;
  if (s[i + 1] == '\r' && i + 2 < s.size() && s[i + 2] == '\n') return 3;
  return 0;
}

// End of a // comment starting at `i`: the newline that ends its logical line.
std::size_t line_comment_end(std::string_view s, std::size_t i) {
  std::size_t j = i;
  while (j < s.size()) {
    if (const std::size_t splice = splice_at(s, j)) {
      j += splice;
    } else if (s[j] == '\n') {
      break;
    } else {
      ++j;
    }
  }
  return j;
}

// End of a string or character literal whose opening quote is at `i`. An
// unterminated literal ends before the newline.
std::size_t quoted_end(std::string_view s, std::size_t i) {
  const char quote = s[i];
  std::size_t j = i + 1;
  while (j < s.size()) {
    if (const std::size_t splice = splice_at(s, j)) {
      j += splice;
    } else if (s[j] == '\\') {
      j = std::min(j + 2, s.size());
    } else if (s[j] == quote) {
      return j + 1;
    } else if (s[j] == '\n') {
      return j;
    } else {
      ++j;
    }
  }
  return j;
}

// End of a raw string literal R"delim( ... )delim" whose opening quote is at `i`.
std::size_t raw_string_end(std::string_view s, std::size_t i) {
  std::size_t j = i + 1;
  while (j < s.size() && s[j] != '(' && s[j] != '\n' && s[j] != '"' && j - i <= 17) ++j;
  if (j >= s.size() || s[j] != '(') return quoted_end(s, i);  // not a raw string after all
  std::string closing(")");
  closing.append(s.substr(i + 1, j - i - 1));
  closing.push_back('"');
  const std::size_t found = s.find(closing, j + 1);
  return found == std::string_view::npos ? s.size() : found + closing.size();
}

std::size_t number_end(std::string_view s, std::size_t i) {
  std::size_t j = i + 1;
  while (j < s.size()) {
    const char c = s[j];
    if (is_ident_char(c) || c == '.') {
      ++j;
    } else if ((c == '+' || c == '-') && (s[j - 1] == 'e' || s[j - 1] == 'E' || s[j - 1] == 'p' || s[j - 1] == 'P')) {
      ++j;
    } else if (c == '\'' && j + 1 < s.size() && is_ident_char(s[j + 1])) {
      ++j;  // digit separator
    } else {
      break;
    }
  }
  return j;
}

// End of the directive whose '#' is at `i`: the end of its logical line, where
// block comments that start on the line may extend it.
std::size_t directive_end(std::string_view s, std::size_t i) {
  std::size_t j = i + 1;
  while (j < s.size()) {
    const char c = s[j];
    if (const std::size_t splice = splice_at(s, j)) {
      j += splice;
    } else if (c == '\n') {
      break;
    } else if (c == '/' && j + 1 < s.size() && s[j + 1] == '*') {
      const std::size_t close = s.find("*/", j + 2);
      if (close == std::string_view::npos) return s.size();
      j = close + 2;
    } else if (c == '/' && j + 1 < s.size() && s[j + 1] == '/') {
      j = line_comment_end(s, j);
      break;
    } else if (c == '"' || c == '\'') {
      j = quoted_end(s, j);
    } else {
      ++j;
    }
  }
  if (j > i && j <= s.size() && s[j - 1] == '\r') --j;
  return j;
}

bool is_literal_prefix(std::string_view word) { return word == "L" || word == "u" || word == "U" || word == "u8"; }
bool is_raw_prefix(std::string_view word) {
  return word == "R" || word == "LR" || word == "uR" || word == "UR" || word == "u8R";
}

}  // namespace

std::vector<Token> lex_c_family(std::string_view s) {
  std::vector<Token> out;
  out.reserve(s.size() / 5 + 16);
  const std::size_t n = s.size();
  bool line_has_token = false;  // a token of any kind started on this line
  bool line_has_code = false;   // a token other than a comment started on this line
  const auto emit = [&](TokenKind kind, std::size_t begin, std::size_t end) {
    out.push_back({kind, !line_has_token, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end)});
    line_has_token = true;
    if (kind != TokenKind::Comment) line_has_code = true;
  };

  std::size_t i = 0;
  if (n >= 3 && s.substr(0, 3) == "\xEF\xBB\xBF") i = 3;  // byte order mark
  while (i < n) {
    const char c = s[i];
    if (c == '\n') {
      line_has_token = false;
      line_has_code = false;
      ++i;
    } else if (is_space(c)) {
      ++i;
    } else if (const std::size_t splice = splice_at(s, i)) {
      i += splice;
    } else if (c == '/' && i + 1 < n && s[i + 1] == '/') {
      std::size_t end = line_comment_end(s, i);
      const std::size_t resume = end;
      if (end > i && s[end - 1] == '\r') --end;
      emit(TokenKind::Comment, i, end);
      i = resume;
    } else if (c == '/' && i + 1 < n && s[i + 1] == '*') {
      const std::size_t close = s.find("*/", i + 2);
      const std::size_t end = close == std::string_view::npos ? n : close + 2;
      const bool multi_line = s.substr(i, end - i).find('\n') != std::string_view::npos;
      emit(TokenKind::Comment, i, end);
      if (multi_line) line_has_code = false;  // what preceded the comment is on an earlier line
      i = end;
    } else if (c == '#' && !line_has_code) {
      const std::size_t end = directive_end(s, i);
      emit(TokenKind::Directive, i, end);
      i = end;
    } else if (is_ident_start(c)) {
      std::size_t j = i + 1;
      while (j < n && is_ident_char(s[j])) ++j;
      const std::string_view word = s.substr(i, j - i);
      if (j < n && s[j] == '"' && is_raw_prefix(word)) {
        const std::size_t end = raw_string_end(s, j);
        emit(TokenKind::String, i, end);
        i = end;
      } else if (j < n && (s[j] == '"' || s[j] == '\'') && is_literal_prefix(word)) {
        const std::size_t end = quoted_end(s, j);
        emit(s[j] == '"' ? TokenKind::String : TokenKind::Character, i, end);
        i = end;
      } else {
        emit(TokenKind::Identifier, i, j);
        i = j;
      }
    } else if (is_digit(c) || (c == '.' && i + 1 < n && is_digit(s[i + 1]))) {
      const std::size_t end = number_end(s, i);
      emit(TokenKind::Number, i, end);
      i = end;
    } else if (c == '"' || c == '\'') {
      const std::size_t end = quoted_end(s, i);
      emit(c == '"' ? TokenKind::String : TokenKind::Character, i, end);
      i = end;
    } else if ((c == ':' && i + 1 < n && s[i + 1] == ':') || (c == '-' && i + 1 < n && s[i + 1] == '>')) {
      emit(TokenKind::Punct, i, i + 2);
      i += 2;
    } else {
      emit(TokenKind::Punct, i, i + 1);
      ++i;
    }
  }
  return out;
}

DirectiveInfo inspect_directive(std::string_view text, const Token& token) {
  DirectiveInfo info;
  const std::size_t end = token.end;
  std::size_t i = token.begin + 1;
  const auto skip_blank = [&] {
    while (i < end) {
      if (const std::size_t splice = splice_at(text, i)) {
        i += splice;
      } else if (text[i] == ' ' || text[i] == '\t') {
        ++i;
      } else if (text[i] == '/' && i + 1 < end && text[i + 1] == '*') {
        const std::size_t close = text.find("*/", i + 2);
        i = close == std::string_view::npos || close + 2 > end ? end : close + 2;
      } else {
        break;
      }
    }
  };
  const auto read_word = [&] {
    const std::size_t begin = i;
    while (i < end && is_ident_char(text[i])) ++i;
    return text.substr(begin, i - begin);
  };

  skip_blank();
  info.keyword = read_word();
  if (info.keyword == "if" || info.keyword == "ifdef" || info.keyword == "ifndef") {
    info.kind = DirectiveKind::If;
  } else if (info.keyword == "else" || info.keyword == "elif" || info.keyword == "elifdef" ||
             info.keyword == "elifndef") {
    info.kind = DirectiveKind::Else;
  } else if (info.keyword == "endif") {
    info.kind = DirectiveKind::Endif;
  } else if (info.keyword == "define") {
    info.kind = DirectiveKind::Define;
    skip_blank();
    info.name = read_word();
    info.name_end = static_cast<std::uint32_t>(i);
    if (i < end && text[i] == '(') {
      info.function_like = true;
      while (i < end && text[i] != ')') ++i;
      if (i < end) ++i;
    }
    skip_blank();
    info.body_begin = static_cast<std::uint32_t>(i);
  }
  return info;
}

LineIndex::LineIndex(std::string_view text) : size_(text.size()) {
  starts_.push_back(0);
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n') starts_.push_back(static_cast<std::uint32_t>(i + 1));
  }
}

std::uint32_t LineIndex::line_of(std::size_t offset) const {
  const auto it = std::upper_bound(starts_.begin(), starts_.end(), static_cast<std::uint32_t>(std::min(offset, size_)));
  return static_cast<std::uint32_t>(it - starts_.begin());
}

std::size_t LineIndex::next_line(std::size_t offset) const {
  const auto it = std::upper_bound(starts_.begin(), starts_.end(), static_cast<std::uint32_t>(std::min(offset, size_)));
  return it == starts_.end() ? size_ : *it;
}

}  // namespace prag::parse::detail
