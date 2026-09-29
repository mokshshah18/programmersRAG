#include "prag/core/text.hpp"

#include <algorithm>
#include <cctype>

namespace prag::text {

namespace {

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
inline bool is_upper(char c) { return c >= 'A' && c <= 'Z'; }
inline bool is_lower(char c) { return c >= 'a' && c <= 'z'; }
inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_alpha(char c) { return is_upper(c) || is_lower(c); }
inline char lower(char c) { return is_upper(c) ? static_cast<char>(c - 'A' + 'a') : c; }

}  // namespace

std::string_view trim(std::string_view s) {
  std::size_t begin = 0;
  std::size_t end = s.size();
  while (begin < end && is_space(s[begin])) ++begin;
  while (end > begin && is_space(s[end - 1])) --end;
  return s.substr(begin, end - begin);
}

std::string to_lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = lower(c);
  return out;
}

bool contains_icase(std::string_view haystack, std::string_view needle) {
  if (needle.empty()) return true;
  if (needle.size() > haystack.size()) return false;
  const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                              [](char a, char b) { return lower(a) == lower(b); });
  return it != haystack.end();
}

std::vector<std::string_view> split_lines(std::string_view s) {
  std::vector<std::string_view> lines;
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t end = s.find('\n', begin);
    if (end == std::string_view::npos) end = s.size();
    std::size_t line_end = end;
    if (line_end > begin && s[line_end - 1] == '\r') --line_end;
    lines.push_back(s.substr(begin, line_end - begin));
    begin = end + 1;
  }
  return lines;
}

std::vector<std::string> split_identifier(std::string_view identifier) {
  std::vector<std::string> words;
  std::string current;
  const auto flush = [&] {
    if (!current.empty()) {
      words.push_back(std::move(current));
      current.clear();
    }
  };
  for (std::size_t i = 0; i < identifier.size(); ++i) {
    const char c = identifier[i];
    if (!is_alpha(c) && !is_digit(c)) {
      flush();
      continue;
    }
    if (!current.empty()) {
      const char prev = identifier[i - 1];
      const bool digit_boundary = is_digit(c) != is_digit(prev);
      const bool hump = is_upper(c) && is_lower(prev);
      // "HTTPServer" splits before the last capital of an acronym run: HTTP | Server.
      const bool acronym_end =
          is_upper(c) && is_upper(prev) && i + 1 < identifier.size() && is_lower(identifier[i + 1]);
      if (digit_boundary || hump || acronym_end) flush();
    }
    current.push_back(lower(c));
  }
  flush();
  return words;
}

std::uint32_t estimate_tokens(std::string_view s) {
  std::uint32_t tokens = 0;
  std::size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (is_space(c)) {
      // Runs of indentation or blank lines cost roughly one token per 4 characters.
      std::size_t j = i;
      while (j < s.size() && is_space(s[j])) ++j;
      const std::size_t run = j - i;
      if (run >= 4) tokens += static_cast<std::uint32_t>(run / 4);
      i = j;
    } else if (is_alpha(c) || c == '_') {
      std::size_t j = i;
      while (j < s.size() && (is_alpha(s[j]) || is_digit(s[j]) || s[j] == '_')) ++j;
      const std::size_t words = split_identifier(s.substr(i, j - i)).size();
      // Long sub-words are split further by byte-pair tokenisers.
      const std::size_t length = j - i;
      tokens += static_cast<std::uint32_t>(std::max<std::size_t>(std::max<std::size_t>(words, 1), length / 6));
      i = j;
    } else if (is_digit(c)) {
      std::size_t j = i;
      while (j < s.size() && (is_digit(s[j]) || is_alpha(s[j]) || s[j] == '.')) ++j;
      tokens += static_cast<std::uint32_t>(1 + (j - i) / 3);
      i = j;
    } else if (static_cast<unsigned char>(c) >= 0x80) {
      std::size_t j = i;
      while (j < s.size() && static_cast<unsigned char>(s[j]) >= 0x80) ++j;
      tokens += static_cast<std::uint32_t>(1 + (j - i) / 3);
      i = j;
    } else {
      ++tokens;
      ++i;
    }
  }
  return tokens;
}

std::uint32_t count_lines(std::string_view s) {
  if (s.empty()) return 0;
  const auto newlines = static_cast<std::uint32_t>(std::count(s.begin(), s.end(), '\n'));
  return s.back() == '\n' ? newlines : newlines + 1;
}

std::string sanitize_utf8(std::string_view s) {
  static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t length = 0;
    std::uint32_t code = 0;
    if (b0 < 0x80) {
      out.push_back(static_cast<char>(b0));
      ++i;
      continue;
    } else if ((b0 & 0xE0) == 0xC0) {
      length = 2;
      code = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
      length = 3;
      code = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
      length = 4;
      code = b0 & 0x07;
    }
    bool valid = length != 0 && i + length <= s.size();
    for (std::size_t k = 1; valid && k < length; ++k) {
      const auto b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) {
        valid = false;
      } else {
        code = (code << 6) | (b & 0x3F);
      }
    }
    if (valid) {
      const bool overlong = (length == 2 && code < 0x80) || (length == 3 && code < 0x800) ||
                            (length == 4 && code < 0x10000);
      const bool surrogate = code >= 0xD800 && code <= 0xDFFF;
      if (overlong || surrogate || code > 0x10FFFF) valid = false;
    }
    if (valid) {
      out.append(s.substr(i, length));
      i += length;
    } else {
      out.append(kReplacement);
      ++i;
    }
  }
  return out;
}

}  // namespace prag::text
