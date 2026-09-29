// Line table and indentation arithmetic.
#include <algorithm>
#include <utility>

#include "internal.hpp"

namespace prag::edit::detail {

namespace {

constexpr std::string_view kByteOrderMark = "\xEF\xBB\xBF";

std::string_view common_prefix(std::string_view a, std::string_view b) {
  std::size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return a.substr(0, n);
}

std::size_t common_suffix_length(std::string_view a, std::string_view b) {
  std::size_t n = 0;
  while (n < a.size() && n < b.size() && a[a.size() - 1 - n] == b[b.size() - 1 - n]) ++n;
  return n;
}

bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// True when every character of `s` equals `c`.
bool consists_of(std::string_view s, char c) {
  return std::all_of(s.begin(), s.end(), [c](char x) { return x == c; });
}

}  // namespace

std::vector<Line> split_lines(std::string_view text) {
  std::vector<Line> lines;
  std::size_t begin = starts_with(text, kByteOrderMark) ? kByteOrderMark.size() : 0;
  while (begin < text.size()) {
    Line line;
    line.begin = begin;
    const std::size_t newline = text.find('\n', begin);
    if (newline == std::string_view::npos) {
      line.content_end = text.size();
      line.end = text.size();
    } else {
      line.content_end = newline > begin && text[newline - 1] == '\r' ? newline - 1 : newline;
      line.end = newline + 1;
    }
    lines.push_back(line);
    begin = line.end;
  }
  return lines;
}

std::vector<std::string_view> line_contents(std::string_view text) {
  std::vector<std::string_view> out;
  for (const Line& line : split_lines(text)) out.push_back(content_of(text, line));
  return out;
}

std::string_view strip_right(std::string_view line) noexcept {
  std::size_t end = line.size();
  while (end > 0 && is_blank(line[end - 1])) --end;
  return line.substr(0, end);
}

std::string_view strip_both(std::string_view line) noexcept {
  line = strip_right(line);
  std::size_t begin = 0;
  while (begin < line.size() && is_blank(line[begin])) ++begin;
  return line.substr(begin);
}

std::string_view indentation_of(std::string_view line) noexcept {
  std::size_t n = 0;
  while (n < line.size() && is_blank(line[n])) ++n;
  return line.substr(0, n);
}

std::string dominant_line_ending(const std::vector<Line>& lines) {
  std::size_t crlf = 0;
  std::size_t lf = 0;
  for (const Line& line : lines) {
    const std::size_t length = line.end - line.content_end;
    if (length == 2) ++crlf;
    if (length == 1) ++lf;
  }
  return crlf > lf ? "\r\n" : "\n";
}

std::string common_indentation(const std::vector<std::string_view>& lines) {
  bool first = true;
  std::string_view common;
  for (const std::string_view line : lines) {
    if (is_blank_line(line)) continue;
    const std::string_view indent = indentation_of(line);
    common = first ? indent : common_prefix(common, indent);
    first = false;
  }
  return std::string(common);
}

bool reindent_replacement(const std::vector<std::string_view>& search_lines,
                          const std::vector<std::string_view>& file_lines, std::vector<std::string>* replacement,
                          std::string* why) {
  const auto fail = [&](const char* reason) {
    if (why) *why = reason;
    return false;
  };
  if (search_lines.size() != file_lines.size()) return fail("internal error: line counts differ");

  // Indentation seen in the SEARCH text and the indentation of the file on the
  // same line. The relation must be a function.
  std::vector<std::pair<std::string_view, std::string_view>> observed;
  for (std::size_t k = 0; k < search_lines.size(); ++k) {
    if (is_blank_line(search_lines[k])) continue;
    const std::string_view from = indentation_of(search_lines[k]);
    const std::string_view to = indentation_of(file_lines[k]);
    const auto it = std::find_if(observed.begin(), observed.end(), [&](const auto& pair) { return pair.first == from; });
    if (it == observed.end()) {
      observed.emplace_back(from, to);
    } else if (it->second != to) {
      return fail("lines with equal indentation in the SEARCH text have different indentation in the file");
    }
  }

  // Uniform difference: SEARCH = from + rest, file = to + rest on every line.
  const std::string search_common = common_indentation(search_lines);
  const std::string file_common = common_indentation(file_lines);
  bool uniform = true;
  for (const auto& [from, to] : observed) {
    if (from.substr(search_common.size()) != to.substr(file_common.size())) uniform = false;
  }
  // Drop what both sides share at the end, so that replacement lines that are
  // less indented than every SEARCH line can still be shifted.
  const std::size_t shared = common_suffix_length(search_common, file_common);
  const std::string_view remove = std::string_view(search_common).substr(0, search_common.size() - shared);
  const std::string_view insert = std::string_view(file_common).substr(0, file_common.size() - shared);
  const bool pure_shift = uniform && (remove.empty() || insert.empty());

  std::vector<std::string> out;
  out.reserve(replacement->size());
  for (const std::string& line : *replacement) {
    if (is_blank_line(line)) {
      out.push_back(line);
      continue;
    }
    const std::string_view indent = indentation_of(line);
    const std::string_view body = std::string_view(line).substr(indent.size());
    const auto it =
        std::find_if(observed.begin(), observed.end(), [&](const auto& pair) { return pair.first == indent; });
    if (it != observed.end()) {
      out.push_back(std::string(it->second) + std::string(body));
      continue;
    }
    if (!pure_shift) {
      return fail("the replacement uses an indentation that does not occur in the SEARCH text, and the "
                  "indentation of the file does not differ from the SEARCH text by one fixed prefix");
    }
    if (!starts_with(indent, remove)) {
      return fail("a replacement line is indented less than the prefix that has to be removed");
    }
    const std::string_view rest = indent.substr(remove.size());
    // Appending spaces to a tab prefix (or the reverse) is what a file with
    // converted indentation would make of a uniform shift; refuse to guess.
    if (!insert.empty() && !rest.empty() && !consists_of(rest, insert.back())) {
      return fail("the replacement would mix tabs and spaces in its indentation");
    }
    out.push_back(std::string(insert) + std::string(rest) + std::string(body));
  }
  *replacement = std::move(out);
  return true;
}

namespace {

// " * text" and " */": the inner lines of a block comment.
bool continues_comment(std::string_view indent, std::string_view body) {
  return indent == " " && !body.empty() && body.front() == '*';
}

}  // namespace

char indentation_style(std::string_view text, const std::vector<Line>& lines) {
  std::size_t tabs = 0;
  std::size_t spaces = 0;
  for (const Line& line : lines) {
    const std::string_view content = content_of(text, line);
    if (is_blank_line(content)) continue;
    const std::string_view indent = indentation_of(content);
    if (indent.empty() || continues_comment(indent, content.substr(indent.size()))) continue;
    ++(indent.front() == '\t' ? tabs : spaces);
  }
  if (tabs != 0 && tabs >= 9 * spaces) return '\t';
  if (spaces != 0 && spaces >= 9 * tabs) return ' ';
  return 0;
}

bool keeps_indentation_style(const std::vector<std::string_view>& replaced,
                             const std::vector<std::string>& replacement, char style, std::string* why) {
  if (style == 0) return true;
  for (const std::string& line : replacement) {
    if (is_blank_line(line)) continue;
    const std::string_view indent = indentation_of(line);
    if (indent.empty() || indent.front() == style) continue;
    if (continues_comment(indent, std::string_view(line).substr(indent.size()))) continue;
    const bool known = std::any_of(replaced.begin(), replaced.end(), [&](std::string_view old_line) {
      return !is_blank_line(old_line) && indentation_of(old_line) == indent;
    });
    if (known) continue;
    if (why) {
      *why = style == '\t' ? "the file is indented with tabs and the replacement brings in lines indented with spaces"
                           : "the file is indented with spaces and the replacement brings in lines indented with tabs";
    }
    return false;
  }
  return true;
}

}  // namespace prag::edit::detail
