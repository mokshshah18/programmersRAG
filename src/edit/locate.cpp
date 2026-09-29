// Locating SEARCH text in a file: the matching ladder and the edit distance.
//
// All comparisons are made line by line on line contents; the terminator of a
// line is not part of its content. A file with CRLF endings is therefore
// matched by SEARCH text written with LF endings on the exact rung.
#include <algorithm>
#include <array>

#include "internal.hpp"

namespace prag::edit {

using detail::Line;

namespace {

enum class Rung { Exact, TrailingSpace, Indentation };

std::string_view normalised(std::string_view line, Rung rung) {
  switch (rung) {
    case Rung::Exact:
      return line;
    case Rung::TrailingSpace:
      return detail::strip_right(line);
    case Rung::Indentation:
      return detail::strip_both(line);
  }
  return line;
}

MatchKind kind_of(Rung rung) {
  switch (rung) {
    case Rung::Exact:
      return MatchKind::Exact;
    case Rung::TrailingSpace:
      return MatchKind::TrailingSpace;
    case Rung::Indentation:
      return MatchKind::Indentation;
  }
  return MatchKind::None;
}

struct Hits {
  std::size_t count = 0;
  std::size_t first = 0;  // index into the scope of the first window that matched
};

// Windows of `pattern.size()` consecutive lines equal to the pattern. Windows
// that overlap each other count separately: each is a distinct place the edit
// could go.
Hits find_windows(const std::vector<std::string_view>& lines, const std::vector<std::string_view>& pattern) {
  Hits hits;
  if (pattern.empty() || lines.size() < pattern.size()) return hits;
  const std::size_t last_start = lines.size() - pattern.size();
  for (std::size_t i = 0; i <= last_start; ++i) {
    if (lines[i] != pattern[0]) continue;
    bool equal = true;
    for (std::size_t k = 1; k < pattern.size() && equal; ++k) equal = lines[i + k] == pattern[k];
    if (!equal) continue;
    if (hits.count == 0) hits.first = i;
    ++hits.count;
  }
  return hits;
}

std::vector<std::string_view> normalise_all(const std::vector<std::string_view>& lines, Rung rung) {
  std::vector<std::string_view> out;
  out.reserve(lines.size());
  for (const std::string_view line : lines) out.push_back(normalised(line, rung));
  return out;
}

// ---- fuzzy rung ------------------------------------------------------------------

struct FuzzyWindow {
  std::size_t first = 0;  // scope line indices, inclusive
  std::size_t last = 0;
  std::size_t distance = 0;
};

struct FuzzyResult {
  std::vector<FuzzyWindow> windows;  // every line-aligned window within the bound that was examined
  bool ambiguous = false;
};

std::string join_lines(const std::vector<std::string_view>& lines, std::vector<std::size_t>* starts,
                       std::vector<std::size_t>* ends) {
  std::string out;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i != 0) out.push_back('\n');
    if (starts) starts->push_back(out.size());
    out.append(lines[i]);
    if (ends) ends->push_back(out.size());
  }
  return out;
}

bool disjoint(const FuzzyWindow& a, const FuzzyWindow& b) { return a.last < b.first || b.last < a.first; }

// Every window of whole lines whose distance to the pattern is within the
// bound.
//
// Windows are enumerated by their first line. A window qualifies only when its
// length differs from the length of the pattern by no more than the bound, so
// every first line has a few candidates. A candidate is compared with the
// pattern only when it passes the counting filter: one edit operation repairs
// one character at most, so a window that lacks more than `bound` of the
// characters of the pattern, or holds more than `bound` characters the pattern
// does not have, is further away than the bound. The filter discards nothing
// that the comparison would accept.
FuzzyResult fuzzy_windows(const std::vector<std::string_view>& lines, const std::vector<std::string_view>& pattern,
                          std::size_t bound) {
  FuzzyResult result;
  std::vector<std::size_t> starts;
  std::vector<std::size_t> ends;
  const std::string text = join_lines(lines, &starts, &ends);
  const std::string wanted = join_lines(pattern, nullptr, nullptr);
  const std::size_t p = wanted.size();
  if (p == 0 || bound == 0 || lines.empty()) return result;

  using Counts = std::array<std::size_t, 256>;
  Counts wanted_counts{};
  for (const char c : wanted) ++wanted_counts[static_cast<unsigned char>(c)];

  Counts counts{};
  for (std::size_t first = 0; first < lines.size() && !result.ambiguous; ++first) {
    counts.fill(0);
    for (std::size_t last = first; last < lines.size(); ++last) {
      const std::size_t length = ends[last] - starts[first];
      if (length > p + bound) break;
      // The line, and the separator in front of it when it is not the first.
      for (std::size_t i = last == first ? starts[first] : ends[last - 1]; i < ends[last]; ++i) {
        ++counts[static_cast<unsigned char>(text[i])];
      }
      if (length + bound < p) continue;

      std::size_t missing = 0;
      std::size_t surplus = 0;
      for (std::size_t c = 0; c < counts.size(); ++c) {
        if (wanted_counts[c] > counts[c]) {
          missing += wanted_counts[c] - counts[c];
        } else {
          surplus += counts[c] - wanted_counts[c];
        }
      }
      if (missing > bound || surplus > bound) continue;

      const std::size_t d = edit_distance(wanted, std::string_view(text).substr(starts[first], length), bound);
      if (d > bound) continue;
      const FuzzyWindow window{first, last, d};
      for (const FuzzyWindow& other : result.windows) {
        if (disjoint(window, other)) result.ambiguous = true;
      }
      result.windows.push_back(window);
    }
  }
  return result;
}

std::uint32_t to_line_number(std::size_t index) { return static_cast<std::uint32_t>(index + 1); }

}  // namespace

std::size_t edit_distance(std::string_view a, std::string_view b, std::size_t bound) {
  if (a.size() < b.size()) std::swap(a, b);
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  // The distance never exceeds the longer length, so a larger bound changes nothing.
  const std::size_t limit = std::min(bound, n);
  if (n - m > limit) return bound + 1;
  if (m == 0) return n;

  const std::size_t inf = limit + 1;
  std::vector<std::size_t> previous(m + 1);
  std::vector<std::size_t> current(m + 1, inf);
  for (std::size_t j = 0; j <= m; ++j) previous[j] = j <= limit ? j : inf;

  // Only the diagonal band |i - j| <= limit can hold values within the limit.
  for (std::size_t i = 1; i <= n; ++i) {
    const std::size_t low = i > limit ? i - limit : 1;
    const std::size_t high = std::min(m, i + limit);
    current[low - 1] = low == 1 ? std::min(i, inf) : inf;
    std::size_t row_minimum = current[low - 1];
    for (std::size_t j = low; j <= high; ++j) {
      const std::size_t substitute = previous[j - 1] + (a[i - 1] != b[j - 1] ? 1 : 0);
      const std::size_t value = std::min({substitute, previous[j] + 1, current[j - 1] + 1, inf});
      current[j] = value;
      row_minimum = std::min(row_minimum, value);
    }
    if (high < m) current[high + 1] = inf;
    if (row_minimum > limit) return bound + 1;
    std::swap(previous, current);
  }
  return previous[m] > limit ? bound + 1 : previous[m];
}

Match locate(std::string_view file_text, std::string_view search, const MatchOptions& options,
             std::size_t scope_begin, std::size_t scope_end) {
  Match match;
  const std::vector<Line> file_lines = detail::split_lines(file_text);
  const std::vector<std::string_view> pattern = detail::line_contents(search);
  if (pattern.empty() || file_lines.empty()) return match;

  // Lines that lie completely inside the scope.
  scope_end = std::min(scope_end, file_text.size());
  std::size_t low = 0;
  while (low < file_lines.size() && file_lines[low].begin < scope_begin) ++low;
  std::size_t high = file_lines.size();
  while (high > low && file_lines[high - 1].content_end > scope_end) --high;
  if (low >= high) return match;

  std::vector<std::string_view> scope;
  scope.reserve(high - low);
  for (std::size_t i = low; i < high; ++i) scope.push_back(detail::content_of(file_text, file_lines[i]));

  const auto place = [&](std::size_t first, std::size_t count) {
    match.begin = file_lines[low + first].begin;
    match.end = file_lines[low + first + count - 1].end;
    match.first_line = to_line_number(low + first);
    match.last_line = to_line_number(low + first + count - 1);
  };

  for (const Rung rung : {Rung::Exact, Rung::TrailingSpace, Rung::Indentation}) {
    const std::vector<std::string_view> wanted = normalise_all(pattern, rung);
    Hits hits = find_windows(normalise_all(scope, rung), wanted);
    if (hits.count == 0) continue;

    // A single match on a tight rung is not enough when the same lines occur
    // elsewhere with other white space: a SEARCH text taken from an older
    // state of the file cannot tell the two places apart.
    if (hits.count == 1 && rung != Rung::Indentation && options.unique_across_rungs) {
      const Hits loose = find_windows(normalise_all(scope, Rung::Indentation), normalise_all(pattern, Rung::Indentation));
      if (loose.count > 1) hits.count = loose.count;
    }

    match.kind = kind_of(rung);
    match.candidates = hits.count;
    match.status = hits.count == 1 ? Status::Ok : Status::Ambiguous;
    place(hits.first, pattern.size());
    if (rung == Rung::Indentation && hits.count == 1) {
      const std::vector<std::string_view> found(scope.begin() + static_cast<std::ptrdiff_t>(hits.first),
                                                scope.begin() + static_cast<std::ptrdiff_t>(hits.first + pattern.size()));
      match.indent_from = detail::common_indentation(pattern);
      match.indent_to = detail::common_indentation(found);
    }
    return match;
  }

  if (!options.allow_fuzzy) return match;
  if (pattern.size() < options.min_fuzzy_lines || pattern.size() > options.max_fuzzy_lines) return match;
  if (!(options.max_distance > 0.0)) return match;

  const std::vector<std::string_view> wanted = normalise_all(pattern, Rung::TrailingSpace);
  std::size_t characters = wanted.size() - 1;  // the separators between the lines
  for (const std::string_view line : wanted) characters += line.size();
  const double scaled = options.max_distance * static_cast<double>(characters);
  const std::size_t bound =
      scaled >= static_cast<double>(characters) ? characters : static_cast<std::size_t>(scaled);
  if (bound == 0) return match;

  const FuzzyResult fuzzy = fuzzy_windows(normalise_all(scope, Rung::TrailingSpace), wanted, bound);
  if (fuzzy.windows.empty()) return match;

  const auto best = std::min_element(fuzzy.windows.begin(), fuzzy.windows.end(),
                                     [](const FuzzyWindow& a, const FuzzyWindow& b) { return a.distance < b.distance; });
  const auto ties = static_cast<std::size_t>(
      std::count_if(fuzzy.windows.begin(), fuzzy.windows.end(),
                    [&](const FuzzyWindow& w) { return w.distance == best->distance; }));

  // Locations: the largest number of windows that do not overlap each other.
  std::vector<FuzzyWindow> by_end = fuzzy.windows;
  std::sort(by_end.begin(), by_end.end(), [](const FuzzyWindow& a, const FuzzyWindow& b) { return a.last < b.last; });
  std::size_t locations = 0;
  std::size_t next_free = 0;
  for (const FuzzyWindow& w : by_end) {
    if (locations != 0 && w.first < next_free) continue;
    ++locations;
    next_free = w.last + 1;
  }

  match.kind = MatchKind::Fuzzy;
  match.distance = best->distance;
  place(best->first, best->last - best->first + 1);
  if (locations > 1 || fuzzy.ambiguous) {
    match.status = Status::Ambiguous;
    match.candidates = std::max<std::size_t>(locations, 2);
  } else if (ties > 1) {
    // Windows of different extent at the same place fit equally well, so the
    // range to replace is not determined.
    match.status = Status::Ambiguous;
    match.candidates = ties;
  } else {
    match.status = Status::Ok;
    match.candidates = 1;
  }
  return match;
}

}  // namespace prag::edit
