// Unified diff of two texts with the O(ND) algorithm of Myers on lines.
//
// Lines are compared together with their terminators, so a change of line
// ending shows up as a changed line, and a missing final newline is reported
// with the usual "\ No newline at end of file" marker.
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "prag/edit/edit.hpp"

namespace prag::edit {

namespace {

enum class Op : std::uint8_t { Keep, Delete, Insert };

struct Step {
  Op op = Op::Keep;
  std::size_t old_index = 0;  // valid for Keep and Delete
  std::size_t new_index = 0;  // valid for Keep and Insert
};

// Lines including their terminators.
std::vector<std::string_view> split_keeping_terminators(std::string_view text) {
  std::vector<std::string_view> lines;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t newline = text.find('\n', begin);
    const std::size_t end = newline == std::string_view::npos ? text.size() : newline + 1;
    lines.push_back(text.substr(begin, end - begin));
    begin = end;
  }
  return lines;
}

// Beyond this number of differing lines the search for a shortest script is
// given up (its memory grows with the square of the number) and the whole
// differing region is reported as replaced.
constexpr std::size_t kMaxDifferences = 2048;

// Shortest edit script between a[0, n) and b[0, m), both given as line ids.
// Returns false when more than kMaxDifferences differences are needed.
bool shortest_script(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b,
                     std::vector<Op>* script) {
  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(a.size());
  const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(b.size());
  const std::ptrdiff_t limit = std::min<std::ptrdiff_t>(n + m, static_cast<std::ptrdiff_t>(kMaxDifferences));

  // furthest[k + limit + 1] is the largest x reached on diagonal k = x - y.
  std::vector<std::ptrdiff_t> furthest(static_cast<std::size_t>(2 * limit + 3), 0);
  std::vector<std::vector<std::ptrdiff_t>> history;  // the diagonals [-d, d] before round d
  const auto at = [&](std::ptrdiff_t k) -> std::ptrdiff_t& { return furthest[static_cast<std::size_t>(k + limit + 1)]; };

  std::ptrdiff_t rounds = -1;
  for (std::ptrdiff_t d = 0; d <= limit && rounds < 0; ++d) {
    std::vector<std::ptrdiff_t> snapshot(static_cast<std::size_t>(2 * d + 1));
    for (std::ptrdiff_t k = -d; k <= d; ++k) snapshot[static_cast<std::size_t>(k + d)] = at(k);
    history.push_back(std::move(snapshot));
    for (std::ptrdiff_t k = -d; k <= d; k += 2) {
      std::ptrdiff_t x = 0;
      if (k == -d || (k != d && at(k - 1) < at(k + 1))) {
        x = at(k + 1);  // step down: a line of b is inserted
      } else {
        x = at(k - 1) + 1;  // step right: a line of a is deleted
      }
      std::ptrdiff_t y = x - k;
      while (x < n && y < m && a[static_cast<std::size_t>(x)] == b[static_cast<std::size_t>(y)]) {
        ++x;
        ++y;
      }
      at(k) = x;
      if (x >= n && y >= m) {
        rounds = d;
        break;
      }
    }
  }
  if (rounds < 0) return false;

  // Walk back from the end through the snapshots.
  std::vector<Op> reversed;
  std::ptrdiff_t x = n;
  std::ptrdiff_t y = m;
  for (std::ptrdiff_t d = rounds; d > 0; --d) {
    const std::vector<std::ptrdiff_t>& before = history[static_cast<std::size_t>(d)];
    const auto seen = [&](std::ptrdiff_t k) { return before[static_cast<std::size_t>(k + d)]; };
    const std::ptrdiff_t k = x - y;
    const bool down = k == -d || (k != d && seen(k - 1) < seen(k + 1));
    const std::ptrdiff_t previous_k = down ? k + 1 : k - 1;
    const std::ptrdiff_t previous_x = seen(previous_k);
    const std::ptrdiff_t previous_y = previous_x - previous_k;
    while (x > previous_x && y > previous_y) {
      reversed.push_back(Op::Keep);
      --x;
      --y;
    }
    reversed.push_back(down ? Op::Insert : Op::Delete);
    if (down) {
      --y;
    } else {
      --x;
    }
  }
  while (x > 0 && y > 0) {
    reversed.push_back(Op::Keep);
    --x;
    --y;
  }
  script->assign(reversed.rbegin(), reversed.rend());
  return true;
}

std::vector<Step> edit_script(const std::vector<std::string_view>& old_lines,
                              const std::vector<std::string_view>& new_lines) {
  // Equal lines at both ends need no search.
  std::size_t prefix = 0;
  while (prefix < old_lines.size() && prefix < new_lines.size() && old_lines[prefix] == new_lines[prefix]) ++prefix;
  std::size_t suffix = 0;
  while (suffix < old_lines.size() - prefix && suffix < new_lines.size() - prefix &&
         old_lines[old_lines.size() - 1 - suffix] == new_lines[new_lines.size() - 1 - suffix]) {
    ++suffix;
  }
  const std::size_t old_middle = old_lines.size() - prefix - suffix;
  const std::size_t new_middle = new_lines.size() - prefix - suffix;

  std::unordered_map<std::string_view, std::uint32_t> ids;
  const auto id_of = [&](std::string_view line) {
    return ids.emplace(line, static_cast<std::uint32_t>(ids.size())).first->second;
  };
  std::vector<std::uint32_t> a(old_middle);
  std::vector<std::uint32_t> b(new_middle);
  for (std::size_t i = 0; i < old_middle; ++i) a[i] = id_of(old_lines[prefix + i]);
  for (std::size_t i = 0; i < new_middle; ++i) b[i] = id_of(new_lines[prefix + i]);

  std::vector<Op> middle;
  if (old_middle == 0 || new_middle == 0 || !shortest_script(a, b, &middle)) {
    middle.assign(old_middle, Op::Delete);
    middle.insert(middle.end(), new_middle, Op::Insert);
  }

  std::vector<Step> steps;
  steps.reserve(prefix + middle.size() + suffix);
  std::size_t i = 0;
  std::size_t j = 0;
  const auto push = [&](Op op) {
    steps.push_back({op, i, j});
    if (op != Op::Insert) ++i;
    if (op != Op::Delete) ++j;
  };
  for (std::size_t k = 0; k < prefix; ++k) push(Op::Keep);
  // Inside a run of changes the deletions are listed before the insertions.
  for (std::size_t k = 0; k < middle.size();) {
    if (middle[k] == Op::Keep) {
      push(Op::Keep);
      ++k;
      continue;
    }
    std::size_t run_end = k;
    while (run_end < middle.size() && middle[run_end] != Op::Keep) ++run_end;
    for (std::size_t r = k; r < run_end; ++r) {
      if (middle[r] == Op::Delete) push(Op::Delete);
    }
    for (std::size_t r = k; r < run_end; ++r) {
      if (middle[r] == Op::Insert) push(Op::Insert);
    }
    k = run_end;
  }
  for (std::size_t k = 0; k < suffix; ++k) push(Op::Keep);
  return steps;
}

void append_range(std::string* out, char sign, std::size_t start, std::size_t count) {
  out->push_back(sign);
  // An empty range is placed after the line before it, line 0 at the start of the file.
  out->append(std::to_string(count == 0 ? start : start + 1));
  if (count != 1) {
    out->push_back(',');
    out->append(std::to_string(count));
  }
}

void append_line(std::string* out, char sign, std::string_view line) {
  out->push_back(sign);
  if (!line.empty() && line.back() == '\n') {
    out->append(line);
  } else {
    out->append(line);
    out->append("\n\\ No newline at end of file\n");
  }
}

}  // namespace

std::string unified_diff(std::string_view old_text, std::string_view new_text, std::string_view old_label,
                         std::string_view new_label, std::size_t context) {
  if (old_text == new_text) return {};
  const std::vector<std::string_view> old_lines = split_keeping_terminators(old_text);
  const std::vector<std::string_view> new_lines = split_keeping_terminators(new_text);
  const std::vector<Step> steps = edit_script(old_lines, new_lines);

  std::string out;
  out.append("--- ").append(old_label).append("\n");
  out.append("+++ ").append(new_label).append("\n");

  std::size_t position = 0;
  while (position < steps.size()) {
    // Next change.
    std::size_t change = position;
    while (change < steps.size() && steps[change].op == Op::Keep) ++change;
    if (change == steps.size()) break;

    // A hunk runs until more than 2 * context unchanged lines separate two changes.
    const std::size_t hunk_begin = change - std::min(context, change - position);
    std::size_t hunk_end = change;  // one past the last step of the hunk
    std::size_t cursor = change;
    while (cursor < steps.size()) {
      if (steps[cursor].op != Op::Keep) {
        ++cursor;
        hunk_end = cursor;
        continue;
      }
      std::size_t run = cursor;
      while (run < steps.size() && steps[run].op == Op::Keep) ++run;
      const bool more_changes = run < steps.size();
      if (!more_changes || run - cursor > 2 * context) break;
      cursor = run;
    }
    const std::size_t trailing = std::min(context, steps.size() - hunk_end);
    const std::size_t hunk_stop = hunk_end + trailing;

    std::size_t old_count = 0;
    std::size_t new_count = 0;
    for (std::size_t s = hunk_begin; s < hunk_stop; ++s) {
      if (steps[s].op != Op::Insert) ++old_count;
      if (steps[s].op != Op::Delete) ++new_count;
    }
    out.append("@@ ");
    append_range(&out, '-', steps[hunk_begin].old_index, old_count);
    out.push_back(' ');
    append_range(&out, '+', steps[hunk_begin].new_index, new_count);
    out.append(" @@\n");
    for (std::size_t s = hunk_begin; s < hunk_stop; ++s) {
      switch (steps[s].op) {
        case Op::Keep:
          append_line(&out, ' ', old_lines[steps[s].old_index]);
          break;
        case Op::Delete:
          append_line(&out, '-', old_lines[steps[s].old_index]);
          break;
        case Op::Insert:
          append_line(&out, '+', new_lines[steps[s].new_index]);
          break;
      }
    }
    position = hunk_stop;
  }
  return out;
}

}  // namespace prag::edit
