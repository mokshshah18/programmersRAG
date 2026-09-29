// The matching ladder, the edit distance, re-indentation and the unified diff.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "../internal.hpp"
#include "edit_test_util.hpp"
#include "prag/edit/edit.hpp"

using namespace prag::edit;
using edit_test::Random;

namespace {

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

// Plain dynamic programme over the full matrix, as a reference.
std::size_t reference_distance(std::string_view a, std::string_view b) {
  std::vector<std::size_t> previous(b.size() + 1);
  std::vector<std::size_t> current(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) previous[j] = j;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    current[0] = i;
    for (std::size_t j = 1; j <= b.size(); ++j) {
      current[j] = std::min({previous[j - 1] + (a[i - 1] != b[j - 1] ? 1 : 0), previous[j] + 1, current[j - 1] + 1});
    }
    std::swap(previous, current);
  }
  return previous[b.size()];
}

std::string join(const std::vector<std::string>& lines, std::size_t first, std::size_t last) {
  std::string out;
  for (std::size_t i = first; i <= last; ++i) {
    if (i != first) out.push_back('\n');
    out.append(lines[i]);
  }
  return out;
}

std::string with_newlines(const std::vector<std::string>& lines) {
  std::string out;
  for (const std::string& line : lines) out.append(line).append("\n");
  return out;
}

MatchOptions fuzzy_options(double max_distance = 0.10, std::size_t min_lines = 4) {
  MatchOptions options;
  options.allow_fuzzy = true;
  options.max_distance = max_distance;
  options.min_fuzzy_lines = min_lines;
  return options;
}

const char* const kTwoFunctions =
    "int first(int x) {\n"    // line 1, offset 0
    "  if (x < 0) {\n"        // line 2
    "    return 0;\n"         // line 3
    "  }\n"                   // line 4
    "  return x;\n"           // line 5
    "}\n"                     // line 6
    "\n"                      // line 7
    "int second(int x) {\n"   // line 8
    "  if (x > 9) {\n"        // line 9
    "    return 0;\n"         // line 10
    "  }\n"                   // line 11
    "  return x + 1;\n"       // line 12
    "}\n";                    // line 13

}  // namespace

// ---- edit distance --------------------------------------------------------------------

PRAG_TEST(edit_distance_known_values) {
  CHECK_EQ(edit_distance("", "", 5), std::size_t{0});
  CHECK_EQ(edit_distance("abc", "", 5), std::size_t{3});
  CHECK_EQ(edit_distance("", "abc", 5), std::size_t{3});
  CHECK_EQ(edit_distance("abc", "abc", 0), std::size_t{0});
  CHECK_EQ(edit_distance("kitten", "sitting", 10), std::size_t{3});
  CHECK_EQ(edit_distance("sitting", "kitten", 10), std::size_t{3});
  CHECK_EQ(edit_distance("flaw", "lawn", 10), std::size_t{2});
  CHECK_EQ(edit_distance("intention", "execution", 100), std::size_t{5});
  CHECK_EQ(edit_distance("kitten", "sitting", kNone), std::size_t{3});
}

PRAG_TEST(edit_distance_stops_at_the_bound) {
  // Within the bound the exact distance, above it bound + 1.
  CHECK_EQ(edit_distance("kitten", "sitting", 3), std::size_t{3});
  CHECK_EQ(edit_distance("kitten", "sitting", 2), std::size_t{3});
  CHECK_EQ(edit_distance("kitten", "sitting", 1), std::size_t{2});
  CHECK_EQ(edit_distance("kitten", "sitting", 0), std::size_t{1});
  CHECK_EQ(edit_distance("abc", "abd", 0), std::size_t{1});
  // The difference of the lengths alone exceeds the bound.
  CHECK_EQ(edit_distance("a", "abcdefgh", 3), std::size_t{4});
  CHECK_EQ(edit_distance(std::string(10000, 'x'), std::string(10000, 'y'), 7), std::size_t{8});
  CHECK_EQ(edit_distance(std::string(5000, 'x'), std::string(5003, 'x'), 7), std::size_t{3});
}

PRAG_TEST(edit_distance_agrees_with_the_reference) {
  Random random(20260929);
  std::size_t above = 0;
  std::size_t within = 0;
  for (int round = 0; round < 3000; ++round) {
    const std::size_t alphabet = random.between(2, 4);
    std::string a(random.below(24), ' ');
    for (char& c : a) c = static_cast<char>('a' + random.below(alphabet));
    std::string b;
    if (random.chance(0.5)) {
      // A few changes to a, so that small distances occur.
      b = a;
      for (std::size_t k = random.below(5); k-- > 0;) {
        const std::size_t at = random.below(b.size() + 1);
        const std::size_t what = random.below(3);
        const char c = static_cast<char>('a' + random.below(alphabet));
        if (what == 0) b.insert(b.begin() + static_cast<std::ptrdiff_t>(at), c);
        if (what == 1 && at < b.size()) b.erase(b.begin() + static_cast<std::ptrdiff_t>(at));
        if (what == 2 && at < b.size()) b[at] = c;
      }
    } else {
      b.assign(random.below(24), ' ');
      for (char& c : b) c = static_cast<char>('a' + random.below(alphabet));
    }
    const std::size_t exact = reference_distance(a, b);
    const std::size_t bound = random.below(12);
    const std::size_t expected = exact <= bound ? exact : bound + 1;
    (exact <= bound ? within : above) += 1;
    CHECK_EQ(edit_distance(a, b, bound), expected);
    CHECK_EQ(edit_distance(b, a, bound), expected);
    CHECK_EQ(edit_distance(a, b, 1000), exact);
  }
  CHECK(within > 500u);
  CHECK(above > 500u);
}

// ---- ladder ---------------------------------------------------------------------------

PRAG_TEST(exact_rung) {
  const std::string file = "a\nb\nc\nd\n";
  for (const char* search : {"b\nc\n", "b\nc", "b\r\nc\r\n"}) {
    const Match m = locate(file, search);
    CHECK(m.status == Status::Ok);
    CHECK(m.kind == MatchKind::Exact);
    CHECK_EQ(m.begin, std::size_t{2});
    CHECK_EQ(m.end, std::size_t{6});
    CHECK_EQ(m.first_line, 2u);
    CHECK_EQ(m.last_line, 3u);
    CHECK_EQ(m.candidates, std::size_t{1});
    CHECK_EQ(m.distance, std::size_t{0});
  }
  // Whole lines only: a part of a line is not a match.
  CHECK(locate("int value = 1;\n", "value").status == Status::NotFound);
  CHECK(locate("ab\n", "a\nb\n").status == Status::NotFound);
}

PRAG_TEST(line_endings_do_not_take_part_in_the_comparison) {
  const std::string file = "a\r\nb\r\nc\r\n";
  const Match m = locate(file, "b\n");
  CHECK(m.status == Status::Ok);
  CHECK(m.kind == MatchKind::Exact);
  CHECK_EQ(m.begin, std::size_t{3});
  CHECK_EQ(m.end, std::size_t{6});

  // A last line without terminator, and a byte order mark in front of the first.
  const Match last = locate("a\nb", "b\n");
  CHECK(last.status == Status::Ok);
  CHECK_EQ(last.begin, std::size_t{2});
  CHECK_EQ(last.end, std::size_t{3});
  const Match first = locate("\xEF\xBB\xBFint a;\nint b;\n", "int a;\n");
  CHECK(first.status == Status::Ok);
  CHECK(first.kind == MatchKind::Exact);
  CHECK_EQ(first.begin, std::size_t{3});
  CHECK_EQ(first.end, std::size_t{10});
  CHECK_EQ(first.first_line, 1u);
}

PRAG_TEST(trailing_space_rung) {
  const Match file_has = locate("x\na  \nb\t\ny\n", "a\nb\n");
  CHECK(file_has.status == Status::Ok);
  CHECK(file_has.kind == MatchKind::TrailingSpace);
  CHECK_EQ(file_has.begin, std::size_t{2});
  CHECK_EQ(file_has.end, std::size_t{9});
  CHECK_EQ(file_has.first_line, 2u);
  CHECK_EQ(file_has.last_line, 3u);

  const Match search_has = locate("x\na\nb\ny\n", "a   \nb \t\n");
  CHECK(search_has.status == Status::Ok);
  CHECK(search_has.kind == MatchKind::TrailingSpace);
  CHECK_EQ(search_has.first_line, 2u);

  // A line of blanks is an empty line on this rung.
  const Match blank = locate("a\n   \nb\n", "a\n\nb\n");
  CHECK(blank.status == Status::Ok);
  CHECK(blank.kind == MatchKind::TrailingSpace);
  // Leading blanks still count.
  CHECK(locate("  a\n", "a \n", {}).kind == MatchKind::Indentation);
}

PRAG_TEST(indentation_rung) {
  const std::string file = "void f() {\n    x = 1;\n    if (y) {\n        z = 2;\n    }\n}\n";
  const Match m = locate(file, "x = 1;\nif (y) {\n    z = 2;\n}\n");
  CHECK(m.status == Status::Ok);
  CHECK(m.kind == MatchKind::Indentation);
  CHECK_EQ(m.first_line, 2u);
  CHECK_EQ(m.last_line, 5u);
  CHECK_EQ(m.indent_from, std::string(""));
  CHECK_EQ(m.indent_to, std::string("    "));
  CHECK_EQ(m.begin, std::size_t{11});
  CHECK_EQ(m.end, file.size() - 2);

  const Match tabs = locate("\tif (a)\n\t\tb();\n", "  if (a)\n    b();\n");
  CHECK(tabs.status == Status::Ok);
  CHECK(tabs.kind == MatchKind::Indentation);
  CHECK_EQ(tabs.indent_from, std::string("  "));
  CHECK_EQ(tabs.indent_to, std::string("\t"));
}

PRAG_TEST(nothing_matches) {
  CHECK(locate("a\nb\n", "c\n").status == Status::NotFound);
  CHECK(locate("a\nb\n", "c\n").kind == MatchKind::None);
  CHECK_EQ(locate("a\nb\n", "c\n").candidates, std::size_t{0});
  CHECK(locate("a\nb\n", "").status == Status::NotFound);
  CHECK(locate("", "a\n").status == Status::NotFound);
  CHECK(locate("", "").status == Status::NotFound);
  CHECK(locate("a\n", "a\nb\n").status == Status::NotFound);
  // The order of the lines matters, and so does every line.
  CHECK(locate("a\nb\nc\n", "b\na\n").status == Status::NotFound);
  CHECK(locate("a\nb\nc\n", "a\nc\n").status == Status::NotFound);
}

PRAG_TEST(several_matches_are_ambiguous) {
  const Match m = locate(kTwoFunctions, "    return 0;\n");
  CHECK(m.status == Status::Ambiguous);
  CHECK(m.kind == MatchKind::Exact);
  CHECK_EQ(m.candidates, std::size_t{2});

  // More lines make the text unique.
  const Match unique = locate(kTwoFunctions, "  if (x > 9) {\n    return 0;\n");
  CHECK(unique.status == Status::Ok);
  CHECK_EQ(unique.first_line, 9u);
  CHECK_EQ(unique.last_line, 10u);

  // Windows that overlap each other are separate places.
  const Match overlapping = locate("\n\n\n", "\n\n");
  CHECK(overlapping.status == Status::Ambiguous);
  CHECK_EQ(overlapping.candidates, std::size_t{2});

  const Match loose = locate("  a();\n    a();\n", "a();\n");
  CHECK(loose.status == Status::Ambiguous);
  CHECK(loose.kind == MatchKind::Indentation);
  CHECK_EQ(loose.candidates, std::size_t{2});
}

PRAG_TEST(a_tight_match_does_not_hide_a_loose_one) {
  // The first statement matches exactly, the second up to white space. Text
  // from an older state of the file could mean either.
  const std::string file = "  value = 1;\n    value = 1;\nvalue = 1;  \n";
  const Match strict = locate(file, "  value = 1;\n");
  CHECK(strict.status == Status::Ambiguous);
  CHECK(strict.kind == MatchKind::Exact);
  CHECK_EQ(strict.candidates, std::size_t{3});
  CHECK_EQ(strict.first_line, 1u);

  MatchOptions plain;
  plain.unique_across_rungs = false;
  const Match first = locate(file, "  value = 1;\n", plain);
  CHECK(first.status == Status::Ok);
  CHECK(first.kind == MatchKind::Exact);
  CHECK_EQ(first.first_line, 1u);
  CHECK_EQ(first.candidates, std::size_t{1});

  const Match second = locate("a;\na;  \n", "a;\n", plain);
  CHECK(second.status == Status::Ok);
  CHECK(second.kind == MatchKind::Exact);
  CHECK(locate("a;\na;  \n", "a;\n").status == Status::Ambiguous);
}

PRAG_TEST(the_scope_limits_the_search) {
  const std::string file = kTwoFunctions;
  const std::size_t second = file.find("int second");
  REQUIRE(second != std::string::npos);

  const Match in_second = locate(file, "    return 0;\n", {}, second, kNone);
  CHECK(in_second.status == Status::Ok);
  CHECK_EQ(in_second.first_line, 10u);
  CHECK(in_second.begin > second);

  const Match in_first = locate(file, "    return 0;\n", {}, 0, second);
  CHECK(in_first.status == Status::Ok);
  CHECK_EQ(in_first.first_line, 3u);

  // Text that exists in the file, but outside of the scope.
  const Match outside = locate(file, "  return x + 1;\n", {}, 0, second);
  CHECK(outside.status == Status::NotFound);
  // A match must lie inside the scope with every line.
  const Match straddling = locate(file, "}\n\nint second(int x) {\n", {}, 0, second);
  CHECK(straddling.status == Status::NotFound);
  CHECK(locate(file, "}\n\nint second(int x) {\n").status == Status::Ok);

  // A scope that does not start at a line start leaves out the line it cuts.
  const Match cut = locate("ab\ncd\nef\n", "ab\n", {}, 1, kNone);
  CHECK(cut.status == Status::NotFound);
  CHECK(locate("ab\ncd\nef\n", "cd\n", {}, 1, kNone).status == Status::Ok);
  // Empty and inverted scopes.
  CHECK(locate(file, "}\n", {}, 5, 5).status == Status::NotFound);
  CHECK(locate(file, "}\n", {}, 50, 10).status == Status::NotFound);
  CHECK(locate(file, "}\n", {}, file.size(), kNone).status == Status::NotFound);
}

// ---- fuzzy rung -----------------------------------------------------------------------

PRAG_TEST(the_fuzzy_rung_is_off_by_default) {
  const std::string file = kTwoFunctions;
  // One character differs: "x > 9" became "x > 8".
  const std::string search = "int second(int x) {\n  if (x > 8) {\n    return 0;\n  }\n";
  CHECK(!MatchOptions().allow_fuzzy);
  const Match off = locate(file, search);
  CHECK(off.status == Status::NotFound);
  CHECK(off.kind == MatchKind::None);

  const Match on = locate(file, search, fuzzy_options());
  CHECK(on.status == Status::Ok);
  CHECK(on.kind == MatchKind::Fuzzy);
  CHECK_EQ(on.distance, std::size_t{1});
  CHECK_EQ(on.candidates, std::size_t{1});
  CHECK_EQ(on.first_line, 8u);
  CHECK_EQ(on.last_line, 11u);
  CHECK_EQ(on.begin, file.find("int second"));
  CHECK_EQ(on.end, file.find("  return x + 1;"));

  // Text that matches on an earlier rung never reaches the fuzzy one.
  const Match exact = locate(file, "int second(int x) {\n  if (x > 9) {\n    return 0;\n  }\n", fuzzy_options());
  CHECK(exact.kind == MatchKind::Exact);
  CHECK_EQ(exact.distance, std::size_t{0});
}

PRAG_TEST(the_fuzzy_rung_respects_the_line_limits) {
  const std::string file = kTwoFunctions;
  const std::string three = "int second(int x) {\n  if (x > 8) {\n    return 0;\n";
  const std::string five = "int second(int x) {\n  if (x > 8) {\n    return 0;\n  }\n  return x + 1;\n";
  CHECK_EQ(MatchOptions().min_fuzzy_lines, std::size_t{4});
  CHECK_EQ(MatchOptions().max_fuzzy_lines, std::size_t{400});

  // Three lines are below the default minimum of four.
  CHECK(locate(file, three, fuzzy_options()).status == Status::NotFound);
  const Match short_allowed = locate(file, three, fuzzy_options(0.10, 3));
  CHECK(short_allowed.status == Status::Ok);
  CHECK(short_allowed.kind == MatchKind::Fuzzy);
  CHECK_EQ(short_allowed.first_line, 8u);
  CHECK_EQ(short_allowed.last_line, 10u);

  MatchOptions capped = fuzzy_options();
  CHECK(locate(file, five, capped).status == Status::Ok);
  capped.max_fuzzy_lines = 4;
  CHECK(locate(file, five, capped).status == Status::NotFound);
  capped.max_fuzzy_lines = 5;
  CHECK(locate(file, five, capped).kind == MatchKind::Fuzzy);
}

PRAG_TEST(the_fuzzy_rung_respects_the_distance_bound) {
  // Four lines of nine characters and three separators: 39 characters. With
  // the default fraction of 0.10 the bound is floor(3.9) = 3.
  const std::string file = "header\nalpha one\nalpha two\nalpha six\nalpha ten\nfooter\n";
  const auto search_with = [](std::size_t changes) {
    std::vector<std::string> lines = {"alpha one", "alpha two", "alpha six", "alpha ten"};
    for (std::size_t k = 0; k < changes; ++k) lines[k][8] = '#';
    return with_newlines(lines);
  };
  for (std::size_t changes = 1; changes <= 3; ++changes) {
    const Match m = locate(file, search_with(changes), fuzzy_options());
    CHECK(m.status == Status::Ok);
    CHECK(m.kind == MatchKind::Fuzzy);
    CHECK_EQ(m.distance, changes);
    CHECK_EQ(m.first_line, 2u);
    CHECK_EQ(m.last_line, 5u);
    CHECK_EQ(m.candidates, std::size_t{1});
  }
  // A fourth change exceeds the bound.
  const Match beyond = locate(file, search_with(4), fuzzy_options());
  CHECK(beyond.status == Status::NotFound);
  CHECK(beyond.kind == MatchKind::None);
  // A larger fraction admits it: floor(0.20 * 39) = 7. No fraction, no match.
  const Match wide = locate(file, search_with(4), fuzzy_options(0.20));
  CHECK(wide.status == Status::Ok);
  CHECK_EQ(wide.distance, std::size_t{4});
  CHECK(locate(file, search_with(1), fuzzy_options(0.0)).status == Status::NotFound);
  CHECK(locate(file, search_with(1), fuzzy_options(0.02)).status == Status::NotFound);
  CHECK(locate(file, search_with(1), fuzzy_options(-1.0)).status == Status::NotFound);
  // Trailing blanks are not counted.
  const Match blanks = locate("header\nalpha one  \nalpha two\t\nalpha six\nalpha ten\nfooter\n", search_with(3),
                              fuzzy_options());
  CHECK(blanks.status == Status::Ok);
  CHECK_EQ(blanks.distance, std::size_t{3});
}

PRAG_TEST(the_fuzzy_rung_requires_a_unique_location) {
  const std::string block_a = "  count = count + 1;\n  total = total + value;\n  if (count > limit) {\n    flush();\n";
  const std::string block_b = "  count = count + 2;\n  total = total + value;\n  if (count > limit) {\n    flush();\n";
  const std::string search = "  count = count + 3;\n  total = total + value;\n  if (count > limit) {\n    flush();\n";
  const std::string one = "void f() {\n" + block_a + "  }\n}\n";
  const std::string two = "void f() {\n" + block_a + "  }\n}\nvoid g() {\n" + block_b + "  }\n}\n";

  const Match unique = locate(one, search, fuzzy_options());
  CHECK(unique.status == Status::Ok);
  CHECK(unique.kind == MatchKind::Fuzzy);
  CHECK_EQ(unique.distance, std::size_t{1});
  CHECK_EQ(unique.first_line, 2u);

  const Match ambiguous = locate(two, search, fuzzy_options());
  CHECK(ambiguous.status == Status::Ambiguous);
  CHECK(ambiguous.kind == MatchKind::Fuzzy);
  CHECK(ambiguous.candidates >= 2u);

  // Inside the scope of one function the location is unique again.
  const std::size_t g = two.find("void g()");
  const Match scoped = locate(two, search, fuzzy_options(), g, kNone);
  CHECK(scoped.status == Status::Ok);
  CHECK_EQ(scoped.first_line, 9u);
  CHECK_EQ(scoped.last_line, 12u);
  CHECK_EQ(scoped.distance, std::size_t{1});
}

PRAG_TEST(the_fuzzy_rung_finds_windows_with_another_number_of_lines) {
  const std::string file =
      "void f() {\n"
      "  prepare_everything(context);\n"
      "  // go\n"
      "  compute_the_result(context, input);\n"
      "  store_the_result(context, output);\n"
      "  release_everything(context);\n"
      "}\n";
  // The short comment line is missing from the SEARCH text: 7 characters and
  // the line separator. The text has 136 characters, which allows 13.
  const std::string search =
      "  prepare_everything(context);\n"
      "  compute_the_result(context, input);\n"
      "  store_the_result(context, output);\n"
      "  release_everything(context);\n";
  const Match m = locate(file, search, fuzzy_options());
  CHECK(m.status == Status::Ok);
  CHECK(m.kind == MatchKind::Fuzzy);
  CHECK_EQ(m.distance, std::size_t{8});
  CHECK_EQ(m.first_line, 2u);
  CHECK_EQ(m.last_line, 6u);
}

// Every line-aligned window is examined with the reference distance, and the
// rule of the header is applied to the result: accept the best window when it
// is the only one at its distance and no window within the bound lies apart
// from another.
PRAG_TEST(the_fuzzy_scan_agrees_with_exhaustive_search) {
  Random random(7);
  const std::vector<std::string> pool = {"ab", "abc", "abd", "b", "", "cab", "abab", "ba", "  ab", "ab  "};
  std::size_t found = 0;
  std::size_t ambiguous = 0;
  std::size_t missing = 0;
  for (int round = 0; round < 1500; ++round) {
    std::vector<std::string> lines(random.between(1, 14));
    for (std::string& line : lines) line = pool[random.below(pool.size())];
    std::vector<std::string> wanted(random.between(1, 4));
    for (std::string& line : wanted) line = pool[random.below(pool.size())];
    const std::string file = with_newlines(lines);
    const std::string search = with_newlines(wanted);

    MatchOptions options = fuzzy_options(0.10 + 0.1 * static_cast<double>(random.below(4)), 1);
    const Match m = locate(file, search, options);
    if (m.kind != MatchKind::Fuzzy && m.kind != MatchKind::None) continue;  // decided on rungs 1 to 3

    // The rung compares without trailing blanks.
    std::vector<std::string> stripped = lines;
    for (std::string& line : stripped) line = std::string(detail::strip_right(line));
    std::vector<std::string> pattern_lines = wanted;
    for (std::string& line : pattern_lines) line = std::string(detail::strip_right(line));
    const std::string pattern = join(pattern_lines, 0, pattern_lines.size() - 1);
    const auto bound = static_cast<std::size_t>(options.max_distance * static_cast<double>(pattern.size()));

    struct Window {
      std::size_t first, last, distance;
    };
    std::vector<Window> windows;
    for (std::size_t first = 0; first < stripped.size(); ++first) {
      for (std::size_t last = first; last < stripped.size(); ++last) {
        const std::size_t d = reference_distance(pattern, join(stripped, first, last));
        if (d <= bound) windows.push_back({first, last, d});
      }
    }
    if (windows.empty() || bound == 0) {
      CHECK(m.status == Status::NotFound);
      CHECK(m.kind == MatchKind::None);
      ++missing;
      continue;
    }
    bool apart = false;
    for (const Window& a : windows) {
      for (const Window& b : windows) apart = apart || a.last < b.first;
    }
    const auto best = std::min_element(windows.begin(), windows.end(),
                                       [](const Window& a, const Window& b) { return a.distance < b.distance; });
    const auto ties = std::count_if(windows.begin(), windows.end(),
                                    [&](const Window& w) { return w.distance == best->distance; });
    CHECK(m.kind == MatchKind::Fuzzy);
    if (apart || ties > 1) {
      CHECK(m.status == Status::Ambiguous);
      CHECK(m.candidates >= 2u);
      ++ambiguous;
    } else {
      CHECK(m.status == Status::Ok);
      CHECK_EQ(m.distance, best->distance);
      CHECK_EQ(static_cast<std::size_t>(m.first_line), best->first + 1);
      CHECK_EQ(static_cast<std::size_t>(m.last_line), best->last + 1);
      CHECK_EQ(m.candidates, std::size_t{1});
      ++found;
    }
  }
  // The generator must reach all three outcomes often enough to mean something.
  CHECK(found > 50u);
  CHECK(ambiguous > 50u);
  CHECK(missing > 50u);
}

// ---- re-indentation ---------------------------------------------------------------------

namespace {

struct Reindented {
  bool ok = false;
  std::vector<std::string> lines;
  std::string why;
};

Reindented reindent(const std::vector<std::string_view>& search, const std::vector<std::string_view>& file,
                    std::vector<std::string> replacement) {
  Reindented out;
  out.lines = std::move(replacement);
  out.ok = detail::reindent_replacement(search, file, &out.lines, &out.why);
  return out;
}

}  // namespace

PRAG_TEST(reindent_adds_the_indentation_the_model_left_out) {
  const Reindented r = reindent({"x = 1;", "if (y) {", "  z = 2;", "}"},
                                {"    x = 1;", "    if (y) {", "      z = 2;", "    }"},
                                {"x = 1;", "", "if (y && w) {", "  z = 2;", "    deeper();", "}", "   "});
  REQUIRE(r.ok);
  const std::vector<std::string> expected = {"    x = 1;", "",           "    if (y && w) {", "      z = 2;",
                                             "        deeper();", "    }", "   "};
  CHECK(r.lines == expected);
}

PRAG_TEST(reindent_removes_indentation_the_model_added) {
  const Reindented r = reindent({"      a();", "        b();"}, {"  a();", "    b();"},
                                {"      a();", "        b();", "    shallower();", "          c();"});
  REQUIRE(r.ok);
  const std::vector<std::string> expected = {"  a();", "    b();", "shallower();", "      c();"};
  CHECK(r.lines == expected);

  // A line that has less indentation than what must be removed.
  const Reindented bad = reindent({"      a();"}, {"  a();"}, {"      a();", "  b();"});
  CHECK(!bad.ok);
  CHECK(!bad.why.empty());
  CHECK_EQ(bad.lines.size(), std::size_t{2});
  CHECK_EQ(bad.lines[1], std::string("  b();"));
}

PRAG_TEST(reindent_with_tabs) {
  // Uniform: the file has one tab more on every line.
  const Reindented uniform = reindent({"a();", "\tb();"}, {"\ta();", "\t\tb();"}, {"a();", "\tb();", "\t\tc();"});
  REQUIRE(uniform.ok);
  const std::vector<std::string> expected = {"\ta();", "\t\tb();", "\t\t\tc();"};
  CHECK(uniform.lines == expected);

  // The file is indented with tabs, the model wrote spaces. Indentation that
  // occurs in the SEARCH text is translated, anything else is refused.
  const Reindented known = reindent({"    a();", "        b();"}, {"\ta();", "\t\tb();"}, {"    a2();", "        b2();"});
  REQUIRE(known.ok);
  const std::vector<std::string> translated = {"\ta2();", "\t\tb2();"};
  CHECK(known.lines == translated);
  CHECK(!reindent({"    a();", "        b();"}, {"\ta();", "\t\tb();"}, {"    a();", "            c();"}).ok);
  CHECK(!reindent({"    a();"}, {"\ta();"}, {"        b();"}).ok);

  // Spaces after a prefix of tabs would mix the two.
  CHECK(!reindent({"a();"}, {"\ta();"}, {"a();", "  b();"}).ok);
  CHECK(reindent({"a();"}, {"\ta();"}, {"a();", "\tb();"}).ok);
}

PRAG_TEST(reindent_refuses_contradictions) {
  // The same indentation in the SEARCH text, two different ones in the file.
  const Reindented r = reindent({"  a();", "  b();"}, {"  a();", "    b();"}, {"  a();", "  b();"});
  CHECK(!r.ok);
  CHECK(!r.why.empty());
  // Different shifts on different lines, and an indentation that was not seen.
  CHECK(!reindent({"a();", "  b();"}, {"  a();", "      b();"}, {"a();", "    c();"}).ok);
  // The same lines translate as long as only known indentation is used.
  const Reindented known = reindent({"a();", "  b();"}, {"  a();", "      b();"}, {"a();", "  c();"});
  REQUIRE(known.ok);
  CHECK_EQ(known.lines[1], std::string("      c();"));
}

// ---- unified diff -----------------------------------------------------------------------

PRAG_TEST(diff_of_equal_texts_is_empty) {
  CHECK_EQ(unified_diff("", "", "a/f", "b/f"), std::string(""));
  CHECK_EQ(unified_diff("a\nb\n", "a\nb\n", "a/f", "b/f"), std::string(""));
  CHECK_EQ(unified_diff("a", "a", "a/f", "b/f"), std::string(""));
}

PRAG_TEST(diff_of_a_changed_line) {
  CHECK_EQ(unified_diff("a\nb\nc\n", "a\nB\nc\n", "a/f.c", "b/f.c"), std::string("--- a/f.c\n"
                                                                                 "+++ b/f.c\n"
                                                                                 "@@ -1,3 +1,3 @@\n"
                                                                                 " a\n"
                                                                                 "-b\n"
                                                                                 "+B\n"
                                                                                 " c\n"));
  CHECK_EQ(unified_diff("a\nb\n", "a\nx\nb\n", "old", "new"), std::string("--- old\n"
                                                                          "+++ new\n"
                                                                          "@@ -1,2 +1,3 @@\n"
                                                                          " a\n"
                                                                          "+x\n"
                                                                          " b\n"));
  CHECK_EQ(unified_diff("a\nb\nc\n", "a\nc\n", "old", "new"), std::string("--- old\n"
                                                                          "+++ new\n"
                                                                          "@@ -1,3 +1,2 @@\n"
                                                                          " a\n"
                                                                          "-b\n"
                                                                          " c\n"));
}

PRAG_TEST(diff_with_empty_files) {
  CHECK_EQ(unified_diff("", "x\ny\n", "/dev/null", "b/new.c"), std::string("--- /dev/null\n"
                                                                           "+++ b/new.c\n"
                                                                           "@@ -0,0 +1,2 @@\n"
                                                                           "+x\n"
                                                                           "+y\n"));
  CHECK_EQ(unified_diff("", "x\n", "a/f", "b/f"), std::string("--- a/f\n"
                                                              "+++ b/f\n"
                                                              "@@ -0,0 +1 @@\n"
                                                              "+x\n"));
  CHECK_EQ(unified_diff("x\n", "", "a/f", "b/f"), std::string("--- a/f\n"
                                                              "+++ b/f\n"
                                                              "@@ -1 +0,0 @@\n"
                                                              "-x\n"));
  CHECK_EQ(unified_diff("x\ny\nz\n", "", "a/f", "b/f"), std::string("--- a/f\n"
                                                                    "+++ b/f\n"
                                                                    "@@ -1,3 +0,0 @@\n"
                                                                    "-x\n"
                                                                    "-y\n"
                                                                    "-z\n"));
  CHECK_EQ(unified_diff("", "x", "a/f", "b/f"), std::string("--- a/f\n"
                                                            "+++ b/f\n"
                                                            "@@ -0,0 +1 @@\n"
                                                            "+x\n"
                                                            "\\ No newline at end of file\n"));
}

PRAG_TEST(diff_without_final_newline) {
  CHECK_EQ(unified_diff("a\nb", "a\nc", "a/f", "b/f"), std::string("--- a/f\n"
                                                                   "+++ b/f\n"
                                                                   "@@ -1,2 +1,2 @@\n"
                                                                   " a\n"
                                                                   "-b\n"
                                                                   "\\ No newline at end of file\n"
                                                                   "+c\n"
                                                                   "\\ No newline at end of file\n"));
  // Only the final newline differs.
  CHECK_EQ(unified_diff("a\nb", "a\nb\n", "a/f", "b/f"), std::string("--- a/f\n"
                                                                     "+++ b/f\n"
                                                                     "@@ -1,2 +1,2 @@\n"
                                                                     " a\n"
                                                                     "-b\n"
                                                                     "\\ No newline at end of file\n"
                                                                     "+b\n"));
  CHECK_EQ(unified_diff("a\nb\n", "a\nb", "a/f", "b/f"), std::string("--- a/f\n"
                                                                     "+++ b/f\n"
                                                                     "@@ -1,2 +1,2 @@\n"
                                                                     " a\n"
                                                                     "-b\n"
                                                                     "+b\n"
                                                                     "\\ No newline at end of file\n"));
  // The unchanged last line of both files has none.
  CHECK_EQ(unified_diff("a\nb\nc", "A\nb\nc", "a/f", "b/f"), std::string("--- a/f\n"
                                                                         "+++ b/f\n"
                                                                         "@@ -1,3 +1,3 @@\n"
                                                                         "-a\n"
                                                                         "+A\n"
                                                                         " b\n"
                                                                         " c\n"
                                                                         "\\ No newline at end of file\n"));
}

PRAG_TEST(diff_hunks_and_context) {
  std::string old_text;
  for (int i = 1; i <= 20; ++i) old_text += "l" + std::to_string(i) + "\n";
  const auto changed = [&](std::initializer_list<int> which) {
    std::string out;
    for (int i = 1; i <= 20; ++i) {
      const bool upper = std::find(which.begin(), which.end(), i) != which.end();
      out += (upper ? "L" : "l") + std::to_string(i) + "\n";
    }
    return out;
  };

  // Far apart: two hunks.
  CHECK_EQ(unified_diff(old_text, changed({2, 18}), "a/f", "b/f"), std::string("--- a/f\n"
                                                                               "+++ b/f\n"
                                                                               "@@ -1,5 +1,5 @@\n"
                                                                               " l1\n"
                                                                               "-l2\n"
                                                                               "+L2\n"
                                                                               " l3\n"
                                                                               " l4\n"
                                                                               " l5\n"
                                                                               "@@ -15,6 +15,6 @@\n"
                                                                               " l15\n"
                                                                               " l16\n"
                                                                               " l17\n"
                                                                               "-l18\n"
                                                                               "+L18\n"
                                                                               " l19\n"
                                                                               " l20\n"));
  // Six unchanged lines between two changes: the contexts touch, one hunk.
  CHECK_EQ(unified_diff(old_text, changed({2, 9}), "a/f", "b/f"), std::string("--- a/f\n"
                                                                              "+++ b/f\n"
                                                                              "@@ -1,12 +1,12 @@\n"
                                                                              " l1\n"
                                                                              "-l2\n"
                                                                              "+L2\n"
                                                                              " l3\n"
                                                                              " l4\n"
                                                                              " l5\n"
                                                                              " l6\n"
                                                                              " l7\n"
                                                                              " l8\n"
                                                                              "-l9\n"
                                                                              "+L9\n"
                                                                              " l10\n"
                                                                              " l11\n"
                                                                              " l12\n"));
  // Seven: two hunks.
  CHECK_EQ(unified_diff(old_text, changed({2, 10}), "a/f", "b/f"), std::string("--- a/f\n"
                                                                               "+++ b/f\n"
                                                                               "@@ -1,5 +1,5 @@\n"
                                                                               " l1\n"
                                                                               "-l2\n"
                                                                               "+L2\n"
                                                                               " l3\n"
                                                                               " l4\n"
                                                                               " l5\n"
                                                                               "@@ -7,7 +7,7 @@\n"
                                                                               " l7\n"
                                                                               " l8\n"
                                                                               " l9\n"
                                                                               "-l10\n"
                                                                               "+L10\n"
                                                                               " l11\n"
                                                                               " l12\n"
                                                                               " l13\n"));
  // Other amounts of context.
  CHECK_EQ(unified_diff(old_text, changed({5}), "a/f", "b/f", 1), std::string("--- a/f\n"
                                                                              "+++ b/f\n"
                                                                              "@@ -4,3 +4,3 @@\n"
                                                                              " l4\n"
                                                                              "-l5\n"
                                                                              "+L5\n"
                                                                              " l6\n"));
  CHECK_EQ(unified_diff("a\nb\nc\n", "a\nc\n", "a/f", "b/f", 0), std::string("--- a/f\n"
                                                                             "+++ b/f\n"
                                                                             "@@ -2 +1,0 @@\n"
                                                                             "-b\n"));
  CHECK_EQ(unified_diff("a\nc\n", "a\nb\nc\n", "a/f", "b/f", 0), std::string("--- a/f\n"
                                                                             "+++ b/f\n"
                                                                             "@@ -1,0 +2 @@\n"
                                                                             "+b\n"));
}

PRAG_TEST(diff_keeps_carriage_returns_and_other_bytes) {
  CHECK_EQ(unified_diff("a\r\nb\r\n", "a\r\nc\r\n", "a/f", "b/f"), std::string("--- a/f\n"
                                                                               "+++ b/f\n"
                                                                               "@@ -1,2 +1,2 @@\n"
                                                                               " a\r\n"
                                                                               "-b\r\n"
                                                                               "+c\r\n"));
  // A changed line ending is a changed line.
  CHECK_EQ(unified_diff("a\r\n", "a\n", "a/f", "b/f"), std::string("--- a/f\n"
                                                                   "+++ b/f\n"
                                                                   "@@ -1 +1 @@\n"
                                                                   "-a\r\n"
                                                                   "+a\n"));
  CHECK_EQ(unified_diff("\tcaf\xC3\xA9 \xFF\n", "\tcafe\n", "a/f", "b/f"), std::string("--- a/f\n"
                                                                                        "+++ b/f\n"
                                                                                        "@@ -1 +1 @@\n"
                                                                                        "-\tcaf\xC3\xA9 \xFF\n"
                                                                                        "+\tcafe\n"));
}

// Applies a unified diff to the old text; the result must be the new text.
namespace {

std::string apply_diff(std::string_view old_text, std::string_view diff) {
  std::vector<std::string> old_lines;
  for (std::size_t begin = 0; begin < old_text.size();) {
    const std::size_t newline = old_text.find('\n', begin);
    const std::size_t end = newline == std::string_view::npos ? old_text.size() : newline + 1;
    old_lines.emplace_back(old_text.substr(begin, end - begin));
    begin = end;
  }
  std::vector<std::string> diff_lines;
  for (std::size_t begin = 0; begin < diff.size();) {
    const std::size_t newline = diff.find('\n', begin);
    const std::size_t end = newline == std::string_view::npos ? diff.size() : newline + 1;
    diff_lines.emplace_back(diff.substr(begin, end - begin));
    begin = end;
  }
  std::string out;
  std::size_t cursor = 0;  // next old line to copy
  for (std::size_t i = 2; i < diff_lines.size(); ++i) {
    const std::string& line = diff_lines[i];
    if (line.compare(0, 2, "@@") == 0) {
      std::size_t start = static_cast<std::size_t>(std::stoul(line.substr(4)));
      const std::size_t comma = line.find(',', 4);
      const std::size_t space = line.find(' ', 4);
      const bool empty_range = comma != std::string::npos && comma < space && line.compare(comma, 3, ",0 ") == 0;
      if (!empty_range) --start;  // an empty range names the line before it
      while (cursor < start) out += old_lines[cursor++];
      continue;
    }
    const bool unterminated =
        i + 1 < diff_lines.size() && diff_lines[i + 1].compare(0, 1, "\\") == 0;
    std::string body = line.substr(1);
    if (unterminated) body.pop_back();
    if (line[0] == ' ' || line[0] == '-') ++cursor;
    if (line[0] == ' ' || line[0] == '+') out += body;
  }
  while (cursor < old_lines.size()) out += old_lines[cursor++];
  return out;
}

}  // namespace

PRAG_TEST(diff_applied_to_the_old_text_gives_the_new_text) {
  Random random(99);
  const std::vector<std::string> pool = {"a", "b", "c", "", "int x;", "}", "{", "return;"};
  std::size_t checked = 0;
  for (int round = 0; round < 1500; ++round) {
    std::vector<std::string> lines(random.below(30));
    for (std::string& line : lines) line = pool[random.below(pool.size())];
    std::vector<std::string> edited = lines;
    for (std::size_t k = random.below(6); k-- > 0;) {
      const std::size_t at = random.below(edited.size() + 1);
      const std::size_t what = random.below(3);
      if (what == 0) edited.insert(edited.begin() + static_cast<std::ptrdiff_t>(at), pool[random.below(pool.size())]);
      if (what == 1 && at < edited.size()) edited.erase(edited.begin() + static_cast<std::ptrdiff_t>(at));
      if (what == 2 && at < edited.size()) edited[at] = "changed " + std::to_string(round);
    }
    std::string old_text = with_newlines(lines);
    std::string new_text = with_newlines(edited);
    if (random.chance(0.2) && !old_text.empty()) old_text.pop_back();
    if (random.chance(0.2) && !new_text.empty()) new_text.pop_back();
    const std::size_t context = random.below(5);
    const std::string diff = unified_diff(old_text, new_text, "a/f", "b/f", context);
    if (old_text == new_text) {
      CHECK(diff.empty());
      continue;
    }
    CHECK(diff.compare(0, 16, "--- a/f\n+++ b/f\n") == 0);
    const std::string applied = apply_diff(old_text, diff);
    if (applied != new_text) {
      std::fprintf(stderr, "  round %d: old\n%s\nnew\n%s\ndiff\n%s\n", round, old_text.c_str(), new_text.c_str(),
                   diff.c_str());
    }
    CHECK(applied == new_text);
    ++checked;
  }
  CHECK(checked > 1000u);
}

PRAG_TEST(diff_of_very_different_texts_is_still_correct) {
  // More differing lines than the search for a shortest script allows.
  std::string old_text;
  std::string new_text;
  for (int i = 0; i < 3000; ++i) {
    old_text += "old line " + std::to_string(i) + "\n";
    new_text += "new line " + std::to_string(i) + "\n";
  }
  old_text = "first\n" + old_text + "last\n";
  new_text = "first\n" + new_text + "last\n";
  const std::string diff = unified_diff(old_text, new_text, "a/f", "b/f");
  CHECK(diff.find("@@ -1,3002 +1,3002 @@\n") != std::string::npos);
  CHECK(apply_diff(old_text, diff) == new_text);
}
