// Parser for edit blocks in model output, and the grammar that constrains the
// output to that form.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "prag/edit/edit.hpp"

using namespace prag::edit;

namespace {

// ---- a small interpreter for GBNF ------------------------------------------------
//
// Reads the subset of the llama.cpp grammar syntax that edit_grammar() uses
// (literals, character classes, groups, alternatives, rule references and the
// repetitions ? * + {m,n}) with the lexical rules of the llama.cpp parser, and
// decides whether the root rule derives a given text. It checks the language
// of the grammar; that llama.cpp itself loads the grammar is checked elsewhere.
namespace gbnf {

struct Node;
using Sequence = std::vector<Node>;
using Alternatives = std::vector<Sequence>;

struct Node {
  enum class Kind { Literal, Class, Any, Rule, Group };
  Kind kind = Kind::Literal;
  std::string literal;
  std::vector<std::pair<unsigned char, unsigned char>> ranges;
  bool negated = false;
  std::string rule;
  std::shared_ptr<Alternatives> group;
  std::size_t min = 1;
  std::size_t max = 1;  // kUnbounded for * + {m,}
};

constexpr std::size_t kUnbounded = static_cast<std::size_t>(-1);

class Grammar {
 public:
  bool parse(std::string_view text) {
    text_ = std::string(text);
    pos_ = 0;
    error_.clear();
    rules_.clear();
    space(true);
    while (ok() && pos_ < text_.size()) rule();
    if (!ok()) return false;
    if (!rules_.count("root")) return fail("no root rule");
    for (const std::string& name : referenced_) {
      if (!rules_.count(name)) return fail("undefined rule " + name);
    }
    return true;
  }

  bool accepts(std::string_view input) const {
    memo_.clear();
    const std::set<std::size_t> ends = match_rule("root", input, 0);
    return ends.count(input.size()) != 0;
  }

  const std::string& error() const { return error_; }
  std::size_t rule_count() const { return rules_.size(); }

 private:
  bool ok() const { return error_.empty(); }
  bool fail(const std::string& message) {
    if (error_.empty()) error_ = message + " at offset " + std::to_string(pos_);
    return false;
  }
  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
  static bool is_word(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
  }

  void space(bool newline_ok) {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == '#') {
        while (pos_ < text_.size() && text_[pos_] != '\n' && text_[pos_] != '\r') ++pos_;
      } else if (c == ' ' || c == '\t' || (newline_ok && (c == '\n' || c == '\r'))) {
        ++pos_;
      } else {
        break;
      }
    }
  }

  std::string name() {
    const std::size_t begin = pos_;
    while (is_word(peek())) ++pos_;
    if (pos_ == begin) fail("expecting a name");
    return text_.substr(begin, pos_ - begin);
  }

  unsigned char character() {
    if (peek() != '\\') return static_cast<unsigned char>(text_[pos_++]);
    const char e = pos_ + 1 < text_.size() ? text_[pos_ + 1] : '\0';
    pos_ += 2;
    switch (e) {
      case 'n':
        return '\n';
      case 't':
        return '\t';
      case 'r':
        return '\r';
      case '\\':
      case '"':
      case '[':
      case ']':
      case '-':
        return static_cast<unsigned char>(e);
      default:
        fail("unknown escape");
        return 0;
    }
  }

  void rule() {
    const std::string rule_name = name();
    space(false);
    if (text_.compare(pos_, 3, "::=") != 0) {
      fail("expecting ::=");
      return;
    }
    pos_ += 3;
    space(true);
    Alternatives body = alternatives(false);
    if (!ok()) return;
    if (peek() == '\r') {
      pos_ += pos_ + 1 < text_.size() && text_[pos_ + 1] == '\n' ? 2 : 1;
    } else if (peek() == '\n') {
      ++pos_;
    } else if (pos_ < text_.size()) {
      fail("expecting a newline or the end");
      return;
    }
    space(true);
    if (rules_.count(rule_name)) {
      fail("rule " + rule_name + " is defined twice");
      return;
    }
    rules_[rule_name] = std::move(body);
  }

  Alternatives alternatives(bool nested) {
    Alternatives out;
    out.push_back(sequence(nested));
    while (ok() && peek() == '|') {
      ++pos_;
      space(true);
      out.push_back(sequence(nested));
    }
    return out;
  }

  void repeat(Sequence* sequence, std::size_t min, std::size_t max) {
    if (sequence->empty()) {
      fail("repetition without an operand");
      return;
    }
    sequence->back().min = min;
    sequence->back().max = max;
  }

  std::size_t integer() {
    std::size_t value = 0;
    if (peek() < '0' || peek() > '9') fail("expecting an integer");
    while (peek() >= '0' && peek() <= '9') value = value * 10 + static_cast<std::size_t>(text_[pos_++] - '0');
    return value;
  }

  Sequence sequence(bool nested) {
    Sequence out;
    while (ok() && pos_ < text_.size()) {
      const char c = peek();
      if (c == '"') {
        ++pos_;
        Node node;
        node.kind = Node::Kind::Literal;
        while (ok() && peek() != '"') {
          if (pos_ >= text_.size()) {
            fail("unterminated literal");
            break;
          }
          node.literal.push_back(static_cast<char>(character()));
        }
        ++pos_;
        space(nested);
        out.push_back(std::move(node));
      } else if (c == '[') {
        ++pos_;
        Node node;
        node.kind = Node::Kind::Class;
        if (peek() == '^') {
          node.negated = true;
          ++pos_;
        }
        while (ok() && peek() != ']') {
          if (pos_ >= text_.size()) {
            fail("unterminated character class");
            break;
          }
          const unsigned char low = character();
          unsigned char high = low;
          if (peek() == '-' && pos_ + 1 < text_.size() && text_[pos_ + 1] != ']') {
            ++pos_;
            high = character();
          }
          node.ranges.emplace_back(low, high);
        }
        ++pos_;
        space(nested);
        out.push_back(std::move(node));
      } else if (c == '(') {
        ++pos_;
        space(true);
        Node node;
        node.kind = Node::Kind::Group;
        node.group = std::make_shared<Alternatives>(alternatives(true));
        if (peek() != ')') {
          fail("expecting )");
          break;
        }
        ++pos_;
        space(nested);
        out.push_back(std::move(node));
      } else if (c == '.') {
        ++pos_;
        space(nested);
        Node node;
        node.kind = Node::Kind::Any;
        out.push_back(std::move(node));
      } else if (c == '*' || c == '+' || c == '?') {
        ++pos_;
        space(nested);
        repeat(&out, c == '+' ? 1 : 0, c == '?' ? 1 : kUnbounded);
      } else if (c == '{') {
        ++pos_;
        space(nested);
        const std::size_t min = integer();
        std::size_t max = min;
        space(nested);
        if (peek() == ',') {
          ++pos_;
          space(nested);
          max = peek() >= '0' && peek() <= '9' ? integer() : kUnbounded;
          space(nested);
        }
        if (peek() != '}') {
          fail("expecting }");
          break;
        }
        ++pos_;
        space(nested);
        repeat(&out, min, max);
      } else if (c == '<' || c == '!') {
        fail("token references are not used by this grammar");
      } else if (is_word(c)) {
        Node node;
        node.kind = Node::Kind::Rule;
        node.rule = name();
        referenced_.insert(node.rule);
        space(nested);
        out.push_back(std::move(node));
      } else {
        break;
      }
    }
    return out;
  }

  // ---- matching: the set of positions at which a construct can end ----

  std::set<std::size_t> match_rule(const std::string& rule_name, std::string_view input, std::size_t at) const {
    const auto key = std::make_pair(rule_name, at);
    const auto known = memo_.find(key);
    if (known != memo_.end()) return known->second;
    std::set<std::size_t> ends = match_alternatives(rules_.at(rule_name), input, at);
    memo_[key] = ends;
    return ends;
  }

  std::set<std::size_t> match_alternatives(const Alternatives& alternatives, std::string_view input,
                                           std::size_t at) const {
    std::set<std::size_t> ends;
    for (const Sequence& sequence : alternatives) {
      std::set<std::size_t> current{at};
      for (const Node& node : sequence) {
        std::set<std::size_t> next;
        for (const std::size_t p : current) {
          for (const std::size_t e : match_repeated(node, input, p)) next.insert(e);
        }
        current = std::move(next);
        if (current.empty()) break;
      }
      ends.insert(current.begin(), current.end());
    }
    return ends;
  }

  std::set<std::size_t> match_repeated(const Node& node, std::string_view input, std::size_t at) const {
    std::set<std::size_t> ends;
    if (node.min == 0) ends.insert(at);
    std::set<std::size_t> current{at};
    for (std::size_t count = 1; count <= node.max && !current.empty(); ++count) {
      std::set<std::size_t> next;
      for (const std::size_t p : current) {
        for (const std::size_t e : match_once(node, input, p)) next.insert(e);
      }
      if (count >= node.min) ends.insert(next.begin(), next.end());
      if (next == current) break;  // no progress: further rounds add nothing
      current = std::move(next);
    }
    return ends;
  }

  std::set<std::size_t> match_once(const Node& node, std::string_view input, std::size_t at) const {
    switch (node.kind) {
      case Node::Kind::Literal:
        if (input.compare(at, node.literal.size(), node.literal) == 0 && at + node.literal.size() <= input.size()) {
          return {at + node.literal.size()};
        }
        return {};
      case Node::Kind::Any:
        if (at < input.size()) return {at + 1};
        return {};
      case Node::Kind::Class: {
        if (at >= input.size()) return {};
        const auto c = static_cast<unsigned char>(input[at]);
        bool inside = false;
        for (const auto& [low, high] : node.ranges) inside = inside || (c >= low && c <= high);
        if (inside != node.negated) return {at + 1};
        return {};
      }
      case Node::Kind::Rule:
        return match_rule(node.rule, input, at);
      case Node::Kind::Group:
        return match_alternatives(*node.group, input, at);
    }
    return {};
  }

  std::string text_;
  std::size_t pos_ = 0;
  std::string error_;
  std::map<std::string, Alternatives> rules_;
  std::set<std::string> referenced_;
  mutable std::map<std::pair<std::string, std::size_t>, std::set<std::size_t>> memo_;
};

}  // namespace gbnf

const char* const kSearchBlock =
    "src/alloc.c\n"
    "<<<<<<< SEARCH\n"
    "  return p;\n"
    "=======\n"
    "  return q;\n"
    ">>>>>>> REPLACE\n";

const char* const kScopedBlock =
    "src/alloc.c\n"
    "@@ mi_heap_malloc\n"
    "<<<<<<< SEARCH\n"
    "  return p;\n"
    "=======\n"
    "  return q;\n"
    ">>>>>>> REPLACE\n";

const char* const kUnitBlock =
    "src/heap.c\n"
    "<<<<<<< UNIT mi_heap_collect\n"
    "void mi_heap_collect(mi_heap_t* heap, bool force) {\n"
    "  mi_heap_collect_ex(heap, force ? MI_FORCE : MI_NORMAL);\n"
    "}\n"
    ">>>>>>> UNIT\n";

}  // namespace

// ---- parser ------------------------------------------------------------------------

PRAG_TEST(parses_a_search_replace_block) {
  const ParseResult r = parse_edit_blocks(kSearchBlock);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 1u);
  CHECK(r.blocks[0].kind == EditKind::SearchReplace);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[0].scope, std::string(""));
  CHECK_EQ(r.blocks[0].search, std::string("  return p;\n"));
  CHECK_EQ(r.blocks[0].replace, std::string("  return q;\n"));
}

PRAG_TEST(parses_the_scope_line) {
  const ParseResult r = parse_edit_blocks(kScopedBlock);
  REQUIRE(r.blocks.size() == 1u);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[0].scope, std::string("mi_heap_malloc"));

  for (const char* scope : {"@@ mi_heap_malloc @@", "@@mi_heap_malloc", "@@ mi_heap_malloc()", "@@  `mi_heap_malloc`"}) {
    const std::string text = std::string("src/alloc.c\n") + scope + "\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n";
    const ParseResult v = parse_edit_blocks(text);
    REQUIRE(v.blocks.size() == 1u);
    CHECK_EQ(v.blocks[0].scope, std::string("mi_heap_malloc"));
    CHECK_EQ(v.blocks[0].path, std::string("src/alloc.c"));
  }
}

PRAG_TEST(parses_a_unit_rewrite) {
  const ParseResult r = parse_edit_blocks(kUnitBlock);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 1u);
  CHECK(r.blocks[0].kind == EditKind::UnitRewrite);
  CHECK_EQ(r.blocks[0].path, std::string("src/heap.c"));
  CHECK_EQ(r.blocks[0].scope, std::string("mi_heap_collect"));
  CHECK_EQ(r.blocks[0].search, std::string(""));
  CHECK_EQ(r.blocks[0].replace, std::string("void mi_heap_collect(mi_heap_t* heap, bool force) {\n"
                                            "  mi_heap_collect_ex(heap, force ? MI_FORCE : MI_NORMAL);\n"
                                            "}\n"));
}

PRAG_TEST(tolerates_prose_and_markdown_fences) {
  const std::string text =
      "Sure! The function returns the wrong pointer. Here is the fix:\n"
      "\n"
      "```c\n"
      "src/alloc.c\n"
      "<<<<<<< SEARCH\n"
      "  return p;\n"
      "=======\n"
      "  return q;\n"
      ">>>>>>> REPLACE\n"
      "```\n"
      "\n"
      "And the header, with the file name outside of the fence:\n"
      "\n"
      "include/mimalloc.h\n"
      "```c\n"
      "<<<<<<< SEARCH\n"
      "#define MI_VERSION 1\n"
      "=======\n"
      "#define MI_VERSION 2\n"
      ">>>>>>> REPLACE\n"
      "```\n"
      "\n"
      "Let me know if anything else is needed.\n";
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 2u);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[0].search, std::string("  return p;\n"));
  CHECK_EQ(r.blocks[0].replace, std::string("  return q;\n"));
  CHECK_EQ(r.blocks[1].path, std::string("include/mimalloc.h"));
  CHECK_EQ(r.blocks[1].search, std::string("#define MI_VERSION 1\n"));
  CHECK_EQ(r.blocks[1].replace, std::string("#define MI_VERSION 2\n"));
}

PRAG_TEST(reads_crlf_output_like_lf_output) {
  std::string text;
  for (const char* p = kScopedBlock; *p; ++p) {
    if (*p == '\n') text.push_back('\r');
    text.push_back(*p);
  }
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 1u);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[0].scope, std::string("mi_heap_malloc"));
  CHECK_EQ(r.blocks[0].search, std::string("  return p;\n"));
  CHECK_EQ(r.blocks[0].replace, std::string("  return q;\n"));
}

PRAG_TEST(accepts_five_to_nine_marker_characters) {
  for (std::size_t n = 3; n <= 11; ++n) {
    const std::string text = "a/b.c\n" + std::string(n, '<') + " SEARCH\nx\n" + std::string(n, '=') + "\ny\n" +
                             std::string(n, '>') + " REPLACE\n";
    const ParseResult r = parse_edit_blocks(text);
    if (n >= 5 && n <= 9) {
      REQUIRE(r.blocks.size() == 1u);
      CHECK_EQ(r.blocks[0].search, std::string("x\n"));
      CHECK_EQ(r.blocks[0].replace, std::string("y\n"));
      CHECK(r.warnings.empty());
    } else {
      CHECK(r.blocks.empty());
    }
  }
  // The three markers of one block need not have the same length.
  const ParseResult mixed = parse_edit_blocks("a/b.c\n<<<<< SEARCH\nx\n=========\ny\n>>>>>>> REPLACE\n");
  REQUIRE(mixed.blocks.size() == 1u);
  CHECK_EQ(mixed.blocks[0].replace, std::string("y\n"));
  // Markers are recognised whatever the case of the keyword, and with blanks after them.
  const ParseResult lower = parse_edit_blocks("a/b.c\n<<<<<<< search  \nx\n=======\t\ny\n>>>>>>> Replace \n");
  REQUIRE(lower.blocks.size() == 1u);
  CHECK_EQ(lower.blocks[0].search, std::string("x\n"));
}

PRAG_TEST(a_block_without_path_is_skipped_or_inherits_the_previous_path) {
  const ParseResult none = parse_edit_blocks("Here is the change:\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n");
  CHECK(none.blocks.empty());
  REQUIRE(none.warnings.size() == 1u);
  CHECK(none.warnings[0].find("no file name") != std::string::npos);

  const std::string text = std::string(kSearchBlock) +
                           "\n"
                           "<<<<<<< SEARCH\n"
                           "  int x;\n"
                           "=======\n"
                           "  long x;\n"
                           ">>>>>>> REPLACE\n";
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 2u);
  CHECK_EQ(r.blocks[1].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[1].search, std::string("  int x;\n"));

  // The end marker of the previous block is never taken for a file name, and a
  // sentence that mentions no file is not one either.
  const ParseResult prose = parse_edit_blocks(std::string(kSearchBlock) +
                                              "Then the second change\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n");
  REQUIRE(prose.blocks.size() == 2u);
  CHECK_EQ(prose.blocks[1].path, std::string("src/alloc.c"));
}

PRAG_TEST(an_unterminated_block_is_skipped) {
  const ParseResult cut = parse_edit_blocks("src/alloc.c\n<<<<<<< SEARCH\n  return p;\n=======\n  return q;\n");
  CHECK(cut.blocks.empty());
  REQUIRE(cut.warnings.size() == 1u);
  CHECK(cut.warnings[0].find("not terminated") != std::string::npos);

  const ParseResult no_divider = parse_edit_blocks("src/alloc.c\n<<<<<<< SEARCH\n  return p;\n");
  CHECK(no_divider.blocks.empty());
  CHECK_EQ(no_divider.warnings.size(), std::size_t{1});

  const ParseResult unit = parse_edit_blocks("src/alloc.c\n<<<<<<< UNIT f\nint f(void) { return 1; }\n");
  CHECK(unit.blocks.empty());
  CHECK_EQ(unit.warnings.size(), std::size_t{1});

  // A block that is cut short does not swallow the one after it.
  const std::string text =
      "src/alloc.c\n"
      "<<<<<<< SEARCH\n"
      "  return p;\n"
      "=======\n"
      "  return q;\n"
      "src/heap.c\n"
      "<<<<<<< SEARCH\n"
      "  int a;\n"
      "=======\n"
      "  int b;\n"
      ">>>>>>> REPLACE\n";
  const ParseResult r = parse_edit_blocks(text);
  REQUIRE(r.blocks.size() == 1u);
  CHECK_EQ(r.blocks[0].path, std::string("src/heap.c"));
  CHECK_EQ(r.blocks[0].search, std::string("  int a;\n"));
  CHECK_EQ(r.blocks[0].replace, std::string("  int b;\n"));
  CHECK_EQ(r.warnings.size(), std::size_t{1});
}

PRAG_TEST(malformed_blocks_are_reported_and_skipped) {
  // End marker before the divider.
  const ParseResult a = parse_edit_blocks("a/b.c\n<<<<<<< SEARCH\nx\n>>>>>>> REPLACE\n");
  CHECK(a.blocks.empty());
  CHECK_EQ(a.warnings.size(), std::size_t{1});
  // Two dividers: it cannot be known which one separates the sections.
  const ParseResult b = parse_edit_blocks("a/b.c\n<<<<<<< SEARCH\nx\n=======\ny\n=======\nz\n>>>>>>> REPLACE\n" +
                                          std::string(kSearchBlock));
  REQUIRE(b.blocks.size() == 1u);
  CHECK_EQ(b.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(b.warnings.size(), std::size_t{1});
  // The two forms mixed up.
  const ParseResult c = parse_edit_blocks("a/b.c\n<<<<<<< UNIT f\nint f;\n>>>>>>> REPLACE\n");
  CHECK(c.blocks.empty());
  CHECK_EQ(c.warnings.size(), std::size_t{1});
  // A unit without a name.
  const ParseResult d = parse_edit_blocks("a/b.c\n<<<<<<< UNIT\nint f;\n>>>>>>> UNIT\n");
  CHECK(d.blocks.empty());
  CHECK_EQ(d.warnings.size(), std::size_t{1});
  // An end marker on its own.
  const ParseResult e = parse_edit_blocks("text\n>>>>>>> REPLACE\nmore text\n");
  CHECK(e.blocks.empty());
  CHECK_EQ(e.warnings.size(), std::size_t{1});
  // Nothing at all.
  const ParseResult f = parse_edit_blocks("");
  CHECK(f.blocks.empty());
  CHECK(f.warnings.empty());
  const ParseResult g = parse_edit_blocks("The code is correct as it is.\n");
  CHECK(g.blocks.empty());
  CHECK(g.warnings.empty());
}

PRAG_TEST(parses_several_blocks_of_both_kinds) {
  const std::string text = std::string("I will make three changes.\n\n") + kSearchBlock + "\n" + kUnitBlock + "\n" +
                           kScopedBlock + "\nDone.\n";
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 3u);
  CHECK(r.blocks[0].kind == EditKind::SearchReplace);
  CHECK(r.blocks[1].kind == EditKind::UnitRewrite);
  CHECK(r.blocks[2].kind == EditKind::SearchReplace);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[1].path, std::string("src/heap.c"));
  CHECK_EQ(r.blocks[2].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[1].scope, std::string("mi_heap_collect"));
  CHECK_EQ(r.blocks[2].scope, std::string("mi_heap_malloc"));
}

PRAG_TEST(removes_decoration_from_the_path_line) {
  const char* const decorated[] = {
      "**src/alloc.c**", "`src/alloc.c`",      "### src/alloc.c",       "File: src/alloc.c", "- `src/alloc.c`:",
      "src/alloc.c:",    "**File:** `src/alloc.c`", "  src/alloc.c  ", "// src/alloc.c",    "\"src/alloc.c\"",
  };
  for (const char* line : decorated) {
    const std::string text = std::string(line) + "\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n";
    const ParseResult r = parse_edit_blocks(text);
    REQUIRE(r.blocks.size() == 1u);
    CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  }
  // Separators and blanks inside the name are kept as written.
  const ParseResult windows = parse_edit_blocks("src\\prim\\windows\\prim.c\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n");
  REQUIRE(windows.blocks.size() == 1u);
  CHECK_EQ(windows.blocks[0].path, std::string("src\\prim\\windows\\prim.c"));
  const ParseResult blanks = parse_edit_blocks("Demo repo/mimalloc/readme.md\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n");
  REQUIRE(blanks.blocks.size() == 1u);
  CHECK_EQ(blanks.blocks[0].path, std::string("Demo repo/mimalloc/readme.md"));
  const ParseResult bare = parse_edit_blocks("alloc.c\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n");
  REQUIRE(bare.blocks.size() == 1u);
  CHECK_EQ(bare.blocks[0].path, std::string("alloc.c"));
}

PRAG_TEST(keeps_the_payload_byte_for_byte) {
  const std::string text =
      "a/b.c\n"
      "<<<<<<< SEARCH\n"
      "\tif (x) {   \n"
      "\n"
      "    y = \"caf\xC3\xA9\";  // <<< not a marker\n"
      "=== neither\n"
      "=======\n"
      ">>>>>>> REPLACE\n";
  const ParseResult r = parse_edit_blocks(text);
  REQUIRE(r.blocks.size() == 1u);
  CHECK_EQ(r.blocks[0].search, std::string("\tif (x) {   \n\n    y = \"caf\xC3\xA9\";  // <<< not a marker\n=== neither\n"));
  CHECK_EQ(r.blocks[0].replace, std::string(""));

  // An empty SEARCH section asks for a new file.
  const ParseResult create = parse_edit_blocks("new/file.c\n<<<<<<< SEARCH\n=======\nint x;\n>>>>>>> REPLACE\n");
  REQUIRE(create.blocks.size() == 1u);
  CHECK_EQ(create.blocks[0].search, std::string(""));
  CHECK_EQ(create.blocks[0].replace, std::string("int x;\n"));

  // The unit form has no divider, so a row of '=' belongs to the text.
  const ParseResult unit = parse_edit_blocks("a/b.c\n<<<<<<< UNIT f\n/*\n=======\n*/\nint f;\n>>>>>>> UNIT\n");
  REQUIRE(unit.blocks.size() == 1u);
  CHECK_EQ(unit.blocks[0].replace, std::string("/*\n=======\n*/\nint f;\n"));
}

PRAG_TEST(prose_with_rows_of_equal_signs_is_not_a_block) {
  const std::string text = std::string("Summary\n=======\n\nThe fix follows.\n\n") + kSearchBlock;
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 1u);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
}

PRAG_TEST(names_of_the_enumerations) {
  CHECK_EQ(std::string(to_string(EditKind::SearchReplace)), std::string("search_replace"));
  CHECK_EQ(std::string(to_string(EditKind::UnitRewrite)), std::string("unit_rewrite"));
  CHECK_EQ(std::string(to_string(MatchKind::None)), std::string("none"));
  CHECK_EQ(std::string(to_string(MatchKind::Exact)), std::string("exact"));
  CHECK_EQ(std::string(to_string(MatchKind::TrailingSpace)), std::string("trailing_space"));
  CHECK_EQ(std::string(to_string(MatchKind::Indentation)), std::string("indentation"));
  CHECK_EQ(std::string(to_string(MatchKind::Fuzzy)), std::string("fuzzy"));
  CHECK_EQ(std::string(to_string(MatchKind::Unit)), std::string("unit"));
  // The names of DESIGN.md section 9.
  CHECK_EQ(std::string(to_string(Status::Ok)), std::string("ok"));
  CHECK_EQ(std::string(to_string(Status::NotFound)), std::string("not_found"));
  CHECK_EQ(std::string(to_string(Status::Ambiguous)), std::string("ambiguous"));
  CHECK_EQ(std::string(to_string(Status::ScopeNotFound)), std::string("scope_not_found"));
  CHECK_EQ(std::string(to_string(Status::OutsideRoots)), std::string("outside_roots"));
  CHECK_EQ(std::string(to_string(Status::FileMissing)), std::string("file_missing"));
  CHECK_EQ(std::string(to_string(Status::Overlap)), std::string("overlap"));
  CHECK_EQ(std::string(to_string(Status::SyntaxRegression)), std::string("syntax_regression"));
  CHECK_EQ(std::string(to_string(Status::IoError)), std::string("io_error"));
  CHECK_EQ(std::string(to_string(Status::Conflict)), std::string("conflict"));
}

// ---- grammar -------------------------------------------------------------------------

PRAG_TEST(the_grammar_is_well_formed) {
  gbnf::Grammar grammar;
  const bool parsed = grammar.parse(edit_grammar());
  if (!parsed) std::fprintf(stderr, "  grammar: %s\n", grammar.error().c_str());
  REQUIRE(parsed);
  CHECK_EQ(grammar.rule_count(), std::size_t{8});
  // Plain ASCII, and no tabs or carriage returns that an editor could have added.
  for (const char c : edit_grammar()) CHECK(c == '\n' || (c >= 0x20 && c < 0x7F));
}

PRAG_TEST(the_grammar_accepts_the_documented_forms) {
  gbnf::Grammar grammar;
  REQUIRE(grammar.parse(edit_grammar()));
  const std::vector<std::string> accepted = {
      kSearchBlock,
      kScopedBlock,
      kUnitBlock,
      std::string("I change the return value and the collector.\n") + kSearchBlock + "\n" + kUnitBlock,
      std::string(kUnitBlock) + kScopedBlock + kSearchBlock,
      // A new file, a deletion, and payload lines that begin with marker characters.
      "new/file.c\n<<<<<<< SEARCH\n=======\nint x;\n>>>>>>> REPLACE\n",
      "a.c\n<<<<<<< SEARCH\nint x;\n=======\n>>>>>>> REPLACE\n",
      "a.c\n<<<<<<< SEARCH\n<< 2;\n=\n>>>> x\n\n== y\n=======\n<\n>>>>>>> REPLACE\n",
      "./Makefile\n<<<<<<< UNIT all\nall: lib\n>>>>>>> UNIT\n",
      "src/a.cpp\n<<<<<<< UNIT Heap::~Heap\nHeap::~Heap() {}\n>>>>>>> UNIT\n",
  };
  for (const std::string& text : accepted) {
    const bool ok = grammar.accepts(text);
    if (!ok) std::fprintf(stderr, "  not accepted:\n%s\n", text.c_str());
    CHECK(ok);
  }
}

PRAG_TEST(the_grammar_rejects_everything_else) {
  gbnf::Grammar grammar;
  REQUIRE(grammar.parse(edit_grammar()));
  const std::string block = kSearchBlock;
  const std::vector<std::string> rejected = {
      "",
      "\n",
      "Only prose.\n",
      "First line.\nSecond line.\n" + block,                                     // two lines of prose
      block + "Trailing remarks.\n",                                             // text after the last block
      "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n",                         // no file name
      "src/alloc.c\n<<<<<<< SEARCH\n  return p;\n=======\n  return q;\n",         // cut short
      block.substr(0, block.size() - 1),                                         // no newline after the end marker
      "a.c\n<<<<<<< SEARCH\nx\n=======\ny\n=======\nz\n>>>>>>> REPLACE\n",         // two dividers
      "a.c\n<<<<<<< SEARCH\nx\n>>>>>>> REPLACE\n",                               // no divider
      "a.c\n<<<<<<< UNIT\nint f;\n>>>>>>> UNIT\n",                                // unit without name
      "a.c\n<<<<<<< UNIT f\n<<<<<<< SEARCH\n>>>>>>> UNIT\n",                       // marker inside the payload
      "a.c\n<<<<<<< UNIT f\nint f;\n>>>>>>> REPLACE\n",                           // forms mixed up
      "a.c\n<<<<< SEARCH\nx\n=====\ny\n>>>>> REPLACE\n",                          // the grammar writes seven characters
      "```c\n" + block + "```\n",                                                // fences
      "my file.c\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n",               // blank in the file name
  };
  for (const std::string& text : rejected) {
    const bool ok = grammar.accepts(text);
    if (ok) std::fprintf(stderr, "  accepted:\n%s\n", text.c_str());
    CHECK(!ok);
  }
}

PRAG_TEST(what_the_grammar_accepts_the_parser_reads) {
  gbnf::Grammar grammar;
  REQUIRE(grammar.parse(edit_grammar()));
  const std::string text = std::string("Two changes follow.\n") + kScopedBlock + "\n" + kUnitBlock +
                           "src/new.c\n<<<<<<< SEARCH\n=======\n== x\n>>>>>>> REPLACE\n";
  REQUIRE(grammar.accepts(text));
  const ParseResult r = parse_edit_blocks(text);
  CHECK(r.warnings.empty());
  REQUIRE(r.blocks.size() == 3u);
  CHECK_EQ(r.blocks[0].path, std::string("src/alloc.c"));
  CHECK_EQ(r.blocks[0].scope, std::string("mi_heap_malloc"));
  CHECK_EQ(r.blocks[0].search, std::string("  return p;\n"));
  CHECK_EQ(r.blocks[0].replace, std::string("  return q;\n"));
  CHECK(r.blocks[1].kind == EditKind::UnitRewrite);
  CHECK_EQ(r.blocks[1].path, std::string("src/heap.c"));
  CHECK_EQ(r.blocks[1].scope, std::string("mi_heap_collect"));
  CHECK_EQ(r.blocks[2].path, std::string("src/new.c"));
  CHECK_EQ(r.blocks[2].search, std::string(""));
  CHECK_EQ(r.blocks[2].replace, std::string("== x\n"));
}
