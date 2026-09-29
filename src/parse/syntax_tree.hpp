// Thin owning wrapper around the tree-sitter parser, private to the parse module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <tree_sitter/api.h>

namespace prag::parse::detail {

// Grammar for a language key ("c", "cpp", "python"); nullptr otherwise.
const TSLanguage* grammar_for(std::string_view language);

// A parsed file. Every call creates a parser of its own, so trees can be built
// from any number of threads at once; the grammars themselves are constant data.
class SyntaxTree {
 public:
  SyntaxTree() = default;
  SyntaxTree(std::string_view language, std::string_view text);
  ~SyntaxTree();
  SyntaxTree(SyntaxTree&& other) noexcept : tree_(other.tree_) { other.tree_ = nullptr; }
  SyntaxTree& operator=(SyntaxTree&& other) noexcept;
  SyntaxTree(const SyntaxTree&) = delete;
  SyntaxTree& operator=(const SyntaxTree&) = delete;

  bool valid() const noexcept { return tree_ != nullptr; }
  TSNode root() const { return ts_tree_root_node(tree_); }
  bool has_errors() const { return valid() && ts_node_has_error(root()); }
  std::size_t count_errors() const;  // ERROR and MISSING nodes

 private:
  TSTree* tree_ = nullptr;
};

// Depth-first traversal without recursion. `visit(node)` returns whether the
// children of the node should be visited.
template <class Visit>
void walk(TSNode start, Visit&& visit) {
  if (ts_node_is_null(start)) return;
  TSTreeCursor cursor = ts_tree_cursor_new(start);
  bool done = false;
  while (!done) {
    const bool descend = visit(ts_tree_cursor_current_node(&cursor));
    if (descend && ts_tree_cursor_goto_first_child(&cursor)) continue;
    while (!ts_tree_cursor_goto_next_sibling(&cursor)) {
      if (!ts_tree_cursor_goto_parent(&cursor)) {
        done = true;
        break;
      }
    }
  }
  ts_tree_cursor_delete(&cursor);
}

inline std::string_view node_text(std::string_view text, TSNode node) {
  const std::uint32_t begin = ts_node_start_byte(node);
  const std::uint32_t end = ts_node_end_byte(node);
  if (begin > end || end > text.size()) return {};
  return text.substr(begin, end - begin);
}

inline TSNode field(TSNode node, std::string_view name) {
  return ts_node_child_by_field_name(node, name.data(), static_cast<std::uint32_t>(name.size()));
}

// What the syntax tree says about one function definition of a C or C++ file.
struct TreeFunction {
  std::uint32_t begin = 0;       // first byte of the definition as tree-sitter delimits it
  std::uint32_t end = 0;
  std::uint32_t body_begin = 0;  // the opening brace of the body
  std::string name;              // declarator name without white space, "" when it could not be read
  bool clean = false;            // no ERROR or MISSING node inside the definition
};

struct TreeCall {
  std::uint32_t offset = 0;  // of the called identifier
  std::string_view name;
};

struct TreeFacts {
  std::vector<TreeFunction> functions;  // ascending body_begin
  std::vector<TreeCall> template_calls; // calls of the form name<args>(...), which a lexer cannot recognise

  // Definition whose body opens at `body_begin`; nullptr when the tree has none.
  const TreeFunction* function_with_body(std::uint32_t body_begin) const;
};

TreeFacts collect_c_family_facts(const SyntaxTree& tree, std::string_view text);

}  // namespace prag::parse::detail
