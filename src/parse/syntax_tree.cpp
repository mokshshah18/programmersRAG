#include "syntax_tree.hpp"

#include <algorithm>
#include <cstring>

#include "lexer.hpp"

extern "C" {
const TSLanguage* tree_sitter_c(void);
const TSLanguage* tree_sitter_cpp(void);
const TSLanguage* tree_sitter_python(void);
}

namespace prag::parse::detail {

namespace {

inline bool is_type(TSNode node, const char* type) { return std::strcmp(ts_node_type(node), type) == 0; }

// Spelling of a declared name as the outline writes it: one space between
// words ("operator new"), none elsewhere ("Scope::name", "operator=="), and
// without the template arguments of scopes ("Scope<T>::name" -> "Scope::name").
std::string spell_name(std::string_view raw) {
  std::string out;
  bool previous_word = false;
  bool in_operator = false;
  int angles = 0;
  for (const Token& token : lex_c_family(raw)) {
    if (token.kind == TokenKind::Comment) continue;
    const std::string_view word = token_text(raw, token);
    if (angles > 0) {
      if (word == "<") ++angles;
      if (word == ">") --angles;
      continue;
    }
    if (word == "<" && previous_word && !in_operator) {
      angles = 1;
      continue;
    }
    const bool is_word = token.kind == TokenKind::Identifier;
    if (is_word && previous_word) out.push_back(' ');
    out.append(word);
    if (word == "operator") in_operator = true;
    if (word == "::") in_operator = false;
    previous_word = is_word;
  }
  return out;
}

// Name node of a declarator: follows the chain of pointer, reference,
// parenthesised and function declarators down to the declared identifier.
TSNode declared_name(TSNode declarator) {
  TSNode node = declarator;
  for (int depth = 0; depth < 32 && !ts_node_is_null(node); ++depth) {
    const char* type = ts_node_type(node);
    if (!std::strcmp(type, "identifier") || !std::strcmp(type, "field_identifier") ||
        !std::strcmp(type, "type_identifier") || !std::strcmp(type, "qualified_identifier") ||
        !std::strcmp(type, "destructor_name") || !std::strcmp(type, "operator_name") ||
        !std::strcmp(type, "operator_cast") || !std::strcmp(type, "template_function")) {
      return node;
    }
    TSNode next = field(node, "declarator");
    if (ts_node_is_null(next)) {
      // parenthesized_declarator and reference_declarator keep the inner declarator as a plain child.
      const std::uint32_t count = ts_node_named_child_count(node);
      for (std::uint32_t i = 0; i < count && ts_node_is_null(next); ++i) {
        const TSNode child = ts_node_named_child(node, i);
        if (std::strstr(ts_node_type(child), "declarator") || is_type(child, "identifier")) next = child;
      }
    }
    node = next;
  }
  return TSNode{};
}

}  // namespace

const TSLanguage* grammar_for(std::string_view language) {
  if (language == "c") return tree_sitter_c();
  if (language == "cpp") return tree_sitter_cpp();
  if (language == "python") return tree_sitter_python();
  return nullptr;
}

SyntaxTree::SyntaxTree(std::string_view language, std::string_view text) {
  const TSLanguage* grammar = grammar_for(language);
  if (grammar == nullptr || text.size() >= 0xFFFFFFFFu) return;
  TSParser* parser = ts_parser_new();
  if (parser == nullptr) return;
  if (ts_parser_set_language(parser, grammar)) {
    tree_ = ts_parser_parse_string(parser, nullptr, text.data(), static_cast<std::uint32_t>(text.size()));
  }
  ts_parser_delete(parser);
}

SyntaxTree::~SyntaxTree() {
  if (tree_ != nullptr) ts_tree_delete(tree_);
}

SyntaxTree& SyntaxTree::operator=(SyntaxTree&& other) noexcept {
  if (this != &other) {
    if (tree_ != nullptr) ts_tree_delete(tree_);
    tree_ = other.tree_;
    other.tree_ = nullptr;
  }
  return *this;
}

std::size_t SyntaxTree::count_errors() const {
  if (!valid()) return 0;
  std::size_t errors = 0;
  walk(root(), [&](TSNode node) {
    if (ts_node_is_error(node) || ts_node_is_missing(node)) ++errors;
    return ts_node_has_error(node);  // subtrees without errors need no visit
  });
  return errors;
}

const TreeFunction* TreeFacts::function_with_body(std::uint32_t body_begin) const {
  const auto it = std::lower_bound(functions.begin(), functions.end(), body_begin,
                                   [](const TreeFunction& f, std::uint32_t value) { return f.body_begin < value; });
  return it != functions.end() && it->body_begin == body_begin ? &*it : nullptr;
}

TreeFacts collect_c_family_facts(const SyntaxTree& tree, std::string_view text) {
  TreeFacts facts;
  if (!tree.valid()) return facts;
  walk(tree.root(), [&](TSNode node) {
    if (!ts_node_is_named(node)) return false;
    const char* type = ts_node_type(node);
    if (!std::strcmp(type, "function_definition")) {
      const TSNode body = field(node, "body");
      if (!ts_node_is_null(body) && !ts_node_is_missing(body) && ts_node_end_byte(body) > ts_node_start_byte(body) &&
          text[ts_node_start_byte(body)] == '{') {
        TreeFunction f;
        f.begin = ts_node_start_byte(node);
        f.end = ts_node_end_byte(node);
        f.body_begin = ts_node_start_byte(body);
        f.clean = !ts_node_has_error(node);
        const TSNode name = declared_name(field(node, "declarator"));
        if (!ts_node_is_null(name)) {
          TSNode shown = name;
          if (is_type(name, "template_function")) {
            const TSNode inner = field(name, "name");
            if (!ts_node_is_null(inner)) shown = inner;
          }
          f.name = spell_name(node_text(text, shown));
          if (is_type(name, "operator_cast")) {  // the node holds the parameter list as well
            const TSNode target = field(name, "type");
            f.name = ts_node_is_null(target) ? std::string() : "operator " + spell_name(node_text(text, target));
          }
        }
        facts.functions.push_back(std::move(f));
      }
    } else if (!std::strcmp(type, "call_expression") && !ts_node_has_error(node)) {
      TSNode callee = field(node, "function");
      if (!ts_node_is_null(callee) && is_type(callee, "qualified_identifier")) {
        // a::b::f<T>(x): the name field of the innermost qualified identifier
        for (int depth = 0; depth < 16 && is_type(callee, "qualified_identifier"); ++depth) {
          const TSNode inner = field(callee, "name");
          if (ts_node_is_null(inner)) break;
          callee = inner;
        }
      } else if (!ts_node_is_null(callee) && is_type(callee, "field_expression")) {
        const TSNode member = field(callee, "field");
        if (!ts_node_is_null(member)) callee = member;
      }
      if (!ts_node_is_null(callee) && (is_type(callee, "template_function") || is_type(callee, "template_method"))) {
        const TSNode name = field(callee, "name");
        if (!ts_node_is_null(name) && (is_type(name, "identifier") || is_type(name, "field_identifier"))) {
          facts.template_calls.push_back({ts_node_start_byte(name), node_text(text, name)});
        }
      }
    }
    return true;
  });
  std::sort(facts.functions.begin(), facts.functions.end(),
            [](const TreeFunction& a, const TreeFunction& b) { return a.body_begin < b.body_begin; });
  std::sort(facts.template_calls.begin(), facts.template_calls.end(),
            [](const TreeCall& a, const TreeCall& b) { return a.offset < b.offset; });
  return facts;
}

}  // namespace prag::parse::detail
