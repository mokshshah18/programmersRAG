// Outline of C and C++ files.
//
// The byte range of every top-level item comes from brace matching over the
// token stream, which macros cannot confuse. What the item is, and what it is
// called, comes from two sources that are compared: the declaration header read
// token by token, and the syntax tree where tree-sitter parsed the item without
// errors. The tree decides between candidate names and recognises definitions
// the header rules do not cover (old-style parameter declarations, declarators
// that return function pointers); the header rules decide wherever the tree has
// errors, which in macro-heavy code is most of the time.
#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>

#include "lexer.hpp"
#include "outline.hpp"

namespace prag::parse::detail {

namespace {

constexpr std::uint32_t kNone = 0xFFFFFFFFu;
constexpr int kMaxNesting = 48;       // containers nested deeper than this are not opened
constexpr std::size_t kMaxTail = 96;  // tokens between the '}' of a type and its ';'
constexpr std::size_t kMaxHeader = 4096;  // tokens of a declaration outside its blocks

using WordSet = std::unordered_set<std::string_view>;

// Identifiers that are never the name of a called or defined function.
const WordSet& reserved_words() {
  static const WordSet words = {
      "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Generic", "_Static_assert", "__alignof__", "__asm",
      "__asm__", "__attribute", "__attribute__", "__declspec", "__except", "__extension__", "__finally",
      "__pragma", "__try", "__typeof", "__typeof__", "alignas", "alignof", "asm", "auto", "bool", "break",
      "case", "catch", "char", "char16_t", "char32_t", "char8_t", "class", "co_await", "co_return",
      "co_yield", "const", "const_cast", "continue", "decltype", "default", "defined", "delete", "do",
      "double", "dynamic_cast", "else", "enum", "explicit", "float", "for", "goto", "if", "int", "long",
      "new", "noexcept", "operator", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
      "static_assert", "static_cast", "struct", "switch", "this", "throw", "try", "typeid", "typename",
      "typeof", "typeof_unqual", "union", "unsigned", "void", "volatile", "wchar_t", "while",
  };
  return words;
}

// Keywords after which an expression, and therefore a call, may start.
const WordSet& expression_keywords() {
  static const WordSet words = {
      "return", "else", "do", "case", "goto", "throw", "new", "delete", "co_return", "co_yield", "co_await",
      "and", "or", "not", "xor", "if", "while", "for", "switch", "sizeof", "default", "try",
  };
  return words;
}

// Keywords whose parenthesised argument is not code.
const WordSet& attribute_keywords() {
  static const WordSet words = {
      "__attribute__", "__attribute", "__declspec", "alignas", "_Alignas", "asm", "__asm__", "__asm", "__pragma",
  };
  return words;
}

// Words that cannot occur in a parameter list.
const WordSet& statement_keywords() {
  static const WordSet words = {
      "return", "if", "else", "while", "for", "switch", "goto", "break", "continue", "do", "case",
  };
  return words;
}

const WordSet& builtin_types() {
  static const WordSet words = {
      "void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "bool", "_Bool",
      "wchar_t", "char8_t", "char16_t", "char32_t", "auto",
  };
  return words;
}

inline bool contains(const WordSet& set, std::string_view word) { return set.find(word) != set.end(); }

inline bool is_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

// MI_FORWARD1, PRAG_TEST: capitals, digits and underscores, at least one capital.
bool is_macro_like(std::string_view word) {
  bool capital = false;
  for (const char c : word) {
    if (c >= 'A' && c <= 'Z') {
      capital = true;
    } else if (!(c >= '0' && c <= '9') && c != '_') {
      return false;
    }
  }
  return capital && word.size() >= 2;
}

// Appends the identifiers in call position among tokens [first, last).
//
// In an expression (a function body, an initialiser, an argument) `name(` is a
// call unless a type or another name directly precedes it, which is how a
// declaration looks. At declaration level, where '*' and '&' belong to
// declarators, only a macro invocation that starts a declaration or follows a
// parameter list counts.
void find_calls(std::string_view text, const std::vector<Token>& tokens, const std::vector<std::uint8_t>* inert,
                std::size_t first, std::size_t last, bool expression, const std::vector<std::string_view>& excluded,
                std::vector<CallSite>& out) {
  enum class Prev { Start, Name, Keyword, Close, Other };
  Prev prev = Prev::Start;
  int parens = 0;
  bool assigned = false;
  const auto visible = [&](std::size_t k) {
    return tokens[k].kind != TokenKind::Comment && (inert == nullptr || (*inert)[k] == 0);
  };
  for (std::size_t k = first; k < last; ++k) {
    if (!visible(k)) continue;
    const Token& token = tokens[k];
    if (token.kind == TokenKind::Directive) {
      prev = Prev::Start;
      continue;
    }
    if (token.kind == TokenKind::Identifier) {
      const std::string_view word = token_text(text, token);
      std::size_t next = k + 1;
      while (next < last && !visible(next)) ++next;
      if (next < last && is_punct(text, tokens[next], '(')) {
        if (contains(attribute_keywords(), word)) {
          int depth = 0;
          std::size_t j = next;
          for (; j < last; ++j) {
            if (is_punct(text, tokens[j], '(')) ++depth;
            if (is_punct(text, tokens[j], ')') && --depth == 0) break;
          }
          k = j;
          prev = Prev::Close;
          continue;
        }
        if (!contains(reserved_words(), word) &&
            std::find(excluded.begin(), excluded.end(), word) == excluded.end()) {
          const bool counts = expression || assigned || parens > 0 ? prev != Prev::Name
                                                                   : prev == Prev::Start || prev == Prev::Close;
          if (counts) out.push_back({token.begin, word});
        }
      }
      prev = contains(expression_keywords(), word) ? Prev::Keyword : Prev::Name;
      continue;
    }
    if (token.kind != TokenKind::Punct) {
      prev = Prev::Other;
      continue;
    }
    switch (text[token.begin]) {
      case '(':
      case '[':
        ++parens;
        prev = Prev::Other;
        break;
      case ')':
      case ']':
        if (parens > 0) --parens;
        prev = Prev::Close;
        break;
      case ';':
        if (parens == 0) assigned = false;
        prev = Prev::Start;
        break;
      case '{':
      case '}':
        prev = Prev::Start;
        break;
      case '=':
        if (parens == 0) assigned = true;
        prev = Prev::Other;
        break;
      default:
        prev = Prev::Other;
        break;
    }
  }
}

struct Declarator {
  bool found = false;
  std::size_t name_first = 0;  // positions in the header
  std::size_t open = 0;
  std::size_t close = 0;
  int strength = 0;  // 2: certainly a parameter list, 1: could also be macro arguments
  std::string name;
};

struct TypeHead {
  bool found = false;
  bool is_enum = false;
  bool is_typedef = false;
  std::string tag;
};

class Scanner {
 public:
  Scanner(std::string_view text, bool cpp, const TreeFacts& facts, const ParseOptions& options)
      : text_(text), cpp_(cpp), facts_(facts), options_(options), lines_(text) {}

  Outline run() {
    tokens_ = lex_c_family(text_);
    inert_.assign(tokens_.size(), 0);
    partner_.assign(tokens_.size(), kNone);
    match_brackets();

    // order of the tree's definitions by their first byte
    tree_starts_.resize(facts_.functions.size());
    for (std::size_t k = 0; k < tree_starts_.size(); ++k) tree_starts_[k] = static_cast<std::uint32_t>(k);
    std::sort(tree_starts_.begin(), tree_starts_.end(), [&](std::uint32_t a, std::uint32_t b) {
      return facts_.functions[a].begin < facts_.functions[b].begin;
    });

    std::size_t i = 0;
    scan_scope(i, tokens_.size(), Context{});

    for (const TreeCall& call : facts_.template_calls) {
      if (!contains(reserved_words(), call.name)) out_.calls.push_back({call.offset, call.name});
    }
    std::sort(out_.calls.begin(), out_.calls.end(), [](const CallSite& a, const CallSite& b) {
      return a.offset != b.offset ? a.offset < b.offset : a.name < b.name;
    });
    out_.calls.erase(std::unique(out_.calls.begin(), out_.calls.end(),
                                 [](const CallSite& a, const CallSite& b) {
                                   return a.offset == b.offset && a.name == b.name;
                                 }),
                     out_.calls.end());
    return std::move(out_);
  }

 private:
  struct Context {
    bool in_type = false;
    std::string prefix;  // "Class::" inside a type that is being split
    std::uint32_t owner = 0;
    int nesting = 0;
  };
  // Indices of the significant tokens of an item; blocks are left out.
  using Header = std::vector<std::uint32_t>;

  enum class Block { Linkage, Namespace, Function, Type, MemberInitialiser, Opaque };
  struct Classified {
    Block kind = Block::Opaque;
    std::string name;
    TypeHead head;
  };

  // ---- tokens ---------------------------------------------------------------

  const Token& tok(std::size_t i) const { return tokens_[i]; }
  std::string_view word(std::size_t i) const { return token_text(text_, tokens_[i]); }
  bool punct(std::size_t i, char c) const { return is_punct(text_, tokens_[i], c); }
  bool punct2(std::size_t i, std::string_view s) const {
    return tokens_[i].kind == TokenKind::Punct && word(i) == s;
  }
  bool ident(std::size_t i) const { return tokens_[i].kind == TokenKind::Identifier; }
  bool ident(std::size_t i, std::string_view s) const { return ident(i) && word(i) == s; }
  bool significant(std::size_t i) const { return inert_[i] == 0 && tokens_[i].kind != TokenKind::Comment; }

  std::size_t next_significant(std::size_t i, std::size_t limit) const {
    while (i < limit && !significant(i)) ++i;
    return i;
  }

  // Index one past the block that opens at `open`.
  std::size_t after_block(std::size_t open, std::size_t limit) const {
    const std::uint32_t close = partner_[open];
    return close == kNone || close >= limit ? limit : static_cast<std::size_t>(close) + 1;
  }

  bool conditional(std::size_t i) const {
    const DirectiveKind kind = inspect_directive(text_, tok(i)).kind;
    return kind == DirectiveKind::If || kind == DirectiveKind::Else || kind == DirectiveKind::Endif;
  }

  // ---- conditional compilation and brace matching ------------------------------

  // Marks the alternatives of a conditional inert, from the #else at `i` to its #endif.
  std::size_t skip_alternatives(std::size_t i) {
    int nested = 0;
    for (std::size_t j = i; j < tokens_.size(); ++j) {
      inert_[j] = 1;
      if (j == i || tokens_[j].kind != TokenKind::Directive) continue;
      const DirectiveKind kind = inspect_directive(text_, tokens_[j]).kind;
      if (kind == DirectiveKind::If) {
        ++nested;
      } else if (kind == DirectiveKind::Endif) {
        if (nested == 0) return j;
        --nested;
      }
    }
    return tokens_.size() - 1;
  }

  // Pairs the brackets. Braces decide: a parenthesis left open inside a block
  // stays without partner when the block closes, and so does one that never
  // closes at all, so that a missing ')' costs one declaration and not the rest
  // of the file. Alternative branches of a conditional are not counted twice:
  // when the first branch leaves a bracket open or closes one (two versions of
  // a function header, say), the other branches are skipped.
  void match_brackets() {
    std::vector<std::uint32_t> open;  // opening brackets without partner so far
    std::size_t open_braces = 0;      // how many of them are braces
    std::vector<std::size_t> open_at_if;
    const auto kind = [&](std::uint32_t index) { return text_[tokens_[index].begin]; };
    const auto pair = [&](std::size_t closer) {
      partner_[open.back()] = static_cast<std::uint32_t>(closer);
      partner_[closer] = open.back();
      open.pop_back();
    };
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
      const Token& token = tokens_[i];
      if (token.kind == TokenKind::Directive) {
        const DirectiveKind directive = inspect_directive(text_, token).kind;
        if (directive == DirectiveKind::If) {
          open_at_if.push_back(open.size());
        } else if (directive == DirectiveKind::Else && !open_at_if.empty()) {
          if (open.size() != open_at_if.back()) {
            i = skip_alternatives(i);
            open_at_if.pop_back();
          }
        } else if (directive == DirectiveKind::Endif && !open_at_if.empty()) {
          open_at_if.pop_back();
        }
        continue;
      }
      if (token.kind != TokenKind::Punct || token.end - token.begin != 1) continue;
      const char c = text_[token.begin];
      if (c == '{' || c == '(' || c == '[') {
        open.push_back(static_cast<std::uint32_t>(i));
        if (c == '{') ++open_braces;
      } else if (c == '}' && open_braces > 0) {
        while (kind(open.back()) != '{') open.pop_back();
        pair(i);
        --open_braces;
      } else if (c == ')' || c == ']') {
        const char wanted = c == ')' ? '(' : '[';
        // a bracket of the other kind in between has no partner
        while (!open.empty() && kind(open.back()) != '{' && kind(open.back()) != wanted) open.pop_back();
        if (!open.empty() && kind(open.back()) == wanted) pair(i);
      }
    }
  }

  // ---- declaration headers --------------------------------------------------------

  // Position of the ')' or ']' matching the bracket at header position `p`; `to` when there is none.
  std::size_t closing(const Header& h, std::size_t p, std::size_t to) const {
    int depth = 0;
    for (std::size_t q = p; q < to; ++q) {
      if (punct(h[q], '(') || punct(h[q], '[')) ++depth;
      if ((punct(h[q], ')') || punct(h[q], ']')) && --depth == 0) return q;
    }
    return to;
  }

  // 0: not a parameter list, 1: single names that could also be macro arguments, 2: parameters.
  int parameter_strength(const Header& h, std::size_t open, std::size_t close) const {
    if (close == open + 1) return 2;
    bool strong = false;
    std::size_t p = open + 1;
    while (p < close) {
      // one parameter: up to the next comma at this level
      std::size_t count = 0;
      std::size_t dots = 0;
      bool defaulted = false;
      int depth = 0;
      std::string_view only;
      for (; p < close; ++p) {
        const std::uint32_t t = h[p];
        if (depth == 0 && punct(t, ',')) break;
        const bool opens = punct(t, '(') || punct(t, '[');
        if (opens) ++depth;
        if (punct(t, ')') || punct(t, ']')) --depth;
        if (defaulted) continue;
        // what stands inside nested brackets is an array size, the parameters of a
        // function pointer or the arguments of an annotation: _In_range_(1, 128)
        if (depth > (opens ? 1 : 0)) continue;
        if (depth == 0 && punct(t, '=')) {
          defaulted = true;
          continue;
        }
        switch (tok(t).kind) {
          case TokenKind::Identifier:
            if (contains(statement_keywords(), word(t))) return 0;
            only = word(t);
            break;
          case TokenKind::Number:
            return 0;
          case TokenKind::Punct: {
            const std::string_view s = word(t);
            if (s == ".") {
              ++dots;
            } else if (s != "*" && s != "&" && s != "::" && s != "<" && s != ">" && s != "[" && s != "]" &&
                       s != "(" && s != ")") {
              return 0;
            }
            break;
          }
          default:
            return 0;
        }
        ++count;
      }
      if (p < close) ++p;  // the comma
      if (dots > 0) {
        if (dots != 3) return 0;
        continue;  // variadic
      }
      if (count == 0) return 0;
      if (count >= 2 || contains(builtin_types(), only)) strong = true;
    }
    return strong ? 2 : 1;
  }

  // Name of a definition whose name a macro makes from its arguments: the
  // arguments joined by underscores. TEST(suite, name) -> suite_name.
  std::string joined_arguments(const Header& h, std::size_t open, std::size_t close) const {
    std::string name;
    for (std::size_t p = open + 1; p < close; ++p) {
      if (!ident(h[p])) continue;
      if (!name.empty() && name.back() != '_') name.push_back('_');
      name.append(word(h[p]));
    }
    return name;
  }

  // The declarators of a function header: every `name(...)` outside brackets
  // whose parentheses could hold parameters, in source order.
  std::vector<Declarator> candidates(const Header& h, std::size_t from, std::size_t to) const {
    std::vector<Declarator> found;
    int squares = 0;
    bool closed = false;  // a parameter list may have been passed
    for (std::size_t p = from; p < to; ++p) {
      const std::uint32_t t = h[p];
      // member initialisers and a trailing return type hold no declarator
      if (squares == 0 && closed && (punct(t, ':') || punct2(t, "->"))) break;
      if (punct(t, '[')) ++squares;
      if (punct(t, ']') && squares > 0) --squares;
      if (!punct(t, '(')) continue;
      std::size_t open = p;
      std::size_t close = closing(h, p, to);
      if (close >= to) break;
      p = close;
      if (squares > 0) continue;
      closed = true;
      if (open == from) continue;

      Declarator d;
      const std::size_t q = open - 1;
      if (ident(h[q], "operator")) {
        // operator()(...): the empty pair belongs to the name
        if (close != open + 1 || close + 1 >= to || !punct(h[close + 1], '(')) continue;
        d.name = "operator()";
        d.name_first = q;
        open = close + 1;
        close = closing(h, open, to);
        if (close >= to) break;
        p = close;
      } else if (ident(h[q])) {
        if (q > from && ident(h[q - 1], "operator")) {
          d.name = "operator " + std::string(word(h[q]));  // new, delete or a conversion
          d.name_first = q - 1;
        } else if (contains(reserved_words(), word(h[q]))) {
          continue;
        } else if (close + 1 < to && punct(h[close + 1], '(') && parameter_strength(h, open, close) == 1) {
          // PASTE(prefix_, SUFFIX)(parameters): the macro makes the name
          d.name = joined_arguments(h, open, close);
          d.name_first = q;
          open = close + 1;
          close = closing(h, open, to);
          if (close >= to) break;
          p = close;
        } else {
          d.name = std::string(word(h[q]));
          d.name_first = q;
        }
      } else if (tok(h[q]).kind == TokenKind::Punct) {
        // operator==, operator[], operator delete[]
        std::size_t r = q;
        std::string symbols;
        for (int steps = 0; steps < 3 && r > from && tok(h[r]).kind == TokenKind::Punct; ++steps, --r) {
          symbols.insert(0, word(h[r]));
        }
        std::string spelled;
        if (r > from && (ident(h[r], "new") || ident(h[r], "delete"))) {
          spelled = " " + std::string(word(h[r]));
          --r;
        }
        if (!ident(h[r], "operator")) continue;
        d.name = "operator" + spelled + symbols;
        d.name_first = r;
      } else {
        continue;
      }

      // ~Name and Scope::Name
      if (d.name_first > from && punct(h[d.name_first - 1], '~')) {
        d.name.insert(0, "~");
        --d.name_first;
      }
      while (d.name_first > from && punct2(h[d.name_first - 1], "::")) {
        if (d.name_first < from + 2) {
          --d.name_first;
          break;
        }
        std::size_t scope = d.name_first - 2;
        if (punct(h[scope], '>')) {  // Scope<T>::
          int angles = 0;
          while (scope > from) {
            if (punct(h[scope], '>')) ++angles;
            if (punct(h[scope], '<') && --angles == 0) break;
            --scope;
          }
          if (scope == from) break;
          --scope;
        }
        if (!ident(h[scope]) || contains(reserved_words(), word(h[scope]))) {
          --d.name_first;
          break;
        }
        d.name.insert(0, std::string(word(h[scope])) + "::");
        d.name_first = scope;
      }
      // operator ns::type(): a conversion to a qualified type
      if (d.name_first > from && ident(h[d.name_first - 1], "operator") && d.name.compare(0, 8, "operator") != 0) {
        d.name.insert(0, "operator ");
        --d.name_first;
      }

      d.open = open;
      d.close = close;
      d.strength = parameter_strength(h, open, close);
      if (d.strength == 0) continue;
      d.found = true;
      found.push_back(std::move(d));
    }
    return found;
  }

  // What may stand between a parameter list and the body.
  bool trailing_ok(const Header& h, std::size_t from, std::size_t to) const {
    int depth = 0;
    for (std::size_t p = from; p < to; ++p) {
      const std::uint32_t t = h[p];
      if (punct(t, '(') || punct(t, '[')) {
        ++depth;
        continue;
      }
      if (punct(t, ')') || punct(t, ']')) {
        if (depth > 0) --depth;
        continue;
      }
      if (depth > 0) continue;
      if (punct(t, ':') || punct2(t, "->")) return true;  // initialisers or a return type follow
      if (ident(t) || punct(t, '&')) continue;
      if (cpp_ && (punct(t, '<') || punct(t, '>') || punct2(t, "::"))) continue;
      return false;
    }
    return true;
  }

  // Chooses the declarator of a function header. `body` is the index of the
  // token that opens the body, or kNone. The syntax tree decides when it parsed
  // the definition without errors and names one of the candidates; otherwise
  // the last parameter list wins, where a list that directly follows another
  // one is taken for the arguments of an attribute macro.
  Declarator find_declarator(const Header& h, std::size_t from, std::size_t to, std::size_t body) const {
    const std::vector<Declarator> found = candidates(h, from, to);
    const TreeFunction* tree = body == kNone ? nullptr : facts_.function_with_body(tok(body).begin);
    if (tree != nullptr && tree->clean && !tree->name.empty()) {
      for (auto it = found.rbegin(); it != found.rend(); ++it) {
        if (it->name == tree->name) return *it;
      }
    }
    const auto after_list = [&](const Declarator& d) {
      return d.name_first > from && punct(h[d.name_first - 1], ')');
    };
    for (const bool allow_after_list : {false, true}) {
      for (const int strength : {2, 1}) {
        for (auto it = found.rbegin(); it != found.rend(); ++it) {
          if (it->strength == strength && (allow_after_list || !after_list(*it))) return *it;
        }
      }
    }
    return {};
  }

  // Skips `template <...>` clauses at the start of a header.
  std::size_t after_template_clauses(const Header& h) const {
    std::size_t p = 0;
    while (p + 1 < h.size() && ident(h[p], "template") && punct(h[p + 1], '<')) {
      int angles = 0;
      std::size_t q = p + 1;
      for (; q < h.size(); ++q) {
        if (punct(h[q], '<')) ++angles;
        if (punct(h[q], '>') && --angles == 0) break;
      }
      if (q >= h.size()) return p;
      p = q + 1;
    }
    return p;
  }

  TypeHead find_type_head(const Header& h) const {
    TypeHead head;
    const std::size_t p = after_template_clauses(h);
    head.is_typedef = p < h.size() && ident(h[p], "typedef");
    int depth = 0;
    std::size_t keyword = h.size();
    for (std::size_t q = p; q < h.size(); ++q) {
      if (punct(h[q], '(') || punct(h[q], '[')) ++depth;
      if ((punct(h[q], ')') || punct(h[q], ']')) && depth > 0) --depth;
      if (depth != 0 || !ident(h[q])) continue;
      const std::string_view w = word(h[q]);
      if (w == "struct" || w == "union" || w == "enum" || w == "class") {
        keyword = q;
        head.is_enum = w == "enum";
        break;
      }
    }
    if (keyword == h.size()) return head;
    head.found = true;
    std::size_t q = keyword + 1;
    if (head.is_enum && q < h.size() && (ident(h[q], "class") || ident(h[q], "struct"))) ++q;
    while (q < h.size()) {
      const std::uint32_t t = h[q];
      if (punct(t, '[') || punct(t, '(')) {  // [[attribute]]
        q = closing(h, q, h.size());
        if (q < h.size()) ++q;
      } else if (ident(t) && q + 1 < h.size() && punct(h[q + 1], '(')) {  // alignas(8), MACRO(x)
        q = closing(h, q + 1, h.size());
        if (q < h.size()) ++q;
      } else if (ident(t) && !ident(t, "final")) {
        head.tag = std::string(word(t));
        ++q;
      } else if (punct2(t, "::") && !head.tag.empty() && q + 1 < h.size() && ident(h[q + 1])) {
        head.tag += "::" + std::string(word(h[q + 1]));
        q += 2;
      } else {
        break;  // base classes, underlying type, template arguments
      }
    }
    return head;
  }

  // Name introduced by a typedef: the identifier in the first of two adjacent
  // parenthesised groups (function and function pointer types), else the last
  // identifier outside brackets.
  std::string typedef_name(const Header& h, std::size_t from, std::size_t to) const {
    std::string last;
    for (std::size_t p = from; p < to; ++p) {
      const std::uint32_t t = h[p];
      if (punct(t, '(') || punct(t, '[')) {
        const std::size_t close = closing(h, p, to);
        if (punct(t, '(') && close + 1 < to && punct(h[close + 1], '(')) {
          for (std::size_t q = close; q > p; --q) {
            if (ident(h[q]) && !contains(reserved_words(), word(h[q]))) return std::string(word(h[q]));
          }
        }
        p = close;
      } else if (ident(t) && !contains(reserved_words(), word(t))) {
        if (p + 1 < to && punct(h[p + 1], '(') && contains(attribute_keywords(), word(t))) continue;
        last = std::string(word(t));
      }
    }
    return last;
  }

  // True when the header ends inside the name of an operator function, so that
  // a following '=' is part of the name (operator=, operator==).
  bool operator_name(const Header& h) const {
    std::size_t steps = 0;
    for (std::size_t p = h.size(); p-- > 0 && steps < 3; ++steps) {
      if (ident(h[p], "operator")) return true;
      if (tok(h[p]).kind != TokenKind::Punct) return false;
    }
    return false;
  }

  // MACRO(...) and nothing else.
  bool lone_invocation(const Header& h) const {
    return h.size() >= 3 && ident(h[0]) && is_macro_like(word(h[0])) && punct(h[1], '(') &&
           closing(h, 1, h.size()) == h.size() - 1;
  }

  // The tree's name for a definition is accepted when it is a name and the
  // header spells it.
  bool spelled_in_header(const TreeFunction& f) const {
    if (f.name.empty() || f.body_begin <= f.begin) return false;
    const bool is_operator = f.name.compare(0, 8, "operator") == 0 || f.name.find("::operator") != std::string::npos;
    if (!is_operator) {
      for (const char c : f.name) {
        if (!is_name_char(c) && c != ':' && c != '~' && c != '$' && static_cast<unsigned char>(c) < 0x80) return false;
      }
    }
    const std::string_view header = text_.substr(f.begin, f.body_begin - f.begin);
    const std::size_t scope = is_operator ? std::string::npos : f.name.rfind("::");
    std::string_view base = scope == std::string::npos ? std::string_view(f.name)
                                                       : std::string_view(f.name).substr(scope + 2);
    if (is_operator) base = "operator";
    if (!base.empty() && base.front() == '~') base.remove_prefix(1);
    return !base.empty() && header.find(base) != std::string_view::npos;
  }

  // What the block that opens at token `open`, after the header `h`, is.
  Classified classify(const Header& h, std::size_t open) const {
    Classified result;
    if (h.empty()) return result;
    // extern "C" { and namespace name {, possibly behind macros: MY_NAMESPACE_BEGIN namespace detail {
    std::size_t words = 0;  // identifiers at the start of the header
    while (words < h.size() && ident(h[words]) && !ident(h[words], "namespace") && !ident(h[words], "extern")) ++words;
    if (words + 2 == h.size() && ident(h[words], "extern") && tok(h[words + 1]).kind == TokenKind::String) {
      result.kind = Block::Linkage;
      return result;
    }
    if (words < h.size() && ident(h[words], "namespace")) {
      const bool name_only = std::all_of(h.begin() + static_cast<std::ptrdiff_t>(words), h.end(), [&](std::uint32_t t) {
        return ident(t) || punct2(t, "::");
      });
      if (name_only) {
        result.kind = Block::Namespace;
        return result;
      }
    }

    const Declarator d = find_declarator(h, 0, h.size(), open);
    if (d.found && trailing_ok(h, d.close + 1, h.size())) {
      // Foo(int x) : a_{x}, b_(x) { ... }: only a brace after ')' or '}' opens the body
      bool initialisers = false;
      int depth = 0;
      for (std::size_t p = d.close + 1; p < h.size(); ++p) {
        if (punct(h[p], '(') || punct(h[p], '[')) ++depth;
        if ((punct(h[p], ')') || punct(h[p], ']')) && depth > 0) --depth;
        if (depth == 0 && punct(h[p], ':')) initialisers = true;
      }
      if (initialisers && (ident(h.back()) || punct(h.back(), '>'))) {
        result.kind = Block::MemberInitialiser;
        return result;
      }
      result.kind = Block::Function;
      result.name = d.name;
      // TEST(suite, name) { ... }: a definition made by a macro is called after the
      // macro's arguments, the only thing that tells such definitions apart
      if (d.name_first == 0 && d.strength == 1 && d.close + 1 == h.size() && is_macro_like(d.name)) {
        const std::string arguments = joined_arguments(h, d.open, d.close);
        if (!arguments.empty()) result.name = arguments;
      }
      return result;
    }

    // A definition only the syntax tree recognises.
    const TreeFunction* tree = facts_.function_with_body(tok(open).begin);
    if (tree != nullptr && tree->clean && tree->begin >= tok(h.front()).begin && spelled_in_header(*tree)) {
      result.kind = Block::Function;
      result.name = tree->name;
      return result;
    }

    result.head = find_type_head(h);
    if (result.head.found) result.kind = Block::Type;
    return result;
  }

  // ---- pieces -----------------------------------------------------------------------

  bool single_line(std::uint32_t begin, std::uint32_t end) const {
    return text_.substr(begin, end - begin).find('\n') == std::string_view::npos;
  }

  bool oversized(const Piece& piece) const {
    const std::size_t limit = piece.kind == PieceKind::Other ? options_.max_region_bytes : options_.max_unit_bytes;
    return piece.end - piece.begin > limit;
  }

  Piece make_piece(PieceKind kind, std::size_t first, std::size_t last, const Context& context) const {
    Piece piece;
    piece.kind = kind;
    piece.begin = tok(first).begin;
    piece.end = tok(last).end;
    piece.owner = context.owner;
    return piece;
  }

  void emit(Piece piece) { out_.pieces.push_back(std::move(piece)); }

  void add_calls(std::size_t first, std::size_t last, bool expression) {
    static const std::vector<std::string_view> none;
    find_calls(text_, tokens_, &inert_, first, last, expression, none, out_.calls);
  }

  // Places where a piece that is too large may be cut: the starts of the lines
  // that follow a statement of the block between `open` and `close`.
  void add_breaks(std::size_t open, std::size_t close, bool commas, Piece& piece) const {
    int depth = 1;
    int parens = 0;
    int level = -1;  // -1: inside a statement, 0: after a statement of the block, 1: after a nested one
    for (std::size_t k = open + 1; k < close && k < tokens_.size(); ++k) {
      const Token& token = tokens_[k];
      if (token.first_on_line && level >= 0 && inert_[k] == 0 && !ident(k, "else") && !ident(k, "catch")) {
        const std::size_t at = lines_.next_line(tokens_[k - 1].end - 1);
        if (at > piece.begin && at <= token.begin) piece.breaks[level].push_back(static_cast<std::uint32_t>(at));
      }
      if (inert_[k] != 0 || token.kind == TokenKind::Comment || token.kind == TokenKind::Directive) continue;
      level = -1;
      if (token.kind != TokenKind::Punct) continue;
      switch (text_[token.begin]) {
        case '(':
        case '[':
          ++parens;
          break;
        case ')':
        case ']':
          if (parens > 0) --parens;
          break;
        case '{':
          ++depth;
          break;
        case '}':
          if (depth > 1) --depth;
          if (parens == 0) level = depth == 1 ? 0 : 1;
          break;
        case ';':
          if (parens == 0) level = depth == 1 ? 0 : 1;
          break;
        case ',':
          if (commas && parens == 0) level = depth == 1 ? 0 : 1;
          break;
        default:
          break;
      }
    }
  }

  void emit_other(std::size_t first, std::size_t last, const Context& context, Glue glue = Glue::None) {
    Piece piece = make_piece(PieceKind::Other, first, last, context);
    piece.glue = glue;
    if (oversized(piece)) {
      for (std::size_t k = first; k <= last; ++k) {
        if (inert_[k] == 0 && punct(k, '{')) {
          add_breaks(k, after_block(k, last + 1) - 1, true, piece);
          break;
        }
      }
    }
    add_calls(first, last + 1, false);
    emit(std::move(piece));
  }

  void emit_directive(std::size_t i, const Context& context) {
    const DirectiveInfo info = inspect_directive(text_, tok(i));
    Piece piece = make_piece(PieceKind::Other, i, i, context);
    if (info.kind == DirectiveKind::If) piece.glue = Glue::If;
    if (info.kind == DirectiveKind::Else) piece.glue = Glue::Else;
    if (info.kind == DirectiveKind::Endif) piece.glue = Glue::Endif;
    if (info.kind == DirectiveKind::Define && !info.name.empty()) {
      piece.kind = PieceKind::Macro;
      piece.name = std::string(info.name);
      piece.callable = info.function_like;
      piece.small = single_line(piece.begin, piece.end);
      piece.signature_end = std::min(info.body_begin, piece.end);
      add_macro_calls(tok(i), info);
    }
    emit(std::move(piece));
  }

  // Calls made by the replacement text of a macro; its parameters are not functions.
  void add_macro_calls(const Token& directive, const DirectiveInfo& info) {
    if (info.body_begin >= directive.end) return;
    std::vector<std::string_view> parameters;
    if (info.function_like) {
      std::size_t p = info.name_end + 1;
      while (p < info.body_begin) {
        while (p < info.body_begin && (!is_name_char(text_[p]) || (text_[p] >= '0' && text_[p] <= '9'))) ++p;
        std::size_t q = p;
        while (q < info.body_begin && is_name_char(text_[q])) ++q;
        if (q > p) parameters.push_back(text_.substr(p, q - p));
        p = q;
      }
    }
    std::vector<Token> body = lex_c_family(text_.substr(info.body_begin, directive.end - info.body_begin));
    for (Token& token : body) {
      token.begin += info.body_begin;
      token.end += info.body_begin;
    }
    find_calls(text_, body, nullptr, 0, body.size(), true, parameters, out_.calls);
  }

  void emit_function(std::size_t first, std::size_t open, std::size_t last, const std::string& name,
                     const Context& context) {
    Piece piece = make_piece(PieceKind::Function, first, last, context);
    piece.name = context.prefix + name;
    piece.signature_end = tok(open).begin;
    if (oversized(piece)) add_breaks(open, after_block(open, last + 1) - 1, false, piece);
    add_calls(first, open, false);
    add_calls(open, last + 1, true);
    emit(std::move(piece));
  }

  // `name(parameters) MACRO(arguments)` on a line of its own: a definition whose
  // body the macro supplies. The identifiers handed to the macro are what the
  // body calls, except those that name a parameter.
  void emit_macro_bodied(std::size_t first, const Header& h, const Declarator& d, std::size_t macro,
                         const Context& context) {
    Piece piece = make_piece(PieceKind::Function, first, h.back(), context);
    piece.name = context.prefix + d.name;
    piece.signature_end = tok(h[macro]).begin;
    add_calls(first, h[macro], false);
    out_.calls.push_back({tok(h[macro]).begin, word(h[macro])});
    std::vector<std::string_view> parameters;
    for (std::size_t p = d.open + 1; p < d.close; ++p) {
      if (ident(h[p]) && (p + 1 == d.close || punct(h[p + 1], ','))) parameters.push_back(word(h[p]));
    }
    for (std::size_t p = macro + 2; p + 1 < h.size(); ++p) {
      const bool argument = ident(h[p]) && (punct(h[p - 1], '(') || punct(h[p - 1], ',')) &&
                            (punct(h[p + 1], ')') || punct(h[p + 1], ','));
      if (argument && !contains(reserved_words(), word(h[p])) &&
          std::find(parameters.begin(), parameters.end(), word(h[p])) == parameters.end()) {
        out_.calls.push_back({tok(h[p]).begin, word(h[p])});
      }
    }
    emit(std::move(piece));
  }

  // Recognises the macro-bodied definition that ends with the ')' at token `i`,
  // the last token of `h`.
  bool macro_bodied(const Header& h, std::size_t i, std::size_t limit, Declarator* declarator,
                    std::size_t* macro) const {
    std::size_t open = h.size();
    int depth = 0;
    for (std::size_t p = h.size(); p-- > 0;) {
      if (punct(h[p], ')')) ++depth;
      if (punct(h[p], '(') && --depth == 0) {
        open = p;
        break;
      }
    }
    if (open == h.size() || open < 4 || !ident(h[open - 1]) || !is_macro_like(word(h[open - 1]))) return false;
    const std::size_t next = next_significant(i + 1, limit);
    if (next < limit) {
      const Token& token = tok(next);
      if (!token.first_on_line) return false;
      if (token.kind == TokenKind::Punct) {
        const char c = text_[token.begin];
        if (c == '{' || c == ';' || c == '=' || c == ',' || c == '(' || c == '[' || c == ':' || c == '-') return false;
      }
    }
    const Declarator d = find_declarator(h, 0, open - 1, kNone);
    if (!d.found || d.strength != 2 || !trailing_ok(h, d.close + 1, open - 1)) return false;
    *declarator = d;
    *macro = open - 1;
    return true;
  }

  // ---- scanning -----------------------------------------------------------------------

  void scan_scope(std::size_t& i, std::size_t limit, const Context& context) {
    while (i < limit) {
      if (inert_[i] != 0) {
        ++i;
        continue;
      }
      const Token& token = tok(i);
      if (token.kind == TokenKind::Comment) {
        emit(make_piece(PieceKind::Comment, i, i, context));
        ++i;
      } else if (token.kind == TokenKind::Directive) {
        emit_directive(i, context);
        ++i;
      } else if (punct(i, ';') || punct(i, '}')) {
        emit_other(i, i, context);  // an empty declaration, or a brace without partner
        ++i;
      } else {
        scan_item(i, limit, context);
      }
    }
  }

  // Opens a block whose content is scanned like the file itself.
  void scan_container(std::size_t first, std::size_t open, std::size_t& i, std::size_t limit, const Context& context) {
    emit_other(first, open, context, Glue::Opener);
    const bool closed = partner_[open] != kNone && partner_[open] < limit;
    const std::size_t close = closed ? static_cast<std::size_t>(partner_[open]) : limit;
    Context inner = context;
    inner.nesting = context.nesting + 1;
    i = open + 1;
    scan_scope(i, close, inner);
    i = close;
    if (closed) {
      emit_other(close, close, context, Glue::Closer);
      i = close + 1;
    }
  }

  // Splits a class into its head and its members when it defines member
  // functions. Returns the owner id of the members, or 0, leaving no trace,
  // when there is nothing to split.
  std::uint32_t split_type(std::size_t first, std::size_t open, const std::string& name, const Context& context) {
    if (partner_[open] == kNone || name.empty() || context.nesting >= kMaxNesting) return 0;
    const std::size_t close = partner_[open];
    const std::size_t pieces_before = out_.pieces.size();
    const std::size_t calls_before = out_.calls.size();
    const std::size_t owners_before = out_.owners.size();

    const std::string qualified = context.prefix + name;
    out_.owners.push_back(qualified);
    Context inner;
    inner.in_type = true;
    inner.prefix = qualified + "::";
    inner.owner = static_cast<std::uint32_t>(out_.owners.size());
    inner.nesting = context.nesting + 1;

    Piece head = make_piece(PieceKind::Type, first, open, inner);
    head.head = true;
    head.name = qualified;
    head.signature_end = tok(open).begin;
    emit(std::move(head));
    add_calls(first, open, false);
    std::size_t i = open + 1;
    scan_scope(i, close, inner);

    const bool has_functions = std::any_of(out_.pieces.begin() + static_cast<std::ptrdiff_t>(pieces_before),
                                           out_.pieces.end(),
                                           [](const Piece& p) { return p.kind == PieceKind::Function; });
    if (!has_functions) {
      out_.pieces.resize(pieces_before);
      out_.calls.resize(calls_before);
      out_.owners.resize(owners_before);
      return 0;
    }
    return inner.owner;
  }

  // A type definition: its body, which may be split, and the declarators
  // between the closing brace and the semicolon.
  void scan_type(std::size_t first, std::size_t open, const TypeHead& head, std::size_t& i, std::size_t limit,
                 const Context& context) {
    const std::size_t close = after_block(open, limit) - 1;
    const std::uint32_t owner = head.is_enum ? 0 : split_type(first, open, head.tag, context);

    Header tail;
    std::size_t last = close;
    int parens = 0;
    bool assigned = false;
    bool terminated = false;
    for (std::size_t k = close + 1; k < limit && tail.size() <= kMaxTail;) {
      if (!significant(k)) {
        ++k;
        continue;
      }
      if (tok(k).kind == TokenKind::Directive || punct(k, '}')) break;
      if (punct(k, '{')) {
        if (!assigned && parens == 0) break;
        k = after_block(k, limit);
        continue;
      }
      if (punct(k, ';')) {
        terminated = true;
        last = k;
        break;
      }
      if (punct(k, '(') || punct(k, '[')) ++parens;
      if ((punct(k, ')') || punct(k, ']')) && parens > 0) --parens;
      if (punct(k, '=') && parens == 0) assigned = true;
      tail.push_back(static_cast<std::uint32_t>(k));
      ++k;
    }
    // Without semicolon the type ends at its brace and what follows is the next item.
    if (!terminated) tail.clear();
    i = last + 1;

    if (owner != 0) {
      Context inner = context;
      inner.owner = owner;
      emit_other(close, last, inner, tail.empty() ? Glue::Closer : Glue::None);
      return;
    }
    std::string name = head.tag;
    if (head.is_typedef) {
      const std::string alias = typedef_name(tail, 0, tail.size());
      if (!alias.empty()) name = alias;
    }
    Piece piece = make_piece(name.empty() ? PieceKind::Other : PieceKind::Type, first, last, context);
    if (!name.empty()) {
      piece.name = context.prefix + name;
      piece.signature_end = tok(open).begin;
    }
    piece.small = single_line(piece.begin, piece.end);
    if (oversized(piece)) add_breaks(open, close, head.is_enum, piece);
    add_calls(first, last + 1, false);
    emit(std::move(piece));
  }

  void finish_declaration(std::size_t first, std::size_t last, const Header& h, const Context& context) {
    const std::size_t p = after_template_clauses(h);
    std::string name;
    if (p < h.size() && ident(h[p], "typedef")) {
      name = typedef_name(h, p + 1, h.size());
    } else if (cpp_ && p + 2 < h.size() && ident(h[p], "using") && ident(h[p + 1]) && punct(h[p + 2], '=')) {
      name = std::string(word(h[p + 1]));
    }
    if (name.empty()) {
      emit_other(first, last, context);
      return;
    }
    Piece piece = make_piece(PieceKind::Type, first, last, context);
    piece.name = context.prefix + name;
    piece.small = single_line(piece.begin, piece.end);
    piece.signature_end = piece.end;
    add_calls(first, last + 1, false);
    emit(std::move(piece));
  }

  // The semicolon that directly follows a block, on the same line, belongs to the block.
  std::size_t with_semicolon(std::size_t last, std::size_t limit) const {
    const std::size_t next = next_significant(last + 1, limit);
    return next < limit && punct(next, ';') && !tok(next).first_on_line ? next : last;
  }

  // Takes the item that starts at token `first` from the syntax tree when the
  // tree has a definition there that is free of errors and whose body is
  // delimited exactly as the braces delimit it. The name is read from the
  // header like any other; the tree's name stands in where the header rules
  // find none.
  bool take_from_tree(std::size_t first, std::size_t limit, const Context& context, std::size_t& i) {
    const auto it = std::lower_bound(tree_starts_.begin(), tree_starts_.end(), tok(first).begin,
                                     [&](std::uint32_t index, std::uint32_t begin) {
                                       return facts_.functions[index].begin < begin;
                                     });
    if (it == tree_starts_.end()) return false;
    const TreeFunction& f = facts_.functions[*it];
    if (f.begin != tok(first).begin || !f.clean || !spelled_in_header(f)) return false;
    Header h;
    std::size_t open = first;
    while (open < limit && tok(open).begin < f.body_begin) {
      if (inert_[open] != 0 || tok(open).kind == TokenKind::Directive) return false;
      if (tok(open).kind == TokenKind::Comment) {
        ++open;
      } else if (punct(open, '{')) {
        open = after_block(open, limit);
        h.push_back(static_cast<std::uint32_t>(open - 1));
      } else {
        h.push_back(static_cast<std::uint32_t>(open++));
      }
    }
    if (open >= limit || tok(open).begin != f.body_begin || !punct(open, '{')) return false;
    if (partner_[open] == kNone || partner_[open] >= limit || tok(partner_[open]).end != f.end) return false;
    const Classified what = classify(h, open);
    const std::size_t last = with_semicolon(partner_[open], limit);
    emit_function(first, open, last, what.kind == Block::Function ? what.name : f.name, context);
    i = last + 1;
    return true;
  }

  // A block inside parentheses or after '=' is normally an initialiser or a
  // statement expression. When the syntax tree knows it as the body of a
  // function that starts on a line of its own after the item began, the
  // parentheses of the item are not balanced: the item ends before that
  // function. Returns the index of the token the function starts with, or kNone.
  std::size_t function_inside(std::size_t first, std::size_t open) const {
    const TreeFunction* f = facts_.function_with_body(tok(open).begin);
    if (f == nullptr || f->begin <= tok(first).begin) return kNone;
    const auto it = std::lower_bound(tokens_.begin() + static_cast<std::ptrdiff_t>(first),
                                     tokens_.begin() + static_cast<std::ptrdiff_t>(open), f->begin,
                                     [](const Token& token, std::uint32_t begin) { return token.begin < begin; });
    const auto k = static_cast<std::size_t>(it - tokens_.begin());
    if (k <= first || k >= open || tok(k).begin != f->begin || !tok(k).first_on_line || inert_[k] != 0) return kNone;
    return k;
  }

  void scan_item(std::size_t& i, std::size_t limit, const Context& context) {
    const std::size_t first = i;
    if (take_from_tree(first, limit, context, i)) return;
    if (scan_declaration(i, limit, context, true)) return;
    // The parentheses of the item never closed. Once more, without counting them.
    i = first;
    scan_declaration(i, limit, context, false);
  }

  // One item, from its first token to its semicolon or the end of its block.
  // Returns false, having done nothing, when `count_parens` is set and the item
  // ends inside parentheses.
  bool scan_declaration(std::size_t& i, std::size_t limit, const Context& context, bool count_parens) {
    const std::size_t first = i;
    Header h;
    int parens = 0;
    bool assigned = false;
    std::size_t last = first;
    while (i < limit && h.size() < kMaxHeader) {
      if (!significant(i)) {
        ++i;
        continue;
      }
      const Token& token = tok(i);
      if (token.kind == TokenKind::Directive) {
        // #define and #include end an item that has no terminator; a conditional may
        // stand inside a declaration, but not after a macro invocation on a line of its own
        if (parens == 0 && (!conditional(i) || lone_invocation(h))) break;
        ++i;
        continue;
      }
      if (token.kind == TokenKind::Identifier && token.first_on_line && parens == 0 && !assigned &&
          lone_invocation(h)) {
        break;  // MACRO(...) on the line above is a declaration of its own
      }
      if (token.kind == TokenKind::Punct) {
        const char c = text_[token.begin];
        if (c == '{' && (parens > 0 || assigned)) {  // a block inside parentheses, an initialiser
          if (const std::size_t next = function_inside(first, i); next != kNone) {
            std::size_t end = next - 1;
            while (end > first && !significant(end)) --end;
            emit_other(first, end, context);
            i = next;
            return true;
          }
          i = after_block(i, limit);
          last = i - 1;
          continue;
        }
        if (c == '{') {
          Classified what = classify(h, i);
          if ((what.kind == Block::Linkage || what.kind == Block::Namespace) && context.nesting >= kMaxNesting) {
            what.kind = Block::Opaque;
          }
          const std::size_t open = i;
          switch (what.kind) {
            case Block::Linkage:
            case Block::Namespace:
              scan_container(first, open, i, limit, context);
              return true;
            case Block::Function:
              last = with_semicolon(after_block(open, limit) - 1, limit);
              i = last + 1;
              emit_function(first, open, last, what.name, context);
              return true;
            case Block::Type:
              scan_type(first, open, what.head, i, limit, context);
              return true;
            case Block::MemberInitialiser:
              i = after_block(open, limit);
              last = i - 1;
              h.push_back(static_cast<std::uint32_t>(last));  // the '}' stands for the initialiser
              continue;
            case Block::Opaque:
              last = with_semicolon(after_block(open, limit) - 1, limit);
              i = last + 1;
              emit_other(first, last, context);
              return true;
          }
        }
        if (c == '}') break;  // a brace without partner
        if (c == ';') {       // inside parentheses as well: they are not balanced then
          last = i++;
          finish_declaration(first, last, h, context);
          return true;
        }
        // only brackets that have their partner inside the item are levels
        const bool paired = partner_[i] != kNone && partner_[i] >= first && partner_[i] < limit;
        if (count_parens && paired && (c == '(' || c == '[')) ++parens;
        if (paired && (c == ')' || c == ']') && parens > 0) --parens;
        if (c == '=' && parens == 0 && !operator_name(h)) assigned = true;
      }
      h.push_back(static_cast<std::uint32_t>(i));
      last = i++;

      if (token.kind != TokenKind::Punct) continue;
      const char c = text_[token.begin];
      if (c == ':' && context.in_type && h.size() == 2 &&
          (ident(h[0], "public") || ident(h[0], "private") || ident(h[0], "protected"))) {
        emit_other(first, last, context, Glue::Opener);  // opens a section of the class
        return true;
      }
      if (c == ')' && parens == 0 && !assigned && count_parens) {
        Declarator declarator;
        std::size_t macro = 0;
        if (macro_bodied(h, last, limit, &declarator, &macro)) {
          emit_macro_bodied(first, h, declarator, macro, context);
          return true;
        }
      }
    }
    if (parens > 0 && count_parens) return false;
    // The item ended without semicolon and without body.
    emit_other(first, last, context);
    if (i <= last) i = last + 1;
    return true;
  }

  std::string_view text_;
  bool cpp_;
  const TreeFacts& facts_;
  const ParseOptions& options_;
  LineIndex lines_;
  std::vector<Token> tokens_;
  std::vector<std::uint8_t> inert_;
  std::vector<std::uint32_t> partner_;
  std::vector<std::uint32_t> tree_starts_;
  Outline out_;
};

}  // namespace

Outline outline_c_family(std::string_view text, bool cpp, const SyntaxTree& tree, const ParseOptions& options) {
  const TreeFacts facts = collect_c_family_facts(tree, text);
  return Scanner(text, cpp, facts, options).run();
}

bool has_cpp_constructs(std::string_view text) {
  const std::vector<Token> tokens = lex_c_family(text);
  std::vector<std::size_t> code;  // tokens other than comments
  code.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i].kind != TokenKind::Comment) code.push_back(i);
  }
  const auto is_word = [&](std::size_t k) { return k < code.size() && tokens[code[k]].kind == TokenKind::Identifier; };
  const auto text_of = [&](std::size_t k) {
    return k < code.size() ? token_text(text, tokens[code[k]]) : std::string_view();
  };
  const auto is_symbol = [&](std::size_t k, std::string_view s) {
    return k < code.size() && tokens[code[k]].kind == TokenKind::Punct && text_of(k) == s;
  };
  for (std::size_t k = 0; k < code.size(); ++k) {
    if (is_symbol(k, "::")) {
      // [[vendor::attribute]] is C as well
      const bool attribute = k >= 2 && is_symbol(k - 2, "[");
      if (k > 0 && is_word(k - 1) && is_word(k + 1) && !attribute) return true;
      continue;
    }
    if (!is_word(k)) continue;
    const std::string_view w = text_of(k);
    if (w == "template" && is_symbol(k + 1, "<")) return true;
    if (w == "namespace" && (is_word(k + 1) || is_symbol(k + 1, "{"))) return true;
    if (w == "class" && is_word(k + 1) &&
        (is_symbol(k + 2, "{") || is_symbol(k + 2, ":") || is_symbol(k + 2, ";") || text_of(k + 2) == "final")) {
      return true;
    }
    if ((w == "public" || w == "private" || w == "protected") && is_symbol(k + 1, ":")) return true;
    if (w == "operator" && k + 1 < code.size() &&
        (tokens[code[k + 1]].kind == TokenKind::Punct || text_of(k + 1) == "new" || text_of(k + 1) == "delete")) {
      return true;
    }
    if ((w == "using" || w == "virtual") && is_word(k + 1)) return true;
    if ((w == "static_cast" || w == "dynamic_cast" || w == "const_cast" || w == "reinterpret_cast") &&
        is_symbol(k + 1, "<")) {
      return true;
    }
  }
  return false;
}

}  // namespace prag::parse::detail
