// Lexer for C and C++ source, private to the parse module.
//
// The outline of a macro-heavy file cannot be read off the syntax tree alone:
// an unknown macro in a declaration makes tree-sitter end a function early or
// start it late. Braces, however, stay balanced whatever the macros expand to,
// provided comments, literals and preprocessor lines are recognised. This lexer
// provides exactly that: comments and whole preprocessor directives are single
// tokens, so a brace inside `#define` or inside a string is never counted.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace prag::parse::detail {

enum class TokenKind : std::uint8_t {
  Identifier,  // also keywords
  Number,
  String,      // with its prefix, raw strings included
  Character,
  Punct,       // one character, except "::" and "->"
  Directive,   // '#' to the end of the logical line, continuation lines included
  Comment,     // one line comment or one block comment
};

struct Token {
  TokenKind kind = TokenKind::Punct;
  bool first_on_line = false;  // nothing but white space precedes it on its line
  std::uint32_t begin = 0;
  std::uint32_t end = 0;  // one past the last byte
};

// Never fails: unterminated literals end at the end of their line, an
// unterminated block comment at the end of the text.
std::vector<Token> lex_c_family(std::string_view text);

enum class DirectiveKind : std::uint8_t {
  If,      // #if, #ifdef, #ifndef
  Else,    // #else, #elif, #elifdef, #elifndef
  Endif,
  Define,
  Other,   // #include, #undef, #pragma, #error, ...
};

struct DirectiveInfo {
  DirectiveKind kind = DirectiveKind::Other;
  std::string_view keyword;        // "define", "ifdef", ...
  std::string_view name;           // Define: the macro name
  bool function_like = false;      // Define: the name is directly followed by '('
  std::uint32_t name_end = 0;      // Define: offset one past the name
  std::uint32_t body_begin = 0;    // Define: offset of the replacement text (after the parameter list)
};

DirectiveInfo inspect_directive(std::string_view text, const Token& token);

inline std::string_view token_text(std::string_view text, const Token& token) {
  return text.substr(token.begin, token.end - token.begin);
}

inline bool is_punct(std::string_view text, const Token& token, char c) {
  return token.kind == TokenKind::Punct && token.end - token.begin == 1 && text[token.begin] == c;
}

// Byte offset <-> line number.
class LineIndex {
 public:
  explicit LineIndex(std::string_view text);

  std::uint32_t line_of(std::size_t offset) const;  // 1-based line containing the byte at `offset`
  std::size_t line_count() const noexcept { return starts_.size(); }
  // Offset of the first byte of the line after the one containing `offset`
  // (the size of the text when there is none).
  std::size_t next_line(std::size_t offset) const;

 private:
  std::vector<std::uint32_t> starts_;
  std::size_t size_ = 0;
};

}  // namespace prag::parse::detail
