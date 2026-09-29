// The outline of a file: the top-level pieces a language front end found, before
// they are turned into units. Private to the parse module.
//
// Front ends (c_family.cpp, python.cpp) only say where things are; everything
// that makes units a partition of the file (comment attachment, boundaries,
// merging, splitting) happens once, in assemble.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "prag/parse/parser.hpp"
#include "syntax_tree.hpp"

namespace prag::parse::detail {

enum class PieceKind : std::uint8_t {
  Function,
  Type,
  Macro,
  Other,    // declarations, directives, anything residual
  Comment,  // a comment between pieces; attached to the piece below it or residual
};

// Structural residue: text that opens or closes something around units and
// says nothing by itself. Where it stands directly between units it goes to the
// unit it closes or opens instead of forming a region.
enum class Glue : std::uint8_t {
  None,
  If,      // #if, #ifdef, #ifndef: opens
  Else,    // #else, #elif: closes and opens
  Endif,   // closes
  Opener,  // extern "C" {, namespace {
  Closer,  // the closing brace of a container or of a class
};

struct Piece {
  PieceKind kind = PieceKind::Other;
  Glue glue = Glue::None;
  bool small = false;     // Type, Macro: on one line, so it may join the surrounding region
  bool callable = false;  // Macro: function-like
  bool head = false;      // Type: opening part of a type whose members became pieces of their own
  std::uint32_t begin = 0;  // first byte of the first token
  std::uint32_t end = 0;    // one past the last byte of the last token
  std::uint32_t signature_end = 0;  // declaration text is [begin, signature_end); 0 = none
  std::uint32_t doc_begin = 0;      // documentation string inside the piece (Python); empty when equal
  std::uint32_t doc_end = 0;
  std::uint32_t owner = 0;  // 0 = file scope, else 1-based index into Outline::owners
  std::string name;
  // Offsets at which the piece may be cut when it is too large, ascending.
  // breaks[0]: between the statements of the body; breaks[1]: between nested statements.
  std::vector<std::uint32_t> breaks[2];
};

struct CallSite {
  std::uint32_t offset = 0;  // of the called identifier
  std::string_view name;     // view into the source text
};

struct Outline {
  std::vector<Piece> pieces;        // ascending, not overlapping
  std::vector<CallSite> calls;      // ascending offsets
  std::vector<std::string> owners;  // qualified names of the types that were split into members
};

// C and C++: brace matching over the token stream, checked against the syntax
// tree where the tree is free of errors. `tree` may be invalid.
Outline outline_c_family(std::string_view text, bool cpp, const SyntaxTree& tree, const ParseOptions& options);

// Python: read off the syntax tree.
Outline outline_python(std::string_view text, const SyntaxTree& tree, const ParseOptions& options);

// No grammar: paragraphs separated by blank lines.
Outline outline_plain(std::string_view text);

// True when the text of a ".h" file uses constructs that only C++ has.
bool has_cpp_constructs(std::string_view text);

// Turns an outline into units that partition `text`.
std::vector<SourceUnit> assemble_units(std::string_view path, std::string_view language, std::string_view text,
                                       Outline outline, const ParseOptions& options);

}  // namespace prag::parse::detail
