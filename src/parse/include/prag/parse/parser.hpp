// Structure-aware splitting of source files into retrievable units, and the
// static call graph of a corpus.
//
// Files are parsed with tree-sitter. Every function definition becomes one unit
// (with its leading comment); type definitions and macros become units of their
// own; whatever top-level text remains is grouped into regions no larger than
// ParseOptions::max_region_bytes. Units are contiguous byte ranges that
// partition the file: the first starts at byte 0, each starts where the
// previous one ends, and the last ends at the end of the file, so concatenating
// the unit texts reproduces the file byte for byte. Blank lines between two
// units belong to the unit that follows them. Files in a language without a
// grammar are split into line-aligned regions.
//
// Names. A member function defined inside a class is a unit of its own, named
// "Class::method" (C++) or "Class.method" (Python); the class is then a Type
// unit that holds what precedes its first member function, and the residue
// between and after the member functions forms regions named "Class#2",
// "Class#3", ... A unit larger than max_unit_bytes is cut at statement
// boundaries; the parts after the first are regions named "name#2", "name#3", ...
// of at most max_unit_bytes each. Regions without a name are residual text and
// at most max_region_bytes large.
// C++ operator functions are named as they are written: "operator new[]",
// "operator==". A definition made by a macro invocation followed by a body,
// `TEST(suite, name) { ... }`, is named after the arguments: "suite_name".
//
// Structure. Text that only opens or closes something around units - a
// conditional directive, `extern "C" {`, the braces of a namespace, the closing
// brace of a class - and stands directly between two units belongs to the unit
// it closes (the one above) or opens (the one below) instead of forming a
// region of a few bytes: `#if X` above a function and `#endif` below it are
// part of the function's unit.
//
// Macro-heavy C. Byte ranges come from brace matching over a token stream in
// which comments, literals and preprocessor lines are single tokens, so they
// do not depend on what tree-sitter makes of unknown macros; the syntax tree
// confirms or decides names where it parsed a definition without errors.
// Braces inside the alternative branches of a conditional are counted once.
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/types.hpp"
#include "prag/graph/call_graph.hpp"

namespace prag::parse {

struct ParseOptions {
  std::size_t max_region_bytes = 4096;  // upper bound for residual regions
  std::size_t max_unit_bytes = 16384;   // larger units are split at statement boundaries
  bool attach_leading_comment = true;   // a comment block directly above a unit belongs to it
  // One-line object-like macros (`#define MI_KiB 1024`) and one-line type
  // declarations without body (`typedef struct mi_heap_s mi_heap_t;`) join the
  // region around them instead of becoming units (DESIGN.md 6.4: small adjacent
  // non-function siblings are merged). Function-like macros, macros that span
  // several lines and types with a body are always units. false: every macro
  // and every type declaration is a unit of its own.
  bool merge_small_definitions = true;
};

struct ParsedFile {
  std::string path;      // normalised
  std::string language;  // "c", "cpp", "python" or "" for the generic fallback
  std::vector<SourceUnit> units;  // ids are 0..n-1 within the file, in source order
  bool had_syntax_errors = false;  // tree-sitter reported ERROR or MISSING nodes
};

// Language key for a path, by extension: ".c" -> "c"; ".h" -> "c" unless the
// text contains C++ constructs; ".cc/.cpp/.cxx/.hpp/.hh" -> "cpp"; ".py" -> "python".
// Returns "" when no grammar applies.
std::string detect_language(std::string_view path, std::string_view text);

// Never throws on malformed source; syntax errors are reported through
// had_syntax_errors and the affected text still ends up in some unit.
ParsedFile parse_source(std::string_view path, std::string_view text, const ParseOptions& options = {});

// Number of ERROR and MISSING nodes tree-sitter reports for `text` in
// `language`; 0 for languages without a grammar. The edit engine uses it to
// refuse edits that make a file less well-formed than it was.
std::size_t count_syntax_errors(std::string_view language, std::string_view text);

// Byte range [begin, end) of the unit called `name` in `text`, leading comment
// included; returns false when no unit has that name. Function definitions are
// preferred to other units; when two or more units of the deciding kind share
// the name (two definitions in alternative #if branches, say), the name is
// ambiguous and the function returns false rather than guess. A name without qualification
// also finds a member ("method" finds "Class::method") when no unit carries the
// name itself. The range of a unit that was split covers all its parts, the
// range of a class all its members.
bool find_unit(std::string_view path, std::string_view text, std::string_view name, std::size_t* begin,
               std::size_t* end);

// Call graph of a corpus. `units[i].id` must equal i. An edge u -> v exists when
// the body of unit u calls an identifier that names function or function-like
// macro v; a name with several definitions yields one edge per definition.
graph::CallGraph build_call_graph(std::span<const SourceUnit> units);

}  // namespace prag::parse
