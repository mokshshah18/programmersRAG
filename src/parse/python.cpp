// Outline of Python files, read off the syntax tree.
//
// Python has no preprocessor, so the tree is reliable and the block structure
// is already resolved by the grammar's scanner. Functions and classes at module
// level become pieces; a class that defines methods is split into its head and
// one piece per method, named Class.method.
#include <algorithm>
#include <cstring>
#include <string>

#include "lexer.hpp"
#include "outline.hpp"

namespace prag::parse::detail {

namespace {

constexpr int kMaxNesting = 48;

inline bool is_type(TSNode node, const char* type) { return std::strcmp(ts_node_type(node), type) == 0; }

class PythonOutline {
 public:
  PythonOutline(std::string_view text, const ParseOptions& options) : text_(text), options_(options), lines_(text) {}

  Outline run(TSNode root) {
    add_children(root, "", 0, 0);
    collect_calls(root);
    return std::move(out_);
  }

 private:
  // The definition inside a decorated_definition, else the node itself.
  static TSNode definition_of(TSNode node) {
    if (!is_type(node, "decorated_definition")) return node;
    const TSNode inner = field(node, "definition");
    return ts_node_is_null(inner) ? node : inner;
  }

  static bool defines_function(TSNode node) { return is_type(definition_of(node), "function_definition"); }

  // End of the ':' that closes the header of a definition.
  std::uint32_t header_end(TSNode definition, TSNode body) const {
    const std::uint32_t limit = ts_node_is_null(body) ? ts_node_end_byte(definition) : ts_node_start_byte(body);
    std::uint32_t end = limit;
    const std::uint32_t count = ts_node_child_count(definition);
    for (std::uint32_t i = 0; i < count; ++i) {
      const TSNode child = ts_node_child(definition, i);
      if (ts_node_start_byte(child) >= limit) break;
      if (!ts_node_is_named(child) && is_type(child, ":")) end = ts_node_end_byte(child);
    }
    return end;
  }

  // The documentation string that opens a body, if any.
  void set_docstring(TSNode body, Piece& piece) const {
    if (ts_node_is_null(body)) return;
    const std::uint32_t count = ts_node_named_child_count(body);
    for (std::uint32_t i = 0; i < count; ++i) {
      const TSNode child = ts_node_named_child(body, i);
      if (is_type(child, "comment")) continue;
      if (is_type(child, "expression_statement") && ts_node_named_child_count(child) == 1) {
        const TSNode value = ts_node_named_child(child, 0);
        if (is_type(value, "string") || is_type(value, "concatenated_string")) {
          piece.doc_begin = ts_node_start_byte(value);
          piece.doc_end = ts_node_end_byte(value);
        }
      }
      return;
    }
  }

  // Cut points between the statements of a block, nested blocks one level down.
  void add_breaks(TSNode block, int level, int depth, Piece& piece) const {
    if (ts_node_is_null(block) || depth > kMaxNesting) return;
    const std::uint32_t count = ts_node_named_child_count(block);
    TSNode previous{};
    for (std::uint32_t i = 0; i < count; ++i) {
      const TSNode child = ts_node_named_child(block, i);
      if (ts_node_end_byte(child) <= ts_node_start_byte(child)) continue;
      if (!ts_node_is_null(previous)) {
        const std::size_t at = lines_.next_line(ts_node_end_byte(previous) - 1);
        if (at > piece.begin && at <= ts_node_start_byte(child)) {
          piece.breaks[level].push_back(static_cast<std::uint32_t>(at));
        }
      }
      previous = child;
      // blocks of compound statements
      walk(child, [&](TSNode node) {
        if (is_type(node, "block")) {
          add_breaks(node, 1, depth + 1, piece);
          return false;
        }
        return true;
      });
    }
  }

  bool oversized(const Piece& piece) const {
    const std::size_t limit = piece.kind == PieceKind::Other ? options_.max_region_bytes : options_.max_unit_bytes;
    return piece.end - piece.begin > limit;
  }

  Piece make_piece(PieceKind kind, TSNode node, std::uint32_t owner) const {
    Piece piece;
    piece.kind = kind;
    piece.begin = ts_node_start_byte(node);
    piece.end = ts_node_end_byte(node);
    piece.owner = owner;
    return piece;
  }

  void add_function(TSNode outer, const std::string& prefix, std::uint32_t owner) {
    const TSNode definition = definition_of(outer);
    const TSNode body = field(definition, "body");
    Piece piece = make_piece(PieceKind::Function, outer, owner);
    piece.name = prefix + std::string(node_text(text_, field(definition, "name")));
    piece.signature_end = header_end(definition, body);
    set_docstring(body, piece);
    if (oversized(piece)) finish_breaks(body, piece);
    out_.pieces.push_back(std::move(piece));
  }

  void finish_breaks(TSNode body, Piece& piece) const {
    add_breaks(body, 0, 0, piece);
    for (auto& level : piece.breaks) {
      std::sort(level.begin(), level.end());
      level.erase(std::unique(level.begin(), level.end()), level.end());
    }
  }

  void add_class(TSNode outer, const std::string& prefix, std::uint32_t owner, int nesting) {
    const TSNode definition = definition_of(outer);
    const TSNode body = field(definition, "body");
    const std::string name = prefix + std::string(node_text(text_, field(definition, "name")));

    bool has_methods = false;
    if (!ts_node_is_null(body) && nesting < kMaxNesting) {
      const std::uint32_t count = ts_node_named_child_count(body);
      for (std::uint32_t i = 0; i < count && !has_methods; ++i) {
        has_methods = defines_function(ts_node_named_child(body, i));
      }
    }

    Piece piece = make_piece(PieceKind::Type, outer, owner);
    piece.name = name;
    piece.signature_end = header_end(definition, body);
    set_docstring(body, piece);
    if (!has_methods) {
      if (oversized(piece)) finish_breaks(body, piece);
      out_.pieces.push_back(std::move(piece));
      return;
    }
    out_.owners.push_back(name);
    const auto id = static_cast<std::uint32_t>(out_.owners.size());
    piece.head = true;
    piece.owner = id;
    piece.end = piece.signature_end;
    out_.pieces.push_back(std::move(piece));
    add_children(body, name + ".", id, nesting + 1);
  }

  void add_children(TSNode parent, const std::string& prefix, std::uint32_t owner, int nesting) {
    const std::uint32_t count = ts_node_named_child_count(parent);
    for (std::uint32_t i = 0; i < count; ++i) {
      const TSNode child = ts_node_named_child(parent, i);
      if (ts_node_end_byte(child) <= ts_node_start_byte(child)) continue;  // MISSING
      const TSNode definition = definition_of(child);
      if (is_type(child, "comment")) {
        out_.pieces.push_back(make_piece(PieceKind::Comment, child, owner));
      } else if (is_type(definition, "function_definition") && !ts_node_is_null(field(definition, "name"))) {
        add_function(child, prefix, owner);
      } else if (is_type(definition, "class_definition") && !ts_node_is_null(field(definition, "name"))) {
        add_class(child, prefix, owner, nesting);
      } else {
        out_.pieces.push_back(make_piece(PieceKind::Other, child, owner));
      }
    }
  }

  // name(...), object.name(...), @name and @object.name
  void collect_calls(TSNode root) {
    const auto add = [&](TSNode callee) {
      if (ts_node_is_null(callee)) return;
      if (is_type(callee, "attribute")) callee = field(callee, "attribute");
      if (!ts_node_is_null(callee) && is_type(callee, "identifier")) {
        out_.calls.push_back({ts_node_start_byte(callee), node_text(text_, callee)});
      }
    };
    walk(root, [&](TSNode node) {
      if (!ts_node_is_named(node)) return false;
      if (is_type(node, "call")) {
        add(field(node, "function"));
      } else if (is_type(node, "decorator") && ts_node_named_child_count(node) > 0) {
        add(ts_node_named_child(node, 0));
      }
      return true;
    });
    std::sort(out_.calls.begin(), out_.calls.end(),
              [](const CallSite& a, const CallSite& b) { return a.offset < b.offset; });
  }

  std::string_view text_;
  const ParseOptions& options_;
  LineIndex lines_;
  Outline out_;
};

}  // namespace

Outline outline_python(std::string_view text, const SyntaxTree& tree, const ParseOptions& options) {
  if (!tree.valid()) return outline_plain(text);
  return PythonOutline(text, options).run(tree.root());
}

Outline outline_plain(std::string_view text) {
  Outline out;
  std::size_t line = 0;
  bool open = false;
  Piece piece;
  while (line < text.size()) {
    std::size_t end = text.find('\n', line);
    if (end == std::string_view::npos) end = text.size();
    std::size_t first = line;
    while (first < end && (text[first] == ' ' || text[first] == '\t' || text[first] == '\r')) ++first;
    std::size_t last = end;
    while (last > first && (text[last - 1] == ' ' || text[last - 1] == '\t' || text[last - 1] == '\r')) --last;
    if (first == last) {  // a blank line ends the paragraph
      if (open) out.pieces.push_back(piece);
      open = false;
    } else {
      if (!open) {
        piece = Piece{};
        piece.begin = static_cast<std::uint32_t>(first);
        open = true;
      }
      piece.end = static_cast<std::uint32_t>(last);
    }
    line = end + 1;
  }
  if (open) out.pieces.push_back(piece);
  return out;
}

}  // namespace prag::parse::detail
