// Shared value types used across modules.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prag {

using UnitId = std::uint32_t;
inline constexpr UnitId kNoUnit = 0xFFFFFFFFu;

enum class UnitKind : std::uint8_t {
  Function = 0,  // a function or method definition
  Type = 1,      // struct, union, enum, class or typedef
  Macro = 2,     // preprocessor definition
  Region = 3,    // residual top-level text that belongs to no other unit
};

const char* to_string(UnitKind kind);

// One retrievable piece of source code. Units are the rows of the vector index:
// unit i owns embedding row i.
struct SourceUnit {
  UnitId id = kNoUnit;
  std::string path;       // absolute, forward slashes
  std::string language;   // "c", "cpp", ... ; empty when parsed by the generic fallback
  UnitKind kind = UnitKind::Region;
  std::string name;       // identifier; empty for regions
  std::string signature;  // declaration text up to (not including) the body
  std::string doc;        // leading comment with comment markers removed
  std::string text;       // full text of the unit, leading comment included
  std::uint32_t start_line = 0;  // 1-based, inclusive
  std::uint32_t end_line = 0;    // 1-based, inclusive
  std::uint32_t start_byte = 0;  // offset into the file, leading comment included
  std::uint32_t end_byte = 0;    // one past the last byte
  std::vector<std::string> callees;  // distinct called identifiers, first-appearance order
  std::uint64_t content_hash = 0;    // hash64(text)
  std::uint32_t tokens = 0;          // estimated or model-counted token length of text
};

// Relation through which the tracer reached a unit.
enum class EdgeKind : std::uint8_t {
  Seed = 0,     // accepted on its own similarity score
  Callee = 1,   // parent calls this unit
  Caller = 2,   // this unit calls parent
  Sibling = 3,  // same density mode as parent
};

const char* to_string(EdgeKind kind);

// One message of a chat exchange with a language model.
struct ChatMessage {
  std::string role;  // "system", "user" or "assistant"
  std::string content;
};

}  // namespace prag
