#include "prag/core/types.hpp"

namespace prag {

const char* to_string(UnitKind kind) {
  switch (kind) {
    case UnitKind::Function: return "function";
    case UnitKind::Type: return "type";
    case UnitKind::Macro: return "macro";
    case UnitKind::Region: return "region";
  }
  return "region";
}

const char* to_string(EdgeKind kind) {
  switch (kind) {
    case EdgeKind::Seed: return "seed";
    case EdgeKind::Callee: return "callee";
    case EdgeKind::Caller: return "caller";
    case EdgeKind::Sibling: return "sibling";
  }
  return "seed";
}

}  // namespace prag
