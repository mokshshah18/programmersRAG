#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "prag/parse/parser.hpp"

namespace prag::parse {

namespace {

// `#define name(` with nothing between the name and the parenthesis.
bool is_function_like_macro(const SourceUnit& unit) {
  if (unit.kind != UnitKind::Macro || unit.name.empty()) return false;
  const std::string_view text = unit.signature.empty() ? std::string_view(unit.text) : std::string_view(unit.signature);
  const std::size_t define = text.find("define");
  if (define == std::string_view::npos) return false;
  const std::size_t name = text.find(unit.name, define + 6);
  if (name == std::string_view::npos) return false;
  const std::size_t after = name + unit.name.size();
  return after < text.size() && text[after] == '(';
}

// Names under which calls reach a definition: the name itself and, for a
// member, the unqualified name. A Python class is called through __init__.
std::vector<std::string> call_names(const SourceUnit& unit) {
  std::vector<std::string> names;
  const std::string& name = unit.name;
  names.push_back(name);
  if (name.compare(0, 8, "operator") == 0) return names;
  const std::size_t scope = name.rfind("::");
  const std::size_t dot = name.rfind('.');
  if (scope != std::string::npos) {
    names.push_back(name.substr(scope + 2));
  } else if (dot != std::string::npos) {
    const std::string member = name.substr(dot + 1);
    if (member == "__init__") {
      const std::string owner = name.substr(0, dot);
      const std::size_t outer = owner.rfind('.');
      names.push_back(outer == std::string::npos ? owner : owner.substr(outer + 1));
    }
    names.push_back(member);
  }
  return names;
}

}  // namespace

graph::CallGraph build_call_graph(std::span<const SourceUnit> units) {
  graph::CallGraph::Definitions definitions;
  for (std::size_t i = 0; i < units.size(); ++i) {
    const SourceUnit& unit = units[i];
    if (unit.name.empty()) continue;
    if (unit.kind != UnitKind::Function && !is_function_like_macro(unit)) continue;
    for (std::string& name : call_names(unit)) {
      if (!name.empty()) definitions[std::move(name)].push_back(static_cast<UnitId>(i));
    }
  }

  std::vector<std::pair<UnitId, UnitId>> edges;
  for (std::size_t i = 0; i < units.size(); ++i) {
    for (const std::string& callee : units[i].callees) {
      const auto found = definitions.find(callee);
      if (found == definitions.end()) continue;
      for (const UnitId target : found->second) {
        if (target != i) edges.emplace_back(static_cast<UnitId>(i), target);
      }
    }
  }
  return graph::CallGraph::from_edges(units.size(), edges, std::move(definitions));
}

}  // namespace prag::parse
