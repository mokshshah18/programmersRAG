#include "prag/graph/call_graph.hpp"

#include <algorithm>

namespace prag::graph {

NeighborGraph NeighborGraph::from_edges(std::size_t n, std::span<const std::pair<UnitId, UnitId>> edges) {
  std::vector<std::pair<UnitId, UnitId>> sorted;
  sorted.reserve(edges.size());
  for (const auto& [from, to] : edges) {
    if (from >= n || to >= n || from == to) continue;
    sorted.emplace_back(from, to);
  }
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  NeighborGraph g;
  g.offsets_.assign(n + 1, 0);
  g.targets_.reserve(sorted.size());
  for (const auto& edge : sorted) ++g.offsets_[static_cast<std::size_t>(edge.first) + 1];
  for (std::size_t i = 0; i < n; ++i) g.offsets_[i + 1] += g.offsets_[i];
  for (const auto& edge : sorted) g.targets_.push_back(edge.second);
  return g;
}

NeighborGraph NeighborGraph::from_csr(std::vector<std::uint32_t> offsets, std::vector<UnitId> targets) {
  NeighborGraph g;
  const bool consistent = !offsets.empty() && offsets.front() == 0 && offsets.back() == targets.size() &&
                          std::is_sorted(offsets.begin(), offsets.end());
  if (!consistent) return g;
  g.offsets_ = std::move(offsets);
  g.targets_ = std::move(targets);
  return g;
}

std::span<const UnitId> NeighborGraph::of(UnitId unit) const noexcept {
  if (static_cast<std::size_t>(unit) + 1 >= offsets_.size()) return {};
  const std::uint32_t begin = offsets_[unit];
  const std::uint32_t end = offsets_[static_cast<std::size_t>(unit) + 1];
  return {targets_.data() + begin, static_cast<std::size_t>(end - begin)};
}

CallGraph CallGraph::from_edges(std::size_t n, std::span<const std::pair<UnitId, UnitId>> edges,
                                Definitions definitions) {
  CallGraph g;
  g.callees_ = NeighborGraph::from_edges(n, edges);
  std::vector<std::pair<UnitId, UnitId>> reversed;
  reversed.reserve(edges.size());
  for (const auto& [from, to] : edges) reversed.emplace_back(to, from);
  g.callers_ = NeighborGraph::from_edges(n, reversed);
  for (auto& [name, ids] : definitions) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }
  g.definitions_ = std::move(definitions);
  return g;
}

std::span<const UnitId> CallGraph::definitions(const std::string& name) const {
  const auto it = definitions_.find(name);
  if (it == definitions_.end()) return {};
  return it->second;
}

}  // namespace prag::graph
