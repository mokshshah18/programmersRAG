// Graphs over the units of a corpus, stored in compressed sparse row form.
//
// CallGraph: an edge u -> v means that the body of unit u calls unit v. Both
// directions are kept. NeighborGraph: a plain directed adjacency list, used for
// the nearest same-mode neighbours of every unit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "prag/core/types.hpp"

namespace prag::graph {

class NeighborGraph {
 public:
  NeighborGraph() = default;

  // Edges may arrive in any order; duplicates and self loops are removed and the
  // targets of every source are sorted ascending. Edges with an end point
  // >= n are ignored.
  static NeighborGraph from_edges(std::size_t n, std::span<const std::pair<UnitId, UnitId>> edges);
  // Takes CSR arrays as they are (offsets has n + 1 entries).
  static NeighborGraph from_csr(std::vector<std::uint32_t> offsets, std::vector<UnitId> targets);

  std::size_t size() const noexcept { return offsets_.empty() ? 0 : offsets_.size() - 1; }
  std::size_t edges() const noexcept { return targets_.size(); }
  bool empty() const noexcept { return targets_.empty(); }

  // Empty span for an id outside the graph.
  std::span<const UnitId> of(UnitId unit) const noexcept;

  const std::vector<std::uint32_t>& offsets() const noexcept { return offsets_; }
  const std::vector<UnitId>& targets() const noexcept { return targets_; }

 private:
  std::vector<std::uint32_t> offsets_;
  std::vector<UnitId> targets_;
};

class CallGraph {
 public:
  using Definitions = std::unordered_map<std::string, std::vector<UnitId>>;

  CallGraph() = default;

  // edges: (caller, callee) pairs. definitions: unit ids defining each name.
  static CallGraph from_edges(std::size_t n, std::span<const std::pair<UnitId, UnitId>> edges,
                              Definitions definitions = {});

  std::size_t size() const noexcept { return callees_.size(); }
  std::size_t edges() const noexcept { return callees_.edges(); }

  std::span<const UnitId> callees(UnitId unit) const noexcept { return callees_.of(unit); }
  std::span<const UnitId> callers(UnitId unit) const noexcept { return callers_.of(unit); }

  // Units defining `name`, ascending; empty when the name is unknown.
  std::span<const UnitId> definitions(const std::string& name) const;

  const NeighborGraph& forward() const noexcept { return callees_; }
  const NeighborGraph& backward() const noexcept { return callers_; }
  const Definitions& all_definitions() const noexcept { return definitions_; }

 private:
  NeighborGraph callees_;
  NeighborGraph callers_;
  Definitions definitions_;
};

}  // namespace prag::graph
