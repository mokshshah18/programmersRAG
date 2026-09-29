#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <utility>
#include <vector>

#include "prag/graph/call_graph.hpp"

using namespace prag;
using namespace prag::graph;

PRAG_TEST(neighbor_graph_sorts_dedupes_and_drops_self_loops) {
  const std::vector<std::pair<UnitId, UnitId>> edges = {{2, 1}, {0, 3}, {0, 1}, {0, 3}, {1, 1}, {3, 0}, {9, 0}, {0, 9}};
  const NeighborGraph g = NeighborGraph::from_edges(4, edges);
  CHECK_EQ(g.size(), std::size_t{4});
  CHECK_EQ(g.edges(), std::size_t{4});
  REQUIRE(g.of(0).size() == 2u);
  CHECK_EQ(g.of(0)[0], UnitId{1});
  CHECK_EQ(g.of(0)[1], UnitId{3});
  CHECK(g.of(1).empty());
  REQUIRE(g.of(2).size() == 1u);
  CHECK_EQ(g.of(2)[0], UnitId{1});
  REQUIRE(g.of(3).size() == 1u);
  CHECK(g.of(4).empty());
  CHECK(g.of(kNoUnit).empty());
}

PRAG_TEST(empty_graph_is_safe) {
  const NeighborGraph g;
  CHECK_EQ(g.size(), std::size_t{0});
  CHECK(g.empty());
  CHECK(g.of(0).empty());
  const NeighborGraph z = NeighborGraph::from_edges(0, {});
  CHECK_EQ(z.size(), std::size_t{0});
  CHECK(z.of(0).empty());
}

PRAG_TEST(csr_round_trip_and_validation) {
  const std::vector<std::pair<UnitId, UnitId>> edges = {{0, 1}, {0, 2}, {2, 0}};
  const NeighborGraph g = NeighborGraph::from_edges(3, edges);
  const NeighborGraph copy = NeighborGraph::from_csr(g.offsets(), g.targets());
  CHECK_EQ(copy.edges(), g.edges());
  CHECK_EQ(copy.of(0).size(), std::size_t{2});
  // Inconsistent arrays are rejected.
  const NeighborGraph bad = NeighborGraph::from_csr({0, 5}, {1});
  CHECK_EQ(bad.size(), std::size_t{0});
}

PRAG_TEST(call_graph_keeps_both_directions) {
  const std::vector<std::pair<UnitId, UnitId>> edges = {{0, 1}, {0, 2}, {1, 2}, {3, 2}};
  CallGraph::Definitions definitions;
  definitions["mi_free"] = {2, 2};
  definitions["mi_malloc"] = {1, 0};
  const CallGraph g = CallGraph::from_edges(4, edges, std::move(definitions));
  CHECK_EQ(g.size(), std::size_t{4});
  CHECK_EQ(g.edges(), std::size_t{4});
  CHECK_EQ(g.callees(0).size(), std::size_t{2});
  REQUIRE(g.callers(2).size() == 3u);
  CHECK_EQ(g.callers(2)[0], UnitId{0});
  CHECK_EQ(g.callers(2)[1], UnitId{1});
  CHECK_EQ(g.callers(2)[2], UnitId{3});
  CHECK(g.callers(0).empty());
  REQUIRE(g.definitions("mi_free").size() == 1u);
  REQUIRE(g.definitions("mi_malloc").size() == 2u);
  CHECK_EQ(g.definitions("mi_malloc")[0], UnitId{0});
  CHECK(g.definitions("unknown").empty());
}
