#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prag/core/clock.hpp"
#include "prag/graph/call_graph.hpp"
#include "prag/retrieve/calibration.hpp"
#include "prag/retrieve/trace.hpp"
#include "sampling.hpp"

using namespace prag;
using namespace prag::retrieve;
using graph::CallGraph;
using graph::NeighborGraph;

namespace {

using Edges = std::vector<std::pair<UnitId, UnitId>>;

// With this calibration, mean 0 and standard deviation 1, the posterior of a
// unit is the logistic function of its score.
const Calibration& logistic() {
  static const Calibration calibration = Calibration::fallback(0.0, 1.0);
  return calibration;
}

// The score whose posterior is (to the precision of a float) p.
float score_for(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }

std::vector<float> scores_for(const std::vector<double>& probabilities) {
  std::vector<float> scores;
  for (const double p : probabilities) scores.push_back(score_for(p));
  return scores;
}

double posterior_of(float score) { return logistic().posterior(static_cast<double>(score)); }

TraceResult run(const std::vector<float>& scores, const CallGraph& calls, const NeighborGraph& siblings,
                const TraceOptions& options = {}) {
  return trace(scores, 0.0, 1.0, logistic(), calls, siblings, options);
}

CallGraph call_graph(std::size_t n, const Edges& edges) { return CallGraph::from_edges(n, edges); }

// The formulas of the model in plain probability arithmetic, independent of
// the implementation (which works on logits).
double lifted(double p, double lambda) {
  const double odds = p / (1.0 - p) * lambda;
  return odds / (1.0 + odds);
}

double evidence(double p_score, double p_parent, double lambda) {
  return p_parent * lifted(p_score, lambda) + (1.0 - p_parent) * p_score;
}

double noisy_or(double p_score, const std::vector<double>& evidences) {
  double survive = 1.0 - p_score;
  for (const double q : evidences) survive *= 1.0 - (q - p_score) / (1.0 - p_score);
  return 1.0 - survive;
}

double damping(std::size_t degree) {
  return 1.0 / (1.0 + std::log2(static_cast<double>(std::max<std::size_t>(1, degree))));
}

// Measurements run in full when the environment variable PRAG_SLOW_TESTS is 1.
bool slow_tests() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  return value != nullptr && std::string(value) == "1";
}

const TracedUnit* find(const TraceResult& result, UnitId id) {
  for (const TracedUnit& unit : result.units) {
    if (unit.id == id) return &unit;
  }
  return nullptr;
}

std::vector<UnitId> ids(const TraceResult& result) {
  std::vector<UnitId> out;
  for (const TracedUnit& unit : result.units) out.push_back(unit.id);
  return out;
}

bool same(const TraceResult& a, const TraceResult& b) {
  if (a.units.size() != b.units.size() || a.seeds != b.seeds || a.evaluated != b.evaluated ||
      a.expansions != b.expansions || a.low_confidence != b.low_confidence || a.truncated != b.truncated ||
      std::memcmp(&a.z_threshold, &b.z_threshold, sizeof(double)) != 0) {
    return false;
  }
  for (std::size_t i = 0; i < a.units.size(); ++i) {
    const TracedUnit& x = a.units[i];
    const TracedUnit& y = b.units[i];
    if (x.id != y.id || x.hops != y.hops || x.parent != y.parent || x.via != y.via ||
        std::memcmp(&x.score, &y.score, sizeof(float)) != 0 || std::memcmp(&x.z, &y.z, sizeof(float)) != 0 ||
        std::memcmp(&x.p_score, &y.p_score, sizeof(float)) != 0 || std::memcmp(&x.p, &y.p, sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ---- (a) seeds ---------------------------------------------------------------

PRAG_TEST(seeds_only) {
  const std::vector<float> scores = scores_for({0.9, 0.3, 0.1, 0.26, 0.6, 0.24});
  const TraceResult result = run(scores, call_graph(6, {}), NeighborGraph());
  CHECK(ids(result) == (std::vector<UnitId>{0, 4, 1, 3}));
  CHECK_EQ(result.seeds, std::size_t{4});
  CHECK_EQ(result.evaluated, std::size_t{0});
  CHECK_EQ(result.expansions, std::size_t{4});
  CHECK(!result.low_confidence);
  CHECK(!result.truncated);
  CHECK_NEAR(result.z_threshold, std::log(0.25 / 0.75), 1e-12);
  for (const TracedUnit& unit : result.units) {
    CHECK_EQ(unit.hops, std::uint16_t{0});
    CHECK_EQ(unit.parent, kNoUnit);
    CHECK(unit.via == EdgeKind::Seed);
    CHECK_EQ(unit.p, unit.p_score);
    CHECK_EQ(unit.score, scores[unit.id]);
    CHECK_EQ(unit.z, scores[unit.id]);
    CHECK_NEAR(unit.p_score, posterior_of(scores[unit.id]), 1e-7);
    CHECK(unit.p >= 0.25f);
  }
  // Standardisation: the same scores on another scale give the same result.
  std::vector<float> shifted;
  for (const float s : scores) shifted.push_back(10.0f + 4.0f * s);
  const TraceResult rescaled = trace(shifted, 10.0, 4.0, logistic(), call_graph(6, {}), NeighborGraph());
  CHECK(ids(rescaled) == ids(result));
  CHECK_NEAR(rescaled.units[0].z, scores[0], 1e-6);
}

PRAG_TEST(empty_and_degenerate_input) {
  const TraceResult none = run({}, call_graph(0, {}), NeighborGraph());
  CHECK(none.units.empty());
  CHECK(!none.low_confidence);
  CHECK_NEAR(none.z_threshold, std::log(0.25 / 0.75), 1e-12);

  // Without spread every unit has z = 0 and posterior 1/2.
  const std::vector<float> flat = {0.3f, 0.3f, 0.3f};
  const TraceResult all = trace(flat, 0.3, 0.0, logistic(), call_graph(3, {}), NeighborGraph());
  CHECK(ids(all) == (std::vector<UnitId>{0, 1, 2}));
  CHECK_NEAR(all.units[0].p, 0.5, 1e-7);
  CHECK_EQ(all.units[0].z, 0.0f);

  // An undefined score is the lowest possible one; graphs smaller or larger
  // than the score vector are tolerated.
  const std::vector<float> broken = {score_for(0.9), std::numeric_limits<float>::quiet_NaN(), score_for(0.2)};
  const TraceResult result = run(broken, call_graph(5, {{0, 1}, {0, 2}, {0, 4}, {4, 0}}), NeighborGraph());
  CHECK(ids(result) == (std::vector<UnitId>{0, 2}));
  const TraceResult small_graph = run(broken, call_graph(1, {}), NeighborGraph());
  CHECK(ids(small_graph) == (std::vector<UnitId>{0}));
}

// ---- (b) acceptance through a confident parent ---------------------------------

PRAG_TEST(callee_is_accepted_through_a_confident_parent) {
  TraceOptions options;
  const double tau = options.tau;
  const double floor = single_edge_floor(tau, options.lift.callee);
  CHECK_NEAR(floor, 0.25 / (0.25 + 4.0 * 0.75), 1e-15);  // 1/13
  const double p_callee = 0.15;
  REQUIRE(p_callee < tau && p_callee > floor);

  const std::vector<float> scores = scores_for({0.99, p_callee});
  const TraceResult result = run(scores, call_graph(2, {{0, 1}}), NeighborGraph(), options);
  REQUIRE(result.units.size() == 2u);
  CHECK_EQ(result.seeds, std::size_t{1});
  CHECK_EQ(result.evaluated, std::size_t{1});
  CHECK_EQ(result.expansions, std::size_t{2});

  const TracedUnit& parent = result.units[0];
  const TracedUnit& callee = result.units[1];
  CHECK_EQ(parent.id, UnitId{0});
  CHECK_EQ(callee.id, UnitId{1});
  CHECK(callee.via == EdgeKind::Callee);
  CHECK_EQ(callee.parent, UnitId{0});
  CHECK_EQ(callee.hops, std::uint16_t{1});

  const double p_parent = posterior_of(scores[0]);
  const double p_score = posterior_of(scores[1]);
  const double expected = evidence(p_score, p_parent, 4.0);
  CHECK_NEAR(callee.p_score, p_score, 1e-7);
  CHECK_NEAR(callee.p, expected, 1e-7);
  CHECK_NEAR(callee.p, noisy_or(p_score, {expected}), 1e-7);  // one neighbour: P = q
  CHECK(callee.p >= static_cast<float>(tau));
  CHECK(callee.p > callee.p_score);
  // By hand: odds 0.15 / 0.85 * 4 give p(v|u) = 0.4138, and 0.99 of the way there is 0.4112.
  CHECK_NEAR(callee.p, 0.4112, 5e-4);
}

PRAG_TEST(caller_and_sibling_edges_use_their_own_lift) {
  TraceOptions options;
  // Unit 1 calls the seed (caller edge, lambda = 2); unit 2 is a sibling of the
  // seed (lambda = 1.5). Both lie between their floor and tau.
  const double p_caller = 0.2;
  const double p_sibling = 0.22;
  REQUIRE(p_caller > single_edge_floor(options.tau, 2.0));
  REQUIRE(p_sibling > single_edge_floor(options.tau, 1.5));
  const std::vector<float> scores = scores_for({0.97, p_caller, p_sibling});
  const NeighborGraph siblings = NeighborGraph::from_edges(3, Edges{{0, 2}});
  const TraceResult result = run(scores, call_graph(3, {{1, 0}}), siblings, options);
  REQUIRE(result.units.size() == 3u);

  const double p_seed = posterior_of(scores[0]);
  const TracedUnit* caller = find(result, 1);
  const TracedUnit* sibling = find(result, 2);
  REQUIRE(caller != nullptr && sibling != nullptr);
  CHECK(caller->via == EdgeKind::Caller);
  CHECK(sibling->via == EdgeKind::Sibling);
  CHECK_NEAR(caller->p, evidence(posterior_of(scores[1]), p_seed, 2.0), 1e-7);
  CHECK_NEAR(sibling->p, evidence(posterior_of(scores[2]), p_seed, 1.5), 1e-7);
  // Larger probability first.
  CHECK(ids(result) == (std::vector<UnitId>{0, 1, 2}));
  CHECK(caller->p > sibling->p);

  // Sibling edges are directed: the seed is not listed for unit 2, which makes
  // no difference here, and a sibling list of the wrong unit gives no edge.
  const NeighborGraph reversed = NeighborGraph::from_edges(3, Edges{{2, 0}});
  const TraceResult without = run(scores, call_graph(3, {{1, 0}}), reversed, options);
  CHECK(ids(without) == (std::vector<UnitId>{0, 1}));
}

PRAG_TEST(evidence_from_several_neighbours_is_a_noisy_or) {
  // Unit 3 is called by unit 0 and by unit 1 (two callers: damping 1/2, so
  // each callee edge lifts by 4^(1/2) = 2) and is a sibling of unit 2.
  const std::vector<float> scores = scores_for({0.9, 0.8, 0.7, 0.12});
  const NeighborGraph siblings = NeighborGraph::from_edges(4, Edges{{2, 3}});
  TraceOptions options;
  options.tau = 0.3;
  const TraceResult result = run(scores, call_graph(4, {{0, 3}, {1, 3}}), siblings, options);
  const TracedUnit* unit = find(result, 3);
  REQUIRE(unit != nullptr);

  const double p_score = posterior_of(scores[3]);
  const double q0 = evidence(p_score, posterior_of(scores[0]), 2.0);
  const double q1 = evidence(p_score, posterior_of(scores[1]), 2.0);
  const double q2 = evidence(p_score, posterior_of(scores[2]), 1.5);
  // After the two callers the probability is still below tau; the sibling,
  // expanded last, carries it over.
  REQUIRE(noisy_or(p_score, {q0, q1}) < options.tau);
  CHECK_NEAR(unit->p, noisy_or(p_score, {q0, q1, q2}), 1e-7);
  CHECK(unit->p >= 0.3f);
  // The parent is the neighbour with the largest gain: the most confident caller.
  CHECK_EQ(unit->parent, UnitId{0});
  CHECK(unit->via == EdgeKind::Callee);
  CHECK_EQ(unit->hops, std::uint16_t{1});
}

PRAG_TEST(a_neighbour_counts_once_through_its_strongest_edge) {
  // Mutual recursion: 0 calls 1 and 1 calls 0. Unit 1 is a callee (lambda 4)
  // and a caller (lambda 2) of the seed, and also its sibling.
  const std::vector<float> scores = scores_for({0.95, 0.1});
  const NeighborGraph siblings = NeighborGraph::from_edges(2, Edges{{0, 1}});
  const TraceResult result = run(scores, call_graph(2, {{0, 1}, {1, 0}}), siblings);
  const TracedUnit* unit = find(result, 1);
  REQUIRE(unit != nullptr);
  CHECK(unit->via == EdgeKind::Callee);
  CHECK_NEAR(unit->p, evidence(posterior_of(scores[1]), posterior_of(scores[0]), 4.0), 1e-7);
  CHECK_EQ(result.evaluated, std::size_t{1});
}

// ---- (c) the floor of a single edge ---------------------------------------------

PRAG_TEST(callee_below_the_floor_is_never_accepted) {
  TraceOptions options;
  const double floor = single_edge_floor(options.tau, options.lift.callee);
  for (const double p_parent : {0.3, 0.9, 0.999999, 1.0 - 1e-12}) {
    for (const double p_callee : {0.9 * floor, 0.99 * floor, 0.05, 0.001}) {
      const std::vector<float> scores = scores_for({p_parent, p_callee});
      const TraceResult result = run(scores, call_graph(2, {{0, 1}}), NeighborGraph(), options);
      CHECK(ids(result) == (std::vector<UnitId>{0}));
      CHECK_EQ(result.evaluated, std::size_t{1});
      CHECK(evidence(posterior_of(scores[1]), posterior_of(scores[0]), 4.0) < options.tau);
    }
  }
  // Just above the floor a parent that is certain suffices, a parent at 0.9 does not.
  const std::vector<float> certain = scores_for({1.0 - 1e-12, 1.01 * floor});
  CHECK_EQ(run(certain, call_graph(2, {{0, 1}}), NeighborGraph(), options).units.size(), std::size_t{2});
  const std::vector<float> doubtful = scores_for({0.9, 1.01 * floor});
  CHECK_EQ(run(doubtful, call_graph(2, {{0, 1}}), NeighborGraph(), options).units.size(), std::size_t{1});

  // Posterior exactly 0 or 1 (a score far outside the range): the clamped
  // logits keep everything finite.
  const std::vector<float> extreme = {1000.0f, -1000.0f, score_for(0.15)};
  const TraceResult result = run(extreme, call_graph(3, {{0, 1}, {0, 2}}), NeighborGraph(), options);
  CHECK(ids(result) == (std::vector<UnitId>{0, 2}));
  CHECK_EQ(result.units[0].p, 1.0f);
  // Two callees of unit 0, each with a single caller: no damping.
  CHECK_NEAR(result.units[1].p, lifted(posterior_of(extreme[2]), 4.0), 1e-7);
}

PRAG_TEST(several_neighbours_lift_a_unit_from_below_the_floor) {
  // Unit 2 is called by unit 0 and calls unit 1: one callee edge (lambda 4)
  // and one caller edge (lambda 2), neither damped. Its posterior 0.07 is
  // below the floor of either edge, 1/13 and 1/7.
  TraceOptions options;
  const double p_unit = 0.07;
  REQUIRE(p_unit < single_edge_floor(options.tau, 4.0));
  const std::vector<float> scores = scores_for({0.999, 0.999, p_unit});
  const TraceResult result = run(scores, call_graph(3, {{0, 2}, {2, 1}}), NeighborGraph(), options);
  const TracedUnit* unit = find(result, 2);
  REQUIRE(unit != nullptr);
  const double p_score = posterior_of(scores[2]);
  const double q_callee = evidence(p_score, posterior_of(scores[0]), 4.0);
  const double q_caller = evidence(p_score, posterior_of(scores[1]), 2.0);
  CHECK(q_callee < options.tau);
  CHECK(q_caller < options.tau);
  CHECK_NEAR(unit->p, noisy_or(p_score, {q_callee, q_caller}), 1e-7);
  CHECK(unit->p >= 0.25f);
  CHECK(unit->via == EdgeKind::Callee);
}

// ---- (d) hub damping -----------------------------------------------------------

PRAG_TEST(hub_damping_rejects_a_callee_with_many_callers) {
  CHECK_EQ(hub_damping(0), 1.0);
  CHECK_EQ(hub_damping(1), 1.0);
  CHECK_NEAR(hub_damping(2), 0.5, 1e-15);
  CHECK_NEAR(hub_damping(8), 0.25, 1e-15);
  CHECK_NEAR(hub_damping(5), damping(5), 1e-15);

  // The callee of the earlier test (posterior 0.15), now called by eight
  // units, seven of which are irrelevant.
  std::vector<double> probabilities = {0.99, 0.15};
  Edges edges = {{0, 1}};
  for (UnitId caller = 2; caller < 9; ++caller) {
    probabilities.push_back(0.01);
    edges.emplace_back(caller, 1);
  }
  const std::vector<float> scores = scores_for(probabilities);
  const TraceResult result = run(scores, call_graph(9, edges), NeighborGraph());
  CHECK(ids(result) == (std::vector<UnitId>{0}));
  CHECK_EQ(result.evaluated, std::size_t{1});
  // Eight callers: lambda = 4^(1/4) = sqrt(2).
  const double reached = evidence(posterior_of(scores[1]), posterior_of(scores[0]), std::sqrt(2.0));
  CHECK(reached < 0.25);
  CHECK_NEAR(reached, 0.199, 1e-3);

  // With a lower threshold the unit is accepted, at exactly that probability.
  TraceOptions options;
  options.tau = 0.19;
  const TraceResult lower = run(scores, call_graph(9, edges), NeighborGraph(), options);
  const TracedUnit* unit = find(lower, 1);
  REQUIRE(unit != nullptr);
  CHECK_NEAR(unit->p, reached, 1e-7);

  // Caller edges are damped by the number of callees of the caller: unit 1
  // calls the seed and three other units, lambda = 2^(1/3).
  const std::vector<float> busy = scores_for({0.99, 0.2, 0.01, 0.01, 0.01});
  const Edges calls = {{1, 0}, {1, 2}, {1, 3}, {1, 4}};
  CHECK(ids(run(busy, call_graph(5, calls), NeighborGraph())) == (std::vector<UnitId>{0}));
  options.tau = 0.23;
  const TraceResult accepted = run(busy, call_graph(5, calls), NeighborGraph(), options);
  const TracedUnit* caller = find(accepted, 1);
  REQUIRE(caller != nullptr);
  CHECK_NEAR(caller->p, evidence(posterior_of(busy[1]), posterior_of(busy[0]), std::cbrt(2.0)), 1e-7);

  // Sibling edges are not damped, however many units list the same sibling.
  const std::vector<float> crowd = scores_for({0.99, 0.22, 0.01, 0.01, 0.01});
  const NeighborGraph siblings = NeighborGraph::from_edges(5, Edges{{0, 1}, {2, 1}, {3, 1}, {4, 1}});
  const TraceResult shared = run(crowd, call_graph(5, {}), siblings);
  const TracedUnit* sibling = find(shared, 1);
  REQUIRE(sibling != nullptr);
  CHECK_NEAR(sibling->p, evidence(posterior_of(crowd[1]), posterior_of(crowd[0]), 1.5), 1e-7);
}

// ---- (e) cycles ----------------------------------------------------------------

PRAG_TEST(cycles_terminate_and_probabilities_are_frozen) {
  // 0 -> 1 -> 2 -> 0. Seen from the seed, unit 1 is a callee and unit 2 a caller.
  const std::vector<float> scores = scores_for({0.95, 0.2, 0.2});
  const TraceResult result = run(scores, call_graph(3, {{0, 1}, {1, 2}, {2, 0}}), NeighborGraph());
  CHECK(ids(result) == (std::vector<UnitId>{0, 1, 2}));
  CHECK_EQ(result.expansions, std::size_t{3});
  CHECK_EQ(result.evaluated, std::size_t{2});

  const double p_seed = posterior_of(scores[0]);
  // The seed keeps its score posterior although both neighbours are accepted
  // after it, and units 1 and 2, accepted in the same expansion, do not raise
  // each other.
  CHECK_EQ(result.units[0].p, result.units[0].p_score);
  CHECK_NEAR(result.units[1].p, evidence(posterior_of(scores[1]), p_seed, 4.0), 1e-7);
  CHECK_NEAR(result.units[2].p, evidence(posterior_of(scores[2]), p_seed, 2.0), 1e-7);
  CHECK(result.units[1].via == EdgeKind::Callee);
  CHECK(result.units[2].via == EdgeKind::Caller);

  // A larger strongly connected graph with two seeds.
  Edges ring;
  std::vector<double> probabilities;
  for (UnitId i = 0; i < 40; ++i) {
    ring.emplace_back(i, (i + 1) % 40);
    ring.emplace_back(i, (i + 7) % 40);
    probabilities.push_back(i % 20 == 0 ? 0.9 : 0.18);
  }
  TraceOptions options;
  options.max_units = 1000;
  options.max_hops = 1000;
  const std::vector<float> ring_scores = scores_for(probabilities);
  const TraceResult around = run(ring_scores, call_graph(40, ring), NeighborGraph(), options);
  CHECK(around.units.size() <= 40u);
  CHECK(around.expansions <= 40u);
  std::set<UnitId> distinct;
  for (const TracedUnit& unit : around.units) distinct.insert(unit.id);
  CHECK_EQ(distinct.size(), around.units.size());  // nobody is accepted twice
  CHECK_EQ(around.seeds, std::size_t{2});
  CHECK(around.units.size() > 2u);
}

// ---- (f) safety bounds ----------------------------------------------------------

PRAG_TEST(max_hops_and_max_units) {
  // A chain 0 -> 1 -> 2 -> 3 -> 4 that the trace follows to its end.
  const std::vector<float> scores = scores_for({0.95, 0.2, 0.2, 0.2, 0.2});
  const CallGraph chain = call_graph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});

  const TraceResult full = run(scores, chain, NeighborGraph());
  REQUIRE(full.units.size() == 5u);
  CHECK(!full.truncated);
  double p = posterior_of(scores[0]);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK_EQ(full.units[i].id, static_cast<UnitId>(i));
    CHECK_EQ(full.units[i].hops, static_cast<std::uint16_t>(i));
    if (i > 0) {
      p = evidence(posterior_of(scores[i]), p, 4.0);
      CHECK_NEAR(full.units[i].p, p, 1e-6);
      CHECK_EQ(full.units[i].parent, static_cast<UnitId>(i - 1));
    }
  }
  // Probabilities decay along the chain towards the fixed point of the map.
  CHECK(full.units[4].p < full.units[1].p);

  TraceOptions options;
  options.max_hops = 2;
  const TraceResult two_hops = run(scores, chain, NeighborGraph(), options);
  CHECK(ids(two_hops) == (std::vector<UnitId>{0, 1, 2}));
  CHECK(!two_hops.truncated);
  CHECK_EQ(two_hops.expansions, std::size_t{2});  // the unit at max_hops is not expanded

  options.max_hops = 0;
  CHECK(ids(run(scores, chain, NeighborGraph(), options)) == (std::vector<UnitId>{0}));

  options = TraceOptions{};
  options.max_units = 3;
  const TraceResult three = run(scores, chain, NeighborGraph(), options);
  CHECK(ids(three) == (std::vector<UnitId>{0, 1, 2}));
  CHECK(three.truncated);

  // The bound is reached but nothing was cut.
  options.max_units = 5;
  const TraceResult exact = run(scores, chain, NeighborGraph(), options);
  CHECK_EQ(exact.units.size(), std::size_t{5});
  CHECK(!exact.truncated);

  // More seeds than max_units: the best ones are kept.
  const std::vector<float> many = scores_for({0.5, 0.9, 0.7, 0.8});
  options.max_units = 2;
  const TraceResult best = run(many, call_graph(4, {}), NeighborGraph(), options);
  CHECK(ids(best) == (std::vector<UnitId>{1, 3}));
  CHECK(best.truncated);
  CHECK_EQ(best.seeds, std::size_t{2});

  options.max_units = 0;
  const TraceResult nothing = run(many, call_graph(4, {}), NeighborGraph(), options);
  CHECK(nothing.units.empty());
  CHECK(nothing.truncated);
}

// Along a chain of callee edges whose units all have the score posterior p,
// P_{k+1} = p + P_k (c - p) with c = p(v|u): the probabilities approach
// P* = p / (1 - c + p) geometrically, with ratio c - p. The trace follows the
// chain while they stay at or above tau. It stops by itself when P* < tau, and
// only at max_hops otherwise; for tau = 0.25 and lambda = 4 the border is
// p = (sqrt(52) - 4) / 18 = 0.1784.
PRAG_TEST(chains_approach_a_fixed_point) {
  const std::size_t length = 40;
  Edges chain;
  for (UnitId i = 0; i + 1 < length; ++i) chain.emplace_back(i, i + 1);
  const CallGraph calls = call_graph(length, chain);
  TraceOptions options;
  options.max_hops = 1000;
  options.max_units = 1000;
  const double border = (std::sqrt(52.0) - 4.0) / 18.0;

  for (const double p : {0.12, 0.17, border - 0.002, border + 0.002, 0.2, 0.24}) {
    std::vector<double> probabilities(length, p);
    probabilities[0] = 0.95;
    const std::vector<float> scores = scores_for(probabilities);
    const TraceResult result = run(scores, calls, NeighborGraph(), options);

    const double p_score = posterior_of(scores[1]);
    const double c = lifted(p_score, 4.0);
    const double fixed_point = p_score / (1.0 - c + p_score);
    // The recurrence, by hand.
    std::size_t expected = 1;
    double last = posterior_of(scores[0]);
    for (std::size_t k = 1; k < length; ++k) {
      const double next = p_score + last * (c - p_score);
      if (next < options.tau) break;
      CHECK_NEAR(next - fixed_point, (last - fixed_point) * (c - p_score), 1e-15);
      last = next;
      ++expected;
    }
    REQUIRE(result.units.size() == expected);
    CHECK_NEAR(result.units.back().p, last, 1e-6);
    CHECK_EQ(result.units.back().hops, static_cast<std::uint16_t>(expected - 1));
    if (fixed_point >= options.tau) {
      CHECK(p > border);
      CHECK_EQ(expected, length);  // to the end of the chain
      CHECK_NEAR(result.units.back().p, fixed_point, 1e-6);
    } else {
      CHECK(p < border);
      CHECK(expected < 8u);
    }
  }
}

PRAG_TEST(sibling_fanout_limits_the_siblings_visited) {
  // The sibling list of the seed, nearest first: 3, 1, 2.
  const std::vector<float> scores = scores_for({0.99, 0.22, 0.22, 0.22});
  const NeighborGraph siblings = NeighborGraph::from_csr({0, 3, 3, 3, 3}, {3, 1, 2});
  TraceOptions options;
  options.sibling_fanout = 2;
  const TraceResult two = run(scores, call_graph(4, {}), siblings, options);
  CHECK(ids(two) == (std::vector<UnitId>{0, 1, 3}));
  options.sibling_fanout = 0;
  CHECK(ids(run(scores, call_graph(4, {}), siblings, options)) == (std::vector<UnitId>{0}));
  options.sibling_fanout = 8;
  CHECK(ids(run(scores, call_graph(4, {}), siblings, options)) == (std::vector<UnitId>{0, 1, 2, 3}));
}

PRAG_TEST(lifts_below_one_give_no_evidence) {
  const std::vector<float> scores = scores_for({0.99, 0.24, 0.24});
  TraceOptions options;
  options.lift.callee = 0.5;
  options.lift.caller = std::numeric_limits<double>::quiet_NaN();
  const TraceResult result = run(scores, call_graph(3, {{0, 1}, {2, 0}}), NeighborGraph(), options);
  CHECK(ids(result) == (std::vector<UnitId>{0}));
  CHECK_EQ(result.evaluated, std::size_t{0});
}

// ---- (g) nothing reaches tau ----------------------------------------------------

PRAG_TEST(keep_best_when_empty) {
  const std::vector<float> scores = scores_for({0.05, 0.2, 0.1, 0.2});
  const CallGraph calls = call_graph(4, {{1, 0}, {1, 2}, {3, 1}});
  const TraceResult result = run(scores, calls, NeighborGraph());
  REQUIRE(result.units.size() == 1u);
  CHECK(result.low_confidence);
  CHECK(!result.truncated);
  CHECK_EQ(result.seeds, std::size_t{0});
  CHECK_EQ(result.expansions, std::size_t{0});
  CHECK_EQ(result.units[0].id, UnitId{1});  // the best score; of two equal ones the lower id
  CHECK(result.units[0].via == EdgeKind::Seed);
  CHECK_EQ(result.units[0].p, result.units[0].p_score);
  CHECK(result.units[0].p < 0.25f);

  TraceOptions options;
  options.keep_best_when_empty = false;
  const TraceResult empty = run(scores, calls, NeighborGraph(), options);
  CHECK(empty.units.empty());
  CHECK(!empty.low_confidence);

  // A threshold no probability reaches.
  options = TraceOptions{};
  options.tau = 1.5;
  const TraceResult impossible = run(scores_for({0.99, 0.5}), call_graph(2, {{0, 1}}), NeighborGraph(), options);
  CHECK(impossible.low_confidence);
  CHECK_EQ(impossible.units.size(), std::size_t{1});
  CHECK(impossible.z_threshold == std::numeric_limits<double>::infinity());
}

// ---- (h) ties and determinism ---------------------------------------------------

PRAG_TEST(ties_break_by_lower_id) {
  // Seeds 1 and 3 have the same score, and so have the callees 4 and 0 of seed 1.
  const float seed = score_for(0.8);
  const float callee = score_for(0.15);
  const std::vector<float> scores = {callee, seed, score_for(0.05), seed, callee};
  const CallGraph calls = call_graph(5, {{1, 4}, {1, 0}});
  const TraceResult result = run(scores, calls, NeighborGraph());
  CHECK(ids(result) == (std::vector<UnitId>{1, 3, 0, 4}));
  CHECK_EQ(result.units[2].p, result.units[3].p);

  // Two seeds of equal probability are expanded in the order of their ids, so
  // the common callee is credited to the lower one.
  const std::vector<float> shared = {score_for(0.1), seed, seed};
  TraceOptions options;
  options.tau = 0.15;
  const TraceResult credited = run(shared, call_graph(3, {{2, 0}, {1, 0}}), NeighborGraph(), options);
  REQUIRE(credited.units.size() == 3u);
  CHECK_EQ(credited.units[2].id, UnitId{0});
  CHECK_EQ(credited.units[2].parent, UnitId{1});

  for (int repeat = 0; repeat < 5; ++repeat) CHECK(same(run(scores, calls, NeighborGraph()), result));
}

// ---- properties on random graphs -----------------------------------------------

namespace {

struct Instance {
  std::vector<float> scores;
  CallGraph calls;
  NeighborGraph siblings;
};

Instance random_instance(std::mt19937_64& rng) {
  Instance instance;
  const std::size_t n = 30 + static_cast<std::size_t>(rng() % 90);
  for (std::size_t i = 0; i < n; ++i) instance.scores.push_back(static_cast<float>(test::normal(rng, -2.5, 1.5)));

  Edges edges;
  const std::size_t calls = (2 + static_cast<std::size_t>(rng() % 3)) * n;
  for (std::size_t e = 0; e < calls; ++e) {
    // Every eighth edge ends in one of three hubs.
    const UnitId to = e % 8 == 0 ? static_cast<UnitId>(rng() % 3) : static_cast<UnitId>(rng() % n);
    edges.emplace_back(static_cast<UnitId>(rng() % n), to);
  }
  instance.calls = CallGraph::from_edges(n, edges);

  std::vector<std::uint32_t> offsets = {0};
  std::vector<UnitId> targets;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t count = static_cast<std::size_t>(rng() % 5);
    std::set<UnitId> chosen;
    while (chosen.size() < count) {
      const UnitId other = static_cast<UnitId>(rng() % n);
      if (other != i && chosen.insert(other).second) targets.push_back(other);  // in the order drawn
    }
    offsets.push_back(static_cast<std::uint32_t>(targets.size()));
  }
  instance.siblings = NeighborGraph::from_csr(std::move(offsets), std::move(targets));
  return instance;
}

bool contains(std::span<const UnitId> list, UnitId id) { return std::find(list.begin(), list.end(), id) != list.end(); }

// P(v) from the accepted set, by the formulas of the model.
double recomputed(UnitId v, const TraceResult& result, const Instance& instance, const TraceOptions& options) {
  const double p_score = posterior_of(instance.scores[v]);
  std::vector<double> evidences;
  for (const TracedUnit& u : result.units) {
    double lambda = 1.0;
    if (contains(instance.calls.callees(u.id), v)) {
      lambda = std::max(lambda, std::pow(options.lift.callee, damping(instance.calls.callers(v).size())));
    }
    if (contains(instance.calls.callers(u.id), v)) {
      lambda = std::max(lambda, std::pow(options.lift.caller, damping(instance.calls.callees(v).size())));
    }
    const std::span<const UnitId> nearest = instance.siblings.of(u.id);
    if (contains(nearest.first(std::min(nearest.size(), options.sibling_fanout)), v)) {
      lambda = std::max(lambda, options.lift.sibling);
    }
    if (lambda > 1.0) evidences.push_back(evidence(p_score, static_cast<double>(u.p), lambda));
  }
  return noisy_or(p_score, evidences);
}

}  // namespace

PRAG_TEST(properties_on_random_graphs) {
  std::mt19937_64 rng(20260929);
  const double thresholds[] = {0.5, 0.25, 0.1};
  std::size_t instances = 0;
  std::size_t traced_units = 0;
  std::size_t rejected_with_evidence = 0;
  std::size_t subset_violations = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const Instance instance = random_instance(rng);
    const std::size_t n = instance.scores.size();
    std::vector<std::set<UnitId>> accepted;
    for (const double tau : thresholds) {
      TraceOptions options;
      options.tau = tau;
      options.max_units = n;  // the safety bounds must not interfere
      options.max_hops = n;
      options.sibling_fanout = 3;
      const TraceResult result = run(instance.scores, instance.calls, instance.siblings, options);
      CHECK(!result.truncated);
      CHECK(same(result, run(instance.scores, instance.calls, instance.siblings, options)));

      std::set<UnitId> members;
      for (std::size_t i = 0; i < result.units.size(); ++i) {
        const TracedUnit& unit = result.units[i];
        CHECK(members.insert(unit.id).second);
        CHECK(unit.p >= unit.p_score);
        CHECK_NEAR(unit.p_score, posterior_of(instance.scores[unit.id]), 1e-7);
        if (result.low_confidence) continue;
        CHECK(unit.p >= static_cast<float>(tau));
        if (i < result.seeds) {
          CHECK(unit.via == EdgeKind::Seed);
          CHECK(unit.p_score >= static_cast<float>(tau));
          if (i > 0) CHECK(result.units[i - 1].p >= unit.p);
        } else {
          // A traced unit was below tau on its score and has an accepted parent
          // that precedes it and is one hop nearer to a seed.
          ++traced_units;
          CHECK(posterior_of(instance.scores[unit.id]) < tau);
          CHECK(unit.via != EdgeKind::Seed);
          const auto parent = std::find_if(result.units.begin(), result.units.begin() + static_cast<std::ptrdiff_t>(i),
                                           [&](const TracedUnit& u) { return u.id == unit.parent; });
          REQUIRE(parent != result.units.begin() + static_cast<std::ptrdiff_t>(i));
          CHECK_EQ(unit.hops, static_cast<std::uint16_t>(parent->hops + 1));
        }
      }
      if (result.low_confidence) {
        CHECK_EQ(result.units.size(), std::size_t{1});
        members.clear();  // the flagged unit is not an accepted one
      } else {
        // Seeds are exactly the units whose score posterior reaches tau.
        std::size_t above = 0;
        for (const float score : instance.scores) above += posterior_of(score) >= tau ? 1 : 0;
        CHECK_EQ(result.seeds, above);
        for (UnitId v = 0; v < n; ++v) {
          if (members.count(v) != 0) continue;
          const double p = recomputed(v, result, instance, options);
          CHECK(p < tau + 1e-6);
          if (p > posterior_of(instance.scores[v])) ++rejected_with_evidence;
        }
      }
      accepted.push_back(std::move(members));
    }
    // tau = 0.5 accepts a subset of tau = 0.25, which accepts a subset of tau = 0.1.
    for (std::size_t t = 0; t + 1 < accepted.size(); ++t) {
      for (const UnitId id : accepted[t]) subset_violations += accepted[t + 1].count(id) == 0 ? 1 : 0;
    }
    ++instances;
  }
  CHECK(instances >= 200u);
  CHECK_EQ(subset_violations, std::size_t{0});
  // The instances exercise the graph part of the trace, not only the seeds.
  CHECK(traced_units > 1000u);
  CHECK(rejected_with_evidence > 1000u);
}

// The accepted set need not grow when tau falls (see trace.hpp). Counts, over
// random graphs and eight thresholds, the units accepted at one threshold and
// missing at a lower one, for the default lifts and for larger ones.
// PRAG_SLOW_TESTS=1 runs 30000 graphs per setting and prints the counts.
PRAG_TEST(measure_monotonicity_in_tau) {
  const bool slow = slow_tests();
  const int graphs = slow ? 30000 : 300;
  const double thresholds[] = {0.9, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05};
  const EdgeLift settings[] = {{4.0, 2.0, 1.5}, {16.0, 8.0, 4.0}, {100.0, 50.0, 20.0}};
  for (const EdgeLift& lift : settings) {
    std::mt19937_64 rng(12345);
    std::size_t lost = 0;
    std::size_t graphs_with_loss = 0;
    std::size_t accepted_units = 0;
    for (int g = 0; g < graphs; ++g) {
      const Instance instance = random_instance(rng);
      const std::size_t n = instance.scores.size();
      std::vector<std::set<UnitId>> accepted;
      for (const double tau : thresholds) {
        TraceOptions options;
        options.tau = tau;
        options.max_units = n;
        options.max_hops = n;
        options.lift = lift;
        const TraceResult result = run(instance.scores, instance.calls, instance.siblings, options);
        std::set<UnitId> members;
        if (!result.low_confidence) {
          for (const TracedUnit& unit : result.units) members.insert(unit.id);
        }
        accepted_units += members.size();
        accepted.push_back(std::move(members));
      }
      std::size_t lost_here = 0;
      for (std::size_t a = 0; a < accepted.size(); ++a) {
        for (std::size_t b = a + 1; b < accepted.size(); ++b) {
          for (const UnitId id : accepted[a]) lost_here += accepted[b].count(id) == 0 ? 1 : 0;
        }
      }
      lost += lost_here;
      graphs_with_loss += lost_here != 0 ? 1 : 0;
    }
    if (lift.callee == 4.0) CHECK_EQ(lost, std::size_t{0});
    CHECK(accepted_units > 0u);
    if (slow) {
      std::printf("  lifts %.0f / %.0f / %.1f: %d graphs, %zu units accepted over 8 thresholds, ", lift.callee,
                  lift.caller, lift.sibling, graphs, accepted_units);
      std::printf("%zu lost at a lower threshold, in %zu graphs\n", lost, graphs_with_loss);
    }
  }
}

// ---- top-k baseline -------------------------------------------------------------

PRAG_TEST(top_k_orders_by_score_and_breaks_ties_by_id) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> scores = {0.2f, 0.9f, nan, 0.5f, 0.9f, -0.3f, 0.5f};
  const std::vector<TracedUnit> top = top_k(scores, 0.0, 1.0, logistic(), 4);
  REQUIRE(top.size() == 4u);
  CHECK_EQ(top[0].id, UnitId{1});
  CHECK_EQ(top[1].id, UnitId{4});
  CHECK_EQ(top[2].id, UnitId{3});
  CHECK_EQ(top[3].id, UnitId{6});
  for (const TracedUnit& unit : top) {
    CHECK_EQ(unit.score, scores[unit.id]);
    CHECK_NEAR(unit.p_score, posterior_of(scores[unit.id]), 1e-7);
    CHECK_EQ(unit.p, unit.p_score);
    CHECK_EQ(unit.hops, std::uint16_t{0});
    CHECK_EQ(unit.parent, kNoUnit);
    CHECK(unit.via == EdgeKind::Seed);
  }
  // k larger than the corpus returns everything, undefined scores last.
  const std::vector<TracedUnit> all = top_k(scores, 0.0, 1.0, logistic(), 100);
  REQUIRE(all.size() == scores.size());
  CHECK_EQ(all[4].id, UnitId{0});
  CHECK_EQ(all[5].id, UnitId{5});
  CHECK_EQ(all[6].id, UnitId{2});
  CHECK(top_k(scores, 0.0, 1.0, logistic(), 0).empty());
  CHECK(top_k({}, 0.0, 1.0, logistic(), 3).empty());
  // Standardisation enters z and the posterior, not the order.
  const std::vector<TracedUnit> scaled = top_k(scores, 0.5, 0.1, logistic(), 1);
  CHECK_NEAR(scaled[0].z, 4.0, 1e-6);
  CHECK_NEAR(scaled[0].p_score, 1.0 / (1.0 + std::exp(-4.0)), 1e-6);
}

// ---- measurements ---------------------------------------------------------------

// 5000 units, 15000 call edges (average degree 6 counting both directions),
// sibling lists of 8, and exactly 30 units above tau on their score. Runs as a
// plain test with few repetitions; PRAG_SLOW_TESTS=1 prints the timings.
PRAG_TEST(measure_trace_on_5000_units) {
  const bool slow = slow_tests();
  const std::size_t n = 5000;
  std::mt19937_64 rng(5000);
  std::vector<float> scores(n);
  for (float& s : scores) s = static_cast<float>(std::min(test::normal(rng, -4.0, 1.2), -1.2));  // posterior < 0.232
  std::set<std::size_t> seeds;
  while (seeds.size() < 30) seeds.insert(static_cast<std::size_t>(rng() % n));
  for (const std::size_t s : seeds) scores[s] = score_for(test::uniform(rng, 0.3, 0.99));

  Edges edges;
  while (edges.size() < 15000) {
    const UnitId from = static_cast<UnitId>(rng() % n);
    const UnitId to = static_cast<UnitId>(rng() % n);
    if (from != to) edges.emplace_back(from, to);
  }
  const CallGraph calls = CallGraph::from_edges(n, edges);
  std::vector<std::uint32_t> offsets = {0};
  std::vector<UnitId> targets;
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 8; ++k) targets.push_back(static_cast<UnitId>(rng() % n));
    offsets.push_back(static_cast<std::uint32_t>(targets.size()));
  }
  const NeighborGraph siblings = NeighborGraph::from_csr(std::move(offsets), std::move(targets));

  const TraceResult result = run(scores, calls, siblings);
  CHECK_EQ(result.seeds, std::size_t{30});
  CHECK(result.units.size() >= 30u);
  CHECK(result.units.size() <= 64u);

  const int repetitions = slow ? 2000 : 20;
  std::vector<double> micros;
  for (int r = 0; r < repetitions; ++r) {
    const Stopwatch watch;
    const TraceResult again = run(scores, calls, siblings);
    micros.push_back(static_cast<double>(watch.elapsed_ns()) / 1e3);
    CHECK_EQ(again.units.size(), result.units.size());
  }
  std::sort(micros.begin(), micros.end());

  // Posterior lookups: the logistic fallback and a fitted table.
  std::vector<double> relevant;
  std::vector<double> other;
  for (int i = 0; i < 400; ++i) relevant.push_back(test::normal(rng, 5.0));
  for (int i = 0; i < 4000; ++i) other.push_back(test::normal(rng));
  FitOptions fit;
  fit.fraction_relevant = 1.0;  // fixed windows: no cross-validation in a timing test
  fit.fraction_nonrelevant = 1.0;
  const Calibration fitted = Calibration::fit(relevant, other, 0.002, fit);
  REQUIRE(fitted.fitted());
  std::vector<double> queries(1 << 16);
  for (double& q : queries) q = test::uniform(rng, -5.0, 9.0);
  const int passes = slow ? 200 : 2;
  double checksum = 0.0;
  const Stopwatch table_watch;
  for (int pass = 0; pass < passes; ++pass) {
    for (const double q : queries) checksum += fitted.posterior(q);
  }
  const double table_ns = static_cast<double>(table_watch.elapsed_ns()) / (static_cast<double>(passes) * 65536.0);
  const Stopwatch logistic_watch;
  for (int pass = 0; pass < passes; ++pass) {
    for (const double q : queries) checksum += logistic().posterior(q);
  }
  const double logistic_ns = static_cast<double>(logistic_watch.elapsed_ns()) / (static_cast<double>(passes) * 65536.0);
  CHECK(checksum > 0.0);

  if (slow) {
    std::printf("  trace, 5000 units, 15000 call edges, 30 seeds: accepted %zu, evaluated %zu, expansions %zu\n",
                result.units.size(), result.evaluated, result.expansions);
    std::printf("  trace time over %d runs: median %.1f us, minimum %.1f us, 95th percentile %.1f us\n", repetitions,
                micros[micros.size() / 2], micros.front(), micros[micros.size() * 95 / 100]);
    std::printf("  posterior lookup, fitted table of 257 points: %.1f ns (%.1f million per second)\n", table_ns,
                1e3 / table_ns);
    std::printf("  posterior lookup, logistic fallback: %.1f ns (%.1f million per second)\n", logistic_ns,
                1e3 / logistic_ns);
  }
}
