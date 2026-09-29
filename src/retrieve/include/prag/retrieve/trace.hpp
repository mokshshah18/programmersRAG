// Probabilistic trace: adaptive retrieval by thresholded relevance probability.
//
// Every unit v starts from its score posterior p_s(v) = P(relevant | z_v). Units
// with p_s(v) >= tau are accepted as seeds. Accepting a unit u raises the
// probability of its neighbours: an edge of kind e carries a likelihood ratio
// lambda_e >= 1 (how much more likely a unit is to be relevant given that its
// neighbour is), so conditional on u being relevant
//
//     p(v | u) = sigmoid( logit p_s(v) + ln lambda_e ).
//
// u is itself relevant only with probability P(u), hence the evidence u gives v is
//
//     q_u(v) = P(u) p(v | u) + (1 - P(u)) p_s(v)   >=  p_s(v),
//
// and evidence from several accepted neighbours combines as a noisy-OR of the
// normalised gains d_u(v) = (q_u(v) - p_s(v)) / (1 - p_s(v)):
//
//     P(v) = 1 - (1 - p_s(v)) * prod_u (1 - d_u(v)).
//
// Candidates are expanded best-first; a unit is accepted the moment P(v) >= tau
// and its probability is frozen at acceptance, which rules out feedback between
// mutually adjacent units. Tracing stops when no candidate reaches tau. With
// tau = c_fp / (c_fp + c_fn) the accept rule minimises expected cost when a
// missed relevant unit costs c_fn and an irrelevant unit in the context costs
// c_fp; tau = 0.25 corresponds to c_fn = 3 c_fp.
//
// The procedure, step by step:
//
//   1. All units with p_s(v) >= tau are accepted, in order of decreasing p_s
//      (ties: lower id first), with P(v) = p_s(v).
//   2. Accepted units wait for expansion in a queue ordered by their frozen
//      probability (ties: lower id first). Expanding u visits its neighbours
//      that are not accepted yet: the callees of u, the callers of u and the
//      first `sibling_fanout` siblings of u. A neighbour reached through more
//      than one kind of edge counts once, through the edge with the largest
//      damped lift. Each visited neighbour v receives the factor 1 - d_u(v).
//   3. The neighbours whose probability reaches tau during the expansion of u
//      are accepted at once, in order of decreasing probability (ties: lower id
//      first), and join the queue.
//
// Hub damping: ln lambda_e of an edge into v is multiplied by
// 1 / (1 + log2(max(1, deg))), where deg is the number of callers of v for a
// callee edge and the number of callees of v for a caller edge. Sibling edges
// are not damped.
//
// Consequences of the model:
//
//   * A single accepted neighbour can lift v to tau only if
//     p_s(v) >= tau / (tau + lambda (1 - tau)) (single_edge_floor). Several
//     neighbours together can lift a unit from below that value.
//   * Along a chain of units of equal score posterior p, joined by edges of
//     lift lambda, P_{k+1} = p + P_k (c - p) with c = p(v | u): the
//     probabilities approach P* = p / (1 - c + p) geometrically. The chain ends
//     by itself only when P* < tau; otherwise `max_hops` ends it. For
//     tau = 0.25 and lambda = 4 the border is p = 0.178.
//   * Every unit is accepted at most once, so the trace ends after at most
//     one expansion per unit, whatever the graph.
//   * Lowering tau accepts units earlier and therefore with a smaller frozen
//     probability, which weakens the evidence they pass on. The accepted set
//     is for that reason not guaranteed to grow when tau falls. With the
//     default lifts no case of a unit lost was found in 30000 random graphs;
//     with lifts of 16, 8 and 4 it happened in about 1 % of them.
//
// All probabilities are combined as logits and log-survival terms
// (logodds.hpp); logits are clamped, so scores with posterior 0 or 1 are safe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "prag/core/types.hpp"
#include "prag/graph/call_graph.hpp"
#include "prag/retrieve/calibration.hpp"

namespace prag::retrieve {

// Lifts below 1 (and undefined values) count as 1: an edge never lowers a probability.
struct EdgeLift {
  double callee = 4.0;   // v is called by an accepted unit
  double caller = 2.0;   // v calls an accepted unit
  double sibling = 1.5;  // v shares a density mode with an accepted unit
};

struct TraceOptions {
  double tau = 0.25;
  std::size_t max_units = 64;  // safety bound on the accepted set
  std::size_t max_hops = 6;    // safety bound on the distance from a seed
  std::size_t sibling_fanout = 8;  // nearest same-mode units considered per accepted unit
  EdgeLift lift;
  bool keep_best_when_empty = true;  // return the top-scoring unit, flagged, when nothing reaches tau
};

struct TracedUnit {
  UnitId id = kNoUnit;
  float score = 0.0f;    // cosine similarity to the query
  float z = 0.0f;        // standardised score
  float p_score = 0.0f;  // posterior from the score alone
  float p = 0.0f;        // posterior at acceptance, p >= p_score
  std::uint16_t hops = 0;           // length of the parent chain back to a seed
  UnitId parent = kNoUnit;          // neighbour that contributed the largest gain
  EdgeKind via = EdgeKind::Seed;    // kind of the edge from the parent
};

struct TraceResult {
  std::vector<TracedUnit> units;  // acceptance order: probability non-increasing among seeds
  std::size_t seeds = 0;
  std::size_t evaluated = 0;      // candidates whose probability was computed
  std::size_t expansions = 0;     // accepted units whose neighbours were visited
  double z_threshold = 0.0;       // score posterior reaches tau at this z
  bool low_confidence = false;    // nothing reached tau; units holds the single best unit
  bool truncated = false;         // stopped by max_units
};

// scores: one cosine per unit id. `siblings` lists, for each unit, its nearest
// units inside the same density mode; pass an empty graph to skip sibling edges.
//
// Further conventions:
//   * z = (score - mean) / stddev; when stddev is not positive and finite,
//     every z is 0. A score that is not finite counts as the lowest possible.
//   * Neighbour ids outside [0, scores.size()) are ignored.
//   * A unit at `max_hops` is accepted but not expanded.
//   * `truncated` is set exactly when the bound `max_units` kept at least one
//     unit out that would have been accepted without it.
//   * `evaluated` counts the distinct units that received evidence from an
//     expanded neighbour; `seeds` counts the accepted seeds.
//   * The low-confidence unit is the one with the largest score (ties: lower
//     id); it is not expanded and needs max_units >= 1.
//   * The result depends only on the arguments: equal inputs give equal outputs.
TraceResult trace(std::span<const float> scores, double mean, double stddev, const Calibration& calibration,
                  const graph::CallGraph& calls, const graph::NeighborGraph& siblings,
                  const TraceOptions& options = {});

// Baseline for comparison: the k best-scoring units, best first (ties: lower
// id first), each with p = p_score, hops 0 and no parent.
std::vector<TracedUnit> top_k(std::span<const float> scores, double mean, double stddev,
                              const Calibration& calibration, std::size_t k);

// 1 / (1 + log2(max(1, degree))): 1 for degree <= 1, 1/2 for 2, 1/3 for 4, 1/4 for 8.
double hub_damping(std::size_t degree) noexcept;

// tau / (tau + lambda (1 - tau)): the smallest score posterior that one fully
// confident neighbour with lift lambda can raise to tau.
double single_edge_floor(double tau, double lambda) noexcept;

}  // namespace prag::retrieve
