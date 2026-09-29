// Baselines for the experiments: retrieval rules the probabilistic trace is
// compared against, and the distribution-free companion of its threshold.
// The fixed-k baseline, top_k, is declared in trace.hpp.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "prag/graph/call_graph.hpp"
#include "prag/retrieve/calibration.hpp"
#include "prag/retrieve/trace.hpp"

namespace prag::retrieve {

// Largest-gap cut-off. With the scores sorted in decreasing order,
// s_(1) >= s_(2) >= ..., the ranking is cut after position
//
//     k = argmax_{1 <= i <= min(k_max, n - 1)}  s_(i) - s_(i+1)
//
// (ties: the smallest i), and the first k + buffer units are returned, best
// first, in the form of top_k. The search is restricted to the top k_max
// positions; the buffer is added after the cut and is not limited by k_max.
// Scores that are not a number are left out. A single unit is returned as it
// is; k_max = 0 returns nothing.
std::vector<TracedUnit> largest_gap_cutoff(std::span<const float> scores, double mean, double stddev,
                                           const Calibration& calibration, std::size_t k_max,
                                           std::size_t buffer = 0);

struct PageRankOptions {
  double continuation = 0.5;   // alpha: probability of following an edge; 1 - alpha restarts at the seeds
  double callee_weight = 1.0;  // weight of the step from a unit to a unit it calls
  double caller_weight = 1.0;  // weight of the step from a unit to a unit that calls it
  double tolerance = 1e-6;     // stop when the l1 change of one iteration is below this
  std::size_t max_iterations = 100;
};

struct PageRankStats {
  std::size_t iterations = 0;
  double change = 0.0;      // l1 change of the last iteration
  bool converged = false;   // the last change was below the tolerance
};

// Personalised PageRank on the call graph: the fixed point of
//
//     r = (1 - alpha) p + alpha W^T r,
//
// where p is the seed vector scaled to sum 1 and W is row-stochastic. Row u of
// W spreads over the callees of u with weight `callee_weight` each and over
// the callers of u with weight `caller_weight` each, divided by their total; a
// unit that is both counts with the sum of the weights. A unit without
// outgoing weight returns its mass to the seeds (its row of W is p), which
// keeps the total mass at 1. The closed form is
// r = (1 - alpha) (I - alpha W^T)^{-1} p.
//
// Power iteration from r = p. One iteration contracts the l1 distance to the
// fixed point by alpha, so the change falls below the tolerance after at most
// ceil(ln(2 / tolerance) / ln(1 / alpha)) + 1 iterations, and the result is
// then within alpha / (1 - alpha) * tolerance of the fixed point.
//
// The number of units is seeds.size(); edges to other ids are ignored. Seed
// entries that are negative or undefined count as zero, and a seed vector
// without positive entry is replaced by the uniform distribution.
// `continuation` is clamped to [0, 0.999999], negative weights to zero. The
// result sums to 1; it is a distribution over units, not a probability of
// relevance per unit.
std::vector<double> personalized_pagerank(const graph::CallGraph& calls, std::span<const double> seeds,
                                          const PageRankOptions& options = {}, PageRankStats* stats = nullptr);

// The k units of largest rank, best first (ties: lower id first), in the form
// of top_k: p and p_score are the score posterior. Units beyond scores.size()
// or rank.size() are ignored.
std::vector<TracedUnit> top_k_by_rank(std::span<const double> rank, std::span<const float> scores, double mean,
                                      double stddev, const Calibration& calibration, std::size_t k);

// Split-conformal score threshold. Returns the largest t such that at least
// k = ceil((n + 1)(1 - alpha)) of the n relevant calibration scores are >= t,
// which is the k-th largest score. If the calibration scores and the score of
// a new relevant unit are exchangeable, the new score is >= t with probability
// at least 1 - alpha.
//
// -infinity when k > n (n too small for the requested alpha, including
// alpha <= 0 and n = 0): only keeping everything gives the guarantee.
// +infinity when alpha >= 1. Scores that are not finite are ignored. A product
// (n + 1)(1 - alpha) within 1e-9 of an integer counts as that integer, so that
// decimal alpha such as 0.1 give the intended k.
double conformal_threshold(std::span<const double> z_relevant, double alpha);

}  // namespace prag::retrieve
