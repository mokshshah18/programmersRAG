#include "prag/retrieve/trace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "prag/retrieve/logodds.hpp"
#include "ranking.hpp"

namespace prag::retrieve {

namespace {

using detail::ranks_before;
using detail::scored_unit;
using detail::Standardiser;

constexpr std::uint32_t kNoSlot = 0xFFFFFFFFu;

double log_lift(double lambda) noexcept { return lambda > 1.0 && std::isfinite(lambda) ? std::log(lambda) : 0.0; }

// A unit that has received evidence but is not accepted.
struct Candidate {
  double logit_score = 0.0;   // clamped logit of p_s
  double log_survival = 0.0;  // ln(1 - P): ln(1 - p_s) plus the evidence received so far
  double best_gain = 0.0;     // largest d_u over the neighbours seen so far
  UnitId parent = kNoUnit;
  EdgeKind via = EdgeKind::Seed;
  std::uint16_t hops = 0;
};

struct Edge {
  UnitId to = kNoUnit;
  double log_lift = 0.0;  // damped
  EdgeKind kind = EdgeKind::Seed;
};

// An accepted unit waiting for expansion.
struct Pending {
  double p = 0.0;
  UnitId id = kNoUnit;
  std::uint32_t slot = 0;  // index in TraceResult::units
};

// For std::priority_queue: the top is the largest p, and among equals the lowest id.
struct PendingOrder {
  bool operator()(const Pending& a, const Pending& b) const noexcept {
    if (a.p != b.p) return a.p < b.p;
    return a.id > b.id;
  }
};

// A unit whose probability has reached tau.
struct Crossing {
  double p = 0.0;
  UnitId id = kNoUnit;
};

bool best_first(const Crossing& a, const Crossing& b) noexcept {
  if (a.p != b.p) return a.p > b.p;
  return a.id < b.id;
}

// ln(1 - d_u(v)) for a unit with score logit `logit_score`, reached through an
// edge of damped log-lift `lift` from a neighbour accepted with logit
// `parent_logit`. From the definitions,
//
//     1 - d = (1 - P(u)) + P(u) r,   r = (1 - p(v|u)) / (1 - p_s(v)),
//
// and ln r = softplus(l) - softplus(l + lift) for l = logit p_s(v).
double log_one_minus_gain(double logit_score, double lift, double parent_logit) noexcept {
  const double log_r = softplus(logit_score) - softplus(logit_score + lift);
  const double stay = -softplus(parent_logit);           // ln(1 - P(u))
  const double move = -softplus(-parent_logit) + log_r;  // ln(P(u) r)
  const double high = std::max(stay, move);
  const double low = std::min(stay, move);
  return std::min(0.0, high + std::log1p(std::exp(low - high)));
}

std::uint16_t next_hop(std::uint16_t hops) noexcept {
  return hops == std::numeric_limits<std::uint16_t>::max() ? hops : static_cast<std::uint16_t>(hops + 1);
}

}  // namespace

double hub_damping(std::size_t degree) noexcept {
  if (degree <= 1) return 1.0;
  return 1.0 / (1.0 + std::log2(static_cast<double>(degree)));
}

double single_edge_floor(double tau, double lambda) noexcept {
  if (!(tau > 0.0)) return 0.0;
  if (!(tau < 1.0)) return 1.0;
  const double lift = lambda > 1.0 && std::isfinite(lambda) ? lambda : 1.0;
  return tau / (tau + lift * (1.0 - tau));
}

TraceResult trace(std::span<const float> scores, double mean, double stddev, const Calibration& calibration,
                  const graph::CallGraph& calls, const graph::NeighborGraph& siblings,
                  const TraceOptions& options) {
  TraceResult result;
  const double tau = options.tau;
  result.z_threshold = calibration.threshold(tau);

  const std::size_t n = scores.size();
  if (n == 0) return result;
  const Standardiser standardise(mean, stddev);

  std::vector<double> score_posterior(n);
  for (std::size_t v = 0; v < n; ++v) score_posterior[v] = calibration.posterior(standardise(scores[v]));

  std::vector<double> accepted_logit;  // frozen logit P(u), parallel to result.units
  std::vector<std::uint8_t> accepted(n, 0);
  std::priority_queue<Pending, std::vector<Pending>, PendingOrder> queue;

  const auto accept = [&](UnitId id, double p, const Candidate* origin) {
    TracedUnit unit = scored_unit(id, scores[id], standardise, calibration);
    unit.p = std::max(static_cast<float>(p), unit.p_score);
    if (origin != nullptr) {
      unit.hops = origin->hops;
      unit.parent = origin->parent;
      unit.via = origin->via;
    }
    accepted[id] = 1;
    accepted_logit.push_back(logit(p));
    queue.push(Pending{p, id, static_cast<std::uint32_t>(result.units.size())});
    result.units.push_back(unit);
  };

  // ---- seeds ------------------------------------------------------------------
  std::vector<Crossing> batch;
  for (std::size_t v = 0; v < n; ++v) {
    if (score_posterior[v] >= tau) batch.push_back(Crossing{score_posterior[v], static_cast<UnitId>(v)});
  }
  std::sort(batch.begin(), batch.end(), best_first);
  for (const Crossing& seed : batch) {
    if (result.units.size() >= options.max_units) {
      result.truncated = true;
      break;
    }
    accept(seed.id, seed.p, nullptr);
  }
  result.seeds = result.units.size();

  if (batch.empty()) {
    if (options.keep_best_when_empty && options.max_units >= 1) {
      UnitId best = 0;
      for (std::size_t v = 1; v < n; ++v) {
        const UnitId id = static_cast<UnitId>(v);
        if (ranks_before(scores[id], id, scores[best], best)) best = id;
      }
      accept(best, score_posterior[best], nullptr);
      result.low_confidence = true;
    }
    return result;
  }

  // ---- expansion --------------------------------------------------------------
  const double logit_tau = logit(tau);
  const double lift_callee = log_lift(options.lift.callee);
  const double lift_caller = log_lift(options.lift.caller);
  const double lift_sibling = log_lift(options.lift.sibling);

  std::vector<std::uint32_t> candidate_slot(n, kNoSlot);
  std::vector<Candidate> candidates;
  std::vector<Edge> edges;

  // Once max_units is reached the loop goes on without accepting, until either
  // a further unit reaches tau (the bound has cut the trace) or nothing is left.
  while (!queue.empty() && !result.truncated) {
    const Pending u = queue.top();
    queue.pop();
    const std::uint16_t hops = result.units[u.slot].hops;
    if (hops >= options.max_hops) continue;
    ++result.expansions;
    const double parent_logit = accepted_logit[u.slot];

    edges.clear();
    const auto add_edge = [&](UnitId to, double lift, EdgeKind kind) {
      if (to >= n || to == u.id || accepted[to] != 0 || !(lift > 0.0)) return;
      edges.push_back(Edge{to, lift, kind});
    };
    for (const UnitId v : calls.callees(u.id)) {
      add_edge(v, lift_callee * hub_damping(calls.callers(v).size()), EdgeKind::Callee);
    }
    for (const UnitId v : calls.callers(u.id)) {
      add_edge(v, lift_caller * hub_damping(calls.callees(v).size()), EdgeKind::Caller);
    }
    const std::span<const UnitId> nearest = siblings.of(u.id);
    for (const UnitId v : nearest.first(std::min(nearest.size(), options.sibling_fanout))) {
      add_edge(v, lift_sibling, EdgeKind::Sibling);
    }
    // One edge per neighbour: the strongest, and among equals the first kind.
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
      if (a.to != b.to) return a.to < b.to;
      if (a.log_lift != b.log_lift) return a.log_lift > b.log_lift;
      return static_cast<int>(a.kind) < static_cast<int>(b.kind);
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.to == b.to; }),
                edges.end());

    batch.clear();
    for (const Edge& edge : edges) {
      if (candidate_slot[edge.to] == kNoSlot) {
        candidate_slot[edge.to] = static_cast<std::uint32_t>(candidates.size());
        Candidate fresh;
        fresh.logit_score = logit(score_posterior[edge.to]);
        fresh.log_survival = -softplus(fresh.logit_score);
        candidates.push_back(fresh);
        ++result.evaluated;
      }
      Candidate& c = candidates[candidate_slot[edge.to]];
      const double factor = log_one_minus_gain(c.logit_score, edge.log_lift, parent_logit);
      const double gain = -std::expm1(factor);
      c.log_survival += factor;
      if (gain > c.best_gain) {
        c.best_gain = gain;
        c.parent = u.id;
        c.via = edge.kind;
        c.hops = next_hop(hops);
      }
      const double logit_p = logit_from_log_survival(c.log_survival);
      if (logit_p >= logit_tau) batch.push_back(Crossing{sigmoid(logit_p), edge.to});
    }

    std::sort(batch.begin(), batch.end(), best_first);
    for (const Crossing& crossing : batch) {
      if (result.units.size() >= options.max_units) {
        result.truncated = true;
        break;
      }
      accept(crossing.id, crossing.p, &candidates[candidate_slot[crossing.id]]);
    }
  }
  return result;
}

std::vector<TracedUnit> top_k(std::span<const float> scores, double mean, double stddev,
                              const Calibration& calibration, std::size_t k) {
  const std::size_t n = scores.size();
  const std::size_t keep = std::min(k, n);
  std::vector<TracedUnit> out;
  if (keep == 0) return out;

  std::vector<UnitId> order(n);
  for (std::size_t v = 0; v < n; ++v) order[v] = static_cast<UnitId>(v);
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(),
                    [&](UnitId a, UnitId b) { return ranks_before(scores[a], a, scores[b], b); });

  const Standardiser standardise(mean, stddev);
  out.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    out.push_back(scored_unit(order[i], scores[order[i]], standardise, calibration));
  }
  return out;
}

}  // namespace prag::retrieve
