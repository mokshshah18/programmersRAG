// Recursive context quality (RCQ) and the pruning that removes context rot.
//
// The prompt context is a tree: context -> files -> units. Every node v carries
//
//     L(v)  length in tokens,
//     M(v)  effective relevant mass in tokens,
//     Q(v)  = M(v) / L(v)  in [0, 1], the quality (density) of the node,
//
// defined recursively. For a unit u with calibrated relevance probability p_u
// and length L(u):  M(u) = p_u L(u). For an inner node with children in prompt
// order u_1 .. u_k and per-node overhead h(v) tokens (headers, separators):
//
//     L(v) = h(v) + sum_j L(u_j),
//     M(v) = sum_j (1 - rho_v(u_j)) M(u_j),
//     rho_v(u_j) = max_{i < j} max(0, cos(e_i, e_j)),
//
// where rho is the redundancy of a child against the siblings placed before it.
// The context rot index of a context C is CRI(C) = 1 - Q(C): the expected
// fraction of prompt tokens that are irrelevant, redundant or overhead.
//
// Only units carry an embedding, so redundancy is evaluated at the units and
// file nodes are not discounted again: rho(u) is taken against every unit placed
// before u in the prompt, in whichever file, which makes a copy of a function
// in a second file as redundant as a copy in the same file. With L(u) the
// length of the unit in its chosen form, unit overhead included,
//
//     L(C) = sum_files ( h + sum_{u in file} L(u) ),
//     M(C) = sum_u (1 - rho(u)) p_u L(u).
//
// Pruning keeps a unit if and only if its discounted quality reaches theta,
//
//     (1 - rho(u)) p_u >= theta,
//
// with rho taken against the kept units that rank before u by probability.
// This guarantees Q >= theta for every kept unit and hence, overhead and
// support units aside, for every inner node:
//
//     Q(C) >= theta L_kept / (L_kept + L_support + L_overhead).
//
// The guarantee does not depend on the order of the prompt: at the fixed point
// every kept pair i, j satisfies (1 - c_ij) min(p_i, p_j) >= theta, so the
// discounted quality of a kept unit reaches theta against any subset of the
// other kept units, in particular against those placed before it in step 5.
//
// Because the redundancy of a unit depends on which units before it are kept,
// prune-and-rescore is iterated: every pass recomputes rho of every candidate
// against the units kept by the previous pass. The kept set is not monotone
// under this iteration; a unit dropped in one pass returns in a later one when
// the unit it duplicated has itself been dropped. The iteration still ends,
// because the status of the j-th candidate in probability order depends only
// on the j - 1 candidates before it: after k passes the first k + 1 statuses
// are final. The fixed point is therefore unique, equals the greedy selection
// in probability order, and is confirmed within |candidates| passes.
//
// When the result still exceeds the token budget, units are first reduced to
// their skeleton (signature and documentation) and then dropped in order of
// increasing discounted mass. The budget step only removes; the redundancy of
// the remaining units can only fall, so the guarantee survives it.
//
// Cost: the clipped cosines of all candidate pairs are computed once, in
// blocked form, O(n^2 dim) time and O(n^2) memory for n candidates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "prag/core/types.hpp"

namespace prag::context {

enum class Form : std::uint8_t {
  Full = 0,      // complete text of the unit
  Skeleton = 1,  // signature and documentation only
};

const char* to_string(Form form);

struct Candidate {
  UnitId id = kNoUnit;
  std::uint32_t group = 0;          // parent in the hierarchy (file index)
  double p = 0.0;                   // calibrated relevance probability
  std::uint32_t tokens_full = 0;
  std::uint32_t tokens_skeleton = 0;  // 0 when the unit has no skeleton form
  const float* embedding = nullptr;   // dim floats, L2-normalised; nullptr disables redundancy
  std::vector<UnitId> requires_units; // units this one needs to be understood (callees, types)
};

struct Options {
  double theta = 0.25;              // minimum discounted quality of a kept unit
  std::uint32_t token_budget = 3000;
  std::uint32_t group_overhead = 12;  // h(v) per file node
  std::uint32_t unit_overhead = 8;    // header tokens per unit
  bool dependency_closure = true;   // add skeletons of required units that were not selected
  double closure_min_p = 0.05;      // required units below this probability are not added
  bool edge_ordering = true;        // strongest groups first and last, weakest in the middle
  std::size_t max_passes = 16;      // the last permitted pass sweeps in rank order, which yields the fixed
                                    // point directly: the limit bounds the work, not the result
  std::size_t dim = 0;              // embedding dimension
  // Share of token_budget, in [0, 1], reserved for full text. The budget step
  // as written reduces every unit to its skeleton before it drops one, so a
  // long candidate list ends as a list of signatures. With a share s > 0 the
  // strongest units by discounted quality, as long as their lengths in full
  // (file overhead included) add up to at most s * token_budget, are neither
  // reduced nor dropped; the step then works on the others. 0 = step 4 of
  // DESIGN.md section 6.9 exactly as written.
  double full_text_share = 0.0;
};

struct Entry {
  UnitId id = kNoUnit;
  std::uint32_t group = 0;
  Form form = Form::Full;
  bool support = false;      // present only because a kept unit requires it
  double p = 0.0;
  double redundancy = 0.0;   // rho against the entries placed before it; 0 for support entries
  double quality = 0.0;      // (1 - rho) p
  std::uint32_t tokens = 0;  // in the chosen form, overhead included
};

struct NodeQuality {
  std::uint32_t group = 0;
  std::uint32_t tokens = 0;  // L
  double mass = 0.0;         // M
  double quality = 0.0;      // Q
  std::uint32_t units = 0;
};

struct Selection {
  std::vector<Entry> entries;        // prompt order
  std::vector<NodeQuality> groups;   // per file, prompt order
  std::uint32_t tokens_before = 0;   // every candidate in full
  std::uint32_t tokens_after = 0;
  double quality_before = 0.0;       // Q of the unpruned context
  double quality_after = 0.0;        // Q of the selection
  double rot_before = 0.0;           // CRI = 1 - Q
  double rot_after = 0.0;
  double compression = 1.0;          // tokens_after / tokens_before
  // Prune-and-rescore passes until the fixed point, the confirming pass
  // included. A candidate below theta fails whatever its redundancy; such
  // candidates are set aside before the first pass and cause no pass of their own.
  std::size_t passes = 0;
  // Every distinct candidate ends in exactly one state, so
  //   entries.size() + dropped_low_quality + dropped_redundant + dropped_budget
  // is the number of distinct candidates. skeletonised and support_added
  // count entries of the selection.
  std::size_t dropped_low_quality = 0;  // p < theta
  std::size_t dropped_redundant = 0;    // p >= theta, (1 - rho) p < theta
  std::size_t dropped_budget = 0;       // selected or added as support, then removed by the budget
  std::size_t skeletonised = 0;         // selected units shown as skeleton
  std::size_t support_added = 0;        // support entries
};

// Candidates may arrive in any order; the result is deterministic.
//
// * The most probable candidate is always selected, also when its probability
//   is below theta or its shortest form exceeds the budget, so a non-empty
//   candidate list never gives an empty context. Support entries aside, it
//   is the only entry that can have a quality below theta, and it is the
//   only reason for tokens_after to exceed token_budget.
// * Support units are taken from the candidates: a required unit that is not
//   a candidate is ignored, because neither its probability nor its length is
//   known, and so is one without a skeleton form (tokens_skeleton = 0). The
//   closure covers the direct requirements of the selected units. A support
//   unit is outside the redundancy computation in both roles.
// * A unit is reduced to its skeleton only when that makes it shorter.
// * In the budget step (1 - rho) is the discount of step 2, taken against the
//   selected units that rank before the unit by probability, and brought up to
//   date whenever a unit is dropped. Of two units with equal values the one
//   that ranks lower (smaller probability, then larger id) goes first.
// * quality_before describes every candidate in Full form, arranged as in
//   step 5 of DESIGN.md section 6.9.
// * A unit id that occurs more than once counts once, with its largest probability.
// * An empty candidate list gives an empty selection with quality 0 and rot 0.
Selection select(std::span<const Candidate> candidates, const Options& options);

// Quality of an arbitrary ordered context, e.g. a fixed top-k baseline, computed
// with the same recursion. All entries are taken in Full form. The list is the
// prompt order; a file node is a maximal run of consecutive entries of one
// group, so a file that is interrupted by another pays its overhead again.
// The fields `before` and `after` of the result are equal.
Selection evaluate(std::span<const Candidate> ordered, const Options& options);

}  // namespace prag::context
