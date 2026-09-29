#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prag/context/rot.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/types.hpp"

using namespace prag;
using namespace prag::context;

namespace {

// ---- helpers ---------------------------------------------------------------

// splitmix64: the same stream on every platform, unlike the <random> distributions.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() { return mix64(state_ += 0x9e3779b97f4a7c15ULL); }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
  double normal() {
    const double u = 1.0 - uniform();
    const double v = uniform();
    return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
  }

 private:
  std::uint64_t state_;
};

constexpr std::size_t kSignDim = 16;
using SignVector = std::array<float, kSignDim>;

// Components +-0.25 in 16 dimensions: the norm is exactly 1 and the cosine of
// two such vectors is (16 - 2 d) / 16, d = number of differing signs, which is
// exact in binary floating point.
SignVector signs(std::initializer_list<int> negative) {
  SignVector v;
  v.fill(0.25f);
  for (const int i : negative) v[static_cast<std::size_t>(i)] = -0.25f;
  return v;
}

Candidate unit(UnitId id, std::uint32_t group, double p, std::uint32_t full, std::uint32_t skeleton = 0,
               const float* embedding = nullptr, std::vector<UnitId> requires_units = {}) {
  Candidate c;
  c.id = id;
  c.group = group;
  c.p = p;
  c.tokens_full = full;
  c.tokens_skeleton = skeleton;
  c.embedding = embedding;
  c.requires_units = std::move(requires_units);
  return c;
}

Options wide_open() {
  Options o;
  o.theta = 0.25;
  o.token_budget = 1000000;
  o.dependency_closure = false;
  return o;
}

// Clipped cosine computed independently of the library: plain sequential sums.
double reference_similarity(const float* a, const float* b, std::size_t dim) {
  if (!a || !b || dim == 0) return 0.0;
  double ab = 0.0;
  double aa = 0.0;
  double bb = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    ab += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    aa += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    bb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (!(aa > 0.0) || !(bb > 0.0)) return 0.0;
  return std::min(1.0, std::max(0.0, ab / std::sqrt(aa * bb)));
}

const Candidate* find(const std::vector<Candidate>& candidates, UnitId id) {
  for (const Candidate& c : candidates) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

const Entry* find(const Selection& s, UnitId id) {
  for (const Entry& e : s.entries) {
    if (e.id == id) return &e;
  }
  return nullptr;
}

// The entry of a unit that has to be in the selection.
const Entry& entry_of(const Selection& s, UnitId id) {
  const Entry* e = find(s, id);
  REQUIRE(e != nullptr);
  return *e;
}

std::vector<UnitId> ids_of(const Selection& s) {
  std::vector<UnitId> ids;
  for (const Entry& e : s.entries) ids.push_back(e.id);
  return ids;
}

std::vector<UnitId> selected_ids(const Selection& s) {  // without support entries, ascending
  std::vector<UnitId> ids;
  for (const Entry& e : s.entries) {
    if (!e.support) ids.push_back(e.id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

bool ranks_before(const Candidate& a, const Candidate& b) {
  if (a.p != b.p) return a.p > b.p;
  return a.id < b.id;
}

// The fixed point of step 2 computed the direct way: one greedy sweep in
// probability order, with the reference cosine.
std::vector<UnitId> greedy_reference(std::vector<Candidate> candidates, double theta, std::size_t dim) {
  std::sort(candidates.begin(), candidates.end(), ranks_before);
  std::vector<std::size_t> kept;
  for (std::size_t j = 0; j < candidates.size(); ++j) {
    double rho = 0.0;
    for (const std::size_t i : kept) {
      rho = std::max(rho, reference_similarity(candidates[i].embedding, candidates[j].embedding, dim));
    }
    if (j == 0 || (1.0 - rho) * candidates[j].p >= theta) kept.push_back(j);
  }
  std::vector<UnitId> ids;
  for (const std::size_t j : kept) ids.push_back(candidates[j].id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Recomputes every number of a selection from its entries and the embeddings
// and compares.
void check_arithmetic(const Selection& s, const std::vector<Candidate>& candidates, const Options& o) {
  std::uint64_t total_tokens = 0;
  double total_mass = 0.0;
  std::size_t g = 0;
  std::uint64_t group_tokens = 0;
  double group_mass = 0.0;
  std::uint32_t group_units = 0;
  const auto close_group = [&] {
    REQUIRE(g < s.groups.size());
    CHECK_EQ(s.groups[g].tokens, static_cast<std::uint32_t>(group_tokens + o.group_overhead));
    CHECK_EQ(s.groups[g].units, group_units);
    CHECK_NEAR(s.groups[g].mass, group_mass, 1e-9);
    CHECK_NEAR(s.groups[g].quality, group_mass / static_cast<double>(group_tokens + o.group_overhead), 1e-12);
    total_tokens += group_tokens + o.group_overhead;
    total_mass += group_mass;
    ++g;
    group_tokens = 0;
    group_mass = 0.0;
    group_units = 0;
  };
  for (std::size_t k = 0; k < s.entries.size(); ++k) {
    const Entry& e = s.entries[k];
    const Candidate* c = find(candidates, e.id);
    REQUIRE(c != nullptr);
    if (k != 0 && s.entries[k - 1].group != e.group) close_group();
    REQUIRE(g < s.groups.size());
    CHECK_EQ(s.groups[g].group, e.group);
    CHECK_EQ(e.group, c->group);

    double rho = 0.0;
    if (!e.support) {
      for (std::size_t i = 0; i < k; ++i) {
        if (s.entries[i].support) continue;
        rho = std::max(rho, reference_similarity(find(candidates, s.entries[i].id)->embedding, c->embedding, o.dim));
      }
    }
    CHECK_NEAR(e.redundancy, rho, 1e-12);
    CHECK_NEAR(e.quality, (1.0 - rho) * e.p, 1e-12);
    // A value, not the reference std::min returns: CHECK_EQ binds its arguments by reference.
    const double clamped = std::min(1.0, std::max(0.0, c->p));
    CHECK_EQ(e.p, clamped);
    const std::uint32_t body = e.form == Form::Skeleton ? c->tokens_skeleton : c->tokens_full;
    CHECK_EQ(e.tokens, body + o.unit_overhead);
    if (e.support) CHECK(e.form == Form::Skeleton);

    group_tokens += e.tokens;
    group_mass += e.quality * static_cast<double>(e.tokens);
    ++group_units;
  }
  if (!s.entries.empty()) close_group();
  CHECK_EQ(g, s.groups.size());
  CHECK_EQ(static_cast<std::uint64_t>(s.tokens_after), total_tokens);
  if (total_tokens != 0) CHECK_NEAR(s.quality_after, total_mass / static_cast<double>(total_tokens), 1e-12);
  CHECK_NEAR(s.rot_after, 1.0 - s.quality_after, 1e-15);
  CHECK_NEAR(s.rot_before, 1.0 - s.quality_before, 1e-15);
  CHECK(s.quality_after >= 0.0 && s.quality_after <= 1.0);
  CHECK(s.quality_before >= 0.0 && s.quality_before <= 1.0);
  if (s.tokens_before != 0) {
    CHECK_NEAR(s.compression, static_cast<double>(s.tokens_after) / static_cast<double>(s.tokens_before), 1e-15);
  }
}

// The unpruned context computed independently: every candidate in full, files
// by their strongest unit (first, last, second, second to last, ...), units of
// a file by id, redundancy against everything placed before. Ids are distinct.
void check_before(const Selection& s, const std::vector<Candidate>& candidates, const Options& o) {
  std::map<std::uint32_t, double> strength;
  for (const Candidate& c : candidates) {
    const double p = std::min(1.0, std::max(0.0, c.p));
    const auto it = strength.find(c.group);
    if (it == strength.end()) {
      strength.emplace(c.group, p);
    } else {
      it->second = std::max(it->second, p);
    }
  }
  std::vector<std::pair<double, std::uint32_t>> ranked;
  for (const auto& [group, p] : strength) ranked.emplace_back(-p, group);
  std::sort(ranked.begin(), ranked.end());  // strength descending, then file index ascending
  std::vector<std::uint32_t> files(ranked.size());
  std::size_t front = 0;
  std::size_t back = ranked.size();
  for (std::size_t r = 0; r < ranked.size(); ++r) {
    if (!o.edge_ordering || r % 2 == 0) {
      files[o.edge_ordering ? front++ : r] = ranked[r].second;
    } else {
      files[--back] = ranked[r].second;
    }
  }
  std::vector<const Candidate*> prompt;
  for (const std::uint32_t file : files) {
    std::vector<const Candidate*> members;
    for (const Candidate& c : candidates) {
      if (c.group == file) members.push_back(&c);
    }
    std::sort(members.begin(), members.end(), [](const Candidate* a, const Candidate* b) { return a->id < b->id; });
    prompt.insert(prompt.end(), members.begin(), members.end());
  }
  double mass = 0.0;
  std::uint64_t length = static_cast<std::uint64_t>(o.group_overhead) * files.size();
  for (std::size_t k = 0; k < prompt.size(); ++k) {
    double rho = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      rho = std::max(rho, reference_similarity(prompt[i]->embedding, prompt[k]->embedding, o.dim));
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(prompt[k]->tokens_full) + o.unit_overhead;
    mass += (1.0 - rho) * std::min(1.0, std::max(0.0, prompt[k]->p)) * static_cast<double>(tokens);
    length += tokens;
  }
  CHECK_EQ(static_cast<std::uint64_t>(s.tokens_before), length);
  CHECK_NEAR(s.quality_before, mass / static_cast<double>(length), 1e-12);
}

// Counters partition the candidates; files are contiguous and in source order.
void check_structure(const Selection& s, const std::vector<Candidate>& candidates) {
  std::set<UnitId> distinct;
  for (const Candidate& c : candidates) distinct.insert(c.id);
  CHECK_EQ(s.entries.size() + s.dropped_low_quality + s.dropped_redundant + s.dropped_budget, distinct.size());

  std::size_t skeletons = 0;
  std::size_t supports = 0;
  std::set<UnitId> seen;
  std::set<std::uint32_t> closed;
  for (std::size_t k = 0; k < s.entries.size(); ++k) {
    const Entry& e = s.entries[k];
    CHECK(seen.insert(e.id).second);  // no unit twice
    if (e.support) ++supports;
    if (!e.support && e.form == Form::Skeleton) ++skeletons;
    if (k != 0) {
      const Entry& previous = s.entries[k - 1];
      if (previous.group == e.group) {
        CHECK(previous.id < e.id);
      } else {
        CHECK(closed.insert(previous.group).second);
        CHECK(closed.count(e.group) == 0u);  // a file is not visited twice
      }
    }
  }
  CHECK_EQ(s.skeletonised, skeletons);
  CHECK_EQ(s.support_added, supports);
  CHECK(s.passes >= 1u);
  CHECK(s.passes <= distinct.size());
}

bool same(const Selection& a, const Selection& b) {
  if (a.entries.size() != b.entries.size() || a.groups.size() != b.groups.size()) return false;
  for (std::size_t i = 0; i < a.entries.size(); ++i) {
    const Entry& x = a.entries[i];
    const Entry& y = b.entries[i];
    if (x.id != y.id || x.group != y.group || x.form != y.form || x.support != y.support || x.p != y.p ||
        x.redundancy != y.redundancy || x.quality != y.quality || x.tokens != y.tokens) {
      return false;
    }
  }
  for (std::size_t i = 0; i < a.groups.size(); ++i) {
    const NodeQuality& x = a.groups[i];
    const NodeQuality& y = b.groups[i];
    if (x.group != y.group || x.tokens != y.tokens || x.mass != y.mass || x.quality != y.quality ||
        x.units != y.units) {
      return false;
    }
  }
  return a.tokens_before == b.tokens_before && a.tokens_after == b.tokens_after &&
         a.quality_before == b.quality_before && a.quality_after == b.quality_after &&
         a.rot_before == b.rot_before && a.rot_after == b.rot_after && a.compression == b.compression &&
         a.passes == b.passes && a.dropped_low_quality == b.dropped_low_quality &&
         a.dropped_redundant == b.dropped_redundant && a.dropped_budget == b.dropped_budget &&
         a.skeletonised == b.skeletonised && a.support_added == b.support_added;
}

// A random candidate set. Embeddings come in clusters, with exact copies among
// them, so that every level of redundancy occurs.
struct Pool {
  std::vector<std::vector<float>> embeddings;
  std::vector<Candidate> candidates;
  std::size_t dim = 0;
};

Pool random_pool(Rng& rng, std::size_t n, std::size_t dim, bool with_requires) {
  Pool pool;
  pool.dim = dim;
  const std::size_t clusters = 1 + rng.below(4);
  std::vector<std::vector<double>> centres(clusters, std::vector<double>(dim));
  for (auto& centre : centres) {
    for (double& x : centre) x = rng.normal();
  }
  const double spread = 0.05 + 0.6 * rng.uniform();
  const double scale = rng.below(10) == 0 ? 0.24 : 1.0;  // one set in ten has nothing above theta = 0.25
  const std::uint32_t files = static_cast<std::uint32_t>(1 + rng.below(6));
  pool.embeddings.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<float>& e = pool.embeddings[i];
    if (dim != 0 && i != 0 && rng.below(8) == 0) {
      e = pool.embeddings[rng.below(i)];  // exact copy of an earlier unit
    } else {
      e.resize(dim);
      const auto& centre = centres[rng.below(clusters)];
      double norm = 0.0;
      for (std::size_t d = 0; d < dim; ++d) {
        const double x = centre[d] + spread * rng.normal();
        e[d] = static_cast<float>(x);
        norm += static_cast<double>(e[d]) * static_cast<double>(e[d]);
      }
      norm = std::sqrt(norm);
      for (std::size_t d = 0; d < dim; ++d) e[d] = static_cast<float>(static_cast<double>(e[d]) / norm);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    // Probabilities on a grid of 1/64, so that ties occur.
    const double p = scale * static_cast<double>(rng.below(65)) / 64.0;
    const auto full = static_cast<std::uint32_t>(20 + rng.below(400));
    const auto skeleton = rng.below(4) == 0 ? 0u : static_cast<std::uint32_t>(5 + rng.below(60));
    Candidate c = unit(static_cast<UnitId>(100 + 7 * i), static_cast<std::uint32_t>(rng.below(files)), p, full,
                       skeleton, dim != 0 ? pool.embeddings[i].data() : nullptr);
    if (with_requires) {
      const std::size_t count = rng.below(4);
      for (std::size_t k = 0; k < count; ++k) {
        c.requires_units.push_back(static_cast<UnitId>(100 + 7 * rng.below(n + 2)));  // may name an unknown unit
      }
    }
    pool.candidates.push_back(std::move(c));
  }
  return pool;
}

const Candidate& top_of(const std::vector<Candidate>& candidates) {
  return *std::min_element(candidates.begin(), candidates.end(), ranks_before);
}

// The five units in two files used by the hand-computed examples.
//   file 0: ids 10, 11, 12    p = 0.9, 0.5, 0.3    L = 100, 50, 200 (8 tokens of unit overhead included)
//   file 1: ids 20, 21        p = 0.8, 0.4         L = 80, 120
// Cosines: (10,11) 1/2, (10,12) 0, (11,12) 1/2, (10,21) 3/4, (11,21) 3/4, (12,21) 1/4, unit 20 orthogonal to all.
struct Example {
  SignVector e10 = signs({});
  SignVector e11 = signs({12, 13, 14, 15});
  SignVector e12 = signs({8, 9, 10, 11, 12, 13, 14, 15});
  SignVector e20 = signs({1, 3, 5, 7, 9, 11, 13, 15});
  SignVector e21 = signs({14, 15});

  std::vector<Candidate> candidates(bool with_embeddings) const {
    const auto e = [&](const SignVector& v) { return with_embeddings ? v.data() : nullptr; };
    return {unit(20, 1, 0.8, 72, 0, e(e20)), unit(12, 0, 0.3, 192, 0, e(e12)), unit(10, 0, 0.9, 92, 0, e(e10)),
            unit(21, 1, 0.4, 112, 0, e(e21)), unit(11, 0, 0.5, 42, 0, e(e11))};
  }
};

}  // namespace

// ---- 1. hand-computed L, M, Q ----------------------------------------------

PRAG_TEST(sign_vectors_have_the_cosines_the_examples_assume) {
  const Example x;
  CHECK_EQ(reference_similarity(x.e10.data(), x.e10.data(), kSignDim), 1.0);
  CHECK_EQ(reference_similarity(x.e10.data(), x.e11.data(), kSignDim), 0.5);
  CHECK_EQ(reference_similarity(x.e10.data(), x.e12.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(x.e11.data(), x.e12.data(), kSignDim), 0.5);
  CHECK_EQ(reference_similarity(x.e10.data(), x.e21.data(), kSignDim), 0.75);
  CHECK_EQ(reference_similarity(x.e11.data(), x.e21.data(), kSignDim), 0.75);
  CHECK_EQ(reference_similarity(x.e12.data(), x.e21.data(), kSignDim), 0.25);
  CHECK_EQ(reference_similarity(x.e20.data(), x.e10.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(x.e20.data(), x.e11.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(x.e20.data(), x.e12.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(x.e20.data(), x.e21.data(), kSignDim), 0.0);
}

PRAG_TEST(hand_computed_quality_without_redundancy) {
  const Example x;
  const std::vector<Candidate> candidates = x.candidates(false);
  const Selection s = select(candidates, wide_open());

  // Every p >= 0.25, nothing is redundant: all five units stay.
  //   file 0: L = 12 + 100 + 50 + 200 = 362   M = 90 + 25 + 60 = 175
  //   file 1: L = 12 + 80 + 120       = 212   M = 64 + 48      = 112
  //   context: L = 574, M = 287, Q = 1/2
  const std::vector<UnitId> expected = {10, 11, 12, 20, 21};
  CHECK(ids_of(s) == expected);
  REQUIRE(s.groups.size() == 2u);
  CHECK_EQ(s.groups[0].group, 0u);
  CHECK_EQ(s.groups[0].tokens, 362u);
  CHECK_EQ(s.groups[0].units, 3u);
  CHECK_NEAR(s.groups[0].mass, 175.0, 1e-12);
  CHECK_NEAR(s.groups[0].quality, 175.0 / 362.0, 1e-12);
  CHECK_EQ(s.groups[1].group, 1u);
  CHECK_EQ(s.groups[1].tokens, 212u);
  CHECK_EQ(s.groups[1].units, 2u);
  CHECK_NEAR(s.groups[1].mass, 112.0, 1e-12);
  CHECK_NEAR(s.groups[1].quality, 112.0 / 212.0, 1e-12);
  CHECK_EQ(s.tokens_before, 574u);
  CHECK_EQ(s.tokens_after, 574u);
  CHECK_NEAR(s.quality_before, 0.5, 1e-12);
  CHECK_NEAR(s.quality_after, 0.5, 1e-12);
  CHECK_NEAR(s.rot_before, 0.5, 1e-12);
  CHECK_NEAR(s.rot_after, 0.5, 1e-12);
  CHECK_NEAR(s.compression, 1.0, 1e-15);
  CHECK_EQ(s.passes, std::size_t{1});
  for (const Entry& e : s.entries) {
    CHECK_EQ(e.redundancy, 0.0);
    CHECK_EQ(e.quality, e.p);
    CHECK(e.form == Form::Full);
    CHECK(!e.support);
  }
  CHECK_EQ(entry_of(s, 12).tokens, 200u);
  check_arithmetic(s, candidates, wide_open());
  check_structure(s, candidates);
}

PRAG_TEST(hand_computed_quality_with_redundancy) {
  const Example x;
  const std::vector<Candidate> candidates = x.candidates(true);
  Options o = wide_open();
  o.dim = kSignDim;
  const Selection s = select(candidates, o);

  // Unpruned context, prompt order 10 11 12 | 20 21:
  //   rho = 0, 1/2, 1/2 | 0, 3/4
  //   M   = 90 + 12.5 + 30 + 64 + 12 = 208.5,  L = 574
  CHECK_EQ(s.tokens_before, 574u);
  CHECK_NEAR(s.quality_before, 208.5 / 574.0, 1e-12);
  CHECK_NEAR(s.rot_before, 1.0 - 208.5 / 574.0, 1e-12);

  // Pruning in probability order 10 20 11 21 12, theta = 1/4:
  //   20: rho 0              keep
  //   11: rho 1/2  q = 0.25  keep (the threshold itself is inside)
  //   21: rho 3/4  q = 0.10  drop
  //   12: rho 1/2  q = 0.15  drop
  // The second pass confirms the set.
  const std::vector<UnitId> expected = {10, 11, 20};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.passes, std::size_t{2});
  CHECK_EQ(s.dropped_redundant, std::size_t{2});
  CHECK_EQ(s.dropped_low_quality, std::size_t{0});
  CHECK_EQ(s.dropped_budget, std::size_t{0});

  //   file 0: L = 12 + 100 + 50 = 162   M = 90 + 0.5 * 25 = 102.5
  //   file 1: L = 12 + 80       =  92   M = 64
  //   context: L = 254, M = 166.5
  REQUIRE(s.entries.size() == 3u);
  CHECK_EQ(s.entries[0].redundancy, 0.0);
  CHECK_EQ(s.entries[1].redundancy, 0.5);
  CHECK_EQ(s.entries[1].quality, 0.25);
  CHECK_EQ(s.entries[2].redundancy, 0.0);
  REQUIRE(s.groups.size() == 2u);
  CHECK_EQ(s.groups[0].tokens, 162u);
  CHECK_NEAR(s.groups[0].mass, 102.5, 1e-12);
  CHECK_NEAR(s.groups[0].quality, 102.5 / 162.0, 1e-12);
  CHECK_EQ(s.groups[1].tokens, 92u);
  CHECK_NEAR(s.groups[1].mass, 64.0, 1e-12);
  CHECK_NEAR(s.groups[1].quality, 64.0 / 92.0, 1e-12);
  CHECK_EQ(s.tokens_after, 254u);
  CHECK_NEAR(s.quality_after, 166.5 / 254.0, 1e-12);
  CHECK_NEAR(s.rot_after, 1.0 - 166.5 / 254.0, 1e-12);
  CHECK_NEAR(s.compression, 254.0 / 574.0, 1e-12);
  check_arithmetic(s, candidates, o);
  check_structure(s, candidates);
  check_before(s, candidates, o);
}

PRAG_TEST(redundancy_follows_the_prompt_order) {
  // One file, unit 1 (p 0.5) stands before unit 2 (p 0.9) in the source, cosine 1/2.
  // Pruning looks at them in probability order: unit 1 has (1 - 1/2) 0.5 = 0.25, it stays.
  // In the prompt unit 1 comes first, so the discount falls on unit 2:
  //   M = 0.5 * 100 + (1 - 1/2) * 0.9 * 100 = 95,  L = 12 + 200
  // (in probability order it would be 90 + 25 = 115).
  const SignVector a = signs({});
  const SignVector b = signs({12, 13, 14, 15});
  const std::vector<Candidate> candidates = {unit(2, 0, 0.9, 92, 0, a.data()), unit(1, 0, 0.5, 92, 0, b.data())};
  Options o = wide_open();
  o.dim = kSignDim;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {1, 2};
  REQUIRE(ids_of(s) == expected);
  CHECK_EQ(s.entries[0].redundancy, 0.0);
  CHECK_EQ(s.entries[0].quality, 0.5);
  CHECK_EQ(s.entries[1].redundancy, 0.5);
  CHECK_EQ(s.entries[1].quality, 0.45);
  CHECK_EQ(s.tokens_after, 212u);
  CHECK_NEAR(s.quality_after, 95.0 / 212.0, 1e-12);
  CHECK_NEAR(s.quality_before, 95.0 / 212.0, 1e-12);
  check_before(s, candidates, o);

  // Both discounted qualities reach theta although the order changed: for a
  // kept pair (1 - c) min(p_i, p_j) >= theta, whichever of the two comes first.
  for (const Entry& e : s.entries) CHECK(e.quality >= o.theta);
}

PRAG_TEST(embeddings_are_ignored_without_a_dimension_or_a_pointer) {
  const Example x;
  std::vector<Candidate> candidates = x.candidates(true);
  const Selection without_dim = select(candidates, wide_open());  // dim = 0
  CHECK_EQ(without_dim.entries.size(), std::size_t{5});
  CHECK_NEAR(without_dim.quality_after, 0.5, 1e-12);

  // Unit 21 without an embedding is redundant to nothing and nothing is redundant to it.
  for (Candidate& c : candidates) {
    if (c.id == 21) c.embedding = nullptr;
  }
  Options o = wide_open();
  o.dim = kSignDim;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {10, 11, 20, 21};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(entry_of(s, 21).redundancy, 0.0);
  check_arithmetic(s, candidates, o);
}

PRAG_TEST(similarity_agrees_with_an_independent_computation) {
  // Two units in one context: the redundancy of the second is their clipped cosine.
  Rng rng(2024);
  for (const std::size_t dim : {std::size_t{1}, std::size_t{7}, std::size_t{16}, std::size_t{33}, std::size_t{384},
                                std::size_t{1024}}) {
    for (int round = 0; round < 20; ++round) {
      std::vector<float> a(dim);
      std::vector<float> b(dim);
      for (std::size_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(rng.normal());
        b[i] = static_cast<float>(0.7 * static_cast<double>(a[i]) + rng.normal());
      }
      if (round % 5 == 4) {  // not normalised, and pointing away
        for (float& v : b) v *= -3.0f;
      }
      const std::vector<Candidate> pair = {unit(0, 0, 0.9, 10, 0, a.data()), unit(1, 0, 0.8, 10, 0, b.data())};
      Options o = wide_open();
      o.dim = dim;
      const Selection s = evaluate(pair, o);
      REQUIRE(s.entries.size() == 2u);
      CHECK_NEAR(s.entries[1].redundancy, reference_similarity(a.data(), b.data(), dim), 1e-12);
      CHECK(s.entries[1].redundancy >= 0.0 && s.entries[1].redundancy <= 1.0);
    }
  }
}

PRAG_TEST(similarity_covers_every_pair_when_the_rows_fill_several_blocks) {
  // The cosines are computed block by block, a block being as many rows as fit
  // 16 KiB together with the row streamed past them: 3 rows of dimension 1024,
  // one row from dimension 2048 on. Rows are unit vectors of the axes, so every
  // cosine is 0, except for one pair of equal rows that is moved through every
  // position: exactly the later row of that pair has redundancy 1.
  struct Case {
    std::size_t dim;
    std::size_t rows;
  };
  for (const Case& x : {Case{1024, 20}, Case{1365, 9}, Case{2048, 7}, Case{5000, 6}, Case{64, 30}}) {
    std::vector<std::vector<float>> axes(x.rows, std::vector<float>(x.dim, 0.0f));
    for (std::size_t k = 0; k < x.rows; ++k) axes[k][(k * 37 + 5) % x.dim] = 1.0f;
    Options o = wide_open();
    o.dim = x.dim;
    std::size_t pairs = 0;
    for (std::size_t a = 0; a < x.rows; ++a) {
      for (std::size_t b = a + 1; b < x.rows; ++b) {
        std::vector<Candidate> list;
        for (std::size_t k = 0; k < x.rows; ++k) {
          list.push_back(unit(static_cast<UnitId>(k), 0, 0.5, 10, 0, axes[k == b ? a : k].data()));
        }
        const Selection s = evaluate(list, o);
        REQUIRE(s.entries.size() == x.rows);
        bool exact = true;
        for (std::size_t k = 0; k < x.rows; ++k) exact = exact && s.entries[k].redundancy == (k == b ? 1.0 : 0.0);
        CHECK(exact);
        ++pairs;
      }
    }
    CHECK_EQ(pairs, x.rows * (x.rows - 1) / 2);
  }

  // Random rows in several blocks, every pair against the plain computation:
  // with two rows moved to the front, the second one has their cosine as redundancy.
  Rng rng(606);
  constexpr std::size_t dim = 1024;
  constexpr std::size_t rows = 14;
  std::vector<std::vector<float>> e(rows, std::vector<float>(dim));
  for (auto& row : e) {
    for (float& v : row) v = static_cast<float>(rng.normal());
  }
  for (std::size_t k = 1; k < rows; k += 2) {  // neighbours are related
    for (std::size_t d = 0; d < dim; ++d) e[k][d] += 2.0f * e[k - 1][d];
  }
  Options o = wide_open();
  o.dim = dim;
  std::vector<Candidate> list;
  for (std::size_t k = 0; k < rows; ++k) list.push_back(unit(static_cast<UnitId>(k), 0, 0.5, 10, 0, e[k].data()));
  const Selection s = evaluate(list, o);
  REQUIRE(s.entries.size() == rows);
  double largest = 0.0;
  for (std::size_t k = 0; k < rows; ++k) {
    double rho = 0.0;
    for (std::size_t i = 0; i < k; ++i) rho = std::max(rho, reference_similarity(e[i].data(), e[k].data(), dim));
    CHECK_NEAR(s.entries[k].redundancy, rho, 1e-12);
    largest = std::max(largest, rho);
  }
  CHECK(largest > 0.8);
}

// ---- 2. threshold guarantee ------------------------------------------------

PRAG_TEST(threshold_guarantee_on_random_candidate_sets) {
  Rng rng(7);
  const double thetas[] = {0.1, 0.25, 0.4};
  const std::uint32_t budgets[] = {1000000u, 3000u, 1200u, 400u, 90u};
  std::size_t sets = 0;
  std::size_t forced = 0;
  std::size_t with_support = 0;
  std::size_t with_redundancy = 0;
  double smallest_margin = 1.0;
  for (int round = 0; round < 600; ++round) {
    const std::size_t n = 1 + rng.below(40);
    const Pool pool = random_pool(rng, n, 12, true);
    Options o;
    o.theta = thetas[rng.below(3)];
    o.token_budget = budgets[rng.below(5)];
    o.dim = pool.dim;
    o.edge_ordering = rng.below(2) == 0;
    const Selection s = select(pool.candidates, o);
    ++sets;

    check_arithmetic(s, pool.candidates, o);
    check_structure(s, pool.candidates);
    check_before(s, pool.candidates, o);

    const Candidate& top = top_of(pool.candidates);
    const Entry* top_entry = find(s, top.id);
    REQUIRE(top_entry != nullptr);  // never an empty context
    CHECK(!top_entry->support);

    std::uint64_t kept_tokens = 0;
    std::uint64_t support_tokens = 0;
    std::size_t kept = 0;
    for (const Entry& e : s.entries) {
      if (e.support) {
        CHECK_EQ(e.redundancy, 0.0);
        CHECK_EQ(e.quality, e.p);
        support_tokens += e.tokens;
        continue;
      }
      ++kept;
      kept_tokens += e.tokens;
      if (e.redundancy > 0.0) ++with_redundancy;
      if (top.p >= o.theta) {
        CHECK(e.quality >= o.theta);
        smallest_margin = std::min(smallest_margin, e.quality - o.theta);
      }
    }
    if (s.support_added != 0) ++with_support;
    if (top.p < o.theta) {
      ++forced;
      CHECK_EQ(kept, std::size_t{1});  // the top-ranked unit alone, kept although below theta
      continue;
    }
    const std::uint64_t overhead = static_cast<std::uint64_t>(o.group_overhead) * s.groups.size();
    CHECK_EQ(kept_tokens + support_tokens + overhead, static_cast<std::uint64_t>(s.tokens_after));
    const double bound = o.theta * static_cast<double>(kept_tokens) / static_cast<double>(s.tokens_after);
    CHECK(s.quality_after >= bound - 1e-12);
  }
  CHECK(sets >= 500u);
  CHECK(sets - forced >= 500u);  // sets on which the guarantee was tested in full
  // The sample exercises every branch the guarantee talks about.
  CHECK(forced >= 20u);
  CHECK(with_support >= 50u);
  CHECK(with_redundancy >= 500u);
  CHECK(smallest_margin >= 0.0);
}

PRAG_TEST(removing_a_child_raises_quality_exactly_when_it_is_below_the_parent) {
  // Q(v \ u) > Q(v)  <=>  (1 - rho(u)) p_u < Q(v), the redundancy of the other
  // children held fixed. Removing the last child leaves the others untouched.
  Rng rng(99);
  Options o = wide_open();
  o.dim = 12;
  o.group_overhead = 0;
  o.unit_overhead = 0;
  std::size_t raised = 0;
  std::size_t lowered = 0;
  for (int round = 0; round < 2000; ++round) {
    Pool pool = random_pool(rng, 2 + rng.below(10), 12, false);
    for (Candidate& c : pool.candidates) c.group = 0;
    const Selection whole = evaluate(pool.candidates, o);
    const Entry last = whole.entries.back();
    pool.candidates.pop_back();
    const Selection rest = evaluate(pool.candidates, o);
    const double gap = last.quality - whole.quality_after;
    if (std::fabs(gap) < 1e-9) continue;  // too close to call in floating point
    CHECK_EQ(rest.quality_after > whole.quality_after, gap < 0.0);
    if (gap < 0.0) ++raised; else ++lowered;
  }
  CHECK(raised > 200u);
  CHECK(lowered > 200u);
}

// ---- 3. fixed point --------------------------------------------------------

PRAG_TEST(a_unit_returns_when_the_unit_it_duplicated_is_dropped) {
  // A (p 0.9), B (p 0.5), C (p 0.45); cos(A,B) = 5/8, cos(B,C) = 3/4, cos(A,C) = 3/8.
  //   pass 1 (against A, B, C): B: (1 - 5/8) 0.5  = 0.1875 drop;  C: (1 - 3/4) 0.45 = 0.1125 drop
  //   pass 2 (against A):       B: drop;                          C: (1 - 3/8) 0.45 = 0.28125 keep
  //   pass 3 confirms {A, C}.
  const SignVector a = signs({});
  const SignVector b = signs({0, 1, 2});
  const SignVector c = signs({0, 1, 2, 3, 4});
  const std::vector<Candidate> candidates = {unit(2, 0, 0.45, 100, 0, c.data()), unit(0, 0, 0.9, 100, 0, a.data()),
                                             unit(1, 0, 0.5, 100, 0, b.data())};
  Options o = wide_open();
  o.dim = kSignDim;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {0, 2};
  CHECK(ids_of(s) == expected);
  CHECK(s.passes > 1u);
  CHECK_EQ(s.passes, std::size_t{3});
  CHECK(s.passes <= candidates.size());
  CHECK_EQ(s.dropped_redundant, std::size_t{1});
  REQUIRE(s.entries.size() == 2u);
  CHECK_EQ(s.entries[1].redundancy, 0.375);
  CHECK_EQ(s.entries[1].quality, 0.28125);
  // L = 12 + 108 + 108 = 228, M = 0.9 * 108 + 0.28125 * 108.
  CHECK_EQ(s.tokens_after, 228u);
  CHECK_NEAR(s.quality_after, (0.9 + 0.28125) * 108.0 / 228.0, 1e-12);
  // Unpruned, prompt order A B C: rho = 0, 5/8, 3/4.
  CHECK_EQ(s.tokens_before, 336u);
  CHECK_NEAR(s.quality_before, (0.9 + 0.375 * 0.5 + 0.25 * 0.45) * 108.0 / 336.0, 1e-12);

  // The limit on the passes bounds the work, not the result.
  for (const std::size_t limit : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
    o.max_passes = limit;
    const Selection limited = select(candidates, o);
    CHECK(ids_of(limited) == expected);
    CHECK(limited.passes <= std::max<std::size_t>(1, limit));
    CHECK_EQ(limited.quality_after, s.quality_after);
  }
}

PRAG_TEST(a_chain_of_duplicates_needs_as_many_passes_as_it_has_units) {
  // e_j = (u_j + u_{j+1}) / sqrt(2): neighbours have cosine 1/2, everything
  // else is orthogonal. With p = 0.4 a unit survives exactly when its
  // predecessor does not, so units 0, 2, 4, 6 stay, and the iteration settles
  // one unit per pass.
  constexpr std::size_t n = 8;
  constexpr std::size_t dim = n + 1;
  std::vector<std::vector<float>> e(n, std::vector<float>(dim, 0.0f));
  std::vector<Candidate> candidates;
  for (std::size_t j = 0; j < n; ++j) {
    e[j][j] = e[j][j + 1] = static_cast<float>(std::sqrt(0.5));
    candidates.push_back(unit(static_cast<UnitId>(j), 0, 0.4, 50, 0, e[j].data()));
  }
  Options o = wide_open();
  o.dim = dim;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {0, 2, 4, 6};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.passes, n);
  CHECK_EQ(s.dropped_redundant, std::size_t{4});
  for (const Entry& entry : s.entries) CHECK_EQ(entry.redundancy, 0.0);

  o.max_passes = 3;
  const Selection limited = select(candidates, o);
  CHECK(ids_of(limited) == expected);
  CHECK_EQ(limited.passes, std::size_t{3});
}

PRAG_TEST(fixed_point_is_the_greedy_selection_and_needs_at_most_n_passes) {
  Rng rng(31337);
  std::size_t multi_pass = 0;
  std::size_t most_passes = 0;
  for (int round = 0; round < 400; ++round) {
    const std::size_t n = 1 + rng.below(48);
    const Pool pool = random_pool(rng, n, 10, false);
    Options o = wide_open();
    o.dim = pool.dim;
    o.max_passes = 1000;
    o.theta = round % 2 == 0 ? 0.25 : 0.15;
    const Selection s = select(pool.candidates, o);
    CHECK(s.passes <= n);
    CHECK(selected_ids(s) == greedy_reference(pool.candidates, o.theta, pool.dim));
    if (s.passes > 2) ++multi_pass;
    most_passes = std::max(most_passes, s.passes);

    o.max_passes = 1 + rng.below(3);
    const Selection limited = select(pool.candidates, o);
    CHECK(limited.passes <= o.max_passes);
    CHECK(ids_of(limited) == ids_of(s));
    CHECK_EQ(limited.quality_after, s.quality_after);
  }
  CHECK(multi_pass > 20u);  // the sample does contain returning units
  CHECK(most_passes >= 4u);
}

// ---- 4. exact duplicates ---------------------------------------------------

PRAG_TEST(only_the_most_probable_copy_of_a_duplicate_is_kept) {
  const SignVector copy = signs({3, 4, 9});
  const SignVector other = signs({0, 1, 2, 3, 4, 5, 6, 7});  // differs from `copy` in 7 signs: cosine 1/8
  const std::vector<Candidate> candidates = {
      unit(5, 0, 0.6, 100, 0, copy.data()), unit(6, 1, 0.9, 100, 0, copy.data()),
      unit(7, 2, 0.7, 100, 0, copy.data()), unit(8, 0, 0.5, 100, 0, other.data())};
  Options o = wide_open();
  o.dim = kSignDim;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {6, 8};  // file 1 (0.9) first, file 0 (0.5) last
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.dropped_redundant, std::size_t{2});
  CHECK_EQ(entry_of(s, 8).redundancy, 0.125);
  check_structure(s, candidates);

  // In the unpruned context the copies carry no mass at all.
  //   prompt order: file 1 [6], file 0 [5, 8], file 2 [7]  (0.9 first, 0.7 last, 0.6 between)
  //   rho = 0 | 1, 1/8 | 1
  //   M = 0.9 * 108 + 0 + 0 + (1 - 1/8) * 0.5 * 108,  L = 3 * 12 + 4 * 108
  CHECK_EQ(s.tokens_before, 468u);
  CHECK_NEAR(s.quality_before, (0.9 + 0.875 * 0.5) * 108.0 / 468.0, 1e-12);

  // Copies that are not normalised and differ by a factor are still copies.
  std::vector<float> scaled(copy.begin(), copy.end());
  for (float& v : scaled) v *= 3.0f;
  const std::vector<Candidate> unnormalised = {unit(1, 0, 0.9, 100, 0, copy.data()),
                                               unit(2, 0, 0.8, 100, 0, scaled.data())};
  const Selection t = select(unnormalised, o);
  CHECK_EQ(t.entries.size(), std::size_t{1});
  CHECK_EQ(t.entries[0].id, UnitId{1});
}

PRAG_TEST(a_unit_id_given_twice_counts_once) {
  const std::vector<Candidate> candidates = {unit(3, 0, 0.4, 100), unit(3, 0, 0.7, 100), unit(4, 0, 0.5, 100)};
  const Selection s = select(candidates, wide_open());
  REQUIRE(s.entries.size() == 2u);
  CHECK_EQ(s.entries[0].id, UnitId{3});
  CHECK_EQ(s.entries[0].p, 0.7);
  CHECK_EQ(s.tokens_before, 12u + 2u * 108u);
  check_structure(s, candidates);
}

// ---- 5. budget -------------------------------------------------------------

PRAG_TEST(budget_reduces_to_skeletons_first_and_drops_afterwards) {
  // One file; L in full 308, 208, 208, 108; as skeleton 38, 28, 28, none.
  const std::vector<Candidate> candidates = {unit(0, 0, 0.9, 300, 30), unit(1, 0, 0.7, 200, 20),
                                             unit(2, 0, 0.5, 200, 20), unit(3, 0, 0.3, 100, 0)};
  struct Expectation {
    std::uint32_t budget;
    std::uint32_t tokens;
    std::vector<UnitId> ids;
    std::vector<Form> forms;
    std::size_t skeletonised;
    std::size_t dropped;
  };
  const Form F = Form::Full;
  const Form S = Form::Skeleton;
  const std::vector<Expectation> expectations = {
      {844, 844, {0, 1, 2, 3}, {F, F, F, F}, 0, 0},  // fits as it is
      {843, 664, {0, 1, 2, 3}, {F, F, S, F}, 1, 0},  // unit 3 is weaker but has no skeleton: unit 2 goes first
      {500, 484, {0, 1, 2, 3}, {F, S, S, F}, 2, 0},
      {400, 214, {0, 1, 2, 3}, {S, S, S, F}, 3, 0},  // every skeleton is used before anything is dropped
      {200, 186, {0, 1, 3}, {S, S, F}, 2, 1},        // masses 0.7*28 = 19.6, 0.5*28 = 14, 0.3*108 = 32.4: unit 2 goes
      {160, 158, {0, 3}, {S, F}, 1, 2},              // then unit 1
      {60, 50, {0}, {S}, 1, 3},                      // then unit 3
      {50, 50, {0}, {S}, 1, 3},
      {40, 50, {0}, {S}, 1, 3},                      // the top unit stays although it does not fit
  };
  for (const Expectation& x : expectations) {
    Options o = wide_open();
    o.token_budget = x.budget;
    const Selection s = select(candidates, o);
    CHECK(ids_of(s) == x.ids);
    REQUIRE(s.entries.size() == x.forms.size());
    for (std::size_t i = 0; i < x.forms.size(); ++i) CHECK(s.entries[i].form == x.forms[i]);
    CHECK_EQ(s.tokens_after, x.tokens);
    CHECK_EQ(s.skeletonised, x.skeletonised);
    CHECK_EQ(s.dropped_budget, x.dropped);
    CHECK_EQ(s.dropped_low_quality, std::size_t{0});
    CHECK_EQ(s.dropped_redundant, std::size_t{0});
    CHECK_EQ(s.tokens_before, 844u);
    if (x.budget >= 50) CHECK(s.tokens_after <= x.budget);
    check_arithmetic(s, candidates, o);
    check_structure(s, candidates);
  }
}

PRAG_TEST(budget_drops_support_entries_before_selected_units) {
  // A requires B and C, which are below theta and come back as skeletons.
  //   A: 108 / 18   D: 108 (no skeleton)   B: support 18   C: support 38   file overhead 12
  const std::vector<Candidate> candidates = {unit(0, 0, 0.9, 100, 10, nullptr, {1, 2}), unit(1, 0, 0.1, 50, 10),
                                             unit(2, 0, 0.2, 60, 30), unit(3, 0, 0.5, 100, 0)};
  struct Expectation {
    std::uint32_t budget;
    std::uint32_t tokens;
    std::vector<UnitId> ids;
    std::size_t support;
    std::size_t skeletonised;
    std::size_t dropped;
  };
  const std::vector<Expectation> expectations = {
      {284, 284, {0, 1, 2, 3}, 2, 0, 0},
      {250, 194, {0, 1, 2, 3}, 2, 1, 0},  // A is reduced before a support entry is dropped
      {180, 176, {0, 2, 3}, 1, 1, 1},     // B (0.1 * 18) before C (0.2 * 38)
      {140, 138, {0, 3}, 0, 1, 2},
      {100, 30, {0}, 0, 1, 3},            // D goes only when no support entry is left
  };
  for (const Expectation& x : expectations) {
    Options o = wide_open();
    o.dependency_closure = true;
    o.token_budget = x.budget;
    const Selection s = select(candidates, o);
    CHECK(ids_of(s) == x.ids);
    CHECK_EQ(s.tokens_after, x.tokens);
    CHECK_EQ(s.support_added, x.support);
    CHECK_EQ(s.skeletonised, x.skeletonised);
    CHECK_EQ(s.dropped_budget, x.dropped);
    CHECK_EQ(s.dropped_low_quality, std::size_t{0});  // B and C are accounted for as support or under the budget
    CHECK(s.tokens_after <= x.budget);
    check_arithmetic(s, candidates, o);
    check_structure(s, candidates);
  }
}

PRAG_TEST(budget_rescoring_keeps_the_survivor_of_a_redundant_pair) {
  // X (top), D, J, Z; J overlaps with D (cosine 1/2), everything else is orthogonal.
  // As skeletons: L = 28, 10, 28, 28, and with the file overhead 106 tokens.
  //   discounted masses  D: 0.8 * 10 = 8    J: (1 - 1/2) * 0.7 * 28 = 9.8    Z: 0.5 * 28 = 14
  // D goes first. That frees J of its redundancy: its mass becomes 0.7 * 28 = 19.6,
  // so the next unit to go is Z and not J.
  const SignVector ex = signs({});
  const SignVector ed = signs({8, 9, 10, 11, 12, 13, 14, 15});
  const SignVector ej = signs({0, 1, 8, 9, 10, 11, 12, 13});
  const SignVector ez = signs({1, 3, 5, 7, 9, 11, 13, 15});
  CHECK_EQ(reference_similarity(ed.data(), ej.data(), kSignDim), 0.5);
  CHECK_EQ(reference_similarity(ex.data(), ej.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(ez.data(), ej.data(), kSignDim), 0.0);
  CHECK_EQ(reference_similarity(ez.data(), ed.data(), kSignDim), 0.0);
  const std::vector<Candidate> candidates = {
      unit(0, 0, 0.9, 200, 20, ex.data()), unit(1, 0, 0.8, 200, 2, ed.data()),
      unit(2, 0, 0.7, 200, 20, ej.data()), unit(3, 0, 0.5, 200, 20, ez.data())};
  Options o = wide_open();
  o.dim = kSignDim;

  o.token_budget = 106;
  Selection s = select(candidates, o);
  CHECK_EQ(s.entries.size(), std::size_t{4});
  CHECK_EQ(s.skeletonised, std::size_t{4});
  CHECK_EQ(s.tokens_after, 106u);
  CHECK_EQ(entry_of(s, 2).redundancy, 0.5);

  o.token_budget = 100;
  s = select(candidates, o);
  std::vector<UnitId> expected = {0, 2, 3};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.tokens_after, 96u);

  o.token_budget = 70;
  s = select(candidates, o);
  expected = {0, 2};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.dropped_budget, std::size_t{2});
  CHECK_EQ(s.tokens_after, 68u);
  CHECK_EQ(entry_of(s, 2).redundancy, 0.0);
  CHECK_EQ(entry_of(s, 2).quality, 0.7);
  check_arithmetic(s, candidates, o);
  check_structure(s, candidates);
}

PRAG_TEST(budget_properties_on_random_candidate_sets) {
  Rng rng(4242);
  std::size_t reduced = 0;
  std::size_t dropped = 0;
  std::size_t top_too_long = 0;
  for (int round = 0; round < 600; ++round) {
    const std::size_t n = 1 + rng.below(30);
    const Pool pool = random_pool(rng, n, 0, round % 2 == 0);  // no embeddings: quality = p
    Options o;
    o.theta = 0.25;
    o.dependency_closure = round % 2 == 0;
    o.token_budget = static_cast<std::uint32_t>(30 + rng.below(3000));
    const Selection s = select(pool.candidates, o);
    check_arithmetic(s, pool.candidates, o);
    check_structure(s, pool.candidates);

    Options unlimited = o;
    unlimited.token_budget = 4000000000u;
    const Selection all = select(pool.candidates, unlimited);
    CHECK_EQ(all.dropped_budget, std::size_t{0});
    CHECK_EQ(all.skeletonised, std::size_t{0});
    CHECK_EQ(s.tokens_before, all.tokens_before);
    CHECK_EQ(s.quality_before, all.quality_before);

    // The budget holds whenever the shortest form of the top-ranked unit fits.
    const Candidate& top = top_of(pool.candidates);
    const bool shrinks = top.tokens_skeleton != 0 && top.tokens_skeleton < top.tokens_full;
    const std::uint64_t shortest = shrinks ? top.tokens_skeleton : top.tokens_full;
    const std::uint64_t least = shortest + o.unit_overhead + o.group_overhead;
    if (least <= o.token_budget) {
      CHECK(s.tokens_after <= o.token_budget);
    } else {
      ++top_too_long;
      CHECK_EQ(s.entries.size(), std::size_t{1});
      CHECK_EQ(static_cast<std::uint64_t>(s.tokens_after), least);
    }
    // The budget step only removes.
    for (const Entry& e : s.entries) {
      const Entry* before = find(all, e.id);
      REQUIRE(before != nullptr);
      CHECK_EQ(before->support, e.support);
    }
    if (all.tokens_after <= o.token_budget) {
      CHECK(same(s, all));
      continue;
    }

    // Order of the skeletons: a selected unit that was reduced is never
    // stronger than one that could have been reduced and was not.
    double strongest_reduced = -1.0;
    double weakest_full = 2.0;
    std::size_t selected_full_shrinkable = 0;
    for (const Entry& e : s.entries) {
      if (e.support) continue;
      const Candidate* c = find(pool.candidates, e.id);
      if (e.form == Form::Skeleton) {
        strongest_reduced = std::max(strongest_reduced, e.p);
      } else if (c->tokens_skeleton != 0 && c->tokens_skeleton < c->tokens_full) {
        weakest_full = std::min(weakest_full, e.p);
        ++selected_full_shrinkable;
      }
    }
    if (s.skeletonised != 0) ++reduced;
    CHECK(strongest_reduced <= weakest_full);

    if (s.dropped_budget == 0) continue;
    ++dropped;
    // Nothing is dropped while a selected unit could still be reduced.
    CHECK_EQ(selected_full_shrinkable, std::size_t{0});
    // Support entries go first, and the weakest selected units after them:
    // a dropped selected unit never has more discounted mass than a survivor.
    const std::vector<UnitId> selected_before = selected_ids(all);
    const std::vector<UnitId> selected_after = selected_ids(s);
    if (selected_after.size() < selected_before.size()) CHECK_EQ(s.support_added, std::size_t{0});
    const auto mass = [&](UnitId id) {
      const Candidate* c = find(pool.candidates, id);
      const bool can_shrink = c->tokens_skeleton != 0 && c->tokens_skeleton < c->tokens_full;
      const std::uint32_t length = (can_shrink ? c->tokens_skeleton : c->tokens_full) + o.unit_overhead;
      return std::min(1.0, std::max(0.0, c->p)) * static_cast<double>(length);
    };
    double heaviest_dropped = -1.0;
    double lightest_survivor = 1e300;
    for (const UnitId id : selected_before) {
      if (id == top.id) continue;
      if (std::binary_search(selected_after.begin(), selected_after.end(), id)) {
        lightest_survivor = std::min(lightest_survivor, mass(id));
      } else {
        heaviest_dropped = std::max(heaviest_dropped, mass(id));
      }
    }
    CHECK(heaviest_dropped <= lightest_survivor);
  }
  CHECK(reduced > 100u);
  CHECK(dropped > 100u);
  CHECK(top_too_long > 5u);
}

PRAG_TEST(full_text_share_keeps_the_strongest_units_in_full) {
  // The units of the budget example; L in full 308, 208, 208, 108, file overhead 12.
  const std::vector<Candidate> candidates = {unit(0, 0, 0.9, 300, 30), unit(1, 0, 0.7, 200, 20),
                                             unit(2, 0, 0.5, 200, 20), unit(3, 0, 0.3, 100, 0)};
  struct Expectation {
    double share;
    std::uint32_t budget;
    std::uint32_t tokens;
    std::vector<UnitId> ids;
    std::vector<Form> forms;
    std::size_t dropped;
  };
  const Form F = Form::Full;
  const Form S = Form::Skeleton;
  const std::vector<Expectation> expectations = {
      {0.5, 844, 844, {0, 1, 2, 3}, {F, F, F, F}, 0},  // fits: the share plays no part
      {0.5, 700, 664, {0, 1, 2, 3}, {F, F, S, F}, 0},  // reserve 350: unit 0 (320); same result as without
      {0.5, 640, 484, {0, 1, 2, 3}, {F, S, S, F}, 0},  // reserve 320: unit 0 just fits
      {0.5, 400, 214, {0, 1, 2, 3}, {S, S, S, F}, 0},  // reserve 200: not even unit 0, step 4 as written
      {0.0, 650, 484, {0, 1, 2, 3}, {F, S, S, F}, 0},  // without a share ...
      {1.0, 650, 636, {0, 1, 3}, {F, F, F}, 1},        // ... and with: units 0 and 1 (528) stay, unit 2 is dropped
      {0.0, 330, 214, {0, 1, 2, 3}, {S, S, S, F}, 0},  // without: four entries, the best one a skeleton
      {1.0, 330, 320, {0}, {F}, 3},                    // with: the best one in full, alone
      {7.0, 330, 320, {0}, {F}, 3},                    // a share above 1 counts as 1
  };
  for (const Expectation& x : expectations) {
    Options o = wide_open();
    o.token_budget = x.budget;
    o.full_text_share = x.share;
    const Selection s = select(candidates, o);
    CHECK(ids_of(s) == x.ids);
    REQUIRE(s.entries.size() == x.forms.size());
    for (std::size_t i = 0; i < x.forms.size(); ++i) CHECK(s.entries[i].form == x.forms[i]);
    CHECK_EQ(s.tokens_after, x.tokens);
    CHECK_EQ(s.dropped_budget, x.dropped);
    CHECK(s.tokens_after <= x.budget);
    check_arithmetic(s, candidates, o);
    check_structure(s, candidates);
  }

  // A reserved unit is not dropped either, however little mass it has.
  //   in full: unit 0: 108, unit 1: 28; reserve 220 holds both (12 + 108 + 28 = 148)
  //   as skeletons: unit 2: 68, mass 0.5 * 68 = 34; unit 3: 68, mass 0.4 * 68 = 27.2
  //   148 + 68 + 68 = 284 > 220: unit 3 goes, although unit 1 has less mass (0.8 * 28 = 22.4)
  const std::vector<Candidate> small = {unit(0, 0, 0.9, 100, 10), unit(1, 0, 0.8, 20, 0), unit(2, 0, 0.5, 400, 60),
                                        unit(3, 0, 0.4, 400, 60)};
  Options o = wide_open();
  o.token_budget = 220;
  o.full_text_share = 1.0;
  const Selection s = select(small, o);
  const std::vector<UnitId> expected = {0, 1, 2};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.tokens_after, 216u);
  CHECK(entry_of(s, 0).form == Form::Full);
  CHECK(entry_of(s, 1).form == Form::Full);
  CHECK(entry_of(s, 2).form == Form::Skeleton);
}

PRAG_TEST(full_text_share_on_random_candidate_sets) {
  Rng rng(9001);
  const double shares[] = {0.25, 0.5, 1.0};
  std::size_t differs = 0;
  std::size_t top_in_full = 0;
  for (int round = 0; round < 600; ++round) {
    const std::size_t n = 1 + rng.below(40);
    const Pool pool = random_pool(rng, n, round % 2 == 0 ? 8 : 0, round % 3 == 0);
    Options o;
    o.dim = pool.dim;
    o.dependency_closure = round % 3 == 0;
    o.token_budget = static_cast<std::uint32_t>(30 + rng.below(3000));
    const Selection plain = select(pool.candidates, o);
    o.full_text_share = shares[rng.below(3)];
    const Selection s = select(pool.candidates, o);
    check_arithmetic(s, pool.candidates, o);
    check_structure(s, pool.candidates);
    check_before(s, pool.candidates, o);
    if (!same(s, plain)) ++differs;

    // What is guaranteed without the share is guaranteed with it.
    const Candidate& top = top_of(pool.candidates);
    const Entry& top_entry = entry_of(s, top.id);
    const bool shrinks = top.tokens_skeleton != 0 && top.tokens_skeleton < top.tokens_full;
    const std::uint64_t shortest = shrinks ? top.tokens_skeleton : top.tokens_full;
    if (shortest + o.unit_overhead + o.group_overhead <= o.token_budget) CHECK(s.tokens_after <= o.token_budget);
    if (top.p >= o.theta) {
      for (const Entry& e : s.entries) {
        if (!e.support) CHECK(e.quality >= o.theta);
      }
    }
    // The best unit is in full whenever its full text fits the reserve.
    const std::uint64_t in_full = static_cast<std::uint64_t>(top.tokens_full) + o.unit_overhead + o.group_overhead;
    const auto reserve = static_cast<std::uint64_t>(o.full_text_share * static_cast<double>(o.token_budget));
    if (in_full <= reserve) {
      CHECK(top_entry.form == Form::Full);
      if (entry_of(plain, top.id).form == Form::Skeleton) ++top_in_full;  // the share made the difference
    }
    // The selection is a subset of the selection without a budget.
    Options unlimited = o;
    unlimited.token_budget = 4000000000u;
    const Selection all = select(pool.candidates, unlimited);
    for (const Entry& e : s.entries) {
      const Entry* before = find(all, e.id);
      REQUIRE(before != nullptr);
      CHECK_EQ(before->support, e.support);
    }
    if (all.tokens_after <= o.token_budget) CHECK(same(s, all));
  }
  CHECK(differs > 50u);
  CHECK(top_in_full > 20u);
}

// ---- 6. dependency closure -------------------------------------------------

PRAG_TEST(dependency_closure_adds_required_units_as_skeletons) {
  // A requires B (p 0.1), C (p 0.02, below closure_min_p), D (selected), itself,
  // an unknown unit and E (p 0.2, no skeleton form). F requires B again.
  const std::vector<Candidate> candidates = {
      unit(0, 0, 0.9, 100, 10, nullptr, {1, 2, 3, 0, 999, 4, 1}),
      unit(1, 1, 0.1, 80, 12),
      unit(2, 1, 0.02, 80, 12),
      unit(3, 0, 0.6, 100, 10),
      unit(4, 1, 0.2, 80, 0),
      unit(5, 2, 0.5, 100, 10, nullptr, {1}),
  };
  Options o = wide_open();
  o.dependency_closure = true;
  const Selection s = select(candidates, o);

  // Files by strength: 0 (0.9), 2 (0.5), 1 (0.1, the support entry): first, last, middle.
  const std::vector<UnitId> expected = {0, 3, 1, 5};
  CHECK(ids_of(s) == expected);
  CHECK_EQ(s.support_added, std::size_t{1});
  CHECK_EQ(s.dropped_low_quality, std::size_t{2});  // C and E
  const Entry* b = find(s, 1);
  REQUIRE(b != nullptr);
  CHECK(b->support);
  CHECK(b->form == Form::Skeleton);
  CHECK_EQ(b->tokens, 20u);
  CHECK_EQ(b->redundancy, 0.0);
  CHECK_EQ(b->quality, 0.1);
  CHECK(!entry_of(s, 0).support);
  CHECK(!entry_of(s, 3).support);
  CHECK(entry_of(s, 3).form == Form::Full);

  //   L = (12 + 108 + 108) + (12 + 20) + (12 + 108) = 380
  //   M = 0.9 * 108 + 0.6 * 108 + 0.1 * 20 + 0.5 * 108 = 218
  CHECK_EQ(s.tokens_after, 380u);
  CHECK_NEAR(s.quality_after, 218.0 / 380.0, 1e-12);
  // The bound with support: theta L_kept / L = 0.25 * 324 / 380.
  CHECK(s.quality_after >= 0.25 * 324.0 / 380.0);
  check_arithmetic(s, candidates, o);
  check_structure(s, candidates);

  o.closure_min_p = 0.01;  // now C qualifies as well
  const Selection lower = select(candidates, o);
  CHECK_EQ(lower.support_added, std::size_t{2});
  CHECK(find(lower, 2) != nullptr);
  CHECK(find(lower, 4) == nullptr);

  o.closure_min_p = 0.15;  // and now not even B
  CHECK_EQ(select(candidates, o).support_added, std::size_t{0});

  o = wide_open();  // closure switched off
  const Selection off = select(candidates, o);
  CHECK_EQ(off.support_added, std::size_t{0});
  CHECK_EQ(off.entries.size(), std::size_t{3});
  CHECK_EQ(off.dropped_low_quality, std::size_t{3});
}

PRAG_TEST(support_entries_stay_out_of_the_redundancy_computation) {
  // A requires B, which is below theta and returns as support.
  //   cos(A,B) = 1/2, cos(A,C) = 1/4, cos(B,C) = 3/4
  // Prompt order A, B, C. B has redundancy 0 by definition, and C is measured
  // against A alone: 1/4, not 3/4.
  const SignVector ea = signs({});
  const SignVector eb = signs({0, 1, 2, 3});
  const SignVector ec = signs({0, 1, 2, 3, 4, 5});
  CHECK_EQ(reference_similarity(ea.data(), eb.data(), kSignDim), 0.5);
  CHECK_EQ(reference_similarity(ea.data(), ec.data(), kSignDim), 0.25);
  CHECK_EQ(reference_similarity(eb.data(), ec.data(), kSignDim), 0.75);
  const std::vector<Candidate> candidates = {unit(0, 0, 0.9, 100, 10, ea.data(), {1}),
                                             unit(1, 0, 0.1, 100, 10, eb.data()),
                                             unit(2, 0, 0.7, 100, 10, ec.data())};
  Options o = wide_open();
  o.dim = kSignDim;
  o.dependency_closure = true;
  const Selection s = select(candidates, o);
  const std::vector<UnitId> expected = {0, 1, 2};
  CHECK(ids_of(s) == expected);
  CHECK(entry_of(s, 1).support);
  CHECK_EQ(entry_of(s, 1).redundancy, 0.0);
  CHECK_EQ(entry_of(s, 1).quality, 0.1);
  CHECK_EQ(entry_of(s, 2).redundancy, 0.25);
  CHECK_EQ(entry_of(s, 2).quality, 0.75 * 0.7);
  //   L = 12 + 108 + 18 + 108 = 246,  M = 0.9 * 108 + 0.1 * 18 + 0.525 * 108
  CHECK_EQ(s.tokens_after, 246u);
  CHECK_NEAR(s.quality_after, (0.9 * 108.0 + 0.1 * 18.0 + 0.525 * 108.0) / 246.0, 1e-12);
  check_arithmetic(s, candidates, o);
  check_structure(s, candidates);

  // A copy of a selected unit that is required by it returns as support too.
  const std::vector<Candidate> copies = {unit(0, 0, 0.9, 100, 10, ea.data(), {1}),
                                         unit(1, 0, 0.8, 100, 10, ea.data())};
  const Selection t = select(copies, o);
  REQUIRE(t.entries.size() == 2u);
  CHECK(t.entries[1].support);
  CHECK_EQ(t.entries[1].redundancy, 0.0);
  CHECK_EQ(t.dropped_redundant, std::size_t{0});
  CHECK_EQ(t.support_added, std::size_t{1});
}

// ---- 7. ordering -----------------------------------------------------------

PRAG_TEST(files_are_arranged_strongest_first_second_strongest_last) {
  // File g has strength 0.9 - 0.1 g; ids inside a file are not in probability order.
  const std::vector<Candidate> candidates = {
      unit(3, 0, 0.5, 50),  unit(1, 0, 0.9, 50),  unit(12, 1, 0.8, 50), unit(10, 1, 0.3, 50),
      unit(20, 2, 0.7, 50), unit(21, 2, 0.4, 50), unit(33, 3, 0.6, 50), unit(30, 3, 0.6, 50),
      unit(41, 4, 0.5, 50), unit(40, 4, 0.1, 50),  // 40 is below theta
  };
  Options o = wide_open();
  Selection s = select(candidates, o);
  std::vector<UnitId> expected = {1, 3, 20, 21, 41, 30, 33, 10, 12};
  CHECK(ids_of(s) == expected);
  REQUIRE(s.groups.size() == 5u);
  const std::uint32_t edge[] = {0, 2, 4, 3, 1};
  for (std::size_t i = 0; i < 5; ++i) CHECK_EQ(s.groups[i].group, edge[i]);
  check_structure(s, candidates);

  o.edge_ordering = false;
  s = select(candidates, o);
  expected = {1, 3, 10, 12, 20, 21, 30, 33, 41};
  CHECK(ids_of(s) == expected);
  for (std::size_t i = 0; i < 5; ++i) CHECK_EQ(s.groups[i].group, static_cast<std::uint32_t>(i));

  // Even and small numbers of files; files of equal strength go by their index.
  const auto order_of = [](std::size_t files) {
    std::vector<Candidate> list;
    for (std::size_t g = 0; g < files; ++g) {
      list.push_back(unit(static_cast<UnitId>(g), static_cast<std::uint32_t>(g),
                          0.9 - 0.05 * static_cast<double>(g / 2), 50));  // pairs of equal strength
    }
    std::vector<std::uint32_t> groups;
    const Selection selection = select(list, wide_open());
    for (const NodeQuality& node : selection.groups) groups.push_back(node.group);
    return groups;
  };
  CHECK(order_of(1) == (std::vector<std::uint32_t>{0}));
  CHECK(order_of(2) == (std::vector<std::uint32_t>{0, 1}));
  CHECK(order_of(3) == (std::vector<std::uint32_t>{0, 2, 1}));
  CHECK(order_of(4) == (std::vector<std::uint32_t>{0, 2, 3, 1}));
  CHECK(order_of(6) == (std::vector<std::uint32_t>{0, 2, 4, 5, 3, 1}));
}

// ---- 8. determinism --------------------------------------------------------

PRAG_TEST(any_permutation_of_the_candidates_gives_the_same_selection) {
  Rng rng(555);
  for (int round = 0; round < 150; ++round) {
    const std::size_t n = 2 + rng.below(40);
    Pool pool = random_pool(rng, n, round % 3 == 0 ? 0 : 9, true);
    Options o;
    o.dim = pool.dim;
    o.token_budget = static_cast<std::uint32_t>(100 + rng.below(4000));
    const Selection reference = select(pool.candidates, o);
    for (int shuffle = 0; shuffle < 4; ++shuffle) {
      for (std::size_t i = pool.candidates.size() - 1; i > 0; --i) {
        std::swap(pool.candidates[i], pool.candidates[rng.below(i + 1)]);
      }
      CHECK(same(select(pool.candidates, o), reference));
    }
    std::reverse(pool.candidates.begin(), pool.candidates.end());
    CHECK(same(select(pool.candidates, o), reference));
    CHECK(same(select(pool.candidates, o), reference));  // and the same call twice
  }
}

// ---- 9. evaluate -----------------------------------------------------------

PRAG_TEST(evaluate_reproduces_hand_computed_numbers_for_a_top_k_list) {
  const Example x;
  std::vector<Candidate> ranked = x.candidates(true);
  std::sort(ranked.begin(), ranked.end(), ranks_before);  // 10 20 11 21 12: the files alternate
  Options o;
  o.dim = kSignDim;
  o.token_budget = 1;  // evaluate neither prunes nor budgets

  // Five runs of one unit each: L = 5 * 12 + 100 + 80 + 50 + 120 + 200 = 610.
  //   rho = 0, 0, 1/2, 3/4, 1/2      M = 90 + 64 + 12.5 + 12 + 30 = 208.5
  const Selection s = evaluate(ranked, o);
  const std::vector<UnitId> expected = {10, 20, 11, 21, 12};
  CHECK(ids_of(s) == expected);
  REQUIRE(s.groups.size() == 5u);
  const std::uint32_t tokens[] = {112, 92, 62, 132, 212};
  const double mass[] = {90.0, 64.0, 12.5, 12.0, 30.0};
  const double rho[] = {0.0, 0.0, 0.5, 0.75, 0.5};
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK_EQ(s.groups[i].tokens, tokens[i]);
    CHECK_EQ(s.groups[i].units, 1u);
    CHECK_NEAR(s.groups[i].mass, mass[i], 1e-12);
    CHECK_NEAR(s.groups[i].quality, mass[i] / static_cast<double>(tokens[i]), 1e-12);
    CHECK_EQ(s.entries[i].redundancy, rho[i]);
    CHECK(s.entries[i].form == Form::Full);
    CHECK(!s.entries[i].support);
  }
  CHECK_EQ(s.tokens_after, 610u);
  CHECK_EQ(s.tokens_before, 610u);
  CHECK_NEAR(s.quality_after, 208.5 / 610.0, 1e-12);
  CHECK_EQ(s.quality_before, s.quality_after);
  CHECK_NEAR(s.rot_after, 1.0 - 208.5 / 610.0, 1e-12);
  CHECK_EQ(s.rot_before, s.rot_after);
  CHECK_EQ(s.compression, 1.0);
  CHECK_EQ(s.passes, std::size_t{0});
  CHECK_EQ(s.dropped_low_quality + s.dropped_redundant + s.dropped_budget, std::size_t{0});

  // Top 3: L = 3 * 12 + 230 = 266, M = 90 + 64 + 12.5 = 166.5.
  const Selection top3 = evaluate(std::span<const Candidate>(ranked.data(), 3), o);
  CHECK_EQ(top3.tokens_after, 266u);
  CHECK_NEAR(top3.quality_after, 166.5 / 266.0, 1e-12);

  // Without embeddings: M = 90 + 64 + 25 + 48 + 60 = 287.
  o.dim = 0;
  CHECK_NEAR(evaluate(ranked, o).quality_after, 287.0 / 610.0, 1e-12);

  // A unit listed twice is a duplicate of itself and adds length only.
  const std::vector<Candidate> twice = {ranked[0], ranked[0]};
  o.dim = kSignDim;
  const Selection t = evaluate(twice, o);
  CHECK_EQ(t.tokens_after, 212u);
  CHECK_EQ(t.entries[1].redundancy, 1.0);
  CHECK_NEAR(t.quality_after, 90.0 / 212.0, 1e-12);
}

PRAG_TEST(evaluate_of_the_selected_units_reproduces_the_selection) {
  Rng rng(808);
  for (int round = 0; round < 100; ++round) {
    const Pool pool = random_pool(rng, 1 + rng.below(30), 8, false);
    Options o = wide_open();
    o.dim = pool.dim;
    const Selection s = select(pool.candidates, o);
    std::vector<Candidate> chosen;
    for (const Entry& e : s.entries) chosen.push_back(*find(pool.candidates, e.id));
    const Selection again = evaluate(chosen, o);
    CHECK_EQ(again.tokens_after, s.tokens_after);
    CHECK_EQ(again.quality_after, s.quality_after);
    REQUIRE(again.entries.size() == s.entries.size());
    for (std::size_t i = 0; i < s.entries.size(); ++i) CHECK_EQ(again.entries[i].quality, s.entries[i].quality);
  }
}

// ---- select against a literal reference --------------------------------------

namespace {

// What select decides, without the numbers that check_arithmetic verifies.
struct Decision {
  struct Item {
    UnitId id = kNoUnit;
    Form form = Form::Full;
    bool support = false;
  };
  std::vector<Item> prompt;  // prompt order
  std::size_t passes = 0;
  std::size_t dropped_low_quality = 0;
  std::size_t dropped_redundant = 0;
  std::size_t dropped_budget = 0;
};

// DESIGN.md section 6.9 and the rules of rot.hpp written down step by step.
// Nothing is carried from one step of a loop to the next: every redundancy and
// every total is recomputed from the current state, with the reference cosine.
// Cubic in the number of candidates. Ids are distinct.
Decision reference_select(const std::vector<Candidate>& input, const Options& o) {
  enum class State { Out, Kept, Support };
  struct Unit {
    const Candidate* c = nullptr;
    double p = 0.0;
    State state = State::Out;
    Form form = Form::Full;
    bool reserved = false;  // Options::full_text_share
    bool over_budget = false;
  };
  Decision out;
  if (input.empty()) return out;

  // Step 1: probability descending, ties by id.
  std::vector<Unit> units;
  for (const Candidate& c : input) {
    Unit u;
    u.c = &c;
    u.p = std::min(1.0, std::max(0.0, c.p));
    units.push_back(u);
  }
  std::sort(units.begin(), units.end(),
            [](const Unit& a, const Unit& b) { return a.p != b.p ? a.p > b.p : a.c->id < b.c->id; });
  const std::size_t n = units.size();
  std::vector<std::vector<double>> cosine(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i != j) cosine[i][j] = reference_similarity(units[i].c->embedding, units[j].c->embedding, o.dim);
    }
  }
  // Redundancy of unit j against the selected units that rank before it.
  const auto rho = [&](std::size_t j) {
    double r = 0.0;
    for (std::size_t i = 0; i < j; ++i) {
      if (units[i].state == State::Kept) r = std::max(r, cosine[i][j]);
    }
    return r;
  };
  const auto length = [&](const Unit& u) {
    return static_cast<std::uint64_t>(u.form == Form::Skeleton ? u.c->tokens_skeleton : u.c->tokens_full) +
           o.unit_overhead;
  };
  const auto total = [&] {
    std::uint64_t sum = 0;
    std::set<std::uint32_t> files;
    for (const Unit& u : units) {
      if (u.state == State::Out) continue;
      sum += length(u);
      files.insert(u.c->group);
    }
    return sum + static_cast<std::uint64_t>(o.group_overhead) * files.size();
  };

  // Step 2: every pass judges every candidate against the units kept by the
  // pass before it. A unit below theta fails whatever its redundancy, so the
  // first pass starts from the units that reach theta, and the top-ranked one.
  for (std::size_t j = 0; j < n; ++j) {
    if (j == 0 || units[j].p >= o.theta) units[j].state = State::Kept;
  }
  for (;;) {
    ++out.passes;
    std::vector<State> next(n, State::Out);
    bool stable = true;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == 0 || (1.0 - rho(j)) * units[j].p >= o.theta) next[j] = State::Kept;
      if (next[j] != units[j].state) stable = false;
    }
    for (std::size_t j = 0; j < n; ++j) units[j].state = next[j];
    if (stable) break;
    REQUIRE(out.passes <= n);
  }

  // Step 3: direct requirements of the selected units, as skeletons.
  if (o.dependency_closure) {
    for (std::size_t j = 0; j < n; ++j) {
      if (units[j].state != State::Kept) continue;
      for (const UnitId required : units[j].c->requires_units) {
        for (Unit& r : units) {
          if (r.c->id != required || r.state != State::Out) continue;
          if (!(r.p >= o.closure_min_p) || r.c->tokens_skeleton == 0) continue;
          r.state = State::Support;
          r.form = Form::Skeleton;
        }
      }
    }
  }

  // Step 4. The weakest unit is the one with the smallest key; among equal
  // keys the one that ranks last.
  if (o.full_text_share > 0.0 && total() > o.token_budget) {
    std::vector<std::pair<double, std::size_t>> strongest;
    for (std::size_t j = 0; j < n; ++j) {
      if (units[j].state == State::Kept) strongest.emplace_back(-(1.0 - rho(j)) * units[j].p, j);
    }
    std::sort(strongest.begin(), strongest.end());  // quality descending, then rank
    const auto reserve = static_cast<std::uint64_t>(std::min(1.0, o.full_text_share) *
                                                    static_cast<double>(o.token_budget));
    std::set<std::uint32_t> files;
    std::uint64_t used = 0;
    for (const auto& entry : strongest) {
      Unit& u = units[entry.second];
      const std::uint64_t in_full = static_cast<std::uint64_t>(u.c->tokens_full) + o.unit_overhead +
                                    (files.count(u.c->group) != 0 ? 0u : o.group_overhead);
      if (used + in_full > reserve) break;
      used += in_full;
      files.insert(u.c->group);
      u.reserved = true;
    }
  }
  while (total() > o.token_budget) {
    std::size_t pick = n;
    double least = 0.0;
    const auto consider = [&](std::size_t j, double key) {
      if (pick == n || key <= least) {
        pick = j;
        least = key;
      }
    };
    for (std::size_t j = 0; j < n; ++j) {
      const Unit& u = units[j];
      const bool shrinks = u.c->tokens_skeleton != 0 && u.c->tokens_skeleton < u.c->tokens_full;
      if (u.state == State::Kept && !u.reserved && u.form == Form::Full && shrinks) {
        consider(j, (1.0 - rho(j)) * u.p);
      }
    }
    if (pick != n) {
      units[pick].form = Form::Skeleton;
      continue;
    }
    for (std::size_t j = 0; j < n; ++j) {
      if (units[j].state == State::Support) consider(j, units[j].p * static_cast<double>(length(units[j])));
    }
    if (pick == n) {
      for (std::size_t j = 1; j < n; ++j) {  // never the top-ranked unit
        if (units[j].state != State::Kept || units[j].reserved) continue;
        consider(j, (1.0 - rho(j)) * units[j].p * static_cast<double>(length(units[j])));
      }
    }
    if (pick == n) break;
    units[pick].state = State::Out;
    units[pick].over_budget = true;
  }

  // Step 5: files by their strongest entry, first, last, second, second to last, ...
  std::map<std::uint32_t, double> strength;
  for (const Unit& u : units) {
    if (u.state == State::Out) {
      if (u.over_budget) {
        ++out.dropped_budget;
      } else if (u.p >= o.theta) {
        ++out.dropped_redundant;
      } else {
        ++out.dropped_low_quality;
      }
      continue;
    }
    const auto it = strength.find(u.c->group);
    if (it == strength.end()) {
      strength.emplace(u.c->group, u.p);
    } else {
      it->second = std::max(it->second, u.p);
    }
  }
  std::vector<std::pair<double, std::uint32_t>> ranked;
  for (const auto& [group, p] : strength) ranked.emplace_back(-p, group);
  std::sort(ranked.begin(), ranked.end());
  std::vector<std::uint32_t> files(ranked.size());
  std::size_t front = 0;
  std::size_t back = ranked.size();
  for (std::size_t r = 0; r < ranked.size(); ++r) {
    if (!o.edge_ordering) {
      files[r] = ranked[r].second;
    } else if (r % 2 == 0) {
      files[front++] = ranked[r].second;
    } else {
      files[--back] = ranked[r].second;
    }
  }
  for (const std::uint32_t file : files) {
    std::vector<Decision::Item> members;
    for (const Unit& u : units) {
      if (u.state == State::Out || u.c->group != file) continue;
      members.push_back(Decision::Item{u.c->id, u.form, u.state == State::Support});
    }
    std::sort(members.begin(), members.end(),
              [](const Decision::Item& a, const Decision::Item& b) { return a.id < b.id; });
    out.prompt.insert(out.prompt.end(), members.begin(), members.end());
  }
  return out;
}

bool agrees(const Selection& s, const Decision& d) {
  if (s.entries.size() != d.prompt.size()) return false;
  for (std::size_t k = 0; k < d.prompt.size(); ++k) {
    const Entry& e = s.entries[k];
    const Decision::Item& x = d.prompt[k];
    if (e.id != x.id || e.form != x.form || e.support != x.support) return false;
  }
  return s.passes == d.passes && s.dropped_low_quality == d.dropped_low_quality &&
         s.dropped_redundant == d.dropped_redundant && s.dropped_budget == d.dropped_budget;
}

}  // namespace

PRAG_TEST(select_agrees_with_a_literal_reference_on_random_candidate_sets) {
  Rng rng(20260929);
  const double thetas[] = {0.1, 0.25, 0.4};
  const double minimum[] = {0.0, 0.05, 0.2};
  const double shares[] = {0.0, 0.0, 0.3, 1.0};
  std::size_t over_budget = 0;
  std::size_t dropped = 0;
  std::size_t with_support = 0;
  std::size_t with_redundant = 0;
  std::size_t several_passes = 0;
  std::size_t reduced = 0;
  for (int round = 0; round < 1500; ++round) {
    const std::size_t n = 1 + rng.below(36);
    const Pool pool = random_pool(rng, n, round % 4 == 0 ? 0 : 10, true);
    Options o;
    o.theta = thetas[rng.below(3)];
    o.token_budget = static_cast<std::uint32_t>(20 + rng.below(round % 5 == 0 ? 20000 : 2500));
    o.group_overhead = static_cast<std::uint32_t>(rng.below(3) * 6);
    o.unit_overhead = static_cast<std::uint32_t>(rng.below(3) * 4);
    o.dependency_closure = rng.below(4) != 0;
    o.closure_min_p = minimum[rng.below(3)];
    o.edge_ordering = rng.below(3) != 0;
    o.full_text_share = shares[rng.below(4)];
    o.max_passes = 1000;
    o.dim = pool.dim;

    const Selection s = select(pool.candidates, o);
    const Decision d = reference_select(pool.candidates, o);
    const bool equal = agrees(s, d);
    CHECK(equal);
    if (!equal) {
      std::printf("  round %d: n = %zu, theta = %g, budget = %u, share = %g: entries %zu vs %zu, passes %zu vs %zu, "
                  "dropped %zu/%zu/%zu vs %zu/%zu/%zu\n",
                  round, n, o.theta, o.token_budget, o.full_text_share, s.entries.size(), d.prompt.size(), s.passes,
                  d.passes, s.dropped_low_quality, s.dropped_redundant, s.dropped_budget, d.dropped_low_quality,
                  d.dropped_redundant, d.dropped_budget);
    }
    check_arithmetic(s, pool.candidates, o);
    check_structure(s, pool.candidates);
    check_before(s, pool.candidates, o);

    // The limit on the passes changes the count and nothing else.
    Options limited = o;
    limited.max_passes = 1 + rng.below(4);
    const Selection t = select(pool.candidates, limited);
    CHECK(ids_of(t) == ids_of(s));
    CHECK_EQ(t.quality_after, s.quality_after);
    CHECK_EQ(t.tokens_after, s.tokens_after);
    CHECK_EQ(t.dropped_redundant, s.dropped_redundant);
    CHECK_EQ(t.dropped_budget, s.dropped_budget);

    Options unlimited = o;
    unlimited.token_budget = 4000000000u;
    if (select(pool.candidates, unlimited).tokens_after > o.token_budget) ++over_budget;
    if (s.dropped_budget != 0) ++dropped;
    if (s.support_added != 0) ++with_support;
    if (s.dropped_redundant != 0) ++with_redundant;
    if (s.skeletonised != 0) ++reduced;
    if (s.passes > 2) ++several_passes;
  }
  // The sample reaches every step of the procedure.
  CHECK(over_budget > 300u);
  CHECK(dropped > 200u);
  CHECK(reduced > 200u);
  CHECK(with_support > 200u);
  CHECK(with_redundant > 300u);
  CHECK(several_passes > 50u);
}

// ---- edge cases ------------------------------------------------------------

PRAG_TEST(empty_and_degenerate_inputs) {
  const Selection none = select(std::span<const Candidate>{}, Options{});
  CHECK(none.entries.empty());
  CHECK(none.groups.empty());
  CHECK_EQ(none.tokens_before, 0u);
  CHECK_EQ(none.tokens_after, 0u);
  CHECK_EQ(none.quality_after, 0.0);
  CHECK_EQ(none.rot_after, 0.0);
  CHECK_EQ(none.compression, 1.0);
  CHECK_EQ(none.passes, std::size_t{0});
  CHECK(evaluate(std::span<const Candidate>{}, Options{}).entries.empty());

  // Nothing reaches theta: the most probable unit is kept, alone.
  const std::vector<Candidate> weak = {unit(1, 0, 0.10, 100, 10), unit(2, 0, 0.20, 100, 10),
                                       unit(3, 1, 0.20, 100, 10)};
  const Selection s = select(weak, wide_open());
  REQUIRE(s.entries.size() == 1u);
  CHECK_EQ(s.entries[0].id, UnitId{2});  // 0.20 twice: the lower id
  CHECK_EQ(s.entries[0].quality, 0.2);
  CHECK_EQ(s.dropped_low_quality, std::size_t{2});
  CHECK_EQ(s.tokens_after, 120u);
  CHECK_NEAR(s.quality_after, 0.2 * 108.0 / 120.0, 1e-12);

  // Probabilities outside [0, 1] and NaN are clamped.
  const std::vector<Candidate> odd = {unit(1, 0, 1.7, 100), unit(2, 0, -0.5, 100),
                                      unit(3, 0, std::nan(""), 100)};
  const Selection t = select(odd, wide_open());
  REQUIRE(t.entries.size() == 1u);
  CHECK_EQ(t.entries[0].p, 1.0);
  CHECK(t.quality_before >= 0.0 && t.quality_before <= 1.0);

  // A zero vector and a vector with a NaN have no direction: no redundancy.
  const std::array<float, 4> zero = {0.0f, 0.0f, 0.0f, 0.0f};
  const std::array<float, 4> bad = {1.0f, std::nanf(""), 0.0f, 0.0f};
  const std::array<float, 4> good = {1.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<Candidate> vectors = {unit(1, 0, 0.9, 100, 0, good.data()), unit(2, 0, 0.8, 100, 0, zero.data()),
                                          unit(3, 0, 0.7, 100, 0, bad.data())};
  Options o = wide_open();
  o.dim = 4;
  const Selection v = select(vectors, o);
  REQUIRE(v.entries.size() == 3u);
  for (const Entry& e : v.entries) CHECK_EQ(e.redundancy, 0.0);

  // theta = 0 keeps everything, copies included; theta above 1 keeps the top unit only.
  const Example x;
  const std::vector<Candidate> five = x.candidates(true);
  o = wide_open();
  o.dim = kSignDim;
  o.theta = 0.0;
  CHECK_EQ(select(five, o).entries.size(), std::size_t{5});
  o.theta = 1.5;
  CHECK_EQ(select(five, o).entries.size(), std::size_t{1});

  CHECK_EQ(std::string(to_string(Form::Full)), std::string("full"));
  CHECK_EQ(std::string(to_string(Form::Skeleton)), std::string("skeleton"));
}

// ---- measurements (PRAG_SLOW_TESTS=1) ---------------------------------------

namespace {

// n units in n / 8 groups of related functions (cosine about 0.6 inside a
// group, about 0 between groups), spread over n / 16 files.
Pool measurement_pool(std::size_t n, std::size_t dim) {
  Rng rng(1000 + n);
  Pool pool;
  pool.dim = dim;
  const std::size_t clusters = std::max<std::size_t>(1, n / 8);
  const std::size_t files = std::max<std::size_t>(1, n / 16);
  std::vector<std::vector<float>> centres(clusters, std::vector<float>(dim));
  for (auto& centre : centres) {
    for (float& x : centre) x = static_cast<float>(rng.normal());
  }
  pool.embeddings.assign(n, std::vector<float>(dim));
  for (std::size_t i = 0; i < n; ++i) {
    const std::vector<float>& centre = centres[rng.below(clusters)];
    double norm = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
      const double x = static_cast<double>(centre[d]) + 0.8 * rng.normal();
      pool.embeddings[i][d] = static_cast<float>(x);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (float& x : pool.embeddings[i]) x = static_cast<float>(static_cast<double>(x) / norm);
  }
  for (std::size_t i = 0; i < n; ++i) {
    Candidate c = unit(static_cast<UnitId>(i), static_cast<std::uint32_t>(rng.below(files)),
                       0.25 + 0.75 * rng.uniform(), static_cast<std::uint32_t>(40 + rng.below(360)),
                       static_cast<std::uint32_t>(10 + rng.below(40)), pool.embeddings[i].data());
    for (std::size_t k = rng.below(4); k > 0; --k) c.requires_units.push_back(static_cast<UnitId>(rng.below(n)));
    pool.candidates.push_back(std::move(c));
  }
  return pool;
}

double median_ms(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

PRAG_TEST(measure_select_time) {
  const char* slow = std::getenv("PRAG_SLOW_TESTS");
  if (!slow || std::string(slow) != "1") {
    std::printf("  skipped (set PRAG_SLOW_TESTS=1 to measure)\n");
    return;
  }
  constexpr std::size_t dim = 1024;
  for (const std::size_t n : {std::size_t{64}, std::size_t{256}, std::size_t{1024}}) {
    const Pool pool = measurement_pool(n, dim);
    Options o;  // theta 0.25, budget 3000, closure on
    o.dim = dim;
    Options unlimited = o;
    unlimited.token_budget = 4000000000u;
    Options no_embeddings = o;
    no_embeddings.dim = 0;
    Options shared = o;
    shared.full_text_share = 0.5;
    const Selection h = select(pool.candidates, shared);
    const int repeats = n >= 1024 ? 9 : 31;
    std::vector<double> with;
    std::vector<double> unbudgeted;
    std::vector<double> without;
    std::vector<double> baseline;
    Selection s;
    Selection u;
    for (int r = 0; r < repeats; ++r) {
      Stopwatch watch;
      s = select(pool.candidates, o);
      with.push_back(watch.elapsed_ms());
      watch.reset();
      u = select(pool.candidates, unlimited);
      unbudgeted.push_back(watch.elapsed_ms());
      watch.reset();
      const Selection plain = select(pool.candidates, no_embeddings);
      without.push_back(watch.elapsed_ms());
      watch.reset();
      const Selection e = evaluate(pool.candidates, o);
      baseline.push_back(watch.elapsed_ms());
      CHECK(!plain.entries.empty());
      CHECK(!e.entries.empty());
    }
    std::sort(with.begin(), with.end());
    std::printf("  n = %4zu, dim = %zu: select %.3f ms (min %.3f, max %.3f); without budget %.3f ms; "
                "without embeddings %.3f ms; evaluate %.3f ms\n",
                n, dim, median_ms(with), with.front(), with.back(), median_ms(unbudgeted), median_ms(without),
                median_ms(baseline));
    std::printf("      passes %zu; redundant %zu, low %zu, budget %zu; entries %zu (skeleton %zu, support %zu); "
                "tokens %u -> %u; Q %.3f -> %.3f (without budget: %zu entries, %u tokens, Q %.3f)\n",
                s.passes, s.dropped_redundant, s.dropped_low_quality, s.dropped_budget, s.entries.size(),
                s.skeletonised, s.support_added, s.tokens_before, s.tokens_after, s.quality_before, s.quality_after,
                u.entries.size(), u.tokens_after, u.quality_after);
    std::size_t full = 0;
    std::size_t full_shared = 0;
    for (const Entry& e : s.entries) full += e.form == Form::Full ? 1 : 0;
    for (const Entry& e : h.entries) full_shared += e.form == Form::Full ? 1 : 0;
    std::printf("      entries in full: %zu of %zu; with full_text_share 0.5: %zu of %zu, tokens %u, Q %.3f\n", full,
                s.entries.size(), full_shared, h.entries.size(), h.tokens_after, h.quality_after);
    check_structure(s, pool.candidates);
    check_arithmetic(s, pool.candidates, o);
    check_before(s, pool.candidates, o);
    check_arithmetic(h, pool.candidates, shared);
    CHECK(s.tokens_after <= o.token_budget);
    CHECK(h.tokens_after <= o.token_budget);
  }
}
