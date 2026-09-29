#include "prag/context/rot.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace prag::context {

const char* to_string(Form form) {
  switch (form) {
    case Form::Full: return "full";
    case Form::Skeleton: return "skeleton";
  }
  return "full";
}

namespace {

// Working set of the blocked similarity computation (DESIGN.md section 6.2).
// This module may depend on core only, so it cannot query the cache topology;
// the block size of the reference machine is used.
constexpr std::size_t kBlockBytes = 16 * 1024;

// NaN maps to 0.
inline double clamp01(double x) noexcept { return x >= 0.0 ? (x <= 1.0 ? x : 1.0) : 0.0; }

inline std::uint32_t saturate(std::uint64_t x) noexcept {
  const std::uint64_t limit = std::numeric_limits<std::uint32_t>::max();
  return static_cast<std::uint32_t>(std::min(x, limit));
}

// Inner product accumulated in double precision. The product of two float32
// values is exact in double, so fused and separate multiply-add give the same
// bits, and the scalar path reproduces the vector path exactly: both keep 16
// partial sums (element e goes to sum e mod 16) and combine them in the same order.
double dot(const float* a, const float* b, std::size_t n) noexcept {
  const std::size_t blocked = n - n % 16;
  double lane[4];
#if defined(__AVX2__)
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  __m256d s2 = _mm256_setzero_pd();
  __m256d s3 = _mm256_setzero_pd();
  for (std::size_t i = 0; i < blocked; i += 16) {
    s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i)), _mm256_cvtps_pd(_mm_loadu_ps(b + i))));
    s1 = _mm256_add_pd(
        s1, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 4)), _mm256_cvtps_pd(_mm_loadu_ps(b + i + 4))));
    s2 = _mm256_add_pd(
        s2, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 8)), _mm256_cvtps_pd(_mm_loadu_ps(b + i + 8))));
    s3 = _mm256_add_pd(
        s3, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 12)), _mm256_cvtps_pd(_mm_loadu_ps(b + i + 12))));
  }
  _mm256_storeu_pd(lane, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
#else
  double s[16] = {};
  for (std::size_t i = 0; i < blocked; i += 16) {
    for (std::size_t k = 0; k < 16; ++k) s[k] += static_cast<double>(a[i + k]) * static_cast<double>(b[i + k]);
  }
  for (std::size_t k = 0; k < 4; ++k) lane[k] = (s[k] + s[k + 4]) + (s[k + 8] + s[k + 12]);
#endif
  double tail = 0.0;
  for (std::size_t i = blocked; i < n; ++i) tail += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + tail;
}

// Clipped cosines max(0, cos(e_i, e_j)) of every pair of rows, stored as the
// strict lower triangle so that the scan "row j against all rows before it" is
// contiguous. A pair with a missing row has similarity 0.
class Similarity {
 public:
  Similarity(const std::vector<const float*>& rows, std::size_t dim) : n_(rows.size()) {
    if (dim == 0 || n_ < 2) return;
    if (std::none_of(rows.begin(), rows.end(), [](const float* r) { return r != nullptr; })) return;

    // Embeddings are L2-normalised by contract; dividing by the norms anyway
    // costs n inner products and makes exact duplicates come out at exactly 1.
    std::vector<double> squared(n_, 0.0);
    for (std::size_t i = 0; i < n_; ++i) {
      if (rows[i]) squared[i] = dot(rows[i], rows[i], dim);
    }
    lower_.assign(n_ * (n_ - 1) / 2, 0.0);
    const auto fill = [&](std::size_t i, std::size_t j) {  // i < j
      if (!rows[i] || !rows[j]) return;
      const double scale = std::sqrt(squared[i] * squared[j]);
      if (!(scale > 0.0) || !std::isfinite(scale)) return;
      lower_[j * (j - 1) / 2 + i] = clamp01(dot(rows[i], rows[j], dim) / scale);
    };

    // Blocked form: a block of rows stays hot while every later row is
    // streamed past it once, so a row is loaded once per block rather than
    // once per pair. The block and the streamed row together fill kBlockBytes.
    const std::size_t rows_per_block = kBlockBytes / (dim * sizeof(float));
    const std::size_t block = rows_per_block > 1 ? rows_per_block - 1 : 1;
    for (std::size_t first = 0; first < n_; first += block) {
      const std::size_t last = std::min(n_, first + block);
      for (std::size_t i = first; i < last; ++i) {
        for (std::size_t j = i + 1; j < last; ++j) fill(i, j);
      }
      for (std::size_t q = last; q < n_; ++q) {
        for (std::size_t i = first; i < last; ++i) fill(i, q);
      }
    }
  }

  bool enabled() const noexcept { return !lower_.empty(); }

  double operator()(std::size_t i, std::size_t j) const noexcept {
    if (lower_.empty() || i == j) return 0.0;
    return i < j ? lower_[j * (j - 1) / 2 + i] : lower_[i * (i - 1) / 2 + j];
  }

  // Similarities of row j to rows 0 .. j-1, contiguous; nullptr when redundancy is disabled.
  const double* before(std::size_t j) const noexcept {
    return lower_.empty() || j == 0 ? nullptr : lower_.data() + j * (j - 1) / 2;
  }

 private:
  std::size_t n_ = 0;
  std::vector<double> lower_;
};

struct Item {
  const Candidate* candidate = nullptr;
  double p = 0.0;  // clamped to [0, 1]
};

// One unit of a prompt, referring to an Item by position.
struct Placed {
  std::size_t item = 0;
  Form form = Form::Full;
  bool support = false;
};

inline std::uint64_t tokens_of(const Candidate& c, Form form, const Options& options) noexcept {
  const std::uint64_t body = form == Form::Skeleton ? c.tokens_skeleton : c.tokens_full;
  return body + options.unit_overhead;
}

// Reducing a unit to its skeleton must make it shorter to be worth doing.
inline bool can_shrink(const Candidate& c) noexcept {
  return c.tokens_skeleton != 0 && c.tokens_skeleton < c.tokens_full;
}

// Step 1. Probability descending, ties by id. A unit id that occurs more than
// once is a caller error; the most probable copy is used, and the remaining
// keys make the choice independent of the input order.
std::vector<Item> rank(std::span<const Candidate> candidates) {
  std::vector<Item> items;
  items.reserve(candidates.size());
  for (const Candidate& c : candidates) items.push_back(Item{&c, clamp01(c.p)});
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    if (a.p != b.p) return a.p > b.p;
    const Candidate& x = *a.candidate;
    const Candidate& y = *b.candidate;
    if (x.id != y.id) return x.id < y.id;
    if (x.group != y.group) return x.group < y.group;
    if (x.tokens_full != y.tokens_full) return x.tokens_full < y.tokens_full;
    return x.tokens_skeleton < y.tokens_skeleton;
  });
  std::unordered_map<UnitId, char> seen;
  seen.reserve(items.size());
  std::size_t out = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (!seen.emplace(items[i].candidate->id, char{1}).second) continue;
    items[out++] = items[i];
  }
  items.resize(out);
  return items;
}

std::vector<const float*> rows_of(const std::vector<Item>& items) {
  std::vector<const float*> rows;
  rows.reserve(items.size());
  for (const Item& item : items) rows.push_back(item.candidate->embedding);
  return rows;
}

// Step 5. Files by maximum probability, strongest first, second strongest
// last, third strongest second, ...; units of a file by ascending id.
void arrange(std::vector<Placed>& placed, const std::vector<Item>& items, bool edge_ordering) {
  struct File {
    std::uint32_t group = 0;
    double strength = 0.0;
  };
  std::vector<File> files;
  std::unordered_map<std::uint32_t, std::size_t> index;
  for (const Placed& entry : placed) {
    const Item& item = items[entry.item];
    const auto [it, inserted] = index.try_emplace(item.candidate->group, files.size());
    if (inserted) {
      files.push_back(File{item.candidate->group, item.p});
    } else {
      files[it->second].strength = std::max(files[it->second].strength, item.p);
    }
  }
  std::sort(files.begin(), files.end(), [](const File& a, const File& b) {
    if (a.strength != b.strength) return a.strength > b.strength;
    return a.group < b.group;
  });
  const std::size_t count = files.size();
  std::unordered_map<std::uint32_t, std::size_t> slot;
  slot.reserve(count);
  for (std::size_t r = 0; r < count; ++r) {
    const std::size_t position = !edge_ordering ? r : (r % 2 == 0 ? r / 2 : count - 1 - r / 2);
    slot.emplace(files[r].group, position);
  }
  std::sort(placed.begin(), placed.end(), [&](const Placed& a, const Placed& b) {
    const Candidate& x = *items[a.item].candidate;
    const Candidate& y = *items[b.item].candidate;
    if (x.group != y.group) return slot.find(x.group)->second < slot.find(y.group)->second;
    if (x.id != y.id) return x.id < y.id;
    return a.item < b.item;
  });
}

struct Scored {
  std::vector<Entry> entries;
  std::vector<NodeQuality> groups;
  std::uint64_t tokens = 0;
  double mass = 0.0;
  double quality = 0.0;
};

// Step 6. L, M and Q of a prompt, bottom-up. A file node is a maximal run of
// consecutive units of one file. Support units are outside the redundancy
// computation in both roles.
Scored score(const std::vector<Placed>& placed, const std::vector<Item>& items, const Similarity& similarity,
             const Options& options) {
  Scored out;
  out.entries.reserve(placed.size());
  std::vector<std::uint64_t> group_tokens;
  for (std::size_t k = 0; k < placed.size(); ++k) {
    const Placed& current = placed[k];
    const Item& item = items[current.item];
    const Candidate& c = *item.candidate;

    double rho = 0.0;
    if (!current.support && similarity.enabled()) {
      for (std::size_t i = 0; i < k; ++i) {
        if (!placed[i].support) rho = std::max(rho, similarity(placed[i].item, current.item));
      }
    }
    const std::uint64_t length = tokens_of(c, current.form, options);

    Entry entry;
    entry.id = c.id;
    entry.group = c.group;
    entry.form = current.form;
    entry.support = current.support;
    entry.p = item.p;
    entry.redundancy = rho;
    entry.quality = (1.0 - rho) * item.p;
    entry.tokens = saturate(length);
    out.entries.push_back(entry);

    if (out.groups.empty() || out.groups.back().group != c.group) {
      NodeQuality node;
      node.group = c.group;
      out.groups.push_back(node);
      group_tokens.push_back(options.group_overhead);
    }
    group_tokens.back() += length;
    out.groups.back().mass += entry.quality * static_cast<double>(length);
    ++out.groups.back().units;
  }
  for (std::size_t g = 0; g < out.groups.size(); ++g) {
    NodeQuality& node = out.groups[g];
    node.tokens = saturate(group_tokens[g]);
    node.quality = group_tokens[g] != 0 ? node.mass / static_cast<double>(group_tokens[g]) : 0.0;
    out.tokens += group_tokens[g];
    out.mass += node.mass;
  }
  out.quality = out.tokens != 0 ? out.mass / static_cast<double>(out.tokens) : 0.0;
  return out;
}

// Largest similarity of row j to the rows before it that are marked in `kept`.
double redundancy(const Similarity& similarity, const std::vector<char>& kept, std::size_t j) noexcept {
  const double* row = similarity.before(j);
  if (!row) return 0.0;
  double rho = 0.0;
  for (std::size_t i = 0; i < j; ++i) {
    if (kept[i] && row[i] > rho) rho = row[i];
  }
  return rho;
}

}  // namespace

Selection select(std::span<const Candidate> candidates, const Options& options) {
  Selection out;
  const std::vector<Item> items = rank(candidates);
  const std::size_t n = items.size();
  if (n == 0) return out;

  const Similarity similarity(rows_of(items), options.dim);
  const double theta = options.theta;

  // ---- step 2: prune and rescore to the fixed point --------------------------
  // Units with p < theta fail for every redundancy, and they form a suffix of
  // the ranking. The top-ranked unit is kept unconditionally.
  std::size_t eligible = 1;
  while (eligible < n && items[eligible].p >= theta) ++eligible;

  std::vector<char> kept(eligible, 1);
  std::vector<char> next(eligible, 1);
  std::vector<double> rho(eligible, 0.0);
  const std::size_t max_passes = std::max<std::size_t>(1, options.max_passes);
  for (;;) {
    ++out.passes;
    // The status of unit j depends only on the units before it. Reading the
    // statuses of the running pass instead of those of the previous one
    // therefore yields the fixed point in a single sweep; the last permitted
    // pass does that, so the result is the fixed point whatever the limit.
    const bool sweep = out.passes >= max_passes;
    const std::vector<char>& reference = sweep ? next : kept;
    for (std::size_t j = 0; j < eligible; ++j) {
      rho[j] = redundancy(similarity, reference, j);
      next[j] = (j == 0 || (1.0 - rho[j]) * items[j].p >= theta) ? 1 : 0;
    }
    const bool stable = sweep || next == kept;
    kept.swap(next);
    if (stable) break;
  }

  enum class State : std::uint8_t { Dropped, Kept, Support };
  std::vector<State> state(n, State::Dropped);
  for (std::size_t j = 0; j < eligible; ++j) {
    if (kept[j]) state[j] = State::Kept;
  }

  // ---- step 3: dependency closure --------------------------------------------
  std::vector<Form> form(n, Form::Full);
  if (options.dependency_closure) {
    std::unordered_map<UnitId, std::size_t> position;
    position.reserve(n);
    for (std::size_t j = 0; j < n; ++j) position.emplace(items[j].candidate->id, j);
    for (std::size_t j = 0; j < eligible; ++j) {
      if (state[j] != State::Kept) continue;
      for (const UnitId required : items[j].candidate->requires_units) {
        const auto it = position.find(required);
        if (it == position.end()) continue;  // not a candidate: nothing is known about it
        const std::size_t k = it->second;
        if (state[k] != State::Dropped) continue;
        if (!(items[k].p >= options.closure_min_p) || items[k].candidate->tokens_skeleton == 0) continue;
        state[k] = State::Support;
        form[k] = Form::Skeleton;
      }
    }
  }

  // ---- step 4: token budget --------------------------------------------------
  std::unordered_map<std::uint32_t, std::size_t> members;  // entries per file
  std::uint64_t total = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (state[j] == State::Dropped) continue;
    total += tokens_of(*items[j].candidate, form[j], options);
    if (++members[items[j].candidate->group] == 1) total += options.group_overhead;
  }
  std::vector<char> over_budget(n, 0);
  const auto remove = [&](std::size_t j) {
    total -= tokens_of(*items[j].candidate, form[j], options);
    if (--members[items[j].candidate->group] == 0) total -= options.group_overhead;
    state[j] = State::Dropped;
    over_budget[j] = 1;
  };
  const auto quality = [&](std::size_t j) { return (1.0 - rho[j]) * items[j].p; };
  // Weakest first; among equals the unit that ranks lower goes first.
  const auto ascending = [](const std::pair<double, std::size_t>& a, const std::pair<double, std::size_t>& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second > b.second;
  };
  const std::uint64_t budget = options.token_budget;

  // Options::full_text_share: the strongest units, as long as their lengths in
  // full add up to no more than that share of the budget, are left as they are.
  std::vector<char> reserved(eligible, 0);
  if (total > budget && options.full_text_share > 0.0) {
    std::vector<std::pair<double, std::size_t>> order;
    for (std::size_t j = 0; j < eligible; ++j) {
      if (state[j] == State::Kept) order.emplace_back(quality(j), j);
    }
    std::sort(order.begin(), order.end(),
              [&](const auto& a, const auto& b) { return ascending(b, a); });  // strongest first
    const double share = std::min(1.0, options.full_text_share);
    const auto limit = static_cast<std::uint64_t>(share * static_cast<double>(budget));
    std::unordered_map<std::uint32_t, char> files;
    std::uint64_t used = 0;
    for (const auto& entry : order) {
      const Candidate& c = *items[entry.second].candidate;
      std::uint64_t length = tokens_of(c, Form::Full, options);
      if (files.find(c.group) == files.end()) length += options.group_overhead;
      if (used + length > limit) break;
      used += length;
      files.emplace(c.group, char{1});
      reserved[entry.second] = 1;
    }
  }

  if (total > budget) {
    std::vector<std::pair<double, std::size_t>> order;
    for (std::size_t j = 0; j < eligible; ++j) {
      if (state[j] == State::Kept && !reserved[j] && can_shrink(*items[j].candidate)) {
        order.emplace_back(quality(j), j);
      }
    }
    std::sort(order.begin(), order.end(), ascending);
    for (const auto& entry : order) {
      if (total <= budget) break;
      const std::size_t j = entry.second;
      total -= items[j].candidate->tokens_full - items[j].candidate->tokens_skeleton;
      form[j] = Form::Skeleton;
    }
  }
  if (total > budget) {
    std::vector<std::pair<double, std::size_t>> order;
    for (std::size_t j = 0; j < n; ++j) {
      if (state[j] != State::Support) continue;
      const double length = static_cast<double>(tokens_of(*items[j].candidate, form[j], options));
      order.emplace_back(items[j].p * length, j);
    }
    std::sort(order.begin(), order.end(), ascending);
    for (const auto& entry : order) {
      if (total <= budget) break;
      remove(entry.second);
    }
  }
  while (total > budget) {
    std::pair<double, std::size_t> weakest{0.0, 0};  // index 0, the top-ranked unit, is never dropped
    for (std::size_t j = 1; j < eligible; ++j) {
      if (state[j] != State::Kept || reserved[j]) continue;
      const double length = static_cast<double>(tokens_of(*items[j].candidate, form[j], options));
      const std::pair<double, std::size_t> current{quality(j) * length, j};
      if (weakest.second == 0 || ascending(current, weakest)) weakest = current;
    }
    const std::size_t dropped = weakest.second;
    if (dropped == 0) break;
    remove(dropped);
    // Units that were discounted because of the dropped one are rescored. No
    // unit returns here: the budget step only removes.
    kept[dropped] = 0;
    for (std::size_t j = dropped + 1; j < eligible; ++j) {
      if (state[j] == State::Kept && rho[j] > 0.0 && similarity(dropped, j) >= rho[j]) {
        rho[j] = redundancy(similarity, kept, j);
      }
    }
  }

  // ---- steps 5 and 6 ---------------------------------------------------------
  std::vector<Placed> selected;
  for (std::size_t j = 0; j < n; ++j) {
    switch (state[j]) {
      case State::Kept:
        selected.push_back(Placed{j, form[j], false});
        if (form[j] == Form::Skeleton) ++out.skeletonised;
        break;
      case State::Support:
        selected.push_back(Placed{j, form[j], true});
        ++out.support_added;
        break;
      case State::Dropped:
        if (over_budget[j]) {
          ++out.dropped_budget;
        } else if (j >= eligible) {
          ++out.dropped_low_quality;
        } else {
          ++out.dropped_redundant;
        }
        break;
    }
  }
  arrange(selected, items, options.edge_ordering);
  Scored after = score(selected, items, similarity, options);

  std::vector<Placed> everything;
  everything.reserve(n);
  for (std::size_t j = 0; j < n; ++j) everything.push_back(Placed{j, Form::Full, false});
  arrange(everything, items, options.edge_ordering);
  const Scored before = score(everything, items, similarity, options);

  out.entries = std::move(after.entries);
  out.groups = std::move(after.groups);
  out.tokens_before = saturate(before.tokens);
  out.tokens_after = saturate(after.tokens);
  out.quality_before = before.quality;
  out.quality_after = after.quality;
  out.rot_before = 1.0 - before.quality;
  out.rot_after = 1.0 - after.quality;
  out.compression =
      before.tokens != 0 ? static_cast<double>(after.tokens) / static_cast<double>(before.tokens) : 1.0;
  return out;
}

Selection evaluate(std::span<const Candidate> ordered, const Options& options) {
  Selection out;
  if (ordered.empty()) return out;

  std::vector<Item> items;
  std::vector<Placed> placed;
  items.reserve(ordered.size());
  placed.reserve(ordered.size());
  for (const Candidate& c : ordered) {
    placed.push_back(Placed{items.size(), Form::Full, false});
    items.push_back(Item{&c, clamp01(c.p)});
  }
  const Similarity similarity(rows_of(items), options.dim);
  Scored scored = score(placed, items, similarity, options);

  out.entries = std::move(scored.entries);
  out.groups = std::move(scored.groups);
  out.tokens_before = out.tokens_after = saturate(scored.tokens);
  out.quality_before = out.quality_after = scored.quality;
  out.rot_before = out.rot_after = 1.0 - scored.quality;
  return out;
}

}  // namespace prag::context
