// The ordered blocked scan, exact rescoring, and the error bounds.
//
// ---- who writes what ------------------------------------------------------------
//
// The encoded queries are prepared before the run and are read-only. `work`
// scores one block against every query of the batch: kernel results go to the
// scratch of the worker, and the score and half-width of a unit go straight
// to their place in the result. Every unit belongs to exactly one block and
// its two values are functions of the block and the query alone, so no two
// workers write the same element and the values do not depend on the
// schedule. The only reduction, the sums behind `mean` and `stddev`, is formed
// per block in `work` and added in `commit`, in storage order.
//
// The scores are not copied in `commit` because `commit` is serial: with the
// copy there, a scan of 100000 vectors spent more time in `commit` than in
// the kernels as soon as several threads or several queries were used (for a
// batch of 64 queries on 8 threads 170 ms against 36 ms, Int8, 16 KiB blocks).
//
// ---- Int8: a true bound ---------------------------------------------------------
//
// Let delta = s / 127, xh = delta * cx the decoded row and qh = delta * cq the
// decoded query, rx = x - xh and rq = q - qh the residuals. A residual
// coordinate is a rounding error (at most delta / 2) when the coordinate lies
// within [-s, s], and the clipping error |x_j| - s (rounded) when it does not;
// the residual is computed from the code that is stored, so it contains both.
// The estimate is <qh, xh> = delta^2 sum_j cq_j cx_j, and
//
//     <q, x> - <qh, xh>  =  <q, x - xh> + <q - qh, xh>  =  <q, rx> + <rq, xh>,
//
// so by the Cauchy-Schwarz inequality
//
//     | <q, x> - <qh, xh> |  <=  ||q|| ||rx||  +  ||rq|| ||xh||.
//
// ||rx|| and ||xh|| are the two side values of the row (rounded upwards),
// ||q|| and ||rq|| are computed when the query is encoded. Nothing is assumed
// about the norm of q or of x. The bound is attained only if each residual is
// parallel to the other vector; rounding errors behave like independent noise,
// so the typical error is smaller by a factor of the order of sqrt(D).
//
// ---- RaBitQ: a probabilistic bound ----------------------------------------------
//
// With c the centroid, d = ||x - c||, dq = ||q - c||, o and q' the normalised
// centred row and query,
//
//     <q, x>  =  d dq <o, q'>  +  <q, c>  +  <x, c>  -  ||c||^2
//             =  d dq <o, q'>  +  <q, c>  +  (||x||^2 - ||c||^2 - d^2) / 2,
//
// the last step because d^2 = ||x||^2 - 2 <x, c> + ||c||^2. Rows are unit
// vectors, so ||x||^2 = 1 and the only unknown is <o, q'>; the largest
// | ||x||^2 - 1 | / 2 over the rows (norm_slack) is added to the half-width.
//
// <o, q'> is estimated by <xbar, q~> / a, a = <obar, o>, xbar = sign(P^T o) /
// sqrt(D), and q~ the rotated query P^T q' quantised to 4 bits per coordinate:
// q~_j = low + step * level_j, level_j in 0..15, rounded up or down at random
// with the probabilities that make q~_j unbiased. Then
//
//     sqrt(D) <xbar, q~>  =  2 low pop(code) + 2 step S - D low - step sum_j level_j,
//     S = sum_j code_j level_j = sum_p 2^p popcount(code AND plane_p).
//
// Two independent errors, both with mean zero:
//   * the code: sqrt(1 - <o, q'>^2) sqrt(1 - a^2) / a * u, with u a coordinate
//     of a uniformly distributed unit vector of R^(D-1), variance 1 / (D - 1);
//   * the rounding of the query: <xbar, r> / a with r_j = q~_j - (P^T q')_j,
//     variance step^2 f_j (1 - f_j) per coordinate (f_j the fractional part of
//     the level), so var <xbar, r> = (step^2 / D) sum_j f_j (1 - f_j), which
//     is known when the query is encoded.
// The half-width is confidence_sigma standard deviations of their sum, with
// sqrt(1 - <o, q'>^2) replaced by its upper bound 1 as in the published bound:
//
//     e  =  confidence_sigma * d * dq * sqrt( (1 - a^2) / (D - 1) + var_q ) / a  +  slack.
//
// For a row at the centroid (d = 0) and for a query at the centroid (dq = 0)
// the product vanishes and the estimate is exact up to the slack.
//
// ---- slack ----------------------------------------------------------------------
//
// Both half-widths contain the rounding errors of float32:
//   * of the exact score as dot_f32 computes it, so that the interval also
//     contains what exact_scores and rescore return: a product passes through
//     at most D / 32 + 13 rounding steps of relative size 2^-24, hence
//     2^-23 (D / 32 + 16) ||q|| max ||x||, which leaves a factor of two;
//   * of the estimate, stored as float32: 2^-24 |estimate|, and |estimate| is
//     at most ||q|| ||x|| plus the half-width, so the term above and a relative
//     increase of the half-width by 2^-22 cover it, together with the
//     conversion of the half-width itself, which must not round downwards.
// RaBitQ also computes d, the centred vectors and the rotation in float32. An
// allowance of 2e-6 d (d + dq) is added for that; against the same estimate
// computed in double precision the tests find differences below 1e-7. It is
// an allowance, not a worst case (which is near 1e-4 if every rounding error
// has the same sign): the interval is probabilistic in any case.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "kernels.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/index/vector_store.hpp"
#include "rng.hpp"
#include "store_impl.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)  // results must not depend on the instruction set the compiler may use
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace prag::index {

namespace {

using Impl = VectorStore::Impl;

constexpr unsigned kLevels = (1u << kQueryBits) - 1;  // 15
constexpr std::size_t kParallelRotation = 8;          // queries from which the rotation runs in parallel
constexpr double kRotationRounding = 2e-6;            // allowance for float32 centring and rotation
constexpr double kInflate = 1.0 + 0x1.0p-22;          // see "slack" above

struct QueryInfo {
  bool zero = false;   // the zero vector: every cosine is zero, exactly
  double norm = 0.0;   // ||q||
  double pivot = 0.0;  // <q, centroid>, the mean of the exact scores
  double slack = 0.0;  // rounding slack, part of every half-width
  // Int8
  double residual_norm = 0.0;  // ||q - decoded code of q||
  // RaBitQ
  double distance = 0.0;           // ||q - centroid||
  double pop_factor = 0.0;         // 2 low / sqrt(D)
  double weighted_factor = 0.0;    // 2 step / sqrt(D)
  double constant = 0.0;           // (D low + step sum_j level_j) / sqrt(D)
  double rounding_variance = 0.0;  // var <xbar, r>
};

struct EncodedQueries {
  std::size_t count = 0;
  std::vector<QueryInfo> info;
  AlignedArray<std::int8_t> codes;     // Int8: count * dim
  AlignedArray<std::uint64_t> planes;  // RaBitQ: count * 4 * words; per query plane after plane
};

bool all_finite(std::span<const float> values) noexcept {
  for (const float v : values) {
    if (!std::isfinite(v)) return false;
  }
  return true;
}

// A non-negative half-width as float32, never below the value given.
// Conversion rounds to nearest, with a relative error of at most 2^-24, so the
// value increased by 2^-22 cannot come out below the value itself.
inline float half_width(double value) noexcept {
  const double inflated = value * kInflate;
  if (!(inflated <= static_cast<double>(std::numeric_limits<float>::max()))) {
    return std::numeric_limits<float>::infinity();  // too large, or not a number
  }
  if (inflated < static_cast<double>(std::numeric_limits<float>::min())) {
    return inflated > 0.0 ? std::numeric_limits<float>::min() : 0.0f;
  }
  return static_cast<float>(inflated);
}

double rounding_slack(const Impl& s, double query_norm) noexcept {
  const double steps = static_cast<double>(s.dim) / 32.0 + 16.0;
  return 0x1.0p-23 * steps * query_norm * (1.0 + 2.0 * s.norm_slack) + 1e-9;
}

void describe_query(const Impl& s, const float* q, QueryInfo& info) noexcept {
  double norm2 = 0.0;
  double pivot = 0.0;
  for (std::size_t j = 0; j < s.dim; ++j) {
    const double v = q[j];
    norm2 += v * v;
    pivot += v * static_cast<double>(s.centroid[j]);
  }
  info.zero = norm2 == 0.0;
  info.norm = std::sqrt(norm2);
  info.pivot = pivot;
  info.slack = rounding_slack(s, info.norm);
}

void encode_int8(const Impl& s, std::span<const float> queries, EncodedQueries& out) {
  const std::size_t dim = s.dim;
  out.codes.reset(out.count * dim);
  const double delta = s.scale / 127.0;
  const double inverse = 127.0 / s.scale;
  for (std::size_t i = 0; i < out.count; ++i) {
    const float* q = queries.data() + i * dim;
    std::int8_t* code = out.codes.data() + i * dim;
    double residual2 = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const double v = q[j];
      double level = std::floor(v * inverse + 0.5);
      level = std::min(127.0, std::max(-127.0, level));
      code[j] = static_cast<std::int8_t>(static_cast<int>(level));
      const double r = v - level * delta;
      residual2 += r * r;
    }
    out.info[i].residual_norm = std::sqrt(residual2);
  }
}

// rotated[i * dim + j] = <row j of the rotation, vector i>. In blocked form: a
// block of rows of the rotation is kept in the cache of the worker and every
// query of the batch is multiplied with it before the next block is fetched.
// Both forms compute the same products, so a query is rotated to the same
// bits alone and in a batch.
void rotate(const Impl& s, const float* vectors, std::size_t count, float* rotated,
            exec::OrderedExecutor& executor) {
  const std::size_t dim = s.dim;
  const Kernels& k = kernels();
  const float* rotation = s.rotation->data();
  if (count < kParallelRotation) {
    for (std::size_t j = 0; j < dim; ++j) {
      const float* row = rotation + j * dim;
      for (std::size_t i = 0; i < count; ++i) rotated[i * dim + j] = k.dot_f32(vectors + i * dim, row, dim);
    }
    return;
  }
  const auto blocks = exec::partition_fixed(dim, dim * sizeof(float), s.options.block_bytes);
  exec::RunOptions run;
  run.window = resolved_window(run, executor);
  ResultRing ring(run.window, blocks.size(), blocks.front().count * count * sizeof(float));
  executor.run(
      blocks,
      [&](const exec::Block& b, unsigned) {
        float* out = reinterpret_cast<float*>(ring.slot(b.seq));
        for (std::size_t i = 0; i < count; ++i) {
          const float* v = vectors + i * dim;
          for (std::size_t r = 0; r < b.count; ++r) {
            out[r * count + i] = k.dot_f32(v, rotation + (b.first + r) * dim, dim);
          }
        }
        return out;
      },
      [&](const exec::Block& b, float*&& out) {
        for (std::size_t r = 0; r < b.count; ++r) {
          for (std::size_t i = 0; i < count; ++i) rotated[i * dim + b.first + r] = out[r * count + i];
        }
      },
      run);
}

void encode_rabitq(const Impl& s, std::span<const float> queries, EncodedQueries& out,
                   exec::OrderedExecutor& executor) {
  const std::size_t dim = s.dim;
  const std::size_t words = s.words;
  const std::size_t count = out.count;
  out.planes.reset(count * kQueryBits * words);

  // Centre and normalise.
  AlignedArray<float> centred(count * dim);
  for (std::size_t i = 0; i < count; ++i) {
    const float* q = queries.data() + i * dim;
    float* v = centred.data() + i * dim;
    double norm2 = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const float diff = q[j] - s.centroid[j];
      v[j] = diff;
      norm2 += static_cast<double>(diff) * static_cast<double>(diff);
    }
    const double distance = std::sqrt(norm2);
    if (distance > 0.0) {
      for (std::size_t j = 0; j < dim; ++j) v[j] = static_cast<float>(static_cast<double>(v[j]) / distance);
    }
    out.info[i].distance = distance;
  }

  AlignedArray<float> rotated(count * dim);
  rotate(s, centred.data(), count, rotated.data(), executor);

  const double sqrt_dim = std::sqrt(static_cast<double>(dim));
  for (std::size_t i = 0; i < count; ++i) {
    const float* q = queries.data() + i * dim;
    const float* r = rotated.data() + i * dim;
    std::uint64_t* planes = out.planes.data() + i * kQueryBits * words;
    QueryInfo& info = out.info[i];

    float lowest = r[0];
    float highest = r[0];
    for (std::size_t j = 1; j < dim; ++j) {
      lowest = std::min(lowest, r[j]);
      highest = std::max(highest, r[j]);
    }
    const double low = lowest;
    const double step = (static_cast<double>(highest) - low) / static_cast<double>(kLevels);

    // The rounding is random but reproducible: the generator is seeded with
    // the seed of the store and the bits of the query.
    SplitMix64 random(hash_combine(s.options.seed, checksum64(q, dim * sizeof(float), 0x7175657279ULL)));
    double level_sum = 0.0;
    double variance = 0.0;
    if (step > 0.0) {
      for (std::size_t j = 0; j < dim; ++j) {
        const double t =
            std::min(static_cast<double>(kLevels), std::max(0.0, (static_cast<double>(r[j]) - low) / step));
        const double base = std::floor(t);
        const double fraction = t - base;
        unsigned level = static_cast<unsigned>(base);
        if (random.uniform() < fraction) ++level;
        if (level > kLevels) level = kLevels;
        variance += fraction * (1.0 - fraction);
        level_sum += static_cast<double>(level);
        const std::uint64_t bit = std::uint64_t{1} << (j % 64);
        for (unsigned p = 0; p < kQueryBits; ++p) {
          if ((level >> p) & 1u) planes[p * words + j / 64] |= bit;
        }
      }
    }
    info.pop_factor = 2.0 * low / sqrt_dim;
    info.weighted_factor = 2.0 * step / sqrt_dim;
    info.constant = (static_cast<double>(dim) * low + step * level_sum) / sqrt_dim;
    info.rounding_variance = step * step * variance / static_cast<double>(dim);
  }
}

// Scratch of one worker, allocated once per scan. What depends on the block
// and not on the query is computed once per block and used for every query.
struct alignas(kCacheLine) ScanScratch {
  AlignedArray<std::int32_t> dots;       // Int8: sum_j cq_j cx_j per vector
  AlignedArray<std::uint32_t> ones;      // RaBitQ: popcount of each code
  AlignedArray<std::uint32_t> weighted;  // RaBitQ: S per vector
  AlignedArray<double> inverse;          // RaBitQ: 1 / a
  AlignedArray<double> base;             // RaBitQ: (1 - ||c||^2 - d^2) / 2
  AlignedArray<double> spread;           // RaBitQ: (1 - a^2) / (D - 1)
  std::uint64_t first_touch = 0;
};

struct Accumulator {
  double sum = 0.0;   // of (score - pivot)
  double sum2 = 0.0;  // of (score - pivot)^2
};

std::vector<ScanResult> scan_queries(const VectorStore& store, std::span<const float> queries,
                                     exec::OrderedExecutor& executor, const exec::RunOptions& options) {
  const Impl& s = store.impl();
  std::vector<ScanResult> results;
  if (s.dim == 0 || queries.empty() || queries.size() % s.dim != 0) {
    if (!queries.empty() || s.dim != 0) {
      PRAG_ERROR("index: a query of %zu floats does not match the dimension %zu", queries.size(), s.dim);
    }
    return results;
  }
  if (!all_finite(queries)) {
    PRAG_ERROR("index: the query contains values that are not finite");
    return results;
  }
  const std::size_t count = queries.size() / s.dim;
  const std::size_t dim = s.dim;
  const std::size_t n = s.n;
  results.resize(count);
  for (ScanResult& r : results) {
    r.scores.assign(n, 0.0f);
    r.error.assign(n, 0.0f);
  }
  if (n == 0) return results;

  // ---- the encoded queries: prepared once, read-only during the scan ----------
  EncodedQueries encoded;
  encoded.count = count;
  encoded.info.resize(count);
  for (std::size_t i = 0; i < count; ++i) describe_query(s, queries.data() + i * dim, encoded.info[i]);
  const bool int8 = s.options.codes == CodeKind::Int8;
  if (int8) {
    encode_int8(s, queries, encoded);
  } else {
    encode_rabitq(s, queries, encoded, executor);
  }

  const Kernels& k = kernels();
  const std::size_t words = s.words;
  const std::size_t capacity = s.max_block_count;
  const double delta2 = (s.scale / 127.0) * (s.scale / 127.0);
  const double inverse_dm1 = dim > 1 ? 1.0 / static_cast<double>(dim - 1) : 0.0;
  const double sigma = s.options.confidence_sigma;
  const double centre_term = 1.0 - s.centroid_norm2;
  const double norm_slack = s.norm_slack;

  exec::RunOptions run = options;
  run.window = resolved_window(options, executor);
  // Slot of a block: the two partial sums of every query.
  ResultRing ring(run.window, s.blocks.size(), count * sizeof(Accumulator));

  std::vector<ScanScratch> scratch(executor.threads());
  for (ScanScratch& w : scratch) {
    if (int8) {
      w.dots.reset(capacity);
    } else {
      w.ones.reset(capacity);
      w.weighted.reset(capacity);
      w.inverse.reset(capacity);
      w.base.reset(capacity);
      w.spread.reset(capacity);
    }
  }
  std::vector<Accumulator> totals(count);

  const exec::RunStats stats = executor.run_with_admit(
      s.blocks,
      // admit: the first read of the payload, in block order
      [&](const exec::Block& b, unsigned worker) {
        std::memcpy(&scratch[worker].first_touch, s.payload(b.seq), std::min<std::size_t>(b.bytes, 8));
      },
      // work: the block against every query
      [&](const exec::Block& b, unsigned worker) {
        ScanScratch& w = scratch[worker];
        Accumulator* sums = reinterpret_cast<Accumulator*>(ring.slot(b.seq));
        const float* side = s.side(b.seq);
        const std::uint32_t* units = s.layout.data() + b.first;
        const std::size_t vectors = b.count;

        if (int8) {
          const std::int8_t* codes = reinterpret_cast<const std::int8_t*>(s.codes(b.seq));
          for (std::size_t i = 0; i < count; ++i) {
            const QueryInfo& info = encoded.info[i];
            Accumulator sum;
            if (!info.zero) {
              const std::int8_t* query = encoded.codes.data() + i * dim;
              for (std::size_t v = 0; v < vectors; ++v) w.dots[v] = k.dot_i8(query, codes + v * dim, dim);
              float* scores = results[i].scores.data();
              float* errors = results[i].error.data();
              for (std::size_t v = 0; v < vectors; ++v) {
                const float estimate = static_cast<float>(delta2 * static_cast<double>(w.dots[v]));
                const double bound = info.norm * static_cast<double>(side[2 * v]) +
                                     info.residual_norm * static_cast<double>(side[2 * v + 1]) + info.slack;
                scores[units[v]] = estimate;
                errors[units[v]] = half_width(bound);
                const double centred = static_cast<double>(estimate) - info.pivot;
                sum.sum += centred;
                sum.sum2 += centred * centred;
              }
            }
            sums[i] = sum;
          }
        } else {
          const std::uint64_t* codes = reinterpret_cast<const std::uint64_t*>(s.codes(b.seq));
          for (std::size_t v = 0; v < vectors; ++v) {
            const double d = side[2 * v];
            const double a = side[2 * v + 1];
            w.ones[v] = k.popcount(codes + v * words, words);
            w.inverse[v] = 1.0 / a;
            w.base[v] = 0.5 * (centre_term - d * d);
            w.spread[v] = std::max(0.0, 1.0 - a * a) * inverse_dm1;
          }
          for (std::size_t i = 0; i < count; ++i) {
            const QueryInfo& info = encoded.info[i];
            Accumulator sum;
            if (!info.zero) {
              const std::uint64_t* planes = encoded.planes.data() + i * kQueryBits * words;
              for (std::size_t v = 0; v < vectors; ++v) {
                w.weighted[v] = k.weighted_popcount(codes + v * words, planes, words);
              }
              float* scores = results[i].scores.data();
              float* errors = results[i].error.data();
              for (std::size_t v = 0; v < vectors; ++v) {
                const double d = side[2 * v];
                const double reach = d * info.distance;
                const double inner = (info.pop_factor * static_cast<double>(w.ones[v]) +
                                      info.weighted_factor * static_cast<double>(w.weighted[v]) - info.constant) *
                                     w.inverse[v];
                const float estimate = static_cast<float>(info.pivot + w.base[v] + reach * inner);
                const double deviation = std::sqrt(w.spread[v] + info.rounding_variance) * w.inverse[v];
                const double bound = sigma * reach * deviation + kRotationRounding * d * (d + info.distance) +
                                     norm_slack + info.slack;
                scores[units[v]] = estimate;
                errors[units[v]] = half_width(bound);
                const double centred = static_cast<double>(estimate) - info.pivot;
                sum.sum += centred;
                sum.sum2 += centred * centred;
              }
            }
            sums[i] = sum;
          }
        }
        return sums;
      },
      // commit: the reduction, in storage order
      [&](const exec::Block&, Accumulator*&& sums) {
        for (std::size_t i = 0; i < count; ++i) {
          totals[i].sum += sums[i].sum;
          totals[i].sum2 += sums[i].sum2;
        }
      },
      run);

  for (std::size_t i = 0; i < count; ++i) {
    const double mean = totals[i].sum / static_cast<double>(n);
    const double variance = totals[i].sum2 / static_cast<double>(n) - mean * mean;
    results[i].mean = encoded.info[i].pivot + mean;
    results[i].stddev = variance > 0.0 ? std::sqrt(variance) : 0.0;
    results[i].stats = stats;
  }
  return results;
}

}  // namespace

ScanResult scan(const VectorStore& store, std::span<const float> query, exec::OrderedExecutor& executor,
                const exec::RunOptions& options) {
  if (query.size() != store.dim() || query.empty()) {
    if (query.size() != store.dim()) {
      PRAG_ERROR("index: a query of %zu floats does not match the dimension %zu", query.size(), store.dim());
    }
    return ScanResult{};
  }
  std::vector<ScanResult> results = scan_queries(store, query, executor, options);
  if (results.empty()) return ScanResult{};
  return std::move(results.front());
}

std::vector<ScanResult> scan_batch(const VectorStore& store, std::span<const float> queries,
                                   exec::OrderedExecutor& executor, const exec::RunOptions& options) {
  return scan_queries(store, queries, executor, options);
}

void rescore(const VectorStore& store, std::span<const float> query, std::span<const UnitId> units,
             ScanResult& result) {
  const Impl& s = store.impl();
  if (query.size() != s.dim || result.scores.size() != s.n || result.error.size() != s.n) {
    PRAG_ERROR("index: rescore called with a query or a result that does not belong to the store");
    return;
  }
  const Kernels& k = kernels();
  for (const UnitId unit : units) {
    if (unit >= s.n) continue;
    result.scores[unit] = k.dot_f32(query.data(), s.rows.data() + static_cast<std::size_t>(unit) * s.dim, s.dim);
    result.error[unit] = 0.0f;
  }
}

std::vector<float> exact_scores(const VectorStore& store, std::span<const float> query) {
  const Impl& s = store.impl();
  std::vector<float> scores;
  if (query.size() != s.dim) {
    PRAG_ERROR("index: a query of %zu floats does not match the dimension %zu", query.size(), s.dim);
    return scores;
  }
  scores.resize(s.n);
  const Kernels& k = kernels();
  for (std::size_t unit = 0; unit < s.n; ++unit) {
    scores[unit] = k.dot_f32(query.data(), s.rows.data() + unit * s.dim, s.dim);
  }
  return scores;
}

}  // namespace prag::index
