// The random rotation of the RaBitQ code.
//
// A matrix G with independent standard normal entries is factorised as
// G^T = Q^T R, R upper triangular, by Gram-Schmidt orthogonalisation of the
// rows of G; the rows of Q are the result. The diagonal of R holds the norms
// of the residuals and is therefore positive, which is the condition under
// which Q is distributed uniformly over the orthogonal group (Haar measure);
// no sign correction is needed.
//
// Modified Gram-Schmidt in double precision: row j is made orthogonal to the
// rows before it one after the other, each time using the current residual.
// The loss of orthogonality is of the order of the condition number of G times
// 1e-16, far below the float32 precision of the stored matrix.
//
// Rows are processed in panels. The projections of a panel row on the rows
// before the panel do not involve the other rows of the panel, so they run in
// parallel, one row per worker, with the row (8 * dim bytes) hot and the
// earlier rows streaming past it. Every row undergoes the same operations in
// the same order as in the sequential algorithm, so the matrix does not depend
// on the panel size or on the number of threads.
#include <algorithm>
#include <cmath>
#include <mutex>

#include "kernels.hpp"
#include "prag/core/hash.hpp"
#include "rng.hpp"
#include "store_impl.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)  // results must not depend on the instruction set the compiler may use
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace prag::index {

namespace {

constexpr std::size_t kPanelRows = 32;
constexpr std::size_t kPanelBlockBytes = 16 * 1024;
constexpr double kDegenerate = 1e-8;  // residual norm below which a row is drawn again

// v <- v - <q, v> q
inline void project_out(double* v, const double* q, std::size_t dim) noexcept {
  subtract_scaled(v, q, dot_f64(q, v, dim), dim);
}

std::shared_ptr<const Rotation> generate(std::size_t dim, std::uint64_t seed, exec::OrderedExecutor& executor) {
  AlignedArray<double> g(dim * dim);
  Gaussian normal(hash_combine(seed, 0x726f746174696f6eULL));
  for (std::size_t i = 0; i < dim * dim; ++i) g[i] = normal.next();
  Gaussian replacement(hash_combine(seed, 0x7265706c6163656dULL));

  for (std::size_t begin = 0; begin < dim; begin += kPanelRows) {
    const std::size_t end = std::min(dim, begin + kPanelRows);
    if (begin != 0) {
      const auto blocks = exec::partition_fixed(end - begin, dim * sizeof(double),
                                                std::max(kPanelBlockBytes, dim * sizeof(double)));
      executor.run(
          blocks,
          [&](const exec::Block& b, unsigned) {
            for (std::size_t r = 0; r < b.count; ++r) {
              double* v = g.data() + (begin + b.first + r) * dim;
              for (std::size_t k = 0; k < begin; ++k) project_out(v, g.data() + k * dim, dim);
            }
            return 0;
          },
          [](const exec::Block&, int&&) {});
    }
    for (std::size_t j = begin; j < end; ++j) {
      double* v = g.data() + j * dim;
      for (std::size_t k = begin; k < j; ++k) project_out(v, g.data() + k * dim, dim);
      double norm = std::sqrt(dot_f64(v, v, dim));
      while (!(norm > kDegenerate)) {
        // The row was (numerically) a combination of the rows before it.
        for (std::size_t i = 0; i < dim; ++i) v[i] = replacement.next();
        for (std::size_t k = 0; k < j; ++k) project_out(v, g.data() + k * dim, dim);
        norm = std::sqrt(dot_f64(v, v, dim));
      }
      const double inverse = 1.0 / norm;
      for (std::size_t i = 0; i < dim; ++i) v[i] *= inverse;
    }
  }

  auto rotation = std::make_shared<Rotation>(dim * dim);
  for (std::size_t i = 0; i < dim * dim; ++i) (*rotation)[i] = static_cast<float>(g[i]);
  return rotation;
}

struct CacheEntry {
  std::size_t dim = 0;
  std::uint64_t seed = 0;
  std::shared_ptr<const Rotation> matrix;
};

// The two most recent matrices. Rebuilding an index with the same dimension
// and seed, the common case, then costs nothing.
std::mutex g_cache_mutex;
CacheEntry g_cache[2];

}  // namespace

std::shared_ptr<const Rotation> make_rotation(std::size_t dim, std::uint64_t seed, exec::OrderedExecutor& executor) {
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for (CacheEntry& entry : g_cache) {
      if (entry.matrix && entry.dim == dim && entry.seed == seed) return entry.matrix;
    }
  }
  std::shared_ptr<const Rotation> matrix = generate(dim, seed, executor);
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache[1] = std::move(g_cache[0]);
    g_cache[0] = CacheEntry{dim, seed, matrix};
  }
  return matrix;
}

}  // namespace prag::index
