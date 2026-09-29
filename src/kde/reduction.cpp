// Dimensionality reduction (centring, principal directions, projection to the
// sphere) and the two diagnostics of embedding geometry.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

#include "kernels.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/kde/directional.hpp"

namespace prag::kde {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

// Rows of the covariance are accumulated from chunks of this many data rows, so
// that two columns of a chunk (8 KiB each) stay in the L1 cache together.
constexpr std::size_t kChunkRows = 2048;
constexpr int kMaxSubspaceIterations = 300;
constexpr int kMaxJacobiSweeps = 64;
// Subspace iteration stops when every wanted Ritz pair has a residual
// |C v - lambda v| below this fraction of the largest eigenvalue.
constexpr double kResidualTolerance = 1e-7;

// Counter-based generator: the k-th normal deviate of a stream depends on the
// seed and on k only, which makes the starting basis independent of the
// standard library.
class NormalStream {
 public:
  explicit NormalStream(std::uint64_t seed) : state_(mix64(seed ^ 0x9e3779b97f4a7c15ULL)) {}

  double next() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    // Box-Muller on two uniforms in (0, 1).
    const double u = uniform();
    const double v = uniform();
    const double radius = std::sqrt(-2.0 * std::log(u));
    spare_ = radius * std::sin(kTwoPi * v);
    has_spare_ = true;
    return radius * std::cos(kTwoPi * v);
  }

 private:
  double uniform() {
    state_ += 0x9e3779b97f4a7c15ULL;
    const std::uint64_t bits = mix64(state_) >> 11;  // 53 random bits
    return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
  }

  std::uint64_t state_;
  double spare_ = 0.0;
  bool has_spare_ = false;
};

// Cyclic Jacobi method for a symmetric k x k matrix (row-major, destroyed).
// On return `values` holds the eigenvalues in decreasing order and column j of
// `vectors` (row-major k x k) the eigenvector of values[j].
void symmetric_eigen(std::vector<double>& a, std::size_t k, std::vector<double>& values,
                     std::vector<double>& vectors) {
  vectors.assign(k * k, 0.0);
  for (std::size_t i = 0; i < k; ++i) vectors[i * k + i] = 1.0;

  for (int sweep = 0; sweep < kMaxJacobiSweeps; ++sweep) {
    double off = 0.0;
    double diagonal = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      diagonal += a[i * k + i] * a[i * k + i];
      for (std::size_t j = i + 1; j < k; ++j) off += a[i * k + j] * a[i * k + j];
    }
    if (off <= 1e-30 * diagonal || off == 0.0) break;

    for (std::size_t p = 0; p + 1 < k; ++p) {
      for (std::size_t q = p + 1; q < k; ++q) {
        const double apq = a[p * k + q];
        if (apq == 0.0) continue;
        const double app = a[p * k + p];
        const double aqq = a[q * k + q];
        // Rotation angle that annihilates a[p][q] (Rutishauser's stable form).
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (std::size_t i = 0; i < k; ++i) {  // columns p and q
          const double aip = a[i * k + p];
          const double aiq = a[i * k + q];
          a[i * k + p] = c * aip - s * aiq;
          a[i * k + q] = s * aip + c * aiq;
        }
        for (std::size_t i = 0; i < k; ++i) {  // rows p and q
          const double api = a[p * k + i];
          const double aqi = a[q * k + i];
          a[p * k + i] = c * api - s * aqi;
          a[q * k + i] = s * api + c * aqi;
        }
        a[p * k + q] = 0.0;
        a[q * k + p] = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
          const double vip = vectors[i * k + p];
          const double viq = vectors[i * k + q];
          vectors[i * k + p] = c * vip - s * viq;
          vectors[i * k + q] = s * vip + c * viq;
        }
      }
    }
  }

  // Decreasing eigenvalues; equal values keep their column order.
  std::vector<std::size_t> order(k);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t x, std::size_t y) { return a[x * k + x] > a[y * k + y]; });
  values.resize(k);
  std::vector<double> sorted(k * k);
  for (std::size_t j = 0; j < k; ++j) {
    values[j] = a[order[j] * k + order[j]];
    for (std::size_t i = 0; i < k; ++i) sorted[i * k + j] = vectors[i * k + order[j]];
  }
  vectors.swap(sorted);
}

// Modified Gram-Schmidt with one re-orthogonalisation pass, on the columns of a
// column-major d x k matrix. A column that lies in the span of its predecessors
// (rank-deficient covariance) is replaced by a fresh random direction.
void orthonormalize(std::vector<double>& q, std::size_t d, std::size_t k, NormalStream& random) {
  for (std::size_t j = 0; j < k; ++j) {
    double* column = q.data() + j * d;
    for (int attempt = 0; attempt < 8; ++attempt) {
      const double before = std::sqrt(detail::dot(column, column, d));
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < j; ++i) {
          const double* previous = q.data() + i * d;
          detail::axpy(-detail::dot(previous, column, d), previous, column, d);
        }
      }
      const double after = std::sqrt(detail::dot(column, column, d));
      if (after > 1e-10 * before && after > 0.0) {
        const double inverse = 1.0 / after;
        for (std::size_t i = 0; i < d; ++i) column[i] *= inverse;
        break;
      }
      for (std::size_t i = 0; i < d; ++i) column[i] = random.next();
    }
  }
}

// Covariance of the rows about `mean`, as a full symmetric d x d matrix
// (row-major). Returns the trace through `trace`.
std::vector<double> covariance(std::span<const float> rows, std::size_t n, std::size_t d,
                               const std::vector<float>& mean, double& trace) {
  std::vector<double> c(d * d, 0.0);
  const std::size_t chunk = std::min(n, kChunkRows);
  std::vector<float> columns(d * chunk);  // column-major copy of one chunk, centred
  for (std::size_t first = 0; first < n; first += chunk) {
    const std::size_t count = std::min(chunk, n - first);
    for (std::size_t r = 0; r < count; ++r) {
      const float* row = rows.data() + (first + r) * d;
      for (std::size_t j = 0; j < d; ++j) columns[j * chunk + r] = row[j] - mean[j];
    }
    // Column i stays in the cache while every column j >= i passes by.
    for (std::size_t i = 0; i < d; ++i) {
      const float* ci = columns.data() + i * chunk;
      double* out = c.data() + i * d;
      for (std::size_t j = i; j < d; ++j) out[j] += detail::dot(ci, columns.data() + j * chunk, count);
    }
  }
  const double scale = n > 1 ? 1.0 / static_cast<double>(n - 1) : 0.0;
  trace = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    for (std::size_t j = i; j < d; ++j) {
      const double v = c[i * d + j] * scale;
      c[i * d + j] = v;
      c[j * d + i] = v;
    }
    trace += c[i * d + i];
  }
  return c;
}

struct Spectrum {
  std::vector<double> values;   // k eigenvalue estimates, decreasing
  std::vector<double> vectors;  // column-major d x k, orthonormal columns
  int iterations = 0;
  bool converged = false;
};

// Leading k eigenpairs of the symmetric positive semi-definite matrix c by
// subspace iteration with a Rayleigh-Ritz step per iteration. The first
// `wanted` pairs decide convergence; the others are guard vectors that speed it up.
Spectrum leading_eigenpairs(const std::vector<double>& c, std::size_t d, std::size_t k, std::size_t wanted,
                            std::uint64_t seed) {
  Spectrum out;
  std::vector<double> small(k * k);
  std::vector<double> rotation;

  if (k >= d) {
    // Nothing to iterate: the subspace is the whole space.
    std::vector<double> full(c);
    symmetric_eigen(full, d, out.values, rotation);
    out.vectors.resize(d * d);
    for (std::size_t j = 0; j < d; ++j) {
      for (std::size_t i = 0; i < d; ++i) out.vectors[j * d + i] = rotation[i * d + j];
    }
    out.converged = true;
    return out;
  }

  NormalStream random(seed);
  std::vector<double> q(d * k);
  for (double& v : q) v = random.next();
  orthonormalize(q, d, k, random);

  std::vector<double> y(d * k);
  std::vector<double> ritz(d * k);
  std::vector<double> image(d * k);
  for (int iteration = 1; iteration <= kMaxSubspaceIterations; ++iteration) {
    out.iterations = iteration;
    // Y = C Q. One row of C stays in the cache while the columns of Q pass by.
    for (std::size_t i = 0; i < d; ++i) {
      const double* row = c.data() + i * d;
      for (std::size_t j = 0; j < k; ++j) y[j * d + i] = detail::dot(row, q.data() + j * d, d);
    }
    // Rayleigh-Ritz: T = Q^T C Q = Q^T Y, symmetrised against rounding.
    for (std::size_t a = 0; a < k; ++a) {
      for (std::size_t b = a; b < k; ++b) {
        const double v = 0.5 * (detail::dot(q.data() + a * d, y.data() + b * d, d) +
                                detail::dot(q.data() + b * d, y.data() + a * d, d));
        small[a * k + b] = v;
        small[b * k + a] = v;
      }
    }
    symmetric_eigen(small, k, out.values, rotation);
    // Ritz vectors V = Q W and their images C V = Y W.
    std::fill(ritz.begin(), ritz.end(), 0.0);
    std::fill(image.begin(), image.end(), 0.0);
    for (std::size_t j = 0; j < k; ++j) {
      for (std::size_t a = 0; a < k; ++a) {
        const double w = rotation[a * k + j];
        detail::axpy(w, q.data() + a * d, ritz.data() + j * d, d);
        detail::axpy(w, y.data() + a * d, image.data() + j * d, d);
      }
    }
    double worst = 0.0;
    for (std::size_t j = 0; j < wanted; ++j) {
      double residual = 0.0;
      const double* v = ritz.data() + j * d;
      const double* cv = image.data() + j * d;
      for (std::size_t i = 0; i < d; ++i) {
        const double e = cv[i] - out.values[j] * v[i];
        residual += e * e;
      }
      worst = std::max(worst, std::sqrt(residual));
    }
    if (worst <= kResidualTolerance * std::fabs(out.values[0]) || out.values[0] <= 0.0) {
      out.converged = true;
      break;
    }
    if (iteration == kMaxSubspaceIterations) break;
    q = image;  // C V spans the same space as C Q and its columns are nearly orthogonal already
    orthonormalize(q, d, k, random);
  }
  out.vectors = ritz;
  return out;
}

// y = normalised basis * (x - mean); `centred` is scratch of size in_dim.
void project(const Reduction& r, const float* x, float* y, double* centred) {
  for (std::size_t i = 0; i < r.in_dim; ++i) centred[i] = static_cast<double>(x[i]) - static_cast<double>(r.mean[i]);
  double norm2 = 0.0;
  // The coordinates are kept in double until the norm is known.
  constexpr std::size_t kStack = 64;
  double stack[kStack];
  std::vector<double> heap;
  double* z = stack;
  if (r.out_dim > kStack) {
    heap.resize(r.out_dim);
    z = heap.data();
  }
  for (std::size_t j = 0; j < r.out_dim; ++j) {
    z[j] = detail::dot(r.basis.data() + j * r.in_dim, centred, r.in_dim);
    norm2 += z[j] * z[j];
  }
  if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
    for (std::size_t j = 0; j < r.out_dim; ++j) y[j] = j == 0 ? 1.0f : 0.0f;
    return;
  }
  const double inverse = 1.0 / std::sqrt(norm2);
  for (std::size_t j = 0; j < r.out_dim; ++j) y[j] = static_cast<float>(z[j] * inverse);
}

// Unit-length copy of the rows in single precision; a zero row stays zero.
std::vector<float> normalized_rows(std::span<const float> rows, std::size_t n, std::size_t dim) {
  std::vector<float> out(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = rows.data() + i * dim;
    const double norm = std::sqrt(detail::dot(x, x, dim));
    const double inverse = norm > 0.0 ? 1.0 / norm : 0.0;
    for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = static_cast<float>(static_cast<double>(x[j]) * inverse);
  }
  return out;
}

// The k best (cosine, index) pairs seen so far, best first. Ties go to the lower index.
class NearestList {
 public:
  NearestList(float* scores, std::uint32_t* ids, std::size_t k) : scores_(scores), ids_(ids), k_(k) {}

  void offer(float score, std::uint32_t id, std::uint32_t& size) const {
    if (size == k_) {
      const float last = scores_[k_ - 1];
      if (score < last || (score == last && id > ids_[k_ - 1])) return;
    }
    std::size_t at = size < k_ ? size : k_ - 1;
    while (at > 0 && (scores_[at - 1] < score || (scores_[at - 1] == score && ids_[at - 1] > id))) {
      scores_[at] = scores_[at - 1];
      ids_[at] = ids_[at - 1];
      --at;
    }
    scores_[at] = score;
    ids_[at] = id;
    if (size < k_) ++size;
  }

 private:
  float* scores_;
  std::uint32_t* ids_;
  std::size_t k_;
};

}  // namespace

void Reduction::apply(const float* x, float* y) const {
  if (out_dim == 0) return;
  // One scratch buffer per thread: apply() is const and may be called concurrently.
  thread_local std::vector<double> centred;
  if (centred.size() < in_dim) centred.resize(in_dim);
  project(*this, x, y, centred.data());
}

double Reduction::explained_variance_ratio() const {
  if (!(total_variance > 0.0)) return 0.0;
  double kept = 0.0;
  for (const float v : variance) kept += static_cast<double>(v);
  return std::min(1.0, kept / total_variance);
}

Reduction fit_reduction(std::span<const float> rows, std::size_t in_dim, std::size_t out_dim, std::size_t drop_top,
                        std::uint64_t seed) {
  Reduction r;
  r.in_dim = in_dim;
  const std::size_t n = in_dim == 0 ? 0 : rows.size() / in_dim;
  if (n == 0) return r;

  std::vector<double> sum(in_dim, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = rows.data() + i * in_dim;
    for (std::size_t j = 0; j < in_dim; ++j) sum[j] += static_cast<double>(row[j]);
  }
  r.mean.resize(in_dim);
  for (std::size_t j = 0; j < in_dim; ++j) r.mean[j] = static_cast<float>(sum[j] / static_cast<double>(n));

  const std::size_t limit = std::min(in_dim, n - 1);
  r.dropped = std::min(drop_top, limit);
  r.out_dim = std::min(out_dim, limit - r.dropped);

  const std::vector<double> c = covariance(rows, n, in_dim, r.mean, r.total_variance);
  if (r.out_dim == 0) return r;

  const std::size_t wanted = r.dropped + r.out_dim;
  const std::size_t guard = std::max<std::size_t>(8, wanted / 2);
  const std::size_t k = std::min(in_dim, wanted + guard);
  const Spectrum spectrum = leading_eigenpairs(c, in_dim, k, wanted, seed);
  if (!spectrum.converged) {
    PRAG_DEBUG("kde: subspace iteration stopped after %d iterations without reaching the residual tolerance",
               spectrum.iterations);
  }

  r.basis.resize(r.out_dim * in_dim);
  r.variance.resize(r.out_dim);
  for (std::size_t j = 0; j < r.out_dim; ++j) {
    const double* v = spectrum.vectors.data() + (r.dropped + j) * in_dim;
    // Sign convention: the component of largest magnitude (the first one among
    // equals) is positive.
    std::size_t pivot = 0;
    for (std::size_t i = 1; i < in_dim; ++i) {
      if (std::fabs(v[i]) > std::fabs(v[pivot])) pivot = i;
    }
    const double sign = v[pivot] < 0.0 ? -1.0 : 1.0;
    for (std::size_t i = 0; i < in_dim; ++i) r.basis[j * in_dim + i] = static_cast<float>(sign * v[i]);
    r.variance[j] = static_cast<float>(std::max(0.0, spectrum.values[r.dropped + j]));
  }
  return r;
}

std::vector<float> reduce(const Reduction& reduction, std::span<const float> rows) {
  const std::size_t n = reduction.in_dim == 0 ? 0 : rows.size() / reduction.in_dim;
  std::vector<float> out(n * reduction.out_dim);
  if (reduction.out_dim == 0) return out;
  std::vector<double> centred(reduction.in_dim);
  for (std::size_t i = 0; i < n; ++i) {
    project(reduction, rows.data() + i * reduction.in_dim, out.data() + i * reduction.out_dim, centred.data());
  }
  return out;
}

double mean_pairwise_cosine(std::span<const float> rows, std::size_t dim) {
  const std::size_t n = dim == 0 ? 0 : rows.size() / dim;
  // sum_{i != j} <u_i, u_j> = |sum_i u_i|^2 - sum_i |u_i|^2 for the unit vectors u_i.
  std::vector<double> total(dim, 0.0);
  double squares = 0.0;
  std::size_t used = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = rows.data() + i * dim;
    const double norm2 = detail::dot(x, x, dim);
    if (!(norm2 > 0.0)) continue;  // a zero row has no direction
    const double inverse = 1.0 / std::sqrt(norm2);
    detail::axpy(inverse, x, total.data(), dim);
    squares += norm2 * inverse * inverse;
    ++used;
  }
  if (used < 2) return 0.0;
  const double pairs = static_cast<double>(used) * static_cast<double>(used - 1);
  return (detail::dot(total.data(), total.data(), dim) - squares) / pairs;
}

double hubness_skewness(std::span<const float> rows, std::size_t dim, std::size_t k) {
  const std::size_t n = dim == 0 ? 0 : rows.size() / dim;
  if (k == 0 || n < 3 || n <= k + 1) return 0.0;  // every point is a neighbour of every other one

  const std::vector<float> unit = normalized_rows(rows, n, dim);
  std::vector<float> scores(n * k);
  std::vector<std::uint32_t> ids(n * k);
  std::vector<std::uint32_t> sizes(n, 0);

  // Blocked, symmetric: each pair of rows is scored once and offered to both
  // neighbour lists. A tile of rows stays in the cache while the tiles after it
  // pass by.
  const std::size_t tile = std::max<std::size_t>(1, (8 * 1024) / (dim * sizeof(float)));
  for (std::size_t a0 = 0; a0 < n; a0 += tile) {
    const std::size_t a1 = std::min(n, a0 + tile);
    for (std::size_t b0 = a0; b0 < n; b0 += tile) {
      const std::size_t b1 = std::min(n, b0 + tile);
      for (std::size_t i = a0; i < a1; ++i) {
        const float* xi = unit.data() + i * dim;
        const NearestList list_i(scores.data() + i * k, ids.data() + i * k, k);
        for (std::size_t j = std::max(b0, i + 1); j < b1; ++j) {
          const float score = detail::dot_single(xi, unit.data() + j * dim, dim);
          list_i.offer(score, static_cast<std::uint32_t>(j), sizes[i]);
          const NearestList list_j(scores.data() + j * k, ids.data() + j * k, k);
          list_j.offer(score, static_cast<std::uint32_t>(i), sizes[j]);
        }
      }
    }
  }

  std::vector<std::uint32_t> occurrences(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t s = 0; s < sizes[i]; ++s) ++occurrences[ids[i * k + s]];
  }
  double mean = 0.0;
  for (const std::uint32_t v : occurrences) mean += static_cast<double>(v);
  mean /= static_cast<double>(n);
  double m2 = 0.0;
  double m3 = 0.0;
  for (const std::uint32_t v : occurrences) {
    const double e = static_cast<double>(v) - mean;
    m2 += e * e;
    m3 += e * e * e;
  }
  m2 /= static_cast<double>(n);
  m3 /= static_cast<double>(n);
  if (!(m2 > 0.0)) return 0.0;
  return m3 / (m2 * std::sqrt(m2));
}

}  // namespace prag::kde
