// Random samples for the kde tests: a small generator with its own normal,
// gamma and beta variates (so that the samples do not depend on the standard
// library), uniform directions, and the von Mises-Fisher sampler of Wood (1994).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prag::kde::testing {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    // splitmix64 expands the seed into the state of xoshiro256**.
    std::uint64_t z = seed;
    for (std::uint64_t& s : state_) {
      z += 0x9e3779b97f4a7c15ULL;
      std::uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      s = x ^ (x >> 31);
    }
  }

  std::uint64_t next() {
    const std::uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const std::uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

  // Uniform in (0, 1).
  double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

  std::size_t below(std::size_t n) { return static_cast<std::size_t>(uniform() * static_cast<double>(n)); }

  // Standard normal (Marsaglia's polar method).
  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    for (;;) {
      const double u = 2.0 * uniform() - 1.0;
      const double v = 2.0 * uniform() - 1.0;
      const double s = u * u + v * v;
      if (s >= 1.0 || s == 0.0) continue;
      const double f = std::sqrt(-2.0 * std::log(s) / s);
      spare_ = v * f;
      has_spare_ = true;
      return u * f;
    }
  }

  // Gamma(shape, 1) by Marsaglia and Tsang (2000).
  double gamma(double shape) {
    if (shape < 1.0) return gamma(shape + 1.0) * std::pow(uniform(), 1.0 / shape);
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      const double x = normal();
      const double t = 1.0 + c * x;
      if (t <= 0.0) continue;
      const double v = t * t * t;
      const double u = uniform();
      if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
    }
  }

  double beta(double a, double b) {
    const double x = gamma(a);
    const double y = gamma(b);
    return x / (x + y);
  }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  std::uint64_t state_[4];
  double spare_ = 0.0;
  bool has_spare_ = false;
};

// A direction distributed uniformly on S^{m-1}.
inline std::vector<double> uniform_direction(Rng& rng, std::size_t m) {
  std::vector<double> x(m);
  double norm2 = 0.0;
  do {
    norm2 = 0.0;
    for (double& v : x) {
      v = rng.normal();
      norm2 += v * v;
    }
  } while (norm2 == 0.0);
  const double inverse = 1.0 / std::sqrt(norm2);
  for (double& v : x) v *= inverse;
  return x;
}

// k orthonormal vectors of R^d (row-major), by Gram-Schmidt on normal vectors.
inline std::vector<double> orthonormal_rows(Rng& rng, std::size_t k, std::size_t d) {
  std::vector<double> rows(k * d);
  for (std::size_t i = 0; i < k; ++i) {
    double* row = rows.data() + i * d;
    for (;;) {
      for (std::size_t j = 0; j < d; ++j) row[j] = rng.normal();
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t p = 0; p < i; ++p) {
          const double* previous = rows.data() + p * d;
          double dot = 0.0;
          for (std::size_t j = 0; j < d; ++j) dot += previous[j] * row[j];
          for (std::size_t j = 0; j < d; ++j) row[j] -= dot * previous[j];
        }
      }
      double norm2 = 0.0;
      for (std::size_t j = 0; j < d; ++j) norm2 += row[j] * row[j];
      if (norm2 < 1e-12) continue;
      const double inverse = 1.0 / std::sqrt(norm2);
      for (std::size_t j = 0; j < d; ++j) row[j] *= inverse;
      break;
    }
  }
  return rows;
}

// One sample of the von Mises-Fisher distribution with mean direction mu (unit
// vector of R^m, m >= 2) and concentration kappa, written to out[0..m).
// Wood (1994): the component along mu is drawn by rejection from a transformed
// beta variate, the orthogonal part is uniform on S^{m-2}.
inline void sample_vmf(Rng& rng, const double* mu, std::size_t m, double kappa, float* out) {
  const double dim = static_cast<double>(m) - 1.0;
  double w = 0.0;
  if (kappa <= 0.0) {
    // Uniform: the component along any axis of a uniform direction.
    w = uniform_direction(rng, m)[0];
  } else {
    const double b = dim / (2.0 * kappa + std::sqrt(4.0 * kappa * kappa + dim * dim));
    const double x0 = (1.0 - b) / (1.0 + b);
    const double c = kappa * x0 + dim * std::log(1.0 - x0 * x0);
    for (;;) {
      const double z = rng.beta(0.5 * dim, 0.5 * dim);
      const double u = rng.uniform();
      w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
      if (kappa * w + dim * std::log(1.0 - x0 * w) - c >= std::log(u)) break;
    }
  }
  // Point with component w along the last axis.
  std::vector<double> x(m);
  const std::vector<double> v = uniform_direction(rng, m - 1);
  const double radius = std::sqrt(std::fmax(0.0, 1.0 - w * w));
  for (std::size_t j = 0; j + 1 < m; ++j) x[j] = radius * v[j];
  x[m - 1] = w;
  // Householder reflection that maps the last axis to mu.
  std::vector<double> h(mu, mu + m);
  h[m - 1] -= 1.0;  // h = mu - e_m
  double h2 = 0.0;
  for (const double value : h) h2 += value * value;
  if (h2 > 1e-24) {
    double dot = 0.0;
    for (std::size_t j = 0; j < m; ++j) dot += h[j] * x[j];
    const double factor = 2.0 * dot / h2;
    for (std::size_t j = 0; j < m; ++j) x[j] -= factor * h[j];
  }
  double norm2 = 0.0;
  for (const double value : x) norm2 += value * value;
  const double inverse = 1.0 / std::sqrt(norm2);
  for (std::size_t j = 0; j < m; ++j) out[j] = static_cast<float>(x[j] * inverse);
}

struct Mixture {
  std::size_t m = 0;
  std::vector<std::vector<double>> means;
  std::vector<double> kappas;
  std::vector<double> weights;  // sum to one
};

struct Sample {
  std::vector<float> rows;            // n * m unit vectors
  std::vector<std::uint32_t> labels;  // component of each row
};

// n points from the mixture. With `balanced` the components receive
// round(weight * n) points each (the last one takes the remainder) and the rows
// are shuffled; otherwise the component of every point is drawn at random.
inline Sample sample_mixture(Rng& rng, const Mixture& mixture, std::size_t n, bool balanced) {
  Sample sample;
  sample.rows.resize(n * mixture.m);
  sample.labels.resize(n);
  const std::size_t k = mixture.means.size();
  if (balanced) {
    std::size_t at = 0;
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t count = static_cast<std::size_t>(std::llround(mixture.weights[c] * static_cast<double>(n)));
      if (c + 1 == k || at + count > n) count = n - at;
      for (std::size_t i = 0; i < count; ++i) sample.labels[at++] = static_cast<std::uint32_t>(c);
    }
    for (std::size_t i = n; i > 1; --i) {  // Fisher-Yates
      const std::size_t j = rng.below(i);
      const std::uint32_t t = sample.labels[i - 1];
      sample.labels[i - 1] = sample.labels[j];
      sample.labels[j] = t;
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const double u = rng.uniform();
      double cumulative = 0.0;
      std::size_t c = 0;
      for (; c + 1 < k; ++c) {
        cumulative += mixture.weights[c];
        if (u < cumulative) break;
      }
      sample.labels[i] = static_cast<std::uint32_t>(c);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = sample.labels[i];
    sample_vmf(rng, mixture.means[c].data(), mixture.m, mixture.kappas[c], sample.rows.data() + i * mixture.m);
  }
  return sample;
}

// Mixture of k components with equal weights and concentration kappa whose
// means are orthonormal, hence well separated.
inline Mixture separated_mixture(Rng& rng, std::size_t m, std::size_t k, double kappa) {
  Mixture mixture;
  mixture.m = m;
  const std::vector<double> means = orthonormal_rows(rng, k, m);
  for (std::size_t c = 0; c < k; ++c) {
    mixture.means.emplace_back(means.begin() + static_cast<std::ptrdiff_t>(c * m),
                               means.begin() + static_cast<std::ptrdiff_t>((c + 1) * m));
    mixture.kappas.push_back(kappa);
    mixture.weights.push_back(1.0 / static_cast<double>(k));
  }
  return mixture;
}

}  // namespace prag::kde::testing
