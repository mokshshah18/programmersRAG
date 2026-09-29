#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <vector>

#include "prag/kde/directional.hpp"
#include "prag/kde/vmf.hpp"
#include "sampling.hpp"

using namespace prag::kde;
using namespace prag::kde::testing;
using prag::exec::OrderedExecutor;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kKiB = 1024;

double dot(const float* a, const float* b, std::size_t m) {
  double s = 0.0;
  for (std::size_t j = 0; j < m; ++j) s += static_cast<double>(a[j]) * static_cast<double>(b[j]);
  return s;
}

double dot(const float* a, const double* b, std::size_t m) {
  double s = 0.0;
  for (std::size_t j = 0; j < m; ++j) s += static_cast<double>(a[j]) * b[j];
  return s;
}

// log fhat at y by the definition: two plain loops, the maximum exponent taken out of the sum.
template <class Point>
double naive_log_density(const std::vector<float>& data, std::size_t m, const Point* y, double kappa,
                         std::size_t skip = std::numeric_limits<std::size_t>::max()) {
  const std::size_t n = data.size() / m;
  std::vector<double> exponents;
  for (std::size_t i = 0; i < n; ++i) {
    if (i != skip) exponents.push_back(kappa * dot(data.data() + i * m, y, m));
  }
  const double top = *std::max_element(exponents.begin(), exponents.end());
  double sum = 0.0;
  for (const double e : exponents) sum += std::exp(e - top);
  return log_vmf_normalizer(m, kappa) - std::log(static_cast<double>(exponents.size())) + top + std::log(sum);
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Rows x = offset + sum_j z_j scale_j u_j + noise * g with z, g standard normal.
std::vector<float> planted_rows(Rng& rng, std::size_t n, std::size_t d, const std::vector<double>& directions,
                                const std::vector<double>& scales, double noise, double offset) {
  std::vector<float> rows(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> x(d, offset);
    for (std::size_t j = 0; j < scales.size(); ++j) {
      const double z = rng.normal() * scales[j];
      for (std::size_t t = 0; t < d; ++t) x[t] += z * directions[j * d + t];
    }
    for (std::size_t t = 0; t < d; ++t) rows[i * d + t] = static_cast<float>(x[t] + noise * rng.normal());
  }
  return rows;
}

// Covariance about the mean stored in the reduction, by the definition.
std::vector<double> naive_covariance(const std::vector<float>& rows, std::size_t d, const std::vector<float>& mean) {
  const std::size_t n = rows.size() / d;
  std::vector<double> c(d * d, 0.0);
  std::vector<double> x(d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) x[t] = static_cast<double>(rows[i * d + t]) - static_cast<double>(mean[t]);
    for (std::size_t a = 0; a < d; ++a) {
      for (std::size_t b = 0; b < d; ++b) c[a * d + b] += x[a] * x[b];
    }
  }
  for (double& v : c) v /= static_cast<double>(n - 1);
  return c;
}

// Largest |C b - lambda b| over the rows b of the basis, in units of the largest eigenvalue.
double worst_residual(const Reduction& r, const std::vector<double>& c) {
  const std::size_t d = r.in_dim;
  double worst = 0.0;
  for (std::size_t j = 0; j < r.out_dim; ++j) {
    const float* b = r.basis.data() + j * d;
    double residual = 0.0;
    for (std::size_t a = 0; a < d; ++a) {
      double cb = 0.0;
      for (std::size_t t = 0; t < d; ++t) cb += c[a * d + t] * static_cast<double>(b[t]);
      const double e = cb - static_cast<double>(r.variance[j]) * static_cast<double>(b[a]);
      residual += e * e;
    }
    worst = std::max(worst, std::sqrt(residual));
  }
  return worst / static_cast<double>(r.variance[0]);
}

void check_orthonormal(const Reduction& r) {
  for (std::size_t a = 0; a < r.out_dim; ++a) {
    for (std::size_t b = a; b < r.out_dim; ++b) {
      const double g = dot(r.basis.data() + a * r.in_dim, r.basis.data() + b * r.in_dim, r.in_dim);
      CHECK_NEAR(g, a == b ? 1.0 : 0.0, 1e-5);
    }
  }
}

Sample planted_mixture(std::uint64_t seed, std::size_t m, std::size_t components, std::size_t n, double kappa) {
  Rng rng(seed);
  const Mixture mixture = separated_mixture(rng, m, components, kappa);
  return sample_mixture(rng, mixture, n, true);
}

}  // namespace

// ---- the sampler used by the tests -----------------------------------------

PRAG_TEST(vmf_sampler_has_the_right_mean_resultant_length) {
  Rng rng(11);
  for (const std::size_t m : {std::size_t{3}, std::size_t{8}, std::size_t{32}}) {
    for (const double kappa : {2.0, 25.0, 300.0}) {
      const std::vector<double> mu = uniform_direction(rng, m);
      const std::size_t n = 4000;
      std::vector<float> x(m);
      std::vector<double> sum(m, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        sample_vmf(rng, mu.data(), m, kappa, x.data());
        CHECK_NEAR(dot(x.data(), x.data(), m), 1.0, 1e-6);
        for (std::size_t j = 0; j < m; ++j) sum[j] += static_cast<double>(x[j]);
      }
      double along = 0.0;
      double norm2 = 0.0;
      for (std::size_t j = 0; j < m; ++j) {
        along += sum[j] * mu[j];
        norm2 += sum[j] * sum[j];
      }
      const double expected = vmf_mean_resultant_length(m, kappa);
      // Standard error of the mean of <mu, x> is at most 1 / sqrt(n) = 0.016.
      CHECK_NEAR(along / static_cast<double>(n), expected, 0.03);
      CHECK_NEAR(std::sqrt(norm2) / static_cast<double>(n), expected, 0.03);
    }
  }
}

// ---- reduction ---------------------------------------------------------------

PRAG_TEST(reduction_recovers_a_planted_subspace) {
  Rng rng(3);
  const std::size_t d = 48;
  const std::size_t n = 900;
  const std::vector<double> scales = {8.0, 5.0, 3.2, 2.0, 1.3, 0.8};
  const std::vector<double> directions = orthonormal_rows(rng, scales.size(), d);
  const std::vector<float> rows = planted_rows(rng, n, d, directions, scales, 0.05, 0.7);

  const Reduction r = fit_reduction(rows, d, 6, 0, 1);
  REQUIRE(r.in_dim == d);
  REQUIRE(r.out_dim == 6u);
  CHECK_EQ(r.dropped, std::size_t{0});
  REQUIRE(r.basis.size() == 6 * d);
  REQUIRE(r.variance.size() == 6u);
  REQUIRE(r.mean.size() == d);
  for (std::size_t t = 0; t < d; ++t) CHECK_NEAR(r.mean[t], 0.7, 0.5);  // offset plus sampling error of the mean
  check_orthonormal(r);

  for (std::size_t j = 0; j < 6; ++j) {
    if (j > 0) CHECK(r.variance[j] <= r.variance[j - 1]);
    // Variance close to the planted one (sampling error of a variance: sqrt(2 / n) = 5 %).
    const double planted = scales[j] * scales[j] + 0.05 * 0.05;
    CHECK(std::fabs(static_cast<double>(r.variance[j]) - planted) <= 0.2 * planted);
    // The direction lies in the planted subspace and is aligned with its own planted direction.
    double inside = 0.0;
    for (std::size_t p = 0; p < 6; ++p) {
      const double c = dot(r.basis.data() + j * d, directions.data() + p * d, d);
      inside += c * c;
    }
    CHECK(inside > 0.999);
    CHECK(std::fabs(dot(r.basis.data() + j * d, directions.data() + j * d, d)) > 0.95);
    // Sign convention: the largest component is positive.
    const float* b = r.basis.data() + j * d;
    std::size_t pivot = 0;
    for (std::size_t t = 1; t < d; ++t) {
      if (std::fabs(b[t]) > std::fabs(b[pivot])) pivot = t;
    }
    CHECK(b[pivot] > 0.0f);
  }

  // Against the sample covariance itself: eigenpairs, and the variance captured.
  const std::vector<double> c = naive_covariance(rows, d, r.mean);
  CHECK(worst_residual(r, c) < 1e-5);
  double trace = 0.0;
  for (std::size_t t = 0; t < d; ++t) trace += c[t * d + t];
  CHECK_NEAR(r.total_variance, trace, 1e-9 * trace);
  double kept = 0.0;
  for (const float v : r.variance) kept += static_cast<double>(v);
  CHECK_NEAR(r.explained_variance_ratio(), kept / trace, 1e-9);
  CHECK(r.explained_variance_ratio() > 0.99);
  CHECK(r.explained_variance_ratio() <= 1.0);
}

PRAG_TEST(reduction_is_deterministic_for_a_seed) {
  Rng rng(5);
  const std::size_t d = 160;
  const std::vector<double> scales = {6.0, 5.0, 4.2, 3.5, 2.9, 2.4, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6};
  const std::vector<double> directions = orthonormal_rows(rng, scales.size(), d);
  const std::vector<float> rows = planted_rows(rng, 500, d, directions, scales, 0.2, 0.0);

  const Reduction a = fit_reduction(rows, d, 8, 0, 42);
  const Reduction b = fit_reduction(rows, d, 8, 0, 42);
  CHECK(same_bits(a.basis, b.basis));
  CHECK(same_bits(a.variance, b.variance));
  CHECK(same_bits(a.mean, b.mean));
  CHECK_EQ(a.total_variance, b.total_variance);

  // Another starting basis converges to the same directions.
  const Reduction other = fit_reduction(rows, d, 8, 0, 43);
  REQUIRE(other.basis.size() == a.basis.size());
  for (std::size_t j = 0; j < 8; ++j) {
    CHECK(dot(a.basis.data() + j * d, other.basis.data() + j * d, d) > 1.0 - 1e-6);
    CHECK_NEAR(a.variance[j], other.variance[j], 1e-5 * a.variance[0]);
  }
  check_orthonormal(a);
  CHECK(worst_residual(a, naive_covariance(rows, d, a.mean)) < 1e-5);
}

PRAG_TEST(reduction_drop_top_removes_the_leading_directions) {
  Rng rng(8);
  const std::size_t d = 40;
  const std::vector<double> scales = {9.0, 6.0, 4.0, 2.5, 1.5, 1.0};
  const std::vector<double> directions = orthonormal_rows(rng, scales.size(), d);
  const std::vector<float> rows = planted_rows(rng, 700, d, directions, scales, 0.05, -0.3);

  const Reduction all = fit_reduction(rows, d, 6, 0, 1);
  const Reduction tail = fit_reduction(rows, d, 4, 2, 1);
  REQUIRE(tail.out_dim == 4u);
  CHECK_EQ(tail.dropped, std::size_t{2});
  check_orthonormal(tail);
  for (std::size_t j = 0; j < 4; ++j) {
    CHECK(dot(tail.basis.data() + j * d, all.basis.data() + (j + 2) * d, d) > 1.0 - 1e-6);
    CHECK_NEAR(tail.variance[j], all.variance[j + 2], 1e-5 * all.variance[0]);
    // Orthogonal to what was dropped.
    CHECK_NEAR(dot(tail.basis.data() + j * d, all.basis.data(), d), 0.0, 1e-5);
    CHECK_NEAR(dot(tail.basis.data() + j * d, all.basis.data() + d, d), 0.0, 1e-5);
  }
  CHECK_NEAR(tail.total_variance, all.total_variance, 1e-12 * all.total_variance);
  CHECK(tail.explained_variance_ratio() < 0.2);
}

PRAG_TEST(reduction_outputs_are_unit_vectors) {
  Rng rng(21);
  const std::size_t d = 64;
  const std::size_t n = 300;
  std::vector<float> rows(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    // Anisotropic cloud on the sphere: a common direction plus noise.
    const std::vector<double> g = uniform_direction(rng, d);
    double norm2 = 0.0;
    std::vector<double> x(d);
    for (std::size_t t = 0; t < d; ++t) {
      x[t] = (t == 0 ? 1.5 : 0.0) + g[t];
      norm2 += x[t] * x[t];
    }
    for (std::size_t t = 0; t < d; ++t) rows[i * d + t] = static_cast<float>(x[t] / std::sqrt(norm2));
  }
  const Reduction r = fit_reduction(rows, d, 12, 0, 1);
  REQUIRE(r.out_dim == 12u);
  const std::vector<float> reduced = reduce(r, rows);
  REQUIRE(reduced.size() == n * 12);
  std::vector<float> single(12);
  for (std::size_t i = 0; i < n; ++i) {
    CHECK_NEAR(dot(reduced.data() + i * 12, reduced.data() + i * 12, 12), 1.0, 1e-6);
    r.apply(rows.data() + i * d, single.data());
    CHECK(std::memcmp(single.data(), reduced.data() + i * 12, 12 * sizeof(float)) == 0);
  }
  // Centring removes the common direction.
  CHECK(mean_pairwise_cosine(rows, d) > 0.5);
  CHECK(std::fabs(mean_pairwise_cosine(reduced, 12)) < 0.05);
  // The mean itself has no direction left; it maps to e_1.
  r.apply(r.mean.data(), single.data());
  CHECK_EQ(single[0], 1.0f);
  for (std::size_t j = 1; j < 12; ++j) CHECK_EQ(single[j], 0.0f);
}

PRAG_TEST(reduction_clamps_the_output_dimension) {
  Rng rng(2);
  const std::size_t d = 20;
  std::vector<float> rows(5 * d);
  for (float& v : rows) v = static_cast<float>(rng.normal());
  const Reduction few = fit_reduction(rows, d, 32, 0, 1);  // five points span four directions
  CHECK_EQ(few.out_dim, std::size_t{4});
  check_orthonormal(few);
  const Reduction dropped = fit_reduction(rows, d, 32, 3, 1);
  CHECK_EQ(dropped.dropped, std::size_t{3});
  CHECK_EQ(dropped.out_dim, std::size_t{1});
  const Reduction too_many = fit_reduction(rows, d, 4, 9, 1);
  CHECK_EQ(too_many.dropped, std::size_t{4});
  CHECK_EQ(too_many.out_dim, std::size_t{0});

  // Every direction requested: the basis is a rotation of the whole space.
  std::vector<float> cloud(200 * 6);
  for (float& v : cloud) v = static_cast<float>(rng.normal());
  const Reduction full = fit_reduction(cloud, 6, 6, 0, 1);
  CHECK_EQ(full.out_dim, std::size_t{6});
  check_orthonormal(full);
  CHECK_NEAR(full.explained_variance_ratio(), 1.0, 1e-6);
  CHECK(worst_residual(full, naive_covariance(cloud, 6, full.mean)) < 1e-5);

  const Reduction none = fit_reduction(std::vector<float>{}, d, 8, 0, 1);
  CHECK_EQ(none.out_dim, std::size_t{0});
  CHECK(reduce(none, std::vector<float>{}).empty());
  const Reduction one = fit_reduction(std::vector<float>(d, 1.0f), d, 8, 0, 1);
  CHECK_EQ(one.out_dim, std::size_t{0});
}

// ---- density -----------------------------------------------------------------

PRAG_TEST(log_density_matches_the_double_loop) {
  const std::size_t m = 24;
  const Sample data = planted_mixture(17, m, 6, 700, 60.0);
  const Sample queries = planted_mixture(18, m, 6, 250, 30.0);
  OrderedExecutor executor(4);
  for (const double kappa : {0.5, 20.0, 400.0, 20000.0}) {
    const std::vector<double> plain = log_density(data.rows, queries.rows, m, kappa, false, executor, 16 * kKiB);
    REQUIRE(plain.size() == 250u);
    for (std::size_t i = 0; i < plain.size(); ++i) {
      const double expected = naive_log_density(data.rows, m, queries.rows.data() + i * m, kappa);
      CHECK(std::fabs(plain[i] - expected) <= 1e-10 * std::fabs(expected));
    }
    const std::vector<double> loo = log_density(data.rows, data.rows, m, kappa, true, executor, 16 * kKiB);
    const std::vector<double> full = log_density(data.rows, data.rows, m, kappa, false, executor, 16 * kKiB);
    REQUIRE(loo.size() == 700u);
    for (std::size_t i = 0; i < loo.size(); ++i) {
      const double expected = naive_log_density(data.rows, m, data.rows.data() + i * m, kappa, i);
      CHECK(std::fabs(loo[i] - expected) <= 1e-10 * std::fabs(expected));
      // Leaving the point out removes its own kernel, the largest term of the sum.
      CHECK(loo[i] < full[i]);
    }
  }
  CHECK(log_density(data.rows, std::vector<float>{}, m, 10.0, false, executor, 16 * kKiB).empty());
}

PRAG_TEST(log_density_is_bit_identical_for_every_thread_count_and_block_size) {
  const std::size_t m = 32;
  const Sample data = planted_mixture(23, m, 8, 900, 80.0);
  OrderedExecutor reference_executor(1);
  const std::vector<double> reference = log_density(data.rows, data.rows, m, 35.0, true, reference_executor, 16 * kKiB);
  const std::vector<double> reference_plain =
      log_density(data.rows, data.rows, m, 35.0, false, reference_executor, 16 * kKiB);
  for (const unsigned threads : {1u, 3u, 8u, 24u}) {
    OrderedExecutor executor(threads);
    for (const std::size_t block : {4 * kKiB, 16 * kKiB, 64 * kKiB}) {
      CHECK(same_bits(log_density(data.rows, data.rows, m, 35.0, true, executor, block), reference));
      CHECK(same_bits(log_density(data.rows, data.rows, m, 35.0, false, executor, block), reference_plain));
    }
  }
}

PRAG_TEST(density_integrates_to_one_on_the_circle) {
  // m = 2: the estimate is a function of the angle and can be integrated by the trapezoidal rule.
  Rng rng(4);
  const double mu[2] = {0.6, 0.8};
  std::vector<float> data(60 * 2);
  for (std::size_t i = 0; i < 60; ++i) sample_vmf(rng, mu, 2, 6.0, data.data() + i * 2);
  const std::size_t steps = 4000;
  std::vector<float> circle(steps * 2);
  for (std::size_t i = 0; i < steps; ++i) {
    const double angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(steps);
    circle[i * 2] = static_cast<float>(std::cos(angle));
    circle[i * 2 + 1] = static_cast<float>(std::sin(angle));
  }
  OrderedExecutor executor(2);
  const std::vector<double> values = log_density(data, circle, 2, 40.0, false, executor, 16 * kKiB);
  double integral = 0.0;
  for (const double v : values) integral += std::exp(v);
  integral *= 2.0 * kPi / static_cast<double>(steps);
  CHECK_NEAR(integral, 1.0, 1e-6);
}

// ---- bandwidth ---------------------------------------------------------------

PRAG_TEST(rule_of_thumb_reproduces_the_closed_form_on_the_sphere) {
  // q = 2 (Garcia-Portugues 2013, after Proposition 2):
  //   h^6 = 8 sinh^2(k) / ( k [ (1 + 4 k^2) sinh(2k) - 2 k cosh(2k) ] n ).
  Rng rng(31);
  for (const double concentration : {1.5, 10.0, 40.0}) {
    const std::vector<double> mu = uniform_direction(rng, 3);
    const std::size_t n = 500;
    std::vector<float> data(n * 3);
    for (std::size_t i = 0; i < n; ++i) sample_vmf(rng, mu.data(), 3, concentration, data.data() + i * 3);

    double sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
      const double norm = std::sqrt(dot(data.data() + i * 3, data.data() + i * 3, 3));
      for (std::size_t j = 0; j < 3; ++j) sum[j] += static_cast<double>(data[i * 3 + j]) / norm;
    }
    const double rbar = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]) / static_cast<double>(n);
    const double k = vmf_kappa_mle(3, rbar);
    CHECK(std::fabs(k - concentration) < 0.2 * concentration);
    const double h6 = 8.0 * std::sinh(k) * std::sinh(k) /
                      (k * ((1.0 + 4.0 * k * k) * std::sinh(2.0 * k) - 2.0 * k * std::cosh(2.0 * k)) *
                       static_cast<double>(n));
    const double expected = std::pow(h6, -1.0 / 3.0);  // kappa = h^-2
    const double value = kappa_rule_of_thumb(data, 3);
    CHECK(std::fabs(value - expected) <= 1e-9 * expected);
  }
  CHECK_EQ(kappa_rule_of_thumb(std::vector<float>{}, 3), 0.0);
  CHECK_EQ(kappa_rule_of_thumb(std::vector<float>{1.0f}, 1), 0.0);
}

PRAG_TEST(rule_of_thumb_agrees_with_the_curvature_integral) {
  // For any q the rule is h^(4+q) = 1 / ( q (4 pi)^(q/2) R(Psi) n ) with, for a
  // vMF density f of concentration k and t = <x, mu>,
  //   Psi = f(t) ( -k t + k^2 (1 - t^2) / q ),   R(Psi) = area(S^{q-1}) int Psi^2 (1 - t^2)^(q/2 - 1) dt.
  // The integral is evaluated here by Simpson's rule in the polar angle.
  Rng rng(32);
  for (const std::size_t m : {std::size_t{2}, std::size_t{5}, std::size_t{8}, std::size_t{32}}) {
    const double q = static_cast<double>(m - 1);
    const std::vector<double> mu = uniform_direction(rng, m);
    const std::size_t n = 400;
    std::vector<float> data(n * m);
    for (std::size_t i = 0; i < n; ++i) sample_vmf(rng, mu.data(), m, 12.0, data.data() + i * m);
    std::vector<double> sum(m, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double norm = std::sqrt(dot(data.data() + i * m, data.data() + i * m, m));
      for (std::size_t j = 0; j < m; ++j) sum[j] += static_cast<double>(data[i * m + j]) / norm;
    }
    double norm2 = 0.0;
    for (const double v : sum) norm2 += v * v;
    const double k = vmf_kappa_mle(m, std::sqrt(norm2) / static_cast<double>(n));

    const int intervals = 200000;
    const double h = kPi / intervals;
    double integral = 0.0;
    for (int i = 0; i <= intervals; ++i) {
      const double angle = h * i;
      const double t = std::cos(angle);
      const double weight = (i == 0 || i == intervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
      const double g = -k * t + k * k * (1.0 - t * t) / q;
      integral += weight * std::exp(2.0 * k * (t - 1.0)) * g * g * std::pow(std::sin(angle), q - 1.0);
    }
    integral *= h / 3.0;
    const double log_equator = m == 2 ? std::log(2.0) : log_sphere_area(m - 1);
    const double log_r = 2.0 * (log_vmf_normalizer(m, k) + k) + log_equator + std::log(integral);
    const double log_h_power = -(std::log(q) + 0.5 * q * std::log(4.0 * kPi) + log_r + std::log(static_cast<double>(n)));
    const double expected = std::exp(-2.0 * log_h_power / (4.0 + q));
    const double value = kappa_rule_of_thumb(data, m);
    CHECK(std::fabs(value - expected) <= 1e-8 * expected);
  }
}

PRAG_TEST(select_kappa_is_close_to_the_best_held_out_likelihood) {
  const std::size_t m = 8;
  Rng rng(77);
  Mixture mixture = separated_mixture(rng, m, 3, 40.0);
  mixture.kappas = {25.0, 40.0, 70.0};
  mixture.weights = {0.5, 0.3, 0.2};
  const Sample train = sample_mixture(rng, mixture, 600, false);
  const Sample held_out = sample_mixture(rng, mixture, 600, false);

  OrderedExecutor executor(4);
  const KappaSelection selection = select_kappa(train.rows, m, executor, 16 * kKiB);
  CHECK(selection.kappa > 0.0);
  CHECK(selection.rule_of_thumb > 0.0);
  REQUIRE(selection.grid.size() >= 13u);
  for (std::size_t i = 1; i < selection.grid.size(); ++i) {
    CHECK(selection.grid[i].first > selection.grid[i - 1].first);
    CHECK_NEAR(selection.grid[i].first / selection.grid[i - 1].first, std::sqrt(2.0), 1e-9);
  }
  for (const auto& point : selection.grid) CHECK(selection.score >= point.second);
  // The score is the mean leave-one-out log-likelihood at the selected concentration.
  const std::vector<double> loo = log_density(train.rows, train.rows, m, selection.kappa, true, executor, 16 * kKiB);
  const double mean_loo = std::accumulate(loo.begin(), loo.end(), 0.0) / static_cast<double>(loo.size());
  CHECK_NEAR(selection.score, mean_loo, 1e-9 * std::fabs(mean_loo));

  const auto held_out_likelihood = [&](double kappa) {
    const std::vector<double> values = log_density(train.rows, held_out.rows, m, kappa, false, executor, 16 * kKiB);
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  };
  double best = -std::numeric_limits<double>::infinity();
  double best_kappa = 0.0;
  for (double kappa = 1.0; kappa < 3000.0; kappa *= 1.05) {
    const double value = held_out_likelihood(kappa);
    if (value > best) {
      best = value;
      best_kappa = kappa;
    }
  }
  const double selected = held_out_likelihood(selection.kappa);
  std::printf("  selected kappa %.2f (rule of thumb %.2f), held-out log-likelihood %.4f; best %.4f at kappa %.2f\n",
              selection.kappa, selection.rule_of_thumb, selected, best, best_kappa);
  CHECK(selected >= best - 0.02 * std::fabs(best));
  CHECK(selected <= best);

  // The selection does not depend on threads or block size.
  for (const unsigned threads : {1u, 7u}) {
    OrderedExecutor other(threads);
    for (const std::size_t block : {4 * kKiB, 64 * kKiB}) {
      const KappaSelection again = select_kappa(train.rows, m, other, block);
      CHECK_EQ(again.kappa, selection.kappa);
      CHECK_EQ(again.score, selection.score);
      CHECK(again.grid == selection.grid);
    }
  }
}

PRAG_TEST(select_kappa_is_not_driven_by_duplicated_rows) {
  const std::size_t m = 8;
  const Sample sample = planted_mixture(55, m, 3, 300, 40.0);
  OrderedExecutor executor(4);
  const KappaSelection clean = select_kappa(sample.rows, m, executor, 16 * kKiB);
  // A third of the rows appear twice.
  std::vector<float> doubled = sample.rows;
  doubled.insert(doubled.end(), sample.rows.begin(), sample.rows.begin() + 100 * m);
  const KappaSelection with_copies = select_kappa(doubled, m, executor, 16 * kKiB);
  CHECK(std::isfinite(with_copies.score));
  CHECK(with_copies.kappa < 2.0 * clean.kappa);
  CHECK(with_copies.kappa > 0.5 * clean.kappa);

  // Degenerate inputs return a usable concentration.
  std::vector<float> same(20 * m, 0.0f);
  for (std::size_t i = 0; i < 20; ++i) same[i * m] = 1.0f;
  const KappaSelection constant = select_kappa(same, m, executor, 16 * kKiB);
  CHECK(constant.kappa > 0.0);
  CHECK(std::isfinite(constant.kappa));
  const KappaSelection single = select_kappa(std::vector<float>(same.begin(), same.begin() + m), m, executor, 16 * kKiB);
  CHECK(single.kappa > 0.0);
  CHECK(std::isfinite(single.kappa));
}

PRAG_TEST(neighbourhood_kappa_is_where_the_neighbours_weigh_as_much_as_the_point) {
  const std::size_t m = 16;
  const Sample sample = planted_mixture(61, m, 5, 400, 80.0);
  const std::size_t n = 400;
  OrderedExecutor executor(4);
  const double kappa = neighbourhood_kappa(sample.rows, m, executor, 16 * kKiB);
  CHECK(kappa > 1.0);
  CHECK(kappa < 1000.0);

  const auto fraction_below_one = [&](double k) {
    std::size_t below = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double weight = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        if (j != i) weight += std::exp(k * (dot(sample.rows.data() + i * m, sample.rows.data() + j * m, m) - 1.0));
      }
      if (weight < 1.0) ++below;
    }
    return static_cast<double>(below) / static_cast<double>(n);
  };
  CHECK(fraction_below_one(kappa) <= 0.05);
  CHECK(fraction_below_one(1.01 * kappa) > 0.05);
  // A larger quantile tolerates more isolated points, hence a narrower kernel.
  CHECK(neighbourhood_kappa(sample.rows, m, executor, 16 * kKiB, 0.25) > kappa);

  for (const unsigned threads : {1u, 5u}) {
    OrderedExecutor other(threads);
    CHECK_EQ(neighbourhood_kappa(sample.rows, m, other, 4 * kKiB), kappa);
    CHECK_EQ(neighbourhood_kappa(sample.rows, m, other, 64 * kKiB), kappa);
  }
}

// ---- mode seeking ------------------------------------------------------------

PRAG_TEST(find_modes_recovers_a_planted_mixture) {
  const std::size_t m = 16;
  const std::size_t n = 400;
  const Sample sample = planted_mixture(101, m, 5, n, 80.0);
  OrderedExecutor executor(8);
  const MeanShiftOptions options;  // concentration chosen automatically
  const Modes modes = find_modes(sample.rows, m, options, executor);

  CHECK_EQ(modes.count(), std::size_t{5});
  CHECK_EQ(modes.m, m);
  CHECK(modes.kappa > 0.0);
  CHECK_EQ(modes.not_converged, std::size_t{0});
  CHECK_EQ(modes.density_decreases, std::size_t{0});
  const double ari = adjusted_rand_index(modes.assignment, sample.labels);
  const double nmi = normalized_mutual_information(modes.assignment, sample.labels);
  std::printf("  kappa %.3f, %zu modes, adjusted Rand index %.4f, normalised mutual information %.4f\n", modes.kappa,
              modes.count(), ari, nmi);
  CHECK(ari > 0.95);
  CHECK(nmi > 0.95);

  REQUIRE(modes.assignment.size() == n);
  REQUIRE(modes.log_density.size() == n);
  REQUIRE(modes.iterations.size() == n);
  REQUIRE(modes.centers.size() == modes.count() * m);
  REQUIRE(modes.center_log_density.size() == modes.count());
  std::vector<std::uint32_t> counted(modes.count(), 0);
  for (const std::uint32_t a : modes.assignment) {
    REQUIRE(a < modes.count());
    ++counted[a];
  }
  for (std::size_t k = 0; k < modes.count(); ++k) {
    CHECK_EQ(counted[k], modes.sizes[k]);
    if (k > 0) CHECK(modes.sizes[k] <= modes.sizes[k - 1]);
    CHECK_NEAR(dot(modes.centers.data() + k * m, modes.centers.data() + k * m, m), 1.0, 1e-6);
    const double expected = naive_log_density(sample.rows, m, modes.centers.data() + k * m, modes.kappa);
    CHECK_NEAR(modes.center_log_density[k], expected, 1e-4);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double expected = naive_log_density(sample.rows, m, sample.rows.data() + i * m, modes.kappa);
    CHECK_NEAR(modes.log_density[i], expected, 1e-4);
    // A mode is at least as dense as every point that climbs to it.
    CHECK(modes.center_log_density[modes.assignment[i]] >= modes.log_density[i] - 1e-5f);
  }
}

PRAG_TEST(density_never_decreases_along_a_trajectory) {
  const std::size_t m = 16;
  const std::size_t n = 400;
  const Sample sample = planted_mixture(101, m, 5, n, 80.0);
  OrderedExecutor executor(8);
  MeanShiftOptions options;
  options.kappa = 20.0;
  const Modes modes = find_modes(sample.rows, m, options, executor);
  CHECK_EQ(modes.density_decreases, std::size_t{0});
  CHECK_EQ(modes.count(), std::size_t{5});

  std::size_t steps = 0;
  for (std::size_t seed = 0; seed < n; ++seed) {
    const Trajectory path = mean_shift_trajectory(sample.rows, m, seed, options);
    REQUIRE(path.points.size() == path.log_density.size() * m);
    REQUIRE(!path.log_density.empty());
    CHECK(path.converged);
    CHECK_EQ(path.steps(), static_cast<std::size_t>(modes.iterations[seed]));
    steps += path.steps();
    double previous = -std::numeric_limits<double>::infinity();
    for (std::size_t s = 0; s < path.log_density.size(); ++s) {
      const double* y = path.points.data() + s * m;
      // Densities recomputed here from the recorded positions, independently of the kernel.
      const double density = naive_log_density(sample.rows, m, y, options.kappa);
      CHECK_NEAR(path.log_density[s], density, 1e-10 * std::fabs(density));
      CHECK(density >= previous - 1e-12 * std::fabs(previous));
      previous = density;
      double norm2 = 0.0;
      for (std::size_t j = 0; j < m; ++j) norm2 += y[j] * y[j];
      CHECK_NEAR(norm2, 1.0, 1e-12);
    }
    // The path starts at the data point and ends at the mode the point is assigned to.
    CHECK(dot(sample.rows.data() + seed * m, path.points.data(), m) > 1.0 - 1e-6);
    const double* end = path.points.data() + path.steps() * m;
    CHECK(dot(modes.centers.data() + modes.assignment[seed] * m, end, m) > options.merge_cosine);
    CHECK_NEAR(modes.log_density[seed], path.log_density.front(), 1e-4);
  }
  CHECK(steps > n);  // the points did move

  CHECK(mean_shift_trajectory(sample.rows, m, n, options).log_density.empty());
  MeanShiftOptions automatic;
  CHECK(mean_shift_trajectory(sample.rows, m, 0, automatic).log_density.empty());
}

PRAG_TEST(find_modes_is_identical_for_every_thread_count_and_block_size) {
  const std::size_t m = 16;
  const Sample sample = planted_mixture(101, m, 5, 400, 80.0);
  MeanShiftOptions options;
  options.kappa = 28.0;  // close to the concentration where modes start to split: the hard case
  OrderedExecutor reference_executor(1);
  const Modes reference = find_modes(sample.rows, m, options, reference_executor);
  CHECK(reference.count() >= 5u);
  for (const unsigned threads : {1u, 3u, 8u, 24u}) {
    OrderedExecutor executor(threads);
    for (const std::size_t block : {4 * kKiB, 16 * kKiB, 64 * kKiB}) {
      options.block_bytes = block;
      const Modes modes = find_modes(sample.rows, m, options, executor);
      CHECK(modes.assignment == reference.assignment);
      CHECK(modes.sizes == reference.sizes);
      CHECK(modes.iterations == reference.iterations);
      CHECK(same_bits(modes.centers, reference.centers));
      CHECK(same_bits(modes.log_density, reference.log_density));
      CHECK(same_bits(modes.center_log_density, reference.center_log_density));
      CHECK_EQ(modes.not_converged, reference.not_converged);
      CHECK_EQ(modes.density_decreases, std::size_t{0});
    }
  }
  // With the automatic concentration as well.
  MeanShiftOptions automatic;
  const Modes first = find_modes(sample.rows, m, automatic, reference_executor);
  OrderedExecutor many(24);
  automatic.block_bytes = 4 * kKiB;
  const Modes second = find_modes(sample.rows, m, automatic, many);
  CHECK_EQ(first.kappa, second.kappa);
  CHECK(first.assignment == second.assignment);
  CHECK(same_bits(first.centers, second.centers));
}

PRAG_TEST(the_likelihood_optimal_concentration_is_too_large_for_mode_seeking) {
  // Documents why MeanShiftOptions::limit_to_neighbourhood exists.
  const std::size_t m = 16;
  const Sample sample = planted_mixture(101, m, 5, 400, 80.0);
  OrderedExecutor executor(8);
  const double cross_validated = select_kappa(sample.rows, m, executor, 16 * kKiB).kappa;
  const double neighbourhood = neighbourhood_kappa(sample.rows, m, executor, 16 * kKiB);
  CHECK(cross_validated > 2.0 * neighbourhood);

  MeanShiftOptions options;
  options.limit_to_neighbourhood = false;
  const Modes raw = find_modes(sample.rows, m, options, executor);
  CHECK_EQ(raw.kappa, cross_validated);
  CHECK(raw.count() > 300u);  // nearly every point is a mode of its own
  CHECK_EQ(raw.density_decreases, std::size_t{0});

  options.limit_to_neighbourhood = true;
  const Modes limited = find_modes(sample.rows, m, options, executor);
  CHECK_EQ(limited.kappa, neighbourhood);
  CHECK_EQ(limited.count(), std::size_t{5});
  std::printf("  cross-validated kappa %.2f gives %zu modes; neighbourhood kappa %.2f gives %zu\n", cross_validated,
              raw.count(), neighbourhood, limited.count());
}

PRAG_TEST(small_groups_are_merged_into_the_nearest_mode) {
  const std::size_t m = 16;
  Rng rng(9);
  const Mixture mixture = separated_mixture(rng, m, 4, 120.0);
  Sample sample = sample_mixture(rng, mixture, 320, true);
  // Three stray points, each close to one component but outside its cloud.
  std::vector<std::uint32_t> nearest;
  for (std::size_t s = 0; s < 3; ++s) {
    const std::vector<double> away = uniform_direction(rng, m);
    std::vector<float> x(m);
    double norm2 = 0.0;
    for (std::size_t j = 0; j < m; ++j) {
      const double v = 0.75 * mixture.means[s][j] + 0.66 * away[j];
      x[j] = static_cast<float>(v);
      norm2 += v * v;
    }
    for (float& v : x) v = static_cast<float>(static_cast<double>(v) / std::sqrt(norm2));
    sample.rows.insert(sample.rows.end(), x.begin(), x.end());
    sample.labels.push_back(static_cast<std::uint32_t>(s));
    nearest.push_back(static_cast<std::uint32_t>(s));
  }
  const std::size_t n = sample.labels.size();
  OrderedExecutor executor(4);
  MeanShiftOptions options;
  options.kappa = 25.0;
  const Modes loose = find_modes(sample.rows, m, options, executor);
  CHECK(loose.count() > 4u);
  for (std::size_t k = 4; k < loose.count(); ++k) CHECK(loose.sizes[k] < 5u);

  options.min_group = 5;
  const Modes merged = find_modes(sample.rows, m, options, executor);
  CHECK_EQ(merged.count(), std::size_t{4});
  CHECK_EQ(std::accumulate(merged.sizes.begin(), merged.sizes.end(), 0u), static_cast<unsigned>(n));
  CHECK_NEAR(adjusted_rand_index(merged.assignment, sample.labels), 1.0, 1e-12);
  // Points that were in large groups keep their grouping.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; j += 37) {
      if (loose.sizes[loose.assignment[i]] >= 5 && loose.sizes[loose.assignment[j]] >= 5) {
        CHECK_EQ(loose.assignment[i] == loose.assignment[j], merged.assignment[i] == merged.assignment[j]);
      }
    }
  }

  // No group is large enough: everything joins the largest one.
  options.min_group = 1000;
  const Modes single = find_modes(sample.rows, m, options, executor);
  CHECK_EQ(single.count(), std::size_t{1});
  CHECK_EQ(single.sizes[0], static_cast<std::uint32_t>(n));
}

PRAG_TEST(modes_are_numbered_by_size) {
  const std::size_t m = 12;
  Rng rng(14);
  Mixture mixture = separated_mixture(rng, m, 3, 150.0);
  mixture.weights = {0.2, 0.5, 0.3};
  const Sample sample = sample_mixture(rng, mixture, 300, true);
  OrderedExecutor executor(4);
  MeanShiftOptions options;
  options.kappa = 25.0;
  const Modes modes = find_modes(sample.rows, m, options, executor);
  REQUIRE(modes.count() == 3u);
  CHECK_EQ(modes.sizes[0], 150u);
  CHECK_EQ(modes.sizes[1], 90u);
  CHECK_EQ(modes.sizes[2], 60u);
  // The largest mode is the component with weight one half, and so on.
  const std::uint32_t component_of_mode[3] = {1, 2, 0};
  for (std::size_t i = 0; i < 300; ++i) CHECK_EQ(sample.labels[i], component_of_mode[modes.assignment[i]]);
  for (std::size_t k = 0; k < 3; ++k) {
    const double cosine = dot(modes.centers.data() + k * m, mixture.means[component_of_mode[k]].data(), m);
    CHECK(cosine > 0.99);
  }
}

PRAG_TEST(layout_lists_points_mode_by_mode) {
  const std::size_t m = 16;
  const Sample sample = planted_mixture(33, m, 5, 300, 90.0);
  OrderedExecutor executor(4);
  MeanShiftOptions options;
  options.kappa = 20.0;
  const Modes modes = find_modes(sample.rows, m, options, executor);
  const std::vector<std::uint32_t> order = layout_by_mode(modes);
  REQUIRE(order.size() == 300u);
  std::vector<std::uint32_t> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) CHECK_EQ(sorted[i], static_cast<std::uint32_t>(i));
  for (std::size_t i = 1; i < order.size(); ++i) {
    const std::uint32_t a = order[i - 1];
    const std::uint32_t b = order[i];
    CHECK(modes.assignment[a] <= modes.assignment[b]);
    if (modes.assignment[a] == modes.assignment[b]) CHECK(modes.log_density[a] >= modes.log_density[b]);
  }
  CHECK(layout_by_mode(Modes{}).empty());
}

PRAG_TEST(find_modes_handles_degenerate_inputs) {
  OrderedExecutor executor(3);
  MeanShiftOptions options;
  options.kappa = 10.0;
  const Modes empty = find_modes(std::vector<float>{}, 8, options, executor);
  CHECK_EQ(empty.count(), std::size_t{0});
  CHECK(empty.assignment.empty());

  const std::vector<float> one = {0.6f, 0.8f, 0.0f};
  const Modes single = find_modes(one, 3, options, executor);
  REQUIRE(single.count() == 1u);
  CHECK_EQ(single.sizes[0], 1u);
  CHECK_EQ(single.not_converged, std::size_t{0});
  CHECK_NEAR(single.centers[0], 0.6, 1e-6);

  // The automatic concentration is defined for one and for two points as well.
  const Modes lonely = find_modes(one, 3, MeanShiftOptions{}, executor);
  CHECK_EQ(lonely.count(), std::size_t{1});
  CHECK(lonely.kappa > 0.0);
  CHECK(std::isfinite(lonely.kappa));
  const std::vector<float> pair = {0.6f, 0.8f, 0.0f, 0.0f, 0.6f, 0.8f};
  const Modes two = find_modes(pair, 3, MeanShiftOptions{}, executor);
  CHECK(two.count() >= 1u);
  CHECK(two.count() <= 2u);
  CHECK(two.kappa > 0.0);
  CHECK(std::isfinite(two.kappa));
  CHECK_EQ(two.not_converged, std::size_t{0});

  // Identical points form one mode; a zero row is tolerated.
  std::vector<float> copies;
  for (int i = 0; i < 40; ++i) copies.insert(copies.end(), one.begin(), one.end());
  copies.insert(copies.end(), {0.0f, 0.0f, 0.0f});
  const Modes same = find_modes(copies, 3, MeanShiftOptions{}, executor);
  CHECK(same.count() <= 2u);
  CHECK(same.sizes[0] >= 40u);
  CHECK_EQ(same.not_converged, std::size_t{0});

  // Without iterations every point stays where it is.
  const Sample sample = planted_mixture(1, 6, 2, 50, 50.0);
  options.max_iterations = 0;
  const Modes frozen = find_modes(sample.rows, 6, options, executor);
  CHECK_EQ(frozen.not_converged, std::size_t{50});
  for (const std::uint16_t steps : frozen.iterations) CHECK_EQ(steps, std::uint16_t{0});
}

// ---- agreement between labelings ---------------------------------------------

PRAG_TEST(agreement_matches_hand_computed_values) {
  using Labels = std::vector<std::uint32_t>;
  // Identical up to the names of the labels.
  const Labels a = {0, 0, 1, 1, 2, 2};
  const Labels renamed = {5, 5, 9, 9, 7, 7};
  CHECK_NEAR(normalized_mutual_information(a, renamed), 1.0, 1e-12);
  CHECK_NEAR(adjusted_rand_index(a, renamed), 1.0, 1e-12);

  // Independent: every cell of the 2 x 2 table holds one point.
  //   I = 0;  pairs together 0, expected 2 * 2 / 6, maximum 2:  ARI = (0 - 2/3) / (2 - 2/3) = -1/2.
  const Labels rows = {0, 0, 1, 1};
  const Labels columns = {0, 1, 0, 1};
  CHECK_NEAR(normalized_mutual_information(rows, columns), 0.0, 1e-12);
  CHECK_NEAR(adjusted_rand_index(rows, columns), -0.5, 1e-12);

  // A single cluster against two: no information, agreement at chance level.
  const Labels one = {3, 3, 3, 3};
  const Labels two = {0, 0, 1, 1};
  CHECK_NEAR(normalized_mutual_information(one, two), 0.0, 1e-12);
  CHECK_NEAR(adjusted_rand_index(one, two), 0.0, 1e-12);
  CHECK_NEAR(normalized_mutual_information(one, one), 1.0, 1e-12);
  CHECK_NEAR(adjusted_rand_index(one, one), 1.0, 1e-12);
  // All singletons against all singletons.
  const Labels singles = {4, 8, 15, 16};
  CHECK_NEAR(adjusted_rand_index(singles, singles), 1.0, 1e-12);
  CHECK_NEAR(normalized_mutual_information(singles, singles), 1.0, 1e-12);
  CHECK_NEAR(adjusted_rand_index(singles, one), 0.0, 1e-12);

  // Partial agreement. Table: {2, 1, 0; 0, 1, 2}.
  //   H(a) = log 2, H(b) = log 3, H(a, b) = (2/3) log 3 + (1/3) log 6, I = (2/3) log 2,
  //   NMI = 2 I / (H(a) + H(b)) = (4/3) log 2 / log 6.
  //   Pairs together 2, in a: 6, in b: 3, all: 15; expected 1.2, maximum 4.5: ARI = 0.8 / 3.3.
  const Labels p = {0, 0, 0, 1, 1, 1};
  const Labels q = {0, 0, 1, 1, 2, 2};
  CHECK_NEAR(normalized_mutual_information(p, q), 4.0 / 3.0 * std::log(2.0) / std::log(6.0), 1e-12);
  CHECK_NEAR(adjusted_rand_index(p, q), 0.8 / 3.3, 1e-12);
  CHECK_NEAR(adjusted_rand_index(q, p), 0.8 / 3.3, 1e-12);
  CHECK_NEAR(normalized_mutual_information(q, p), normalized_mutual_information(p, q), 1e-15);

  // {0, 0, 1, 2} against {0, 0, 1, 1}: pairs together 1, in a: 1, in b: 2, all: 6;
  // ARI = (1 - 1/3) / (3/2 - 1/3) = 4/7.
  CHECK_NEAR(adjusted_rand_index(Labels{0, 0, 1, 2}, Labels{0, 0, 1, 1}), 4.0 / 7.0, 1e-12);

  CHECK_EQ(normalized_mutual_information(Labels{}, Labels{}), 0.0);
  CHECK_EQ(adjusted_rand_index(Labels{0, 1}, Labels{0, 1, 2}), 0.0);
}

// ---- diagnostics ----------------------------------------------------------------

PRAG_TEST(mean_pairwise_cosine_measures_the_common_direction) {
  Rng rng(41);
  const std::size_t d = 256;
  const std::size_t n = 600;
  std::vector<float> isotropic(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    const std::vector<double> x = uniform_direction(rng, d);
    for (std::size_t t = 0; t < d; ++t) isotropic[i * d + t] = static_cast<float>(x[t]);
  }
  CHECK(std::fabs(mean_pairwise_cosine(isotropic, d)) < 0.005);

  // x = c u + g with |u| = |g| = 1: for large d the cosine of two such points is
  // c^2 / (c^2 + 1), here 0.64 / 1.64 = 0.390.
  const std::vector<double> u = uniform_direction(rng, d);
  const double c = 0.8;
  std::vector<float> shifted(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    const std::vector<double> g = uniform_direction(rng, d);
    for (std::size_t t = 0; t < d; ++t) shifted[i * d + t] = static_cast<float>(c * u[t] + g[t]);  // not normalised
  }
  CHECK_NEAR(mean_pairwise_cosine(shifted, d), c * c / (c * c + 1.0), 0.01);

  // Against the definition, on a sample small enough for the double loop.
  const std::size_t few = 40;
  double sum = 0.0;
  for (std::size_t i = 0; i < few; ++i) {
    for (std::size_t j = 0; j < few; ++j) {
      if (i == j) continue;
      const float* a = shifted.data() + i * d;
      const float* b = shifted.data() + j * d;
      sum += dot(a, b, d) / std::sqrt(dot(a, a, d) * dot(b, b, d));
    }
  }
  const std::vector<float> head(shifted.begin(), shifted.begin() + static_cast<std::ptrdiff_t>(few * d));
  CHECK_NEAR(mean_pairwise_cosine(head, d), sum / static_cast<double>(few * (few - 1)), 1e-12);

  CHECK_EQ(mean_pairwise_cosine(std::vector<float>{}, d), 0.0);
  CHECK_EQ(mean_pairwise_cosine(std::vector<float>(d, 1.0f), d), 0.0);
}

PRAG_TEST(hubness_grows_with_dimension_and_anisotropy) {
  Rng rng(43);
  const std::size_t n = 500;
  // Low dimension, isotropic.
  std::vector<float> low(n * 3);
  for (std::size_t i = 0; i < n; ++i) {
    const std::vector<double> x = uniform_direction(rng, 3);
    for (std::size_t t = 0; t < 3; ++t) low[i * 3 + t] = static_cast<float>(x[t]);
  }
  // High dimension, a common direction plus noise of varying length: points
  // with little noise are close to the mean direction, hence to everybody.
  const std::size_t d = 128;
  const std::vector<double> u = uniform_direction(rng, d);
  std::vector<float> high(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      high[i * d + t] = static_cast<float>(u[t] + rng.normal() / std::sqrt(static_cast<double>(d)));
    }
  }
  const double low_skew = hubness_skewness(low, 3, 10);
  const double high_skew = hubness_skewness(high, d, 10);
  std::printf("  skewness of N_10: %.3f (3 dimensions, isotropic), %.3f (128 dimensions, anisotropic)\n", low_skew,
              high_skew);
  CHECK(high_skew > low_skew + 0.5);
  CHECK(high_skew > 1.0);
  CHECK(std::fabs(low_skew) < 0.5);

  // Against the definition.
  const std::size_t few = 60;
  const std::size_t k = 5;
  const std::vector<float> head(high.begin(), high.begin() + static_cast<std::ptrdiff_t>(few * d));
  std::vector<double> occurrences(few, 0.0);
  for (std::size_t i = 0; i < few; ++i) {
    std::vector<std::pair<double, std::size_t>> others;
    for (std::size_t j = 0; j < few; ++j) {
      if (j == i) continue;
      const float* a = head.data() + i * d;
      const float* b = head.data() + j * d;
      others.emplace_back(-dot(a, b, d) / std::sqrt(dot(a, a, d) * dot(b, b, d)), j);
    }
    std::sort(others.begin(), others.end());
    for (std::size_t s = 0; s < k; ++s) occurrences[others[s].second] += 1.0;
  }
  double m2 = 0.0;
  double m3 = 0.0;
  for (const double v : occurrences) {
    m2 += (v - static_cast<double>(k)) * (v - static_cast<double>(k));
    m3 += std::pow(v - static_cast<double>(k), 3);
  }
  m2 /= static_cast<double>(few);
  m3 /= static_cast<double>(few);
  CHECK_NEAR(hubness_skewness(head, d, k), m3 / std::pow(m2, 1.5), 1e-9);

  // Fewer points than neighbours asked for: every point is a neighbour of all the others.
  CHECK_EQ(hubness_skewness(std::vector<float>(low.begin(), low.begin() + 9 * 3), 3, 10), 0.0);
  CHECK_EQ(hubness_skewness(std::vector<float>{}, 3, 10), 0.0);
}
