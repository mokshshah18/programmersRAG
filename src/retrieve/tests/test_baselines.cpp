#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "prag/graph/call_graph.hpp"
#include "prag/retrieve/baselines.hpp"
#include "prag/retrieve/calibration.hpp"
#include "sampling.hpp"

using namespace prag;
using namespace prag::retrieve;
using graph::CallGraph;

namespace {

using Edges = std::vector<std::pair<UnitId, UnitId>>;

const Calibration& logistic() {
  static const Calibration calibration = Calibration::fallback(0.0, 1.0);
  return calibration;
}

std::vector<UnitId> ids(const std::vector<TracedUnit>& units) {
  std::vector<UnitId> out;
  for (const TracedUnit& unit : units) out.push_back(unit.id);
  return out;
}

std::vector<TracedUnit> cut(const std::vector<float>& scores, std::size_t k_max, std::size_t buffer = 0) {
  return largest_gap_cutoff(scores, 0.0, 1.0, logistic(), k_max, buffer);
}

double sum(const std::vector<double>& v) {
  double total = 0.0;
  for (const double x : v) total += x;
  return total;
}

double l1_distance(const std::vector<double>& a, const std::vector<double>& b) {
  double total = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) total += std::fabs(a[i] - b[i]);
  return total;
}

using Matrix3 = std::array<std::array<double, 3>, 3>;

// Solves M x = b by Gaussian elimination with partial pivoting.
std::array<double, 3> solve(Matrix3 m, std::array<double, 3> b) {
  for (std::size_t column = 0; column < 3; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < 3; ++row) {
      if (std::fabs(m[row][column]) > std::fabs(m[pivot][column])) pivot = row;
    }
    std::swap(m[column], m[pivot]);
    std::swap(b[column], b[pivot]);
    for (std::size_t row = column + 1; row < 3; ++row) {
      const double factor = m[row][column] / m[column][column];
      for (std::size_t k = column; k < 3; ++k) m[row][k] -= factor * m[column][k];
      b[row] -= factor * b[column];
    }
  }
  std::array<double, 3> x{};
  for (std::size_t row = 3; row-- > 0;) {
    double value = b[row];
    for (std::size_t k = row + 1; k < 3; ++k) value -= m[row][k] * x[k];
    x[row] = value / m[row][row];
  }
  return x;
}

// (1 - a) (I - a W^T)^{-1} p for a row-stochastic W.
std::array<double, 3> closed_form(const Matrix3& w, const std::array<double, 3>& p, double a) {
  Matrix3 m{};
  std::array<double, 3> b{};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) m[i][j] = (i == j ? 1.0 : 0.0) - a * w[j][i];
    b[i] = (1.0 - a) * p[i];
  }
  return solve(m, b);
}

std::vector<double> normal_sample(std::size_t n, double mean, double sd, std::mt19937_64& rng) {
  std::vector<double> out(n);
  for (double& x : out) x = test::normal(rng, mean, sd);
  return out;
}

// Measurements run in full when the environment variable PRAG_SLOW_TESTS is 1.
bool slow_tests() {
  const char* value = std::getenv("PRAG_SLOW_TESTS");
  return value != nullptr && std::string(value) == "1";
}

double coverage(const std::vector<double>& fresh, double threshold) {
  std::size_t kept = 0;
  for (const double z : fresh) kept += z >= threshold ? 1 : 0;
  return static_cast<double>(kept) / static_cast<double>(fresh.size());
}

}  // namespace

// ---- largest gap ---------------------------------------------------------------

PRAG_TEST(largest_gap_on_constructed_rankings) {
  // Sorted: 1, 0.875, 0.75 | 0.25, 0.125, 0 | -0.5. Gaps 1/8, 1/8, 1/2, 1/8, 1/8, 1/2.
  const std::vector<float> scores = {0.25f, 1.0f, -0.5f, 0.75f, 0.0f, 0.875f, 0.125f};
  CHECK(ids(cut(scores, 10)) == (std::vector<UnitId>{1, 5, 3}));  // of two equal gaps the first
  CHECK(ids(cut(scores, 3)) == (std::vector<UnitId>{1, 5, 3}));
  CHECK(ids(cut(scores, 2)) == (std::vector<UnitId>{1}));         // gaps 1 and 2 are equal
  CHECK(ids(cut(scores, 1)) == (std::vector<UnitId>{1}));
  CHECK(cut(scores, 0).empty());

  // The buffer extends the cut and is limited by the corpus only.
  CHECK(ids(cut(scores, 10, 2)) == (std::vector<UnitId>{1, 5, 3, 0, 6}));
  CHECK(ids(cut(scores, 2, 1)) == (std::vector<UnitId>{1, 5}));
  CHECK_EQ(cut(scores, 10, 100).size(), scores.size());
  CHECK_EQ(cut(scores, 10, std::numeric_limits<std::size_t>::max()).size(), scores.size());

  // A gap near the end wins when it is the largest within k_max.
  const std::vector<float> late = {9.0f, 8.0f, 7.0f, 6.0f, 1.0f};
  CHECK_EQ(cut(late, 10).size(), std::size_t{4});
  CHECK_EQ(cut(late, 3).size(), std::size_t{1});  // all gaps within the top 3 are equal

  // The output has the form of top_k.
  const std::vector<TracedUnit> units = cut(scores, 10);
  CHECK_EQ(units[0].score, 1.0f);
  CHECK_NEAR(units[0].p_score, 1.0 / (1.0 + std::exp(-1.0)), 1e-7);
  CHECK_EQ(units[0].p, units[0].p_score);
  CHECK(units[0].via == EdgeKind::Seed);
  CHECK_EQ(units[0].parent, kNoUnit);
}

PRAG_TEST(largest_gap_edge_cases) {
  CHECK(cut({}, 5).empty());
  CHECK(ids(cut({0.3f}, 5)) == (std::vector<UnitId>{0}));
  // Equal scores: every gap is zero, the cut falls after the first unit.
  CHECK(ids(cut({0.5f, 0.5f, 0.5f}, 5)) == (std::vector<UnitId>{0}));
  // Undefined scores are left out of the ranking.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(ids(cut({nan, 0.2f, nan, 0.9f, 0.8f}, 5)) == (std::vector<UnitId>{3, 4}));
  CHECK(cut({nan, nan}, 5).empty());
}

// ---- personalised PageRank -----------------------------------------------------

PRAG_TEST(pagerank_matches_the_closed_form_solved_by_hand) {
  // 0 -> 1, 0 -> 2, 1 -> 2, forward steps only, restart at unit 0, alpha = 1/2.
  // Rows of W: (0, 1/2, 1/2), (0, 0, 1), and (1, 0, 0) for unit 2, which has no
  // callee and returns its mass to the seed. r = (1 - a) p + a W^T r reads
  //     r0 = 1/2 + r2 / 2,   r1 = r0 / 4,   r2 = r0 / 4 + r1 / 2 = 3 r0 / 8,
  // hence r0 = 8/13, r1 = 2/13, r2 = 3/13.
  const CallGraph calls = CallGraph::from_edges(3, Edges{{0, 1}, {0, 2}, {1, 2}});
  PageRankOptions options;
  options.continuation = 0.5;
  options.callee_weight = 1.0;
  options.caller_weight = 0.0;
  options.tolerance = 1e-14;
  options.max_iterations = 200;
  PageRankStats stats;
  const std::vector<double> seeds = {1.0, 0.0, 0.0};
  const std::vector<double> rank = personalized_pagerank(calls, seeds, options, &stats);
  REQUIRE(rank.size() == 3u);
  CHECK(stats.converged);
  CHECK_NEAR(rank[0], 8.0 / 13.0, 1e-12);
  CHECK_NEAR(rank[1], 2.0 / 13.0, 1e-12);
  CHECK_NEAR(rank[2], 3.0 / 13.0, 1e-12);
  CHECK_NEAR(sum(rank), 1.0, 1e-15);

  const Matrix3 w = {{{0.0, 0.5, 0.5}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}}};
  const std::array<double, 3> solved = closed_form(w, {1.0, 0.0, 0.0}, 0.5);
  for (std::size_t i = 0; i < 3; ++i) CHECK_NEAR(rank[i], solved[i], 1e-12);
}

PRAG_TEST(pagerank_weights_both_directions) {
  // The same graph with backward steps of weight 1/2 and a spread-out seed.
  // Unit 0: callees 1, 2.            Row (0, 1/2, 1/2).
  // Unit 1: callee 2, caller 0.      Weights 1 and 1/2: row (1/3, 0, 2/3).
  // Unit 2: callers 0, 1.            Row (1/2, 1/2, 0).
  const CallGraph calls = CallGraph::from_edges(3, Edges{{0, 1}, {0, 2}, {1, 2}});
  const Matrix3 w = {{{0.0, 0.5, 0.5}, {1.0 / 3.0, 0.0, 2.0 / 3.0}, {0.5, 0.5, 0.0}}};
  // Unscaled seeds: the restart distribution is (0.5, 0.3, 0.2).
  const std::vector<double> seeds = {5.0, 3.0, 2.0};
  for (const double alpha : {0.5, 0.85, 0.1}) {
    PageRankOptions options;
    options.continuation = alpha;
    options.callee_weight = 1.0;
    options.caller_weight = 0.5;
    options.tolerance = 1e-14;
    options.max_iterations = 1000;
    PageRankStats stats;
    const std::vector<double> rank = personalized_pagerank(calls, seeds, options, &stats);
    const std::array<double, 3> solved = closed_form(w, {0.5, 0.3, 0.2}, alpha);
    CHECK(stats.converged);
    for (std::size_t i = 0; i < 3; ++i) CHECK_NEAR(rank[i], solved[i], 1e-12);
    CHECK_NEAR(sum(rank), 1.0, 1e-15);
  }
  // Only the ratio of the weights matters.
  PageRankOptions a;
  a.callee_weight = 1.0;
  a.caller_weight = 0.5;
  PageRankOptions b;
  b.callee_weight = 4.0;
  b.caller_weight = 2.0;
  const std::vector<double> rank_a = personalized_pagerank(calls, seeds, a);
  const std::vector<double> rank_b = personalized_pagerank(calls, seeds, b);
  CHECK(l1_distance(rank_a, rank_b) < 1e-15);
}

PRAG_TEST(pagerank_sums_to_one_and_converges) {
  std::mt19937_64 rng(77);
  const std::size_t n = 400;
  Edges edges;
  for (int e = 0; e < 1200; ++e) {
    // Edges point from lower to higher ids: a graph with many units that call
    // nothing, and with forward steps only these have no way out.
    const UnitId a = static_cast<UnitId>(rng() % n);
    const UnitId b = static_cast<UnitId>(rng() % n);
    if (a != b) edges.emplace_back(std::min(a, b), std::max(a, b));
  }
  // Units 390..399 stay isolated.
  std::erase_if(edges, [](const auto& edge) { return edge.first >= 390 || edge.second >= 390; });
  const CallGraph calls = CallGraph::from_edges(n, edges);
  std::vector<double> seeds(n, 0.0);
  seeds[3] = 2.0;
  seeds[17] = 1.0;
  seeds[250] = 1.0;

  for (const double backward : {0.0, 1.0}) {
    PageRankOptions options;  // alpha 0.5, tolerance 1e-6
    options.caller_weight = backward;
    PageRankStats stats;
    const std::vector<double> rank = personalized_pagerank(calls, seeds, options, &stats);
    REQUIRE(rank.size() == n);
    CHECK(stats.converged);
    CHECK(stats.change < 1e-6);
    // Contraction by alpha per iteration: ln(2 / 1e-6) / ln 2 = 20.9 iterations suffice.
    CHECK(stats.iterations <= 22u);
    CHECK_NEAR(sum(rank), 1.0, 1e-12);
    for (const double r : rank) CHECK(r >= 0.0);
    CHECK(rank[3] > rank[17]);
    CHECK(rank[3] >= 0.5 * 0.5);  // at least the restart mass of the seed
    for (std::size_t i = 390; i < n; ++i) CHECK_EQ(rank[i], 0.0);  // isolated and not a seed

    // Geometric convergence: the distance to the fixed point shrinks at least
    // by alpha with every iteration.
    options.tolerance = 0.0;
    options.max_iterations = 200;
    const std::vector<double> limit = personalized_pagerank(calls, seeds, options);
    // Stopping at a change below the tolerance leaves at most
    // alpha / (1 - alpha) times the tolerance to the fixed point.
    CHECK(l1_distance(rank, limit) <= 1e-6);
    double previous = 2.0;
    for (std::size_t k = 1; k <= 12; ++k) {
      options.max_iterations = k;
      PageRankStats partial;
      const std::vector<double> after = personalized_pagerank(calls, seeds, options, &partial);
      CHECK_EQ(partial.iterations, k);
      CHECK(!partial.converged);
      const double distance = l1_distance(after, limit);
      CHECK(distance <= 0.5 * previous + 1e-12);
      CHECK(distance <= 2.0 * std::pow(0.5, static_cast<double>(k)) + 1e-12);
      previous = distance;
    }
  }
}

PRAG_TEST(pagerank_edge_cases) {
  const CallGraph calls = CallGraph::from_edges(4, Edges{{0, 1}, {1, 2}, {2, 3}});
  CHECK(personalized_pagerank(calls, {}).empty());

  // Without positive seed the restart is uniform; negative and undefined
  // entries count as zero.
  const std::vector<double> zero(4, 0.0);
  const std::vector<double> uniform = personalized_pagerank(calls, zero);
  CHECK_NEAR(sum(uniform), 1.0, 1e-12);
  for (const double r : uniform) CHECK(r > 0.0);
  const std::vector<double> dirty = {-3.0, std::numeric_limits<double>::quiet_NaN(), 2.0, 0.0};
  const std::vector<double> clean = {0.0, 0.0, 1.0, 0.0};
  CHECK(l1_distance(personalized_pagerank(calls, dirty), personalized_pagerank(calls, clean)) < 1e-15);

  // alpha = 0: the walk never leaves the seeds.
  PageRankOptions stay;
  stay.continuation = 0.0;
  const std::vector<double> seeds = {0.25, 0.75, 0.0, 0.0};
  CHECK(l1_distance(personalized_pagerank(calls, seeds, stay), seeds) < 1e-15);

  // No iteration: the seed distribution.
  PageRankOptions none;
  none.max_iterations = 0;
  PageRankStats stats;
  CHECK(l1_distance(personalized_pagerank(calls, seeds, none, &stats), seeds) < 1e-15);
  CHECK_EQ(stats.iterations, std::size_t{0});
  CHECK(!stats.converged);

  // Fewer seeds than units in the graph: the graph is cut to the seeds, and
  // unit 1 loses its callee.
  const std::vector<double> two = {1.0, 0.0};
  PageRankOptions forward;
  forward.caller_weight = 0.0;
  forward.tolerance = 1e-14;
  forward.max_iterations = 200;
  const std::vector<double> rank = personalized_pagerank(calls, two, forward);
  REQUIRE(rank.size() == 2u);
  // r0 = 1/2 + r1 / 2, r1 = r0 / 2: r0 = 2/3.
  CHECK_NEAR(rank[0], 2.0 / 3.0, 1e-12);
  CHECK_NEAR(rank[1], 1.0 / 3.0, 1e-12);

  // The result does not depend on anything but the arguments.
  CHECK(personalized_pagerank(calls, seeds) == personalized_pagerank(calls, seeds));
}

PRAG_TEST(top_k_by_rank_orders_by_rank) {
  const std::vector<double> rank = {0.1, 0.4, 0.1, 0.4, 0.0};
  const std::vector<float> scores = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<TracedUnit> top = top_k_by_rank(rank, scores, 0.0, 1.0, logistic(), 3);
  CHECK(ids(top) == (std::vector<UnitId>{1, 3, 0}));
  CHECK_EQ(top[0].score, 2.0f);
  CHECK_NEAR(top[0].p_score, 1.0 / (1.0 + std::exp(-2.0)), 1e-7);
  CHECK_EQ(top_k_by_rank(rank, scores, 0.0, 1.0, logistic(), 10).size(), std::size_t{5});
  CHECK(top_k_by_rank(rank, scores, 0.0, 1.0, logistic(), 0).empty());
}

// ---- conformal threshold -------------------------------------------------------

PRAG_TEST(conformal_threshold_hand_examples) {
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> z = {7, 1, 9, 3, 5, 2, 8, 4, 6};  // n = 9
  // k = ceil(10 (1 - alpha)); the threshold is the k-th largest score.
  CHECK_EQ(conformal_threshold(z, 0.5), 5.0);   // k = 5
  CHECK_EQ(conformal_threshold(z, 0.2), 2.0);   // k = 8
  CHECK_EQ(conformal_threshold(z, 0.25), 2.0);  // k = ceil(7.5) = 8
  CHECK_EQ(conformal_threshold(z, 0.1), 1.0);   // k = 9: 10 * 0.9 must not round up to 10
  CHECK_EQ(conformal_threshold(z, 0.3), 3.0);   // k = 7: 10 * 0.7 must not round up to 8
  CHECK_EQ(conformal_threshold(z, 0.91), 9.0);  // k = 1
  CHECK(conformal_threshold(z, 0.05) == -inf);  // k = 10 > n
  CHECK(conformal_threshold(z, 0.0) == -inf);
  CHECK(conformal_threshold(z, -1.0) == -inf);
  CHECK(conformal_threshold(z, 1.0) == inf);
  CHECK_EQ(conformal_threshold(z, 0.95), 9.0);  // k = ceil(0.5) = 1
  CHECK(conformal_threshold(z, 1.0 - 1e-12) == inf);  // k = 0: nothing has to be kept
  CHECK(conformal_threshold({}, 0.1) == -inf);
  CHECK(conformal_threshold(z, std::numeric_limits<double>::quiet_NaN()) == -inf);

  // Ties and undefined scores.
  const std::vector<double> tied = {2, 2, 2, 5, 5, std::numeric_limits<double>::quiet_NaN(), inf};
  CHECK_EQ(conformal_threshold(tied, 0.5), 2.0);  // n = 5, k = 3: the third largest is 2
  CHECK_EQ(conformal_threshold(tied, 0.6), 2.0);  // k = ceil(2.4) = 3
  CHECK_EQ(conformal_threshold(tied, 0.7), 5.0);  // k = ceil(1.8) = 2
}

PRAG_TEST(conformal_threshold_is_the_largest_admissible_threshold) {
  std::mt19937_64 rng(101);
  for (int trial = 0; trial < 300; ++trial) {
    const std::size_t n = 1 + static_cast<std::size_t>(rng() % 200);
    std::vector<double> z(n);
    for (double& v : z) v = std::round(10.0 * test::normal(rng, 0.0, 3.0)) / 10.0;  // with ties
    const double alpha = test::uniform(rng, 0.01, 0.6);
    const double needed = std::ceil(static_cast<double>(n + 1) * (1.0 - alpha) - 1e-9);
    const double t = conformal_threshold(z, alpha);
    if (needed > static_cast<double>(n)) {
      CHECK(t == -std::numeric_limits<double>::infinity());
      continue;
    }
    const auto at_least = [&](double threshold) {
      return static_cast<double>(std::count_if(z.begin(), z.end(), [&](double v) { return v >= threshold; }));
    };
    CHECK(at_least(t) >= needed);
    // Any larger threshold keeps fewer than k scores: it is enough to try the
    // next larger score.
    double next = std::numeric_limits<double>::infinity();
    for (const double v : z) {
      if (v > t) next = std::min(next, v);
    }
    if (std::isfinite(next)) CHECK(at_least(next) < needed);
    CHECK(std::find(z.begin(), z.end(), t) != z.end());
  }
}

PRAG_TEST(conformal_threshold_covers_fresh_relevant_scores) {
  std::mt19937_64 rng(2026);
  const std::vector<double> calibration = normal_sample(5000, 5.0, 1.0, rng);
  const std::vector<double> fresh = normal_sample(2000, 5.0, 1.0, rng);
  for (const double alpha : {0.05, 0.1, 0.2}) {
    const double t = conformal_threshold(calibration, alpha);
    const double covered = coverage(fresh, t);
    CHECK(covered >= 1.0 - alpha - 0.02);
    // Not vacuous: the threshold is near the alpha quantile of N(5, 1), and the
    // coverage is not far above the target either.
    CHECK(covered <= 1.0 - alpha + 0.03);
    CHECK(std::isfinite(t));
  }
  CHECK(conformal_threshold(calibration, 0.05) < conformal_threshold(calibration, 0.1));
  CHECK(conformal_threshold(calibration, 0.1) < conformal_threshold(calibration, 0.2));
  CHECK_NEAR(conformal_threshold(calibration, 0.1), 5.0 - 1.2816, 0.06);
}

// Coverage over many calibration and test draws: the guarantee is on the
// average, single draws scatter around it. PRAG_SLOW_TESTS=1 prints the numbers.
PRAG_TEST(conformal_coverage_over_many_draws) {
  const bool slow = slow_tests();
  const int draws = slow ? 1000 : 40;
  std::mt19937_64 rng(4711);
  for (const std::size_t n : {std::size_t{100}, std::size_t{5000}}) {
    for (const double alpha : {0.05, 0.1, 0.2}) {
      double mean = 0.0;
      double lowest = 1.0;
      int within = 0;
      for (int d = 0; d < draws; ++d) {
        const std::vector<double> calibration = normal_sample(n, 5.0, 1.0, rng);
        const std::vector<double> fresh = normal_sample(2000, 5.0, 1.0, rng);
        const double covered = coverage(fresh, conformal_threshold(calibration, alpha));
        mean += covered / draws;
        lowest = std::min(lowest, covered);
        within += covered >= 1.0 - alpha - 0.02 ? 1 : 0;
      }
      // The average coverage is k / (n + 1) >= 1 - alpha; allow three standard
      // errors of the mean.
      const double deviation = std::sqrt(alpha * (1.0 - alpha) * (1.0 / static_cast<double>(n) + 1.0 / 2000.0));
      CHECK(mean >= 1.0 - alpha - 3.0 * deviation / std::sqrt(static_cast<double>(draws)));
      if (slow) {
        std::printf("  conformal, n = %zu, alpha = %.2f: mean coverage %.4f, lowest %.4f, ", n, alpha, mean, lowest);
        std::printf(">= 1 - alpha - 0.02 in %d of %d draws\n", within, draws);
      }
    }
  }
}
