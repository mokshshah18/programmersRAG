// Directional mean shift, mode merging and the storage layout derived from it.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "blocking.hpp"
#include "kernels.hpp"
#include "prag/kde/directional.hpp"
#include "prag/kde/vmf.hpp"

namespace prag::kde {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

enum class Phase : std::uint8_t {
  Moving,   // takes another step after the next sweep
  Landing,  // has stopped; the next sweep evaluates the density where it stands
  Done,
};

struct Problem {
  const float* data = nullptr;
  std::size_t n = 0;
  std::size_t m = 0;
  double kappa = 0.0;
  double tolerance = 0.0;
  std::uint32_t max_iterations = 0;
  const std::vector<exec::Block>* tiles = nullptr;
};

// Per-point results, written by the worker that owns the point.
struct Paths {
  std::vector<double> end_points;   // n * m, unit vectors
  std::vector<double> start_value;  // log sum_i exp(kappa (<x_i, y> - 1)) at the start
  std::vector<double> end_value;    // the same at the end point
  std::vector<std::uint32_t> steps;
  std::vector<std::uint8_t> converged;

  Paths(std::size_t n, std::size_t m)
      : end_points(n * m), start_value(n, -kInf), end_value(n, -kInf), steps(n, 0), converged(n, 0) {}
};

// State of the seeds of one batch; allocated once per worker.
struct SeedScratch {
  std::vector<double> position;   // max_batch * m
  std::vector<double> sum;        // max_batch * m, weighted sum of the data rows
  std::vector<double> reference;  // largest exponent met in the current sweep
  std::vector<double> weight;     // sum of exp(exponent - reference)
  std::vector<double> previous;   // density value at the previous position
  std::vector<std::uint32_t> steps;
  std::vector<Phase> phase;
  std::vector<std::uint8_t> converged;
  std::vector<std::uint32_t> active;

  void reserve(std::size_t max_batch, std::size_t m) {
    if (!reference.empty()) return;
    position.resize(max_batch * m);
    sum.resize(max_batch * m);
    reference.resize(max_batch);
    weight.resize(max_batch);
    previous.resize(max_batch);
    steps.resize(max_batch);
    phase.resize(max_batch);
    converged.resize(max_batch);
    active.reserve(max_batch);
  }

  static std::size_t bytes_per_seed(std::size_t m) {
    return 2 * m * sizeof(double) + 3 * sizeof(double) + 2 * sizeof(std::uint32_t) + 2;
  }
};

struct BatchOutcome {
  std::size_t not_converged = 0;
  std::size_t density_decreases = 0;
};

// One sweep of a tile for one seed: adds the rows of the tile, in order, to the
// kernel sums of the seed. The sums are kept relative to the largest exponent
// met so far (`reference`), so that no concentration makes them underflow.
inline void sweep_moving(const Problem& p, const exec::Block& tile, const double* position, double* sum,
                         double& reference, double& weight) {
  const float* rows = p.data + tile.first * p.m;
  for (std::size_t r = 0; r < tile.count; ++r) {
    const float* x = rows + r * p.m;
    const double e = p.kappa * (detail::dot(x, position, p.m) - 1.0);
    if (e > reference) {
      const double f = std::exp(reference - e);
      weight = weight * f + 1.0;
      detail::scale_add(f, x, sum, p.m);
      reference = e;
    } else {
      const double w = std::exp(e - reference);
      weight += w;
      if (w != 0.0) detail::axpy(w, x, sum, p.m);
    }
  }
}

// The same sweep without the weighted sum, for a seed that has stopped.
inline void sweep_landing(const Problem& p, const exec::Block& tile, const double* position, double& reference,
                          double& weight) {
  const float* rows = p.data + tile.first * p.m;
  for (std::size_t r = 0; r < tile.count; ++r) {
    const double e = p.kappa * (detail::dot(rows + r * p.m, position, p.m) - 1.0);
    if (e > reference) {
      weight = weight * std::exp(reference - e) + 1.0;
      reference = e;
    } else {
      weight += std::exp(e - reference);
    }
  }
}

// Runs every mean-shift iteration of the seeds of one batch. `observe` is
// called with (seed, position, value) for every position whose density has
// been evaluated, in path order.
template <class Observer>
BatchOutcome run_batch(const Problem& p, const exec::Block& batch, SeedScratch& s, Paths& paths,
                       Observer&& observe) {
  BatchOutcome outcome;
  const std::size_t m = p.m;
  s.active.clear();
  for (std::size_t q = 0; q < batch.count; ++q) {
    const float* x = p.data + (batch.first + q) * m;
    double* position = s.position.data() + q * m;
    const double norm2 = detail::dot(x, x, m);
    if (norm2 > 0.0 && std::isfinite(norm2)) {
      const double inverse = 1.0 / std::sqrt(norm2);
      for (std::size_t j = 0; j < m; ++j) position[j] = static_cast<double>(x[j]) * inverse;
    } else {
      for (std::size_t j = 0; j < m; ++j) position[j] = j == 0 ? 1.0 : 0.0;
    }
    s.previous[q] = -kInf;
    s.steps[q] = 0;
    s.converged[q] = 0;
    s.phase[q] = p.max_iterations == 0 ? Phase::Landing : Phase::Moving;
    s.active.push_back(static_cast<std::uint32_t>(q));
  }

  while (!s.active.empty()) {
    for (const std::uint32_t q : s.active) {
      s.reference[q] = -kInf;
      s.weight[q] = 0.0;
      if (s.phase[q] == Phase::Moving) std::fill_n(s.sum.begin() + static_cast<std::ptrdiff_t>(q * m), m, 0.0);
    }
    // The tile stays in the cache while every active seed of the batch uses it.
    for (const exec::Block& tile : *p.tiles) {
      for (const std::uint32_t q : s.active) {
        const double* position = s.position.data() + q * m;
        if (s.phase[q] == Phase::Moving) {
          sweep_moving(p, tile, position, s.sum.data() + q * m, s.reference[q], s.weight[q]);
        } else {
          sweep_landing(p, tile, position, s.reference[q], s.weight[q]);
        }
      }
    }

    std::size_t kept = 0;
    for (const std::uint32_t q : s.active) {
      const std::size_t seed = batch.first + q;
      double* position = s.position.data() + q * m;
      const double value = s.reference[q] + std::log(s.weight[q]);
      observe(seed, static_cast<const double*>(position), value);
      if (s.steps[q] == 0) paths.start_value[seed] = value;
      if (value < s.previous[q] - 1e-12 * std::max(1.0, std::fabs(s.previous[q]))) ++outcome.density_decreases;
      s.previous[q] = value;

      bool finished = s.phase[q] == Phase::Landing;
      if (!finished) {
        const double* sum = s.sum.data() + q * m;
        const double norm = std::sqrt(detail::dot(sum, sum, m));
        if (!(norm > 0.0) || !std::isfinite(norm)) {
          // The weighted sum has no direction: the seed stays where it is, and
          // the density there has just been evaluated.
          s.converged[q] = 1;
          finished = true;
        } else {
          const double inverse = 1.0 / norm;
          double shift = 0.0;  // 1 - <new, old> = |new - old|^2 / 2 for unit vectors
          for (std::size_t j = 0; j < m; ++j) {
            const double next = sum[j] * inverse;
            const double delta = next - position[j];
            shift += delta * delta;
            position[j] = next;
          }
          shift *= 0.5;
          ++s.steps[q];
          if (shift <= p.tolerance) {
            s.converged[q] = 1;
            s.phase[q] = Phase::Landing;
          } else if (s.steps[q] >= p.max_iterations) {
            s.phase[q] = Phase::Landing;
          }
        }
      }
      if (finished) {
        s.phase[q] = Phase::Done;
        std::copy_n(position, m, paths.end_points.begin() + static_cast<std::ptrdiff_t>(seed * m));
        paths.end_value[seed] = value;
        paths.steps[seed] = s.steps[q];
        paths.converged[seed] = s.converged[q];
        if (s.converged[q] == 0) ++outcome.not_converged;
      } else {
        s.active[kept++] = q;
      }
    }
    s.active.resize(kept);
  }
  return outcome;
}

Problem make_problem(std::span<const float> data, std::size_t m, double kappa, const MeanShiftOptions& options,
                     const std::vector<exec::Block>& tiles) {
  Problem p;
  p.data = data.data();
  p.m = m;
  p.n = data.size() / m;
  p.kappa = kappa;
  p.tolerance = options.tolerance;
  p.max_iterations = static_cast<std::uint32_t>(std::clamp(options.max_iterations, 0, 1'000'000));
  p.tiles = &tiles;
  return p;
}

// log fhat(y) = log C_m(kappa) - log n + kappa + log sum_i exp(kappa (<x_i, y> - 1)).
double density_offset(std::size_t n, std::size_t m, double kappa) {
  return log_vmf_normalizer(m, kappa) - std::log(static_cast<double>(n)) + kappa;
}

struct Group {
  std::uint32_t centre = 0;  // point whose end point is the centre of the group
  std::uint32_t size = 0;
  std::uint32_t lowest = 0;  // lowest member index
};

}  // namespace

Modes find_modes(std::span<const float> data, std::size_t m, const MeanShiftOptions& options,
                 exec::OrderedExecutor& executor) {
  Modes modes;
  modes.m = m;
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  if (n == 0) return modes;

  modes.kappa = options.kappa;
  if (!(modes.kappa > 0.0)) {
    modes.kappa = select_kappa(data, m, executor, options.block_bytes).kappa;
    if (options.limit_to_neighbourhood) {
      modes.kappa = std::min(modes.kappa, neighbourhood_kappa(data, m, executor, options.block_bytes));
    }
  }
  const double kappa = modes.kappa;

  // ---- mean shift from every point ------------------------------------------
  const detail::Layout layout =
      detail::make_layout(n, m * sizeof(float), n, SeedScratch::bytes_per_seed(m), options.block_bytes);
  const Problem problem = make_problem(data, m, kappa, options, layout.tiles);
  Paths paths(n, m);
  std::vector<SeedScratch> scratch(executor.threads());

  executor.run(
      layout.batches,
      [&](const exec::Block& batch, unsigned worker) {
        SeedScratch& s = scratch[worker];
        s.reserve(layout.max_batch, m);
        return run_batch(problem, batch, s, paths, [](std::size_t, const double*, double) {});
      },
      [&](const exec::Block&, BatchOutcome&& outcome) {
        modes.not_converged += outcome.not_converged;
        modes.density_decreases += outcome.density_decreases;
      });

  // ---- merge end points into modes ------------------------------------------
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::uint32_t a, std::uint32_t b) { return paths.end_value[a] > paths.end_value[b]; });

  std::vector<Group> groups;
  std::vector<std::uint32_t> group_of(n, 0);
  for (const std::uint32_t i : order) {
    const double* end = paths.end_points.data() + static_cast<std::size_t>(i) * m;
    std::size_t best = groups.size();
    double best_cosine = options.merge_cosine;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const double cosine = detail::dot(end, paths.end_points.data() + static_cast<std::size_t>(groups[g].centre) * m, m);
      if (cosine > best_cosine) {
        best_cosine = cosine;
        best = g;
      }
    }
    if (best == groups.size()) groups.push_back(Group{i, 0, i});
    Group& group = groups[best];
    ++group.size;
    group.lowest = std::min(group.lowest, i);
    group_of[i] = static_cast<std::uint32_t>(best);
  }

  // ---- merge groups that are too small --------------------------------------
  std::vector<std::uint32_t> target(groups.size());
  std::iota(target.begin(), target.end(), 0u);
  if (options.min_group > 1) {
    std::vector<std::uint8_t> kept(groups.size(), 0);
    bool any = false;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      kept[g] = groups[g].size >= options.min_group ? 1 : 0;
      any = any || kept[g] != 0;
    }
    if (!any) {
      // No group is large enough: keep the largest (the densest among equals).
      std::size_t largest = 0;
      for (std::size_t g = 1; g < groups.size(); ++g) {
        if (groups[g].size > groups[largest].size) largest = g;
      }
      kept[largest] = 1;
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (kept[g] != 0) continue;
      const double* centre = paths.end_points.data() + static_cast<std::size_t>(groups[g].centre) * m;
      double best_cosine = -kInf;
      for (std::size_t h = 0; h < groups.size(); ++h) {
        if (kept[h] == 0) continue;
        const double cosine =
            detail::dot(centre, paths.end_points.data() + static_cast<std::size_t>(groups[h].centre) * m, m);
        if (cosine > best_cosine) {
          best_cosine = cosine;
          target[g] = static_cast<std::uint32_t>(h);
        }
      }
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const std::uint32_t h = target[g];
      if (h == g) continue;
      groups[h].size += groups[g].size;
      groups[h].lowest = std::min(groups[h].lowest, groups[g].lowest);
      groups[g].size = 0;
    }
  }

  // ---- number the modes ------------------------------------------------------
  std::vector<std::uint32_t> ranking;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (groups[g].size != 0) ranking.push_back(static_cast<std::uint32_t>(g));
  }
  std::sort(ranking.begin(), ranking.end(), [&](std::uint32_t a, std::uint32_t b) {
    if (groups[a].size != groups[b].size) return groups[a].size > groups[b].size;
    const double da = paths.end_value[groups[a].centre];
    const double db = paths.end_value[groups[b].centre];
    if (da != db) return da > db;
    return groups[a].lowest < groups[b].lowest;
  });
  std::vector<std::uint32_t> number(groups.size(), 0);
  for (std::size_t k = 0; k < ranking.size(); ++k) number[ranking[k]] = static_cast<std::uint32_t>(k);

  const double offset = density_offset(n, m, kappa);
  modes.centers.resize(ranking.size() * m);
  modes.sizes.resize(ranking.size());
  modes.center_log_density.resize(ranking.size());
  for (std::size_t k = 0; k < ranking.size(); ++k) {
    const Group& group = groups[ranking[k]];
    const double* centre = paths.end_points.data() + static_cast<std::size_t>(group.centre) * m;
    for (std::size_t j = 0; j < m; ++j) modes.centers[k * m + j] = static_cast<float>(centre[j]);
    modes.sizes[k] = group.size;
    modes.center_log_density[k] = static_cast<float>(offset + paths.end_value[group.centre]);
  }
  modes.assignment.resize(n);
  modes.log_density.resize(n);
  modes.iterations.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    modes.assignment[i] = number[target[group_of[i]]];
    modes.log_density[i] = static_cast<float>(offset + paths.start_value[i]);
    modes.iterations[i] = static_cast<std::uint16_t>(std::min<std::uint32_t>(paths.steps[i], 0xFFFFu));
  }
  return modes;
}

Trajectory mean_shift_trajectory(std::span<const float> data, std::size_t m, std::size_t seed,
                                 const MeanShiftOptions& options) {
  Trajectory out;
  out.m = m;
  const std::size_t n = m == 0 ? 0 : data.size() / m;
  if (seed >= n || !(options.kappa > 0.0)) return out;

  const detail::Layout layout =
      detail::make_layout(n, m * sizeof(float), 1, SeedScratch::bytes_per_seed(m), options.block_bytes);
  const Problem problem = make_problem(data, m, options.kappa, options, layout.tiles);
  Paths paths(n, m);
  SeedScratch scratch;
  scratch.reserve(1, m);
  const double offset = density_offset(n, m, options.kappa);
  const exec::Block batch{0, seed, 1, SeedScratch::bytes_per_seed(m)};
  run_batch(problem, batch, scratch, paths, [&](std::size_t, const double* position, double value) {
    out.points.insert(out.points.end(), position, position + m);
    out.log_density.push_back(offset + value);
  });
  out.converged = paths.converged[seed] != 0;
  return out;
}

std::vector<std::uint32_t> layout_by_mode(const Modes& modes) {
  std::vector<std::uint32_t> order(modes.assignment.size());
  std::iota(order.begin(), order.end(), 0u);
  const bool has_density = modes.log_density.size() == modes.assignment.size();
  std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
    if (modes.assignment[a] != modes.assignment[b]) return modes.assignment[a] < modes.assignment[b];
    if (has_density && modes.log_density[a] != modes.log_density[b]) return modes.log_density[a] > modes.log_density[b];
    return a < b;
  });
  return order;
}

}  // namespace prag::kde
