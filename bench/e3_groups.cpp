// E3: density groups: geometry before and after reduction, bandwidths, number of
// modes, agreement with structure that was not used, stability, graph locality.
#include <map>

#include "common.hpp"
#include "prag/core/clock.hpp"
#include "prag/kde/directional.hpp"

namespace bench {

namespace {

namespace kde = prag::kde;

std::vector<float> centre_and_normalise(const std::vector<float>& rows, std::size_t n, std::size_t dim) {
  std::vector<double> mean(dim, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < dim; ++j) mean[j] += static_cast<double>(rows[i * dim + j]);
  for (double& m : mean) m /= D(n);
  std::vector<float> out(rows.size());
  for (std::size_t i = 0; i < n; ++i) {
    double norm = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const double v = static_cast<double>(rows[i * dim + j]) - mean[j];
      out[i * dim + j] = static_cast<float>(v);
      norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0)
      for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = static_cast<float>(static_cast<double>(out[i * dim + j]) / norm);
  }
  return out;
}

json iteration_stats(const kde::Modes& m) {
  std::vector<double> it(m.iterations.begin(), m.iterations.end());
  return json{{"mean", mean_of(it)}, {"median", median(it)}, {"max", quantile(it, 1.0)}, {"not_converged", m.not_converged},
              {"density_decreases", m.density_decreases}};
}

std::size_t singleton_modes(const kde::Modes& m) {
  std::size_t s = 0;
  for (const auto size : m.sizes) s += size == 1 ? 1 : 0;
  return s;
}

std::vector<std::uint32_t> top_level_directory(const Context& ctx) {
  const auto& c = *ctx.corpus;
  std::vector<std::string> common;
  {
    bool first = true;
    for (const std::string& root : c.roots) {
      std::vector<std::string> parts;
      std::size_t start = 0;
      for (;;) {
        const std::size_t slash = root.find('/', start);
        parts.push_back(root.substr(start, slash == std::string::npos ? std::string::npos : slash - start));
        if (slash == std::string::npos) break;
        start = slash + 1;
      }
      if (first) {
        common = parts;
        first = false;
      } else {
        std::size_t k = 0;
        while (k < common.size() && k < parts.size() && common[k] == parts[k]) ++k;
        common.resize(k);
      }
    }
  }
  std::string prefix;
  for (const std::string& p : common) prefix += p + "/";
  std::map<std::string, std::uint32_t> ids;
  std::vector<std::uint32_t> labels(c.units.size());
  for (const auto& u : c.units) {
    std::string rel = u.path.compare(0, prefix.size(), prefix) == 0 ? u.path.substr(prefix.size()) : u.path;
    const std::size_t slash = rel.find('/');
    const std::string top = slash == std::string::npos ? "." : rel.substr(0, slash);
    labels[u.id] = ids.emplace(top, static_cast<std::uint32_t>(ids.size())).first->second;
  }
  return labels;
}

json agreement(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
  return json{{"ari", kde::adjusted_rand_index(a, b)}, {"nmi", kde::normalized_mutual_information(a, b)}};
}

std::size_t distinct(const std::vector<std::uint32_t>& v) {
  std::vector<std::uint32_t> s = v;
  std::sort(s.begin(), s.end());
  return static_cast<std::size_t>(std::unique(s.begin(), s.end()) - s.begin());
}

}  // namespace

int run_e3(Context& ctx) {
  const std::string started = iso_now();
  const auto& c = *ctx.corpus;
  const std::size_t n = ctx.n(), dim = ctx.dim();
  const std::vector<float>& rows = corpus_rows(ctx);
  auto& executor = ctx.engine->executor();
  const std::size_t block_bytes = ctx.engine->block_bytes();
  json body = json::object();

  // Geometry before and after centring and reduction.
  body["geometry"] = guarded("geometry", [&] {
    const std::size_t engine_dim = c.reduction.out_dim;
    const std::vector<float> centred = centre_and_normalise(rows, n, dim);
    json out{{"raw", json{{"dim", dim}, {"anisotropy", kde::mean_pairwise_cosine(rows, dim)},
                          {"hubness_skewness", kde::hubness_skewness(rows, dim, 10)}}},
             {"centred", json{{"dim", dim}, {"anisotropy", kde::mean_pairwise_cosine(centred, dim)},
                              {"hubness_skewness", kde::hubness_skewness(centred, dim, 10)}}},
             {"engine_index", json{{"reduced_dim", engine_dim},
                                   {"anisotropy_before_reported", c.anisotropy_before},
                                   {"anisotropy_after_reported", c.anisotropy_after}}}};
    out["reduced"] = json::object();
    for (const std::size_t d : {8u, 16u, 32u, 64u}) {
      const kde::Reduction red = kde::fit_reduction(rows, dim, d, 0, 1);
      const std::vector<float> reduced = kde::reduce(red, rows);
      out["reduced"][std::to_string(d)] = json{{"dim", red.out_dim},
                                               {"explained_variance", red.explained_variance_ratio()},
                                               {"anisotropy", kde::mean_pairwise_cosine(reduced, red.out_dim)},
                                               {"hubness_skewness", kde::hubness_skewness(reduced, red.out_dim, 10)}};
    }
    return out;
  });

  // Bandwidth, number of modes and mean-shift cost per reduced dimension.
  body["per_dimension"] = guarded("per_dimension", [&] {
    json out = json::object();
    const std::vector<std::size_t> dims = ctx.quick() ? std::vector<std::size_t>{8, 32} : std::vector<std::size_t>{8, 16, 32, 64};
    for (const std::size_t d : dims) {
      progress("E3 dimension %zu", d);
      const kde::Reduction red = kde::fit_reduction(rows, dim, d, 0, 1);
      const std::vector<float> reduced = kde::reduce(red, rows);
      const std::size_t m = red.out_dim;
      const kde::KappaSelection sel = kde::select_kappa(reduced, m, executor, block_bytes);
      const double limit = kde::neighbourhood_kappa(reduced, m, executor, block_bytes);
      kde::MeanShiftOptions mo;
      mo.block_bytes = block_bytes;
      mo.min_group = ctx.engine->settings().get<std::size_t>("kde.min_group", 3);
      prag::Stopwatch watch;
      const kde::Modes modes = kde::find_modes(reduced, m, mo, executor);
      const double seconds = watch.elapsed_s();
      out[std::to_string(d)] = json{{"dim", m},
                                    {"kappa_rule_of_thumb", sel.rule_of_thumb},
                                    {"kappa_cross_validated", sel.kappa},
                                    {"cv_log_likelihood", sel.score},
                                    {"kappa_neighbourhood_limit", limit},
                                    {"kappa_used", modes.kappa},
                                    {"modes", modes.count()},
                                    {"singleton_modes", singleton_modes(modes)},
                                    {"mean_shift_iterations", iteration_stats(modes)},
                                    {"mean_shift_seconds", seconds}};
    }
    return out;
  });

  // Number of modes against kappa on the engine's reduction.
  body["modes_vs_kappa"] = guarded("modes_vs_kappa", [&] {
    const std::size_t m = c.reduction.out_dim;
    const double limit = kde::neighbourhood_kappa(c.reduced, m, executor, block_bytes);
    json rows_out = json::array();
    std::vector<double> kappas;
    for (const double f : ctx.quick() ? std::vector<double>{0.5, 1.0, 2.0} : std::vector<double>{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0})
      kappas.push_back(limit * f);
    if (c.modes.kappa > 0.0) kappas.push_back(c.modes.kappa);
    std::sort(kappas.begin(), kappas.end());
    for (const double kappa : kappas) {
      kde::MeanShiftOptions mo;
      mo.kappa = kappa;
      mo.min_group = 1;
      mo.block_bytes = block_bytes;
      const kde::Modes modes = kde::find_modes(c.reduced, m, mo, executor);
      rows_out.push_back(json{{"kappa", kappa},
                              {"kappa_over_neighbourhood_limit", limit > 0.0 ? kappa / limit : kNaN},
                              {"modes", modes.count()},
                              {"singleton_modes", singleton_modes(modes)},
                              {"largest_mode", modes.sizes.empty() ? 0u : modes.sizes.front()},
                              {"mean_iterations", iteration_stats(modes)["mean"]}});
    }
    return json{{"dim", m}, {"neighbourhood_limit", limit}, {"min_group", 1}, {"rows", rows_out}};
  });

  // Agreement with file, directory and spherical k-means; stability; call-graph locality.
  body["agreement"] = guarded("agreement", [&] {
    const std::size_t m = c.reduction.out_dim;
    const std::vector<std::uint32_t>& groups = c.modes.assignment;
    const std::size_t k = c.modes.count();
    std::vector<std::uint32_t> files(c.file_of_unit.begin(), c.file_of_unit.end());
    const std::vector<std::uint32_t> dirs = top_level_directory(ctx);
    const std::vector<std::uint32_t> km = spherical_kmeans(c.reduced.data(), n, m, k, ctx.seed + 21);
    json out{{"groups", k},
             {"files", distinct(files)},
             {"top_level_directories", distinct(dirs)},
             {"vs_files", agreement(groups, files)},
             {"vs_top_level_directories", agreement(groups, dirs)},
             {"vs_spherical_kmeans", agreement(groups, km)},
             {"kmeans_same_k", distinct(km)}};
    // Reference: the agreement of k-means with the same structure, as a yardstick.
    out["kmeans_vs_files"] = agreement(km, files);
    out["kmeans_vs_top_level_directories"] = agreement(km, dirs);

    const std::size_t resamples = ctx.quick() ? 3 : 20;
    std::vector<double> ari, nmi, counts;
    std::uint64_t state = ctx.seed + 31;
    for (std::size_t b = 0; b < resamples; ++b) {
      std::vector<char> in(n, 0);
      for (std::size_t i = 0; i < n; ++i) in[static_cast<std::size_t>(rng_below(state, n))] = 1;
      std::vector<std::size_t> picked;
      for (std::size_t i = 0; i < n; ++i)
        if (in[i]) picked.push_back(i);
      std::vector<float> sub(picked.size() * m);
      for (std::size_t r = 0; r < picked.size(); ++r)
        std::copy(c.reduced.begin() + static_cast<std::ptrdiff_t>(picked[r] * m),
                  c.reduced.begin() + static_cast<std::ptrdiff_t>((picked[r] + 1) * m), sub.begin() + static_cast<std::ptrdiff_t>(r * m));
      kde::MeanShiftOptions mo;
      mo.kappa = c.modes.kappa;
      mo.min_group = ctx.engine->settings().get<std::size_t>("kde.min_group", 3);
      mo.block_bytes = block_bytes;
      const kde::Modes sample = kde::find_modes(sub, m, mo, executor);
      std::vector<std::uint32_t> original(picked.size());
      for (std::size_t r = 0; r < picked.size(); ++r) original[r] = groups[picked[r]];
      ari.push_back(kde::adjusted_rand_index(original, sample.assignment));
      nmi.push_back(kde::normalized_mutual_information(original, sample.assignment));
      counts.push_back(D(sample.count()));
      progress("E3 stability resample %zu / %zu", b + 1, resamples);
    }
    out["stability"] = json{{"resamples", resamples},
                            {"sample", "points drawn with replacement, duplicates removed (about 63 %)"},
                            {"kappa", c.modes.kappa},
                            {"ari", to_json(bootstrap_mean(ari, ctx.seed + 32))},
                            {"nmi", to_json(bootstrap_mean(nmi, ctx.seed + 32))},
                            {"modes_mean", mean_of(counts)},
                            {"ari_values", ari},
                            {"nmi_values", nmi}};

    // Share of call-graph edges inside one group, against random assignment with the same group sizes.
    std::size_t edges = 0, inside = 0;
    const auto& fwd = c.calls.forward();
    for (std::size_t u = 0; u < n; ++u)
      for (const UnitId v : fwd.of(static_cast<UnitId>(u))) {
        if (v == u) continue;
        ++edges;
        inside += groups[u] == groups[v] ? 1 : 0;
      }
    const double observed = edges ? D(inside) / D(edges) : kNaN;
    const std::size_t permutations = ctx.quick() ? 50 : 500;
    std::vector<double> random_share;
    std::vector<std::uint32_t> shuffled = groups;
    std::size_t at_least = 0;
    for (std::size_t r = 0; r < permutations; ++r) {
      shuffle(shuffled, ctx.seed + 41 + r);
      std::size_t same = 0;
      for (std::size_t u = 0; u < n; ++u)
        for (const UnitId v : fwd.of(static_cast<UnitId>(u))) {
          if (v == u) continue;
          same += shuffled[u] == shuffled[v] ? 1 : 0;
        }
      const double share = edges ? D(same) / D(edges) : kNaN;
      random_share.push_back(share);
      at_least += share >= observed ? 1 : 0;
    }
    out["call_graph"] = json{{"edges", edges},
                             {"observed_share_same_group", observed},
                             {"random_share_mean", mean_of(random_share)},
                             {"random_share_q025", quantile(random_share, 0.025)},
                             {"random_share_q975", quantile(random_share, 0.975)},
                             {"permutations", permutations},
                             {"permutations_at_least_observed", at_least},
                             {"p_value_add_one", D(at_least + 1) / D(permutations + 1)}};
    return out;
  });

  write_result(ctx, "e3_groups", started,
               json{{"dims", json::array({8, 16, 32, 64})},
                    {"engine_dim", c.reduction.out_dim},
                    {"engine_kappa", c.modes.kappa},
                    {"engine_min_group", ctx.engine->settings().get<std::size_t>("kde.min_group", 3)},
                    {"bootstrap_stability_resamples", ctx.quick() ? 3 : 20},
                    {"random_assignment_permutations", ctx.quick() ? 50 : 500},
                    {"kmeans_seed", ctx.seed + 21}},
               std::move(body));
  return 0;
}

}  // namespace bench
