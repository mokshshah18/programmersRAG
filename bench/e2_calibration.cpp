// E2: calibration of the score posterior.
#include "common.hpp"
#include "prag/retrieve/kde1d.hpp"

namespace bench {

namespace {

namespace rt = prag::retrieve;

// Smallest z on a fine grid at which p reaches tau (linear interpolation inside the step); null when none does.
json first_crossing(const std::function<double(double)>& p, double tau) {
  double previous_z = -6.0, previous_p = p(previous_z);
  if (previous_p >= tau) return previous_z;
  for (double z = -5.99; z <= 30.0; z += 0.01) {
    const double v = p(z);
    if (v >= tau) {
      const double span = v - previous_p;
      return span > 0.0 ? previous_z + (tau - previous_p) / span * (z - previous_z) : z;
    }
    previous_z = z;
    previous_p = v;
  }
  return nullptr;
}

json evaluate_set(Context& ctx, QuerySet& set) {
  const std::size_t n = ctx.n();
  const std::vector<Scored> scored = score_queries(ctx, set);
  const Labeled fit = collect_labeled(set, scored, kCalibration, ctx.quick() ? 64 : 400, ctx.seed + 3);
  const Labeled held = collect_labeled(set, scored, kHeldout, 0, ctx.seed + 4);
  const Labeled test = collect_labeled(set, scored, kTest, 0, ctx.seed + 5);
  const double prior = 1.0 / D(n);
  const bool enough = fit.queries >= 20;
  const rt::Calibration cal = enough ? rt::Calibration::fit(fit.z_relevant, fit.z_nonrelevant, prior)
                                     : rt::Calibration::fallback();
  const rt::Calibration fallback = rt::Calibration::fallback();
  const PlattFit platt = fit_platt(fit.samples, prior);

  struct Variant {
    std::string name;
    std::function<double(double)> p;
    json extra;
  };
  std::vector<Variant> variants;
  variants.push_back({"kde_posterior", [&](double z) { return cal.posterior(z); },
                      json{{"fitted", cal.fitted()},
                           {"fraction_relevant", cal.fraction_relevant()},
                           {"fraction_nonrelevant", cal.fraction_nonrelevant()}}});
  variants.push_back({"kde_posterior_without_isotonic",
                      [&](double z) {
                        const double r = cal.likelihood_ratio(z);
                        const double a = cal.prior() * r;
                        return a / (a + 1.0 - cal.prior());
                      },
                      json{{"note", "Bayes rule on the tabulated density ratio, before the monotone regression"}}});
  variants.push_back({"logistic_fallback", [&](double z) { return fallback.posterior(z); },
                      json{{"midpoint", 4.0}, {"slope", 1.5}, {"note", "not fitted to any data"}}});
  variants.push_back({"platt", [&](double z) { return platt.posterior(z); },
                      json{{"ok", platt.ok}, {"a", platt.a}, {"b", platt.b},
                           {"note", "logistic regression on the calibration split, intercept shifted to the prior 1/n"}}});

  json out_variants = json::object();
  for (const Variant& v : variants) {
    json row = v.extra;
    row["heldout"] = reliability_json(v.p, held.samples);
    row["test"] = reliability_json(v.p, test.samples);
    row["z_at_tau_0.25"] = first_crossing(v.p, 0.25);
    out_variants[v.name] = std::move(row);
  }
  {  // Cross-check of the reliability code against the library's.
    const rt::Reliability lib = rt::reliability(cal, held.samples);
    out_variants["kde_posterior"]["library_heldout"] =
        json{{"ece", lib.expected_calibration_error}, {"brier", lib.brier}, {"log_loss", lib.log_loss}};
    out_variants["kde_posterior"]["z_at_tau_0.25_library"] = cal.threshold(0.25);
  }

  // Class-conditional densities (Gaussian kernel, Silverman bandwidth) and the tabulated calibration.
  json densities = json::object();
  {
    std::vector<double> negatives = fit.z_nonrelevant;
    if (negatives.size() > 20000) {
      std::vector<double> thinned;
      for (std::size_t i = 0; i < 20000; ++i) thinned.push_back(negatives[i * negatives.size() / 20000]);
      negatives = std::move(thinned);
    }
    double lo = 1e300, hi = -1e300;
    for (const double z : fit.z_relevant) { lo = std::min(lo, z); hi = std::max(hi, z); }
    for (const double z : negatives) { lo = std::min(lo, z); hi = std::max(hi, z); }
    if (lo < hi) {
      const rt::Kde1d f1 = rt::Kde1d::fit(fit.z_relevant), f0 = rt::Kde1d::fit(negatives);
      json grid = json::array(), d1 = json::array(), d0 = json::array();
      const int points = 160;
      for (int i = 0; i < points; ++i) {
        const double z = lo + (hi - lo) * i / (points - 1);
        grid.push_back(z);
        d1.push_back(f1.pdf(z));
        d0.push_back(f0.pdf(z));
      }
      densities = json{{"estimator", "Gaussian kernel, Silverman bandwidth (plain, not the local-quadratic fit of the calibration)"},
                       {"bandwidth_relevant", f1.bandwidth()},
                       {"bandwidth_nonrelevant", f0.bandwidth()},
                       {"relevant_samples", fit.z_relevant.size()},
                       {"nonrelevant_samples_used", negatives.size()},
                       {"z", grid},
                       {"f_relevant", d1},
                       {"f_nonrelevant", d0}};
    }
    json table_z = json::array(), table_p = json::array(), table_r = json::array();
    for (std::size_t i = 0; i < cal.grid().size(); ++i) {
      table_z.push_back(cal.grid()[i]);
      table_p.push_back(cal.table()[i]);
      table_r.push_back(cal.likelihood_ratio(cal.grid()[i]));
    }
    densities["calibration_grid"] = json{{"z", table_z}, {"posterior", table_p}, {"likelihood_ratio", table_r}};
  }
  return json{{"queries", json{{"calibration", fit.queries}, {"heldout", held.queries}, {"test", test.queries}}},
              {"samples", json{{"calibration", fit.samples.size()}, {"heldout", held.samples.size()}, {"test", test.samples.size()}}},
              {"prior", prior},
              {"calibration_fitted", enough && cal.fitted()},
              {"variants", out_variants},
              {"densities", densities}};
}

}  // namespace

int run_e2(Context& ctx) {
  const std::string started = iso_now();
  QuerySets& sets = query_sets(ctx);
  json body = json::object();
  body["q_doc"] = guarded("q_doc", [&] { return evaluate_set(ctx, sets.doc); });
  body["q_gen"] = sets.gen.ok() ? guarded("q_gen", [&] { return evaluate_set(ctx, sets.gen); }) : skipped(sets.gen.skipped);
  write_result(ctx, "e2_calibration", started,
               json{{"bins", 10},
                    {"calibration_negatives_per_query", ctx.quick() ? 64 : 400},
                    {"heldout_and_test_negatives", "all other units (true prior 1/n)"},
                    {"reliability_split", "heldout"}},
               std::move(body));
  return 0;
}

}  // namespace bench
