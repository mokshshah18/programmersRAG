// E1: retrieval quality of the probabilistic trace against fixed-k, largest-gap,
// personalised PageRank and the split-conformal rule; quantised codes against
// exact scoring.
#include <map>
#include <unordered_set>

#include "common.hpp"
#include "prag/core/clock.hpp"
#include "prag/index/vector_store.hpp"
#include "prag/kde/directional.hpp"
#include "prag/retrieve/baselines.hpp"
#include "prag/retrieve/trace.hpp"

namespace bench {

namespace {

using prag::graph::CallGraph;
using prag::graph::NeighborGraph;
namespace rt = prag::retrieve;

struct Selected {
  std::vector<UnitId> ids;  // method order
  std::size_t evaluated = 0, expansions = 0;
  bool truncated = false, low_confidence = false;
};

struct Env {
  Context* ctx;
  const rt::Calibration* cal;
  rt::TraceOptions base;
  CallGraph no_calls;
  NeighborGraph no_siblings;
  double conformal_z[2] = {0, 0};
};

Selected from_traced(const std::vector<rt::TracedUnit>& units) {
  Selected s;
  for (const auto& t : units) s.ids.push_back(t.id);
  return s;
}

struct Method {
  std::string name;
  json parameters;
  std::function<Selected(const Scored&)> run;
};

std::vector<Method> make_methods(Env& env) {
  Context& ctx = *env.ctx;
  const auto& c = *ctx.corpus;
  std::vector<Method> methods;
  for (const std::size_t k : {1u, 3u, 5u, 10u, 20u}) {
    methods.push_back({"topk_" + std::to_string(k), json{{"k", k}}, [&env, k](const Scored& s) {
                         return from_traced(rt::top_k(s.score, s.mean, s.sd, *env.cal, k));
                       }});
  }
  methods.push_back({"largest_gap", json{{"k_max", 20}, {"buffer", 0}}, [&env](const Scored& s) {
                       return from_traced(rt::largest_gap_cutoff(s.score, s.mean, s.sd, *env.cal, 20, 0));
                     }});
  for (const std::size_t k : {5u, 10u}) {
    methods.push_back({"ppr_top" + std::to_string(k), json{{"k", k}, {"continuation", 0.5}, {"seeds", "score posterior"}},
                       [&env, &c, k](const Scored& s) {
                         std::vector<double> seeds(s.score.size());
                         for (std::size_t i = 0; i < seeds.size(); ++i) seeds[i] = env.cal->posterior(s.z(i));
                         const auto rank = rt::personalized_pagerank(c.calls, seeds);
                         return from_traced(rt::top_k_by_rank(rank, s.score, s.mean, s.sd, *env.cal, k));
                       }});
  }
  for (const double tau : {0.10, 0.25, 0.50}) {
    char label[16];
    std::snprintf(label, sizeof label, "%.2f", tau);
    for (const bool graph : {true, false}) {
      methods.push_back({std::string(graph ? "trace_tau" : "trace_nograph_tau") + label,
                         json{{"tau", tau}, {"graph_edges", graph}},
                         [&env, &c, tau, graph](const Scored& s) {
                           rt::TraceOptions to = env.base;
                           to.tau = tau;
                           const rt::TraceResult r =
                               rt::trace(s.score, s.mean, s.sd, *env.cal, graph ? c.calls : env.no_calls,
                                         graph ? c.siblings : env.no_siblings, to);
                           Selected out = from_traced(r.units);
                           out.evaluated = r.evaluated;
                           out.expansions = r.expansions;
                           out.truncated = r.truncated;
                           out.low_confidence = r.low_confidence;
                           return out;
                         }});
    }
  }
  int index = 0;
  for (const double alpha : {0.1, 0.2}) {
    const int which = index++;
    methods.push_back({"conformal_alpha" + std::string(alpha == 0.1 ? "0.1" : "0.2"), json{{"alpha", alpha}},
                       [&env, which](const Scored& s) {
                         Selected out;
                         const double t = env.conformal_z[which];
                         std::vector<UnitId> ids;
                         for (std::size_t j = 0; j < s.score.size(); ++j)
                           if (s.z(j) >= t) ids.push_back(static_cast<UnitId>(j));
                         std::stable_sort(ids.begin(), ids.end(), [&](UnitId a, UnitId b) { return s.score[a] > s.score[b]; });
                         out.ids = std::move(ids);
                         return out;
                       }});
  }
  return methods;
}

// ---- codes -----------------------------------------------------------------

struct CodeRun {
  std::vector<UnitId> ids;
  std::size_t rescored = 0;
  std::vector<float> estimates;
};

CodeRun code_trace(Context& ctx, const prag::index::VectorStore& store, const std::vector<float>& q,
                   const rt::Calibration& cal, const rt::TraceOptions& to) {
  const auto& c = *ctx.corpus;
  const std::size_t n = c.units.size();
  prag::index::ScanResult sr = prag::index::scan(store, q, ctx.engine->executor());
  CodeRun out;
  out.estimates = sr.scores;
  const double mean = sr.mean, sd = sr.stddev;
  std::vector<char> exact(n, 0);
  auto rescore = [&](std::vector<UnitId>& ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](UnitId v) { return v >= n || exact[v]; }), ids.end());
    if (ids.empty()) return false;
    for (const UnitId v : ids) exact[v] = 1;
    prag::index::rescore(store, q, ids, sr);
    out.rescored += ids.size();
    return true;
  };
  {
    std::vector<UnitId> ids;
    const double zt = cal.threshold(to.tau);
    if (sd > 0.0 && std::isfinite(sd) && zt != std::numeric_limits<double>::infinity()) {
      const double st = std::isfinite(zt) ? mean + zt * sd : -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < n; ++j)
        if (static_cast<double>(sr.scores[j]) + static_cast<double>(sr.error[j]) >= st) ids.push_back(static_cast<UnitId>(j));
    }
    rescore(ids);
  }
  rt::TraceResult tr;
  for (int round = 0; round < 64; ++round) {
    tr = rt::trace(sr.scores, mean, sd, cal, c.calls, c.siblings, to);
    std::vector<UnitId> ids;
    for (const auto& t : tr.units) {
      ids.push_back(t.id);
      for (const UnitId v : c.calls.callees(t.id)) ids.push_back(v);
      for (const UnitId v : c.calls.callers(t.id)) ids.push_back(v);
      std::size_t k = 0;
      for (const UnitId v : c.siblings.of(t.id)) {
        if (k++ >= to.sibling_fanout) break;
        ids.push_back(v);
      }
    }
    if (tr.low_confidence && !tr.units.empty()) {
      const float best = sr.scores[tr.units.front().id];
      for (std::size_t j = 0; j < n; ++j)
        if (!exact[j] && sr.scores[j] + sr.error[j] >= best) ids.push_back(static_cast<UnitId>(j));
    }
    if (!rescore(ids)) break;
  }
  for (const auto& t : tr.units) out.ids.push_back(t.id);
  return out;
}

json code_comparison(Context& ctx, const QuerySet& set, const std::vector<Scored>& scored, const rt::Calibration& cal,
                     const rt::TraceOptions& to, const Env& env) {
  const auto& c = *ctx.corpus;
  const std::vector<float>& rows = rows_for(ctx, set);
  const std::size_t n = ctx.n(), dim = ctx.dim();
  const std::vector<std::uint32_t> layout = prag::kde::layout_by_mode(c.modes);
  json result = json::object();
  for (const prag::index::CodeKind kind : {prag::index::CodeKind::Int8, prag::index::CodeKind::RaBitQ}) {
    prag::index::BuildOptions options = c.store.options();
    options.codes = kind;
    options.block_bytes = ctx.engine->block_bytes();
    prag::index::VectorStore store;
    prag::Stopwatch watch;
    store.build(rows, dim, layout, c.modes.assignment, options, ctx.engine->executor());
    const double build_s = watch.elapsed_s();
    std::vector<double> retention, jaccard, rescored, top10, hit_exact, hit_code, size_exact, size_code;
    for (std::size_t i = 0; i < set.queries.size(); ++i) {
      const Query& q = set.queries[i];
      const Scored& s = scored[i];
      if (q.split != kTest || !s.ok) continue;
      const rt::TraceResult exact = rt::trace(s.score, s.mean, s.sd, cal, c.calls, c.siblings, to);
      std::unordered_set<UnitId> a;
      for (const auto& t : exact.units) a.insert(t.id);
      const CodeRun run = code_trace(ctx, store, s.q, cal, to);
      std::unordered_set<UnitId> b(run.ids.begin(), run.ids.end());
      std::size_t common = 0;
      for (const UnitId id : a) common += b.count(id);
      retention.push_back(a.empty() ? kNaN : D(common) / D(a.size()));
      jaccard.push_back(a.empty() && b.empty() ? 1.0 : D(common) / D(a.size() + b.size() - common));
      rescored.push_back(D(run.rescored));
      hit_exact.push_back(a.count(q.unit) ? 1.0 : 0.0);
      hit_code.push_back(b.count(q.unit) ? 1.0 : 0.0);
      size_exact.push_back(D(a.size()));
      size_code.push_back(D(b.size()));
      // Top-10 of the raw estimates against the exact top-10 (no rescoring).
      std::vector<UnitId> order(n);
      for (std::size_t j = 0; j < n; ++j) order[j] = static_cast<UnitId>(j);
      auto top = [&](const std::vector<float>& sc) {
        std::vector<UnitId> o = order;
        std::partial_sort(o.begin(), o.begin() + 10, o.end(), [&](UnitId x, UnitId y) { return sc[x] != sc[y] ? sc[x] > sc[y] : x < y; });
        o.resize(10);
        return o;
      };
      const auto te = top(s.score), tc = top(run.estimates);
      std::size_t overlap = 0;
      for (const UnitId x : te) overlap += std::find(tc.begin(), tc.end(), x) != tc.end() ? 1 : 0;
      top10.push_back(D(overlap) / 10.0);
    }
    const std::uint64_t seed = ctx.seed + 11;
    result[prag::index::to_string(kind)] = json{
        {"bytes_per_code", store.bytes_per_code()},
        {"build_seconds", build_s},
        {"queries", retention.size()},
        {"oracle_recall_exact", to_json(bootstrap_mean(hit_exact, seed))},
        {"oracle_recall_codes", to_json(bootstrap_mean(hit_code, seed))},
        {"set_recall_retention", to_json(bootstrap_mean(retention, seed))},
        {"set_jaccard", to_json(bootstrap_mean(jaccard, seed))},
        {"set_size_exact", to_json(bootstrap_mean(size_exact, seed))},
        {"set_size_codes", to_json(bootstrap_mean(size_code, seed))},
        {"rescored_units", to_json(bootstrap_mean(rescored, seed))},
        {"rescored_share_of_corpus", mean_of(rescored) / D(n)},
        {"top10_overlap_without_rescoring", to_json(bootstrap_mean(top10, seed))}};
  }
  (void)env;
  return result;
}

// ---- one query set ---------------------------------------------------------

json evaluate_set(Context& ctx, QuerySet& set, const json& utility) {
  const auto& c = *ctx.corpus;
  const std::size_t n = ctx.n();
  const std::vector<Scored> scored = score_queries(ctx, set);
  const Labeled fit = collect_labeled(set, scored, kCalibration, ctx.quick() ? 64 : 400, ctx.seed + 3);
  const Labeled held = collect_labeled(set, scored, kHeldout, 0, ctx.seed + 4);
  rt::Calibration cal = rt::Calibration::fallback();
  if (fit.queries >= 20) cal = rt::Calibration::fit(fit.z_relevant, fit.z_nonrelevant, 1.0 / D(n));

  Env env;
  env.ctx = &ctx;
  env.cal = &cal;
  env.base.max_units = ctx.engine->settings().get<std::size_t>("retrieve.max_units", 64);
  env.base.max_hops = ctx.engine->settings().get<std::size_t>("retrieve.max_hops", 6);
  env.base.lift = c.lifts;
  env.no_calls = CallGraph::from_edges(n, {}, {});
  env.no_siblings = NeighborGraph::from_edges(n, {});
  json conformal = json::object();
  int idx = 0;
  for (const double alpha : {0.1, 0.2}) {
    env.conformal_z[idx] = rt::conformal_threshold(held.z_relevant, alpha);
    conformal[alpha == 0.1 ? "0.1" : "0.2"] = json{{"z_threshold", env.conformal_z[idx]}, {"heldout_queries", held.queries}};
    ++idx;
  }

  // Useful units by query id (Q-gen only).
  std::map<std::string, std::unordered_set<UnitId>> useful;
  bool have_utility = false;
  if (set.name == "q_gen" && utility.is_object() && utility.contains("queries")) {
    have_utility = true;
    for (const json& row : utility["queries"])
      for (const json& cand : row["candidates"])
        if (cand.value("useful", false)) useful[row.value("query", std::string())].insert(cand.value("unit", prag::kNoUnit));
    for (const json& row : utility["queries"]) useful[row.value("query", std::string())];  // queries without useful units
  }

  std::vector<Method> methods = make_methods(env);
  json out_methods = json::object();
  std::vector<std::size_t> test;
  for (std::size_t i = 0; i < set.queries.size(); ++i)
    if (set.queries[i].split == kTest && scored[i].ok) test.push_back(i);
  progress("E1 %s: %zu test queries, %zu calibration queries", set.name.c_str(), test.size(), fit.queries);

  for (const Method& m : methods) {
    std::vector<double> hit, size, precision, tokens, rr, u_recall, u_precision, evaluated, expansions, truncated, low;
    for (const std::size_t i : test) {
      const Query& q = set.queries[i];
      const Scored& s = scored[i];
      const Selected sel = m.run(s);
      std::vector<char> in(n, 0);
      double tok = 0.0;
      for (const UnitId id : sel.ids) {
        in[id] = 1;
        tok += static_cast<double>(c.units[id].tokens);
      }
      const bool is_hit = in[q.unit] != 0;
      hit.push_back(is_hit ? 1.0 : 0.0);
      size.push_back(D(sel.ids.size()));
      precision.push_back(sel.ids.empty() ? kNaN : (is_hit ? 1.0 / D(sel.ids.size()) : 0.0));
      tokens.push_back(tok);
      // Rank of the relevant unit in the method's order, continued by score.
      double rank;
      const auto pos = std::find(sel.ids.begin(), sel.ids.end(), q.unit);
      if (pos != sel.ids.end()) {
        rank = D(static_cast<std::size_t>(pos - sel.ids.begin()) + 1);
      } else {
        std::size_t better = 0;
        for (std::size_t j = 0; j < n; ++j)
          if (!in[j] && (s.score[j] > s.score[q.unit] || (s.score[j] == s.score[q.unit] && j < q.unit))) ++better;
        rank = D(sel.ids.size() + better + 1);
      }
      rr.push_back(1.0 / rank);
      evaluated.push_back(D(sel.evaluated));
      expansions.push_back(D(sel.expansions));
      truncated.push_back(sel.truncated ? 1.0 : 0.0);
      low.push_back(sel.low_confidence ? 1.0 : 0.0);
      if (have_utility) {
        const auto it = useful.find(q.id);
        if (it != useful.end() && !it->second.empty()) {
          std::size_t got = 0, non_oracle = 0;
          for (const UnitId id : sel.ids) {
            if (id == q.unit) continue;
            ++non_oracle;
            got += it->second.count(id);
          }
          u_recall.push_back(D(got) / D(it->second.size()));
          u_precision.push_back(non_oracle ? D(got) / D(non_oracle) : kNaN);
        }
      }
    }
    const std::uint64_t seed = ctx.seed + 100;
    json row{{"parameters", m.parameters},
             {"queries", test.size()},
             {"recall", to_json(bootstrap_mean(hit, seed))},
             {"set_size_mean", to_json(bootstrap_mean(size, seed))},
             {"set_size_median", median(size)},
             {"precision", to_json(bootstrap_mean(precision, seed))},
             {"tokens_mean", to_json(bootstrap_mean(tokens, seed))},
             {"mrr", to_json(bootstrap_mean(rr, seed))}};
    if (m.name.rfind("trace", 0) == 0) {
      row["trace"] = json{{"evaluated_mean", mean_of(evaluated)}, {"expansions_mean", mean_of(expansions)},
                          {"truncated_share", mean_of(truncated)}, {"low_confidence_share", mean_of(low)}};
    }
    if (have_utility) {
      row["utility"] = json{{"queries_with_useful_units", u_recall.size()},
                            {"recall", to_json(bootstrap_mean(u_recall, seed))},
                            {"precision", to_json(bootstrap_mean(u_precision, seed))},
                            {"precision_definition", "useful units in the set over units of the set other than the relevant unit"}};
    } else {
      row["utility"] = skipped(set.name == "q_gen" ? "no utility labels (chat model needed)" : "utility labels exist for Q-gen only");
    }
    out_methods[m.name] = std::move(row);
  }
  out_methods["trace_nodamp_tau0.25"] =
      skipped("prag::retrieve::TraceOptions has no switch for hub damping; see the report of the bench module (requests)");

  json z_at = json::object();
  for (const double tau : {0.10, 0.25, 0.50}) {
    char label[16];
    std::snprintf(label, sizeof label, "%.2f", tau);
    z_at[label] = cal.threshold(tau);
  }
  json result{{"calibration",
               json{{"fitted", cal.fitted()},
                    {"fit_queries", fit.queries},
                    {"relevant_samples", cal.relevant_samples()},
                    {"nonrelevant_samples", cal.nonrelevant_samples()},
                    {"prior", cal.prior()},
                    {"z_at_tau", z_at},
                    {"conformal", conformal}}},
              {"test_queries", test.size()},
              {"edge_lifts", json{{"callee", c.lifts.callee}, {"caller", c.lifts.caller}, {"sibling", c.lifts.sibling}}},
              {"methods", out_methods}};
  if (set.name == "q_doc") {
    // Upper bound for orientation: the same questions against the index that still contains the documentation
    // (the question text is then part of the relevant unit).
    const std::vector<float>& full = corpus_rows(ctx);
    const std::size_t dim = ctx.dim();
    std::vector<double> top1, top10, rr;
    for (const std::size_t i : test) {
      const Query& q = set.queries[i];
      const float own = prag::index::dot_f32(scored[i].q.data(), &full[static_cast<std::size_t>(q.unit) * dim], dim);
      std::size_t better = 0;
      for (std::size_t j = 0; j < n; ++j)
        if (j != q.unit && prag::index::dot_f32(scored[i].q.data(), &full[j * dim], dim) > own) ++better;
      top1.push_back(better == 0 ? 1.0 : 0.0);
      top10.push_back(better < 10 ? 1.0 : 0.0);
      rr.push_back(1.0 / D(better + 1));
    }
    result["reference_with_documentation_in_index"] =
        json{{"note", "leakage reference, not a result: the relevant unit still contains the question text"},
             {"recall_at_1", to_json(bootstrap_mean(top1, ctx.seed + 101))},
             {"recall_at_10", to_json(bootstrap_mean(top10, ctx.seed + 101))},
             {"mrr", to_json(bootstrap_mean(rr, ctx.seed + 101))}};
  }
  result["codes"] = guarded("codes " + set.name, [&] {
    return code_comparison(ctx, set, scored, cal, env.base, env);
  });
  return result;
}

}  // namespace

int run_e1(Context& ctx) {
  const std::string started = iso_now();
  QuerySets& sets = query_sets(ctx);
  json body = json::object();
  body["q_doc"] = guarded("q_doc", [&] { return evaluate_set(ctx, sets.doc, sets.utility); });
  body["q_gen"] = sets.gen.ok() ? guarded("q_gen", [&] { return evaluate_set(ctx, sets.gen, sets.utility); })
                                : skipped(sets.gen.skipped);
  write_result(ctx, "e1_retrieval", started,
               json{{"bootstrap_resamples", 1000},
                    {"bootstrap_seed", ctx.seed + 100},
                    {"calibration_negatives_per_query", ctx.quick() ? 64 : 400},
                    {"scores", "exact float32 cosine; Q-doc against documentation-free embeddings"},
                    {"ranking_for_mrr", "method order, then the remaining units by score"},
                    {"precision", "relevant unit in the set divided by the set size"}},
               std::move(body));
  return 0;
}

}  // namespace bench
