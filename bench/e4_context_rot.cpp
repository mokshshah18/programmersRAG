// E4: context rot. Loss of the reference answer under the serving model for
// contexts with distractors, top-k, the trace and the pruned trace.
#include <map>
#include <unordered_set>

#include "common.hpp"
#include "prag/context/prompt.hpp"
#include "prag/context/rot.hpp"
#include "prag/retrieve/trace.hpp"

namespace bench {

namespace {

namespace cx = prag::context;
namespace rt = prag::retrieve;

struct Row {
  std::string query, condition;
  std::size_t distractors = 0;
  double tokens = 0, quality = kNaN, rot = kNaN, loss = kNaN, prompt_ms = kNaN, prompt_tokens = kNaN;
  bool truncated = false;
  std::size_t units = 0;
};

struct Builder {
  Context& ctx;
  prag::llm::LanguageModel& chat;
  const rt::Calibration& cal;
  cx::Options options;

  cx::Candidate candidate(UnitId id, double p) const {
    const auto& c = *ctx.corpus;
    const prag::SourceUnit& u = c.units[id];
    cx::Candidate cand;
    cand.id = id;
    cand.group = c.file_of_unit[id];
    cand.p = p;
    cand.tokens_full = static_cast<std::uint32_t>(std::max(0, chat.count_tokens(u.text)));
    cand.tokens_skeleton = static_cast<std::uint32_t>(std::max(0, chat.count_tokens(cx::skeleton_of(u))));
    cand.embedding = c.store.vector(id).data();
    const auto callees = c.calls.callees(id);
    cand.requires_units.assign(callees.begin(), callees.end());
    return cand;
  }

  // Files by their strongest unit (descending), units inside a file in source order.
  std::vector<UnitId> order(std::vector<UnitId> ids, const std::map<UnitId, double>& p) const {
    const auto& c = *ctx.corpus;
    std::map<std::uint32_t, double> strongest;
    for (const UnitId id : ids) {
      auto& s = strongest[c.file_of_unit[id]];
      s = std::max(s, p.at(id));
    }
    std::stable_sort(ids.begin(), ids.end(), [&](UnitId a, UnitId b) {
      const auto fa = c.file_of_unit[a], fb = c.file_of_unit[b];
      if (fa != fb) return strongest[fa] != strongest[fb] ? strongest[fa] > strongest[fb] : fa < fb;
      return c.units[a].start_byte < c.units[b].start_byte;
    });
    return ids;
  }
};

}  // namespace

int run_e4(Context& ctx) {
  const std::string started = iso_now();
  const json settings{{"conditions", "none, oracle, oracle + k hard (2,4,8,16), oracle + k random, top-k (5,10,20), trace, trace + RCQ"},
                      {"reference_answer", "documentation of the oracle function"},
                      {"loss", "negative log-likelihood per token of the reference answer after the prompt, serving model"},
                      {"queries", "Q-gen test queries with a reference answer"}};
  const auto chat = ctx.chat();
  if (!chat) {
    write_result(ctx, "e4_context_rot", started, settings, skipped("no chat model loaded"));
    return 0;
  }
  QuerySets& sets = query_sets(ctx);
  if (!sets.gen.ok()) {
    write_result(ctx, "e4_context_rot", started, settings, skipped("Q-gen is not available: " + sets.gen.skipped));
    return 0;
  }
  const auto& c = *ctx.corpus;
  const std::size_t n = ctx.n();
  const std::vector<Scored> scored = score_queries(ctx, sets.gen);
  const Labeled fit = collect_labeled(sets.gen, scored, kCalibration, ctx.quick() ? 64 : 400, ctx.seed + 3);
  rt::Calibration cal = rt::Calibration::fallback();
  if (fit.queries >= 20) cal = rt::Calibration::fit(fit.z_relevant, fit.z_nonrelevant, 1.0 / D(n));

  cx::Options co;
  co.theta = ctx.engine->settings().get<double>("context.theta", 0.25);
  co.token_budget = ctx.engine->settings().get<std::uint32_t>("context.token_budget", 2400);
  co.full_text_share = ctx.engine->settings().get<double>("context.full_text_share", 0.5);
  co.unit_overhead = ctx.engine->settings().get<std::uint32_t>("context.unit_overhead", 25);
  co.group_overhead = ctx.engine->settings().get<std::uint32_t>("context.group_overhead", 8);
  co.dim = ctx.dim();
  Builder builder{ctx, *chat, cal, co};
  rt::TraceOptions to;
  to.max_units = ctx.engine->settings().get<std::size_t>("retrieve.max_units", 64);
  to.max_hops = ctx.engine->settings().get<std::size_t>("retrieve.max_hops", 6);
  to.lift = c.lifts;

  std::vector<std::size_t> test;
  for (std::size_t i = 0; i < sets.gen.queries.size(); ++i)
    if (sets.gen.queries[i].split == kTest && scored[i].ok && !sets.gen.queries[i].reference.empty()) test.push_back(i);
  const std::size_t cap = ctx.quick() ? 2 : 40;
  if (test.size() > cap) test.resize(cap);
  progress("E4: %zu test queries", test.size());

  prag::llm::GenerateOptions probe;
  probe.max_tokens = 1;
  probe.temperature = 0.0f;
  std::vector<Row> rows;
  std::size_t done = 0;
  for (const std::size_t qi : test) {
    const Query& q = sets.gen.queries[qi];
    const Scored& s = scored[qi];
    std::map<UnitId, double> p;
    auto prob = [&](UnitId id) {
      auto it = p.find(id);
      if (it == p.end()) it = p.emplace(id, cal.posterior(s.z(id))).first;
      return it->second;
    };
    // Rankings of the non-relevant units: by score (hard) and a fixed random order.
    std::vector<UnitId> by_score;
    for (std::size_t j = 0; j < n; ++j)
      if (j != q.unit) by_score.push_back(static_cast<UnitId>(j));
    std::stable_sort(by_score.begin(), by_score.end(), [&](UnitId a, UnitId b) { return s.score[a] > s.score[b]; });
    std::vector<UnitId> random_order = by_score;
    shuffle(random_order, ctx.seed + 400 + qi);

    struct Condition {
      std::string name;
      std::size_t distractors;
      std::vector<UnitId> ids;
      bool rcq = false;
    };
    std::vector<Condition> conditions;
    conditions.push_back({"none", 0, {}, false});
    conditions.push_back({"oracle", 0, {q.unit}, false});
    for (const std::size_t k : {2u, 4u, 8u, 16u}) {
      std::vector<UnitId> hard{q.unit};
      hard.insert(hard.end(), by_score.begin(), by_score.begin() + static_cast<std::ptrdiff_t>(k));
      conditions.push_back({"oracle+hard" + std::to_string(k), k, hard, false});
      std::vector<UnitId> rnd{q.unit};
      rnd.insert(rnd.end(), random_order.begin(), random_order.begin() + static_cast<std::ptrdiff_t>(k));
      conditions.push_back({"oracle+random" + std::to_string(k), k, rnd, false});
    }
    for (const std::size_t k : {5u, 10u, 20u}) {
      const auto top = rt::top_k(s.score, s.mean, s.sd, cal, k);
      std::vector<UnitId> ids;
      std::size_t distractors = 0;
      for (const auto& t : top) {
        ids.push_back(t.id);
        distractors += t.id == q.unit ? 0 : 1;
      }
      conditions.push_back({"top" + std::to_string(k), distractors, ids, false});
    }
    const rt::TraceResult tr = rt::trace(s.score, s.mean, s.sd, cal, c.calls, c.siblings, to);
    {
      std::vector<UnitId> ids;
      std::size_t distractors = 0;
      for (const auto& t : tr.units) {
        ids.push_back(t.id);
        distractors += t.id == q.unit ? 0 : 1;
      }
      conditions.push_back({"trace", distractors, ids, false});
      conditions.push_back({"trace+rcq", distractors, ids, true});
    }

    const auto oracle_messages = messages_for(ctx, q.question, {q.unit}, false);
    (void)oracle_messages;
    for (const Condition& cond : conditions) {
      Row row;
      row.query = q.id;
      row.condition = cond.name;
      row.distractors = cond.distractors;
      cx::Selection selection;
      if (cond.rcq) {
        std::vector<cx::Candidate> candidates;
        std::unordered_set<UnitId> have;
        for (const UnitId id : cond.ids) {
          candidates.push_back(builder.candidate(id, prob(id)));
          have.insert(id);
        }
        for (const UnitId id : cond.ids)
          for (const UnitId v : c.calls.callees(id))
            if (have.insert(v).second) candidates.push_back(builder.candidate(v, prob(v)));
        selection = cx::select(candidates, co);
      } else if (!cond.ids.empty()) {
        const std::vector<UnitId> ordered = builder.order(cond.ids, [&] {
          std::map<UnitId, double> pm;
          for (const UnitId id : cond.ids) pm[id] = prob(id);
          return pm;
        }());
        std::vector<cx::Candidate> candidates;
        for (const UnitId id : ordered) candidates.push_back(builder.candidate(id, prob(id)));
        selection = cx::evaluate(candidates, co);
      }
      row.units = selection.entries.size();
      row.tokens = static_cast<double>(selection.tokens_after);
      row.quality = cond.ids.empty() ? kNaN : selection.quality_after;
      row.rot = cond.ids.empty() ? kNaN : selection.rot_after;
      cx::PromptOptions po;
      po.root = c.roots.empty() ? std::string() : c.roots.front();
      const auto messages = cx::build_messages(q.question, c.units, selection, {}, {}, po);
      row.loss = answer_loss(*chat, messages, q.reference, &row.truncated);
      const prag::llm::GenerateStats g = chat->generate(messages, probe, [](std::string_view) { return true; });
      row.prompt_ms = g.prompt_ms;
      row.prompt_tokens = static_cast<double>(g.prompt_tokens);
      rows.push_back(row);
    }
    progress("E4 query %zu / %zu", ++done, test.size());
  }

  // Aggregation.
  std::map<std::string, std::vector<const Row*>> by_condition;
  std::map<std::string, double> oracle_loss;
  for (const Row& r : rows) {
    by_condition[r.condition].push_back(&r);
    if (r.condition == "oracle") oracle_loss[r.query] = r.loss;
  }
  json conditions = json::object();
  json curve = json::array();
  for (const auto& [name, list] : by_condition) {
    std::vector<double> loss, tokens, quality, rot, ms, distractors;
    std::size_t truncated = 0;
    for (const Row* r : list) {
      loss.push_back(r->loss);
      tokens.push_back(r->tokens);
      quality.push_back(r->quality);
      rot.push_back(r->rot);
      ms.push_back(r->prompt_ms);
      distractors.push_back(D(r->distractors));
      truncated += r->truncated ? 1 : 0;
    }
    const std::uint64_t seed = ctx.seed + 500;
    conditions[name] = json{{"queries", list.size()},
                            {"loss", to_json(bootstrap_mean(loss, seed))},
                            {"tokens", to_json(bootstrap_mean(tokens, seed))},
                            {"quality", to_json(bootstrap_mean(quality, seed))},
                            {"rot_index", to_json(bootstrap_mean(rot, seed))},
                            {"prompt_ms", to_json(bootstrap_mean(ms, seed))},
                            {"distractors_mean", mean_of(distractors)},
                            {"truncated_prompts", truncated}};
    curve.push_back(json{{"condition", name}, {"distractors_mean", mean_of(distractors)}, {"tokens_mean", mean_of(tokens)},
                         {"loss_mean", mean_of(loss)}});
  }
  std::vector<double> rot_x, increase, tokens_x, ms_y;
  for (const Row& r : rows) {
    if (std::isfinite(r.prompt_ms) && r.tokens > 0) {
      tokens_x.push_back(r.prompt_tokens);
      ms_y.push_back(r.prompt_ms);
    }
    if (r.condition == "none" || r.condition == "oracle" || !std::isfinite(r.rot)) continue;
    const auto it = oracle_loss.find(r.query);
    if (it == oracle_loss.end()) continue;
    rot_x.push_back(r.rot);
    increase.push_back(r.loss - it->second);
  }
  json per_context = json::array();
  for (const Row& r : rows)
    per_context.push_back(json{{"query", r.query}, {"condition", r.condition}, {"distractors", r.distractors},
                               {"units", r.units}, {"tokens", r.tokens}, {"quality", r.quality}, {"rot_index", r.rot},
                               {"loss", r.loss}, {"prompt_ms", r.prompt_ms}, {"prompt_tokens", r.prompt_tokens},
                               {"truncated", r.truncated}});
  json body{{"queries", test.size()},
            {"conditions", conditions},
            {"rot_curve", curve},
            {"spearman_rot_vs_loss_increase", json{{"rho", spearman(rot_x, increase)}, {"n", rot_x.size()}}},
            {"latency_vs_tokens", json{{"spearman_prompt_tokens_vs_prompt_ms", spearman(tokens_x, ms_y)}, {"n", tokens_x.size()}}},
            {"contexts", per_context},
            {"calibration", json{{"fitted", cal.fitted()}, {"fit_queries", fit.queries}}}};
  write_result(ctx, "e4_context_rot", started,
               json{{"conditions", settings["conditions"]},
                    {"reference_answer", settings["reference_answer"]},
                    {"loss", settings["loss"]},
                    {"max_test_queries", cap},
                    {"theta", co.theta},
                    {"token_budget", co.token_budget},
                    {"context_order", "files by strongest unit, source order inside a file (no edge ordering) for unpruned contexts; RCQ selection order for the pruned trace"},
                    {"hard_distractors", "highest-scoring non-relevant units"},
                    {"random_seed", ctx.seed + 400}},
               std::move(body));
  return 0;
}

}  // namespace bench
