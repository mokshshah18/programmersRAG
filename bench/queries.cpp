// `prag-bench queries`: Q-doc, Q-gen, the split, and the utility labels.
#include <unordered_set>

#include "common.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"
#include "prag/llm/model.hpp"

namespace bench {

namespace {

constexpr double kMinGain = 0.02;  // nats per token, docs/EXPERIMENTS.md

std::string generate_question(prag::llm::LanguageModel& chat, const prag::SourceUnit& unit, bool strict) {
  std::string body = strip_leading_comment(unit.text, unit.language);
  if (body.size() > 6000) body.resize(6000);
  const std::vector<prag::ChatMessage> messages = {
      {"system", "You write questions about source code."},
      {"user",
       "Write one question a developer would ask that this function answers. Do not use the function name. "
       "Reply with the question only." +
           (strict ? " The question must not contain the word `" + unit.name +
                         "` and must not quote any identifier of the code; describe what the function does in plain words."
                   : std::string()) +
           "\n\n```\n" + body + "\n```"}};
  prag::llm::GenerateOptions go;
  go.max_tokens = 64;
  go.temperature = 0.0f;
  std::string out;
  chat.generate(messages, go, [&](std::string_view piece) {
    out += piece;
    return true;
  });
  for (const std::string_view line : prag::text::split_lines(out)) {
    const std::string_view t = prag::text::trim(line);
    if (!t.empty()) return std::string(t);
  }
  return {};
}

std::string reference_answer(const prag::SourceUnit& u) {
  std::string doc(prag::text::trim(u.doc));
  if (doc.size() > 600) doc.resize(600);
  return doc;
}

json utility_labels(Context& ctx, const QuerySet& gen, std::size_t subset) {
  const auto chat = ctx.chat();
  if (!chat) return skipped("no chat model loaded");
  if (!gen.ok()) return skipped("Q-gen is not available");
  const prag::engine::Corpus& c = *ctx.corpus;
  const std::size_t per_kind = 3;
  json rows = json::array();
  std::size_t done = 0;
  std::uint64_t state = prag::hash_combine(ctx.seed, 0x77);
  for (const Query& q : gen.queries) {
    if (q.split != kTest || done >= subset) continue;
    if (q.reference.empty()) continue;
    const auto oracle_messages = messages_for(ctx, q.question, {q.unit}, false);
    const double base = answer_loss(*chat, oracle_messages, q.reference);
    if (!std::isfinite(base)) continue;
    std::vector<std::pair<UnitId, const char*>> candidates;
    std::unordered_set<UnitId> seen{q.unit};
    auto add = [&](std::span<const UnitId> ids, const char* kind) {
      std::size_t taken = 0;
      for (const UnitId v : ids) {
        if (taken >= per_kind) break;
        if (!seen.insert(v).second) continue;
        candidates.emplace_back(v, kind);
        ++taken;
      }
    };
    add(c.calls.callees(q.unit), "callee");
    add(c.calls.callers(q.unit), "caller");
    add(c.siblings.of(q.unit), "group");
    const std::size_t structural = candidates.size();
    for (std::size_t tries = 0; tries < structural * 8 + 8 && candidates.size() < 2 * structural; ++tries) {
      const auto v = static_cast<UnitId>(rng_below(state, ctx.n()));
      if (seen.insert(v).second) candidates.emplace_back(v, "random");
    }
    json cand = json::array();
    std::size_t useful = 0;
    for (const auto& [v, kind] : candidates) {
      const double with = answer_loss(*chat, messages_for(ctx, q.question, {q.unit, v}, false), q.reference);
      const double gain = base - with;
      const bool is_useful = std::isfinite(gain) && gain >= kMinGain;
      useful += is_useful ? 1 : 0;
      cand.push_back(json{{"unit", v}, {"kind", kind}, {"gain", gain}, {"useful", is_useful}});
    }
    rows.push_back(json{{"query", q.id}, {"unit", q.unit}, {"base_nll", base}, {"useful", useful}, {"candidates", cand}});
    ++done;
    progress("utility labels %zu / %zu", done, subset);
  }
  return json{{"min_gain", kMinGain}, {"queries", rows}};
}

}  // namespace

void build_query_sets(Context& ctx, QuerySets* sets) {
  sets->doc = QuerySet{};
  sets->gen = QuerySet{};
  sets->doc.name = "q_doc";
  sets->gen.name = "q_gen";
  const auto& splits = unit_splits(ctx);
  const auto& units = ctx.corpus->units;
  json info = json::object();

  // Q-doc: the documentation is the question, the function is the relevant unit.
  std::vector<UnitId> documented;
  for (const auto& u : units)
    if (splits[u.id] >= 0 && count_words(u.doc) >= 6) documented.push_back(u.id);
  info["doc_eligible"] = documented.size();
  const std::size_t doc_cap = ctx.quick() ? 80 : documented.size();
  {
    std::vector<UnitId> pool = documented;
    shuffle(pool, ctx.seed + 1);
    if (pool.size() > doc_cap) pool.resize(doc_cap);
    std::sort(pool.begin(), pool.end());
    for (const UnitId id : pool) {
      Query q;
      q.id = "doc-" + std::to_string(id);
      q.unit = id;
      q.question = std::string(prag::text::trim(units[id].doc));
      q.split = splits[id];
      sets->doc.queries.push_back(std::move(q));
    }
  }

  // Q-gen: a question written by the chat model from the body of a documented function.
  const auto chat = ctx.chat();
  if (!chat) {
    sets->gen.skipped = "no chat model loaded";
  } else {
    std::vector<UnitId> pool;
    for (const UnitId id : documented)
      if (units[id].tokens >= 16) pool.push_back(id);
    shuffle(pool, ctx.seed + 2);
    const std::size_t want = ctx.quick() ? 8 : 240;
    std::size_t attempted = 0, discarded = 0;
    for (const UnitId id : pool) {
      if (sets->gen.queries.size() >= want) break;
      ++attempted;
      std::string question = generate_question(*chat, units[id], false);
      if (question.empty() || prag::text::contains_icase(question, units[id].name))
        question = generate_question(*chat, units[id], true);  // second attempt with a stricter instruction
      if (question.empty() || prag::text::contains_icase(question, units[id].name)) {
        ++discarded;
        continue;
      }
      Query q;
      q.id = "gen-" + std::to_string(id);
      q.unit = id;
      q.question = question;
      q.split = splits[id];
      q.reference = reference_answer(units[id]);
      sets->gen.queries.push_back(std::move(q));
      if (sets->gen.queries.size() % 20 == 0) progress("Q-gen %zu / %zu", sets->gen.queries.size(), want);
    }
    info["gen_attempted"] = attempted;
    info["gen_discarded_empty_or_name"] = discarded;
    if (sets->gen.queries.empty()) sets->gen.skipped = "the chat model produced no usable question";
  }

  sets->utility = utility_labels(ctx, sets->gen, ctx.quick() ? 2 : 40);
  info["doc_min_words"] = 6;
  info["gen_requested"] = ctx.quick() ? 8 : 240;
  info["split_fractions"] = json{{"calibration", 0.5}, {"heldout", 0.25}, {"test", 0.25}};
  info["split_seed"] = ctx.seed;
  sets->info = info;
}

int run_queries(Context& ctx) {
  const std::string started = iso_now();
  auto sets = std::make_unique<QuerySets>();
  sets->fingerprint = corpus_fingerprint(ctx);
  build_query_sets(ctx, sets.get());
  json body = query_sets_json(ctx, *sets);
  write_result(ctx, "queries", started,
               json{{"min_gain_nats_per_token", kMinGain},
                    {"doc_min_words", 6},
                    {"note", "Q-gen questions and utility labels need the chat model"}},
               std::move(body));
  ctx.sets = std::move(sets);
  return 0;
}

}  // namespace bench
