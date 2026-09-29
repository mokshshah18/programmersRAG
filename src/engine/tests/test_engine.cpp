#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <atomic>
#include <thread>

#include "engine_test_util.hpp"
#include "prag/engine/engine.hpp"
#include "tune_stub.hpp"

using namespace prag;
using namespace prag::engine;
using nlohmann::json;
namespace fsys = std::filesystem;

static const char* kQuestion = "How is the Fletcher checksum of a packet buffer computed?";

static bool first_is_planted(const QueryResult& r, const Corpus& c) {
  return r.error.empty() && !r.units.empty() && c.units[r.units[0].traced.id].name == "fletcher_packet_checksum";
}

PRAG_TEST(build_query_history_persistence) {
  const auto dir = engine_test::make_workspace("basic");
  const std::string settings = (dir / "settings.json").string();
  std::string first_id;
  float first_p = 0.0f;
  {
    Engine e(settings);
    e.open();
    CHECK(!e.index_ready());
    std::vector<std::string> stages;
    const IndexReport rep = e.build_index([&](const IndexProgress& p) {
      if (stages.empty() || stages.back() != p.stage) stages.push_back(p.stage);
      return true;
    });
    REQUIRE(rep.error.empty());
    CHECK_EQ(rep.files, std::size_t{12});
    CHECK(rep.units >= 36);
    CHECK(rep.functions >= 36);
    CHECK(rep.call_edges > 0);
    CHECK_EQ(rep.embedded, rep.units);
    CHECK_EQ(rep.reused, std::size_t{0});
    CHECK(rep.groups >= 1);
    CHECK(rep.seconds.count("total") == 1);
    CHECK(stages.size() >= 8 && stages.front() == "discover" && stages.back() == "done");
    CHECK(e.index_ready());
    const json st = e.status_json();
    CHECK(st["index"]["ready"].get<bool>());
    CHECK_EQ(st["index"]["files"].get<std::size_t>(), std::size_t{12});

    const auto c = e.corpus();
    REQUIRE(c != nullptr);
    CHECK(c->calibration_info.queries + c->calibration_info.heldout_queries > 0);
    std::string streamed;
    bool retrieval_seen = false;
    const QueryResult r = e.query(
        kQuestion, {}, [&](const QueryResult& partial) { retrieval_seen = partial.answer.empty(); },
        [&](std::string_view piece) {
          streamed += piece;
          return true;
        });
    REQUIRE(first_is_planted(r, *c));
    CHECK(retrieval_seen);
    CHECK(r.extractive);
    CHECK_EQ(streamed, r.answer);
    CHECK(r.answer.find("fletcher_packet_checksum") != std::string::npos);
    CHECK(r.answer.find("net/checksum.c:") != std::string::npos);
    CHECK_EQ(r.scan.admission_violations, std::uint64_t{0});
    CHECK_EQ(r.scan.commit_violations, std::uint64_t{0});
    CHECK(!r.selection.entries.empty());
    CHECK(!r.messages.empty());
    first_id = r.id;
    first_p = r.units[0].traced.p;

    const auto h = e.history();
    REQUIRE(h.size() == 1);
    CHECK_EQ(h[0].id, r.id);
    CHECK(e.rate(r.id, 1));
    CHECK_EQ(e.history()[0].rating, 1);
    CHECK(!e.rate("q-none", 1));

    // A second query sees the first as history.
    const QueryResult r2 = e.query("fletcher checksum of a packet", {});
    CHECK_EQ(r2.history_total, std::size_t{1});
    CHECK(r2.id != r.id);

    // An index-relevant setting marks the index stale; a rebuild reuses every embedding.
    e.update_settings(json{{"kde", {{"dim", 8}}}});
    CHECK(!e.index_ready());
    e.update_settings(json{{"generate", {{"max_tokens", 256}}}});
    const IndexReport rep2 = e.build_index();
    REQUIRE(rep2.error.empty());
    CHECK_EQ(rep2.reused, rep2.units);
    CHECK_EQ(rep2.embedded, std::size_t{0});
    CHECK(e.index_ready());
    CHECK_EQ(e.corpus()->reduction.out_dim, std::size_t{8});
    e.update_settings(json{{"kde", {{"dim", 32}}}});
    REQUIRE(e.build_index().error.empty());
    CHECK(first_is_planted(e.query(kQuestion, {false, 0.0, true, false}), *e.corpus()));
  }
  {
    // Persistence round trip.
    Engine e(settings);
    e.open();
    REQUIRE(e.corpus() != nullptr);
    CHECK(e.index_ready());
    CHECK_EQ(e.history().size(), std::size_t{2});
    CHECK_EQ(e.history()[0].rating, 1);
    const QueryResult r = e.query(kQuestion, {false, 0.0, true, false, false});
    REQUIRE(first_is_planted(r, *e.corpus()));
    CHECK_EQ(r.units[0].traced.p, first_p);
    CHECK(r.id != first_id);
    CHECK(e.last_index_report().has_value());
    CHECK(e.clear_history());
    CHECK(e.history().empty());
  }
}

PRAG_TEST(concurrent_queries_during_rebuild) {
  const auto dir = engine_test::make_workspace("concurrent");
  Engine e((dir / "settings.json").string());
  e.open();
  REQUIRE(e.build_index().error.empty());
  std::atomic<bool> stop{false};
  std::atomic<int> good{0}, bad{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 8 || !stop; ++i) {
        const QueryResult r = e.query(kQuestion, {false, 0.0, true, false});
        const auto c = e.corpus();
        if (first_is_planted(r, *c)) ++good;
        else ++bad;
        if (i > 200) break;
      }
    });
  }
  e.update_settings(json{{"kde", {{"dim", 16}}}});
  const IndexReport rep = e.build_index();
  stop = true;
  for (auto& th : threads) th.join();
  CHECK(rep.error.empty());
  CHECK_EQ(bad.load(), 0);
  CHECK(good.load() >= 32);
  CHECK(e.index_ready());
}

PRAG_TEST(edit_flow) {
  const auto dir = engine_test::make_workspace("edit");
  Engine e((dir / "settings.json").string());
  e.open();
  REQUIRE(e.build_index().error.empty());
  const auto file = dir / "corpus" / "ds" / "ring.c";
  const std::string before = engine_test::read_text(file);

  edit::EditBlock b;
  b.kind = edit::EditKind::UnitRewrite;
  b.path = "ds/ring.c";
  b.scope = "ring_reset";
  b.replace =
      "/* Resets the circular ring queue to an empty state without freeing. */\n"
      "int ring_reset(void* self) {\n  (void)self;\n  return zebra_marker_value();\n}\n";
  const edit::Transaction tx = e.preview_edit(std::vector<edit::EditBlock>{b});
  REQUIRE(tx.status == edit::Status::Ok);
  CHECK(!e.transaction_committed(tx.id));
  CHECK_EQ(engine_test::read_text(file), before);

  edit::Transaction applied;
  REQUIRE(e.apply_edit(tx.id, &applied) == edit::Status::Ok);
  CHECK(e.transaction_committed(tx.id));
  const std::string after = engine_test::read_text(file);
  CHECK(after.find("zebra_marker_value") != std::string::npos);
  auto contains_marker = [&] {
    for (const SourceUnit& u : e.corpus()->units)
      if (u.text.find("zebra_marker_value") != std::string::npos) return true;
    return false;
  };
  CHECK(contains_marker());
  const auto rep = e.last_index_report();
  REQUIRE(rep.has_value());
  CHECK(rep->embedded >= 1 && rep->embedded <= 3);
  const QueryResult r = e.query("zebra marker value", {false, 0.0, true, false});
  REQUIRE(r.error.empty() && !r.units.empty());
  CHECK_EQ(e.corpus()->units[r.units[0].traced.id].name, std::string("ring_reset"));

  const json tj = transaction_json(applied, true, e.corpus()->roots);
  CHECK_EQ(tj["status"].get<std::string>(), std::string("ok"));
  CHECK_EQ(tj["files"][0]["display_path"].get<std::string>(), std::string("ds/ring.c"));

  REQUIRE(e.revert_edit(tx.id) == edit::Status::Ok);
  CHECK(!e.transaction_committed(tx.id));
  CHECK_EQ(engine_test::read_text(file), before);
  CHECK(!contains_marker());

  // Edit text as a model would write it.
  std::vector<std::string> warnings;
  const edit::Transaction tx2 = e.preview_edit(
      "ds/list.c\n<<<<<<< SEARCH\n  (void)self;\n  return 0;\n=======\n  (void)self;\n  return 1;\n>>>>>>> REPLACE\n",
      &warnings);
  CHECK(tx2.status == edit::Status::Ok);
  CHECK(e.apply_edit("tx-unknown") == edit::Status::NotFound);
}

// Checks that `j[key]` exists and has the type tested by `is`.
#define HAS(j, key, is) CHECK((j).contains(key) && (j)[key].is())

PRAG_TEST(json_views) {
  const auto dir = engine_test::make_workspace("json");
  Engine e((dir / "settings.json").string());
  e.open();
  const json progress0 = e.index_progress_json();
  CHECK(progress0["report"].is_null());
  REQUIRE(e.build_index().error.empty());

  const json st = e.status_json();
  HAS(st, "version", is_string);
  HAS(st, "index", is_object);
  HAS(st, "models", is_object);
  HAS(st, "cache", is_object);
  HAS(st, "tune", is_object);
  HAS(st, "roots", is_array);
  const json& ix = st["index"];
  for (const char* k : {"ready", "building", "calibrated"}) HAS(ix, k, is_boolean);
  for (const char* k : {"files", "units", "functions", "call_edges", "groups", "dim", "reduced_dim", "blocks",
                        "block_bytes"})
    HAS(ix, k, is_number_unsigned);
  HAS(ix, "kappa", is_number);
  for (const char* k : {"codes", "embedder", "built_at"}) HAS(ix, k, is_string);
  for (const char* k : {"llama", "chat_loaded", "embedding_loaded"}) HAS(st["models"], k, is_boolean);
  for (const char* k : {"chat", "backend"}) HAS(st["models"], k, is_string);
  const json& ca = st["cache"];
  HAS(ca, "detected", is_boolean);
  for (const char* k : {"logical_cpus", "l1d_min_bytes", "l1d_per_thread_min_bytes", "line_bytes", "block_bytes",
                        "threads"})
    HAS(ca, k, is_number_unsigned);
  HAS(ca, "simd", is_string);
  HAS(ca, "classes", is_array);

  const json pr = e.index_progress_json();
  for (const char* k : {"running"}) HAS(pr, k, is_boolean);
  for (const char* k : {"stage", "message"}) HAS(pr, k, is_string);
  for (const char* k : {"done", "total"}) HAS(pr, k, is_number_unsigned);
  HAS(pr, "report", is_object);
  const json& rp = pr["report"];
  for (const char* k : {"files", "units", "functions", "call_edges", "embedded", "reused", "groups", "reduced_dim",
                        "blocks", "block_bytes"})
    HAS(rp, k, is_number_unsigned);
  for (const char* k : {"kappa", "explained_variance", "anisotropy_before", "anisotropy_after"}) HAS(rp, k, is_number);
  HAS(rp, "seconds", is_object);
  for (const char* k : {"discover", "parse", "graph", "embed", "density", "codes", "calibrate", "save", "total"})
    HAS(rp["seconds"], k, is_number);
  HAS(rp, "error", is_string);

  const json g = e.groups_json();
  HAS(g, "kappa", is_number);
  HAS(g, "reduced_dim", is_number_unsigned);
  REQUIRE(g.contains("groups") && !g["groups"].empty() && g.contains("units") && !g["units"].empty());
  const json& g0 = g["groups"][0];
  for (const char* k : {"id", "size"}) HAS(g0, k, is_number_unsigned);
  for (const char* k : {"log_density", "x", "y"}) HAS(g0, k, is_number);
  HAS(g0, "label", is_string);
  for (const json& u : g["units"]) {
    CHECK(u["x"].get<double>() >= -1.0 && u["x"].get<double>() <= 1.0);
    CHECK(u["y"].get<double>() >= -1.0 && u["y"].get<double>() <= 1.0);
  }
  const json& gu = g["units"][0];
  for (const char* k : {"id", "group"}) HAS(gu, k, is_number_unsigned);
  for (const char* k : {"x", "y", "log_density"}) HAS(gu, k, is_number);
  for (const char* k : {"name", "path", "kind"}) HAS(gu, k, is_string);

  const json u = e.unit_json(0);
  for (const char* k : {"id", "start_line", "end_line", "group", "tokens"}) HAS(u, k, is_number_unsigned);
  for (const char* k : {"path", "name", "kind", "signature", "doc", "text"}) HAS(u, k, is_string);
  for (const char* k : {"callees", "callers"}) HAS(u, k, is_array);
  CHECK(e.unit_json(1u << 30).contains("error"));

  const QueryResult r = e.query(kQuestion, {});
  const json rj = retrieval_json(r, *e.corpus());
  HAS(rj, "id", is_string);
  for (const char* k : {"tau", "z_threshold"}) HAS(rj, k, is_number);
  for (const char* k : {"low_confidence", "truncated"}) HAS(rj, k, is_boolean);
  REQUIRE(rj.contains("units") && !rj["units"].empty());
  const json& ru = rj["units"][0];
  for (const char* k : {"id", "start_line", "end_line", "group", "tokens", "hops"}) HAS(ru, k, is_number_unsigned);
  for (const char* k : {"score", "z", "p_score", "p", "redundancy", "quality"}) HAS(ru, k, is_number);
  for (const char* k : {"path", "name", "kind", "via", "form", "preview"}) HAS(ru, k, is_string);
  for (const char* k : {"in_context", "support"}) HAS(ru, k, is_boolean);
  CHECK(ru.contains("parent") && (ru["parent"].is_null() || ru["parent"].is_number_unsigned()));
  const json& cx = rj["context"];
  for (const char* k : {"tokens_before", "tokens_after", "passes", "dropped_low_quality", "dropped_redundant",
                        "dropped_budget", "skeletonised", "support_added", "history_kept", "history_total"})
    HAS(cx, k, is_number_unsigned);
  for (const char* k : {"quality_before", "quality_after", "rot_before", "rot_after", "compression"})
    HAS(cx, k, is_number);
  HAS(cx, "groups", is_array);
  if (!cx["groups"].empty()) {
    HAS(cx["groups"][0], "path", is_string);
    HAS(cx["groups"][0], "tokens", is_number_unsigned);
    HAS(cx["groups"][0], "quality", is_number);
    HAS(cx["groups"][0], "units", is_number_unsigned);
  }
  const json& sc = rj["scan"];
  for (const char* k : {"blocks", "block_bytes", "threads", "wall_us", "work_us", "gate_wait_us",
                        "admission_violations", "commit_violations", "rescored"})
    HAS(sc, k, is_number_unsigned);
  for (const char* k : {"codes", "simd"}) HAS(sc, k, is_string);
  for (const char* k : {"seeds", "evaluated", "expansions"}) HAS(rj["trace"], k, is_number_unsigned);

  const json dj = done_json(r);
  for (const char* k : {"id", "answer", "stop_reason", "model"}) HAS(dj, k, is_string);
  for (const char* k : {"prompt_tokens", "completion_tokens"}) HAS(dj, k, is_number_integer);
  for (const char* k : {"prompt_ms", "completion_ms", "total_ms"}) HAS(dj, k, is_number);
  HAS(dj, "extractive", is_boolean);

  const json hj = e.history_json();
  REQUIRE(hj.is_array() && hj.size() == 1);
  for (const char* k : {"id", "time", "question", "answer"}) HAS(hj[0], k, is_string);
  HAS(hj[0], "units", is_array);
  HAS(hj[0], "rating", is_number_integer);

  edit::EditBlock b;
  b.path = "ds/bits.c";
  b.search = "  (void)self;\n  return 0;\n";
  b.replace = "  (void)self;\n  return 2;\n";
  const edit::Transaction tx = e.preview_edit(std::vector<edit::EditBlock>{b});
  const json tj = transaction_json(tx, false, e.corpus()->roots, {"w"});
  for (const char* k : {"id", "status"}) HAS(tj, k, is_string);
  HAS(tj, "committed", is_boolean);
  HAS(tj, "warnings", is_array);
  REQUIRE(tj.contains("files") && tj["files"].size() == 1);
  const json& tf = tj["files"][0];
  for (const char* k : {"path", "display_path", "status", "diff", "message"}) HAS(tf, k, is_string);
  HAS(tf, "creates_file", is_boolean);
  for (const char* k : {"syntax_errors_before", "syntax_errors_after"}) HAS(tf, k, is_number_unsigned);
  REQUIRE(tf.contains("matches") && tf["matches"].size() == 1);
  HAS(tf["matches"][0], "kind", is_string);
  for (const char* k : {"first_line", "last_line", "distance", "candidates"})
    HAS(tf["matches"][0], k, is_number_unsigned);

  CHECK_EQ(display_path("C:/a/b/c.c", {"C:/a"}), std::string("b/c.c"));
  CHECK_EQ(display_path("C:/x/c.c", {"C:/a"}), std::string("C:/x/c.c"));
}
