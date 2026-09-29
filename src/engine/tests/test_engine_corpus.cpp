// Engine on the mimalloc corpus with the hashing embedder; skipped when the corpus is missing.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <cstring>
#include <filesystem>

#include "engine_test_util.hpp"
#include "prag/engine/engine.hpp"
#include "tune_stub.hpp"

using namespace prag;
using namespace prag::engine;
using nlohmann::json;
namespace fsys = std::filesystem;

static const char* kQuestion = "How is the thread free list of a page collected with an atomic exchange?";

struct Outcome {
  std::vector<UnitId> ids;
  std::vector<float> p;
  std::vector<float> score;
  std::size_t in_context = 0;
  std::string answer;
};

static Outcome run(unsigned threads, IndexReport* report, QueryResult* result) {
  const auto dir = fsys::temp_directory_path() / ("prag_engine_mi" + std::to_string(threads));
  std::error_code ec;
  fsys::remove_all(dir, ec);
  fsys::create_directories(dir);
  const json settings{{"corpus", {{"roots", {PRAG_DEMO_REPO}}}},
                      {"paths", {{"data", "data"}}},
                      {"exec", {{"threads", threads}}}};
  engine_test::write_text(dir / "settings.json", settings.dump(2));
  Engine e((dir / "settings.json").string());
  e.open();
  *report = e.build_index();
  Outcome o;
  if (!report->error.empty()) return o;
  *result = e.query(kQuestion, {false, 0.0, true, false});
  {
    const CalibrationInfo& ci = e.corpus()->calibration_info;
    std::printf("calibration (%u threads): fitted %d, %s, %zu + %zu queries, ece %.3f, conformal z %.2f; "
                "query z_threshold %.2f, top z %.2f, p_score %.3f
",
                threads, ci.fitted, ci.source.c_str(), ci.queries, ci.heldout_queries,
                ci.heldout.expected_calibration_error, ci.conformal_z, result->trace.z_threshold,
                result->units.empty() ? 0.0 : result->units[0].traced.z,
                result->units.empty() ? 0.0 : result->units[0].traced.p_score);
  }
  for (const RetrievedUnit& u : result->units) {
    o.ids.push_back(u.traced.id);
    o.p.push_back(u.traced.p);
    o.score.push_back(u.traced.score);
    o.in_context += u.in_context;
  }
  o.answer = result->answer;
  return o;
}

PRAG_TEST(mimalloc_build_and_thread_invariance) {
  if (!fsys::exists(PRAG_DEMO_REPO)) {
    std::printf("SKIP: corpus not found at %s\n", PRAG_DEMO_REPO);
    return;
  }
  IndexReport rep24;
  QueryResult q24;
  const Outcome o24 = run(24, &rep24, &q24);
  REQUIRE(rep24.error.empty());
  std::printf("mimalloc: %zu files, %zu units, %zu functions, %zu call edges, %zu groups, kappa %.2f, "
              "reduced %zu (explained %.3f), anisotropy %.3f -> %.3f, %zu blocks of %zu bytes\n",
              rep24.files, rep24.units, rep24.functions, rep24.call_edges, rep24.groups, rep24.kappa,
              rep24.reduced_dim, rep24.explained_variance, rep24.anisotropy_before, rep24.anisotropy_after,
              rep24.blocks, rep24.block_bytes);
  for (const auto& [stage, s] : rep24.seconds) std::printf("  %-10s %8.3f s\n", stage.c_str(), s);
  CHECK(rep24.files >= 40);
  CHECK(rep24.units >= 800);
  CHECK(rep24.functions >= 500);
  CHECK(rep24.call_edges >= 1000);
  CHECK(rep24.groups >= 2 && rep24.groups < rep24.units / 3);
  CHECK(rep24.blocks >= 1);
  REQUIRE(q24.error.empty());
  REQUIRE(!o24.ids.empty());
  CHECK(o24.in_context >= 1);
  CHECK(q24.trace.seeds >= 1);
  CHECK_EQ(q24.scan.admission_violations, std::uint64_t{0});
  CHECK_EQ(q24.scan.commit_violations, std::uint64_t{0});
  CHECK_EQ(q24.scan.threads, 24u);
", q24.trace.z_threshold > 0, 0ul, 0.0, 0.0, q24.trace.z_threshold, q24.units.empty() ? 0.0 : q24.units[0].traced.z, q24.units.empty() ? 0.0 : q24.units[0].traced.p_score);
  std::printf("query: %zu units (%zu seeds, %zu evaluated), %zu rescored, retrieve %.1f ms\n", o24.ids.size(),
              q24.trace.seeds, q24.trace.evaluated, q24.scan.rescored, q24.retrieve_ms);

  for (const unsigned threads : {1u, 4u}) {
    IndexReport rep;
    QueryResult q;
    const Outcome o = run(threads, &rep, &q);
    REQUIRE(rep.error.empty());
    CHECK_EQ(rep.units, rep24.units);
    CHECK_EQ(rep.groups, rep24.groups);
    CHECK_EQ(q.scan.admission_violations, std::uint64_t{0});
    CHECK_EQ(q.scan.commit_violations, std::uint64_t{0});
    CHECK(o.ids == o24.ids);
    CHECK(o.p.size() == o24.p.size() && std::memcmp(o.p.data(), o24.p.data(), o.p.size() * sizeof(float)) == 0);
    CHECK(o.score.size() == o24.score.size() &&
          std::memcmp(o.score.data(), o24.score.data(), o.score.size() * sizeof(float)) == 0);
    CHECK_EQ(o.answer, o24.answer);
  }
}
