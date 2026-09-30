// prag-bench <experiment> --settings prag.json --out paper/results [--quick]
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>

#include "common.hpp"
#include "prag/core/clock.hpp"

namespace {

struct Experiment {
  const char* name;
  int (*run)(bench::Context&);
};

const Experiment kExperiments[] = {
    {"queries", bench::run_queries},          {"e1_retrieval", bench::run_e1}, {"e2_calibration", bench::run_e2},
    {"e3_groups", bench::run_e3},             {"e4_context_rot", bench::run_e4}, {"e5_execution", bench::run_e5},
    {"e6_tuning", bench::run_e6},             {"e7_edits", bench::run_e7},
};

void usage() {
  std::fprintf(stderr,
               "usage: prag-bench <experiment> [--settings prag.json] [--out paper/results] [--quick]\n"
               "experiments: queries e1_retrieval e2_calibration e3_groups e4_context_rot e5_execution e6_tuning "
               "e7_edits all\n");
}

int run_all(bench::Context& ctx) {
  using bench::json;
  json summary = json::array();
  int failures = 0;
  const std::string started = bench::iso_now();
  for (const Experiment& e : kExperiments) {
    bench::progress("=== %s ===", e.name);
    prag::Stopwatch watch;
    json row{{"experiment", e.name}};
    try {
      const int rc = e.run(ctx);
      row["status"] = rc == 0 ? "ok" : "failed";
      if (rc != 0) {
        row["error"] = "exit code " + std::to_string(rc);
        ++failures;
      }
    } catch (const std::exception& ex) {
      row["status"] = "failed";
      row["error"] = ex.what();
      ++failures;
      bench::progress("%s failed: %s", e.name, ex.what());
    }
    row["seconds"] = watch.elapsed_s();
    summary.push_back(row);
  }
  bench::write_result(ctx, "all", started, json{{"experiments", "queries, e1 .. e7 in order"}},
                      json{{"runs", summary}, {"failures", failures}});
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  bench::Args args;
  if (argc < 2) {
    usage();
    return 2;
  }
  args.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") {
      args.quick = true;
    } else if ((a == "--settings" || a == "--out") && i + 1 < argc) {
      (a == "--settings" ? args.settings : args.out) = argv[++i];
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      usage();
      return 2;
    }
  }
  const Experiment* chosen = nullptr;
  for (const Experiment& e : kExperiments)
    if (args.command == e.name) chosen = &e;
  if (!chosen && args.command != "all") {
    usage();
    return 2;
  }
  try {
    prag::Stopwatch watch;
    auto ctx = bench::open_context(args);
    const int rc = chosen ? chosen->run(*ctx) : run_all(*ctx);
    bench::progress("%s finished in %.1f s", args.command.c_str(), watch.elapsed_s());
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "prag-bench: %s\n", e.what());
    return 1;
  }
}
