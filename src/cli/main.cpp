// prag: command line front end of the engine (DESIGN.md section 6.14).
//
// Exit codes: 0 success, 1 failure, 2 usage error.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"
#include "prag/engine/engine.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/index/vector_store.hpp"
#include "prag/llm/model.hpp"
#include "prag/server/server.hpp"

using nlohmann::json;
using namespace prag;

namespace {

constexpr int kOk = 0;
constexpr int kFail = 1;
constexpr int kUsage = 2;

const char* kUsageText =
    "usage: prag [--settings <file>] <command> [options]\n"
    "\n"
    "commands:\n"
    "  serve [--host H] [--port N] [--frontend DIR]   run the web interface and the API\n"
    "  index [--quiet] [--json]                       build or rebuild the index\n"
    "  ask <question...> [--edit] [--json] [--tau T]  answer a question (streams to stdout)\n"
    "  edit <file|-> [--yes] [--revert] [--json]      preview (default) or apply a file of edit blocks;\n"
    "                                                 --yes is required to write; --revert applies and\n"
    "                                                 then restores the files (a round trip check)\n"
    "  find-files                                     print the files the corpus settings select\n"
    "  groups [--json]                                density modes of the index\n"
    "  calibrate [--quiet]                            re-estimate the calibration\n"
    "  tune status|on|off|run                         background fine-tuning\n"
    "  models                                         list configured model files\n"
    "  doctor                                         check the installation\n"
    "\n"
    "--settings defaults to prag.json in the current directory.\n"
    "exit codes: 0 success, 1 failure, 2 usage error\n";

std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop.store(true); }

struct Args {
  std::string settings = "prag.json";
  std::string command;
  std::vector<std::string> positional;
  std::map<std::string, std::string> values;  // --name value
  std::set<std::string> flags;                // --name
  std::string error;

  bool flag(const std::string& name) const { return flags.count(name) != 0; }
  std::string value(const std::string& name, const std::string& fallback = {}) const {
    const auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
  }
  bool has(const std::string& name) const { return values.count(name) != 0; }
};

Args parse_args(int argc, char** argv) {
  static const std::set<std::string> boolean = {"--edit", "--json", "--yes", "--quiet", "--revert", "--help"};
  static const std::set<std::string> valued = {"--settings", "--host", "--port", "--frontend", "--tau"};
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "-h") {
      a.flags.insert("--help");
    } else if (s.rfind("--", 0) == 0 && s.size() > 2) {
      std::string name = s, inline_value;
      bool has_inline = false;
      const auto eq = s.find('=');
      if (eq != std::string::npos) {
        name = s.substr(0, eq);
        inline_value = s.substr(eq + 1);
        has_inline = true;
      }
      if (boolean.count(name)) {
        a.flags.insert(name);
      } else if (valued.count(name)) {
        if (has_inline) {
          a.values[name] = inline_value;
        } else if (i + 1 < argc) {
          a.values[name] = argv[++i];
        } else {
          a.error = "option " + name + " needs a value";
          return a;
        }
      } else {
        a.error = "unknown option " + name;
        return a;
      }
    } else if (a.command.empty()) {
      a.command = s;
    } else {
      a.positional.push_back(s);
    }
  }
  if (a.has("--settings")) a.settings = a.value("--settings");
  return a;
}

int usage_error(const std::string& message) {
  std::fprintf(stderr, "prag: %s\n\n%s", message.c_str(), kUsageText);
  return kUsage;
}

std::string settings_dir(const std::string& settings_path) {
  const std::string full = fs::normalize(settings_path);
  const auto slash = full.find_last_of('/');
  return slash == std::string::npos ? std::string(".") : full.substr(0, slash);
}

std::string resolve(const std::string& base_dir, const std::string& p) {
  if (p.empty()) return {};
  std::error_code ec;
  if (std::filesystem::path(p).is_absolute()) return fs::normalize(p);
  return fs::normalize(base_dir + "/" + p);
}

std::vector<std::string> corpus_roots(const Settings& settings, const std::string& base_dir) {
  std::vector<std::string> roots;
  for (const std::string& r : settings.get<std::vector<std::string>>("corpus.roots", {}))
    if (!r.empty()) roots.push_back(resolve(base_dir, r));
  return roots;
}

// Progress lines on stderr: one per stage change.
engine::IndexProgressCallback progress_printer(bool quiet) {
  auto last = std::make_shared<std::string>();
  return [quiet, last](const engine::IndexProgress& p) {
    if (!quiet && p.stage != *last) {
      *last = p.stage;
      std::fprintf(stderr, "[%s] %s\n", p.stage.c_str(), p.message.c_str());
    }
    return !g_stop.load();
  };
}

void print_transaction(const json& tx) {
  std::printf("transaction %s: %s\n", tx.value("id", std::string()).c_str(), tx.value("status", std::string()).c_str());
  for (const json& f : tx.value("files", json::array())) {
    std::printf("\n%s  [%s]%s\n", f.value("display_path", std::string()).c_str(), f.value("status", std::string()).c_str(),
                f.value("creates_file", false) ? " (new file)" : "");
    const std::string msg = f.value("message", std::string());
    if (!msg.empty()) std::printf("  %s\n", msg.c_str());
    const std::string diff = f.value("diff", std::string());
    if (!diff.empty()) std::printf("%s%s", diff.c_str(), diff.back() == '\n' ? "" : "\n");
  }
  for (const json& w : tx.value("warnings", json::array())) std::printf("warning: %s\n", w.get<std::string>().c_str());
}

// ---- commands --------------------------------------------------------------

int cmd_serve(const Args& a) {
  engine::Engine e(a.settings);
  e.open();
  const std::string base = settings_dir(a.settings);
  server::ServerOptions so;
  so.host = a.value("--host", e.settings().get<std::string>("server.host", "127.0.0.1"));
  so.port = e.settings().get<int>("server.port", 8000);
  if (a.has("--port")) {
    try {
      so.port = std::stoi(a.value("--port"));
    } catch (...) {
      return usage_error("--port must be an integer");
    }
    if (so.port < 0 || so.port > 65535) return usage_error("--port must lie in 0..65535");
  }
  so.frontend_dir = a.has("--frontend") ? a.value("--frontend")
                                        : resolve(base, e.settings().get<std::string>("paths.frontend", "frontend"));
  server::Server srv(e, so);
  std::string error;
  if (!srv.bind(&error)) {
    std::fprintf(stderr, "prag: %s\n", error.c_str());
    return kFail;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::printf("listening on http://%s:%d/  (Ctrl+C to stop)\n", so.host.c_str(), srv.port());
  if (!e.index_ready()) std::printf("no index yet: run \"prag index\" or use the web interface\n");
  std::fflush(stdout);
  std::atomic<bool> finished{false};
  std::thread watcher([&] {
    while (!finished.load() && !g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    srv.stop();
  });
  srv.run();
  finished = true;
  watcher.join();
  std::printf("stopped\n");
  return kOk;
}

int cmd_index(const Args& a) {
  engine::Engine e(a.settings);
  e.open();
  std::signal(SIGINT, on_signal);
  const engine::IndexReport r = e.build_index(progress_printer(a.flag("--quiet")));
  if (!r.error.empty()) {
    std::fprintf(stderr, "prag: index build failed: %s\n", r.error.c_str());
    return kFail;
  }
  if (a.flag("--json")) {
    std::printf("%s\n", engine::report_json(r).dump(2).c_str());
  } else {
    std::printf("indexed %zu files, %zu units (%zu functions, %zu call edges), %zu groups; embedded %zu, reused %zu; %.2f s\n",
                r.files, r.units, r.functions, r.call_edges, r.groups, r.embedded, r.reused,
                r.seconds.count("total") ? r.seconds.at("total") : 0.0);
  }
  return kOk;
}

int cmd_ask(const Args& a) {
  std::string question;
  for (const std::string& w : a.positional) question += (question.empty() ? "" : " ") + w;
  if (question.empty()) return usage_error("ask needs a question");
  engine::QueryOptions qo;
  qo.edit = a.flag("--edit");
  if (a.has("--tau")) {
    try {
      qo.tau = std::stod(a.value("--tau"));
    } catch (...) {
      return usage_error("--tau must be a number");
    }
    if (!(qo.tau > 0.0 && qo.tau < 1.0)) return usage_error("--tau must lie in (0, 1)");
  }
  engine::Engine e(a.settings);
  e.open();
  if (!e.corpus()) {
    std::fprintf(stderr, "prag: no index found; run \"prag index\" first\n");
    return kFail;
  }
  std::signal(SIGINT, on_signal);
  const bool as_json = a.flag("--json");
  json retrieval;
  bool streamed = false;
  const engine::QueryResult r = e.query(
      question, qo,
      [&](const engine::QueryResult& partial) {
        if (as_json) {
          if (const auto c = e.corpus()) retrieval = engine::retrieval_json(partial, *c);
        }
      },
      [&](std::string_view piece) {
        if (!as_json) {
          std::fwrite(piece.data(), 1, piece.size(), stdout);
          std::fflush(stdout);
        }
        streamed = true;
        return !g_stop.load();
      });
  if (!r.error.empty()) {
    if (streamed && !as_json) std::printf("\n");
    std::fprintf(stderr, "prag: %s\n", r.error.c_str());
    return kFail;
  }
  std::vector<std::string> roots;
  if (const auto c = e.corpus()) roots = c->roots;
  if (as_json) {
    json out{{"retrieval", retrieval}, {"done", engine::done_json(r)}};
    if (r.transaction) out["edit"] = engine::transaction_json(*r.transaction, false, roots, r.edit_warnings);
    std::printf("%s\n", out.dump(2, ' ', false, json::error_handler_t::replace).c_str());
    return kOk;
  }
  if (!streamed) std::fputs(r.answer.c_str(), stdout);
  std::printf("\n");
  if (r.transaction) {
    std::printf("\n--- proposed edit (not applied; save the answer to a file and use \"prag edit <file> --yes\") ---\n");
    print_transaction(engine::transaction_json(*r.transaction, false, roots, r.edit_warnings));
  }
  return kOk;
}

int cmd_edit(const Args& a) {
  if (a.positional.size() != 1) return usage_error("edit needs exactly one file of edit blocks (or - for stdin)");
  if (a.flag("--revert") && !a.flag("--yes")) return usage_error("--revert writes files temporarily and needs --yes");
  std::string text;
  if (a.positional[0] == "-") {
    text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
  } else {
    const auto content = fs::read_file(a.positional[0]);
    if (!content) {
      std::fprintf(stderr, "prag: cannot read %s\n", a.positional[0].c_str());
      return kFail;
    }
    text = *content;
  }
  engine::Engine e(a.settings);
  e.open();
  std::vector<std::string> roots;
  if (const auto c = e.corpus()) roots = c->roots;
  else roots = corpus_roots(e.settings(), settings_dir(a.settings));
  std::vector<std::string> warnings;
  edit::Transaction tx = e.preview_edit(text, &warnings);
  const bool as_json = a.flag("--json");
  auto show = [&](const edit::Transaction& t) {
    const json j = engine::transaction_json(t, e.transaction_committed(t.id), roots, warnings);
    if (as_json) std::printf("%s\n", j.dump(2, ' ', false, json::error_handler_t::replace).c_str());
    else print_transaction(j);
  };
  if (tx.files.empty()) {
    std::fprintf(stderr, "prag: no edit blocks found in the input\n");
    return kFail;
  }
  if (tx.status != edit::Status::Ok) {
    show(tx);
    std::fprintf(stderr, "prag: the edit cannot be applied (%s)\n", edit::to_string(tx.status));
    return kFail;
  }
  if (!a.flag("--yes")) {
    show(tx);
    std::fprintf(stderr, "preview only; nothing was written (use --yes to apply)\n");
    return kOk;
  }
  edit::Transaction applied;
  const edit::Status st = e.apply_edit(tx.id, &applied);
  show(applied);
  if (st != edit::Status::Ok) {
    std::fprintf(stderr, "prag: apply failed (%s)\n", edit::to_string(st));
    return kFail;
  }
  std::fprintf(stderr, "applied\n");
  if (a.flag("--revert")) {
    edit::Transaction reverted;
    const edit::Status rs = e.revert_edit(tx.id, &reverted);
    if (rs != edit::Status::Ok) {
      std::fprintf(stderr, "prag: revert failed (%s)\n", edit::to_string(rs));
      return kFail;
    }
    std::fprintf(stderr, "reverted\n");
  }
  return kOk;
}

int cmd_find_files(const Args& a) {
  Settings s;
  if (!s.load(a.settings)) {
    std::fprintf(stderr, "prag: %s is not valid JSON\n", a.settings.c_str());
    return kFail;
  }
  const auto roots = corpus_roots(s, settings_dir(a.settings));
  if (roots.empty()) {
    std::fprintf(stderr, "prag: no corpus roots configured (setting corpus.roots)\n");
    return kFail;
  }
  fs::WalkOptions wo;
  wo.extensions = s.get<std::vector<std::string>>("corpus.extensions", {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".py"});
  wo.exclude_substrings =
      s.get<std::vector<std::string>>("corpus.exclude", {"/.git/", "/build/", "/out/", "/.deps/", "/node_modules/"});
  wo.max_file_bytes = s.get<std::uint64_t>("corpus.max_file_bytes", 1048576);
  std::set<std::string> files;
  for (const std::string& r : roots) {
    if (!fs::exists(r)) std::fprintf(stderr, "warning: corpus root does not exist: %s\n", r.c_str());
    for (const std::string& f : fs::walk(r, wo)) files.insert(f);
  }
  for (const std::string& f : files) std::printf("%s\n", f.c_str());
  std::fprintf(stderr, "%zu file(s)\n", files.size());
  return files.empty() ? kFail : kOk;
}

int cmd_groups(const Args& a) {
  engine::Engine e(a.settings);
  e.open();
  if (!e.corpus()) {
    std::fprintf(stderr, "prag: no index found; run \"prag index\" first\n");
    return kFail;
  }
  const json g = e.groups_json();
  if (a.flag("--json")) {
    std::printf("%s\n", g.dump(2, ' ', false, json::error_handler_t::replace).c_str());
    return kOk;
  }
  std::printf("kappa %.3f, reduced dimension %d, %zu group(s)\n", g.value("kappa", 0.0), g.value("reduced_dim", 0),
              g["groups"].size());
  std::printf("%5s %6s %12s  %s\n", "group", "size", "log_density", "label");
  for (const json& k : g["groups"])
    std::printf("%5d %6d %12.3f  %s\n", k.value("id", 0), k.value("size", 0), k.value("log_density", 0.0),
                k.value("label", std::string()).c_str());
  return kOk;
}

int cmd_calibrate(const Args& a) {
  engine::Engine e(a.settings);
  e.open();
  if (!e.corpus()) {
    std::fprintf(stderr, "prag: no index found; run \"prag index\" first\n");
    return kFail;
  }
  std::signal(SIGINT, on_signal);
  std::string error;
  if (!e.calibrate({}, progress_printer(a.flag("--quiet")), &error)) {
    std::fprintf(stderr, "prag: calibration failed: %s\n", error.c_str());
    return kFail;
  }
  const auto c = e.corpus();
  std::printf("calibration %s: %zu fit queries, %zu held-out\n", c && c->calibration_info.fitted ? "fitted" : "not fitted",
              c ? c->calibration_info.queries : std::size_t{0}, c ? c->calibration_info.heldout_queries : std::size_t{0});
  return kOk;
}

int cmd_tune(const Args& a) {
  if (a.positional.size() != 1) return usage_error("tune needs one of: status, on, off, run");
  const std::string& sub = a.positional[0];
  if (sub != "status" && sub != "on" && sub != "off" && sub != "run") return usage_error("tune needs one of: status, on, off, run");
  engine::Engine e(a.settings);
  e.open();
  if (sub == "on" || sub == "off") {
    const bool on = sub == "on";
    e.update_settings(json{{"tune", {{"enabled", on}}}});
    e.tuning().set_enabled(on);
  } else if (sub == "run") {
    std::string error;
    if (!e.tuning().run_now(&error)) {
      std::fprintf(stderr, "prag: tuning run failed: %s\n", error.empty() ? "unavailable or already running" : error.c_str());
      return kFail;
    }
  }
  std::printf("%s\n", tune::to_json(e.tuning().status()).dump(2, ' ', false, json::error_handler_t::replace).c_str());
  return kOk;
}

int cmd_models(const Args& a) {
  Settings s;
  if (!s.load(a.settings)) {
    std::fprintf(stderr, "prag: %s is not valid JSON\n", a.settings.c_str());
    return kFail;
  }
  const std::string base = settings_dir(a.settings);
  const char* roles[][2] = {{"embedding", "models.embedding"}, {"chat", "models.chat"}, {"tuning_base", "models.tuning_base"}};
  int status = kOk;
  std::printf("%-12s %-10s %-7s %s\n", "role", "size", "exists", "path");
  for (const auto& r : roles) {
    const std::string configured = s.get<std::string>(r[1], "");
    if (configured.empty()) {
      std::printf("%-12s %-10s %-7s %s\n", r[0], "-", "-", "(not configured)");
      continue;
    }
    const std::string path = resolve(base, configured);
    const bool exists = fs::exists(path) && !fs::is_directory(path);
    char size[32] = "-";
    if (exists) std::snprintf(size, sizeof size, "%.1f MiB", static_cast<double>(fs::file_size(path)) / (1024.0 * 1024.0));
    std::printf("%-12s %-10s %-7s %s\n", r[0], size, exists ? "yes" : "no", path.c_str());
    if (!exists) status = kFail;
  }
  return status;
}

int cmd_doctor(const Args& a) {
  int problems = 0;
  const exec::CacheTopology& t = exec::cache_topology();
  std::printf("version        2.0.0\n");
  std::printf("cache          %s\n", exec::describe(t).c_str());
  std::printf("block size     %zu bytes\n", exec::recommended_block_bytes(t));
  std::printf("simd kernel    %s\n", index::simd_kernel());
  std::printf("llama.cpp      %s (%s)\n", llm::available() ? "available" : "not available", llm::backend_info().c_str());

  Settings s;
  const bool exists = fs::exists(a.settings);
  if (!s.load(a.settings)) {
    std::printf("settings       %s: NOT VALID JSON\n", a.settings.c_str());
    return kFail;
  }
  std::printf("settings       %s%s\n", fs::normalize(a.settings).c_str(), exists ? "" : " (missing; defaults in use)");
  const std::string base = settings_dir(a.settings);
  const auto roots = corpus_roots(s, base);
  if (roots.empty()) std::printf("corpus         no roots configured\n");
  for (const std::string& r : roots) {
    const bool ok = fs::exists(r);
    std::printf("corpus root    %s%s\n", r.c_str(), ok ? "" : " (MISSING)");
    if (!ok) ++problems;
  }
  const std::string frontend = resolve(base, s.get<std::string>("paths.frontend", "frontend"));
  std::printf("frontend       %s%s\n", frontend.c_str(), fs::is_directory(frontend) ? "" : " (not found)");
  for (const char* key : {"models.embedding", "models.chat", "models.tuning_base"}) {
    const std::string m = s.get<std::string>(key, "");
    if (m.empty()) continue;
    const std::string path = resolve(base, m);
    const bool ok = fs::exists(path);
    std::printf("%-14s %s%s\n", key, path.c_str(), ok ? "" : " (MISSING)");
    if (!ok) ++problems;
  }
  {
    engine::Engine e(a.settings);
    e.open();
    const auto c = e.corpus();
    if (c) std::printf("index          %zu files, %zu units, built %s%s\n", c->files.size(), c->units.size(), c->built_at.c_str(),
                       e.index_ready() ? "" : " (stale)");
    else std::printf("index          none\n");
  }
  std::printf("%s\n", problems ? "problems found" : "ok");
  return problems ? kFail : kOk;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);
  if (!a.error.empty()) return usage_error(a.error);
  if (a.flag("--help") || a.command == "help") {
    std::fputs(kUsageText, stdout);
    return kOk;
  }
  if (a.command.empty()) return usage_error("no command given");
  try {
    const std::string& c = a.command;
    if (c == "serve") return cmd_serve(a);
    if (c == "index") return cmd_index(a);
    if (c == "ask") return cmd_ask(a);
    if (c == "edit") return cmd_edit(a);
    if (c == "find-files") return cmd_find_files(a);
    if (c == "groups") return cmd_groups(a);
    if (c == "calibrate") return cmd_calibrate(a);
    if (c == "tune") return cmd_tune(a);
    if (c == "models") return cmd_models(a);
    if (c == "doctor") return cmd_doctor(a);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "prag: %s\n", ex.what());
    return kFail;
  }
  return usage_error("unknown command \"" + a.command + "\"");
}
