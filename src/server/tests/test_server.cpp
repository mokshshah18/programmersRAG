#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <httplib.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

#include "prag/server/server.hpp"

using namespace prag;
using nlohmann::json;
namespace fsys = std::filesystem;

namespace {

void write_text(const fsys::path& p, const std::string& text) {
  fsys::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << text;
}

std::string read_text(const fsys::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string module_source(const std::string& prefix, const std::string& words) {
  std::string s = "#include <stdlib.h>\n\n";
  s += "/* Initialises the " + words + " and allocates its storage. */\n";
  s += "int " + prefix + "_init(void* self, int capacity) {\n  if (capacity <= 0) return -1;\n  return " + prefix +
       "_reset(self);\n}\n\n";
  s += "/* Resets the " + words + " to an empty state without freeing. */\n";
  s += "int " + prefix + "_reset(void* self) {\n  (void)self;\n  return 0;\n}\n\n";
  s += "/* Releases every resource held by the " + words + ". */\n";
  s += "void " + prefix + "_destroy(void* self) {\n  " + prefix + "_reset(self);\n  free(self);\n}\n";
  return s;
}

// A workspace: corpus, frontend, secret file outside the frontend and a settings file.
fsys::path make_workspace(const std::string& name) {
  const auto dir = fsys::temp_directory_path() / ("prag_srv_" + name);
  std::error_code ec;
  fsys::remove_all(dir, ec);
  fsys::create_directories(dir);
  write_text(dir / "corpus/ring.c", module_source("ring", "circular ring queue of fixed capacity"));
  write_text(dir / "corpus/table.c", module_source("table", "open addressing hash table with string keys"));
  write_text(dir / "corpus/arena.c", module_source("arena", "arena region allocator for temporary memory"));
  write_text(dir / "corpus/checksum.c",
             "#include <stdint.h>\n#include <stddef.h>\n\n"
             "/* Computes the Fletcher checksum of a packet buffer using two running modular sums. */\n"
             "uint32_t fletcher_packet_checksum(const uint8_t* buffer, size_t length) {\n"
             "  uint32_t sum1 = 0, sum2 = 0;\n  for (size_t i = 0; i < length; ++i) {\n"
             "    sum1 = (sum1 + buffer[i]) % 65535;\n    sum2 = (sum2 + sum1) % 65535;\n  }\n"
             "  return (sum2 << 16) | sum1;\n}\n");
  write_text(dir / "frontend/index.html", "<!DOCTYPE html><title>t</title><p>hello</p>");
  write_text(dir / "frontend/app.js", "export const x = 1;\n");
  write_text(dir / "frontend/style.css", "body{}\n");
  write_text(dir / "secret.txt", "TOP SECRET");
  const json settings{{"corpus", {{"roots", {"corpus"}}}},
                      {"paths", {{"data", "data"}, {"frontend", "frontend"}}},
                      {"exec", {{"threads", 2}}}};
  write_text(dir / "settings.json", settings.dump(2));
  return dir;
}

struct Event {
  std::string name;
  json data;
};

std::vector<Event> parse_events(const std::string& body) {
  std::vector<Event> out;
  std::size_t pos = 0;
  while (pos < body.size()) {
    std::size_t end = body.find("\n\n", pos);
    if (end == std::string::npos) end = body.size();
    const std::string block = body.substr(pos, end - pos);
    pos = end + 2;
    Event e;
    std::size_t l = 0;
    while (l < block.size()) {
      std::size_t nl = block.find('\n', l);
      if (nl == std::string::npos) nl = block.size();
      const std::string line = block.substr(l, nl - l);
      l = nl + 1;
      if (line.rfind("event: ", 0) == 0) e.name = line.substr(7);
      if (line.rfind("data: ", 0) == 0) e.data = json::parse(line.substr(6), nullptr, false);
    }
    if (!e.name.empty()) out.push_back(std::move(e));
  }
  return out;
}

// Engine with a built index and a server on an unused port.
struct Fixture {
  fsys::path dir;
  std::unique_ptr<engine::Engine> engine;
  std::unique_ptr<server::Server> server;
  std::thread thread;
  int port = 0;

  explicit Fixture(const std::string& name, bool build = true) {
    dir = make_workspace(name);
    engine = std::make_unique<engine::Engine>((dir / "settings.json").string());
    engine->open();
    if (build) {
      const engine::IndexReport r = engine->build_index();
      if (!r.error.empty()) std::fprintf(stderr, "index build failed: %s\n", r.error.c_str());
    }
    server::ServerOptions so;
    so.port = 0;
    so.frontend_dir = (dir / "frontend").string();
    server = std::make_unique<server::Server>(*engine, so);
    std::string error;
    if (!server->bind(&error)) std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    port = server->port();
    thread = std::thread([this] { server->run(); });
    for (int i = 0; i < 100; ++i) {
      httplib::Client c("127.0.0.1", port);
      if (auto r = c.Get("/api/status"); r && r->status == 200) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  ~Fixture() {
    server->stop();
    if (thread.joinable()) thread.join();
  }
  httplib::Client client() const {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(60, 0);
    return c;
  }
};

json parse(const httplib::Result& r) { return r ? json::parse(r->body, nullptr, false) : json(); }

bool wait_index_idle(httplib::Client& c) {
  for (int i = 0; i < 600; ++i) {
    const json p = parse(c.Get("/api/index/progress"));
    if (p.is_object() && !p.value("running", true)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

}  // namespace

PRAG_TEST(status_settings_and_static_files) {
  Fixture f("basic");
  auto c = f.client();
  REQUIRE(f.port > 0);

  auto r = c.Get("/api/status");
  REQUIRE(r && r->status == 200);
  const json st = json::parse(r->body);
  CHECK_EQ(st["version"].get<std::string>(), std::string("2.0.0"));
  CHECK(st["index"]["ready"].get<bool>());
  CHECK(st.contains("models") && st.contains("cache") && st.contains("tune") && st.contains("roots"));
  CHECK(r->has_header("Cache-Control"));
  CHECK(!r->has_header("Access-Control-Allow-Origin"));

  r = c.Get("/api/settings");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body)["corpus"]["roots"].is_array());
  r = c.Post("/api/settings", R"({"retrieve":{"tau":0.3}})", "application/json");
  REQUIRE(r && r->status == 200);
  CHECK_NEAR(json::parse(r->body)["retrieve"]["tau"].get<double>(), 0.3, 1e-9);
  CHECK(read_text(f.dir / "settings.json").find("0.3") != std::string::npos);  // persisted
  r = c.Post("/api/settings", R"({"retrieve":{"tau":0.25}})", "application/json");
  CHECK(r && r->status == 200);

  // Frontend.
  r = c.Get("/");
  REQUIRE(r && r->status == 200);
  CHECK(r->get_header_value("Content-Type").rfind("text/html", 0) == 0);
  CHECK_EQ(r->get_header_value("Cache-Control"), std::string("no-store"));
  CHECK(r->body.find("hello") != std::string::npos);
  r = c.Get("/app.js");
  REQUIRE(r && r->status == 200);
  CHECK(r->get_header_value("Content-Type").find("javascript") != std::string::npos);
  r = c.Get("/style.css");
  REQUIRE(r && r->status == 200);
  CHECK(r->get_header_value("Content-Type").rfind("text/css", 0) == 0);
  r = c.Get("/missing.txt");
  CHECK(r && r->status == 404);

  // Path traversal never reaches the file outside the frontend directory.
  for (const char* p : {"/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/%2e%2e%2fsecret.txt",
                        "/..%5csecret.txt", "/sub/../../secret.txt", "//secret.txt", "/C:/Windows/win.ini"}) {
    r = c.Get(p);
    CHECK(r && r->status >= 400);
    if (r) CHECK(r->body.find("TOP SECRET") == std::string::npos);
  }
  httplib::Client raw("127.0.0.1", f.port);
  auto t = raw.Get("/../secret.txt");
  CHECK(!t || (t->status >= 400 && t->body.find("TOP SECRET") == std::string::npos));

  // Unknown API paths give JSON.
  r = c.Get("/api/nothing");
  REQUIRE(r && r->status == 404);
  CHECK(json::parse(r->body).contains("error"));
  r = c.Post("/api/nothing", "{}", "application/json");
  REQUIRE(r && r->status == 404);
  CHECK(json::parse(r->body).contains("error"));
  CHECK(r->get_header_value("Content-Type").find("application/json") != std::string::npos);
}

PRAG_TEST(history_feedback_groups_unit_tune) {
  Fixture f("misc");
  auto c = f.client();
  auto r = c.Get("/api/history");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body).is_array());

  r = c.Post("/api/query", R"({"question":"How is the Fletcher checksum of a packet computed?"})", "application/json");
  REQUIRE(r && r->status == 200);
  const auto events = parse_events(r->body);
  REQUIRE(!events.empty());
  const std::string qid = events.back().data.value("id", std::string());
  CHECK(!qid.empty());

  r = c.Get("/api/history");
  const json h = json::parse(r->body);
  REQUIRE(h.size() == 1);
  CHECK_EQ(h[0]["id"].get<std::string>(), qid);

  r = c.Post("/api/feedback", json{{"id", qid}, {"rating", 1}}.dump(), "application/json");
  CHECK(r && r->status == 200);
  CHECK_EQ(json::parse(c.Get("/api/history")->body)[0]["rating"].get<int>(), 1);
  r = c.Post("/api/feedback", json{{"id", "q-nope"}, {"rating", 1}}.dump(), "application/json");
  CHECK(r && r->status == 404);
  r = c.Post("/api/feedback", json{{"id", qid}, {"rating", 5}}.dump(), "application/json");
  CHECK(r && r->status == 400);
  r = c.Post("/api/feedback", R"({"rating":1})", "application/json");
  CHECK(r && r->status == 400);

  r = c.Get("/api/groups");
  REQUIRE(r && r->status == 200);
  const json g = json::parse(r->body);
  CHECK(g["groups"].is_array() && !g["groups"].empty());
  CHECK(g["units"].is_array() && !g["units"].empty());

  r = c.Get("/api/unit?id=0");
  REQUIRE(r && r->status == 200);
  const json u = json::parse(r->body);
  CHECK(u.contains("text") && u.contains("path") && u.contains("callees"));
  CHECK(c.Get("/api/unit?id=99999999") && c.Get("/api/unit?id=99999999")->status == 404);
  CHECK(c.Get("/api/unit?id=abc")->status == 400);
  CHECK(c.Get("/api/unit")->status == 400);

  r = c.Get("/api/tune");
  REQUIRE(r && r->status == 200);
  const json tu = json::parse(r->body);
  CHECK(tu.contains("available") && tu.contains("enabled") && tu.contains("stage") && tu.contains("runs"));
  CHECK(c.Post("/api/tune", R"({"enabled":"yes"})", "application/json")->status == 400);
  r = c.Post("/api/tune", R"({"enabled":true})", "application/json");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body).contains("stage"));
  r = c.Post("/api/tune", R"({"enabled":false})", "application/json");
  CHECK(r && r->status == 200);
  r = c.Post("/api/tune/run", "", "application/json");  // unavailable without llama.cpp
  CHECK(r && (r->status == 400 || r->status == 202 || r->status == 409));

  r = c.Delete("/api/history");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(c.Get("/api/history")->body).empty());
}

PRAG_TEST(query_event_stream_both_modes) {
  Fixture f("query");
  auto c = f.client();
  for (const bool edit : {false, true}) {
    const json body{{"question", "How is the Fletcher checksum of a packet buffer computed?"}, {"edit", edit}, {"tau", 0.25}};
    auto r = c.Post("/api/query", body.dump(), "application/json");
    REQUIRE(r && r->status == 200);
    CHECK(r->get_header_value("Content-Type").rfind("text/event-stream", 0) == 0);
    const auto ev = parse_events(r->body);
    REQUIRE(ev.size() >= 3);
    CHECK_EQ(ev.front().name, std::string("retrieval"));
    CHECK_EQ(ev.back().name, std::string("done"));
    bool retrieval_first = true;
    std::string streamed;
    for (std::size_t i = 1; i + 1 < ev.size(); ++i) {
      if (ev[i].name == "retrieval") retrieval_first = false;
      if (ev[i].name == "token") streamed += ev[i].data.value("text", std::string());
      CHECK(ev[i].name == "token" || ev[i].name == "edit");
    }
    CHECK(retrieval_first);
    const json& ret = ev.front().data;
    CHECK(ret.contains("id") && ret.contains("units") && ret.contains("context") && ret.contains("scan") && ret.contains("trace"));
    CHECK(ret["units"].is_array() && !ret["units"].empty());
    CHECK(ret["units"][0].contains("preview"));
    CHECK_EQ(ret["units"][0]["name"].get<std::string>(), std::string("fletcher_packet_checksum"));
    const json& done = ev.back().data;
    CHECK_EQ(done["id"].get<std::string>(), ret["id"].get<std::string>());
    CHECK(!done["answer"].get<std::string>().empty());
    CHECK_EQ(done["answer"].get<std::string>(), streamed);
    CHECK(done.contains("prompt_tokens") && done.contains("stop_reason") && done.contains("extractive"));
  }

  // Validation.
  auto r = c.Post("/api/query", "{not json", "application/json");
  REQUIRE(r && r->status == 400);
  CHECK(json::parse(r->body).contains("error"));
  CHECK(c.Post("/api/query", "{}", "application/json")->status == 400);
  CHECK(c.Post("/api/query", R"({"question":""})", "application/json")->status == 400);
  CHECK(c.Post("/api/query", R"({"question":"x","tau":7})", "application/json")->status == 400);
  CHECK(c.Post("/api/query", R"({"question":"x","edit":"yes"})", "application/json")->status == 400);
  CHECK(c.Post("/api/query", "[1,2]", "application/json")->status == 400);
  // Oversized body.
  const std::string big(1024 * 1024 + 4096, 'a');
  r = c.Post("/api/query", json{{"question", big}}.dump(), "application/json");
  CHECK(r && r->status == 413);
  if (r) CHECK(json::parse(r->body, nullptr, false).contains("error"));
  // Cross-origin state-changing requests are refused.
  httplib::Headers h{{"Origin", "http://evil.example"}};
  r = c.Post("/api/history", h, "{}", "application/json");
  CHECK(r && r->status == 403);
  r = c.Delete("/api/history", h);
  CHECK(r && r->status == 403);
}

PRAG_TEST(query_without_index_reports_error_event) {
  Fixture f("noindex", false);
  auto c = f.client();
  auto r = c.Post("/api/query", R"({"question":"anything"})", "application/json");
  REQUIRE(r && (r->status == 200 || r->status == 409));
  if (r->status == 200) {
    const auto ev = parse_events(r->body);
    REQUIRE(!ev.empty());
    CHECK_EQ(ev.back().name, std::string("error"));
    CHECK(ev.back().data.contains("error"));
  }
}

PRAG_TEST(index_calibrate_and_conflicts) {
  Fixture f("jobs", false);
  auto c = f.client();
  auto r = c.Post("/api/calibrate", "", "application/json");
  CHECK(r && r->status == 409);  // no index yet

  CHECK(c.Post("/api/index", "{bad", "application/json")->status == 400);
  r = c.Post("/api/index", "", "application/json");
  REQUIRE(r && r->status == 202);
  // While the build runs a second start is refused (the build of a tiny corpus can finish first,
  // so the loop tries until it observes the conflict or the job ends).
  bool saw_conflict = false;
  for (int i = 0; i < 50 && !saw_conflict; ++i) {
    auto r2 = c.Post("/api/index", "", "application/json");
    if (r2 && r2->status == 409) saw_conflict = true;
    if (r2 && r2->status == 202) break;  // previous job already finished and a new one started
  }
  (void)saw_conflict;  // timing dependent; the deterministic check is conflict_while_job_running
  REQUIRE(wait_index_idle(c));
  for (int i = 0; i < 200; ++i) {
    if (!json::parse(c.Get("/api/status")->body)["job"]["running"].get<bool>()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  const json p = json::parse(c.Get("/api/index/progress")->body);
  CHECK(p.contains("stage") && p.contains("done") && p.contains("total") && p.contains("report"));
  REQUIRE(p["report"].is_object());
  CHECK_EQ(p["report"]["files"].get<int>(), 4);
  CHECK(json::parse(c.Get("/api/status")->body)["index"]["ready"].get<bool>());

  r = c.Post("/api/calibrate", R"({"max_queries":40})", "application/json");
  REQUIRE(r && r->status == 202);
  for (int i = 0; i < 400; ++i) {
    if (!json::parse(c.Get("/api/status")->body)["job"]["running"].get<bool>()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  const json job = json::parse(c.Get("/api/status")->body)["job"];
  CHECK(!job["running"].get<bool>());
  CHECK_EQ(job["error"].get<std::string>(), std::string());
}

PRAG_TEST(conflict_while_job_running) {
  // A large enough corpus keeps the build busy while the second start is issued.
  const auto dir = make_workspace("conflict");
  for (int i = 0; i < 150; ++i)
    write_text(dir / "corpus" / ("gen" + std::to_string(i) + ".c"), module_source("gen" + std::to_string(i), "generated module number " + std::to_string(i)));
  engine::Engine e((dir / "settings.json").string());
  e.open();
  server::ServerOptions so;
  so.port = 0;
  so.frontend_dir = (dir / "frontend").string();
  server::Server s(e, so);
  REQUIRE(s.bind());
  std::thread t([&] { s.run(); });
  httplib::Client c("127.0.0.1", s.port());
  for (int i = 0; i < 100 && !c.Get("/api/status"); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  auto r = c.Post("/api/index", "", "application/json");
  REQUIRE(r && r->status == 202);
  bool conflict = false;
  for (int i = 0; i < 20 && !conflict; ++i) {
    auto r2 = c.Post("/api/index", "", "application/json");
    conflict = r2 && r2->status == 409;
  }
  CHECK(conflict);
  auto r3 = c.Post("/api/calibrate", "", "application/json");
  CHECK(r3 && (r3->status == 409));
  s.stop();
  t.join();
}

PRAG_TEST(edit_preview_apply_revert) {
  Fixture f("edit");
  auto c = f.client();
  const std::string file = (f.dir / "corpus/ring.c").generic_string();
  const std::string original = read_text(f.dir / "corpus/ring.c");

  const json blocks{{"blocks", json::array({json{{"path", file},
                                                 {"search", "  if (capacity <= 0) return -1;\n"},
                                                 {"replace", "  if (capacity < 1) return -2;\n"}}})}};
  auto r = c.Post("/api/edit/preview", blocks.dump(), "application/json");
  REQUIRE(r && r->status == 200);
  json tx = json::parse(r->body);
  CHECK_EQ(tx["status"].get<std::string>(), std::string("ok"));
  CHECK(!tx["committed"].get<bool>());
  REQUIRE(tx["files"].size() == 1);
  CHECK(tx["files"][0]["diff"].get<std::string>().find("capacity < 1") != std::string::npos);
  CHECK_EQ(read_text(f.dir / "corpus/ring.c"), original);  // preview writes nothing
  const std::string id = tx["id"].get<std::string>();

  // The text form: model output with an edit block.
  const std::string text = "Change it:\n\n" + file + "\n<<<<<<< SEARCH\n  if (capacity <= 0) return -1;\n=======\n  if (capacity < 1) return -2;\n>>>>>>> REPLACE\n";
  r = c.Post("/api/edit/preview", json{{"text", text}}.dump(), "application/json");
  REQUIRE(r && r->status == 200);
  CHECK_EQ(json::parse(r->body)["id"].get<std::string>(), id);

  CHECK(c.Post("/api/edit/preview", "{}", "application/json")->status == 400);
  CHECK(c.Post("/api/edit/apply", "{}", "application/json")->status == 400);
  CHECK(c.Post("/api/edit/apply", R"({"id":"tx-none"})", "application/json")->status == 404);
  CHECK(c.Post("/api/edit/revert", R"({"id":"tx-none"})", "application/json")->status == 404);

  r = c.Post("/api/edit/apply", json{{"id", id}}.dump(), "application/json");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body)["committed"].get<bool>());
  CHECK(read_text(f.dir / "corpus/ring.c").find("capacity < 1") != std::string::npos);

  r = c.Post("/api/edit/revert", json{{"id", id}}.dump(), "application/json");
  REQUIRE(r && r->status == 200);
  CHECK(!json::parse(r->body)["committed"].get<bool>());
  CHECK_EQ(read_text(f.dir / "corpus/ring.c"), original);

  // A search text that does not exist is reported, not applied.
  const json bad{{"blocks", json::array({json{{"path", file}, {"search", "no such text\n"}, {"replace", "x\n"}}})}};
  r = c.Post("/api/edit/preview", bad.dump(), "application/json");
  REQUIRE(r && r->status == 200);
  CHECK(json::parse(r->body)["status"].get<std::string>() != "ok");
}

#ifdef PRAG_CLI_EXE
namespace {

// Runs prag.exe with `args`, captures stdout and stderr together, returns the exit code.
int run_cli(const std::string& args, const fsys::path& out_file) {
#ifdef _WIN32
  const std::string cmd = "\"\"" + std::string(PRAG_CLI_EXE) + "\" " + args + " > \"" + out_file.string() + "\" 2>&1\"";
#else
  const std::string cmd = "\"" + std::string(PRAG_CLI_EXE) + "\" " + args + " > \"" + out_file.string() + "\" 2>&1";
#endif
  return std::system(cmd.c_str());
}

}  // namespace

PRAG_TEST(cli_commands) {
  if (!fsys::exists(PRAG_CLI_EXE)) {
    std::printf("  skipped: %s not built\n", PRAG_CLI_EXE);
    return;
  }
  const auto dir = make_workspace("cli");
  const std::string s = "--settings \"" + (dir / "settings.json").string() + "\"";
  const auto out = dir / "out.txt";

  CHECK_EQ(run_cli("doctor " + s, out), 0);
  CHECK(read_text(out).find("ok") != std::string::npos);
  CHECK_EQ(run_cli("find-files " + s, out), 0);
  {
    const std::string o = read_text(out);
    CHECK(o.find("ring.c") != std::string::npos && o.find("checksum.c") != std::string::npos);
  }
  CHECK_EQ(run_cli("models " + s, out), 0);
  CHECK(!read_text(out).empty());

  // Not indexed yet: ask fails with exit code 1.
  CHECK_EQ(run_cli("ask \"What does ring_init do?\" " + s, out), 1);

  CHECK_EQ(run_cli("index " + s, out), 0);
  CHECK(read_text(out).find("indexed") != std::string::npos);
  CHECK_EQ(run_cli("ask \"How is the Fletcher checksum of a packet buffer computed?\" " + s, out), 0);
  CHECK(read_text(out).find("fletcher_packet_checksum") != std::string::npos);
  CHECK_EQ(run_cli("ask \"How is the Fletcher checksum of a packet buffer computed?\" --json " + s, out), 0);
  {
    const std::string o = read_text(out);
    const auto brace = o.find('{');
    const json j = brace == std::string::npos ? json() : json::parse(o.substr(brace), nullptr, false);
    CHECK(j.is_object() && j.contains("retrieval") && j.contains("done"));
  }
  CHECK_EQ(run_cli("groups " + s, out), 0);
  CHECK(!read_text(out).empty());
  CHECK_EQ(run_cli("tune status " + s, out), 0);
  CHECK(read_text(out).find("stage") != std::string::npos);

  // edit: preview writes nothing, --yes writes, --revert restores.
  const std::string file = (dir / "corpus/ring.c").generic_string();
  const std::string original = read_text(dir / "corpus/ring.c");
  write_text(dir / "edit.txt", file + "\n<<<<<<< SEARCH\n  if (capacity <= 0) return -1;\n=======\n  if (capacity < 1) return -2;\n>>>>>>> REPLACE\n");
  CHECK_EQ(run_cli("edit \"" + (dir / "edit.txt").string() + "\" " + s, out), 0);
  CHECK_EQ(read_text(dir / "corpus/ring.c"), original);
  CHECK_EQ(run_cli("edit \"" + (dir / "edit.txt").string() + "\" --yes --revert " + s, out), 0);
  CHECK_EQ(read_text(dir / "corpus/ring.c"), original);
  CHECK_EQ(run_cli("edit \"" + (dir / "edit.txt").string() + "\" --yes " + s, out), 0);
  CHECK(read_text(dir / "corpus/ring.c").find("capacity < 1") != std::string::npos);

  // Usage errors.
  CHECK_EQ(run_cli("frobnicate", out), 2);
  CHECK_EQ(run_cli("", out), 2);
  CHECK_EQ(run_cli("ask", out), 2);
  CHECK_EQ(run_cli("index --bogus", out), 2);
  CHECK_EQ(run_cli("tune sideways " + s, out), 2);
  CHECK_EQ(run_cli("edit " + s, out), 2);
}
#endif
