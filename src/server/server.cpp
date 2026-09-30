#include "prag/server/server.hpp"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"

namespace prag::server {

namespace {

using nlohmann::json;
namespace stdfs = std::filesystem;

constexpr std::size_t kMaxBody = 1024 * 1024;

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

void send_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(dump(body), "application/json; charset=utf-8");
}

void send_error(httplib::Response& res, int status, const std::string& message) {
  send_json(res, status, json{{"error", message}});
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

const char* content_type_for(const std::string& ext) {
  static const struct {
    const char* ext;
    const char* type;
  } table[] = {{".html", "text/html; charset=utf-8"},
               {".htm", "text/html; charset=utf-8"},
               {".css", "text/css; charset=utf-8"},
               {".js", "text/javascript; charset=utf-8"},
               {".mjs", "text/javascript; charset=utf-8"},
               {".json", "application/json; charset=utf-8"},
               {".map", "application/json; charset=utf-8"},
               {".webmanifest", "application/manifest+json; charset=utf-8"},
               {".txt", "text/plain; charset=utf-8"},
               {".md", "text/plain; charset=utf-8"},
               {".svg", "image/svg+xml"},
               {".png", "image/png"},
               {".jpg", "image/jpeg"},
               {".jpeg", "image/jpeg"},
               {".gif", "image/gif"},
               {".webp", "image/webp"},
               {".ico", "image/x-icon"},
               {".woff", "font/woff"},
               {".woff2", "font/woff2"},
               {".ttf", "font/ttf"},
               {".wasm", "application/wasm"}};
  for (const auto& e : table)
    if (ext == e.ext) return e.type;
  return "application/octet-stream";
}

// Parses an optional JSON body. `allow_empty` accepts a missing body as {}.
bool parse_body(const httplib::Request& req, bool allow_empty, json* out, httplib::Response& res) {
  if (req.body.empty()) {
    if (allow_empty) {
      *out = json::object();
      return true;
    }
    send_error(res, 400, "request body is empty; expected a JSON object");
    return false;
  }
  json j = json::parse(req.body, nullptr, false);
  if (j.is_discarded()) {
    send_error(res, 400, "request body is not valid JSON");
    return false;
  }
  if (!j.is_object()) {
    send_error(res, 400, "request body must be a JSON object");
    return false;
  }
  *out = std::move(j);
  return true;
}

stdfs::path canonical_or_absolute(const stdfs::path& p) {
  std::error_code ec;
  stdfs::path c = stdfs::weakly_canonical(p, ec);
  if (ec) {
    c = stdfs::absolute(p, ec);
    if (ec) return p;
  }
  return c;
}

}  // namespace

struct Server::Impl {
  engine::Engine& engine;
  ServerOptions options;
  httplib::Server svr;
  int bound_port = 0;
  bool bound = false;
  std::atomic<bool> stopping{false};
  std::string frontend_root;  // normalised absolute path, empty when unusable

  // One long operation at a time (index build, calibration, tuning run).
  std::mutex job_mu;
  std::thread job_thread;
  std::atomic<bool> job_running{false};
  std::atomic<bool> job_cancel{false};
  std::string job_name;
  std::string job_error;

  Impl(engine::Engine& e, ServerOptions o) : engine(e), options(std::move(o)) {
    frontend_root = fs::normalize(canonical_or_absolute(stdfs::path(options.frontend_dir)).generic_string());
    configure();
    routes();
  }

  ~Impl() {
    stopping = true;
    job_cancel = true;
    svr.stop();
    join_job();
  }

  void join_job() {
    std::lock_guard<std::mutex> lock(job_mu);
    if (job_thread.joinable()) job_thread.join();
  }

  // Starts `fn` on the job thread. False when another job is running.
  bool start_job(const std::string& name, std::function<std::string()> fn) {
    std::lock_guard<std::mutex> lock(job_mu);
    if (job_running.load() || engine.index_building()) return false;
    if (job_thread.joinable()) job_thread.join();
    job_running = true;
    job_cancel = false;
    job_name = name;
    job_error.clear();
    job_thread = std::thread([this, fn = std::move(fn)] {
      std::string error;
      try {
        error = fn();
      } catch (const std::exception& ex) {
        error = ex.what();
      } catch (...) {
        error = "unknown error";
      }
      std::lock_guard<std::mutex> inner(job_mu_error);
      job_error = error;
      job_running = false;
    });
    return true;
  }
  std::mutex job_mu_error;

  json job_json() {
    std::lock_guard<std::mutex> lock(job_mu_error);
    return json{{"running", job_running.load()}, {"name", job_name}, {"error", job_error}};
  }

  std::vector<std::string> roots() const {
    if (const auto c = engine.corpus()) return c->roots;
    std::vector<std::string> out;
    const json st = engine.status_json();
    if (st.contains("roots") && st["roots"].is_array())
      for (const json& r : st["roots"])
        if (r.is_string()) out.push_back(r.get<std::string>());
    return out;
  }

  json tx_json(const edit::Transaction& tx, bool committed, const std::vector<std::string>& warnings = {}) const {
    return engine::transaction_json(tx, committed, roots(), warnings);
  }

  void configure() {
    svr.set_payload_max_length(kMaxBody);
    svr.set_read_timeout(30, 0);
    svr.set_write_timeout(30, 0);
    svr.set_keep_alive_max_count(100);
    svr.new_task_queue = [n = std::max(2, options.worker_threads)] { return new httplib::ThreadPool(static_cast<size_t>(n)); };

    // Cross-site request protection: a browser sends Origin on state-changing
    // requests; refuse those whose origin is not this server's own address.
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
      if (req.method != "GET" && req.method != "HEAD" && req.has_header("Origin")) {
        std::string origin = req.get_header_value("Origin");
        const auto scheme = origin.find("://");
        if (scheme != std::string::npos) origin = origin.substr(scheme + 3);
        if (lower(origin) != lower(req.get_header_value("Host"))) {
          send_error(res, 403, "cross-origin request refused");
          return httplib::Server::HandlerResponse::Handled;
        }
      }
      return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
      if (!res.has_header("Cache-Control")) res.set_header("Cache-Control", "no-store");
      res.set_header("X-Content-Type-Options", "nosniff");
    });
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
      if (res.body.empty() || !res.has_header("Content-Type")) {
        const std::string message = res.status == 413   ? "request body too large (limit 1 MiB)"
                                    : res.status == 404 ? "not found"
                                    : res.status == 405 ? "method not allowed"
                                                        : "request failed";
        (void)req;
        send_error(res, res.status, message);
      }
    });
    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
      std::string message = "internal error";
      try {
        if (ep) std::rethrow_exception(ep);
      } catch (const std::exception& ex) {
        message = ex.what();
      } catch (...) {
      }
      send_error(res, 500, message);
    });
  }

  void routes() {
    using Req = httplib::Request;
    using Res = httplib::Response;

    svr.Get("/api/status", [this](const Req&, Res& res) {
      json st = engine.status_json();
      st["job"] = job_json();
      send_json(res, 200, st);
    });

    svr.Get("/api/settings", [this](const Req&, Res& res) { send_json(res, 200, engine.settings().snapshot()); });
    svr.Post("/api/settings", [this](const Req& req, Res& res) {
      json patch;
      if (!parse_body(req, false, &patch, res)) return;
      engine.update_settings(patch);
      send_json(res, 200, engine.settings().snapshot());
    });

    svr.Post("/api/index", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, true, &body, res)) return;
      const bool started = start_job("index", [this]() -> std::string {
        const engine::IndexReport r =
            engine.build_index([this](const engine::IndexProgress&) { return !job_cancel.load(); });
        return r.error;
      });
      if (!started) return send_error(res, 409, "another long operation is already running");
      send_json(res, 202, json{{"started", true}});
    });
    svr.Get("/api/index/progress", [this](const Req&, Res& res) { send_json(res, 200, engine.index_progress_json()); });

    svr.Post("/api/query", [this](const Req& req, Res& res) { handle_query(req, res); });

    svr.Get("/api/history", [this](const Req&, Res& res) { send_json(res, 200, engine.history_json()); });
    svr.Delete("/api/history", [this](const Req&, Res& res) {
      if (!engine.clear_history()) return send_error(res, 500, "could not clear the history");
      send_json(res, 200, json{{"ok", true}});
    });

    svr.Post("/api/feedback", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, false, &body, res)) return;
      if (!body.contains("id") || !body["id"].is_string()) return send_error(res, 400, "field \"id\" (string) is required");
      if (!body.contains("rating") || !body["rating"].is_number_integer())
        return send_error(res, 400, "field \"rating\" must be 1 or -1");
      const long long rating = body["rating"].get<long long>();
      if (rating != 1 && rating != -1) return send_error(res, 400, "field \"rating\" must be 1 or -1");
      if (!engine.rate(body["id"].get<std::string>(), static_cast<int>(rating)))
        return send_error(res, 404, "no such history entry");
      send_json(res, 200, json{{"ok", true}});
    });

    svr.Get("/api/groups", [this](const Req&, Res& res) { send_json(res, 200, engine.groups_json()); });

    svr.Get("/api/unit", [this](const Req& req, Res& res) {
      if (!req.has_param("id")) return send_error(res, 400, "query parameter \"id\" is required");
      const std::string text = req.get_param_value("id");
      if (text.empty() || text.size() > 10 || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return send_error(res, 400, "query parameter \"id\" must be a non-negative integer");
      const json unit = engine.unit_json(static_cast<UnitId>(std::stoull(text)));
      if (unit.contains("error")) return send_json(res, 404, unit);
      send_json(res, 200, unit);
    });

    svr.Get("/api/tune", [this](const Req&, Res& res) { send_json(res, 200, tune::to_json(engine.tuning().status())); });
    svr.Post("/api/tune", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, false, &body, res)) return;
      if (!body.contains("enabled") || !body["enabled"].is_boolean())
        return send_error(res, 400, "field \"enabled\" (boolean) is required");
      const bool enabled = body["enabled"].get<bool>();
      engine.update_settings(json{{"tune", {{"enabled", enabled}}}});
      engine.tuning().set_enabled(enabled);
      send_json(res, 200, tune::to_json(engine.tuning().status()));
    });
    svr.Post("/api/tune/run", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, true, &body, res)) return;
      if (!engine.tuning().status().available)
        return send_error(res, 400, "tuning is not available (needs llama.cpp and models.tuning_base)");
      const bool started = start_job("tune", [this]() -> std::string {
        std::string error;
        if (!engine.tuning().run_now(&error)) return error.empty() ? "the tuning run did not start" : error;
        return {};
      });
      if (!started) return send_error(res, 409, "another long operation is already running");
      send_json(res, 202, json{{"started", true}});
    });

    svr.Post("/api/calibrate", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, true, &body, res)) return;
      if (!engine.corpus()) return send_error(res, 409, "no index has been built yet");
      engine::CalibrationOptions opt;
      if (body.contains("max_queries") && body["max_queries"].is_number_unsigned())
        opt.max_queries = std::min<std::size_t>(body["max_queries"].get<std::size_t>(), 100000);
      const bool started = start_job("calibrate", [this, opt]() -> std::string {
        std::string error;
        const bool ok = engine.calibrate(
            opt, [this](const engine::IndexProgress&) { return !job_cancel.load(); }, &error);
        return ok ? std::string() : (error.empty() ? "calibration failed" : error);
      });
      if (!started) return send_error(res, 409, "another long operation is already running");
      send_json(res, 202, json{{"started", true}});
    });

    svr.Post("/api/edit/preview", [this](const Req& req, Res& res) {
      json body;
      if (!parse_body(req, false, &body, res)) return;
      std::vector<std::string> warnings;
      edit::Transaction tx;
      if (body.contains("text")) {
        if (!body["text"].is_string()) return send_error(res, 400, "field \"text\" must be a string");
        tx = engine.preview_edit(body["text"].get<std::string>(), &warnings);
      } else if (body.contains("blocks")) {
        if (!body["blocks"].is_array()) return send_error(res, 400, "field \"blocks\" must be an array");
        std::vector<edit::EditBlock> blocks;
        for (const json& b : body["blocks"]) {
          if (!b.is_object() || !b.contains("path") || !b["path"].is_string() || !b.contains("replace") ||
              !b["replace"].is_string())
            return send_error(res, 400, "each block needs string fields \"path\" and \"replace\"");
          edit::EditBlock eb;
          eb.path = b["path"].get<std::string>();
          eb.replace = b["replace"].get<std::string>();
          if (b.contains("search") && b["search"].is_string()) eb.search = b["search"].get<std::string>();
          if (b.contains("scope") && b["scope"].is_string()) eb.scope = b["scope"].get<std::string>();
          const bool rewrite = b.contains("kind") ? (b["kind"].is_string() && b["kind"].get<std::string>() == "unit_rewrite")
                                                  : (!eb.scope.empty() && eb.search.empty());
          eb.kind = rewrite ? edit::EditKind::UnitRewrite : edit::EditKind::SearchReplace;
          blocks.push_back(std::move(eb));
        }
        tx = engine.preview_edit(blocks);
      } else {
        return send_error(res, 400, "provide \"text\" (model output) or \"blocks\"");
      }
      send_json(res, 200, tx_json(tx, engine.transaction_committed(tx.id), warnings));
    });
    svr.Post("/api/edit/apply", [this](const Req& req, Res& res) { handle_edit(req, res, true); });
    svr.Post("/api/edit/revert", [this](const Req& req, Res& res) { handle_edit(req, res, false); });

    // Static frontend and unknown paths.
    svr.Get(R"(/.*)", [this](const Req& req, Res& res) { handle_static(req, res); });
  }

  void handle_edit(const httplib::Request& req, httplib::Response& res, bool apply) {
    json body;
    if (!parse_body(req, false, &body, res)) return;
    if (!body.contains("id") || !body["id"].is_string()) return send_error(res, 400, "field \"id\" (string) is required");
    const std::string id = body["id"].get<std::string>();
    edit::Transaction tx;
    const edit::Status st = apply ? engine.apply_edit(id, &tx) : engine.revert_edit(id, &tx);
    if (tx.id.empty()) return send_error(res, 404, "no such transaction");
    json out = tx_json(tx, engine.transaction_committed(id));
    if (st != edit::Status::Ok) {
      out["error"] = std::string(apply ? "edit not applied: " : "edit not reverted: ") + edit::to_string(st) +
                     (tx.message.empty() ? std::string() : " (" + tx.message + ")");
      return send_json(res, 409, out);
    }
    send_json(res, 200, out);
  }

  void handle_query(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parse_body(req, false, &body, res)) return;
    if (!body.contains("question") || !body["question"].is_string() || body["question"].get<std::string>().empty())
      return send_error(res, 400, "field \"question\" (non-empty string) is required");
    engine::QueryOptions opt;
    if (body.contains("edit")) {
      if (!body["edit"].is_boolean()) return send_error(res, 400, "field \"edit\" must be a boolean");
      opt.edit = body["edit"].get<bool>();
    }
    if (body.contains("tau") && !body["tau"].is_null()) {
      if (!body["tau"].is_number()) return send_error(res, 400, "field \"tau\" must be a number");
      const double tau = body["tau"].get<double>();
      if (!(tau > 0.0 && tau < 1.0)) return send_error(res, 400, "field \"tau\" must lie in (0, 1)");
      opt.tau = tau;
    }
    const std::string question = body["question"].get<std::string>();

    res.status = 200;
    res.set_header("Cache-Control", "no-store");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("text/event-stream", [this, question, opt](std::size_t, httplib::DataSink& sink) {
      bool alive = true;
      auto send = [&](const char* event, const json& data) {
        if (!alive) return false;
        const std::string text = std::string("event: ") + event + "\ndata: " + dump(data) + "\n\n";
        alive = sink.write(text.data(), text.size());
        return alive;
      };
      try {
        const engine::QueryResult result = engine.query(
            question, opt,
            [&](const engine::QueryResult& partial) {
              if (const auto c = engine.corpus()) send("retrieval", engine::retrieval_json(partial, *c));
            },
            [&](std::string_view piece) {
              send("token", json{{"text", std::string(piece)}});
              return alive && !stopping.load();
            });
        if (!result.error.empty()) {
          send("error", json{{"error", result.error}});
        } else {
          if (result.transaction) send("edit", tx_json(*result.transaction, false, result.edit_warnings));
          send("done", engine::done_json(result));
        }
      } catch (const std::exception& ex) {
        send("error", json{{"error", ex.what()}});
      } catch (...) {
        send("error", json{{"error", "unknown error"}});
      }
      sink.done();
      return true;
    });
  }

  void handle_static(const httplib::Request& req, httplib::Response& res) {
    const std::string& path = req.path;
    if (path == "/api" || path.rfind("/api/", 0) == 0) return send_error(res, 404, "unknown API endpoint");
    if (path.find('\0') != std::string::npos || path.find('\\') != std::string::npos || path.find(':') != std::string::npos)
      return send_error(res, 403, "forbidden");
    if (frontend_root.empty() || !fs::is_directory(frontend_root)) return send_error(res, 404, "frontend not found");

    std::string rel = path.substr(1);
    std::string candidate = fs::normalize(frontend_root + "/" + rel);
    if (!fs::is_within(candidate, frontend_root)) return send_error(res, 403, "forbidden");
    if (fs::is_directory(candidate)) candidate = fs::normalize(candidate + "/index.html");
    // Resolve links and compare again, so that a link inside the frontend cannot lead out of it.
    const std::string resolved = fs::normalize(canonical_or_absolute(stdfs::path(candidate)).generic_string());
    const std::string resolved_root = fs::normalize(canonical_or_absolute(stdfs::path(frontend_root)).generic_string());
    if (!fs::is_within(resolved, resolved_root)) return send_error(res, 403, "forbidden");
    std::error_code ec;
    if (!stdfs::is_regular_file(stdfs::path(resolved), ec)) return send_error(res, 404, "not found");
    const auto content = fs::read_file(resolved);
    if (!content) return send_error(res, 404, "not found");
    res.status = 200;
    res.set_header("Cache-Control", "no-store");
    res.set_content(*content, content_type_for(lower(stdfs::path(resolved).extension().string())));
  }
};

Server::Server(engine::Engine& engine, ServerOptions options) : impl_(std::make_unique<Impl>(engine, std::move(options))) {}
Server::~Server() = default;

bool Server::bind(std::string* error) {
  Impl& d = *impl_;
  if (d.bound) return true;
  int port = d.options.port;
  if (port == 0) {
    port = d.svr.bind_to_any_port(d.options.host);
  } else if (!d.svr.bind_to_port(d.options.host, port)) {
    port = -1;
  }
  if (port <= 0) {
    if (error) *error = "cannot bind " + d.options.host + ":" + std::to_string(d.options.port) + " (address in use or unavailable)";
    return false;
  }
  d.bound_port = port;
  d.bound = true;
  return true;
}

int Server::port() const { return impl_->bound_port; }

bool Server::run() {
  if (!impl_->bound) return false;
  impl_->svr.listen_after_bind();
  return true;
}

void Server::stop() {
  impl_->stopping = true;
  impl_->job_cancel = true;
  impl_->svr.stop();
}

bool Server::running() const { return impl_->svr.is_running(); }

}  // namespace prag::server
