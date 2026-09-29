// HTTP interface of the engine (DESIGN.md sections 6.14 and 9).
#pragma once

#include <memory>
#include <string>

#include "prag/engine/engine.hpp"

namespace prag::server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8000;
  std::string frontend_dir = "frontend";
  int worker_threads = 8;
};

class Server {
 public:
  // The engine must outlive the server.
  Server(engine::Engine& engine, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Binds the socket. Returns false (with the reason in *error) when the
  // address is unavailable. With port 0 an unused port is chosen.
  bool bind(std::string* error = nullptr);
  int port() const;

  // Serves until stop() is called from another thread. Returns false when the
  // server was not bound.
  bool run();
  void stop();
  bool running() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace prag::server
