#include "prag/core/log.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace prag::log {

namespace {

std::atomic<int> g_level{static_cast<int>(Level::Info)};

std::mutex& output_mutex() {
  static std::mutex m;
  return m;
}

const char* tag(Level level) {
  switch (level) {
    case Level::Debug: return "debug";
    case Level::Info: return "info ";
    case Level::Warn: return "warn ";
    case Level::Error: return "error";
    case Level::Off: break;
  }
  return "     ";
}

}  // namespace

void set_level(Level level) noexcept { g_level.store(static_cast<int>(level), std::memory_order_relaxed); }

Level level() noexcept { return static_cast<Level>(g_level.load(std::memory_order_relaxed)); }

void write(Level message_level, const char* format, ...) {
  if (static_cast<int>(message_level) < g_level.load(std::memory_order_relaxed)) return;
  if (message_level == Level::Off) return;

  char buffer[2048];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);

  std::lock_guard<std::mutex> lock(output_mutex());
  std::fprintf(stderr, "[%s] %s\n", tag(message_level), buffer);
  std::fflush(stderr);
}

}  // namespace prag::log
