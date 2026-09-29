// Thread-safe levelled logging to stderr.
#pragma once

namespace prag::log {

enum class Level : int { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

void set_level(Level level) noexcept;
Level level() noexcept;

// printf-style; a newline is appended. Messages below the current level are dropped.
void write(Level level, const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

}  // namespace prag::log

#define PRAG_DEBUG(...) ::prag::log::write(::prag::log::Level::Debug, __VA_ARGS__)
#define PRAG_INFO(...) ::prag::log::write(::prag::log::Level::Info, __VA_ARGS__)
#define PRAG_WARN(...) ::prag::log::write(::prag::log::Level::Warn, __VA_ARGS__)
#define PRAG_ERROR(...) ::prag::log::write(::prag::log::Level::Error, __VA_ARGS__)
