// Shared helpers of the engine tests: a synthetic C corpus in a temporary directory.
#pragma once

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace engine_test {

inline void write_text(const std::filesystem::path& p, const std::string& text) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << text;
}

inline std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

struct Theme {
  const char* file;
  const char* prefix;
  const char* words;  // what the functions work on
};

// Twelve files, three documented functions each, plus the planted function in net/checksum.c.
inline void write_corpus(const std::filesystem::path& root) {
  const Theme themes[] = {
      {"ds/ring.c", "ring", "circular ring queue of fixed capacity"},
      {"ds/table.c", "table", "open addressing hash table with string keys"},
      {"text/builder.c", "sbuild", "growable string builder for text output"},
      {"math/matrix.c", "mat", "dense matrix of double precision values"},
      {"ds/list.c", "dlist", "doubly linked list of generic nodes"},
      {"ds/bits.c", "bitset", "compact bitset of boolean flags"},
      {"mem/arena.c", "arena", "arena region allocator for temporary memory"},
      {"time/date.c", "date", "calendar date parsing and formatting in ISO style"},
      {"io/log.c", "logger", "leveled logger writing messages to a stream"},
      {"sys/pool.c", "tpool", "worker thread pool with a job queue"},
      {"text/json.c", "jesc", "json string escaping of control characters"},
  };
  for (const Theme& t : themes) {
    const std::string p = t.prefix;
    std::string s = "#include <stdlib.h>\n\n";
    s += "/* Initialises the " + std::string(t.words) + " and allocates its storage. */\n";
    s += "int " + p + "_init(void* self, int capacity) {\n  if (capacity <= 0) return -1;\n  return " + p +
         "_reset(self);\n}\n\n";
    s += "/* Resets the " + std::string(t.words) + " to an empty state without freeing. */\n";
    s += "int " + p + "_reset(void* self) {\n  (void)self;\n  return 0;\n}\n\n";
    s += "/* Releases every resource held by the " + std::string(t.words) + ". */\n";
    s += "void " + p + "_destroy(void* self) {\n  " + p + "_reset(self);\n  free(self);\n}\n";
    write_text(root / t.file, s);
  }
  write_text(root / "net/checksum.c",
             "#include <stdint.h>\n#include <stddef.h>\n\n"
             "/* Adds two sixteen bit words with end around carry for the internet checksum. */\n"
             "static uint16_t ones_add(uint16_t a, uint16_t b) {\n  uint32_t s = (uint32_t)a + b;\n"
             "  return (uint16_t)((s & 0xffff) + (s >> 16));\n}\n\n"
             "/* Computes the Fletcher checksum of a packet buffer using two running modular sums. */\n"
             "uint32_t fletcher_packet_checksum(const uint8_t* buffer, size_t length) {\n"
             "  uint32_t sum1 = 0, sum2 = 0;\n  for (size_t i = 0; i < length; ++i) {\n"
             "    sum1 = (sum1 + buffer[i]) % 65535;\n    sum2 = (sum2 + sum1) % 65535;\n  }\n"
             "  return (sum2 << 16) | sum1;\n}\n\n"
             "/* Verifies a received packet by recomputing its Fletcher checksum and comparing. */\n"
             "int packet_verify(const uint8_t* buffer, size_t length, uint32_t expected) {\n"
             "  return fletcher_packet_checksum(buffer, length) == expected;\n}\n");
}

// A fresh directory under the system temporary directory with a corpus and a settings file.
inline std::filesystem::path make_workspace(const std::string& name, const nlohmann::json& extra = {}) {
  const auto dir = std::filesystem::temp_directory_path() / ("prag_engine_" + name);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
  write_corpus(dir / "corpus");
  nlohmann::json settings{{"corpus", {{"roots", {"corpus"}}}}, {"paths", {{"data", "data"}}},
                          {"exec", {{"threads", 4}}}};
  if (!extra.is_null()) settings.merge_patch(extra);
  write_text(dir / "settings.json", settings.dump(2));
  return dir;
}

}  // namespace engine_test
