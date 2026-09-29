#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <filesystem>
#include <string>

#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/settings.hpp"
#include "prag/core/text.hpp"
#include "prag/core/types.hpp"

using namespace prag;

namespace {

std::string temp_dir(const char* name) {
  const auto dir = std::filesystem::temp_directory_path() / "prag-core-tests" / name;
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  return fs::normalize(dir.string());
}

}  // namespace

PRAG_TEST(hash_is_deterministic_and_seeded) {
  CHECK_EQ(hash64("mi_malloc"), hash64("mi_malloc"));
  CHECK(hash64("mi_malloc") != hash64("mi_malloc "));
  CHECK(hash64("mi_malloc", 1) != hash64("mi_malloc", 2));
  CHECK(hash64("") != 0u);
}

PRAG_TEST(split_identifier_handles_snake_camel_and_acronyms) {
  const auto a = text::split_identifier("mi_heapMallocAligned2");
  REQUIRE(a.size() == 5u);
  CHECK_EQ(a[0], std::string("mi"));
  CHECK_EQ(a[1], std::string("heap"));
  CHECK_EQ(a[2], std::string("malloc"));
  CHECK_EQ(a[3], std::string("aligned"));
  CHECK_EQ(a[4], std::string("2"));

  const auto b = text::split_identifier("HTTPServer");
  REQUIRE(b.size() == 2u);
  CHECK_EQ(b[0], std::string("http"));
  CHECK_EQ(b[1], std::string("server"));

  CHECK(text::split_identifier("").empty());
  CHECK(text::split_identifier("___").empty());
}

PRAG_TEST(estimate_tokens_is_monotone_in_content) {
  CHECK_EQ(text::estimate_tokens(""), 0u);
  const std::uint32_t one = text::estimate_tokens("return x;");
  const std::uint32_t two = text::estimate_tokens("return x;\nreturn y;");
  CHECK(one >= 3u);
  CHECK(two > one);
  // Whitespace alone is nearly free.
  CHECK(text::estimate_tokens("   ") == 0u);
}

PRAG_TEST(count_lines_and_split_lines_agree) {
  CHECK_EQ(text::count_lines(""), 0u);
  CHECK_EQ(text::count_lines("a"), 1u);
  CHECK_EQ(text::count_lines("a\n"), 1u);
  CHECK_EQ(text::count_lines("a\nb"), 2u);
  const auto lines = text::split_lines("a\r\nb\n\nc");
  REQUIRE(lines.size() == 4u);
  CHECK_EQ(std::string(lines[0]), std::string("a"));
  CHECK_EQ(std::string(lines[1]), std::string("b"));
  CHECK_EQ(std::string(lines[2]), std::string(""));
  CHECK_EQ(std::string(lines[3]), std::string("c"));
}

PRAG_TEST(sanitize_utf8_replaces_invalid_sequences) {
  CHECK_EQ(text::sanitize_utf8("plain"), std::string("plain"));
  CHECK_EQ(text::sanitize_utf8("caf\xC3\xA9"), std::string("caf\xC3\xA9"));
  CHECK_EQ(text::sanitize_utf8("a\xFF" "b"), std::string("a\xEF\xBF\xBD" "b"));
  // Truncated two-byte sequence at the end of the input.
  CHECK_EQ(text::sanitize_utf8("a\xC3"), std::string("a\xEF\xBF\xBD"));
  // Overlong encoding of '/'.
  CHECK_EQ(text::sanitize_utf8("\xC0\xAF"), std::string("\xEF\xBF\xBD\xEF\xBF\xBD"));
}

PRAG_TEST(atomic_write_round_trips_and_replaces) {
  const std::string dir = temp_dir("atomic");
  const std::string path = dir + "/nested/file.txt";
  REQUIRE(fs::write_file_atomic(path, "first"));
  CHECK_EQ(fs::read_file(path).value_or(""), std::string("first"));
  REQUIRE(fs::write_file_atomic(path, std::string("second\0binary", 13)));
  CHECK_EQ(fs::read_file(path).value_or(""), std::string("second\0binary", 13));
  CHECK_EQ(fs::file_size(path), 13u);
  // No temporary files are left behind.
  std::size_t entries = 0;
  for (const auto& e : std::filesystem::directory_iterator(dir + "/nested")) {
    (void)e;
    ++entries;
  }
  CHECK_EQ(entries, 1u);
}

PRAG_TEST(is_within_rejects_traversal_and_prefix_collisions) {
  const std::string dir = temp_dir("within");
  CHECK(fs::is_within(dir + "/a/b.c", dir));
  CHECK(fs::is_within(dir, dir));
  CHECK(!fs::is_within(dir + "/../outside.c", dir));
  CHECK(!fs::is_within(dir + "-sibling/file.c", dir));
  CHECK(!fs::is_within(dir + "/a/../../escape.c", dir));
}

PRAG_TEST(walk_filters_by_extension_size_and_exclusion) {
  const std::string dir = temp_dir("walk");
  REQUIRE(fs::write_file_atomic(dir + "/src/a.c", "int a;"));
  REQUIRE(fs::write_file_atomic(dir + "/src/b.H", "int b;"));
  REQUIRE(fs::write_file_atomic(dir + "/src/c.txt", "text"));
  REQUIRE(fs::write_file_atomic(dir + "/build/gen.c", "int g;"));
  REQUIRE(fs::write_file_atomic(dir + "/src/big.c", std::string(4096, 'x')));

  fs::WalkOptions options;
  options.extensions = {".c", ".h"};
  options.exclude_substrings = {"/build/"};
  options.max_file_bytes = 1024;
  const auto files = fs::walk(dir, options);
  REQUIRE(files.size() == 2u);
  CHECK(files[0].ends_with("/src/a.c"));
  CHECK(files[1].ends_with("/src/b.H"));

  const auto single = fs::walk(dir + "/src/a.c", options);
  REQUIRE(single.size() == 1u);
}

PRAG_TEST(settings_dotted_keys_defaults_and_merge) {
  Settings s;
  CHECK_NEAR(s.get<double>("retrieve.tau", 0.25), 0.25, 0.0);
  s.set("retrieve.tau", 0.4);
  s.set("server.port", 8123);
  CHECK_NEAR(s.get<double>("retrieve.tau", 0.25), 0.4, 1e-12);
  CHECK_EQ(s.get<int>("server.port", 8000), 8123);
  // Wrong type falls back instead of throwing.
  CHECK_EQ(s.get<std::string>("server.port", "fallback"), std::string("fallback"));
  CHECK(s.has("retrieve.tau"));
  CHECK(!s.has("retrieve.missing"));

  s.merge(nlohmann::json::parse(R"({"retrieve": {"max_units": 32}, "server": {"port": 9000}})"));
  CHECK_NEAR(s.get<double>("retrieve.tau", 0.0), 0.4, 1e-12);
  CHECK_EQ(s.get<int>("retrieve.max_units", 0), 32);
  CHECK_EQ(s.get<int>("server.port", 0), 9000);

  const std::string dir = temp_dir("settings");
  REQUIRE(s.save(dir + "/prag.json"));
  Settings loaded;
  REQUIRE(loaded.load(dir + "/prag.json"));
  CHECK_EQ(loaded.get<int>("server.port", 0), 9000);

  REQUIRE(fs::write_file_atomic(dir + "/broken.json", "{ not json"));
  CHECK(!loaded.load(dir + "/broken.json"));
  CHECK_EQ(loaded.get<int>("server.port", 0), 9000);

  Settings missing;
  CHECK(missing.load(dir + "/does-not-exist.json"));
}

PRAG_TEST(enum_names) {
  CHECK_EQ(std::string(to_string(UnitKind::Function)), std::string("function"));
  CHECK_EQ(std::string(to_string(EdgeKind::Callee)), std::string("callee"));
}
