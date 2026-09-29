// Transactions: prepare, commit, revert and recover, and the safety properties
// S2 (ambiguity), S3 (containment), S4 (atomicity), S5 (byte preservation) and
// S6 (overlap) of the edit module.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "../internal.hpp"
#include "edit_test_util.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/core/text.hpp"
#include "prag/edit/edit.hpp"

using namespace prag;
using namespace prag::edit;
using edit_test::options_for;
using edit_test::read_file;
using edit_test::search_replace;
using edit_test::temp_dir;
using edit_test::unit_rewrite;
using edit_test::write_file;

namespace {

const char* const kSource =
    "static int counter = 0;\n"  // line 1
    "\n"                         // line 2
    "int first(int x) {\n"       // line 3
    "  if (x < 0) {\n"           // line 4
    "    return 0;\n"            // line 5
    "  }\n"                      // line 6
    "  return x;\n"              // line 7
    "}\n"                        // line 8
    "\n"                         // line 9
    "int second(int x) {\n"      // line 10
    "  if (x > 9) {\n"           // line 11
    "    return 0;\n"            // line 12
    "  }\n"                      // line 13
    "  return x + 1;\n"          // line 14
    "}\n";                       // line 15

std::string replaced(std::string text, std::string_view from, std::string_view to, std::size_t occurrence = 0) {
  std::size_t at = text.find(from);
  for (std::size_t k = 0; k < occurrence && at != std::string::npos; ++k) at = text.find(from, at + 1);
  if (at == std::string::npos) return "<text to replace not found>";
  return text.replace(at, from.size(), to);
}

std::string to_crlf(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (c == '\n') out.push_back('\r');
    out.push_back(c);
  }
  return out;
}

// Stand-in for parse::find_unit. A unit ends with a line that starts with '}';
// the blank lines before a unit belong to it, as they do in the real parser.
bool locate_unit(std::string_view, std::string_view text, std::string_view name, std::size_t* begin,
                 std::size_t* end) {
  const std::string call = " " + std::string(name) + "(";
  std::size_t unit_begin = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    const std::size_t newline = text.find('\n', pos);
    const std::size_t next = newline == std::string_view::npos ? text.size() : newline + 1;
    if (text[pos] == '}') {
      if (text.substr(unit_begin, next - unit_begin).find(call) != std::string_view::npos) {
        *begin = unit_begin;
        *end = next;
        return true;
      }
      unit_begin = next;
    }
    pos = next;
  }
  return false;
}

// Stand-in for parse::count_syntax_errors: unbalanced braces.
std::size_t count_errors(std::string_view, std::string_view text) {
  const auto open = std::count(text.begin(), text.end(), '{');
  const auto close = std::count(text.begin(), text.end(), '}');
  return static_cast<std::size_t>(open > close ? open - close : close - open);
}

struct QuietLog {
  QuietLog() : previous(log::level()) { log::set_level(log::Level::Off); }
  ~QuietLog() { log::set_level(previous); }
  log::Level previous;
};

struct FaultGuard {
  explicit FaultGuard(detail::Fault fault) { detail::set_fault(fault); }
  ~FaultGuard() { detail::set_fault(detail::Fault::None); }
};

// commit() and revert(); trouble with the file system is printed, because it
// is the one failure that does not come from the test itself.
Status committed(Transaction& tx, const ApplyOptions& options) {
  const Status status = commit(tx, options);
  if (status == Status::IoError) std::printf("  (commit: %s: %s)\n", to_string(status), tx.message.c_str());
  return status;
}

Status reverted(const Transaction& tx) {
  const Status status = revert(tx);
  if (status == Status::IoError) std::printf("  (revert: %s)\n", to_string(status));
  return status;
}

bool valid_id(const std::string& id) {
  if (id.size() != 19 || id.compare(0, 3, "tx-") != 0) return false;
  return std::all_of(id.begin() + 3, id.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

}  // namespace

// ---- the basic cycle --------------------------------------------------------------------

PRAG_TEST(prepare_commit_revert) {
  const std::string root = temp_dir("cycle");
  const std::string path = root + "/src/a.c";
  REQUIRE(write_file(path, kSource));
  const ApplyOptions options = options_for(root);

  Transaction tx = prepare({search_replace("src/a.c", "  return x + 1;\n", "  counter++;\n  return x + 2;\n")}, options);
  CHECK(tx.status == Status::Ok);
  CHECK(valid_id(tx.id));
  CHECK(tx.message.empty());
  REQUIRE(tx.files.size() == 1u);
  const FileProposal& file = tx.files[0];
  CHECK(file.status == Status::Ok);
  CHECK_EQ(file.path, path);
  CHECK_EQ(file.display_path, std::string("src/a.c"));
  CHECK_EQ(file.original, std::string(kSource));
  CHECK_EQ(file.original_hash, hash64(kSource));
  const std::string expected = replaced(kSource, "  return x + 1;\n", "  counter++;\n  return x + 2;\n");
  CHECK_EQ(file.updated, expected);
  CHECK(!file.creates_file);
  CHECK_EQ(file.line_ending, std::string("\n"));
  CHECK(file.message.empty());
  REQUIRE(file.matches.size() == 1u);
  CHECK(file.matches[0].kind == MatchKind::Exact);
  CHECK(file.matches[0].status == Status::Ok);
  CHECK_EQ(file.matches[0].first_line, 14u);
  CHECK_EQ(file.matches[0].last_line, 14u);
  CHECK_EQ(file.matches[0].candidates, std::size_t{1});
  CHECK_EQ(file.diff, std::string("--- a/src/a.c\n"
                                  "+++ b/src/a.c\n"
                                  "@@ -11,5 +11,6 @@\n"
                                  "   if (x > 9) {\n"
                                  "     return 0;\n"
                                  "   }\n"
                                  "-  return x + 1;\n"
                                  "+  counter++;\n"
                                  "+  return x + 2;\n"
                                  " }\n"));
  // prepare() writes nothing.
  CHECK_EQ(read_file(path), std::string(kSource));
  CHECK(!fs::exists(options.backup_dir));

  CHECK(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(path), expected);
  CHECK(!tx.journal_path.empty());
  CHECK(!fs::exists(tx.journal_path));  // removed at the commit point
  CHECK(fs::is_within(tx.journal_path, options.backup_dir));
  REQUIRE(!tx.files[0].backup_path.empty());
  CHECK(fs::is_within(tx.files[0].backup_path, options.backup_dir));
  CHECK_EQ(read_file(tx.files[0].backup_path), std::string(kSource));

  CHECK(reverted(tx) == Status::Ok);
  CHECK_EQ(read_file(path), std::string(kSource));
  // Reverting twice changes nothing.
  CHECK(reverted(tx) == Status::Ok);
  CHECK_EQ(read_file(path), std::string(kSource));
  // Only the target is left in the source directory: no temporary files.
  CHECK_EQ(edit_test::count_entries(root + "/src"), std::size_t{1});
}

PRAG_TEST(the_identifier_follows_the_content) {
  const std::string root = temp_dir("id");
  REQUIRE(write_file(root + "/a.c", kSource));
  const ApplyOptions options = options_for(root);
  const Transaction one = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  const Transaction same = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  const Transaction other = prepare({search_replace("a.c", "  return x;\n", "  return +x;\n")}, options);
  CHECK(valid_id(one.id));
  CHECK_EQ(one.id, same.id);
  CHECK(one.id != other.id);

  const Transaction nothing = prepare({}, options);
  CHECK(nothing.status == Status::NotFound);
  CHECK(nothing.files.empty());
  CHECK(!nothing.message.empty());
  Transaction copy = nothing;
  CHECK(committed(copy, options) == Status::NotFound);
  CHECK(reverted(nothing) == Status::NotFound);
}

PRAG_TEST(commit_needs_a_backup_directory) {
  const std::string root = temp_dir("nobackup");
  REQUIRE(write_file(root + "/a.c", kSource));
  ApplyOptions options = options_for(root);
  options.backup_dir.clear();
  Transaction tx = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  REQUIRE(tx.status == Status::Ok);
  CHECK(committed(tx, options) == Status::IoError);
  CHECK(!tx.message.empty());
  CHECK_EQ(read_file(root + "/a.c"), std::string(kSource));
}

PRAG_TEST(a_dry_run_writes_nothing) {
  const std::string root = temp_dir("dry");
  REQUIRE(write_file(root + "/a.c", kSource));
  ApplyOptions options = options_for(root);
  options.dry_run = true;
  Transaction tx = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n"),
                            search_replace("new.c", "", "int fresh;\n")},
                           options);
  REQUIRE(tx.status == Status::Ok);
  REQUIRE(tx.files.size() == 2u);
  CHECK_EQ(tx.files[0].updated, replaced(kSource, "  return x;\n", "  return -x;\n"));
  CHECK(!tx.files[0].diff.empty());
  CHECK(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(root + "/a.c"), std::string(kSource));
  CHECK(!fs::exists(root + "/new.c"));
  CHECK(!fs::exists(options.backup_dir));
  CHECK(tx.journal_path.empty());
  CHECK(tx.files[0].backup_path.empty());
  CHECK_EQ(edit_test::count_entries(root), std::size_t{1});

  // A dry run still notices that the file has changed.
  REQUIRE(write_file(root + "/a.c", "int other;\n"));
  CHECK(committed(tx, options) == Status::Conflict);
}

PRAG_TEST(without_backups_revert_uses_the_snapshot) {
  const std::string root = temp_dir("nokeep");
  REQUIRE(write_file(root + "/a.c", kSource));
  ApplyOptions options = options_for(root);
  options.keep_backup = false;
  Transaction tx = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  REQUIRE(committed(tx, options) == Status::Ok);
  CHECK(tx.files[0].backup_path.empty());
  CHECK_EQ(edit_test::count_entries(options.backup_dir), std::size_t{0});
  CHECK_EQ(read_file(root + "/a.c"), replaced(kSource, "  return x;\n", "  return -x;\n"));
  CHECK(reverted(tx) == Status::Ok);
  CHECK_EQ(read_file(root + "/a.c"), std::string(kSource));
}

// ---- S2: ambiguity ------------------------------------------------------------------------

PRAG_TEST(s2_ambiguous_text_is_not_applied) {
  const std::string root = temp_dir("s2");
  const std::string path = root + "/a.c";
  REQUIRE(write_file(path, kSource));
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;

  Transaction tx = prepare({search_replace("a.c", "    return 0;\n", "    return -1;\n")}, options);
  CHECK(tx.status == Status::Ambiguous);
  REQUIRE(tx.files.size() == 1u);
  CHECK(tx.files[0].status == Status::Ambiguous);
  CHECK(!tx.files[0].message.empty());
  CHECK(!tx.message.empty());
  CHECK_EQ(tx.files[0].updated, std::string(kSource));
  CHECK(tx.files[0].diff.empty());
  REQUIRE(tx.files[0].matches.size() == 1u);
  CHECK(tx.files[0].matches[0].status == Status::Ambiguous);
  CHECK_EQ(tx.files[0].matches[0].candidates, std::size_t{2});
  CHECK(committed(tx, options) == Status::Ambiguous);
  CHECK(reverted(tx) == Status::Ambiguous);
  CHECK_EQ(read_file(path), std::string(kSource));
  CHECK(!fs::exists(options.backup_dir));

  // With a scope that holds one occurrence the same block is applied.
  Transaction scoped = prepare({search_replace("a.c", "    return 0;\n", "    return -1;\n", "second")}, options);
  CHECK(scoped.status == Status::Ok);
  REQUIRE(scoped.files.size() == 1u);
  CHECK_EQ(scoped.files[0].matches[0].first_line, 12u);
  const std::string expected = replaced(kSource, "    return 0;\n", "    return -1;\n", 1);
  CHECK_EQ(scoped.files[0].updated, expected);
  CHECK(committed(scoped, options) == Status::Ok);
  CHECK_EQ(read_file(path), expected);
  CHECK(reverted(scoped) == Status::Ok);

  Transaction other = prepare({search_replace("a.c", "    return 0;\n", "    return -1;\n", "first")}, options);
  CHECK(other.status == Status::Ok);
  CHECK_EQ(other.files[0].matches[0].first_line, 5u);
  CHECK_EQ(other.files[0].updated, replaced(kSource, "    return 0;\n", "    return -1;\n", 0));
}

PRAG_TEST(s2_the_scope_must_exist) {
  const std::string root = temp_dir("s2scope");
  REQUIRE(write_file(root + "/a.c", kSource));
  ApplyOptions options = options_for(root);

  // Without a locator a scope cannot be resolved.
  const Transaction none = prepare({search_replace("a.c", "    return 0;\n", "    return -1;\n", "second")}, options);
  CHECK(none.status == Status::ScopeNotFound);
  options.unit_locator = locate_unit;
  const Transaction unknown = prepare({search_replace("a.c", "    return 0;\n", "    return -1;\n", "third")}, options);
  CHECK(unknown.status == Status::ScopeNotFound);
  CHECK(unknown.files[0].message.find("third") != std::string::npos);
  // Text of another unit is not found inside the scope.
  const Transaction elsewhere = prepare({search_replace("a.c", "  return x + 1;\n", "  return x;\n", "first")}, options);
  CHECK(elsewhere.status == Status::NotFound);
  // Two occurrences inside the scope.
  REQUIRE(write_file(root + "/b.c", "int twice(int x) {\n  x++;\n  x++;\n  return x;\n}\n"));
  const Transaction twice = prepare({search_replace("b.c", "  x++;\n", "  x--;\n", "twice")}, options);
  CHECK(twice.status == Status::Ambiguous);
  // A locator that reports nonsense is not trusted.
  options.unit_locator = [](std::string_view, std::string_view text, std::string_view, std::size_t* begin,
                            std::size_t* end) {
    *begin = 10;
    *end = text.size() + 1;
    return true;
  };
  CHECK(prepare({search_replace("a.c", "  return x;\n", "  return 1;\n", "first")}, options).status ==
        Status::ScopeNotFound);
  CHECK_EQ(read_file(root + "/a.c"), std::string(kSource));
}

// ---- S3: containment ------------------------------------------------------------------------

PRAG_TEST(s3_paths_inside_the_root) {
  const std::string root = temp_dir("s3in");
  const std::string path = root + "/src/a.c";
  REQUIRE(write_file(path, kSource));
  const ApplyOptions options = options_for(root);
  const std::string expected = replaced(kSource, "  return x;\n", "  return -x;\n");

  for (const std::string& written :
       {std::string("src/a.c"), std::string("./src/a.c"), std::string("src/../src/a.c"), std::string("src//a.c"),
        std::string("src\\a.c"), path, std::string("  src/a.c  ")}) {
    const Transaction tx = prepare({search_replace(written, "  return x;\n", "  return -x;\n")}, options);
    if (tx.status != Status::Ok) std::fprintf(stderr, "  path: %s\n", written.c_str());
    CHECK(tx.status == Status::Ok);
    REQUIRE(tx.files.size() == 1u);
    CHECK_EQ(tx.files[0].path, path);
    CHECK_EQ(tx.files[0].display_path, std::string("src/a.c"));
    CHECK_EQ(tx.files[0].updated, expected);
  }
  // A sentence in front of the name of an existing file.
  const Transaction sentence = prepare({search_replace("the change goes into src/a.c.", "  return x;\n", "  return -x;\n")}, options);
  CHECK(sentence.status == Status::Ok);
  CHECK_EQ(sentence.files[0].path, path);
}

PRAG_TEST(s3_paths_outside_the_root_are_refused) {
  const std::string base = temp_dir("s3out");
  const std::string root = base + "/root";
  const std::string outside = base + "/outside.c";
  const std::string sibling = base + "/root-other/a.c";
  REQUIRE(write_file(root + "/src/a.c", kSource));
  REQUIRE(write_file(outside, kSource));
  REQUIRE(write_file(sibling, kSource));
  const ApplyOptions options = options_for(root);

  const std::vector<std::string> refused = {
      "../outside.c",
      "src/../../outside.c",
      "src/../../root-other/a.c",
      "../root-other/a.c",
      "..\\outside.c",
      "src\\..\\..\\outside.c",
      outside,
      sibling,
      base + "/root/../outside.c",
      "../../../../../../../../../../outside.c",
      "/outside.c",
#ifdef _WIN32
      "C:outside.c",
      "\\\\localhost\\share\\a.c",
      "src/a.c:stream",
#endif
  };
  for (const std::string& written : refused) {
    // Both an edit of an existing file and the creation of a new one.
    for (const char* search : {"  return x;\n", ""}) {
      Transaction tx = prepare({search_replace(written, search, "  return -x;\n")}, options);
      if (tx.status != Status::OutsideRoots) {
        std::fprintf(stderr, "  path: %s gives %s\n", written.c_str(), to_string(tx.status));
      }
      CHECK(tx.status == Status::OutsideRoots);
      REQUIRE(tx.files.size() == 1u);
      CHECK(tx.files[0].status == Status::OutsideRoots);
      CHECK(tx.files[0].updated.empty());
      CHECK(committed(tx, options) == Status::OutsideRoots);
    }
  }
  CHECK_EQ(read_file(outside), std::string(kSource));
  CHECK_EQ(read_file(sibling), std::string(kSource));
  CHECK_EQ(read_file(root + "/src/a.c"), std::string(kSource));
  CHECK(!fs::exists(options.backup_dir));
  CHECK_EQ(edit_test::count_entries(base), std::size_t{6});  // 3 files and 3 directories, as before

  // No root at all, and no name at all.
  ApplyOptions closed = options;
  closed.allowed_roots.clear();
  CHECK(prepare({search_replace("src/a.c", "  return x;\n", "  return -x;\n")}, closed).status == Status::OutsideRoots);
  CHECK(prepare({search_replace("", "  return x;\n", "  return -x;\n")}, options).status == Status::FileMissing);
  CHECK(prepare({search_replace("   ", "", "int x;\n")}, options).status == Status::FileMissing);
  // A directory is not a file.
  CHECK(prepare({search_replace("src", "", "int x;\n")}, options).status == Status::IoError);
}

PRAG_TEST(s3_commit_checks_the_roots_again) {
  const std::string base = temp_dir("s3commit");
  const std::string root = base + "/root";
  const std::string other = base + "/other";
  REQUIRE(write_file(root + "/a.c", kSource));
  REQUIRE(write_file(other + "/a.c", kSource));
  const ApplyOptions options = options_for(root);

  // A proposal that was prepared for one root is not written under another,
  // and a path that was changed afterwards is not written at all.
  Transaction tx = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options_for(other));
  REQUIRE(tx.status == Status::Ok);
  CHECK(committed(tx, options) == Status::OutsideRoots);
  CHECK(!tx.message.empty());
  Transaction forged = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  REQUIRE(forged.status == Status::Ok);
  forged.files[0].path = root + "/../other/a.c";
  CHECK(committed(forged, options) == Status::OutsideRoots);
  forged.files[0].path = root + "/a.c";
  forged.id = "tx-../../x";
  CHECK(committed(forged, options) == Status::IoError);
  CHECK_EQ(read_file(root + "/a.c"), std::string(kSource));
  CHECK_EQ(read_file(other + "/a.c"), std::string(kSource));
  CHECK(!fs::exists(options.backup_dir));
}

PRAG_TEST(s3_several_roots) {
  const std::string base = temp_dir("s3roots");
  const std::string one = base + "/one";
  const std::string two = base + "/two";
  REQUIRE(write_file(one + "/only1.c", kSource));
  REQUIRE(write_file(two + "/only2.c", kSource));
  REQUIRE(write_file(one + "/both.c", kSource));
  REQUIRE(write_file(two + "/both.c", kSource));
  ApplyOptions options;
  options.allowed_roots = {one, two};
  options.backup_dir = base + "/edits";

  const Transaction first = prepare({search_replace("only1.c", "  return x;\n", "  return 1;\n")}, options);
  CHECK(first.status == Status::Ok);
  CHECK_EQ(first.files[0].path, one + "/only1.c");
  const Transaction second = prepare({search_replace("only2.c", "  return x;\n", "  return 2;\n")}, options);
  CHECK(second.status == Status::Ok);
  CHECK_EQ(second.files[0].path, two + "/only2.c");
  CHECK_EQ(second.files[0].display_path, std::string("only2.c"));

  // The same relative name under two roots: the file that is meant is unknown.
  const Transaction both = prepare({search_replace("both.c", "  return x;\n", "  return 3;\n")}, options);
  CHECK(both.status == Status::Ambiguous);
  CHECK(!both.files[0].message.empty());
  // The absolute path says which one.
  const Transaction exact = prepare({search_replace(two + "/both.c", "  return x;\n", "  return 3;\n")}, options);
  CHECK(exact.status == Status::Ok);
  CHECK_EQ(exact.files[0].path, two + "/both.c");
  // A new file goes below the first root.
  const Transaction fresh = prepare({search_replace("dir/new.c", "", "int x;\n")}, options);
  CHECK(fresh.status == Status::Ok);
  CHECK_EQ(fresh.files[0].path, one + "/dir/new.c");

  // A root that is a single file admits that file and nothing next to it.
  ApplyOptions single;
  single.allowed_roots = {one + "/only1.c"};
  single.backup_dir = base + "/edits";
  const Transaction inside = prepare({search_replace("only1.c", "  return x;\n", "  return 1;\n")}, single);
  CHECK(inside.status == Status::Ok);
  CHECK_EQ(inside.files[0].path, one + "/only1.c");
  CHECK(prepare({search_replace(one + "/only1.c", "  return x;\n", "  return 1;\n")}, single).status == Status::Ok);
  CHECK(prepare({search_replace("both.c", "  return x;\n", "  return 1;\n")}, single).status == Status::OutsideRoots);
  CHECK(prepare({search_replace(one + "/both.c", "  return x;\n", "  return 1;\n")}, single).status ==
        Status::OutsideRoots);
}

PRAG_TEST(s3_spellings_of_one_file_are_one_file) {
  const std::string root = temp_dir("s3spell");
  const std::string path = root + "/src/a.c";
  REQUIRE(write_file(path, kSource));
  const ApplyOptions options = options_for(root);

  // Two blocks for the same lines through two spellings must collide.
  const Transaction tx = prepare({search_replace("src/a.c", "  return x;\n", "  return 1;\n"),
                                  search_replace("./src/../src/a.c", "  return x;\n", "  return 2;\n")},
                                 options);
  CHECK(tx.status == Status::Overlap);
  REQUIRE(tx.files.size() == 1u);
  CHECK_EQ(tx.files[0].matches.size(), std::size_t{2});

#ifdef _WIN32
  // Windows: case and backslashes make no difference, and the proposal names
  // the file as it is spelled on disk.
  for (const char* written : {"SRC/A.C", "Src\\a.C", "src\\A.c"}) {
    const Transaction upper = prepare({search_replace(written, "  return x;\n", "  return 1;\n")}, options);
    CHECK(upper.status == Status::Ok);
    REQUIRE(upper.files.size() == 1u);
    CHECK_EQ(upper.files[0].path, path);
    CHECK_EQ(upper.files[0].display_path, std::string("src/a.c"));
  }
  const Transaction mixed = prepare({search_replace("src/a.c", "  return x;\n", "  return 1;\n"),
                                     search_replace("SRC\\A.C", "  return x + 1;\n", "  return 2;\n")},
                                    options);
  CHECK(mixed.status == Status::Ok);
  REQUIRE(mixed.files.size() == 1u);
  CHECK_EQ(mixed.files[0].matches.size(), std::size_t{2});
  CHECK_EQ(mixed.files[0].updated,
           replaced(replaced(kSource, "  return x;\n", "  return 1;\n"), "  return x + 1;\n", "  return 2;\n"));

  // The root may be spelled differently as well.
  std::string shouted = text::to_lower(root);
  std::replace(shouted.begin(), shouted.end(), '/', '\\');
  ApplyOptions other = options;
  other.allowed_roots = {shouted + "\\"};
  Transaction through = prepare({search_replace(path, "  return x;\n", "  return 1;\n")}, other);
  CHECK(through.status == Status::Ok);
  CHECK(fs::is_within(through.files[0].path, root));
  CHECK_EQ(through.files[0].display_path, std::string("src/a.c"));
  CHECK(committed(through, other) == Status::Ok);
  CHECK_EQ(read_file(path), replaced(kSource, "  return x;\n", "  return 1;\n"));
  CHECK(reverted(through) == Status::Ok);
  // Case does not open a door either.
  std::string escaped = text::to_lower(root) + "/../OUTSIDE.C";
  CHECK(prepare({search_replace(escaped, "", "int x;\n")}, options).status == Status::OutsideRoots);
#endif
}

PRAG_TEST(s3_links_do_not_lead_out_of_the_root) {
  const std::string base = temp_dir("s3link");
  const std::string root = base + "/root";
  const std::string outside = base + "/outside";
  REQUIRE(write_file(root + "/a.c", kSource));
  REQUIRE(write_file(outside + "/secret.c", kSource));
  std::error_code ec;
  std::filesystem::create_directory_symlink(std::filesystem::path(outside), std::filesystem::path(root) / "link", ec);
  if (ec) {
    std::printf("  (skipped: this account may not create symbolic links: %s)\n", ec.message().c_str());
    return;
  }
  const ApplyOptions options = options_for(root);
  // Lexically inside the root, in fact outside of it.
  Transaction tx = prepare({search_replace("link/secret.c", "  return x;\n", "  return -x;\n")}, options);
  CHECK(tx.status == Status::OutsideRoots);
  CHECK(prepare({search_replace("link/new.c", "", "int x;\n")}, options).status == Status::OutsideRoots);
  CHECK(committed(tx, options) == Status::OutsideRoots);
  CHECK_EQ(read_file(outside + "/secret.c"), std::string(kSource));
  CHECK(!fs::exists(outside + "/new.c"));
  std::filesystem::remove(std::filesystem::path(root) / "link", ec);
}

// ---- S4: atomic transactions ---------------------------------------------------------------

namespace {

struct ThreeFiles {
  std::string root;
  ApplyOptions options;
  std::string a, b, c;                  // c does not exist before the transaction
  std::string a_before, b_before;       // with bytes that a text tool might damage
  std::string a_after, b_after, c_after;
  std::vector<EditBlock> blocks;
};

ThreeFiles three_files(const char* name) {
  ThreeFiles t;
  t.root = temp_dir(name);
  t.options = options_for(t.root);
  t.a = t.root + "/a.c";
  t.b = t.root + "/sub/b.c";
  t.c = t.root + "/sub/new/c.c";
  t.a_before = std::string(kSource) + "/* caf\xC3\xA9 \xFF\xFE */\n\tint tab;   \n";
  t.b_before = to_crlf(kSource) + "int last_line_without_newline;";
  REQUIRE(write_file(t.a, t.a_before));
  REQUIRE(write_file(t.b, t.b_before));
  t.blocks = {search_replace("a.c", "  return x;\n", "  return -x;\n"),
              search_replace("sub/b.c", "  return x + 1;\n", "  return x + 2;\n"),
              search_replace("sub/new/c.c", "", "int created;\n")};
  t.a_after = replaced(t.a_before, "  return x;\n", "  return -x;\n");
  t.b_after = replaced(t.b_before, "  return x + 1;\r\n", "  return x + 2;\r\n");
  t.c_after = "int created;\n";
  return t;
}

void check_untouched(const ThreeFiles& t) {
  CHECK_EQ(read_file(t.a), t.a_before);
  CHECK_EQ(read_file(t.b), t.b_before);
  CHECK(!fs::exists(t.c));
}

}  // namespace

PRAG_TEST(s4_a_transaction_over_several_files) {
  const ThreeFiles t = three_files("s4ok");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(tx.status == Status::Ok);
  REQUIRE(tx.files.size() == 3u);
  CHECK_EQ(tx.files[0].updated, t.a_after);
  CHECK_EQ(tx.files[1].updated, t.b_after);
  CHECK_EQ(tx.files[1].line_ending, std::string("\r\n"));
  CHECK_EQ(tx.files[2].updated, t.c_after);
  CHECK(tx.files[2].creates_file);
  check_untouched(t);

  CHECK(committed(tx, t.options) == Status::Ok);
  CHECK_EQ(read_file(t.a), t.a_after);
  CHECK_EQ(read_file(t.b), t.b_after);
  CHECK_EQ(read_file(t.c), t.c_after);
  CHECK(!fs::exists(tx.journal_path));
  CHECK_EQ(read_file(tx.files[0].backup_path), t.a_before);
  CHECK_EQ(read_file(tx.files[1].backup_path), t.b_before);
  CHECK(tx.files[2].backup_path.empty());
  // Nothing to recover after a commit.
  CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
  CHECK_EQ(read_file(t.a), t.a_after);

  // A second commit of the same proposal meets files that have changed.
  CHECK(committed(tx, t.options) == Status::Conflict);
  CHECK_EQ(read_file(t.a), t.a_after);

  CHECK(reverted(tx) == Status::Ok);
  check_untouched(t);
}

PRAG_TEST(s4_one_failing_file_leaves_every_file_untouched) {
  ThreeFiles t = three_files("s4fail");
  // The block for the second file does not match.
  t.blocks[1].search = "  return x + 100;\n";
  Transaction tx = prepare(t.blocks, t.options);
  CHECK(tx.status == Status::NotFound);
  REQUIRE(tx.files.size() == 3u);
  CHECK(tx.files[0].status == Status::Ok);
  CHECK(tx.files[1].status == Status::NotFound);
  CHECK(tx.files[2].status == Status::Ok);
  CHECK(tx.message.find("sub/b.c") != std::string::npos);
  CHECK(committed(tx, t.options) == Status::NotFound);
  check_untouched(t);
  CHECK(!fs::exists(t.options.backup_dir));
  CHECK(reverted(tx) == Status::NotFound);
  check_untouched(t);

  // The same when the status of the transaction has been tampered with.
  tx.status = Status::Ok;
  CHECK(committed(tx, t.options) == Status::NotFound);
  check_untouched(t);
}

PRAG_TEST(s4_a_failed_write_restores_the_files_replaced_before) {
  const ThreeFiles t = three_files("s4write");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(tx.status == Status::Ok);
  {
    const QuietLog quiet;
    const FaultGuard fault(detail::Fault::FailSecondWrite);
    CHECK(committed(tx, t.options) == Status::IoError);
  }
  CHECK(!tx.message.empty());
  check_untouched(t);
  CHECK(tx.journal_path.empty());
  CHECK(!fs::exists(t.options.backup_dir + "/" + detail::journal_name(tx.id)));
  CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
  // The proposal is still good.
  CHECK(committed(tx, t.options) == Status::Ok);
  CHECK_EQ(read_file(t.a), t.a_after);
  CHECK_EQ(read_file(t.b), t.b_after);
  CHECK_EQ(read_file(t.c), t.c_after);
}

#ifdef _WIN32
PRAG_TEST(s4_a_target_that_cannot_be_replaced) {
  // Windows refuses to replace a file that another handle holds open.
  const ThreeFiles t = three_files("s4lock");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(tx.status == Status::Ok);
  {
    const QuietLog quiet;
    std::ifstream held(std::filesystem::path(t.b), std::ios::binary);
    REQUIRE(held.good());
    CHECK(committed(tx, t.options) == Status::IoError);
    // The first file had been replaced and is restored.
    CHECK_EQ(read_file(t.a), t.a_before);
  }
  check_untouched(t);
  CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
  CHECK_EQ(edit_test::count_entries(t.root + "/sub"), std::size_t{1});  // no temporary file is left
  CHECK(committed(tx, t.options) == Status::Ok);
  CHECK_EQ(read_file(t.b), t.b_after);
}
#endif

PRAG_TEST(s4_a_crash_before_the_commit_point_is_rolled_back) {
  struct Point {
    detail::Fault fault;
    const char* name;
    bool a_replaced, b_replaced, c_created;
  };
  const Point points[] = {
      {detail::Fault::CrashAfterJournal, "s4crash1", false, false, false},
      {detail::Fault::CrashAfterFirstWrite, "s4crash2", true, false, false},
      {detail::Fault::CrashBeforeCommitPoint, "s4crash3", true, true, true},
  };
  for (const Point& point : points) {
    const ThreeFiles t = three_files(point.name);
    Transaction tx = prepare(t.blocks, t.options);
    REQUIRE(tx.status == Status::Ok);
    {
      const FaultGuard fault(point.fault);
      CHECK(committed(tx, t.options) != Status::Ok);
    }
    // What the crash left behind.
    REQUIRE(!tx.journal_path.empty());
    CHECK(fs::exists(tx.journal_path));
    CHECK_EQ(read_file(t.a), point.a_replaced ? t.a_after : t.a_before);
    CHECK_EQ(read_file(t.b), point.b_replaced ? t.b_after : t.b_before);
    CHECK_EQ(fs::exists(t.c), point.c_created);

    CHECK_EQ(recover(t.options.backup_dir), std::size_t{1});
    check_untouched(t);
    CHECK(!fs::exists(tx.journal_path));
    CHECK_EQ(edit_test::count_entries(t.options.backup_dir), std::size_t{0});
    // Recovery is idempotent.
    CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
    check_untouched(t);

    // After the recovery the edit can be made as if nothing had happened.
    Transaction again = prepare(t.blocks, t.options);
    REQUIRE(again.status == Status::Ok);
    CHECK_EQ(again.id, tx.id);
    CHECK(committed(again, t.options) == Status::Ok);
    CHECK_EQ(read_file(t.a), t.a_after);
    CHECK_EQ(read_file(t.b), t.b_after);
    CHECK_EQ(read_file(t.c), t.c_after);
    CHECK(reverted(again) == Status::Ok);
    check_untouched(t);
  }
}

PRAG_TEST(s4_recover_does_not_overwrite_what_it_does_not_know) {
  const QuietLog quiet;
  const ThreeFiles t = three_files("s4unknown");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(tx.status == Status::Ok);
  {
    const FaultGuard fault(detail::Fault::CrashBeforeCommitPoint);
    CHECK(committed(tx, t.options) != Status::Ok);
  }
  // Somebody edits a file between the crash and the recovery.
  REQUIRE(write_file(t.b, "edited by hand\n"));
  CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
  CHECK_EQ(read_file(t.b), std::string("edited by hand\n"));
  CHECK_EQ(read_file(t.a), t.a_after);  // nothing was done by halves
  CHECK(!fs::exists(tx.journal_path));
  CHECK(fs::exists(tx.journal_path + ".unresolved"));
  // The backups are kept for whoever sorts it out.
  CHECK_EQ(read_file(t.options.backup_dir + "/" + tx.id + "/0.bak"), t.a_before);
  CHECK_EQ(recover(t.options.backup_dir), std::size_t{0});
}

PRAG_TEST(s4_recover_ignores_what_is_not_a_journal) {
  const QuietLog quiet;
  const std::string root = temp_dir("s4junk");
  const std::string target = root + "/a.c";
  REQUIRE(write_file(target, kSource));
  const std::string dir = root + "/.edits";
  CHECK_EQ(recover(dir), std::size_t{0});  // the directory does not exist
  REQUIRE(write_file(dir + "/tx-0000000000000001.journal", "not json"));
  REQUIRE(write_file(dir + "/tx-0000000000000002.journal", "{\"format\": 1, \"id\": \"tx-0000000000000002\"}"));
  REQUIRE(write_file(dir + "/tx-0000000000000003.journal",
                     "{\"format\": 1, \"id\": \"tx-../../../x\", \"files\": [{\"path\": \"" + target +
                         "\", \"backup\": \"\", \"created\": true, \"original_hash\": \"0\", \"updated_hash\": \"0\"}]}"));
  REQUIRE(write_file(dir + "/tx-0000000000000004.journal",
                     "{\"format\": 1, \"id\": \"tx-0000000000000004\", \"files\": [{\"path\": 5}]}"));
  REQUIRE(write_file(dir + "/notes.txt", "{\"format\": 1}"));
  // A backup outside of the journal directory is not read.
  REQUIRE(write_file(root + "/elsewhere.bak", "int replaced;\n"));
  char hashes[128];
  std::snprintf(hashes, sizeof(hashes), "\"original_hash\": \"%016llx\", \"updated_hash\": \"%016llx\"",
                static_cast<unsigned long long>(hash64("int replaced;\n")),
                static_cast<unsigned long long>(hash64(kSource)));
  REQUIRE(write_file(dir + "/tx-0000000000000005.journal",
                     "{\"format\": 1, \"id\": \"tx-0000000000000005\", \"files\": [{\"path\": \"" + target +
                         "\", \"backup\": \"" + root + "/elsewhere.bak\", \"created\": false, " + hashes + "}]}"));
  CHECK_EQ(recover(dir), std::size_t{0});
  CHECK_EQ(read_file(target), std::string(kSource));
  CHECK(fs::exists(dir + "/notes.txt"));
}

PRAG_TEST(s4_a_file_that_changed_after_prepare_is_a_conflict) {
  const ThreeFiles t = three_files("s4conflict");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(tx.status == Status::Ok);
  // The second file changes on disk, in a place the edit does not touch.
  const std::string changed = replaced(t.b_before, "static int counter = 0;", "static int counter = 1;");
  REQUIRE(write_file(t.b, changed));
  CHECK(committed(tx, t.options) == Status::Conflict);
  CHECK(tx.message.find("b.c") != std::string::npos);
  CHECK_EQ(read_file(t.a), t.a_before);
  CHECK_EQ(read_file(t.b), changed);
  CHECK(!fs::exists(t.c));
  CHECK(!fs::exists(t.options.backup_dir));

  // A file that is to be created, but exists by now.
  REQUIRE(write_file(t.b, t.b_before));
  REQUIRE(write_file(t.c, "int somebody_else;\n"));
  CHECK(committed(tx, t.options) == Status::Conflict);
  CHECK_EQ(read_file(t.c), std::string("int somebody_else;\n"));
  CHECK_EQ(read_file(t.a), t.a_before);
  // A file that is to be edited, but is gone.
  REQUIRE(fs::remove_file(t.c));
  REQUIRE(fs::remove_file(t.a));
  CHECK(committed(tx, t.options) == Status::Conflict);
  CHECK(!fs::exists(t.a));
  CHECK_EQ(read_file(t.b), t.b_before);
  CHECK(!fs::exists(t.options.backup_dir));

  // With the snapshot back on disk the proposal is good again.
  REQUIRE(write_file(t.a, t.a_before));
  CHECK(committed(tx, t.options) == Status::Ok);
  CHECK_EQ(read_file(t.b), t.b_after);
}

PRAG_TEST(s4_revert_restores_the_exact_bytes) {
  const ThreeFiles t = three_files("s4revert");
  Transaction tx = prepare(t.blocks, t.options);
  REQUIRE(committed(tx, t.options) == Status::Ok);
  CHECK(reverted(tx) == Status::Ok);
  check_untouched(t);
  CHECK_EQ(fs::file_size(t.a), static_cast<std::uint64_t>(t.a_before.size()));
  CHECK_EQ(fs::file_size(t.b), static_cast<std::uint64_t>(t.b_before.size()));

  // A file that was edited after the commit is not overwritten, and then no
  // file of the transaction is.
  Transaction second = prepare(t.blocks, t.options);
  REQUIRE(committed(second, t.options) == Status::Ok);
  REQUIRE(write_file(t.b, t.b_after + "\r\n// more work\r\n"));
  CHECK(reverted(second) == Status::Conflict);
  CHECK_EQ(read_file(t.a), t.a_after);
  CHECK_EQ(read_file(t.b), t.b_after + "\r\n// more work\r\n");
  CHECK_EQ(read_file(t.c), t.c_after);

  // A damaged backup is not used; the snapshot in memory is.
  REQUIRE(write_file(t.b, t.b_after));
  REQUIRE(write_file(second.files[0].backup_path, "garbage"));
  CHECK(reverted(second) == Status::Ok);
  check_untouched(t);
}

// ---- S5: byte preservation ------------------------------------------------------------------

namespace {

// Prepares and commits one block against a file with the given content and
// returns what is on disk afterwards.
std::string edit_file(const char* name, const std::string& content, const std::string& search,
                      const std::string& replace, Status expected = Status::Ok, MatchKind kind = MatchKind::Exact) {
  const std::string root = temp_dir(name);
  const std::string path = root + "/f.c";
  REQUIRE(write_file(path, content));
  const ApplyOptions options = options_for(root);
  Transaction tx = prepare({search_replace("f.c", search, replace)}, options);
  if (tx.status != expected) std::fprintf(stderr, "  %s: %s %s\n", name, to_string(tx.status), tx.message.c_str());
  CHECK(tx.status == expected);
  REQUIRE(tx.files.size() == 1u);
  if (expected == Status::Ok) CHECK(tx.files[0].matches[0].kind == kind);
  CHECK(committed(tx, options) == expected);
  const std::string on_disk = read_file(path);
  if (expected == Status::Ok) {
    CHECK_EQ(on_disk, tx.files[0].updated);
    CHECK(reverted(tx) == Status::Ok);
    CHECK_EQ(read_file(path), content);
  } else {
    CHECK_EQ(on_disk, content);
  }
  return on_disk;
}

}  // namespace

PRAG_TEST(s5_line_endings_are_preserved) {
  const std::string lf = "one\ntwo\nthree\nfour\n";
  CHECK_EQ(edit_file("s5lf", lf, "two\nthree\n", "2\n2.5\n3\n"), std::string("one\n2\n2.5\n3\nfour\n"));
  // The same block against the same file with CRLF endings.
  const std::string crlf = "one\r\ntwo\r\nthree\r\nfour\r\n";
  CHECK_EQ(edit_file("s5crlf", crlf, "two\nthree\n", "2\n2.5\n3\n"), std::string("one\r\n2\r\n2.5\r\n3\r\nfour\r\n"));
  // The block itself written with CRLF, against both.
  CHECK_EQ(edit_file("s5crlf2", crlf, "two\r\nthree\r\n", "2\r\n3\r\n"), std::string("one\r\n2\r\n3\r\nfour\r\n"));
  CHECK_EQ(edit_file("s5lf2", lf, "two\r\nthree\r\n", "2\r\n3\r\n"), std::string("one\n2\n3\nfour\n"));
  // Deleting and emptying lines.
  CHECK_EQ(edit_file("s5del", crlf, "two\n", ""), std::string("one\r\nthree\r\nfour\r\n"));
  CHECK_EQ(edit_file("s5empty", crlf, "two\n", "\n"), std::string("one\r\n\r\nthree\r\nfour\r\n"));

  // Mixed endings: lines that are not replaced keep their own; new lines
  // take the ending that is more frequent in the file, the last new line the
  // ending of the last line it replaces.
  const std::string mixed = "one\r\ntwo\nthree\r\nfour\r\n";
  CHECK_EQ(edit_file("s5mixed", mixed, "two\n", "2\n2.5\n"), std::string("one\r\n2\r\n2.5\nthree\r\nfour\r\n"));
  CHECK_EQ(edit_file("s5mixed2", mixed, "three\n", "3\n"), std::string("one\r\ntwo\n3\r\nfour\r\n"));
}

PRAG_TEST(s5_a_missing_final_newline_is_preserved) {
  const std::string text = "one\ntwo\nthree";
  CHECK_EQ(edit_file("s5nl1", text, "three\n", "3\n"), std::string("one\ntwo\n3"));
  CHECK_EQ(edit_file("s5nl2", text, "three\n", "3\n4\n"), std::string("one\ntwo\n3\n4"));
  CHECK_EQ(edit_file("s5nl3", text, "two\n", "2\n"), std::string("one\n2\nthree"));
  CHECK_EQ(edit_file("s5nl4", text, "two\nthree\n", "2\n"), std::string("one\n2"));
  // Removing the last line does not give the file a final newline.
  CHECK_EQ(edit_file("s5nl5", text, "three\n", ""), std::string("one\ntwo"));
  CHECK_EQ(edit_file("s5nl6", "one\r\ntwo\r\nthree", "two\nthree\n", ""), std::string("one"));
  CHECK_EQ(edit_file("s5nl7", text, "one\ntwo\nthree\n", ""), std::string(""));
  CHECK_EQ(edit_file("s5nl8", "only", "only", "other"), std::string("other"));
  // And a final newline that exists is kept.
  CHECK_EQ(edit_file("s5nl9", "one\ntwo\nthree\n", "three", "3"), std::string("one\ntwo\n3\n"));
  CHECK_EQ(edit_file("s5nl10", "one\ntwo\nthree\n", "three\n", ""), std::string("one\ntwo\n"));
}

PRAG_TEST(s5_bytes_outside_the_replaced_range_are_preserved) {
  // Tabs, trailing blanks, UTF-8, bytes that are not valid UTF-8, a form feed,
  // a lone carriage return and a byte order mark.
  const std::string before = "\xEF\xBB\xBF// caf\xC3\xA9 \xE2\x82\xAC \xFF\xFE\x80\n\tint tab;\t \n\f\nint a\rb;\n";
  const std::string after = "\n\t\tchar* s = \"\xC3\xBC\xF0\x9F\x98\x80\";   \n\x01\x7F last \xC0\xAF";
  const std::string target = "int value = 1;\n";
  for (const char* ending : {"\n", "\r\n"}) {
    std::string file = before + target + after;
    std::string expected = before + "int value = 2;\n" + after;
    if (ending[0] == '\r') {
      // Only the terminator of the target line; every other byte as it is.
      file = before + "int value = 1;\r\n" + after;
      expected = before + "int value = 2;\r\n" + after;
    }
    CHECK_EQ(edit_file("s5bytes", file, target, "int value = 2;\n"), expected);
  }
  // The replacement may carry any bytes as well.
  CHECK_EQ(edit_file("s5bytes2", before + target + after, target, "int value = 3; // \xC3\xA9\xFF\t \n"),
           before + "int value = 3; // \xC3\xA9\xFF\t \n" + after);
  // The first line of a file with a byte order mark.
  CHECK_EQ(edit_file("s5bom", "\xEF\xBB\xBFint a;\nint b;\n", "int a;\n", "long a;\n"),
           std::string("\xEF\xBB\xBFlong a;\nint b;\n"));

  // Trailing blanks of the lines that are replaced go with them, those of
  // other lines stay.
  CHECK_EQ(edit_file("s5trail", "a  \nb\t\nc \n", "b\n", "B\n", Status::Ok, MatchKind::TrailingSpace),
           std::string("a  \nB\nc \n"));
  // A file with a zero byte is not text.
  edit_file("s5nul", std::string("int a;\n\0int b;\n", 15), "int a;\n", "long a;\n", Status::IoError);
}

PRAG_TEST(s5_the_indentation_rung_changes_the_replacement_only) {
  const std::string file = "void f(void) {\n\tif (a) {\n\t\tb();\n\t}\n  odd();   \n}\n";
  // The model left out one level of indentation.
  CHECK_EQ(edit_file("s5indent", file, "if (a) {\n\tb();\n}\n", "if (a && c) {\n\tb();\n\td();\n}\n", Status::Ok,
                     MatchKind::Indentation),
           std::string("void f(void) {\n\tif (a && c) {\n\t\tb();\n\t\td();\n\t}\n  odd();   \n}\n"));
  // Spaces for tabs: lines whose indentation is known are translated.
  CHECK_EQ(edit_file("s5indent2", file, "    if (a) {\n        b();\n    }\n",
                     "    if (a) {\n        b2();\n    }\n", Status::Ok, MatchKind::Indentation),
           std::string("void f(void) {\n\tif (a) {\n\t\tb2();\n\t}\n  odd();   \n}\n"));
  // Indentation that cannot be translated: nothing is written.
  const std::string root = temp_dir("s5indent3");
  REQUIRE(write_file(root + "/f.c", file));
  const Transaction tx = prepare({search_replace("f.c", "    if (a) {\n        b();\n    }\n",
                                                 "    if (a) {\n            deeper();\n    }\n")},
                                 options_for(root));
  CHECK(tx.status == Status::NotFound);
  REQUIRE(tx.files.size() == 1u);
  CHECK(tx.files[0].matches[0].kind == MatchKind::Indentation);
  CHECK_EQ(tx.files[0].matches[0].first_line, 2u);
  CHECK(tx.files[0].message.find("indentation") != std::string::npos);
  CHECK_EQ(tx.files[0].updated, file);
}

PRAG_TEST(s5_the_fuzzy_rung_end_to_end) {
  const std::string root = temp_dir("s5fuzzy");
  REQUIRE(write_file(root + "/f.c", kSource));
  ApplyOptions options = options_for(root);
  // "x > 8" for "x > 9": no rung matches unless the fuzzy one is enabled.
  const std::vector<EditBlock> blocks = {search_replace(
      "f.c", "int second(int x) {\n  if (x > 8) {\n    return 0;\n  }\n",
      "int second(int x) {\n  if (x > 8) {\n    return 1;\n  }\n")};
  CHECK(prepare(blocks, options).status == Status::NotFound);
  options.match.allow_fuzzy = true;
  Transaction tx = prepare(blocks, options);
  CHECK(tx.status == Status::Ok);
  CHECK(tx.files[0].matches[0].kind == MatchKind::Fuzzy);
  CHECK_EQ(tx.files[0].matches[0].distance, std::size_t{1});
  CHECK_EQ(tx.files[0].updated,
           replaced(kSource, "  if (x > 9) {\n    return 0;\n", "  if (x > 8) {\n    return 1;\n"));
}

// ---- S6: overlap ------------------------------------------------------------------------------

PRAG_TEST(s6_overlapping_blocks_are_refused) {
  const std::string root = temp_dir("s6");
  const std::string path = root + "/a.c";
  REQUIRE(write_file(path, kSource));
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;

  const std::vector<std::vector<EditBlock>> overlapping = {
      // The same lines twice.
      {search_replace("a.c", "  return x;\n", "  return 1;\n"), search_replace("a.c", "  return x;\n", "  return 2;\n")},
      // Lines 5-6 and 6-7.
      {search_replace("a.c", "    return 0;\n  }\n  return x;\n", "  }\n"),
       search_replace("a.c", "  }\n  return x;\n}\n", "}\n")},
      // One block inside the other.
      {search_replace("a.c", "int second(int x) {\n  if (x > 9) {\n    return 0;\n  }\n  return x + 1;\n}\n", "\n"),
       search_replace("a.c", "  if (x > 9) {\n", "  if (x > 10) {\n")},
      // A unit and a line of that unit.
      {unit_rewrite("a.c", "second", "int second(int x) {\n  return x;\n}\n"),
       search_replace("a.c", "  return x + 1;\n", "  return x + 2;\n")},
      // The same unit twice.
      {unit_rewrite("a.c", "first", "int first(int x) {\n  return 1;\n}\n"),
       unit_rewrite("a.c", "first", "int first(int x) {\n  return 2;\n}\n")},
  };
  for (const std::vector<EditBlock>& blocks : overlapping) {
    for (const bool reversed : {false, true}) {
      std::vector<EditBlock> ordered = blocks;
      if (reversed) std::reverse(ordered.begin(), ordered.end());
      Transaction tx = prepare(ordered, options);
      CHECK(tx.status == Status::Overlap);
      REQUIRE(tx.files.size() == 1u);
      CHECK_EQ(tx.files[0].matches.size(), std::size_t{2});
      CHECK_EQ(tx.files[0].updated, std::string(kSource));
      CHECK(!tx.files[0].message.empty());
      CHECK(committed(tx, options) == Status::Overlap);
    }
  }
  CHECK_EQ(read_file(path), std::string(kSource));
  CHECK(!fs::exists(options.backup_dir));
}

PRAG_TEST(s6_the_order_of_the_blocks_does_not_matter) {
  const std::string root = temp_dir("s6order");
  const std::string path = root + "/a.c";
  REQUIRE(write_file(path, kSource));
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;

  // Four blocks; the second and the third touch neighbouring lines, the
  // first changes the number of lines, the last rewrites a unit.
  std::vector<EditBlock> blocks = {
      search_replace("a.c", "static int counter = 0;\n", "static int counter = 0;\nstatic int limit = 9;\n\n"),
      search_replace("a.c", "  if (x < 0) {\n", "  if (x <= 0) {\n"),
      search_replace("a.c", "    return 0;\n  }\n  return x;\n", "    return -1;\n  }\n"),
      unit_rewrite("a.c", "second", "int second(int x) {\n  return x > limit ? 0 : x + 1;\n}\n"),
  };
  const std::string expected =
      "static int counter = 0;\n"
      "static int limit = 9;\n"
      "\n"
      "\n"
      "int first(int x) {\n"
      "  if (x <= 0) {\n"
      "    return -1;\n"
      "  }\n"
      "}\n"
      "\n"
      "int second(int x) {\n"
      "  return x > limit ? 0 : x + 1;\n"
      "}\n";
  std::vector<std::size_t> order = {0, 1, 2, 3};
  std::size_t permutations = 0;
  std::string id;
  do {
    std::vector<EditBlock> ordered;
    for (const std::size_t i : order) ordered.push_back(blocks[i]);
    const Transaction tx = prepare(ordered, options);
    CHECK(tx.status == Status::Ok);
    REQUIRE(tx.files.size() == 1u);
    CHECK_EQ(tx.files[0].updated, expected);
    // The matches are reported in the order of the blocks.
    REQUIRE(tx.files[0].matches.size() == 4u);
    for (std::size_t k = 0; k < order.size(); ++k) {
      const std::uint32_t first_line[] = {1, 4, 5, 10};
      const MatchKind kind = order[k] == 3 ? MatchKind::Unit : MatchKind::Exact;
      CHECK_EQ(tx.files[0].matches[k].first_line, first_line[order[k]]);
      CHECK(tx.files[0].matches[k].kind == kind);
    }
    if (id.empty()) id = tx.id;
    CHECK_EQ(tx.id, id);
    ++permutations;
  } while (std::next_permutation(order.begin(), order.end()));
  CHECK_EQ(permutations, std::size_t{24});

  Transaction tx = prepare(blocks, options);
  CHECK(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(path), expected);
}

// ---- unit rewrite -------------------------------------------------------------------------------

PRAG_TEST(unit_rewrite_replaces_the_unit) {
  const std::string root = temp_dir("unit");
  const std::string path = root + "/a.c";
  REQUIRE(write_file(path, kSource));
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;

  // The unit "second" begins with the blank line before it. The blank line
  // stays, whether or not the model surrounds its text with blank lines.
  const std::string expected =
      replaced(kSource, "int second(int x) {\n  if (x > 9) {\n    return 0;\n  }\n  return x + 1;\n}\n",
               "int second(int x) {\n  return (x + 1) % 10;\n}\n");
  for (const char* text : {"int second(int x) {\n  return (x + 1) % 10;\n}\n",
                           "\n\nint second(int x) {\n  return (x + 1) % 10;\n}\n\n", "int second(int x) {\r\n  return (x + 1) % 10;\r\n}"}) {
    Transaction tx = prepare({unit_rewrite("a.c", "second", text)}, options);
    CHECK(tx.status == Status::Ok);
    REQUIRE(tx.files.size() == 1u);
    CHECK_EQ(tx.files[0].updated, expected);
    REQUIRE(tx.files[0].matches.size() == 1u);
    const Match& m = tx.files[0].matches[0];
    CHECK(m.kind == MatchKind::Unit);
    CHECK_EQ(m.first_line, 10u);
    CHECK_EQ(m.last_line, 15u);
    CHECK_EQ(m.begin, std::string(kSource).find("int second"));
    CHECK_EQ(m.end, std::string(kSource).size());
    CHECK_EQ(m.candidates, std::size_t{1});
  }
  Transaction tx = prepare({unit_rewrite("a.c", "first", "int first(int x) {\n  return x < 0 ? 0 : x;\n}\n")}, options);
  REQUIRE(tx.status == Status::Ok);
  CHECK(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(path),
           std::string("int first(int x) {\n  return x < 0 ? 0 : x;\n}\n\nint second(int x) {\n  if (x > 9) {\n"
                       "    return 0;\n  }\n  return x + 1;\n}\n"));
  CHECK(reverted(tx) == Status::Ok);
  CHECK_EQ(read_file(path), std::string(kSource));

  // Failures.
  CHECK(prepare({unit_rewrite("a.c", "third", "int third;\n")}, options).status == Status::ScopeNotFound);
  CHECK(prepare({unit_rewrite("a.c", "", "int third;\n")}, options).status == Status::ScopeNotFound);
  CHECK(prepare({unit_rewrite("a.c", "second", "")}, options).status == Status::NotFound);
  CHECK(prepare({unit_rewrite("a.c", "second", "\n  \n")}, options).status == Status::NotFound);
  CHECK(prepare({unit_rewrite("missing.c", "second", "int second;\n")}, options).status == Status::FileMissing);
  ApplyOptions no_locator = options_for(root);
  CHECK(prepare({unit_rewrite("a.c", "second", "int second;\n")}, no_locator).status == Status::ScopeNotFound);
}

PRAG_TEST(unit_rewrite_keeps_line_endings_and_indentation) {
  const std::string root = temp_dir("unit2");
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;

  // CRLF file whose last unit has no final newline.
  const std::string crlf = "int first(int x) {\r\n  return x;\r\n}\r\n\r\nint second(int x) {\r\n  return x + 1;\r\n}";
  REQUIRE(write_file(root + "/crlf.c", crlf));
  const Transaction last = prepare({unit_rewrite("crlf.c", "second", "int second(int x) {\n  return 2;\n}\n")}, options);
  CHECK(last.status == Status::Ok);
  CHECK_EQ(last.files[0].updated,
           std::string("int first(int x) {\r\n  return x;\r\n}\r\n\r\nint second(int x) {\r\n  return 2;\r\n}"));
  const Transaction head = prepare({unit_rewrite("crlf.c", "first", "int first(int x) {\n  return 1;\n}\n")}, options);
  CHECK(head.status == Status::Ok);
  CHECK_EQ(head.files[0].updated,
           std::string("int first(int x) {\r\n  return 1;\r\n}\r\n\r\nint second(int x) {\r\n  return x + 1;\r\n}"));

  // A unit that is nested in the file, written by the model from the first column.
  const std::string nested =
      "class Heap:\n"
      "    def size(self):\n"
      "        return self.n\n"
      "\n"
      "    def grow(self, k):\n"
      "        self.n += k\n";
  REQUIRE(write_file(root + "/heap.py", nested));
  options.unit_locator = [&](std::string_view, std::string_view text, std::string_view name, std::size_t* begin,
                             std::size_t* end) {
    if (name != "size") return false;
    *begin = text.find("    def size");
    *end = text.find("\n    def grow");  // the blank line belongs to the next unit
    return true;
  };
  const Transaction method =
      prepare({unit_rewrite("heap.py", "size", "def size(self):\n    if self.n < 0:\n        return 0\n    return self.n\n")},
              options);
  CHECK(method.status == Status::Ok);
  CHECK_EQ(method.files[0].updated, std::string("class Heap:\n"
                                                "    def size(self):\n"
                                                "        if self.n < 0:\n"
                                                "            return 0\n"
                                                "        return self.n\n"
                                                "\n"
                                                "    def grow(self, k):\n"
                                                "        self.n += k\n"));
  CHECK_EQ(method.files[0].matches[0].indent_from, std::string(""));
  CHECK_EQ(method.files[0].matches[0].indent_to, std::string("    "));
  // Tabs in the new text of a unit that is indented with spaces: refused.
  const Transaction tabs = prepare({unit_rewrite("heap.py", "size", "def size(self):\n\treturn self.n\n")}, options);
  CHECK(tabs.status == Status::NotFound);
  CHECK(!tabs.files[0].message.empty());
}

// ---- syntax hook ----------------------------------------------------------------------------------

PRAG_TEST(edits_that_add_syntax_errors_are_refused) {
  const std::string root = temp_dir("syntax");
  const std::string path = root + "/a.c";
  REQUIRE(write_file(path, kSource));
  ApplyOptions options = options_for(root);
  std::vector<std::string> seen;
  options.syntax_check = [&](std::string_view p, std::string_view text) {
    seen.emplace_back(p);
    return count_errors(p, text);
  };

  // The closing brace of the condition is lost.
  Transaction bad = prepare({search_replace("a.c", "    return 0;\n  }\n  return x;\n", "    return 0;\n  return x;\n")}, options);
  CHECK(bad.status == Status::SyntaxRegression);
  REQUIRE(bad.files.size() == 1u);
  CHECK_EQ(bad.files[0].syntax_errors_before, std::size_t{0});
  CHECK_EQ(bad.files[0].syntax_errors_after, std::size_t{1});
  CHECK(!bad.files[0].message.empty());
  CHECK(!bad.files[0].diff.empty());  // what was refused can still be shown
  CHECK_EQ(bad.files[0].updated, std::string(kSource));
  CHECK(committed(bad, options) == Status::SyntaxRegression);
  CHECK_EQ(read_file(path), std::string(kSource));
  REQUIRE(seen.size() == 2u);
  CHECK_EQ(seen[0], path);  // the hook receives the resolved path

  Transaction good = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n")}, options);
  CHECK(good.status == Status::Ok);
  CHECK_EQ(good.files[0].syntax_errors_before, std::size_t{0});
  CHECK_EQ(good.files[0].syntax_errors_after, std::size_t{0});

  // Counts are compared, zero is not required: a file that has errors may be
  // edited as long as it does not get more of them.
  const std::string broken = std::string(kSource) + "int unfinished(void) {\n";
  REQUIRE(write_file(root + "/b.c", broken));
  const Transaction same = prepare({search_replace("b.c", "  return x;\n", "  return -x;\n")}, options);
  CHECK(same.status == Status::Ok);
  CHECK_EQ(same.files[0].syntax_errors_before, std::size_t{1});
  CHECK_EQ(same.files[0].syntax_errors_after, std::size_t{1});
  const Transaction better = prepare({search_replace("b.c", "int unfinished(void) {\n", "int unfinished(void) {\n}\n")}, options);
  CHECK(better.status == Status::Ok);
  CHECK_EQ(better.files[0].syntax_errors_after, std::size_t{0});
  const Transaction worse = prepare({search_replace("b.c", "  return x;\n", "  {{ return x;\n")}, options);
  CHECK(worse.status == Status::SyntaxRegression);
  CHECK_EQ(worse.files[0].syntax_errors_before, std::size_t{1});
  CHECK_EQ(worse.files[0].syntax_errors_after, std::size_t{3});

  // New files are checked against an empty file.
  CHECK(prepare({search_replace("new.c", "", "int f(void) {\n")}, options).status == Status::SyntaxRegression);
  CHECK(prepare({search_replace("new.c", "", "int f(void) {\n}\n")}, options).status == Status::Ok);
  // One bad file stops the transaction.
  Transaction both = prepare({search_replace("a.c", "  return x;\n", "  return -x;\n"),
                              search_replace("b.c", "  return x;\n", "  { return x;\n")},
                             options);
  CHECK(both.status == Status::SyntaxRegression);
  CHECK(committed(both, options) == Status::SyntaxRegression);
  CHECK_EQ(read_file(path), std::string(kSource));
  CHECK_EQ(read_file(root + "/b.c"), broken);
}

// ---- new files --------------------------------------------------------------------------------------

PRAG_TEST(an_empty_search_section_creates_a_file) {
  const std::string root = temp_dir("create");
  const std::string path = root + "/include/new/header.h";
  const ApplyOptions options = options_for(root);

  Transaction tx = prepare({search_replace("include/new/header.h", "", "#pragma once\n\nint answer(void);\n")}, options);
  CHECK(tx.status == Status::Ok);
  REQUIRE(tx.files.size() == 1u);
  const FileProposal& file = tx.files[0];
  CHECK(file.creates_file);
  CHECK_EQ(file.path, path);
  CHECK_EQ(file.display_path, std::string("include/new/header.h"));
  CHECK_EQ(file.original, std::string(""));
  CHECK_EQ(file.updated, std::string("#pragma once\n\nint answer(void);\n"));
  CHECK_EQ(file.diff, std::string("--- /dev/null\n"
                                  "+++ b/include/new/header.h\n"
                                  "@@ -0,0 +1,3 @@\n"
                                  "+#pragma once\n"
                                  "+\n"
                                  "+int answer(void);\n"));
  REQUIRE(file.matches.size() == 1u);
  CHECK(file.matches[0].status == Status::Ok);
  CHECK(file.matches[0].kind == MatchKind::None);
  CHECK(!fs::exists(root + "/include"));

  CHECK(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(path), std::string("#pragma once\n\nint answer(void);\n"));
  CHECK(tx.files[0].backup_path.empty());
  CHECK(reverted(tx) == Status::Ok);
  CHECK(!fs::exists(path));
  CHECK(reverted(tx) == Status::Ok);

  // Anything else on a path that does not exist.
  CHECK(prepare({search_replace("missing.c", "int a;\n", "int b;\n")}, options).status == Status::FileMissing);
  const Transaction twice = prepare({search_replace("n.c", "", "int a;\n"), search_replace("n.c", "", "int b;\n")}, options);
  CHECK(twice.status == Status::Overlap);
  CHECK_EQ(twice.files.size(), std::size_t{1});
  CHECK(prepare({search_replace("n.c", "", "int a;\n"), search_replace("n.c", "int a;\n", "int b;\n")}, options).status ==
        Status::FileMissing);

  // An existing file is never overwritten by an empty SEARCH section, unless it is empty.
  REQUIRE(write_file(root + "/full.c", "int a;\n"));
  REQUIRE(write_file(root + "/empty.c", ""));
  const Transaction full = prepare({search_replace("full.c", "", "int b;\n")}, options);
  CHECK(full.status == Status::NotFound);
  CHECK_EQ(full.files[0].updated, std::string("int a;\n"));
  Transaction empty = prepare({search_replace("empty.c", "", "int b;\n")}, options);
  CHECK(empty.status == Status::Ok);
  CHECK(!empty.files[0].creates_file);
  CHECK_EQ(empty.files[0].updated, std::string("int b;\n"));
  CHECK(committed(empty, options) == Status::Ok);
  CHECK_EQ(read_file(root + "/empty.c"), std::string("int b;\n"));
  CHECK(reverted(empty) == Status::Ok);
  CHECK_EQ(read_file(root + "/empty.c"), std::string(""));
  CHECK(fs::exists(root + "/empty.c"));
}

// ---- from model output to disk ------------------------------------------------------------------------

PRAG_TEST(model_output_is_applied_end_to_end) {
  const std::string root = temp_dir("endtoend");
  REQUIRE(write_file(root + "/src/a.c", to_crlf(kSource)));
  ApplyOptions options = options_for(root);
  options.unit_locator = locate_unit;
  options.syntax_check = count_errors;

  const std::string answer =
      "The guard should return -1, and the second function can be one expression.\n"
      "\n"
      "```c\n"
      "src/a.c\n"
      "@@ first\n"
      "<<<<<<< SEARCH\n"
      "  if (x < 0) {\n"
      "    return 0;\n"
      "=======\n"
      "  if (x < 0) {\n"
      "    return -1;\n"
      ">>>>>>> REPLACE\n"
      "```\n"
      "\n"
      "```c\n"
      "src/a.c\n"
      "<<<<<<< UNIT second\n"
      "int second(int x) {\n"
      "  return x > 9 ? 0 : x + 1;\n"
      "}\n"
      ">>>>>>> UNIT\n"
      "```\n"
      "\n"
      "src/notes.txt\n"
      "<<<<<<< SEARCH\n"
      "=======\n"
      "first() returns -1 for negative input.\n"
      ">>>>>>> REPLACE\n";
  const ParseResult parsed = parse_edit_blocks(answer);
  CHECK(parsed.warnings.empty());
  REQUIRE(parsed.blocks.size() == 3u);
  Transaction tx = prepare(parsed.blocks, options);
  CHECK(tx.status == Status::Ok);
  REQUIRE(tx.files.size() == 2u);
  CHECK_EQ(tx.files[0].matches.size(), std::size_t{2});
  REQUIRE(committed(tx, options) == Status::Ok);
  CHECK_EQ(read_file(root + "/src/a.c"), to_crlf("static int counter = 0;\n"
                                                 "\n"
                                                 "int first(int x) {\n"
                                                 "  if (x < 0) {\n"
                                                 "    return -1;\n"
                                                 "  }\n"
                                                 "  return x;\n"
                                                 "}\n"
                                                 "\n"
                                                 "int second(int x) {\n"
                                                 "  return x > 9 ? 0 : x + 1;\n"
                                                 "}\n"));
  CHECK_EQ(read_file(root + "/src/notes.txt"), std::string("first() returns -1 for negative input.\n"));
  CHECK(reverted(tx) == Status::Ok);
  CHECK_EQ(read_file(root + "/src/a.c"), to_crlf(kSource));
  CHECK(!fs::exists(root + "/src/notes.txt"));
}
