// The parse module against the evaluation corpus: the mimalloc sources under
// PRAG_DEMO_REPO. Every test is skipped when the corpus is not there.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "checks.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/graph/call_graph.hpp"
#include "prag/parse/parser.hpp"

#ifdef PRAG_PARSE_TEST_WITH_EXEC
#include "prag/exec/block.hpp"
#include "prag/exec/ordered_executor.hpp"
#endif

using namespace prag;
using namespace prag::parse;
using parse_checks::check_partition;

namespace {

struct Corpus {
  bool present = false;
  std::string root;                // normalised, without trailing slash
  std::vector<std::string> paths;  // sorted: include/ before src/
  std::vector<std::string> texts;
  std::vector<ParsedFile> files;   // parsed one after the other
  double parse_seconds = 0.0;
  std::size_t bytes = 0;

  std::string relative(std::size_t i) const { return paths[i].substr(root.size() + 1); }
};

const Corpus& corpus() {
  static const Corpus instance = [] {
    Corpus c;
    c.root = fs::normalize(PRAG_DEMO_REPO);
    while (!c.root.empty() && c.root.back() == '/') c.root.pop_back();
    if (!fs::is_directory(c.root + "/src") || !fs::is_directory(c.root + "/include")) return c;
    fs::WalkOptions options;
    options.extensions = {".c", ".h"};
    for (const char* directory : {"/include", "/src"}) {
      for (std::string& path : fs::walk(c.root + directory, options)) c.paths.push_back(std::move(path));
    }
    for (const std::string& path : c.paths) {
      c.texts.push_back(fs::read_file(path).value_or(std::string()));
      c.bytes += c.texts.back().size();
    }
    c.present = !c.paths.empty();
    const Stopwatch watch;
    for (std::size_t i = 0; i < c.paths.size(); ++i) c.files.push_back(parse_source(c.paths[i], c.texts[i]));
    c.parse_seconds = watch.elapsed_s();
    return c;
  }();
  return instance;
}

bool skip(const char* test) {
  if (corpus().present) return false;
  std::printf("  skipped %s: no corpus at %s\n", test, PRAG_DEMO_REPO);
  return true;
}

std::size_t index_of(const Corpus& c, std::string_view relative) {
  for (std::size_t i = 0; i < c.paths.size(); ++i) {
    if (c.relative(i) == relative) return i;
  }
  return c.paths.size();
}

std::size_t count_kind(const ParsedFile& file, UnitKind kind) {
  return static_cast<std::size_t>(
      std::count_if(file.units.begin(), file.units.end(), [kind](const SourceUnit& u) { return u.kind == kind; }));
}

// The units of the corpus with dense ids, as the index numbers them.
std::vector<SourceUnit> all_units(const Corpus& c) {
  std::vector<SourceUnit> units;
  for (const ParsedFile& file : c.files) {
    for (SourceUnit unit : file.units) {
      unit.id = static_cast<UnitId>(units.size());
      units.push_back(std::move(unit));
    }
  }
  return units;
}

UnitId function_in(const std::vector<SourceUnit>& units, std::string_view path_suffix, std::string_view name) {
  for (const SourceUnit& unit : units) {
    if (unit.kind == UnitKind::Function && unit.name == name && unit.path.size() >= path_suffix.size() &&
        unit.path.compare(unit.path.size() - path_suffix.size(), path_suffix.size(), path_suffix) == 0) {
      return unit.id;
    }
  }
  return kNoUnit;
}

bool is_direct_source(std::string_view relative) {
  return relative.substr(0, 4) == "src/" && relative.find('/', 4) == std::string_view::npos &&
         relative.size() > 2 && relative.substr(relative.size() - 2) == ".c";
}

}  // namespace

// 1. Every file parses and its units partition it.
PRAG_TEST(corpus_files_are_partitioned) {
  if (skip("corpus_files_are_partitioned")) return;
  const Corpus& c = corpus();
  CHECK(c.paths.size() >= 40u);
  std::size_t units = 0;
  std::size_t with_errors = 0;
  std::size_t errors = 0;
  for (std::size_t i = 0; i < c.files.size(); ++i) {
    const ParsedFile& file = c.files[i];
    CHECK(!c.texts[i].empty());
    CHECK_EQ(file.path, c.paths[i]);
    CHECK(file.language == "c" || file.language == "cpp");
    check_partition(c.texts[i], file, c.relative(i));
    units += file.units.size();
    // the flag and the count agree
    const std::size_t count = count_syntax_errors(file.language, c.texts[i]);
    CHECK_EQ(count > 0, file.had_syntax_errors);
    errors += count;
    if (file.had_syntax_errors) ++with_errors;
  }
  std::printf("  corpus: %zu files, %zu bytes, %zu units; tree-sitter reports errors in %zu files (%zu nodes)\n",
              c.paths.size(), c.bytes, units, with_errors, errors);
  // what makes the corpus hard: most files do not parse cleanly
  CHECK(with_errors > c.paths.size() / 2);
}

// 2. Functions everybody knows, where they are defined. Every entry was looked
// up in the sources.
PRAG_TEST(well_known_functions_are_found) {
  if (skip("well_known_functions_are_found")) return;
  const Corpus& c = corpus();
  const std::pair<const char*, const char*> known[] = {
      // allocation
      {"src/alloc.c", "mi_malloc"},
      {"src/alloc.c", "mi_zalloc"},
      {"src/alloc.c", "mi_calloc"},
      {"src/alloc.c", "mi_realloc"},
      {"src/alloc.c", "mi_malloc_small"},
      {"src/alloc.c", "mi_heap_malloc"},
      {"src/alloc.c", "mi_theap_malloc"},
      {"src/alloc.c", "mi_page_malloc_zero"},
      {"src/alloc.c", "mi_expand"},
      {"src/alloc.c", "mi_strdup"},
      {"src/alloc.c", "mi_new"},
      {"src/alloc-aligned.c", "mi_malloc_aligned"},
      {"src/alloc-aligned.c", "mi_zalloc_aligned"},
      {"src/alloc-posix.c", "mi_posix_memalign"},
      {"src/alloc-override.c", "mi_malloc_size"},
      // free
      {"src/free.c", "mi_free"},
      {"src/free.c", "mi_free_size"},
      {"src/free.c", "mi_usable_size"},
      {"src/free.c", "_mi_free_generic"},
      // heap
      {"src/heap.c", "mi_heap_new"},
      {"src/heap.c", "mi_heap_delete"},
      {"src/heap.c", "mi_heap_destroy"},
      {"src/theap.c", "mi_heap_collect"},
      {"src/theap.c", "mi_collect"},
      // page
      {"src/page.c", "_mi_page_retire"},
      {"src/page.c", "_mi_page_free"},
      {"src/page.c", "_mi_malloc_generic"},
      {"src/page-queue.c", "mi_page_queue_remove"},
      {"src/page-map.c", "_mi_page_map_init"},
      // arena
      {"src/arena.c", "_mi_arenas_alloc"},
      {"src/arena.c", "mi_arenas_try_find_free"},
      {"src/arena.c", "mi_reserve_os_memory"},
      {"src/arena.c", "mi_manage_os_memory"},
      {"src/arena.c", "mi_arena_purge"},
      {"src/arena.c", "mi_debug_show_arenas"},
      {"src/arena.c", "mi_heap_visit_blocks"},
      {"src/bitmap.c", "mi_bitmap_init"},
      {"src/bitmap.c", "mi_bitmap_setN"},
      // os
      {"src/os.c", "_mi_os_alloc"},
      {"src/os.c", "_mi_os_free"},
      {"src/os.c", "_mi_os_commit"},
      {"src/os.c", "_mi_os_decommit"},
      {"src/os.c", "_mi_os_page_size"},
      // options
      {"src/options.c", "mi_option_get"},
      {"src/options.c", "mi_option_set"},
      {"src/options.c", "mi_option_is_enabled"},
      {"src/options.c", "_mi_options_init"},
      {"src/options.c", "_mi_error_message"},
      {"src/options.c", "mi_version"},
      // stats
      {"src/stats.c", "mi_stats_reset"},
      {"src/stats.c", "mi_stats_print"},
      {"src/stats.c", "_mi_clock_now"},
      {"src/stats.c", "mi_process_info"},
      // init
      {"src/init.c", "mi_thread_init"},
      {"src/init.c", "mi_thread_done"},
      {"src/init.c", "mi_process_init"},
  };
  std::size_t found = 0;
  for (const auto& [path, name] : known) {
    const std::size_t i = index_of(c, path);
    REQUIRE(i < c.paths.size());
    const SourceUnit* unit = parse_checks::find_unit_named(c.files[i], name, UnitKind::Function);
    if (unit == nullptr) {
      parse_checks::fail(__FILE__, __LINE__, path, std::string("function not found: ") + name);
      continue;
    }
    ++found;
    // the unit is the definition: its declaration names the function and its text has a body
    const std::string call = std::string(name) + "(";
    PARSE_EXPECT(unit->signature.find(call) != std::string::npos, path, std::string(name) + ": signature");
    const std::size_t body = unit->text.find('{', unit->text.find(call));
    PARSE_EXPECT(body != std::string::npos, path, std::string(name) + ": no body");
    PARSE_EXPECT(unit->text.find_last_of('}') != std::string::npos && unit->text.find_last_of('}') > body, path,
                 std::string(name) + ": body not closed");
    // find_unit agrees
    std::size_t begin = 0;
    std::size_t end = 0;
    PARSE_EXPECT(find_unit(c.paths[i], c.texts[i], name, &begin, &end), path, std::string(name) + ": find_unit");
    PARSE_EXPECT(begin <= unit->start_byte && unit->end_byte <= end, path, std::string(name) + ": find_unit range");
  }
  std::printf("  %zu of %zu well-known functions found\n", found, sizeof(known) / sizeof(known[0]));
  CHECK_EQ(found, sizeof(known) / sizeof(known[0]));

  // declaration macros in front of the name, the body on the next line
  const std::size_t arena = index_of(c, "src/arena.c");
  REQUIRE(arena < c.paths.size());
  const SourceUnit* find_free = parse_checks::find_unit_named(c.files[arena], "mi_arenas_try_find_free", UnitKind::Function);
  REQUIRE(find_free != nullptr);
  CHECK_EQ(find_free->doc, std::string("allocate slices from the arenas"));
  CHECK(find_free->signature.find("static mi_decl_noinline void* mi_arenas_try_find_free( mi_heap_t* heap,") == 0);
  const std::size_t alloc = index_of(c, "src/alloc.c");
  REQUIRE(alloc < c.paths.size());
  const SourceUnit* mi_malloc = parse_checks::find_unit_named(c.files[alloc], "mi_malloc", UnitKind::Function);
  REQUIRE(mi_malloc != nullptr);
  CHECK_EQ(mi_malloc->signature,
           std::string("mi_decl_nodiscard mi_decl_restrict void* mi_malloc(size_t size) mi_attr_noexcept"));
  CHECK_EQ(mi_malloc->text, std::string("\nmi_decl_nodiscard mi_decl_restrict void* mi_malloc(size_t size) "
                                        "mi_attr_noexcept {\n  return mi_theap_malloc(_mi_theap_default(), size);\n}\n"));
}

// 3. Coverage against an independent count of top-level function bodies.
PRAG_TEST(top_level_functions_are_covered) {
  if (skip("top_level_functions_are_covered")) return;
  const Corpus& c = corpus();
  std::size_t total = 0;
  std::size_t total_found = 0;
  std::size_t direct_total = 0;
  std::size_t direct_found = 0;
  std::printf("  %-36s %8s %8s %8s %9s\n", "file", "counted", "found", "units", "coverage");
  for (std::size_t i = 0; i < c.files.size(); ++i) {
    const std::string relative = c.relative(i);
    const ParsedFile& file = c.files[i];
    const std::vector<std::size_t> bodies = parse_checks::top_level_function_bodies(c.texts[i]);
    std::size_t found = 0;
    for (const std::size_t body : bodies) {
      const bool covered = std::any_of(file.units.begin(), file.units.end(), [body](const SourceUnit& u) {
        return u.kind == UnitKind::Function && u.start_byte <= body && body < u.end_byte;
      });
      if (covered) ++found;
    }
    const double coverage = bodies.empty() ? 1.0 : static_cast<double>(found) / static_cast<double>(bodies.size());
    std::printf("  %-36s %8zu %8zu %8zu %8.1f%%\n", relative.c_str(), bodies.size(), found,
                count_kind(file, UnitKind::Function), 100.0 * coverage);
    total += bodies.size();
    total_found += found;
    if (relative.substr(0, 4) == "src/" && relative.substr(relative.size() - 2) == ".c") {
      PARSE_EXPECT(coverage >= 0.95, relative, "fewer than 95% of the counted function bodies are function units");
    }
    if (is_direct_source(relative)) {
      direct_total += bodies.size();
      direct_found += found;
      // the counter would be useless if it counted nothing
      if (relative != "src/static.c") PARSE_EXPECT(bodies.size() >= 5, relative, "the counter found almost nothing");
    }
  }
  std::printf("  src/*.c: %zu of %zu counted bodies covered (%.2f%%)\n", direct_found, direct_total,
              direct_total == 0 ? 100.0 : 100.0 * static_cast<double>(direct_found) / static_cast<double>(direct_total));
  std::printf("  all files: %zu of %zu counted bodies covered (%.2f%%)\n", total_found, total,
              total == 0 ? 100.0 : 100.0 * static_cast<double>(total_found) / static_cast<double>(total));
  CHECK(direct_total > 800u);
  CHECK(static_cast<double>(direct_found) >= 0.95 * static_cast<double>(direct_total));
}

// 4. Function units have proper names and hold one function each.
PRAG_TEST(function_units_hold_one_named_function) {
  if (skip("function_units_hold_one_named_function")) return;
  const Corpus& c = corpus();
  std::size_t functions = 0;
  std::size_t plain = 0;
  std::size_t operators = 0;
  std::size_t members = 0;
  for (std::size_t i = 0; i < c.files.size(); ++i) {
    const std::string relative = c.relative(i);
    const ParsedFile& file = c.files[i];
    const std::vector<std::size_t> bodies = parse_checks::top_level_function_bodies(c.texts[i]);
    // names that are not C identifiers exist only where C++ is written
    const bool has_cpp = file.language == "cpp" || c.texts[i].find("__cplusplus") != std::string::npos;
    for (const SourceUnit& unit : file.units) {
      if (unit.kind != UnitKind::Function) continue;
      ++functions;
      PARSE_EXPECT(!unit.name.empty(), relative, "function without name at line " + std::to_string(unit.start_line));
      PARSE_EXPECT(parse_checks::is_function_name(unit.name), relative, "not a function name: '" + unit.name + "'");
      if (parse_checks::is_identifier(unit.name)) {
        ++plain;
      } else {
        PARSE_EXPECT(has_cpp, relative, "not a C identifier in a C file: '" + unit.name + "'");
        if (unit.name.find("operator") != std::string::npos) {
          ++operators;
        } else {
          ++members;
        }
      }
      PARSE_EXPECT(!unit.signature.empty(), relative, unit.name + ": no signature");
      const auto inside = std::count_if(bodies.begin(), bodies.end(), [&unit](std::size_t body) {
        return unit.start_byte <= body && body < unit.end_byte;
      });
      PARSE_EXPECT(inside <= 1, relative,
                   unit.name + ": " + std::to_string(inside) + " top-level function bodies in one unit");
    }
  }
  std::printf("  %zu function units: %zu C identifiers, %zu C++ operator functions, %zu C++ members\n", functions,
              plain, operators, members);
  CHECK(functions > 1000u);
  CHECK_EQ(plain + operators + members, functions);
}

// 5. The call graph of the corpus.
PRAG_TEST(call_graph_of_the_corpus) {
  if (skip("call_graph_of_the_corpus")) return;
  const Corpus& c = corpus();
  const std::vector<SourceUnit> units = all_units(c);
  const Stopwatch watch;
  const graph::CallGraph g = build_call_graph(units);
  const double seconds = watch.elapsed_s();
  std::size_t functions = 0;
  std::size_t types = 0;
  std::size_t macros = 0;
  std::size_t regions = 0;
  std::size_t calling = 0;
  for (const SourceUnit& unit : units) {
    functions += unit.kind == UnitKind::Function ? 1 : 0;
    types += unit.kind == UnitKind::Type ? 1 : 0;
    macros += unit.kind == UnitKind::Macro ? 1 : 0;
    regions += unit.kind == UnitKind::Region ? 1 : 0;
    calling += g.callees(unit.id).empty() ? 0 : 1;
  }
  std::printf("  call graph: %zu units (%zu functions, %zu types, %zu macros, %zu regions), %zu edges, "
              "%zu defined names, %zu units with callees, built in %.1f ms\n",
              units.size(), functions, types, macros, regions, g.edges(), g.all_definitions().size(), calling,
              seconds * 1e3);
  CHECK_EQ(g.size(), units.size());
  CHECK(g.edges() > 2 * functions);

  const auto has_edge = [&](UnitId from, UnitId to) {
    const auto callees = g.callees(from);
    const auto callers = g.callers(to);
    return std::find(callees.begin(), callees.end(), to) != callees.end() &&
           std::find(callers.begin(), callers.end(), from) != callers.end();
  };
  // alloc.c: mi_malloc returns mi_theap_malloc(_mi_theap_default(), size)
  const UnitId mi_malloc = function_in(units, "/src/alloc.c", "mi_malloc");
  const UnitId mi_theap_malloc = function_in(units, "/src/alloc.c", "mi_theap_malloc");
  REQUIRE(mi_malloc != kNoUnit);
  REQUIRE(mi_theap_malloc != kNoUnit);
  CHECK(has_edge(mi_malloc, mi_theap_malloc));
  CHECK(!has_edge(mi_theap_malloc, mi_malloc));
  // across files: alloc.c: mi_theap_reallocf frees with mi_free, defined in free.c
  const UnitId reallocf = function_in(units, "/src/alloc.c", "mi_theap_reallocf");
  const UnitId mi_free = function_in(units, "/src/free.c", "mi_free");
  REQUIRE(reallocf != kNoUnit);
  REQUIRE(mi_free != kNoUnit);
  CHECK(has_edge(reallocf, mi_free));
  // through a macro-made body: alloc-override.c: void free(void* p) MI_FORWARD0(mi_free, p)
  const UnitId libc_free = function_in(units, "/src/alloc-override.c", "free");
  REQUIRE(libc_free != kNoUnit);
  CHECK(has_edge(libc_free, mi_free));

  // widely used helpers
  const UnitId page_size = function_in(units, "/src/os.c", "_mi_os_page_size");
  REQUIRE(page_size != kNoUnit);
  std::printf("  callers: mi_free %zu, _mi_os_page_size %zu, mi_theap_malloc %zu\n", g.callers(mi_free).size(),
              g.callers(page_size).size(), g.callers(mi_theap_malloc).size());
  CHECK(g.callers(page_size).size() >= 5u);
  CHECK(g.callers(mi_free).size() >= 10u);
  for (const UnitId caller : g.callers(page_size)) {
    REQUIRE(caller < units.size());
    CHECK(parse_checks::calls(units[caller], "_mi_os_page_size"));
  }
  CHECK(!g.definitions("mi_malloc").empty());
  // a function-like macro is a node: internal.h defines mi_likely
  bool macro_defined = false;
  for (const UnitId id : g.definitions("mi_likely")) macro_defined |= units[id].kind == UnitKind::Macro;
  CHECK(macro_defined);
}

// 8. Parsing is re-entrant: sixteen workers give what one thread gives.
PRAG_TEST(parallel_parsing_gives_the_same_units) {
  if (skip("parallel_parsing_gives_the_same_units")) return;
#ifdef PRAG_PARSE_TEST_WITH_EXEC
  const Corpus& c = corpus();
  std::vector<std::uint32_t> sizes;
  for (const std::string& text : c.texts) sizes.push_back(static_cast<std::uint32_t>(text.size()));
  // Files are the items; nearly every file exceeds a block and gets one of its own.
  const std::vector<exec::Block> blocks = exec::partition_by_bytes(sizes, 16 * 1024);

  exec::OrderedExecutor executor(16);
  CHECK_EQ(executor.threads(), 16u);
  double best_parallel = 0.0;
  double best_sequential = c.parse_seconds;
  for (int round = 0; round < 3; ++round) {
    std::vector<ParsedFile> parsed;
    const Stopwatch watch;
    const exec::RunStats stats = executor.run(
        blocks,
        [&](const exec::Block& block, unsigned) {
          std::vector<ParsedFile> files;
          for (std::size_t i = block.first; i < block.first + block.count; ++i) {
            files.push_back(parse_source(c.paths[i], c.texts[i]));
          }
          return files;
        },
        [&](const exec::Block&, std::vector<ParsedFile>&& files) {
          for (ParsedFile& file : files) parsed.push_back(std::move(file));
        });
    const double seconds = watch.elapsed_s();
    if (round == 0 || seconds < best_parallel) best_parallel = seconds;
    CHECK_EQ(stats.admission_violations, std::uint64_t{0});
    CHECK_EQ(stats.commit_violations, std::uint64_t{0});
    REQUIRE(parsed.size() == c.files.size());
    for (std::size_t i = 0; i < parsed.size(); ++i) {
      PARSE_EXPECT(parse_checks::same_units(parsed[i], c.files[i]), c.relative(i),
                   "parallel and sequential parsing differ");
    }

    // once more on one thread, for the comparison of times
    const Stopwatch sequential;
    std::size_t units = 0;
    for (std::size_t i = 0; i < c.paths.size(); ++i) units += parse_source(c.paths[i], c.texts[i]).units.size();
    best_sequential = std::min(best_sequential, sequential.elapsed_s());
    CHECK(units > 0u);
  }
  std::printf("  parse time for %zu files, %zu bytes: sequential %.1f ms (%.1f MB/s), 16 workers %.1f ms, "
              "speed-up %.1fx (best of 3)\n",
              c.paths.size(), c.bytes, best_sequential * 1e3,
              static_cast<double>(c.bytes) / 1e6 / std::max(best_sequential, 1e-9), best_parallel * 1e3,
              best_sequential / std::max(best_parallel, 1e-9));
#else
  std::printf("  skipped parallel_parsing_gives_the_same_units: built without the exec module\n");
#endif
}

// Options change the units, never the partition.
PRAG_TEST(corpus_is_partitioned_under_other_options) {
  if (skip("corpus_is_partitioned_under_other_options")) return;
  const Corpus& c = corpus();
  ParseOptions small;
  small.max_region_bytes = 512;
  small.max_unit_bytes = 1024;
  ParseOptions plain;
  plain.attach_leading_comment = false;
  plain.merge_small_definitions = false;
  std::size_t units_small = 0;
  std::size_t units_plain = 0;
  std::size_t macros_plain = 0;
  for (std::size_t i = 0; i < c.paths.size(); ++i) {
    const ParsedFile a = parse_source(c.paths[i], c.texts[i], small);
    check_partition(c.texts[i], a, c.relative(i) + " (small limits)", small);
    units_small += a.units.size();
    const ParsedFile b = parse_source(c.paths[i], c.texts[i], plain);
    check_partition(c.texts[i], b, c.relative(i) + " (nothing attached or merged)", plain);
    units_plain += b.units.size();
    macros_plain += count_kind(b, UnitKind::Macro);
    // the functions are the same whatever the options
    CHECK_EQ(count_kind(b, UnitKind::Function), count_kind(c.files[i], UnitKind::Function));
  }
  std::printf("  units with limits 512/1024: %zu; with every definition a unit: %zu (%zu macros)\n", units_small,
              units_plain, macros_plain);
  CHECK(units_small > units_plain / 2);
}
