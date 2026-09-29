// Unit tests of the parse module on hand-written sources. The tests against the
// mimalloc corpus are in test_parse_corpus.cpp.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "checks.hpp"
#include "prag/core/text.hpp"
#include "prag/graph/call_graph.hpp"
#include "prag/parse/parser.hpp"

using namespace prag;
using namespace prag::parse;
using parse_checks::calls;
using parse_checks::check_partition;
using parse_checks::find_unit_named;

namespace {

// C in the style of mimalloc: declaration macros, conditional compilation,
// annotations, bodies supplied by macros.
const char kC[] = R"src(/* Sample in the style of mimalloc. */
#include "mimalloc.h"
#include <string.h>  // memset

#define MI_KiB 1024
#define mi_align_up(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef struct mi_block_s mi_block_t;
typedef void (mi_cdecl mi_output_fun)(const char* msg, void* arg);

// A page of blocks.
typedef struct mi_page_s {
  mi_block_t* free;   // "{" in a comment
  size_t used;
} mi_page_t;

typedef enum mi_kind_e { MI_SMALL, MI_LARGE } mi_kind_t;

static const char* names[] = { "small", "large {" };


// Allocate `size` bytes.
// Returns NULL when out of memory.
mi_decl_nodiscard extern inline mi_decl_restrict void* mi_malloc(size_t size) mi_attr_noexcept {
  if mi_unlikely(size == 0) { return NULL; }
  void* p = mi_heap_malloc(mi_heap_get_default(), mi_align_up(size, 8));
  return p;
}

#if MI_DEBUG
static mi_decl_noinline void mi_check(const mi_page_t* page)
{
  mi_assert_internal(page != NULL);
  if (page->used > MI_KiB) mi_check(page);
}
#endif

#ifdef _WIN32
static size_t mi_page_size(int unused) {
#else
static size_t mi_page_size(void) {
#endif
  return 4096;
}

#ifdef __cplusplus
extern "C" {
#endif

_Success_(return != 0) _Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(_Size)
void* __cdecl _expand(_Pre_notnull_ void* _Block, _In_ size_t _Size) {
  return mi_expand(_Block, _Size);
}

mi_decl_export void* malloc(size_t size)  MI_FORWARD1(mi_malloc, size)
mi_decl_export void  free(void* p)        MI_FORWARD0(mi_free, p)

#ifdef __cplusplus
}
#endif

int old_style(a, b) int a; int b; { return a + b; }

MI_PASTE(mi_register_, MI_SUFFIX)(int x)
{
  return x;
}
)src";

const char kCpp[] = R"src(// Sample C++ file.
#include <string>
#include <vector>

#define SQUARE(x) ((x) * (x))
#define LIMIT 10

namespace geometry {

// A point in the plane.
struct Point {
  double x = 0;
  double y = 0;
};

/* Shapes have an area. */
class Shape {
 public:
  explicit Shape(std::string name) : name_(std::move(name)), id_{next_id()} {}
  virtual ~Shape() = default;

  // Area in square units.
  virtual double area() const = 0;

  const std::string& name() const { return name_; }

  template <class Visitor>
  void visit(Visitor&& visitor) const {
    visitor(*this);
  }

  bool operator==(const Shape& other) const { return id_ == other.id_; }

 private:
  static int next_id();
  std::string name_;
  int id_;
};

class Circle final : public Shape {
 public:
  Circle(double r);
  double area() const override;

 private:
  double r_;
};

Circle::Circle(double r) : Shape("circle"), r_(r) {}

double Circle::area() const {
  return 3.14159 * SQUARE(r_);
}

int Shape::next_id() {
  static int id = 0;
  return ++id;
}

// Largest of two values.
template <typename T>
T largest(T a, T b) {
  return a < b ? b : a;
}

template <typename T>
std::vector<T> repeat(const T& value, int n) {
  std::vector<T> out;
  for (int i = 0; i < n; ++i) out.push_back(largest<T>(value, value));
  return out;
}

enum class Colour : unsigned char { Red, Green, Blue };

using Points = std::vector<Point>;

}  // namespace geometry

int main() {
  geometry::Circle c(2.0);
  auto twice = [](double v) { return 2 * v; };
  return static_cast<int>(twice(c.area())) > LIMIT ? 0 : 1;
}
)src";

const char kPython[] = R"src(#!/usr/bin/env python3
"""Sample module."""
import functools
import os
from typing import List

LIMIT = 10


def cached(fn):
    """Decorator that caches results."""
    store = {}

    @functools.wraps(fn)
    def wrapper(*args):
        if args not in store:
            store[args] = fn(*args)
        return store[args]

    return wrapper


# Fibonacci numbers,
# computed recursively.
@cached
def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)


class Stack:
    """A stack of integers."""

    capacity = LIMIT

    def __init__(self, items: List[int] = ()):
        self.items = list(items)

    def push(self, item):
        self.items.append(item)
        return self

    @property
    def top(self):
        # the last element
        return self.items[-1]

    @staticmethod
    def of(*items):
        return Stack(items)

    class Empty(Exception):
        pass

    async def drain(self):
        while self.items:
            yield self.items.pop()


class Marker:
    pass


def main():
    s = Stack.of(1, 2, 3).push(fib(10))
    print(s.top, os.getcwd())


if __name__ == "__main__":
    main()
)src";

std::vector<std::string> names_of(const ParsedFile& file, UnitKind kind) {
  std::vector<std::string> names;
  for (const SourceUnit& unit : file.units) {
    if (unit.kind == kind) names.push_back(unit.name);
  }
  return names;
}

std::size_t count_named(const ParsedFile& file, std::string_view name) {
  std::size_t count = 0;
  for (const SourceUnit& unit : file.units) {
    if (unit.name == name) ++count;
  }
  return count;
}

bool starts_with(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }
bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}
bool has(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

std::string with_crlf(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (c == '\n') out.push_back('\r');
    out.push_back(c);
  }
  return out;
}

// A function of `statements` statements, each on a line of its own.
std::string long_function(const std::string& name, int statements) {
  std::string text = "// A long function.\nstatic int " + name + "(int x)\n{\n";
  for (int i = 0; i < statements; ++i) {
    text += "  x = step_" + std::to_string(i % 7) + "(x, " + std::to_string(i) + ");\n";
  }
  text += "  return " + name + "(x);\n}\n";
  return text;
}

}  // namespace

PRAG_TEST(language_is_detected_by_extension_and_content) {
  CHECK_EQ(detect_language("src/alloc.c", ""), std::string("c"));
  CHECK_EQ(detect_language("SRC\\ALLOC.C", ""), std::string("c"));
  CHECK_EQ(detect_language("a.cc", ""), std::string("cpp"));
  CHECK_EQ(detect_language("a.cpp", ""), std::string("cpp"));
  CHECK_EQ(detect_language("a.cxx", ""), std::string("cpp"));
  CHECK_EQ(detect_language("a.hpp", ""), std::string("cpp"));
  CHECK_EQ(detect_language("a.hh", ""), std::string("cpp"));
  CHECK_EQ(detect_language("tool.py", ""), std::string("python"));
  CHECK_EQ(detect_language("notes.txt", "class A {};"), std::string(""));
  CHECK_EQ(detect_language("Makefile", ""), std::string(""));
  CHECK_EQ(detect_language("dir.c/readme", ""), std::string(""));
  CHECK_EQ(detect_language(".c", ""), std::string(""));
  CHECK_EQ(detect_language("", ""), std::string(""));

  // A header is C unless it uses what only C++ has.
  CHECK_EQ(detect_language("a.h", "typedef struct s { int class_; } s_t;\nint f(int);\n"), std::string("c"));
  CHECK_EQ(detect_language("a.h", "#ifdef __cplusplus\nextern \"C\" {\n#endif\nint f(void);\n#ifdef __cplusplus\n}\n#endif\n"),
           std::string("c"));
  CHECK_EQ(detect_language("a.h", "// class Foo { public: };\n/* template <class T> */\nint f(void);\n"),
           std::string("c"));
  CHECK_EQ(detect_language("a.h", "const char* s = \"namespace x { a::b }\";\n"), std::string("c"));
  CHECK_EQ(detect_language("a.h", "#define CAST(x) static_cast<int>(x)\nint f(void);\n"), std::string("c"));
  CHECK_EQ(detect_language("a.h", "class Foo { public: int x; };\n"), std::string("cpp"));
  CHECK_EQ(detect_language("a.h", "template <class T> T f(T x);\n"), std::string("cpp"));
  CHECK_EQ(detect_language("a.h", "namespace n { int f(); }\n"), std::string("cpp"));
  CHECK_EQ(detect_language("a.h", "void* operator new(std::size_t n);\n"), std::string("cpp"));
  CHECK_EQ(detect_language("a.h", "int f() { return std::max(1, 2); }\n"), std::string("cpp"));
}

PRAG_TEST(c_functions_are_found_behind_macros) {
  const ParsedFile file = parse_source("sample.c", kC);
  CHECK_EQ(file.language, std::string("c"));
  CHECK(file.had_syntax_errors);  // the declaration macros are not C
  REQUIRE(check_partition(kC, file, "sample.c"));

  const std::vector<std::string> expected = {"mi_malloc", "mi_check",  "mi_page_size", "_expand",
                                             "malloc",    "free",      "old_style",    "mi_register_MI_SUFFIX"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);

  const SourceUnit* mi_malloc = find_unit_named(file, "mi_malloc", UnitKind::Function);
  REQUIRE(mi_malloc != nullptr);
  CHECK_EQ(mi_malloc->signature,
           std::string("mi_decl_nodiscard extern inline mi_decl_restrict void* mi_malloc(size_t size) mi_attr_noexcept"));
  CHECK_EQ(mi_malloc->doc, std::string("Allocate `size` bytes.\nReturns NULL when out of memory."));
  // blank lines belong to the unit that follows them, the comment to the function
  CHECK(starts_with(mi_malloc->text, "\n\n// Allocate `size` bytes.\n"));
  CHECK(ends_with(mi_malloc->text, "  return p;\n}\n"));
  CHECK_EQ(mi_malloc->start_line, 20u);
  CHECK_EQ(mi_malloc->end_line, 28u);
  const std::vector<std::string> callees = {"mi_unlikely", "mi_heap_malloc", "mi_heap_get_default", "mi_align_up"};
  REQUIRE(mi_malloc->callees.size() == callees.size());
  for (std::size_t i = 0; i < callees.size(); ++i) CHECK_EQ(mi_malloc->callees[i], callees[i]);

  // the conditional around a function is part of its unit; a function does not call itself
  const SourceUnit* mi_check = find_unit_named(file, "mi_check", UnitKind::Function);
  REQUIRE(mi_check != nullptr);
  CHECK(starts_with(mi_check->text, "\n#if MI_DEBUG\nstatic"));
  CHECK(ends_with(mi_check->text, "}\n#endif\n"));
  CHECK_EQ(mi_check->signature, std::string("static mi_decl_noinline void mi_check(const mi_page_t* page)"));
  REQUIRE(mi_check->callees.size() == 1u);
  CHECK_EQ(mi_check->callees[0], std::string("mi_assert_internal"));

  // two alternative headers, one body, one unit
  CHECK_EQ(count_named(file, "mi_page_size"), std::size_t{1});
  const SourceUnit* page_size = find_unit_named(file, "mi_page_size", UnitKind::Function);
  REQUIRE(page_size != nullptr);
  CHECK(has(page_size->text, "mi_page_size(int unused)"));
  CHECK(has(page_size->text, "mi_page_size(void)"));
  CHECK(ends_with(page_size->text, "  return 4096;\n}\n"));

  // annotations with arguments in front of the declarator
  const SourceUnit* expand = find_unit_named(file, "_expand", UnitKind::Function);
  REQUIRE(expand != nullptr);
  CHECK(has(expand->text, "extern \"C\" {"));  // opened directly above the function
  CHECK(has(expand->text, "_Success_(return != 0)"));
  CHECK(calls(*expand, "mi_expand"));

  // bodies supplied by a macro: the forwarded function is what the body calls
  const SourceUnit* malloc_unit = find_unit_named(file, "malloc", UnitKind::Function);
  REQUIRE(malloc_unit != nullptr);
  CHECK_EQ(malloc_unit->signature, std::string("mi_decl_export void* malloc(size_t size)"));
  CHECK(calls(*malloc_unit, "mi_malloc"));
  CHECK(calls(*malloc_unit, "MI_FORWARD1"));
  CHECK(!calls(*malloc_unit, "size"));
  const SourceUnit* free_unit = find_unit_named(file, "free", UnitKind::Function);
  REQUIRE(free_unit != nullptr);
  CHECK(calls(*free_unit, "mi_free"));
  CHECK_EQ(malloc_unit->end_byte, free_unit->start_byte);

  // old-style parameter declarations come from the syntax tree
  const SourceUnit* old_style = find_unit_named(file, "old_style", UnitKind::Function);
  REQUIRE(old_style != nullptr);
  CHECK(has(old_style->text, "int a; int b; { return a + b; }"));

  // types and macros
  const SourceUnit* page = find_unit_named(file, "mi_page_t", UnitKind::Type);
  REQUIRE(page != nullptr);
  CHECK_EQ(page->signature, std::string("typedef struct mi_page_s"));
  CHECK_EQ(page->doc, std::string("A page of blocks."));
  CHECK(ends_with(page->text, "} mi_page_t;\n"));
  const SourceUnit* align = find_unit_named(file, "mi_align_up", UnitKind::Macro);
  REQUIRE(align != nullptr);
  CHECK_EQ(align->signature, std::string("#define mi_align_up(x, a)"));
  CHECK(align->callees.empty());  // x and a are parameters

  // small definitions are part of the regions around them
  CHECK_EQ(count_named(file, "MI_KiB"), std::size_t{0});
  CHECK_EQ(count_named(file, "mi_block_t"), std::size_t{0});
  CHECK_EQ(count_named(file, "mi_kind_t"), std::size_t{0});
  CHECK(file.units.front().kind == UnitKind::Region);
  CHECK(has(file.units.front().text, "#define MI_KiB 1024"));
  for (const SourceUnit& unit : file.units) {
    if (unit.kind != UnitKind::Region) continue;
    CHECK(unit.name.empty());
    CHECK(unit.signature.empty());
    CHECK(unit.doc.empty());
  }
}

PRAG_TEST(every_definition_is_a_unit_when_small_ones_are_not_merged) {
  ParseOptions options;
  options.merge_small_definitions = false;
  const ParsedFile file = parse_source("sample.c", kC, options);
  REQUIRE(check_partition(kC, file, "sample.c", options));
  CHECK(find_unit_named(file, "MI_KiB", UnitKind::Macro) != nullptr);
  CHECK(find_unit_named(file, "mi_align_up", UnitKind::Macro) != nullptr);
  CHECK(find_unit_named(file, "mi_block_t", UnitKind::Type) != nullptr);
  CHECK(find_unit_named(file, "mi_output_fun", UnitKind::Type) != nullptr);
  CHECK(find_unit_named(file, "mi_kind_t", UnitKind::Type) != nullptr);
  CHECK(find_unit_named(file, "mi_page_t", UnitKind::Type) != nullptr);
  CHECK_EQ(names_of(file, UnitKind::Function).size(), std::size_t{8});
}

PRAG_TEST(leading_comments_can_stay_outside) {
  ParseOptions options;
  options.attach_leading_comment = false;
  const ParsedFile file = parse_source("sample.c", kC, options);
  REQUIRE(check_partition(kC, file, "sample.c", options));
  const SourceUnit* mi_malloc = find_unit_named(file, "mi_malloc", UnitKind::Function);
  REQUIRE(mi_malloc != nullptr);
  CHECK(mi_malloc->doc.empty());
  CHECK(!has(mi_malloc->text, "Allocate"));
  CHECK(starts_with(mi_malloc->text, "mi_decl_nodiscard"));
  REQUIRE(mi_malloc->id > 0);
  CHECK(has(file.units[mi_malloc->id - 1].text, "// Allocate `size` bytes."));
}

PRAG_TEST(cpp_classes_methods_and_templates) {
  const ParsedFile file = parse_source("sample.cpp", kCpp);
  CHECK_EQ(file.language, std::string("cpp"));
  CHECK(!file.had_syntax_errors);
  REQUIRE(check_partition(kCpp, file, "sample.cpp"));

  const std::vector<std::string> expected = {"Shape::Shape",   "Shape::name",  "Shape::visit",   "Shape::operator==",
                                             "Circle::Circle", "Circle::area", "Shape::next_id", "largest",
                                             "repeat",         "main"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);

  // a class without member function definitions is one unit
  const SourceUnit* point = find_unit_named(file, "Point", UnitKind::Type);
  REQUIRE(point != nullptr);
  CHECK_EQ(point->doc, std::string("A point in the plane."));
  CHECK_EQ(point->signature, std::string("struct Point"));
  CHECK(ends_with(point->text, "  double y = 0;\n};\n"));
  const SourceUnit* circle = find_unit_named(file, "Circle", UnitKind::Type);
  REQUIRE(circle != nullptr);
  CHECK_EQ(circle->signature, std::string("class Circle final : public Shape"));
  CHECK(has(circle->text, "double area() const override;"));
  CHECK(ends_with(circle->text, "  double r_;\n};\n"));

  // a class with member function definitions: head, members, residue
  const SourceUnit* shape = find_unit_named(file, "Shape", UnitKind::Type);
  REQUIRE(shape != nullptr);
  CHECK_EQ(shape->doc, std::string("Shapes have an area."));
  CHECK(ends_with(shape->text, "class Shape {\n public:\n"));
  const SourceUnit* constructor = find_unit_named(file, "Shape::Shape", UnitKind::Function);
  REQUIRE(constructor != nullptr);
  CHECK_EQ(constructor->id, shape->id + 1);
  CHECK_EQ(constructor->signature,
           std::string("explicit Shape(std::string name) : name_(std::move(name)), id_{next_id()}"));
  CHECK(ends_with(constructor->text, "id_{next_id()} {}\n"));
  CHECK(calls(*constructor, "next_id"));
  const SourceUnit* residue = find_unit_named(file, "Shape#2", UnitKind::Region);
  REQUIRE(residue != nullptr);
  CHECK(has(residue->text, "virtual ~Shape() = default;"));
  CHECK(has(residue->text, "virtual double area() const = 0;"));
  const SourceUnit* tail = find_unit_named(file, "Shape#3", UnitKind::Region);
  REQUIRE(tail != nullptr);
  CHECK(has(tail->text, "static int next_id();"));
  CHECK(ends_with(tail->text, "  int id_;\n};\n"));
  const SourceUnit* visit = find_unit_named(file, "Shape::visit", UnitKind::Function);
  REQUIRE(visit != nullptr);
  CHECK_EQ(visit->signature, std::string("template <class Visitor> void visit(Visitor&& visitor) const"));
  const SourceUnit* equal = find_unit_named(file, "Shape::operator==", UnitKind::Function);
  REQUIRE(equal != nullptr);
  CHECK(has(equal->text, "return id_ == other.id_;"));

  // a declaration is not a definition
  CHECK_EQ(count_named(file, "Circle::area"), std::size_t{1});
  CHECK_EQ(count_named(file, "Shape::area"), std::size_t{0});
  const SourceUnit* area = find_unit_named(file, "Circle::area", UnitKind::Function);
  REQUIRE(area != nullptr);
  CHECK_EQ(area->signature, std::string("double Circle::area() const"));
  CHECK(calls(*area, "SQUARE"));

  const SourceUnit* largest = find_unit_named(file, "largest", UnitKind::Function);
  REQUIRE(largest != nullptr);
  CHECK_EQ(largest->doc, std::string("Largest of two values."));
  CHECK_EQ(largest->signature, std::string("template <typename T> T largest(T a, T b)"));
  const SourceUnit* repeat = find_unit_named(file, "repeat", UnitKind::Function);
  REQUIRE(repeat != nullptr);
  CHECK(calls(*repeat, "largest"));  // largest<T>(...): only the syntax tree sees this call
  CHECK(calls(*repeat, "push_back"));
  CHECK(!calls(*repeat, "for"));

  const SourceUnit* main_unit = find_unit_named(file, "main", UnitKind::Function);
  REQUIRE(main_unit != nullptr);
  CHECK(calls(*main_unit, "area"));
  CHECK(calls(*main_unit, "twice"));
  CHECK(!calls(*main_unit, "static_cast"));
  CHECK(!calls(*main_unit, "c"));  // geometry::Circle c(2.0) declares

  CHECK(find_unit_named(file, "SQUARE", UnitKind::Macro) != nullptr);
  CHECK_EQ(count_named(file, "LIMIT"), std::size_t{0});
  CHECK_EQ(count_named(file, "Colour"), std::size_t{0});  // one line: part of a region
}

PRAG_TEST(python_functions_classes_and_decorators) {
  const ParsedFile file = parse_source("sample.py", kPython);
  CHECK_EQ(file.language, std::string("python"));
  CHECK(!file.had_syntax_errors);
  REQUIRE(check_partition(kPython, file, "sample.py"));

  const std::vector<std::string> expected = {"cached",   "fib",      "Stack.__init__", "Stack.push",
                                             "Stack.top", "Stack.of", "Stack.drain",    "main"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);

  // a nested function stays inside the function that defines it
  const SourceUnit* cached = find_unit_named(file, "cached", UnitKind::Function);
  REQUIRE(cached != nullptr);
  CHECK_EQ(cached->signature, std::string("def cached(fn):"));
  CHECK_EQ(cached->doc, std::string("Decorator that caches results."));
  CHECK(has(cached->text, "def wrapper(*args):"));
  CHECK(ends_with(cached->text, "    return wrapper\n"));
  CHECK_EQ(count_named(file, "wrapper"), std::size_t{0});
  CHECK(calls(*cached, "wraps"));
  CHECK(calls(*cached, "fn"));

  // decorators and the comment above them belong to the function
  const SourceUnit* fib = find_unit_named(file, "fib", UnitKind::Function);
  REQUIRE(fib != nullptr);
  CHECK_EQ(fib->signature, std::string("@cached def fib(n: int) -> int:"));
  CHECK_EQ(fib->doc, std::string("Fibonacci numbers,\ncomputed recursively."));
  CHECK(starts_with(fib->text, "\n\n# Fibonacci numbers,\n# computed recursively.\n@cached\ndef fib"));
  REQUIRE(fib->callees.size() == 1u);  // not itself
  CHECK_EQ(fib->callees[0], std::string("cached"));

  const SourceUnit* stack = find_unit_named(file, "Stack", UnitKind::Type);
  REQUIRE(stack != nullptr);
  CHECK_EQ(stack->signature, std::string("class Stack:"));
  CHECK_EQ(stack->doc, std::string("A stack of integers."));
  CHECK(has(stack->text, "capacity = LIMIT"));
  CHECK(!has(stack->text, "def __init__"));
  const SourceUnit* top = find_unit_named(file, "Stack.top", UnitKind::Function);
  REQUIRE(top != nullptr);
  CHECK_EQ(top->signature, std::string("@property def top(self):"));
  CHECK(starts_with(top->text, "\n    @property\n    def top(self):\n"));
  const SourceUnit* drain = find_unit_named(file, "Stack.drain", UnitKind::Function);
  REQUIRE(drain != nullptr);
  CHECK_EQ(drain->signature, std::string("async def drain(self):"));
  CHECK(find_unit_named(file, "Stack.Empty", UnitKind::Type) != nullptr);
  const SourceUnit* marker = find_unit_named(file, "Marker", UnitKind::Type);
  REQUIRE(marker != nullptr);
  CHECK(ends_with(marker->text, "class Marker:\n    pass\n"));

  const SourceUnit* main_unit = find_unit_named(file, "main", UnitKind::Function);
  REQUIRE(main_unit != nullptr);
  const std::vector<std::string> callees = {"push", "of", "fib", "print", "getcwd"};
  for (const std::string& callee : callees) CHECK(calls(*main_unit, callee));

  CHECK(file.units.front().kind == UnitKind::Region);
  CHECK(has(file.units.front().text, "import functools"));
  CHECK(file.units.back().kind == UnitKind::Region);
  CHECK(has(file.units.back().text, "if __name__ == \"__main__\":"));
  CHECK(calls(file.units.back(), "main"));
}

PRAG_TEST(large_functions_are_split_at_statements) {
  const std::string head = "#include <stdio.h>\n\n";
  const std::string function = long_function("big", 2400);
  const std::string text = head + function + "\nint after(void) { return big(1); }\n";
  REQUIRE(function.size() > 3 * 16384);

  const ParseOptions options;
  const ParsedFile file = parse_source("big.c", text);
  REQUIRE(check_partition(text, file, "big.c"));

  const SourceUnit* first = find_unit_named(file, "big", UnitKind::Function);
  REQUIRE(first != nullptr);
  CHECK_EQ(first->doc, std::string("A long function."));
  CHECK_EQ(first->signature, std::string("static int big(int x)"));
  CHECK(has(first->text, "static int big(int x)\n{\n  x = step_0(x, 0);\n"));

  // the parts follow one another, are named big#2, big#3, ... and are regions
  std::size_t parts = 1;
  std::string joined = first->text;
  for (std::size_t i = first->id + 1; i < file.units.size(); ++i) {
    const SourceUnit& unit = file.units[i];
    if (unit.name != "big#" + std::to_string(parts + 1)) break;
    CHECK(unit.kind == UnitKind::Region);
    CHECK(unit.signature.empty());
    CHECK(unit.doc.empty());
    CHECK(calls(unit, "step_0"));
    CHECK(!calls(unit, "big"));
    joined += unit.text;
    ++parts;
  }
  const std::size_t needed = (function.size() + options.max_unit_bytes - 1) / options.max_unit_bytes;
  CHECK(parts >= needed);
  CHECK(parts <= needed + 1);
  CHECK(has(joined, function));
  // every cut is between two statements, and the parts are of similar size
  for (std::size_t i = first->id; i + 1 < first->id + parts; ++i) {
    CHECK(ends_with(file.units[i].text, ");\n"));
    CHECK(starts_with(file.units[i + 1].text, "  x = step_"));
    CHECK(file.units[i].text.size() > options.max_unit_bytes / 2);
  }
  CHECK(ends_with(file.units[first->id + parts - 1].text, "  return big(x);\n}\n"));
  const SourceUnit* after = find_unit_named(file, "after", UnitKind::Function);
  REQUIRE(after != nullptr);
  CHECK_EQ(after->id, first->id + parts);
  CHECK(calls(*after, "big"));

  // find_unit gives the function as a whole
  std::size_t begin = 0;
  std::size_t end = 0;
  REQUIRE(find_unit("big.c", text, "big", &begin, &end));
  CHECK_EQ(begin, static_cast<std::size_t>(first->start_byte));
  CHECK_EQ(end, static_cast<std::size_t>(after->start_byte));
  CHECK(!find_unit("big.c", text, "big#2", &begin, &end));
}

PRAG_TEST(splitting_descends_into_nested_blocks_and_lines) {
  ParseOptions options;
  options.max_unit_bytes = 2048;
  options.max_region_bytes = 1024;

  // one statement holds everything: cuts between the statements inside it
  std::string nested = "void nested(int n)\n{\n  while (n > 0) {\n";
  for (int i = 0; i < 300; ++i) nested += "    n = reduce(n, " + std::to_string(i) + ");\n";
  nested += "  }\n}\n";
  const ParsedFile a = parse_source("nested.c", nested, options);
  REQUIRE(check_partition(nested, a, "nested.c", options));
  REQUIRE(a.units.size() >= 4u);
  CHECK(a.units[0].kind == UnitKind::Function);
  CHECK_EQ(a.units[0].name, std::string("nested"));
  for (std::size_t i = 0; i + 1 < a.units.size(); ++i) {
    CHECK_EQ(a.units[i + 1].name, "nested#" + std::to_string(i + 2));
    CHECK(ends_with(a.units[i].text, ");\n"));
  }

  // a table: cuts after the entries
  std::string table = "static const int table[] = {\n";
  for (int i = 0; i < 600; ++i) table += "  " + std::to_string(i * 7919) + ",\n";
  table += "};\n";
  const ParsedFile b = parse_source("table.c", table, options);
  REQUIRE(check_partition(table, b, "table.c", options));
  REQUIRE(b.units.size() >= 5u);
  for (std::size_t i = 0; i + 1 < b.units.size(); ++i) {
    CHECK(b.units[i].kind == UnitKind::Region);
    CHECK(ends_with(b.units[i].text, ",\n"));
  }

  // a line longer than any limit: cut between characters, never inside one
  std::string line = "const char* s = \"";
  for (int i = 0; i < 1500; ++i) line += "\xC3\xA9\xE2\x82\xAC";  // two and three byte sequences
  line += "\";\n";
  const ParsedFile c = parse_source("line.c", line, options);
  REQUIRE(check_partition(line, c, "line.c", options));
  CHECK(c.units.size() >= line.size() / options.max_region_bytes);
  for (const SourceUnit& unit : c.units) CHECK_EQ(text::sanitize_utf8(unit.text), unit.text);

  // a type that is too large
  std::string type = "// Many fields.\ntypedef struct wide_s {\n";
  for (int i = 0; i < 200; ++i) type += "  unsigned long field_" + std::to_string(i) + ";\n";
  type += "} wide_t;\n";
  const ParsedFile d = parse_source("wide.h", type, options);
  REQUIRE(check_partition(type, d, "wide.h", options));
  REQUIRE(d.units.size() >= 2u);
  CHECK(d.units[0].kind == UnitKind::Type);
  CHECK_EQ(d.units[0].name, std::string("wide_t"));
  CHECK_EQ(d.units[0].doc, std::string("Many fields."));
  CHECK(d.units[1].kind == UnitKind::Region);
  CHECK_EQ(d.units[1].name, std::string("wide_t#2"));
  CHECK(ends_with(d.units[0].text, ";\n"));

  // Python: between the statements of the body
  std::string python = "def long(x):\n    \"\"\"Long.\"\"\"\n";
  for (int i = 0; i < 200; ++i) python += "    x = step(x, " + std::to_string(i) + ")\n";
  python += "    return x\n";
  const ParsedFile e = parse_source("long.py", python, options);
  REQUIRE(check_partition(python, e, "long.py", options));
  REQUIRE(e.units.size() >= 3u);
  CHECK(e.units[0].kind == UnitKind::Function);
  CHECK_EQ(e.units[0].name, std::string("long"));
  CHECK_EQ(e.units[0].doc, std::string("Long."));
  for (std::size_t i = 0; i + 1 < e.units.size(); ++i) {
    CHECK_EQ(e.units[i + 1].name, "long#" + std::to_string(i + 2));
    CHECK(e.units[i + 1].kind == UnitKind::Region);
    CHECK(starts_with(e.units[i + 1].text, "    x = step(x, "));
  }
}

PRAG_TEST(residual_text_is_packed_into_regions) {
  std::string text = "/* Prototypes. */\n#pragma once\n\n";
  for (int i = 0; i < 400; ++i) {
    if (i % 25 == 0) text += "\n// group " + std::to_string(i / 25) + "\n";
    text += "int function_number_" + std::to_string(i) + "(int first, int second);\n";
  }
  const ParseOptions options;
  const ParsedFile file = parse_source("protos.h", text);
  REQUIRE(check_partition(text, file, "protos.h"));
  const std::size_t needed = (text.size() + options.max_region_bytes - 1) / options.max_region_bytes;
  CHECK(file.units.size() >= needed);
  CHECK(file.units.size() <= needed + 1);
  for (const SourceUnit& unit : file.units) {
    CHECK(unit.kind == UnitKind::Region);
    CHECK(unit.text.size() <= options.max_region_bytes);
    CHECK(unit.text.size() >= options.max_region_bytes / 2);  // about equal sizes
    CHECK(ends_with(unit.text, ");\n"));                       // whole declarations
    CHECK(unit.callees.empty());                                // a prototype is not a call
  }
}

PRAG_TEST(syntax_errors_are_counted) {
  const std::string good = "int add(int a, int b) { return a + b; }\n";
  const std::string bad = "int add(int a, int b) { return a + ; }\nint sub(int a int b) { return a - b; }\n";
  CHECK_EQ(count_syntax_errors("c", good), std::size_t{0});
  CHECK_EQ(count_syntax_errors("cpp", good), std::size_t{0});
  CHECK(count_syntax_errors("c", bad) >= 2u);
  CHECK(count_syntax_errors("cpp", bad) >= 2u);
  CHECK(count_syntax_errors("c", "int f(void) { return 1;\n") >= 1u);  // the brace is missing
  CHECK_EQ(count_syntax_errors("c", ""), std::size_t{0});
  CHECK_EQ(count_syntax_errors("python", kPython), std::size_t{0});
  CHECK(count_syntax_errors("python", "def f(:\n    return 1\n") >= 1u);
  CHECK_EQ(count_syntax_errors("cpp", kCpp), std::size_t{0});
  // no grammar, no count
  CHECK_EQ(count_syntax_errors("", bad), std::size_t{0});
  CHECK_EQ(count_syntax_errors("rust", bad), std::size_t{0});
  // what the edit engine relies on: damaging a file raises the count
  const std::size_t before = count_syntax_errors("c", kC);
  std::string damaged(kC);
  damaged.replace(damaged.find("return p;"), 9, "return p +;");
  CHECK(count_syntax_errors("c", damaged) > before);
  // deterministic
  CHECK_EQ(count_syntax_errors("c", kC), before);

  CHECK(parse_source("good.c", good).had_syntax_errors == false);
  CHECK(parse_source("bad.c", bad).had_syntax_errors == true);
}

PRAG_TEST(find_unit_returns_the_range_of_a_name) {
  const std::string_view text = kC;
  std::size_t begin = 1;
  std::size_t end = 1;
  REQUIRE(find_unit("sample.c", text, "mi_malloc", &begin, &end));
  const std::string_view found = text.substr(begin, end - begin);
  CHECK(starts_with(found, "\n\n// Allocate `size` bytes.\n"));  // leading comment included
  CHECK(ends_with(found, "  return p;\n}\n"));
  const ParsedFile file = parse_source("sample.c", text);
  const SourceUnit* unit = find_unit_named(file, "mi_malloc", UnitKind::Function);
  REQUIRE(unit != nullptr);
  CHECK_EQ(begin, static_cast<std::size_t>(unit->start_byte));
  CHECK_EQ(end, static_cast<std::size_t>(unit->end_byte));

  // types and macros have names as well
  REQUIRE(find_unit("sample.c", text, "mi_page_t", &begin, &end));
  CHECK(has(text.substr(begin, end - begin), "typedef struct mi_page_s {"));
  REQUIRE(find_unit("sample.c", text, "mi_align_up", &begin, &end));
  CHECK(has(text.substr(begin, end - begin), "#define mi_align_up(x, a)"));

  // unknown names, and names that only occur inside other units
  begin = end = 7;
  CHECK(!find_unit("sample.c", text, "mi_heap_malloc", &begin, &end));
  CHECK(!find_unit("sample.c", text, "", &begin, &end));
  CHECK(!find_unit("sample.c", text, "MI_KiB", &begin, &end));
  CHECK_EQ(begin, std::size_t{7});  // untouched on failure
  CHECK_EQ(end, std::size_t{7});
  CHECK(find_unit("sample.c", text, "free", nullptr, nullptr));

  // several units of one name: the first function definition wins
  const std::string twice = "#define get(x) (x)\nstruct get { int a; };\nint get(void) { return 1; }\n"
                            "#if X\nint get(void) { return 2; }\n#endif\n";
  ParseOptions options;
  REQUIRE(find_unit("twice.c", twice, "get", &begin, &end));
  CHECK_EQ(std::string(twice.substr(begin, end - begin)), std::string("int get(void) { return 1; }\n"));

  // a broken file: what can be delimited is found
  const std::string broken = "int first(void) { return 1; }\n\nint second(int x { return x +; }\n\n"
                             "static int third(void)\n{\n  return 3;\n}\n\nint fourth(void) {\n  return 4;\n";
  CHECK(parse_source("broken.c", broken).had_syntax_errors);
  REQUIRE(find_unit("broken.c", broken, "first", &begin, &end));
  CHECK_EQ(std::string(broken.substr(begin, end - begin)), std::string("int first(void) { return 1; }\n"));
  REQUIRE(find_unit("broken.c", broken, "third", &begin, &end));
  CHECK_EQ(std::string(broken.substr(begin, end - begin)),
           std::string("\nstatic int third(void)\n{\n  return 3;\n}\n"));
  REQUIRE(find_unit("broken.c", broken, "fourth", &begin, &end));  // runs to the end of the file
  CHECK_EQ(end, broken.size());

  // C++: a class covers its members, a member is found by its own name
  const std::string_view cpp = kCpp;
  REQUIRE(find_unit("sample.cpp", cpp, "Shape", &begin, &end));
  CHECK(has(cpp.substr(begin, end - begin), "/* Shapes have an area. */\nclass Shape {"));
  CHECK(ends_with(cpp.substr(begin, end - begin), "  int id_;\n};\n"));
  CHECK(!has(cpp.substr(begin, end - begin), "class Circle"));
  REQUIRE(find_unit("sample.cpp", cpp, "Shape::name", &begin, &end));
  CHECK(has(cpp.substr(begin, end - begin), "const std::string& name() const { return name_; }"));
  REQUIRE(find_unit("sample.cpp", cpp, "area", &begin, &end));
  CHECK(has(cpp.substr(begin, end - begin), "double Circle::area() const {"));
  REQUIRE(find_unit("sample.cpp", cpp, "largest", &begin, &end));
  CHECK(has(cpp.substr(begin, end - begin), "// Largest of two values.\ntemplate <typename T>"));

  // Python
  const std::string_view python = kPython;
  REQUIRE(find_unit("sample.py", python, "Stack.push", &begin, &end));
  CHECK(has(python.substr(begin, end - begin), "def push(self, item):"));
  CHECK(!has(python.substr(begin, end - begin), "def top"));
  REQUIRE(find_unit("sample.py", python, "drain", &begin, &end));
  CHECK(has(python.substr(begin, end - begin), "async def drain(self):"));
  REQUIRE(find_unit("sample.py", python, "Stack", &begin, &end));
  CHECK(starts_with(python.substr(begin, end - begin), "\n\nclass Stack:"));
  CHECK(ends_with(python.substr(begin, end - begin), "yield self.items.pop()\n"));
}

PRAG_TEST(broken_sources_are_still_partitioned) {
  const std::vector<std::string> sources = {
      "",
      "\n",
      "   \n\t\n",
      "}",
      "{",
      "}}}}{{{{",
      ";;;",
      "int f(void) {",
      "int f(void) { return 1; } }",
      "/* never closed\nint f(void) { return 1; }\n",
      "const char* s = \"never closed\nint f(void) { return 1; }\n",
      "char c = 'x\nint f(void) { return 1; }\n",
      "#if A\nint f(void) {\n#elif B\nint f(int a) {\n#else\nint f(long a) {\n",
      "#endif\n#endif\n#else\nint f(void) { return 1; }\n",
      "#define OPEN {\n#define CLOSE }\nint f(void) OPEN return 1; CLOSE\n",
      "int f(int a, int b { return a; }\nint g(void) { return 2; }\n",
      "struct s { int a;\nint f(void) { return 1; }\n",
      "int f(void) { return 1; }\n#",
      "#define A \\",
      "\xEF\xBB\xBFint f(void) { return 1; }\n",
      std::string("int f(void) { return 1; }") + '\0' + "int g(void) { return 2; }\n",
      "int \xFF\xFE(void) { return 1; }\n",
      "class A { public: void f() { } int x; ",
      "namespace a { namespace b { void f() {} } ",
      "void f() try { } catch (...) { }\n",
      "template <class T> struct S { void f(); }; template <class T> void S<T>::f() {}\n",
      "R\"x(raw { string )x\" int f(void) { return 1; }\n",
      "int a = 1'000'000; int f(void) { return 'a'; }\n",
  };
  for (std::size_t i = 0; i < sources.size(); ++i) {
    for (const char* path : {"x.c", "x.cpp", "x.h", "x.py", "x.txt"}) {
      const ParsedFile file = parse_source(path, sources[i]);
      check_partition(sources[i], file, std::string(path) + " source " + std::to_string(i));
    }
  }
  // a function after a stretch of nonsense is still found
  const ParsedFile file = parse_source("x.c", "int f(int a, int b { return a; }\n\nint g(void) { return 2; }\n");
  CHECK(find_unit_named(file, "g", UnitKind::Function) != nullptr);
  // the byte order mark is part of the first unit
  const ParsedFile marked = parse_source("x.c", "\xEF\xBB\xBFint f(void) { return 1; }\n");
  REQUIRE(marked.units.size() == 1u);
  CHECK(marked.units[0].kind == UnitKind::Function);
  CHECK_EQ(marked.units[0].name, std::string("f"));
}

PRAG_TEST(damaged_samples_are_still_partitioned) {
  // Deterministic damage: cut the samples short, and drop or insert characters
  // that carry structure.
  const char* const insertions[] = {"{", "}", "(", ")", "\"", "'", "/*", "*/", "#if X\n", "#else\n", "#endif\n", ";", "\\\n"};
  const struct {
    const char* path;
    std::string_view text;
  } samples[] = {{"sample.c", kC}, {"sample.cpp", kCpp}, {"sample.py", kPython}};
  std::uint64_t state = 0x2545F4914F6CDD1DULL;
  const auto next = [&state] {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  std::size_t parsed = 0;
  for (const auto& sample : samples) {
    for (std::size_t cut = 0; cut <= sample.text.size(); cut += 37) {
      const std::string text(sample.text.substr(0, cut));
      check_partition(text, parse_source(sample.path, text), std::string(sample.path) + " cut at " + std::to_string(cut));
      ++parsed;
    }
    for (int round = 0; round < 120; ++round) {
      std::string text(sample.text);
      for (int edit = 0; edit < 3; ++edit) {
        const std::size_t at = static_cast<std::size_t>(next() % text.size());
        if (next() % 2 == 0) {
          text.erase(at, 1 + static_cast<std::size_t>(next() % 3));
        } else {
          text.insert(at, insertions[next() % (sizeof(insertions) / sizeof(insertions[0]))]);
        }
      }
      check_partition(text, parse_source(sample.path, text),
                      std::string(sample.path) + " damage round " + std::to_string(round));
      ++parsed;
    }
  }
  CHECK(parsed > 400u);
}

PRAG_TEST(text_without_grammar_becomes_line_regions) {
  std::string text = "Title\n=====\n\n";
  for (int i = 0; i < 300; ++i) {
    text += "Paragraph " + std::to_string(i) + " has a first line\nand a second line.\n";
    if (i % 3 == 2) text += "\n";
  }
  const ParseOptions options;
  const ParsedFile file = parse_source("notes.md", text);
  CHECK_EQ(file.language, std::string(""));
  CHECK(!file.had_syntax_errors);
  REQUIRE(check_partition(text, file, "notes.md"));
  CHECK(file.units.size() >= text.size() / options.max_region_bytes);
  for (const SourceUnit& unit : file.units) {
    CHECK(unit.kind == UnitKind::Region);
    CHECK(unit.name.empty());
    CHECK(unit.language.empty());
    CHECK(unit.callees.empty());
    CHECK(ends_with(unit.text, "\n"));  // line-aligned
  }
  // regions end where paragraphs end
  for (std::size_t i = 0; i + 1 < file.units.size(); ++i) CHECK(ends_with(file.units[i].text, "and a second line.\n"));
}

PRAG_TEST(empty_and_blank_files) {
  for (const char* path : {"x.c", "x.cpp", "x.py", "x.txt"}) {
    const ParsedFile empty = parse_source(path, "");
    CHECK(empty.units.empty());
    CHECK(!empty.had_syntax_errors);
    CHECK(!empty.path.empty());
    const std::string blank = "\n\n   \n";
    const ParsedFile file = parse_source(path, blank);
    REQUIRE(check_partition(blank, file, path));
    REQUIRE(file.units.size() == 1u);
    CHECK(file.units[0].kind == UnitKind::Region);
    CHECK_EQ(file.units[0].start_line, 1u);
    CHECK_EQ(file.units[0].end_line, 3u);
  }
  // a comment alone is a region, a file without final newline is complete
  const std::string text = "// only a comment";
  const ParsedFile file = parse_source("x.c", text);
  REQUIRE(check_partition(text, file, "x.c"));
  REQUIRE(file.units.size() == 1u);
  CHECK_EQ(file.units[0].end_line, 1u);
}

PRAG_TEST(paths_are_normalised) {
  const ParsedFile file = parse_source("some\\dir\\..\\file.c", "int f(void) { return 1; }\n");
  CHECK(!has(file.path, "\\"));
  CHECK(!has(file.path, ".."));
  CHECK(ends_with(file.path, "/some/file.c"));
  REQUIRE(file.units.size() == 1u);
  CHECK_EQ(file.units[0].path, file.path);
}

PRAG_TEST(line_endings_do_not_matter) {
  const std::string text = with_crlf(kC);
  const ParsedFile dos = parse_source("sample.c", text);
  const ParsedFile unix_file = parse_source("sample.c", kC);
  REQUIRE(check_partition(text, dos, "sample.c with CRLF"));
  REQUIRE(dos.units.size() == unix_file.units.size());
  for (std::size_t i = 0; i < dos.units.size(); ++i) {
    const SourceUnit& a = dos.units[i];
    const SourceUnit& b = unix_file.units[i];
    CHECK(a.kind == b.kind);
    CHECK_EQ(a.name, b.name);
    CHECK_EQ(a.signature, b.signature);
    CHECK_EQ(a.doc, b.doc);
    CHECK_EQ(a.start_line, b.start_line);
    CHECK_EQ(a.end_line, b.end_line);
    CHECK_EQ(a.text, with_crlf(b.text));
    CHECK(a.callees == b.callees);
  }
  const std::string python = with_crlf(kPython);
  const ParsedFile file = parse_source("sample.py", python);
  REQUIRE(check_partition(python, file, "sample.py with CRLF"));
  CHECK_EQ(names_of(file, UnitKind::Function).size(), std::size_t{8});
}

PRAG_TEST(conditional_branches_are_counted_once) {
  // Every branch opens the block; the code after the conditional closes it once.
  const std::string text =
      "int before(void) { return 0; }\n"
      "\n"
      "void loop(int n) {\n"
      "#if defined(FAST)\n"
      "  for (int i = 0; i < n; i += 2) {\n"
      "#elif defined(SAFE)\n"
      "  for (int i = 0; i < n && check(i); i++) {\n"
      "#else\n"
      "  for (int i = 0; i < n; i++) {\n"
      "#endif\n"
      "    work(i);\n"
      "  }\n"
      "}\n"
      "\n"
      "#ifdef A\n"
      "int chosen(void) { return 1; }\n"
      "#else\n"
      "int chosen(void) { return 2; }\n"
      "#endif\n"
      "\n"
      "int after(void) { return 3; }\n";
  const ParsedFile file = parse_source("branches.c", text);
  REQUIRE(check_partition(text, file, "branches.c"));
  const std::vector<std::string> expected = {"before", "loop", "chosen", "chosen", "after"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);
  const SourceUnit* loop = find_unit_named(file, "loop", UnitKind::Function);
  REQUIRE(loop != nullptr);
  CHECK(ends_with(loop->text, "    work(i);\n  }\n}\n"));
  CHECK(calls(*loop, "work"));
  CHECK_EQ(file.units.size(), std::size_t{5});  // the directives went to the functions they enclose
}

PRAG_TEST(containers_are_transparent) {
  const std::string text =
      "namespace outer {\n"
      "namespace {\n"
      "\n"
      "int hidden() { return 1; }\n"
      "\n"
      "}  // namespace\n"
      "\n"
      "inline namespace v1 {\n"
      "struct Box {\n"
      "  int get() const { return value; }\n"
      "  struct Lid {\n"
      "    bool open() { return true; }\n"
      "  };\n"
      "  int value;\n"
      "};\n"
      "}\n"
      "\n"
      "extern \"C\" {\n"
      "int c_function(void) { return hidden(); }\n"
      "}\n"
      "\n"
      "extern \"C\" int single(void) { return 2; }\n"
      "\n"
      "}  // namespace outer\n";
  const ParsedFile file = parse_source("containers.cpp", text);
  REQUIRE(check_partition(text, file, "containers.cpp"));
  const std::vector<std::string> expected = {"hidden", "Box::get", "Box::Lid::open", "c_function", "single"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);
  CHECK(find_unit_named(file, "Box", UnitKind::Type) != nullptr);
  CHECK(find_unit_named(file, "Box::Lid", UnitKind::Type) != nullptr);
  const SourceUnit* c_function = find_unit_named(file, "c_function", UnitKind::Function);
  REQUIRE(c_function != nullptr);
  CHECK(calls(*c_function, "hidden"));
}

PRAG_TEST(access_specifiers_go_to_the_head_or_to_the_member_below) {
  const std::string text =
      "class Gate {\n"
      " public:\n"
      "  int open() { return 1; }\n"
      "\n"
      " private:\n"
      "  int close() { return 2; }\n"
      "  int state_;\n"
      "};\n"
      "\n"
      "class Guarded {\n"
      "#if FEATURE\n"
      "  int fast() { return 1; }\n"
      "#endif\n"
      "  int slow() { return 2; }\n"
      "};\n";
  const ParsedFile file = parse_source("access.cpp", text);
  REQUIRE(check_partition(text, file, "access.cpp"));
  const std::vector<std::string> expected = {"Gate::open", "Gate::close", "Guarded::fast", "Guarded::slow"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);

  // what precedes the first member function belongs to the head of the type
  const SourceUnit* gate = find_unit_named(file, "Gate", UnitKind::Type);
  REQUIRE(gate != nullptr);
  CHECK_EQ(gate->text, std::string("class Gate {\n public:\n"));
  const SourceUnit* open = find_unit_named(file, "Gate::open", UnitKind::Function);
  REQUIRE(open != nullptr);
  CHECK_EQ(open->text, std::string("  int open() { return 1; }\n"));
  // between two members the specifier opens the section of the member below it
  const SourceUnit* close = find_unit_named(file, "Gate::close", UnitKind::Function);
  REQUIRE(close != nullptr);
  CHECK_EQ(close->text, std::string("\n private:\n  int close() { return 2; }\n"));

  // a conditional stays with the member it encloses, directly below the head as well
  const SourceUnit* guarded = find_unit_named(file, "Guarded", UnitKind::Type);
  REQUIRE(guarded != nullptr);
  CHECK_EQ(guarded->text, std::string("\nclass Guarded {\n"));
  const SourceUnit* fast = find_unit_named(file, "Guarded::fast", UnitKind::Function);
  REQUIRE(fast != nullptr);
  CHECK_EQ(fast->text, std::string("#if FEATURE\n  int fast() { return 1; }\n#endif\n"));
  // the brace that closes the class goes to the unit above it
  const SourceUnit* slow = find_unit_named(file, "Guarded::slow", UnitKind::Function);
  REQUIRE(slow != nullptr);
  CHECK(ends_with(slow->text, "  int slow() { return 2; }\n};\n"));
}

PRAG_TEST(macro_made_definitions_are_named_after_their_arguments) {
  const std::string text =
      "PRAG_TEST(first_case) {\n  CHECK(run(1));\n}\n\n"
      "TEST(Suite, Second) { EXPECT_TRUE(run(2)); }\n\n"
      "TEST_F(Fixture, Third)\n{\n  EXPECT_TRUE(run(3));\n}\n";
  const ParsedFile file = parse_source("cases.cpp", text);
  REQUIRE(check_partition(text, file, "cases.cpp"));
  const std::vector<std::string> expected = {"first_case", "Suite_Second", "Fixture_Third"};
  const std::vector<std::string> functions = names_of(file, UnitKind::Function);
  REQUIRE(functions.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(functions[i], expected[i]);
  CHECK(calls(file.units[0], "PRAG_TEST"));
  CHECK(calls(file.units[0], "run"));
}

PRAG_TEST(call_graph_resolves_names_over_the_corpus) {
  // two files; ids are made dense the way the index does it
  const std::string a =
      "#define twice(x) helper((x) * 2)\n"
      "#define LIMIT 3\n"
      "static int helper(int x) { return x + LIMIT; }\n"
      "int entry(int x) { return twice(x) + other(x) + missing(x); }\n";
  const std::string b =
      "static int helper(int x) { return -x; }\n"
      "int other(int x) { return helper(x) + other(x - 1); }\n";
  std::vector<SourceUnit> units;
  const std::vector<std::pair<std::string, const std::string*>> files = {{"a.c", &a}, {"b.c", &b}};
  for (const auto& [path, text] : files) {
    const ParsedFile file = parse_source(path, *text);
    REQUIRE(check_partition(*text, file, path));
    for (SourceUnit unit : file.units) {
      unit.id = static_cast<UnitId>(units.size());
      units.push_back(std::move(unit));
    }
  }
  const auto id_of = [&](std::string_view path, std::string_view name) {
    for (const SourceUnit& unit : units) {
      if (unit.name == name && ends_with(unit.path, path)) return unit.id;
    }
    return kNoUnit;
  };
  const UnitId twice = id_of("a.c", "twice");
  const UnitId helper_a = id_of("a.c", "helper");
  const UnitId helper_b = id_of("b.c", "helper");
  const UnitId entry = id_of("a.c", "entry");
  const UnitId other = id_of("b.c", "other");
  REQUIRE(twice != kNoUnit);
  REQUIRE(helper_a != kNoUnit);
  REQUIRE(helper_b != kNoUnit);
  REQUIRE(entry != kNoUnit);
  REQUIRE(other != kNoUnit);

  const graph::CallGraph g = build_call_graph(units);
  CHECK_EQ(g.size(), units.size());
  const auto has_edge = [&](UnitId from, UnitId to) {
    const auto callees = g.callees(from);
    const auto callers = g.callers(to);
    return std::find(callees.begin(), callees.end(), to) != callees.end() &&
           std::find(callers.begin(), callers.end(), from) != callers.end();
  };
  // entry calls the macro and the function of the other file; what is not defined gives no edge
  CHECK(has_edge(entry, twice));
  CHECK(has_edge(entry, other));
  CHECK_EQ(g.callees(entry).size(), std::size_t{2});
  // a name with two definitions: one edge per definition
  CHECK(has_edge(twice, helper_a));
  CHECK(has_edge(twice, helper_b));
  CHECK(has_edge(other, helper_a));
  CHECK(has_edge(other, helper_b));
  // recursion is not an edge
  CHECK(!has_edge(other, other));
  CHECK_EQ(g.edges(), std::size_t{6});
  CHECK_EQ(g.callers(helper_a).size(), std::size_t{2});
  CHECK(g.callers(entry).empty());

  REQUIRE(g.definitions("helper").size() == 2u);
  CHECK_EQ(g.definitions("helper")[0], helper_a);
  CHECK_EQ(g.definitions("helper")[1], helper_b);
  CHECK_EQ(g.definitions("twice").size(), std::size_t{1});
  CHECK(g.definitions("LIMIT").empty());    // not function-like
  CHECK(g.definitions("missing").empty());

  CHECK_EQ(build_call_graph({}).size(), std::size_t{0});
}

PRAG_TEST(call_graph_reaches_members) {
  ParsedFile cpp = parse_source("sample.cpp", kCpp);
  graph::CallGraph g = build_call_graph(cpp.units);
  const SourceUnit* main_unit = find_unit_named(cpp, "main", UnitKind::Function);
  const SourceUnit* area = find_unit_named(cpp, "Circle::area", UnitKind::Function);
  const SourceUnit* square = find_unit_named(cpp, "SQUARE", UnitKind::Macro);
  const SourceUnit* repeat = find_unit_named(cpp, "repeat", UnitKind::Function);
  const SourceUnit* largest = find_unit_named(cpp, "largest", UnitKind::Function);
  REQUIRE(main_unit != nullptr && area != nullptr && square != nullptr && repeat != nullptr && largest != nullptr);
  const auto has_edge = [&](const SourceUnit* from, const SourceUnit* to) {
    const auto callees = g.callees(from->id);
    return std::find(callees.begin(), callees.end(), to->id) != callees.end();
  };
  CHECK(has_edge(main_unit, area));  // c.area() reaches Circle::area
  CHECK(has_edge(area, square));
  CHECK(has_edge(repeat, largest));
  REQUIRE(g.definitions("area").size() == 1u);
  CHECK_EQ(g.definitions("Circle::area").size(), std::size_t{1});

  const ParsedFile python = parse_source("sample.py", kPython);
  g = build_call_graph(python.units);
  const SourceUnit* py_main = find_unit_named(python, "main", UnitKind::Function);
  const SourceUnit* push = find_unit_named(python, "Stack.push", UnitKind::Function);
  const SourceUnit* of = find_unit_named(python, "Stack.of", UnitKind::Function);
  const SourceUnit* init = find_unit_named(python, "Stack.__init__", UnitKind::Function);
  const SourceUnit* fib = find_unit_named(python, "fib", UnitKind::Function);
  const SourceUnit* cached = find_unit_named(python, "cached", UnitKind::Function);
  REQUIRE(py_main != nullptr && push != nullptr && of != nullptr && init != nullptr && fib != nullptr &&
          cached != nullptr);
  CHECK(has_edge(py_main, push));
  CHECK(has_edge(py_main, fib));
  CHECK(has_edge(fib, cached));  // the decorator
  CHECK(has_edge(of, init));     // Stack(items) constructs
  CHECK(g.callees(python.units.back().id).size() == 1u);  // the module calls main
}

PRAG_TEST(parsing_is_deterministic) {
  const ParsedFile a = parse_source("sample.c", kC);
  const ParsedFile b = parse_source("sample.c", kC);
  CHECK(parse_checks::same_units(a, b));
  const ParsedFile c = parse_source("sample.py", kPython);
  const ParsedFile d = parse_source("sample.py", kPython);
  CHECK(parse_checks::same_units(c, d));
  CHECK(!parse_checks::same_units(a, c));
}

PRAG_TEST(limits_of_zero_are_survived) {
  ParseOptions options;
  options.max_region_bytes = 0;
  options.max_unit_bytes = 0;
  const ParsedFile file = parse_source("sample.c", kC, options);
  CHECK(!file.units.empty());
  std::string joined;
  for (const SourceUnit& unit : file.units) joined += unit.text;
  CHECK_EQ(joined, std::string(kC));
}
