// Minimal self-registering unit test harness (no external dependency).
//
//   PRAG_TEST(name) { CHECK(cond); CHECK_EQ(a, b); CHECK_NEAR(a, b, tol); REQUIRE(cond); }
//   int main(int argc, char** argv) { return prag::test::run_all(argc, argv); }   // or define PRAG_TEST_MAIN
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace prag::test {

struct Case {
  const char* name;
  std::function<void()> body;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

struct Failure : std::exception {
  const char* what() const noexcept override { return "REQUIRE failed"; }
};

inline int& failures() {
  static int count = 0;
  return count;
}

inline void report(const char* file, int line, const std::string& message) {
  ++failures();
  std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, message.c_str());
}

template <class A, class B>
std::string describe(const char* expr_a, const char* expr_b, const A& a, const B& b, const char* op) {
  std::ostringstream os;
  os << expr_a << ' ' << op << ' ' << expr_b << "  (" << a << " vs " << b << ')';
  return os.str();
}

// Runs every registered case whose name contains the optional filter argument.
inline int run_all(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  int failed_cases = 0;
  for (const Case& c : registry()) {
    if (filter && !std::strstr(c.name, filter)) continue;
    const int before = failures();
    try {
      c.body();
    } catch (const Failure&) {
    } catch (const std::exception& e) {
      report(c.name, 0, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      report(c.name, 0, "unexpected non-standard exception");
    }
    ++ran;
    const bool ok = failures() == before;
    if (!ok) ++failed_cases;
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", c.name);
  }
  std::printf("%d case(s) run, %d failed\n", ran, failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace prag::test

#define PRAG_TEST(name)                                                        \
  static void prag_test_##name();                                              \
  static ::prag::test::Registrar prag_registrar_##name(#name, prag_test_##name); \
  static void prag_test_##name()

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) ::prag::test::report(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define REQUIRE(cond)                                                       \
  do {                                                                      \
    if (!(cond)) {                                                          \
      ::prag::test::report(__FILE__, __LINE__, "REQUIRE(" #cond ")");       \
      throw ::prag::test::Failure();                                        \
    }                                                                       \
  } while (0)

#define CHECK_EQ(a, b)                                                                       \
  do {                                                                                       \
    const auto& prag_a = (a);                                                                \
    const auto& prag_b = (b);                                                                \
    if (!(prag_a == prag_b))                                                                 \
      ::prag::test::report(__FILE__, __LINE__, ::prag::test::describe(#a, #b, prag_a, prag_b, "==")); \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                \
  do {                                                                                       \
    const double prag_a = static_cast<double>(a);                                            \
    const double prag_b = static_cast<double>(b);                                            \
    if (!(std::fabs(prag_a - prag_b) <= static_cast<double>(tol)))                           \
      ::prag::test::report(__FILE__, __LINE__, ::prag::test::describe(#a, #b, prag_a, prag_b, "~=")); \
  } while (0)

#ifdef PRAG_TEST_MAIN
int main(int argc, char** argv) { return ::prag::test::run_all(argc, argv); }
#endif
