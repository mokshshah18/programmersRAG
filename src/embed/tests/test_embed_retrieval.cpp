// Realistic check of the hashing embedder that needs no model: known-item
// retrieval of mimalloc functions. The query is the name of a function split
// into words ("mi heap malloc"); the documents are about 200 functions of the
// corpus. Two document forms are measured:
//
//   body      the text between the braces only, so the name itself is absent
//             and the words of the name must be found in what the function does;
//   rendered  render_for_embedding of the unit (path, name, signature, body),
//             which is the form the index stores.
//
// The measured fractions are printed; the assertions are loose lower bounds.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "prag/core/fs.hpp"
#include "prag/core/text.hpp"
#include "prag/embed/hashing.hpp"

using namespace prag;
using namespace prag::embed;

namespace {

struct Function {
  std::string path;
  std::string name;
  std::string signature;  // the declaration on one line
  std::string body;       // from the opening to the closing brace
  std::string raw;        // the lines of the file, from the declaration to the closing brace
};

bool is_name_start(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }
bool is_name_byte(char c) { return is_name_start(c) || (c >= '0' && c <= '9'); }

// Function definitions in the layout mimalloc uses throughout: the declaration
// starts in column 0 and ends with '{' at the end of a line (at most
// `kHeaderLines` lines later), and the body ends at the next '}' in column 0.
// This is the line-based equivalent of the regular expression
//   ^[A-Za-z_][^;={}#]*\b(\w+)\s*\([^;={}]*\{\s*$   ...   ^\}
std::vector<Function> find_functions(const std::string& path, std::string_view source) {
  constexpr std::size_t kHeaderLines = 6;
  static const std::set<std::string_view> kNotAName = {"if", "for", "while", "switch", "do", "else", "return",
                                                       "sizeof", "defined"};
  std::vector<Function> functions;
  const std::vector<std::string_view> lines = text::split_lines(source);
  std::size_t i = 0;
  while (i < lines.size()) {
    if (lines[i].empty() || !is_name_start(lines[i][0])) {
      ++i;
      continue;
    }
    std::string header;
    std::size_t open = i;
    bool found = false;
    for (; open < lines.size() && open < i + kHeaderLines; ++open) {
      std::string_view line = lines[open];
      if (open > i && (line.empty() || line[0] == '#' || line[0] == '}')) break;
      if (const std::size_t comment = line.find("//"); comment != std::string_view::npos) line = line.substr(0, comment);
      line = text::trim(line);
      if (line.empty()) continue;
      if (!header.empty()) header.push_back(' ');
      header.append(line);
      if (line.back() == ';' || line.back() == '}') break;
      if (line.back() == '{') {
        found = true;
        break;
      }
    }
    if (!found) {
      ++i;
      continue;
    }
    header.pop_back();  // the brace
    const std::size_t paren = header.find('(');
    std::size_t name_end = paren == std::string::npos ? 0 : paren;
    while (name_end > 0 && header[name_end - 1] == ' ') --name_end;
    std::size_t name_begin = name_end;
    while (name_begin > 0 && is_name_byte(header[name_begin - 1])) --name_begin;
    const std::string name = header.substr(name_begin, name_end - name_begin);

    std::size_t close = open + 1;
    while (close < lines.size() && !(!lines[close].empty() && lines[close][0] == '}')) ++close;
    const bool is_function = paren != std::string::npos && header.find('=') == std::string::npos && !name.empty() &&
                             is_name_start(name[0]) && kNotAName.count(name) == 0 && close < lines.size();
    if (!is_function) {
      ++i;
      continue;
    }
    Function f;
    f.path = path;
    f.name = name;
    f.signature = std::string(text::trim(header));
    f.body = "{\n";
    for (std::size_t k = open + 1; k <= close; ++k) {
      f.body.append(lines[k]);
      f.body.push_back('\n');
    }
    for (std::size_t k = i; k <= close; ++k) {
      f.raw.append(lines[k]);
      f.raw.push_back('\n');
    }
    functions.push_back(std::move(f));
    i = close + 1;
  }
  return functions;
}

// About `wanted` functions with distinct names, evenly spread over the corpus
// in path and source order.
std::vector<Function> sample_functions(std::size_t wanted) {
  fs::WalkOptions options;
  options.extensions = {".c"};
  std::vector<Function> all;
  std::set<std::string> seen;
  for (const std::string& path : fs::walk(std::string(PRAG_DEMO_REPO) + "/src", options)) {
    const auto source = fs::read_file(path);
    if (!source) continue;
    for (Function& f : find_functions(path, *source)) {
      if (seen.insert(f.name).second) all.push_back(std::move(f));
    }
  }
  if (all.size() <= wanted) return all;
  std::vector<Function> sample;
  sample.reserve(wanted);
  for (std::size_t k = 0; k < wanted; ++k) sample.push_back(std::move(all[k * all.size() / wanted]));
  return sample;
}

std::string query_of(const Function& f) {
  std::string query;
  for (const std::string& word : text::split_identifier(f.name)) {
    if (!query.empty()) query.push_back(' ');
    query.append(word);
  }
  return query;
}

struct Recall {
  std::size_t queries = 0;
  double top1 = 0.0;
  double top5 = 0.0;
  double top10 = 0.0;
  double mrr = 0.0;
};

// Rank of document q for query q among all documents; equal scores rank in index order.
Recall known_item_recall(Embedder& embedder, const std::vector<std::string>& queries,
                         const std::vector<std::string>& documents) {
  const std::size_t n = queries.size();
  const std::size_t dim = embedder.dim();
  std::vector<std::string_view> views(documents.begin(), documents.end());
  std::vector<float> d(n * dim);
  embedder.embed(views, Role::Document, d.data());
  views.assign(queries.begin(), queries.end());
  std::vector<float> q(n * dim);
  embedder.embed(views, Role::Query, q.data());

  Recall recall;
  recall.queries = n;
  std::vector<double> scores(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double dot = 0.0;
      for (std::size_t k = 0; k < dim; ++k) dot += static_cast<double>(q[i * dim + k]) * d[j * dim + k];
      scores[j] = dot;
    }
    std::size_t rank = 1;
    for (std::size_t j = 0; j < n; ++j) {
      if (scores[j] > scores[i] || (scores[j] == scores[i] && j < i)) ++rank;
    }
    if (rank <= 1) recall.top1 += 1.0;
    if (rank <= 5) recall.top5 += 1.0;
    if (rank <= 10) recall.top10 += 1.0;
    recall.mrr += 1.0 / static_cast<double>(rank);
  }
  if (n > 0) {
    recall.top1 /= static_cast<double>(n);
    recall.top5 /= static_cast<double>(n);
    recall.top10 /= static_cast<double>(n);
    recall.mrr /= static_cast<double>(n);
  }
  return recall;
}

void print(const char* documents, const char* embedder, const Recall& r) {
  std::printf("  documents = %-8s  embedder = %-24s  n = %3zu  top1 %.3f  top5 %.3f  top10 %.3f  mrr %.3f\n", documents,
              embedder, r.queries, r.top1, r.top5, r.top10, r.mrr);
}

struct Corpus {
  std::vector<std::string> queries;   // the name, split into words
  std::vector<std::string> bodies;    // the text between the braces
  std::vector<std::string> rendered;  // render_for_embedding of the unit
};

Corpus make_corpus(const std::vector<Function>& functions) {
  Corpus corpus;
  for (const Function& f : functions) {
    corpus.queries.push_back(query_of(f));
    corpus.bodies.push_back(f.body);
    SourceUnit unit;
    unit.path = f.path;
    unit.language = "c";
    unit.kind = UnitKind::Function;
    unit.name = f.name;
    unit.signature = f.signature;
    unit.text = f.signature + " " + f.body;
    corpus.rendered.push_back(render_for_embedding(unit, kHashingMaxTokens));
  }
  return corpus;
}

// `s` without comments and with every run of white space as one blank.
std::string on_one_line(std::string_view s) {
  std::string out;
  bool blank = false;
  std::size_t i = 0;
  while (i < s.size()) {
    if (s.substr(i, 2) == "/*") {
      const std::size_t close = s.find("*/", i + 2);
      i = close == std::string_view::npos ? s.size() : close + 2;
      blank = true;
    } else if (s.substr(i, 2) == "//") {
      const std::size_t line_end = s.find('\n', i);
      i = line_end == std::string_view::npos ? s.size() : line_end;
      blank = true;
    } else if (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
      blank = true;
      ++i;
    } else {
      if (blank && !out.empty()) out.push_back(' ');
      blank = false;
      out.push_back(s[i++]);
    }
  }
  return out;
}

bool corpus_is_missing() {
  if (fs::is_directory(std::string(PRAG_DEMO_REPO) + "/src")) return false;
  std::printf("  skipped: no corpus at %s\n", PRAG_DEMO_REPO);
  return true;
}

}  // namespace

PRAG_TEST(function_finder_reads_the_mimalloc_layout) {
  const std::string source =
      "#include <stdio.h>\n"
      "static int counter = 0;\n"
      "static const int table[2] = {\n  1, 2\n};\n"
      "int declared(int x);\n"
      "\n"
      "// comment\n"
      "static inline void* mi_heap_malloc(mi_heap_t* heap, size_t size) mi_attr_noexcept {\n"
      "  if (size == 0) {\n    return NULL;\n  }\n"
      "  return mi_heap_malloc_small(heap, size);\n"
      "}\n"
      "\n"
      "bool _mi_page_is_valid(mi_page_t* page,\n"
      "                       void* arg)\n"
      "{\n"
      "  return page != NULL;\n"
      "}\n";
  const std::vector<Function> functions = find_functions("x.c", source);
  REQUIRE(functions.size() == 2u);
  CHECK_EQ(functions[0].name, std::string("mi_heap_malloc"));
  CHECK_EQ(functions[0].signature,
           std::string("static inline void* mi_heap_malloc(mi_heap_t* heap, size_t size) mi_attr_noexcept"));
  CHECK_EQ(functions[0].body,
           std::string("{\n  if (size == 0) {\n    return NULL;\n  }\n  return mi_heap_malloc_small(heap, size);\n}\n"));
  CHECK_EQ(functions[1].name, std::string("_mi_page_is_valid"));
  CHECK_EQ(functions[1].body, std::string("{\n  return page != NULL;\n}\n"));
  CHECK_EQ(query_of(functions[1]), std::string("mi page is valid"));
  CHECK_EQ(functions[1].signature, std::string("bool _mi_page_is_valid(mi_page_t* page, void* arg)"));
  CHECK_EQ(functions[1].raw,
           std::string("bool _mi_page_is_valid(mi_page_t* page,\n                       void* arg)\n{\n  return page != NULL;\n}\n"));
  CHECK_EQ(on_one_line("a  /* b */ c // d\n\te"), std::string("a c e"));
}

// The index stores the declaration as parse delivers it, on one line, while
// the text of the unit has it as the file does. A declaration that spans
// several lines must not be rendered a second time with the body.
PRAG_TEST(declarations_of_mimalloc_are_rendered_once) {
  if (corpus_is_missing()) return;
  const std::vector<Function> functions = sample_functions(static_cast<std::size_t>(-1));
  REQUIRE(functions.size() >= 200u);
  std::size_t checked = 0;
  std::size_t laid_out = 0;  // declarations that do not occur literally in the text
  std::size_t repeated = 0;
  std::size_t incomplete = 0;
  for (const Function& f : functions) {
    // The line scanner also finds code that was commented out, together with
    // the ends of the comment; that is not a declaration.
    if (f.signature.find("/*") != std::string::npos || f.signature.find("*/") != std::string::npos) continue;
    ++checked;
    SourceUnit unit;
    unit.path = f.path;
    unit.language = "c";
    unit.kind = UnitKind::Function;
    unit.name = f.name;
    unit.signature = on_one_line(f.signature);
    unit.text = f.raw;
    if (f.raw.find(unit.signature) == std::string::npos) ++laid_out;

    const std::string rendered = render_for_embedding(unit, static_cast<std::size_t>(-1));
    const std::size_t header = rendered.find(unit.signature);
    REQUIRE(header != std::string::npos);
    const std::string body = on_one_line(std::string_view(rendered).substr(header + unit.signature.size()));
    if (body.find(unit.signature) != std::string::npos) {
      if (++repeated <= 3) std::printf("  rendered twice: %s\n", unit.signature.c_str());
    }
    // Nothing of the body is lost on the way.
    if (!body.ends_with(on_one_line(f.body))) {
      if (++incomplete <= 3) std::printf("  body not complete: %s\n", unit.signature.c_str());
    }
  }
  std::printf("  %zu declarations, %zu of them laid out differently from their signature, %zu rendered twice\n", checked,
              laid_out, repeated);
  CHECK(checked + 5 >= functions.size());
  CHECK_EQ(repeated, std::size_t{0});
  CHECK_EQ(incomplete, std::size_t{0});
  if (laid_out == 0) std::printf("  note: the corpus has no declaration that spans several lines\n");
}

PRAG_TEST(function_names_retrieve_their_own_function_in_mimalloc) {
  if (corpus_is_missing()) return;
  const std::vector<Function> functions = sample_functions(200);
  REQUIRE(functions.size() >= 100u);
  const Corpus corpus = make_corpus(functions);

  const auto standard = make_hashing_embedder();
  HashingOptions plain;
  plain.char_trigrams = false;
  const auto without_trigrams = make_hashing_embedder(plain);
  HashingOptions wide;
  wide.dim = 1024;
  const auto wider = make_hashing_embedder(wide);

  const Recall body = known_item_recall(*standard, corpus.queries, corpus.bodies);
  const Recall full = known_item_recall(*standard, corpus.queries, corpus.rendered);
  print("body", "hashing 384", body);
  print("body", "hashing 384, no trigrams", known_item_recall(*without_trigrams, corpus.queries, corpus.bodies));
  print("body", "hashing 1024", known_item_recall(*wider, corpus.queries, corpus.bodies));
  print("rendered", "hashing 384", full);
  print("rendered", "hashing 384, no trigrams", known_item_recall(*without_trigrams, corpus.queries, corpus.rendered));
  print("rendered", "hashing 1024", known_item_recall(*wider, corpus.queries, corpus.rendered));

  // Chance level for 200 documents: top1 0.005, top5 0.025, top10 0.05. The
  // bounds are far below the measured values, because the corpus is a working
  // copy that other experiments edit.
  CHECK(body.top1 >= 0.20);
  CHECK(body.top5 >= 0.40);
  CHECK(body.top10 >= 0.50);
  CHECK(full.top1 >= 0.60);
  CHECK(full.top5 >= 0.80);
  CHECK(full.top10 >= 0.90);
}

PRAG_TEST(retrieval_among_all_functions_of_mimalloc) {
  if (corpus_is_missing()) return;
  const std::vector<Function> functions = sample_functions(static_cast<std::size_t>(-1));
  REQUIRE(functions.size() >= 200u);
  const Corpus corpus = make_corpus(functions);
  const auto standard = make_hashing_embedder();
  const Recall body = known_item_recall(*standard, corpus.queries, corpus.bodies);
  const Recall full = known_item_recall(*standard, corpus.queries, corpus.rendered);
  print("body", "hashing 384", body);
  print("rendered", "hashing 384", full);
  CHECK(body.top10 >= 0.30);
  CHECK(full.top10 >= 0.75);
}
