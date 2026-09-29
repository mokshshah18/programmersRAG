#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "prag/core/clock.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"
#include "prag/embed/hashing.hpp"

using namespace prag;
using namespace prag::embed;

namespace {

double dot(const std::vector<float>& a, const std::vector<float>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
  return sum;
}

double norm(const std::vector<float>& a) { return std::sqrt(dot(a, a)); }

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool all_finite(const std::vector<float>& a) {
  for (const float x : a) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// Hash of the bit patterns, independent of the byte order of the machine.
std::uint64_t checksum(const std::vector<float>& v) {
  std::uint64_t h = 0;
  for (const float x : v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof bits);
    h = hash_combine(h, bits);
  }
  return h;
}

std::vector<float> embed_text(std::string_view text, const HashingOptions& options = {}) {
  return make_hashing_embedder(options)->embed_one(text, Role::Document);
}

// Deterministic pseudo-random source text of about `bytes` bytes.
std::string synthetic_source(std::size_t bytes, std::uint64_t seed) {
  std::string out;
  out.reserve(bytes + 64);
  std::uint64_t state = seed;
  const auto word = [&] {
    state = mix64(state + 0x9e3779b97f4a7c15ULL);
    std::uint64_t r = state % 4096;  // a vocabulary of 4096 words
    r = mix64(r);
    const std::size_t length = 3 + r % 8;
    std::string w;
    for (std::size_t k = 0; k < length; ++k) {
      r = mix64(r + k);
      w.push_back(static_cast<char>('a' + r % 26));
    }
    return w;
  };
  while (out.size() < bytes) {
    out += "  " + word() + "_" + word() + "(" + word() + ", " + word() + ");\n";
  }
  return out;
}

const char* const kMalloc =
    "static void* mi_heap_malloc(mi_heap_t* heap, size_t size) {\n"
    "  return mi_heap_malloc_small(heap, size);\n"
    "}\n";
const char* const kZalloc =
    "void* mi_heap_zalloc(mi_heap_t* heap, size_t size) {\n"
    "  return mi_heap_malloc_zero(heap, size, true);\n"
    "}\n";
const char* const kJson =
    "int parse_json_number(const char* input, double* value) {\n"
    "  char* end = NULL;\n"
    "  *value = strtod(input, &end);\n"
    "  return end != input;\n"
    "}\n";

SourceUnit sample_unit() {
  SourceUnit unit;
  unit.id = 7;
  unit.path = "C:/corpus/mimalloc/src/alloc.c";
  unit.language = "c";
  unit.kind = UnitKind::Function;
  unit.name = "mi_heap_malloc";
  unit.signature = "static void* mi_heap_malloc(mi_heap_t* heap, size_t size)";
  unit.doc = "Allocate `size` bytes in `heap`.";
  unit.text =
      "\n// Allocate `size` bytes in `heap`.\n"
      "static void* mi_heap_malloc(mi_heap_t* heap, size_t size) {\n"
      "  return mi_heap_malloc_small(heap, size);\n"
      "}\n";
  return unit;
}

const char* const kSampleHeader =
    "src/alloc.c\n"
    "function mi_heap_malloc\n"
    "static void* mi_heap_malloc(mi_heap_t* heap, size_t size)";

}  // namespace

// ---- hashing embedder --------------------------------------------------------

PRAG_TEST(same_text_gives_the_same_bits) {
  const auto first = make_hashing_embedder();
  const auto second = make_hashing_embedder();
  const std::vector<float> a = first->embed_one(kMalloc, Role::Document);
  const std::vector<float> b = first->embed_one(kMalloc, Role::Document);
  const std::vector<float> c = second->embed_one(kMalloc, Role::Document);
  CHECK(same_bits(a, b));
  CHECK(same_bits(a, c));
  // Interleaved work does not leave traces in the embedder.
  (void)first->embed_one(kJson, Role::Document);
  CHECK(same_bits(a, first->embed_one(kMalloc, Role::Document)));
}

PRAG_TEST(golden_values_pin_the_scheme_across_platforms) {
  // Computed once with MSVC 14.51 on x86-64. Any conforming platform must
  // reproduce them bit for bit; a deliberate change of the scheme changes the
  // version in id() and these values together.
  HashingOptions plain;
  plain.char_trigrams = false;
  HashingOptions small;
  small.dim = 64;
  small.seed = 42;
  CHECK_EQ(checksum(embed_text(kMalloc)), 7845085307177462836ULL);
  CHECK_EQ(checksum(embed_text(kMalloc, plain)), 13312111391845307521ULL);
  CHECK_EQ(checksum(embed_text(kJson, small)), 9097088741390060743ULL);
  CHECK_EQ(checksum(embed_text("")), 7914101959971130846ULL);
}

PRAG_TEST(rows_have_unit_norm) {
  const auto embedder = make_hashing_embedder();
  for (const char* text : {kMalloc, kZalloc, kJson, "x", "mi", "a b c d e f g", "heap heap heap heap", "naive caf\xC3\xA9"}) {
    const std::vector<float> v = embedder->embed_one(text, Role::Document);
    CHECK_NEAR(norm(v), 1.0, 1e-6);
    CHECK(all_finite(v));
  }
}

PRAG_TEST(dimension_follows_the_options) {
  CHECK_EQ(make_hashing_embedder()->dim(), std::size_t{384});
  for (const std::size_t dim : {std::size_t{1}, std::size_t{2}, std::size_t{64}, std::size_t{384}, std::size_t{1000}}) {
    HashingOptions options;
    options.dim = dim;
    const auto embedder = make_hashing_embedder(options);
    CHECK_EQ(embedder->dim(), dim);
    // Exactly dim floats per text are written.
    const std::string_view texts[2] = {kMalloc, kJson};
    std::vector<float> out(2 * dim + 3, 7.0f);
    embedder->embed(texts, Role::Document, out.data());
    for (std::size_t k = 2 * dim; k < out.size(); ++k) CHECK_EQ(out[k], 7.0f);
    const std::vector<float> row0(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(dim));
    const std::vector<float> row1(out.begin() + static_cast<std::ptrdiff_t>(dim),
                                  out.begin() + static_cast<std::ptrdiff_t>(2 * dim));
    CHECK_NEAR(norm(row0), 1.0, 1e-6);
    CHECK_NEAR(norm(row1), 1.0, 1e-6);
  }
  HashingOptions zero;
  zero.dim = 0;
  CHECK_EQ(make_hashing_embedder(zero)->dim(), std::size_t{384});
  CHECK(make_hashing_embedder()->max_tokens() == kHashingMaxTokens);
}

PRAG_TEST(shared_identifiers_mean_higher_similarity) {
  const std::vector<float> malloc_unit = embed_text(kMalloc);
  const std::vector<float> zalloc_unit = embed_text(kZalloc);
  const std::vector<float> json_unit = embed_text(kJson);
  const double related = dot(malloc_unit, zalloc_unit);
  const double unrelated_a = dot(malloc_unit, json_unit);
  const double unrelated_b = dot(zalloc_unit, json_unit);
  CHECK(related > 0.5);
  CHECK(related > unrelated_a + 0.3);
  CHECK(related > unrelated_b + 0.3);

  // A question in plain words finds the unit that uses these words.
  const auto embedder = make_hashing_embedder();
  const std::vector<float> query = embedder->embed_one("allocate from a heap: heap malloc", Role::Query);
  CHECK(dot(query, malloc_unit) > dot(query, json_unit) + 0.2);
  const std::vector<float> other = embedder->embed_one("parse a json number", Role::Query);
  CHECK(dot(other, json_unit) > dot(other, malloc_unit) + 0.2);
}

PRAG_TEST(identifier_style_does_not_matter) {
  const std::vector<float> snake = embed_text("mi_heap_malloc");
  CHECK(same_bits(snake, embed_text("miHeapMalloc")));
  CHECK(same_bits(snake, embed_text("MiHeapMalloc")));
  CHECK(same_bits(snake, embed_text("MI_HEAP_MALLOC")));
  CHECK(same_bits(snake, embed_text("mi heap malloc")));
  CHECK(same_bits(snake, embed_text("  mi__heap \t malloc  ")));
  // The same holds inside running text.
  const std::vector<float> a = embed_text("p = mi_heap_malloc_aligned(heap, size);");
  const std::vector<float> b = embed_text("p = miHeapMallocAligned(heap, size);");
  CHECK(same_bits(a, b));
  // The order of the words does matter, through the bigrams.
  const double reordered = dot(snake, embed_text("malloc_heap_mi"));
  CHECK(reordered < 0.95);
  CHECK(reordered > 0.4);
}

PRAG_TEST(bigrams_need_adjacent_words) {
  const std::vector<float> phrase = embed_text("heap malloc");
  CHECK(same_bits(phrase, embed_text("heap \t  malloc")));
  CHECK(same_bits(phrase, embed_text("heap_malloc")));
  const std::vector<float> separated = embed_text("heap, malloc");
  CHECK(!same_bits(phrase, separated));
  CHECK(same_bits(separated, embed_text("heap\nmalloc")));
  CHECK(same_bits(separated, embed_text("heap(malloc)")));
  CHECK(same_bits(separated, embed_text("heap 16 malloc")));
  // An identifier without any sub-word is a token too: it separates its neighbours.
  CHECK(same_bits(separated, embed_text("heap _ malloc")));
  CHECK(same_bits(separated, embed_text("heap ___ malloc")));
  CHECK(!same_bits(embed_text("for in"), embed_text("for _ in")));
  CHECK(same_bits(embed_text("for, in"), embed_text("for _ in")));
  // Underscores inside or around an identifier with sub-words change nothing.
  CHECK(same_bits(phrase, embed_text("_heap__malloc_")));
  CHECK(same_bits(phrase, embed_text("__heap malloc__")));
}

PRAG_TEST(line_ends_of_both_kinds_give_the_same_embedding) {
  const std::string unix_text = "static void* f(heap_t* heap) {\n  return heap_malloc(heap);\n}\n\n// done\n";
  std::string windows_text;
  for (const char c : unix_text) {
    if (c == '\n') windows_text.push_back('\r');
    windows_text.push_back(c);
  }
  CHECK(same_bits(embed_text(unix_text), embed_text(windows_text)));
}

PRAG_TEST(numeric_literals_are_not_features) {
  const std::vector<float> x = embed_text("size");
  CHECK(same_bits(x, embed_text("size 123 0x7F 1.5e3 10UL")));
  CHECK(same_bits(x, embed_text("42; size; 7")));
  // Digits inside an identifier are sub-words.
  CHECK(!same_bits(embed_text("sha"), embed_text("sha256")));
}

PRAG_TEST(text_without_features_maps_to_one_fixed_unit_vector) {
  const auto embedder = make_hashing_embedder();
  const std::vector<float> empty = embedder->embed_one("", Role::Document);
  CHECK_NEAR(norm(empty), 1.0, 1e-6);
  const float magnitude = 1.0f / std::sqrt(384.0f);
  bool positive = false;
  bool negative = false;
  for (const float x : empty) {
    CHECK_NEAR(std::fabs(x), magnitude, 1e-7);
    positive = positive || x > 0.0f;
    negative = negative || x < 0.0f;
  }
  CHECK(positive && negative);

  for (const char* text : {" ", "   \t  ", "\n", "\r\n\r\n", " \n\t \r\n ", "(){};", "+-*/ = == !=", "123 456", "___", "0x10"}) {
    CHECK(same_bits(empty, embedder->embed_one(text, Role::Document)));
    CHECK(same_bits(empty, embedder->embed_one(text, Role::Query)));
  }
  const std::string zeros(16, '\0');
  CHECK(same_bits(empty, embedder->embed_one(zeros, Role::Document)));
  CHECK(same_bits(empty, embedder->embed_one(std::string_view(), Role::Document)));

  // It depends on the seed and is nearly orthogonal to real texts.
  HashingOptions other;
  other.seed = 1;
  CHECK(!same_bits(empty, embed_text("", other)));
  CHECK(std::fabs(dot(empty, embed_text(kMalloc))) < 0.3);
  CHECK(std::fabs(dot(empty, embed_text(kJson))) < 0.3);

  // Other dimensions, odd ones included.
  for (const std::size_t dim : {std::size_t{1}, std::size_t{7}, std::size_t{1024}}) {
    HashingOptions options;
    options.dim = dim;
    const std::vector<float> v = embed_text(" \n", options);
    CHECK_EQ(v.size(), dim);
    CHECK_NEAR(norm(v), 1.0, 1e-6);
  }
}

PRAG_TEST(weights_follow_the_documented_scheme) {
  // "alpha beta beta gamma_delta alpha beta", evaluated by hand.
  const std::string text = "alpha beta beta gamma_delta alpha beta";
  const std::vector<std::string> words = {"alpha", "beta", "beta", "gamma", "delta", "alpha", "beta"};
  for (const bool trigrams : {false, true}) {
    HashingOptions options;
    options.dim = 96;
    options.seed = 0xABCDEF;
    options.char_trigrams = trigrams;

    const std::uint64_t word_seed = hash_combine(options.seed, 1);
    const std::uint64_t bigram_seed = hash_combine(options.seed, 2);
    const std::uint64_t trigram_seed = hash_combine(options.seed, 3);
    std::map<std::uint64_t, int> tf[3];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint64_t key = hash64(words[i], word_seed);
      ++tf[0][key];
      if (i > 0) ++tf[1][hash_combine(hash_combine(bigram_seed, hash64(words[i - 1], word_seed)), key)];
      if (trigrams) {
        const std::string marked = "<" + words[i] + ">";
        for (std::size_t k = 0; k + 3 <= marked.size(); ++k) ++tf[2][hash64(marked.substr(k, 3), trigram_seed)];
      }
    }
    CHECK_EQ(tf[0].size(), std::size_t{4});
    CHECK_EQ(tf[1].size(), std::size_t{5});
    const double class_weight[3] = {1.0, kBigramWeight, kTrigramWeight};
    std::vector<double> expected(options.dim, 0.0);
    for (int c = 0; c < 3; ++c) {
      for (const auto& [key, count] : tf[c]) {
        const double weight = class_weight[c] * (1.0 + std::log(static_cast<double>(count)));
        const std::size_t index = static_cast<std::size_t>((key & 0x7FFFFFFFFFFFFFFFULL) % options.dim);
        expected[index] += (key >> 63) ? -weight : weight;
      }
    }
    double squares = 0.0;
    for (const double x : expected) squares += x * x;
    REQUIRE(squares > 0.0);

    const std::vector<float> actual = embed_text(text, options);
    REQUIRE(actual.size() == options.dim);
    for (std::size_t i = 0; i < options.dim; ++i) CHECK_NEAR(actual[i], expected[i] / std::sqrt(squares), 1e-6);
  }
}

PRAG_TEST(term_frequency_counts_sub_linearly) {
  HashingOptions options;
  options.char_trigrams = false;
  // One word, repeated on separate lines so that no bigram is formed: the
  // direction does not depend on the count.
  const std::vector<float> once = embed_text("heap", options);
  std::string many;
  for (int i = 0; i < 1000; ++i) many += "heap\n";
  CHECK(same_bits(once, embed_text(many, options)));

  // "page" 1 time against "heap" e^k times: the ratio of the two weights is 1 + ln tf.
  for (const int count : {1, 2, 20, 300, 5000}) {
    std::string text = "page\n";
    for (int i = 0; i < count; ++i) text += "heap\n";
    const std::vector<float> v = embed_text(text, options);
    const double heap = std::fabs(dot(v, once));
    const double page = std::fabs(dot(v, embed_text("page", options)));
    REQUIRE(page > 0.0);
    CHECK_NEAR(heap / page, 1.0 + std::log(static_cast<double>(count)), 1e-5);
  }
}

PRAG_TEST(trigrams_relate_spelling_variants) {
  HashingOptions plain;
  plain.char_trigrams = false;
  const double with_trigrams = dot(embed_text("zalloc"), embed_text("malloc"));
  const double without = dot(embed_text("zalloc", plain), embed_text("malloc", plain));
  CHECK(with_trigrams > 0.25);
  CHECK(with_trigrams < 0.9);
  CHECK(with_trigrams > without + 0.2);
  CHECK(dot(embed_text("aligned"), embed_text("alignment")) > dot(embed_text("aligned"), embed_text("segment")));
}

PRAG_TEST(very_long_input) {
  const std::string text = synthetic_source(4u << 20, 1);
  REQUIRE(text.size() >= (4u << 20));
  const auto embedder = make_hashing_embedder();
  const Stopwatch watch;
  const std::vector<float> a = embedder->embed_one(text, Role::Document);
  const double seconds = watch.elapsed_s();
  std::printf("  4 MiB embedded in %.0f ms\n", seconds * 1e3);
  CHECK(seconds < 10.0);
  CHECK_NEAR(norm(a), 1.0, 1e-6);
  CHECK(all_finite(a));
  CHECK(same_bits(a, embedder->embed_one(text, Role::Document)));
  // Another sample of the same vocabulary is close, source code of another
  // kind is not: the signal survives the collisions of 10^5 distinct features.
  const double same_kind = dot(a, embed_text(synthetic_source(1u << 20, 2)));
  const double other_kind = dot(a, embed_text(kMalloc));
  std::printf("  cosine to a sample of the same vocabulary %.3f, to other code %.3f\n", same_kind, other_kind);
  CHECK(same_kind > 0.5);
  CHECK(std::fabs(other_kind) < 0.2);

  // One identifier of 1 MiB, and one word a million times.
  const std::string giant(1u << 20, 'a');
  const std::vector<float> g = embedder->embed_one(giant, Role::Document);
  CHECK_NEAR(norm(g), 1.0, 1e-6);
  std::string repeated;
  repeated.reserve(5u << 20);
  for (int i = 0; i < 1000000; ++i) repeated += "heap ";
  const std::vector<float> r = embedder->embed_one(repeated, Role::Document);
  CHECK_NEAR(norm(r), 1.0, 1e-6);
  CHECK(all_finite(r));
  // The logarithm keeps the counts from taking over: the same three kinds of
  // feature as in "heap heap", in somewhat different proportions.
  CHECK(dot(r, embed_text("heap heap")) > 0.95);
}

PRAG_TEST(bytes_outside_ascii_are_handled) {
  const auto embedder = make_hashing_embedder();
  const std::vector<float> empty = embedder->embed_one("", Role::Document);
  const std::vector<float> accented = embedder->embed_one("caf\xC3\xA9 na\xC3\xAFve", Role::Document);
  CHECK_NEAR(norm(accented), 1.0, 1e-6);
  CHECK(!same_bits(accented, empty));
  const std::vector<float> cjk = embedder->embed_one("\xE5\x86\x85\xE5\xAD\x98", Role::Document);
  CHECK_NEAR(norm(cjk), 1.0, 1e-6);
  CHECK(!same_bits(cjk, empty));
  // Invalid UTF-8 and control characters.
  const std::string broken = std::string("a\xFF\xFE") + '\0' + "b\x01\x80";
  const std::vector<float> v = embedder->embed_one(broken, Role::Document);
  CHECK_NEAR(norm(v), 1.0, 1e-6);
  CHECK(all_finite(v));
}

PRAG_TEST(batch_equals_one_by_one) {
  const std::string long_text = synthetic_source(20000, 3);
  const std::vector<std::string_view> texts = {kMalloc, "", kJson, "   ", long_text, kZalloc, "x", kMalloc};
  for (const std::size_t dim : {std::size_t{384}, std::size_t{50}}) {
    HashingOptions options;
    options.dim = dim;
    const auto embedder = make_hashing_embedder(options);
    std::vector<float> batch(texts.size() * dim);
    embedder->embed(texts, Role::Document, batch.data());
    for (std::size_t t = 0; t < texts.size(); ++t) {
      const std::vector<float> one = embedder->embed_one(texts[t], Role::Document);
      CHECK(std::memcmp(one.data(), batch.data() + t * dim, dim * sizeof(float)) == 0);
    }
    // Equal texts give equal rows, wherever they stand in the batch.
    CHECK(std::memcmp(batch.data(), batch.data() + 7 * dim, dim * sizeof(float)) == 0);
    // An empty batch writes nothing.
    float untouched = 3.0f;
    embedder->embed(std::span<const std::string_view>(), Role::Document, &untouched);
    CHECK_EQ(untouched, 3.0f);
  }
}

PRAG_TEST(roles_are_embedded_identically) {
  const auto embedder = make_hashing_embedder();
  for (const char* text : {kMalloc, "heap malloc", ""}) {
    CHECK(same_bits(embedder->embed_one(text, Role::Query), embedder->embed_one(text, Role::Document)));
  }
}

PRAG_TEST(any_number_of_threads_gives_the_same_bits) {
  std::vector<std::string> texts;
  for (std::uint64_t k = 0; k < 96; ++k) texts.push_back(synthetic_source(200 + 50 * k, 100 + k));
  const std::size_t dim = 384;
  const auto shared = make_hashing_embedder();
  std::vector<float> serial(texts.size() * dim);
  for (std::size_t t = 0; t < texts.size(); ++t) {
    const std::string_view one[1] = {texts[t]};
    shared->embed(one, Role::Document, serial.data() + t * dim);
  }

  for (const unsigned threads : {1u, 2u, 3u, 8u}) {
    for (const bool own_embedder : {true, false}) {
      std::vector<float> parallel(texts.size() * dim, 0.0f);
      std::vector<std::thread> pool;
      for (unsigned w = 0; w < threads; ++w) {
        pool.emplace_back([&, w] {
          const auto own = make_hashing_embedder();
          Embedder& embedder = own_embedder ? *own : *shared;
          // Worker w takes the texts w, w + threads, ... in batches of up to four.
          std::vector<std::string_view> batch;
          std::vector<std::size_t> rows;
          std::vector<float> out;
          const auto flush = [&] {
            out.resize(batch.size() * dim);
            embedder.embed(batch, Role::Document, out.data());
            for (std::size_t b = 0; b < rows.size(); ++b) {
              std::memcpy(parallel.data() + rows[b] * dim, out.data() + b * dim, dim * sizeof(float));
            }
            batch.clear();
            rows.clear();
          };
          for (std::size_t t = w; t < texts.size(); t += threads) {
            batch.push_back(texts[t]);
            rows.push_back(t);
            if (batch.size() == 4) flush();
          }
          if (!batch.empty()) flush();
        });
      }
      for (std::thread& thread : pool) thread.join();
      CHECK(same_bits(serial, parallel));
    }
  }
}

PRAG_TEST(id_encodes_dimension_seed_and_options) {
  CHECK_EQ(make_hashing_embedder()->id(), std::string("hashing-v1:dim=384,seed=0x70726167,trigrams=1,wb=1,wt=0.5"));
  CHECK_EQ(make_hashing_embedder()->id(), make_hashing_embedder(HashingOptions{})->id());
  HashingOptions dim;
  dim.dim = 256;
  HashingOptions seed;
  seed.seed = 0xFFFFFFFFFFFFFFFFULL;
  HashingOptions plain;
  plain.char_trigrams = false;
  const std::vector<std::string> ids = {make_hashing_embedder()->id(), make_hashing_embedder(dim)->id(),
                                        make_hashing_embedder(seed)->id(), make_hashing_embedder(plain)->id()};
  for (std::size_t a = 0; a < ids.size(); ++a) {
    for (std::size_t b = a + 1; b < ids.size(); ++b) CHECK(ids[a] != ids[b]);
  }
  CHECK(ids[1].find("dim=256") != std::string::npos);
  CHECK(ids[2].find("seed=0xffffffffffffffff") != std::string::npos);
  CHECK(ids[3].find("trigrams=0") != std::string::npos);
  // dim = 0 is the default dimension and says so.
  HashingOptions zero;
  zero.dim = 0;
  CHECK_EQ(make_hashing_embedder(zero)->id(), ids[0]);
}

PRAG_TEST(id_and_embedding_do_not_depend_on_the_locale) {
  // The id is compared with the one stored in the index, so a host program that
  // selects the locale of the user must not change it ("wt=0,5").
  const std::string expected_id = make_hashing_embedder()->id();
  const std::vector<float> expected_row = embed_text(kMalloc);
  const char* const current = std::setlocale(LC_ALL, nullptr);
  const std::string saved = current ? current : "C";
  bool switched = false;
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "de-DE", "German_Germany.1252", "fr_FR.UTF-8", "fr-FR"}) {
    if (std::setlocale(LC_ALL, name) == nullptr) continue;
    char probe[16];
    std::snprintf(probe, sizeof probe, "%g", 0.5);
    if (std::string(probe) != "0,5") continue;
    switched = true;
    const std::string id = make_hashing_embedder()->id();
    const std::vector<float> row = embed_text(kMalloc);
    std::setlocale(LC_ALL, saved.c_str());
    CHECK_EQ(id, expected_id);
    CHECK(same_bits(row, expected_row));
    break;
  }
  std::setlocale(LC_ALL, saved.c_str());
  if (!switched) std::printf("  skipped: no locale with a decimal comma is installed\n");
  CHECK_EQ(expected_id, std::string("hashing-v1:dim=384,seed=0x70726167,trigrams=1,wb=1,wt=0.5"));
}

PRAG_TEST(options_change_the_embedding) {
  const std::vector<float> base = embed_text(kMalloc);
  HashingOptions seed;
  seed.seed = 12345;
  HashingOptions plain;
  plain.char_trigrams = false;
  CHECK(!same_bits(base, embed_text(kMalloc, seed)));
  CHECK(!same_bits(base, embed_text(kMalloc, plain)));
  // Different seeds are different random projections: unrelated coordinates,
  // but similar geometry.
  CHECK(std::fabs(dot(base, embed_text(kMalloc, seed))) < 0.3);
  const double here = dot(embed_text(kMalloc), embed_text(kZalloc));
  const double there = dot(embed_text(kMalloc, seed), embed_text(kZalloc, seed));
  CHECK_NEAR(here, there, 0.15);
}

// ---- render_for_embedding ----------------------------------------------------

PRAG_TEST(render_has_path_tail_kind_name_signature_doc_and_body) {
  const std::string text = render_for_embedding(sample_unit(), 4096);
  const std::string expected = std::string(kSampleHeader) +
                               "\n"
                               "Allocate `size` bytes in `heap`.\n"
                               "{\n"
                               "  return mi_heap_malloc_small(heap, size);\n"
                               "}";
  CHECK_EQ(text, expected);
  // The documentation and the declaration appear once.
  CHECK_EQ(text.find("Allocate"), text.rfind("Allocate"));
  CHECK_EQ(text.find("static void*"), text.rfind("static void*"));
}

PRAG_TEST(render_leaves_out_empty_parts) {
  SourceUnit region;
  region.path = "C:/corpus/mimalloc/src/alloc.c";
  region.kind = UnitKind::Region;
  region.text = "\n\n#include \"mimalloc.h\"\n#include \"mimalloc/internal.h\"\n\n";
  CHECK_EQ(render_for_embedding(region, 4096),
           std::string("src/alloc.c\nregion\n#include \"mimalloc.h\"\n#include \"mimalloc/internal.h\""));

  SourceUnit bare;
  bare.kind = UnitKind::Macro;
  bare.name = "MI_KiB";
  CHECK_EQ(render_for_embedding(bare, 4096), std::string("macro MI_KiB"));

  SourceUnit nothing;
  CHECK_EQ(render_for_embedding(nothing, 4096), std::string("region"));

  SourceUnit type;
  type.path = "include/mimalloc/types.h";
  type.kind = UnitKind::Type;
  type.name = "mi_heap_t";
  type.doc = "\n  A heap owns pages.  \n\n  One per thread.\n";
  type.text = "int x;";
  CHECK_EQ(render_for_embedding(type, 4096),
           std::string("mimalloc/types.h\ntype mi_heap_t\nA heap owns pages.\nOne per thread.\nint x;"));
}

PRAG_TEST(render_takes_the_last_two_path_components) {
  SourceUnit unit;
  unit.kind = UnitKind::Function;
  unit.name = "f";
  const auto first_line = [&](const char* path) {
    unit.path = path;
    const std::string text = render_for_embedding(unit, 100);
    return text.substr(0, text.find('\n'));
  };
  CHECK_EQ(first_line("C:/corpus/mimalloc/src/alloc.c"), std::string("src/alloc.c"));
  CHECK_EQ(first_line("/home/user/mimalloc/src/prim/unix/prim.c"), std::string("unix/prim.c"));
  CHECK_EQ(first_line("src/alloc.c"), std::string("src/alloc.c"));
  CHECK_EQ(first_line("alloc.c"), std::string("alloc.c"));
  CHECK_EQ(first_line("C:\\corpus\\src\\alloc.c"), std::string("src/alloc.c"));
  CHECK_EQ(first_line("/alloc.c"), std::string("/alloc.c"));
  CHECK_EQ(first_line(""), std::string("function f"));
}

PRAG_TEST(render_puts_the_signature_on_one_line) {
  SourceUnit unit = sample_unit();
  unit.signature = "static void*\n    mi_heap_malloc(mi_heap_t* heap,\n\t\t   size_t size)  ";
  unit.text =
      "// Allocate `size` bytes in `heap`.\n"
      "static void*\n    mi_heap_malloc(mi_heap_t* heap,\n\t\t   size_t size)\n"
      "{\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string(kSampleHeader) + "\nAllocate `size` bytes in `heap`.\n{\n  return 0;\n}");
}

PRAG_TEST(render_finds_a_declaration_that_was_normalised) {
  // parse hands over the declaration on one line: white space collapsed,
  // comments and line splices removed. It must still be found in the text, or
  // the declaration would be rendered twice.
  const std::string expected = std::string(kSampleHeader) + "\nAllocate `size` bytes in `heap`.\n{\n  return 0;\n}";
  SourceUnit unit = sample_unit();
  for (const char* declaration : {
           "static void*   mi_heap_malloc(mi_heap_t* heap, size_t size) {\n",
           "static void*\nmi_heap_malloc(mi_heap_t* heap,\n               size_t size)\n{\n",
           "static void*\r\n\tmi_heap_malloc(mi_heap_t* heap,\r\n\t\tsize_t size)\r\n{\r\n",
           "static void* /* owned */ mi_heap_malloc(mi_heap_t* heap,  // where\n    size_t size) {\n",
           "static void* \\\n  mi_heap_malloc(mi_heap_t* heap, \\\n  size_t size) {\n",
       }) {
    unit.text = std::string("// Allocate `size` bytes in `heap`.\n") + declaration + "  return 0;\n}\n";
    std::string text = render_for_embedding(unit, 4096);
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    CHECK_EQ(text, expected);
    CHECK_EQ(text.find("mi_heap_malloc("), text.rfind("mi_heap_malloc("));
  }
  // A comment in front of a bracket leaves a blank in the normalised form.
  unit.signature = "static void* mi_heap_malloc(mi_heap_t* heap, size_t size )";
  unit.text = "static void* mi_heap_malloc(mi_heap_t* heap, size_t size /* bytes */) {\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string("src/alloc.c\nfunction mi_heap_malloc\nstatic void* mi_heap_malloc(mi_heap_t* heap, size_t size )\n"
                       "Allocate `size` bytes in `heap`.\n{\n  return 0;\n}"));

  // A signature that was not normalised is found in the same way.
  unit.signature = "static void*\n    mi_heap_malloc(mi_heap_t* heap,   size_t size)";
  unit.text = "// Allocate `size` bytes in `heap`.\nstatic void* mi_heap_malloc(mi_heap_t* heap, size_t size)\n{\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096), expected);

  // What follows the declaration belongs to the body, a comment included.
  unit = sample_unit();
  unit.text = "static void*\nmi_heap_malloc(mi_heap_t* heap, size_t size) // may fail\n{\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string(kSampleHeader) + "\nAllocate `size` bytes in `heap`.\n// may fail\n{\n  return 0;\n}");

  // White space is not invented: "size_t size" does not match "size_tsize",
  // and "heap," does not match "heap ,".
  unit = sample_unit();
  unit.doc.clear();
  unit.text = "static void* mi_heap_malloc(mi_heap_t* heap, size_tsize) {\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string(kSampleHeader) + "\nstatic void* mi_heap_malloc(mi_heap_t* heap, size_tsize) {\n  return 0;\n}");
  unit.text = "static void* mi_heap_malloc(mi_heap_t* heap , size_t size) {\n  return 0;\n}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string(kSampleHeader) + "\nstatic void* mi_heap_malloc(mi_heap_t* heap , size_t size) {\n  return 0;\n}");

  // Python: a decorator on a line of its own, '#' is not a C comment and
  // "//" is an operator.
  SourceUnit python;
  python.language = "python";
  python.kind = UnitKind::Function;
  python.name = "half";
  python.signature = "@cache def half(n, d=2 // 1):";
  python.text = "@cache\ndef half(n,\n         d=2 // 1):\n    return n // d\n";
  CHECK_EQ(render_for_embedding(python, 4096),
           std::string("function half\n@cache def half(n, d=2 // 1):\n    return n // d"));
}

PRAG_TEST(render_leaves_out_a_python_documentation_string) {
  // parse puts the documentation string into `doc`; it must not come back
  // with the body.
  SourceUnit unit;
  unit.language = "python";
  unit.kind = UnitKind::Function;
  unit.name = "area";
  unit.signature = "def area(r):";
  unit.doc = "Area of a circle.\nThe radius is in metres.";
  const std::string expected =
      "function area\ndef area(r):\nArea of a circle.\nThe radius is in metres.\n    return 3.14159 * r * r";
  for (const char* literal : {
           "\"\"\"Area of a circle.\n\n    The radius is in metres.\n    \"\"\"",
           "'''Area of a circle.\n    The radius is in metres.'''",
           "r\"\"\"Area of a circle.\n    The radius is in metres.\"\"\"",
           "Rb'''Area of a circle.'''",
           "\"Area of a circle.\"",
           "'Area of a circle.'   ",
           "\"\"\"\n    Area of a circle.\n    The radius \\\"r\\\" is in metres.\n    \"\"\"",
           "\"\"\"Area of a circle.\r\n    The radius is in metres.\r\n    \"\"\"\r",
       }) {
    unit.text = std::string("def area(r):\n    ") + literal + "\n    return 3.14159 * r * r\n";
    CHECK_EQ(render_for_embedding(unit, 4096), expected);
  }
  // A comment behind the documentation string is text of the body.
  unit.text = "def area(r):\n    'Area of a circle.'  # short\n    return 3.14159 * r * r\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string("function area\ndef area(r):\nArea of a circle.\nThe radius is in metres.\n"
                       "# short\n    return 3.14159 * r * r"));
  // With a comment above the definition as well.
  unit.text = "# Area of a circle.\ndef area(r):\n    \"\"\"The radius is in metres.\"\"\"\n    return 3.14159 * r * r\n";
  CHECK_EQ(render_for_embedding(unit, 4096), expected);

  // Text that the documentation does not hold stays: a string that is no
  // documentation string, a string that is part of an expression, another
  // language, and a unit without documentation.
  unit.text = "def area(r):\n    \"\"\"Something else.\"\"\"\n    return 3.14159 * r * r\n";
  CHECK(render_for_embedding(unit, 4096).find("Something else.") != std::string::npos);
  unit.text = "def area(r):\n    \"Area of a circle.\".format(r)\n    return 3.14159 * r * r\n";
  CHECK(render_for_embedding(unit, 4096).ends_with("    \"Area of a circle.\".format(r)\n    return 3.14159 * r * r"));
  unit.text = "def area(r):\n    \"\"\"Area of a circle.\n    return 3.14159 * r * r\n";  // not closed
  CHECK(render_for_embedding(unit, 4096).ends_with("    \"\"\"Area of a circle.\n    return 3.14159 * r * r"));
  unit.language = "c";
  unit.text = "def area(r):\n    \"Area of a circle.\"\n    return 3.14159 * r * r\n";
  CHECK(render_for_embedding(unit, 4096).ends_with("    \"Area of a circle.\"\n    return 3.14159 * r * r"));
  unit.language = "python";
  unit.doc.clear();
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string("function area\ndef area(r):\n    \"Area of a circle.\"\n    return 3.14159 * r * r"));
}

PRAG_TEST(render_falls_back_to_the_text_without_its_comment) {
  // The signature does not occur in the text, however white space is read
  // (its producer dropped the names of the parameters).
  SourceUnit unit = sample_unit();
  unit.signature = "static void* mi_heap_malloc(mi_heap_t*, size_t)";
  unit.text =
      "/* Allocate `size` bytes\n   in `heap`. */\n"
      "// second comment\n"
      "static void*   mi_heap_malloc(mi_heap_t* heap, size_t size) {\n"
      "  return 0;\n"
      "}\n";
  CHECK_EQ(render_for_embedding(unit, 4096),
           std::string("src/alloc.c\nfunction mi_heap_malloc\nstatic void* mi_heap_malloc(mi_heap_t*, size_t)") +
               "\nAllocate `size` bytes in `heap`.\n"
               "static void*   mi_heap_malloc(mi_heap_t* heap, size_t size) {\n  return 0;\n}");

  // Without documentation nothing is taken for a comment.
  SourceUnit plain;
  plain.kind = UnitKind::Region;
  plain.text = "// not a documented unit\nint x;\n";
  CHECK_EQ(render_for_embedding(plain, 4096), std::string("region\n// not a documented unit\nint x;"));

  // Python comments start with '#', which is a directive in C.
  SourceUnit python;
  python.language = "python";
  python.kind = UnitKind::Function;
  python.name = "area";
  python.signature = "def area(r):";
  python.doc = "Area of a circle.";
  python.text = "# Area of a circle.\ndef area(r):\n    return 3.14159 * r * r\n";
  CHECK_EQ(render_for_embedding(python, 4096),
           std::string("function area\ndef area(r):\nArea of a circle.\n    return 3.14159 * r * r"));
  SourceUnit directive;
  directive.language = "c";
  directive.kind = UnitKind::Macro;
  directive.name = "MI_KiB";
  directive.doc = "One kibibyte.";
  directive.text = "// One kibibyte.\n#define MI_KiB 1024\n";
  CHECK_EQ(render_for_embedding(directive, 4096), std::string("macro MI_KiB\nOne kibibyte.\n#define MI_KiB 1024"));
}

PRAG_TEST(render_cuts_the_body_at_a_line_end) {
  SourceUnit unit = sample_unit();
  std::vector<std::string> lines;
  std::string body = " {\n";
  for (int i = 0; i < 400; ++i) {
    lines.push_back("  total_" + std::to_string(i) + " = total_" + std::to_string(i) + " + page_size(heap, " +
                    std::to_string(i) + ");");
    body += lines.back() + "\n";
  }
  body += "}\n";
  unit.text = "// Allocate `size` bytes in `heap`.\n" + unit.signature + body;

  const std::string complete = render_for_embedding(unit, 1u << 20);
  CHECK(complete.ends_with(lines.back() + "\n}"));
  const std::string prefix = std::string(kSampleHeader) + "\nAllocate `size` bytes in `heap`.\n{\n";
  REQUIRE(complete.starts_with(prefix));

  std::size_t previous_size = 0;
  for (const std::size_t budget : {std::size_t{60}, std::size_t{100}, std::size_t{256}, std::size_t{1000}, std::size_t{2048}}) {
    const std::string text = render_for_embedding(unit, budget);
    CHECK(text::estimate_tokens(text) <= budget);
    CHECK(text.size() < complete.size());
    CHECK(text.size() > previous_size);
    previous_size = text.size();
    // The header and the documentation are complete, the body is a prefix...
    REQUIRE(text.starts_with(prefix));
    REQUIRE(complete.starts_with(text));
    // ... that ends with a whole line ...
    const std::size_t kept = static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')) - 4;
    REQUIRE(kept >= 1u);
    REQUIRE(kept < lines.size());
    CHECK(text.ends_with(lines[kept - 1]));
    // ... and the next line would not fit any more.
    CHECK(text::estimate_tokens(text + "\n" + lines[kept]) > budget);
    // The budget is used: no more than one line is missing to it.
    CHECK(text::estimate_tokens(text) + text::estimate_tokens(lines[kept]) + 2 > budget);
  }
}

PRAG_TEST(render_never_cuts_the_header_lines) {
  const SourceUnit unit = sample_unit();
  const std::uint32_t header_tokens = text::estimate_tokens(kSampleHeader);
  for (const std::size_t budget : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{header_tokens - 1},
                                   std::size_t{header_tokens}}) {
    CHECK_EQ(render_for_embedding(unit, budget), std::string(kSampleHeader));
  }
  // Room for the documentation but not for the first line of the body.
  const std::string with_doc = std::string(kSampleHeader) + "\nAllocate `size` bytes in `heap`.";
  CHECK_EQ(render_for_embedding(unit, text::estimate_tokens(with_doc)), with_doc);
  CHECK_EQ(render_for_embedding(unit, text::estimate_tokens(with_doc) + 1), with_doc + "\n{");
}

PRAG_TEST(render_cuts_the_documentation_only_when_it_does_not_fit_alone) {
  SourceUnit unit = sample_unit();
  std::string doc;
  std::string comment;
  for (int i = 0; i < 200; ++i) {
    const std::string line = "Sentence number " + std::to_string(i) + " of the documentation.";
    doc += line + "\n";
    comment += "// " + line + "\n";
  }
  unit.doc = doc;
  unit.text = comment + unit.signature + " {\n  return 0;\n}\n";

  const std::string complete = render_for_embedding(unit, 1u << 20);
  CHECK(complete.ends_with("of the documentation.\n{\n  return 0;\n}"));
  const std::uint32_t doc_end = text::estimate_tokens(complete.substr(0, complete.find("\n{")));

  // The documentation fits: the body is cut, the documentation is not.
  const std::string cut_body = render_for_embedding(unit, doc_end + 1);
  CHECK(cut_body.ends_with("Sentence number 199 of the documentation.\n{"));

  // It does not fit: it is cut at a line end and no body follows.
  const std::string cut_doc = render_for_embedding(unit, 300);
  CHECK(text::estimate_tokens(cut_doc) <= 300u);
  CHECK(cut_doc.starts_with(std::string(kSampleHeader) + "\nSentence number 0 of the documentation.\n"));
  CHECK(cut_doc.ends_with("of the documentation."));
  CHECK(cut_doc.find("return") == std::string::npos);
  CHECK(cut_doc.find('{') == std::string::npos);
  CHECK(text::estimate_tokens(cut_doc) > 280u);
}

PRAG_TEST(render_cuts_inside_a_line_that_is_too_long_alone) {
  SourceUnit unit;
  unit.path = "src/generated.c";
  unit.kind = UnitKind::Region;
  std::string line;
  for (int i = 0; i < 5000; ++i) line += "value_" + std::to_string(i) + ", ";
  unit.text = line;
  const std::string text = render_for_embedding(unit, 200);
  const std::string header = "src/generated.c\nregion\n";
  REQUIRE(text.starts_with(header));
  CHECK(text::estimate_tokens(text) <= 200u);
  CHECK(text::estimate_tokens(text) > 190u);
  const std::string body = text.substr(header.size());
  REQUIRE(line.starts_with(body));
  // Cut between two tokens, not inside a word, and without trailing blanks.
  REQUIRE(!body.empty());
  const auto in_word = [](char c) { return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z'); };
  CHECK(!(in_word(body.back()) && in_word(line[body.size()])));
  CHECK(body.back() != ' ');

  // One word longer than the budget, in multi-byte characters: the cut is
  // between two characters.
  std::string word;
  for (int i = 0; i < 4000; ++i) word += "\xC3\xA9";
  unit.text = word;
  const std::string cut = render_for_embedding(unit, 100);
  REQUIRE(cut.starts_with(header));
  CHECK(text::estimate_tokens(cut) <= 100u);
  CHECK(cut.size() > header.size() + 100);
  CHECK(cut.size() < header.size() + word.size());
  CHECK_EQ((cut.size() - header.size()) % 2, std::size_t{0});
  CHECK_EQ(text::sanitize_utf8(cut), cut);

  // The same word behind its indentation: the budget is used in the same way,
  // the body is not given up.
  for (const std::string& long_word : {word, std::string(9000, 'q')}) {
    const std::string lines = "      " + long_word + "\nnext line\n";
    unit.text = "\n" + lines;
    const std::string indented = render_for_embedding(unit, 100);
    REQUIRE(indented.starts_with(header + "      "));
    CHECK(text::estimate_tokens(indented) <= 100u);
    CHECK(text::estimate_tokens(indented) > 90u);
    CHECK((header + lines).starts_with(indented));
    CHECK_EQ(text::sanitize_utf8(indented), indented);
  }
  // No room for a single character of it: nothing but the header.
  unit.text = "\n      " + word;
  CHECK_EQ(render_for_embedding(unit, text::estimate_tokens(header)), std::string("src/generated.c\nregion"));
}

PRAG_TEST(render_keeps_line_ends_of_both_kinds) {
  SourceUnit unit;
  unit.kind = UnitKind::Region;
  std::string body;
  for (int i = 0; i < 100; ++i) body += "int value_" + std::to_string(i) + " = " + std::to_string(i) + ";\r\n";
  unit.text = body;
  const std::string text = render_for_embedding(unit, 120);
  CHECK(text::estimate_tokens(text) <= 120u);
  CHECK(text.ends_with(";"));
  CHECK(text.find("\r\n") != std::string::npos);
  CHECK(("region\n" + body).starts_with(text));
}

PRAG_TEST(render_is_deterministic_and_feeds_the_embedder) {
  const SourceUnit unit = sample_unit();
  CHECK_EQ(render_for_embedding(unit, 512), render_for_embedding(unit, 512));
  const auto embedder = make_hashing_embedder();
  const std::vector<float> row = embedder->embed_one(render_for_embedding(unit, embedder->max_tokens()), Role::Document);
  const std::vector<float> query = embedder->embed_one("mi heap malloc", Role::Query);
  const std::vector<float> other = embedder->embed_one("parse json number", Role::Query);
  CHECK(dot(row, query) > 0.5);
  CHECK(dot(row, query) > dot(row, other) + 0.4);
  // A huge budget is not a problem.
  CHECK_EQ(render_for_embedding(unit, static_cast<std::size_t>(-1)), render_for_embedding(unit, 4096));
}
