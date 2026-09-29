// Model-free embedding by feature hashing, and the text rendering of units.
//
// The hashing embedder maps a text to a unit vector without any model file.
// It is a pure function of the text and the options: the same input gives the
// same bits on every run, every platform and for every number of threads.
//
//   1. Tokens. An identifier is a maximal run of ASCII letters, digits and
//      underscores that starts with a letter or an underscore. A run that
//      starts with a digit is a numeric literal; it extends over letters,
//      digits, underscores and '.' (0x7F, 1.5e3, 10UL) and is skipped. A
//      maximal run of bytes >= 0x80 is a token of its own and is taken as it
//      is. Line ends of both kinds (LF, CR LF) are read alike.
//   2. Sub-words. Identifiers are split with text::split_identifier, so
//      mi_heap_malloc, miHeapMalloc and MI_HEAP_MALLOC all give {mi, heap, malloc}.
//   3. Features.
//        word     every sub-word;
//        bigram   every ordered pair of adjacent sub-words, where adjacent means
//                 next to each other inside one identifier, or in two tokens
//                 separated by blanks (space, tab) only. Pairs across
//                 punctuation, a line end, a number or an identifier without
//                 sub-words ("_") are not formed: they are accidents of
//                 layout, not phrases;
//        trigram  (HashingOptions::char_trigrams) the character trigrams of
//                 "<w>" for every ASCII sub-word w of four or more characters,
//                 e.g. heap -> "<he", "hea", "eap", "ap>". They make spelling
//                 variants (alloc, malloc, zalloc) partially similar.
//   4. Weights. A feature that occurs tf times in the text has the weight
//      c (1 + ln tf), with c = 1 for words, kBigramWeight for bigrams and
//      kTrigramWeight for trigrams.
//   5. Hashing. key = hash64(feature bytes, hash_combine(seed, class)) for words
//      and trigrams, hash_combine(hash_combine(class seed, key(w1)), key(w2)) for
//      bigrams. The low 63 bits of the key modulo `dim` select the coordinate, bit
//      63 selects the sign. Collisions add up; the random signs make the inner
//      product of two hashed vectors an unbiased estimate of the inner product
//      of the feature vectors (Weinberger et al., 2009).
//   6. Normalisation to unit length. A text without any feature (empty, blank,
//      punctuation or numbers only), or whose features cancel exactly, maps to
//      one fixed unit vector that depends on the seed and the dimension only;
//      all its coordinates have magnitude 1/sqrt(dim), so it is nearly
//      orthogonal to every sparse feature vector.
//
// The weights are accumulated in double precision in a fixed order (feature
// class, then key), ln is evaluated from IEEE-754 basic operations instead of
// the C library, and no product is fused with an addition, so the result does
// not depend on the compiler, the library or the instruction set.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "prag/core/embedder.hpp"
#include "prag/core/types.hpp"

namespace prag::embed {

struct HashingOptions {
  std::size_t dim = 384;            // 0 selects the default, 384
  std::uint64_t seed = 0x70726167;  // "prag"
  bool char_trigrams = true;        // adds robustness to identifier spelling variants
};

// Class weights of step 4. They were chosen on the known-item experiment of
// tests/test_embed_retrieval.cpp (all functions of the corpus) and are part of
// the embedder id.
inline constexpr double kBigramWeight = 1.0;
inline constexpr double kTrigramWeight = 0.5;

// Rendering budget reported by the hashing embedder as max_tokens(). The
// embedder itself never truncates: a longer text is embedded completely.
inline constexpr std::size_t kHashingMaxTokens = 8192;

// Deterministic signed feature hashing of identifier sub-words, word bigrams and
// character trigrams with sub-linear term-frequency weighting. It needs no model
// file, which makes it the embedder of unit tests and the fallback when no
// embedding model is configured.
//
// Role::Query and Role::Document are embedded identically. The returned object
// keeps no state between calls, so embed() may also be called from several
// threads at once. id() has the form
// "hashing-v1:dim=384,seed=0x70726167,trigrams=1,wb=1,wt=0.5"
// whatever the locale of the program is.
std::unique_ptr<Embedder> make_hashing_embedder(const HashingOptions& options = {});

// Text handed to an embedder for a unit:
//
//   src/alloc.c                         last two components of the path
//   function mi_heap_malloc             kind and name (kind alone when unnamed)
//   void* mi_heap_malloc(mi_heap_t* heap, size_t size)     signature, on one line
//   Allocate `size` bytes in `heap`.    documentation, blank lines removed
//   { ... }                             body
//
// Empty parts are left out. The body is `text` without what the lines above
// already hold:
//   * when `doc` is not empty the leading comment is skipped;
//   * when the signature occurs in the text, everything up to its end. It is
//     looked for literally first, and then with white space read loosely: a
//     blank of the signature stands for any run of white space, line splices
//     and (except in Python) comments of the text, which is the form in which
//     parse delivers a declaration that was spread over several lines. A
//     signature that is not found leaves the declaration in the body;
//   * in Python, when `doc` is not empty, the documentation string that opens
//     the body, provided its first line of text is found in `doc`.
//
// The result is limited to `max_tokens` as counted by text::estimate_tokens.
// The three header lines are never cut. The body is cut first, at the end of a
// line (inside the line only when the first line alone is too long); the
// documentation is cut, in the same way, only when it does not fit the budget
// that the header lines leave. Hence estimate_tokens(result) <= max_tokens
// whenever the header lines fit, and the result consists of the header lines
// alone when they do not.
std::string render_for_embedding(const SourceUnit& unit, std::size_t max_tokens);

}  // namespace prag::embed
