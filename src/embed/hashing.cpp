#include "prag/embed/hashing.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"

namespace prag::embed {

namespace {

constexpr std::size_t kDefaultDim = 384;
constexpr std::uint64_t kSignBit = 1ULL << 63;
constexpr std::size_t kMinTrigramWord = 4;  // shorter sub-words are covered by the word feature

// Feature classes; the class is mixed into the seed, so the word "abc" and the
// trigram "abc" are different features.
enum : std::uint64_t { kWordClass = 1, kBigramClass = 2, kTrigramClass = 3, kEmptyClass = 4 };

inline bool is_letter(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_word_byte(unsigned char c) { return is_letter(c) || is_digit(c) || c == '_'; }

// a * b, rounded to double before anything else happens to it. A compiler that
// contracts floating-point expressions may otherwise fuse the product with a
// following addition (fused multiply-add), which rounds once instead of twice
// and changes the last bit on the machines that have the instruction. Reading
// the product back from a volatile object rules that out on every compiler.
inline double product(double a, double b) {
  volatile double rounded = a * b;
  return rounded;
}

// ln(n) for n >= 1 from additions, multiplications and divisions only. These
// are correctly rounded on every IEEE-754 platform, unlike the log of the C
// library, so the feature weights have the same bits everywhere.
double log_count(std::size_t n) {
  // n = m 2^e with m in [sqrt(1/2), sqrt(2)); both scalings are exact.
  double m = static_cast<double>(n);
  int e = 0;
  while (m >= 2.0) {
    m *= 0.5;
    ++e;
  }
  if (m > 1.4142135623730951) {
    m *= 0.5;
    ++e;
  }
  // ln(m) = 2 atanh(z) = 2 z (1 + z^2/3 + z^4/5 + ...), z = (m - 1) / (m + 1).
  // |z| < 0.172, so the first omitted term (order 26) is below 1e-21.
  const double z = (m - 1.0) / (m + 1.0);
  const double z2 = product(z, z);
  double series = 0.0;
  for (int k = 12; k >= 0; --k) series = product(series, z2) + 1.0 / static_cast<double>(2 * k + 1);
  const double ln_m = product(2.0 * z, series);
  return product(static_cast<double>(e), 0.6931471805599453) + ln_m;
}

// Shortest decimal form that reads back as `value`: 1 -> "1", 0.5 -> "0.5".
// Unlike printf it does not follow the locale, which may write "0,5".
std::string shortest(double value) {
  char buffer[32];
  const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof buffer, value);
  return std::string(buffer, result.ptr);
}

// Memory of one embed() call, reused for every text of the batch.
struct Scratch {
  std::vector<std::uint64_t> keys[3];  // one key per occurrence: words, bigrams, trigrams
  std::vector<double> sum;
  std::string marked;  // "<word>"
};

class HashingEmbedder final : public Embedder {
 public:
  explicit HashingEmbedder(const HashingOptions& options)
      : dim_(options.dim == 0 ? kDefaultDim : options.dim),
        seed_(options.seed),
        trigrams_(options.char_trigrams),
        word_seed_(hash_combine(options.seed, kWordClass)),
        bigram_seed_(hash_combine(options.seed, kBigramClass)),
        trigram_seed_(hash_combine(options.seed, kTrigramClass)) {
    // ln(tf) for the common small counts.
    log_table_.resize(257);
    for (std::size_t n = 1; n < log_table_.size(); ++n) log_table_[n] = log_count(n);

    const std::uint64_t empty_seed = hash_combine(options.seed, kEmptyClass);
    const auto magnitude = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim_)));
    fallback_.resize(dim_);
    for (std::size_t i = 0; i < dim_; ++i) {
      fallback_[i] = (hash_combine(empty_seed, i) & kSignBit) ? -magnitude : magnitude;
    }
  }

  std::size_t dim() const override { return dim_; }
  std::size_t max_tokens() const override { return kHashingMaxTokens; }

  std::string id() const override {
    char buffer[96];
    std::snprintf(buffer, sizeof buffer, "hashing-v1:dim=%zu,seed=0x%llx,trigrams=%d", dim_,
                  static_cast<unsigned long long>(seed_), trigrams_ ? 1 : 0);
    return std::string(buffer) + ",wb=" + shortest(kBigramWeight) + ",wt=" +
           shortest(trigrams_ ? kTrigramWeight : 0.0);
  }

  void embed(std::span<const std::string_view> texts, Role, float* out) override {
    Scratch scratch;
    scratch.sum.resize(dim_);
    for (std::size_t t = 0; t < texts.size(); ++t) embed_text(texts[t], scratch, out + t * dim_);
  }

 private:
  // Fills scratch.keys with one key per feature occurrence.
  void collect(std::string_view text, Scratch& scratch) const {
    for (auto& keys : scratch.keys) keys.clear();
    std::uint64_t previous = 0;
    bool adjacent = false;  // a sub-word precedes, with nothing but blanks in between

    const auto add_word = [&](std::string_view word, bool ascii) {
      const std::uint64_t key = hash64(word, word_seed_);
      scratch.keys[0].push_back(key);
      if (adjacent) scratch.keys[1].push_back(hash_combine(hash_combine(bigram_seed_, previous), key));
      previous = key;
      adjacent = true;
      if (trigrams_ && ascii && word.size() >= kMinTrigramWord) {
        scratch.marked.assign(1, '<');
        scratch.marked.append(word);
        scratch.marked.push_back('>');
        const std::string_view marked = scratch.marked;
        for (std::size_t k = 0; k + 3 <= marked.size(); ++k) {
          scratch.keys[2].push_back(hash64(marked.substr(k, 3), trigram_seed_));
        }
      }
    };

    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
      const auto c = static_cast<unsigned char>(text[i]);
      std::size_t j = i + 1;
      if (c == ' ' || c == '\t') {
        // Blanks keep adjacency.
      } else if (is_letter(c) || c == '_') {
        while (j < n && is_word_byte(static_cast<unsigned char>(text[j]))) ++j;
        const std::vector<std::string> words = text::split_identifier(text.substr(i, j - i));
        if (words.empty()) adjacent = false;  // "_": a token without sub-words separates its neighbours
        for (const std::string& word : words) add_word(word, true);
      } else if (c >= 0x80) {
        while (j < n && static_cast<unsigned char>(text[j]) >= 0x80) ++j;
        add_word(text.substr(i, j - i), false);
      } else if (is_digit(c)) {
        // Numeric literal with its suffix: 0x7F, 1.5e3, 10UL.
        while (j < n && (is_word_byte(static_cast<unsigned char>(text[j])) || text[j] == '.')) ++j;
        adjacent = false;
      } else {
        adjacent = false;  // punctuation, line ends, control characters
      }
      i = j;
    }
  }

  // Adds the weights of one feature class to `sum`, in ascending key order.
  void accumulate(std::vector<std::uint64_t>& keys, double class_weight, std::vector<double>& sum) const {
    std::sort(keys.begin(), keys.end());
    std::size_t first = 0;
    while (first < keys.size()) {
      std::size_t last = first + 1;
      while (last < keys.size() && keys[last] == keys[first]) ++last;
      const std::size_t tf = last - first;
      const double ln_tf = tf < log_table_.size() ? log_table_[tf] : log_count(tf);
      const double weight = product(1.0 + ln_tf, class_weight);
      const std::uint64_t key = keys[first];
      const auto index = static_cast<std::size_t>((key & ~kSignBit) % dim_);
      if (key & kSignBit) {
        sum[index] -= weight;
      } else {
        sum[index] += weight;
      }
      first = last;
    }
  }

  void embed_text(std::string_view text, Scratch& scratch, float* out) const {
    collect(text, scratch);
    std::fill(scratch.sum.begin(), scratch.sum.end(), 0.0);
    accumulate(scratch.keys[0], 1.0, scratch.sum);
    accumulate(scratch.keys[1], kBigramWeight, scratch.sum);
    accumulate(scratch.keys[2], kTrigramWeight, scratch.sum);

    double squares = 0.0;
    for (const double value : scratch.sum) squares += product(value, value);
    if (!(squares > 0.0)) {
      // No feature, or features that cancel exactly.
      std::copy(fallback_.begin(), fallback_.end(), out);
      return;
    }
    const double norm = std::sqrt(squares);
    for (std::size_t i = 0; i < dim_; ++i) out[i] = static_cast<float>(scratch.sum[i] / norm);
  }

  std::size_t dim_;
  std::uint64_t seed_;
  bool trigrams_;
  std::uint64_t word_seed_;
  std::uint64_t bigram_seed_;
  std::uint64_t trigram_seed_;
  std::vector<double> log_table_;
  std::vector<float> fallback_;
};

}  // namespace

std::unique_ptr<Embedder> make_hashing_embedder(const HashingOptions& options) {
  return std::make_unique<HashingEmbedder>(options);
}

}  // namespace prag::embed
