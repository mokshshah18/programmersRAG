// Shared infrastructure of the experiment drivers: arguments, the engine and
// its corpus, the query sets, result files with the meta block, and the small
// statistical tools (bootstrap, Wilson interval, Spearman, Platt scaling,
// spherical k-means).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prag/core/types.hpp"
#include "prag/engine/engine.hpp"
#include "prag/retrieve/calibration.hpp"

namespace bench {

using json = nlohmann::json;
using prag::UnitId;

inline double D(std::size_t x) { return static_cast<double>(x); }
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Args {
  std::string command;
  std::string settings = "prag.json";
  std::string out = "paper/results";
  bool quick = false;
};

// ---- query sets ------------------------------------------------------------

enum Split : int { kCalibration = 0, kHeldout = 1, kTest = 2 };

struct Query {
  std::string id;
  UnitId unit = prag::kNoUnit;  // the relevant unit
  std::string question;
  int split = kTest;
  std::string reference;  // reference answer (documentation of the unit), may be empty
};

struct QuerySet {
  std::string name;
  std::vector<Query> queries;
  std::string skipped;  // non-empty: the set could not be built, the reason
  bool ok() const { return skipped.empty(); }
};

struct QuerySets {
  QuerySet doc;   // Q-doc
  QuerySet gen;   // Q-gen
  json utility;   // utility labels or {"skipped": ...}
  json info;      // counts and parameters used to build the sets
  std::string fingerprint;
};

// ---- context ---------------------------------------------------------------

struct Context {
  Args args;
  std::string repo_dir;   // repository root (PRAG_SOURCE_DIR)
  std::string demo_dir;   // the corpus checkout (PRAG_DEMO_REPO)
  std::string data_dir;   // engine data directory of this run
  std::string out_dir;    // absolute result directory
  std::string settings_path;
  json settings;          // effective settings document handed to the engine
  std::unique_ptr<prag::engine::Engine> engine;
  std::shared_ptr<const prag::engine::Corpus> corpus;
  std::uint64_t seed = 20260930;

  std::unique_ptr<QuerySets> sets;
  std::vector<float> rows_cache;
  std::vector<float> docfree_cache;

  bool quick() const { return args.quick; }
  std::shared_ptr<prag::llm::LanguageModel> chat() const { return engine->chat_model(); }
  std::size_t n() const { return corpus->units.size(); }
  std::size_t dim() const { return corpus->store.dim(); }
};

// Opens the engine on the settings file, builds the index when it is missing or stale.
std::unique_ptr<Context> open_context(const Args& args);

// Stderr progress line: "[bench] ...".
void progress(const char* format, ...);
std::string iso_now();

// ---- result files ----------------------------------------------------------

json machine_json();
std::string git_head(const std::string& directory);
std::size_t peak_working_set_bytes();  // process-wide peak, 0 when unknown

// {"experiment", "started", "finished", "git_commit", "corpus", "machine", "models", "settings", "seed"}.
json make_meta(Context& ctx, const std::string& experiment, const std::string& started, const json& settings);
// Writes <out>/<name>.json with `meta` first, the body members after it, and "quick" when set.
void write_result(Context& ctx, const std::string& name, const std::string& started, const json& settings,
                  json body);
// Runs `f`; an exception becomes {"error": "..."} so one failing part does not lose the others.
json guarded(const std::string& part, const std::function<json()>& f);
json skipped(const std::string& reason);

// ---- random numbers (SplitMix-based, identical on every platform) ----------

std::uint64_t rng_next(std::uint64_t& state);
double rng_unit(std::uint64_t& state);  // [0, 1)
std::uint64_t rng_below(std::uint64_t& state, std::uint64_t n);
template <class T>
void shuffle(std::vector<T>& v, std::uint64_t seed) {
  std::uint64_t state = seed * 0x9e3779b97f4a7c15ULL + 12345;
  for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng_below(state, i)]);
}

// ---- statistics ------------------------------------------------------------

struct Ci {
  double mean = kNaN, lo = kNaN, hi = kNaN;
  std::size_t n = 0;
};
// Percentile bootstrap of the mean over `reps` resamples; non-finite values are dropped.
Ci bootstrap_mean(const std::vector<double>& values, std::uint64_t seed, std::size_t reps = 1000);
json to_json(const Ci& ci);

struct Wilson {
  double p = kNaN, lo = kNaN, hi = kNaN;
  std::size_t k = 0, n = 0;
};
Wilson wilson(std::size_t k, std::size_t n, double z = 1.959963984540054);
json to_json(const Wilson& w);

double mean_of(const std::vector<double>& v);
double quantile(std::vector<double> v, double q);  // linear interpolation, q in [0, 1]
double median(std::vector<double> v);
double spearman(const std::vector<double>& x, const std::vector<double>& y);
// median, q1, q3, iqr, min, max, reps, all in the unit of the input
json timing_json(const std::vector<double>& samples);

// ---- models ----------------------------------------------------------------

// P(relevant | z) = sigmoid(a z + b); fitted by Newton's method on the labelled
// scores and shifted to the class prior `prior` (the samples may contain the
// classes in another proportion).
struct PlattFit {
  double a = 0.0, b = 0.0;
  bool ok = false;
  double posterior(double z) const { return 1.0 / (1.0 + std::exp(-(a * z + b))); }
};
PlattFit fit_platt(const std::vector<prag::retrieve::LabeledScore>& samples, double prior);

// Cosine k-means on unit vectors (k-means++ start, `iterations` Lloyd steps).
std::vector<std::uint32_t> spherical_kmeans(const float* x, std::size_t n, std::size_t m, std::size_t k,
                                            std::uint64_t seed, int iterations = 50);

// Reliability diagram, ECE, Brier score and log loss of a probability function.
json reliability_json(const std::function<double(double)>& posterior,
                      const std::vector<prag::retrieve::LabeledScore>& samples, std::size_t bins = 10);

// ---- corpus data -----------------------------------------------------------

// Identifies the corpus content and the embedder (stored with the query sets).
std::string corpus_fingerprint(const Context& ctx);
// Loads <out>/queries.json when it matches this corpus and mode, builds the sets otherwise.
QuerySets& query_sets(Context& ctx);
// Builds the sets and the utility labels (queries.cpp).
void build_query_sets(Context& ctx, QuerySets* sets);
json query_sets_json(Context& ctx, const QuerySets& sets);

// Embeddings of all units, contiguous (unit i at i * dim).
const std::vector<float>& corpus_rows(Context& ctx);
// Embeddings of all units rendered without their leading comment (the index
// Q-doc retrieval runs against); cached on disk in the data directory.
const std::vector<float>& docfree_rows(Context& ctx);
// Rows the retrieval for a query set is scored against.
const std::vector<float>& rows_for(Context& ctx, const QuerySet& set);
// Text of a unit without its leading comment block.
std::string strip_leading_comment(std::string_view text, std::string_view language);
std::size_t count_words(std::string_view s);

// Split of a function unit (kCalibration / kHeldout / kTest), -1 for other units.
const std::vector<int>& unit_splits(Context& ctx);

// Exact scoring of the questions of a set against `rows_for(set)`.
struct Scored {
  std::vector<float> score;  // cosine per unit
  double mean = 0.0, sd = 0.0;
  std::vector<float> q;      // question embedding
  bool ok = false;           // sd positive and finite
  double z(std::size_t unit) const { return (static_cast<double>(score[unit]) - mean) / sd; }
};
std::vector<Scored> score_queries(Context& ctx, const QuerySet& set);

// Labelled standardised scores of the queries of one split: the relevant unit
// against `negatives_per_query` random other units (0 = all other units).
struct Labeled {
  std::vector<double> z_relevant, z_nonrelevant;
  std::vector<prag::retrieve::LabeledScore> samples;
  std::size_t queries = 0;
};
Labeled collect_labeled(const QuerySet& set, const std::vector<Scored>& scored, int split,
                        std::size_t negatives_per_query, std::uint64_t seed);

// Messages for a question over the given units in the serving format (as the engine builds them).
std::vector<prag::ChatMessage> messages_for(const Context& ctx, const std::string& question,
                                            const std::vector<UnitId>& units, bool edit_mode);
// Mean negative log-likelihood per token of `answer` after `messages`; NaN when nothing was scored.
double answer_loss(prag::llm::LanguageModel& model, const std::vector<prag::ChatMessage>& messages,
                   const std::string& answer, bool* truncated = nullptr);

// ---- experiments -----------------------------------------------------------

int run_queries(Context& ctx);
int run_e1(Context& ctx);
int run_e2(Context& ctx);
int run_e3(Context& ctx);
int run_e4(Context& ctx);
int run_e5(Context& ctx);
int run_e6(Context& ctx);
int run_e7(Context& ctx);

}  // namespace bench
