// Training example construction (DESIGN.md section 6.12).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <unordered_set>

#include "prag/core/hash.hpp"
#include "prag/tune/pipeline.hpp"

namespace prag::tune {
namespace {

constexpr const char* kAbsentAnswer = "The supplied code does not contain the answer.";

std::size_t count_words(const std::string& text) {
  std::size_t words = 0;
  bool in_word = false;
  for (unsigned char c : text) {
    const bool space = std::isspace(c) != 0;
    if (!space && !in_word) ++words;
    in_word = !space;
  }
  return words;
}

std::string trim(std::string text) {
  const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
  std::size_t begin = 0;
  while (begin < text.size() && space(static_cast<unsigned char>(text[begin]))) ++begin;
  return text.substr(begin);
}

// First sentence of the documentation, whitespace collapsed.
std::string first_sentence(const std::string& doc) {
  std::string flat;
  bool pending_space = false;
  for (unsigned char c : doc) {
    if (std::isspace(c)) {
      pending_space = !flat.empty();
    } else {
      if (pending_space) flat += ' ';
      pending_space = false;
      flat += static_cast<char>(c);
    }
  }
  for (std::size_t i = 0; i < flat.size(); ++i) {
    if (flat[i] == '.' && (i + 1 == flat.size() || flat[i + 1] == ' ')) return flat.substr(0, i + 1);
  }
  return flat;
}

std::string question_for(const SourceUnit& unit) {
  const std::string name = unit.name.empty() ? std::string("this code") : "`" + unit.name + "`";
  switch (unit.kind) {
    case UnitKind::Function: return "What does " + name + " do?";
    case UnitKind::Type: return "What is " + name + " and what is it used for?";
    case UnitKind::Macro: return "What is the purpose of the macro " + name + "?";
    case UnitKind::Region: break;
  }
  return "What is the purpose of this code?";
}

std::string answer_for(const SourceUnit& unit) {
  std::string quoted = trim(unit.signature);
  if (quoted.empty()) {
    // First lines of the unit text.
    std::size_t end = 0;
    for (int lines = 0; lines < 6 && end < unit.text.size(); ++lines) {
      const std::size_t nl = unit.text.find('\n', end);
      end = nl == std::string::npos ? unit.text.size() : nl + 1;
    }
    quoted = trim(unit.text.substr(0, end));
  }
  std::string answer = first_sentence(unit.doc);
  answer += "\n\n";
  if (!unit.name.empty()) answer += "It is `" + unit.name + "`:\n";
  answer += "```" + unit.language + "\n" + quoted + "\n```";
  return answer;
}

std::string hex64(std::uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(value));
  return buffer;
}

std::string make_id(const Example& e, std::uint64_t content_hash) {
  std::uint64_t h = hash64(e.question, content_hash);
  h = hash_combine(h, e.oracle_present ? 1 : 0);
  for (UnitId id : e.context) h = hash_combine(h, id);
  return "ex-" + hex64(h);
}

struct Candidate {
  const SourceUnit* unit;
  std::uint64_t key;
};

}  // namespace

ExampleSet build_examples(const CorpusView& corpus, const ExampleOptions& options) {
  ExampleSet set;

  std::vector<Candidate> candidates;
  for (const SourceUnit& unit : corpus.units) {
    if (count_words(unit.doc) < options.min_doc_words) continue;
    candidates.push_back({&unit, mix64(unit.content_hash ^ mix64(options.seed ^ 0x5bd1e995ULL))});
  }
  // A unit's place in the order depends only on its content and the seed.
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.key != b.key ? a.key < b.key : a.unit->id < b.unit->id;
  });

  const std::size_t n = candidates.size();
  const double hold_fraction = std::clamp(options.holdout_fraction, 0.0, 1.0);
  std::size_t holdout_units = static_cast<std::size_t>(std::llround(hold_fraction * static_cast<double>(n)));
  if (hold_fraction > 0.0 && holdout_units == 0 && n >= 2) holdout_units = 1;
  holdout_units = std::min(holdout_units, n);

  const std::size_t cap = options.max_examples;
  const std::size_t hold_take = std::min(
      holdout_units, static_cast<std::size_t>(std::llround(hold_fraction * static_cast<double>(cap))));
  const std::size_t train_take = std::min(n - holdout_units, cap > hold_take ? cap - hold_take : 0);

  std::unordered_set<UnitId> holdout_ids;
  for (std::size_t i = 0; i < holdout_units; ++i) holdout_ids.insert(candidates[i].unit->id);

  const double fraction = std::clamp(options.oracle_fraction, 0.0, 1.0);

  auto build = [&](std::size_t first, std::size_t count, bool is_holdout, std::vector<Example>* out) {
    for (std::size_t i = 0; i < count; ++i) {
      const SourceUnit& unit = *candidates[first + i].unit;
      // Stratified: the running share of present oracles tracks `fraction` within one example.
      bool present = std::floor(static_cast<double>(i + 1) * fraction + 1e-9) >
                     std::floor(static_cast<double>(i) * fraction + 1e-9);

      std::vector<UnitId> distractors;
      if (corpus.distractors) {
        std::unordered_set<UnitId> seen{unit.id};
        for (UnitId id : corpus.distractors(unit.id, options.distractors + 2)) {
          if (distractors.size() >= options.distractors) break;
          if (!seen.insert(id).second) continue;
          if (!is_holdout && holdout_ids.count(id)) continue;  // held-out units stay out of training prompts
          distractors.push_back(id);
        }
      }
      if (distractors.empty() && !present) present = true;  // nothing to show instead of the oracle

      Example e;
      e.question = question_for(unit);
      e.oracle = unit.id;
      e.oracle_present = present;
      e.source = "documentation";
      e.context = distractors;
      if (present) {
        // Deterministic position of the oracle among the distractors.
        const std::size_t pos = static_cast<std::size_t>(mix64(candidates[first + i].key) % (e.context.size() + 1));
        e.context.insert(e.context.begin() + static_cast<std::ptrdiff_t>(pos), unit.id);
        e.answer = answer_for(unit);
      } else {
        e.answer = kAbsentAnswer;
      }
      e.id = make_id(e, unit.content_hash);
      out->push_back(std::move(e));
    }
  };

  build(0, hold_take, true, &set.holdout);
  build(holdout_units, train_take, false, &set.train);
  return set;
}

std::string render_training_text(const ExampleSet& examples, const CorpusView& corpus,
                                 const std::function<std::string(std::span<const ChatMessage>)>& chat_template) {
  std::string text;
  for (const Example& e : examples.train) {
    std::vector<ChatMessage> messages;
    if (corpus.prompt) messages = corpus.prompt(e.question, e.context);
    if (messages.empty()) messages.push_back({"user", e.question});
    messages.push_back({"assistant", e.answer});
    if (chat_template) {
      text += chat_template(messages);
    } else {
      for (const ChatMessage& m : messages) text += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    }
    if (!text.empty() && text.back() != '\n') text += '\n';
  }
  return text;
}

}  // namespace prag::tune
