// Prompt assembly from a pruned selection, and relevance-gated conversation memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "prag/context/rot.hpp"
#include "prag/core/types.hpp"

namespace prag::context {

struct Turn {
  std::string question;
  std::string answer;
  std::uint32_t tokens = 0;  // 0 = estimated from the text by gate_history
  double p = 0.0;  // relevance of the turn to the current question (filled by gate_history)
};

struct HistoryOptions {
  double theta = 0.25;       // turns below this relevance are left out
  double recency_half_life = 2.0;  // turns; the prior of a turn halves every half_life turns back
  std::uint32_t token_budget = 600;
  std::size_t max_turns = 6;
};

// Relevance of a past turn combines a recency prior with the semantic posterior
// by noisy-OR:  p = 1 - (1 - prior)(1 - posterior),  prior = 2^{-age / half_life}
// with age 1 for the most recent turn. `semantic_posterior[i]` is the calibrated
// probability that turn i is relevant to the current question; a missing value
// counts as 0. `turns` is ordered oldest first. Turns with p >= theta are taken
// in order of decreasing p (the more recent first among equals) until
// max_turns are kept; a turn that does not fit the remaining token budget is
// passed over. A half-life <= 0 switches the recency prior off. Returns the
// indices of the turns to keep, oldest first.
std::vector<std::size_t> gate_history(std::span<Turn> turns, std::span<const double> semantic_posterior,
                                      const HistoryOptions& options = {});

// Text of a unit in the requested form, preceded by a one-line header
// "// <relative path>:<first>-<last>  <kind> <name>". A path outside `root` is
// shown in full. The result ends with a newline.
std::string render_unit(const SourceUnit& unit, Form form, std::string_view root);

// Signature and documentation of a unit, with the body replaced by "{ ... }".
// Functions: the documentation as a line comment (at most 8 lines), then the
// signature followed by " { ... }" (" ..." in Python). Other kinds, and
// functions without a signature: the first lines of the text, at most 8 lines
// and 640 bytes, followed by a "// ..." line when text was left out.
std::string skeleton_of(const SourceUnit& unit);

struct PromptOptions {
  std::string system_prompt;  // empty = built-in instruction for code question answering
  bool edit_mode = false;     // true = instruct the model to answer with edit blocks
  std::string root;           // paths are shown relative to this directory
};

// The built-in system instruction: answer only from the supplied code, cite
// units as path:line, say plainly when the context does not contain the
// answer, keep answers concise. With edit_mode it also describes the two edit
// block forms of prag/edit/edit.hpp, with one example of each.
std::string default_system_prompt(bool edit_mode);

// Messages for a chat model: system instruction, the kept history turns, and a
// user message holding the context followed by the question.
//
// The context lists the entries of the selection in order, under one
// "## File <path>" line per file. Each unit is a header line
//     ### <relative path>:<first>-<last>  <kind> <name>
// (followed by "(skeleton)" or "(skeleton, support)" where that applies) and a
// fenced code block. Unit text is untrusted: every run of three or more
// backticks inside it is broken up, so that no line of a unit can close the
// fence around it, and the header fields are kept on one line. In edit mode
// the edit instructions are appended to a custom system prompt as well.
// Entries whose unit is not in `units` and turn indices outside `history` are
// skipped. Every message is valid UTF-8, so that the prompt can be sent as
// JSON: an invalid byte sequence in unit text, in a header field, in a history
// turn, in the question or in a custom system prompt is replaced by U+FFFD.
std::vector<ChatMessage> build_messages(std::string_view question, std::span<const SourceUnit> units,
                                    const Selection& selection, std::span<const Turn> history,
                                    std::span<const std::size_t> kept_turns, const PromptOptions& options);

}  // namespace prag::context
