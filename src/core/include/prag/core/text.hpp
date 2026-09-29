// Small text utilities shared by the parser, embedder and context assembler.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace prag::text {

std::string_view trim(std::string_view s);
std::string to_lower(std::string_view s);
bool contains_icase(std::string_view haystack, std::string_view needle);
std::vector<std::string_view> split_lines(std::string_view s);  // without line terminators

// Splits an identifier into lower-case words at underscores, digit boundaries
// and camelCase humps: "mi_heapMallocAligned2" -> {"mi", "heap", "malloc", "aligned", "2"}.
std::vector<std::string> split_identifier(std::string_view identifier);

// Model-free token estimate for source text: one token per identifier sub-word,
// number or punctuation mark, which tracks byte-pair tokenisers closely on code.
std::uint32_t estimate_tokens(std::string_view s);

// Number of lines in `s`: newline count, plus one when the text does not end
// in a newline. Zero for empty text.
std::uint32_t count_lines(std::string_view s);

// Replaces invalid UTF-8 sequences by U+FFFD so the text can be embedded in JSON.
std::string sanitize_utf8(std::string_view s);

}  // namespace prag::text
