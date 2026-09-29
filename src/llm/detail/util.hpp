// Helpers that do not depend on llama.cpp. Private to the llm module.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace prag::llm::detail {

// Number of physical cores (hyper-threads counted once); at least 1.
int physical_cores();

// requested > 0 is returned unchanged; 0 (or less) selects min(physical cores, 8).
// Language models stop scaling beyond that on this class of machine, because
// the matrix kernels are limited by memory bandwidth, not by core count.
int resolve_threads(int requested);

// Length of the longest prefix of `bytes` that does not end inside a multi-byte
// UTF-8 sequence. Bytes that can never start or continue a sequence are not
// held back: only an incomplete tail is.
std::size_t utf8_complete_prefix(std::string_view bytes);

// Removes leading continuation bytes and an incomplete trailing sequence, so
// that a slice cut at arbitrary byte positions is whole characters only.
std::string_view utf8_trim_partial(std::string_view bytes);

// "C:/models/qwen-q4_k_m.gguf" -> "qwen-q4_k_m".
std::string file_stem(std::string_view path);

std::string to_upper(std::string_view s);
std::string to_lower(std::string_view s);

}  // namespace prag::llm::detail
