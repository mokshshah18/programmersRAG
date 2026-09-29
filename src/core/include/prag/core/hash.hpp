// Non-cryptographic 64-bit hashing (FNV-1a core with an avalanche finaliser).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace prag {

// splitmix64 finaliser: full avalanche, bijective.
constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

constexpr std::uint64_t hash64(std::string_view bytes, std::uint64_t seed = 0) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ULL ^ mix64(seed);
  for (const char c : bytes) {
    h ^= static_cast<unsigned char>(c);
    h *= 0x100000001b3ULL;
  }
  return mix64(h);
}

constexpr std::uint64_t hash_combine(std::uint64_t a, std::uint64_t b) noexcept {
  return mix64(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
}

}  // namespace prag
