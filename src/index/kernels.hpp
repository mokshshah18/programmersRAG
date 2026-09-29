// Private to the index module: the kernel table selected at start-up.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prag::index {

struct Kernels {
  const char* name;

  std::int32_t (*dot_i8)(const std::int8_t* a, const std::int8_t* b, std::size_t dim);
  float (*dot_f32)(const float* a, const float* b, std::size_t dim);
  std::uint64_t (*and_popcount)(const std::uint64_t* a, const std::uint64_t* b, std::size_t words);

  // Number of ones in a code.
  std::uint32_t (*popcount)(const std::uint64_t* code, std::size_t words);

  // RaBitQ query kernel. `planes` holds the four bit planes of the quantised
  // query one after the other (planes[p * words + w] is word w of plane p).
  // Returns
  //     sum_p 2^p * popcount(code AND plane_p)  =  sum_j code_j * level_j.
  std::uint32_t (*weighted_popcount)(const std::uint64_t* code, const std::uint64_t* planes, std::size_t words);
};

// The table for this CPU. Selected once, on first use.
const Kernels& kernels() noexcept;

// Every table this CPU can run: "scalar", then "avx2" and "avx2-vnni" where
// they are compiled in and supported. The last one is the table selected.
const std::vector<const Kernels*>& usable_kernels();

// Double precision helpers with a fixed order of operations (one portable
// implementation). dot: eight partial sums, p[j mod 8], combined as
// ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)), remainder added one by one.
double dot_f64(const double* a, const double* b, std::size_t n) noexcept;
// v <- v - factor * q, element by element, product rounded before the subtraction.
void subtract_scaled(double* v, const double* q, double factor, std::size_t n) noexcept;

}  // namespace prag::index
