// Private to the index module: cache-line aligned arrays of trivial elements.
#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace prag::index {

inline constexpr std::size_t kCacheLine = 64;

inline constexpr std::size_t round_up(std::size_t value, std::size_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Owns `size()` zero-initialised elements starting on a 64-byte boundary. The
// allocation is a whole number of cache lines, so two arrays never share one.
template <class T>
class AlignedArray {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);

 public:
  AlignedArray() noexcept = default;
  explicit AlignedArray(std::size_t count) { reset(count); }
  ~AlignedArray() { release(); }

  AlignedArray(AlignedArray&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
  AlignedArray& operator=(AlignedArray&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  AlignedArray(const AlignedArray&) = delete;
  AlignedArray& operator=(const AlignedArray&) = delete;

  void reset(std::size_t count) {
    release();
    if (count == 0) return;
    const std::size_t bytes = round_up(count * sizeof(T), kCacheLine);
    data_ = static_cast<T*>(::operator new(bytes, std::align_val_t{kCacheLine}));
    std::memset(static_cast<void*>(data_), 0, bytes);
    size_ = count;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }

 private:
  void release() noexcept {
    if (data_ != nullptr) ::operator delete(static_cast<void*>(data_), std::align_val_t{kCacheLine});
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace prag::index
