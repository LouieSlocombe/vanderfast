// Overflow-checked size arithmetic.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "vanderfast/error.hpp"

namespace vanderfast {

template <class T>
T checked_mul(T a, T b, const char* what) {
  T result;
  if (__builtin_mul_overflow(a, b, &result)) {
    fail(ErrorCode::SizeOverflow, std::string(what) + " overflows");
  }
  return result;
}

template <class T>
T checked_add(T a, T b, const char* what) {
  T result;
  if (__builtin_add_overflow(a, b, &result)) {
    fail(ErrorCode::SizeOverflow, std::string(what) + " overflows");
  }
  return result;
}

// Narrow a 64-bit count to a 32-bit index type, failing loudly on overflow.
inline std::int32_t checked_int32(std::int64_t value, const char* what) {
  if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
    fail(ErrorCode::SizeOverflow, std::string(what) + " exceeds the 32-bit index range");
  }
  return static_cast<std::int32_t>(value);
}

}  // namespace vanderfast
