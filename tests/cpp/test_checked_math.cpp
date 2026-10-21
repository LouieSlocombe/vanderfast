#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>

#include "common/checked_math.hpp"
#include "vanderfast/error.hpp"

using vanderfast::Error;
using vanderfast::ErrorCode;

TEST_CASE("checked arithmetic detects overflow") {
  CHECK(vanderfast::checked_mul<std::size_t>(3, 4, "x") == 12);
  CHECK(vanderfast::checked_add<std::size_t>(3, 4, "x") == 7);
  const auto big = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(vanderfast::checked_mul<std::size_t>(big, 2, "x"), Error);
  CHECK_THROWS_AS(vanderfast::checked_add<std::size_t>(big, 1, "x"), Error);
  CHECK(vanderfast::checked_int32(7, "x") == 7);
  try {
    vanderfast::checked_int32(std::int64_t{1} << 40, "atom count");
    FAIL("expected an overflow");
  } catch (const Error& error) {
    CHECK(error.code() == ErrorCode::SizeOverflow);
  }
  CHECK_THROWS_AS(vanderfast::checked_int32(-1, "x"), Error);
}

TEST_CASE("error codes have names") {
  CHECK(std::string(vanderfast::to_string(ErrorCode::ContextBusy)) == "ContextBusy");
  CHECK(std::string(vanderfast::to_string(ErrorCode::Internal)) == "Internal");
}
