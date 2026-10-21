#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

#include "vanderfast/error.hpp"
#include "vanderfast/parameters.hpp"

using vanderfast::D3Parameters;
using vanderfast::Error;

TEST_CASE("parameter validation and activity rules") {
  D3Parameters p;
  p.s6 = 1.0;
  p.s8 = 0.7875;
  p.a1 = 0.4289;
  p.a2 = 4.4407;
  p.s9 = 1.0;
  CHECK_NOTHROW(p.validate());
  CHECK(p.pair_active());
  CHECK(p.atm_active());
  CHECK(p.atm_exponent() == 16.0);
  const auto fingerprint = p.fingerprint();
  p.s9 = 1e-17;
  CHECK_FALSE(p.atm_active());
  CHECK(p.fingerprint() != fingerprint);
  p.s6 = 0.0;
  p.s8 = 0.0;
  CHECK_FALSE(p.pair_active());
  p.alp = 0.0;
  CHECK_THROWS_AS(p.validate(), Error);
  p.alp = 14.0;
  p.a2 = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(p.validate(), Error);
  p.a2 = 4.0;
  p.cutoffs.disp3 = -1.0;
  CHECK_THROWS_AS(p.validate(), Error);
}
