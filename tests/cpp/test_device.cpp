#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "vanderfast/context.hpp"
#include "vanderfast/error.hpp"

using vanderfast::D3Context;
using vanderfast::D3Parameters;

TEST_CASE("a one-atom molecule has zero dispersion on the device") {
  if (D3Context::device_count() == 0) SKIP("no CUDA device");
  D3Parameters p;
  p.s6 = 1.0;
  p.s8 = 0.7875;
  p.a1 = 0.4289;
  p.a2 = 4.4407;
  std::int32_t numbers[] = {18};
  D3Context context(numbers, 1, p);
  std::vector<double> x{0, 0, 0};
  vanderfast::Geometry geometry;
  geometry.positions = x.data();
  geometry.n_atoms = 1;
  auto result = context.evaluate(geometry, vanderfast::kEnergy | vanderfast::kGradient);
  CHECK(result.energy == 0.0);
  CHECK(result.gradient.size() == 3);
  CHECK(context.stats().evaluations == 1);
}
