#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "core/lattice.hpp"
#include "vanderfast/error.hpp"

using vanderfast::Error;
using vanderfast::Geometry;

namespace {

Geometry make(const std::vector<double>& positions, std::array<double, 9> cell,
              std::array<bool, 3> pbc) {
  Geometry g;
  g.positions = positions.data();
  g.n_atoms = positions.size() / 3;
  g.cell = cell;
  g.pbc = pbc;
  return g;
}

}  // namespace

TEST_CASE("molecules use the identity basis") {
  std::vector<double> x{0, 0, 0, 1, 2, 3};
  auto lattice = vanderfast::make_lattice(make(x, {}, {false, false, false}));
  CHECK(lattice.n_active == 0);
  CHECK(vanderfast::image_ranges(lattice, x.data(), 2, 50.0) == std::array<int, 3>{0, 0, 0});
}

TEST_CASE("plane spacings and image ranges of a triclinic cell") {
  std::vector<double> x{0.1, 0.2, 0.3};
  std::array<double, 9> cell{5, 0, 0, 1, 4, 0, 0.5, 0.5, 3};
  auto lattice = vanderfast::make_lattice(make(x, cell, {true, true, true}));
  CHECK(lattice.n_active == 3);
  CHECK(lattice.spacing[2] == Catch::Approx(3.0));
  auto ranges = vanderfast::image_ranges(lattice, x.data(), 1, 10.0);
  CHECK(ranges[2] == 4);
  CHECK(vanderfast::cell_volume(cell) == Catch::Approx(60.0));
}

TEST_CASE("partial periodicity completes the basis orthogonally") {
  std::vector<double> x{0, 0, 0};
  std::array<double, 9> cell{2.46, 0, 0, -1.23, 2.13, 0, 1.0, 0.5, 12.0};
  auto slab = vanderfast::make_lattice(make(x, cell, {true, true, false}));
  CHECK(slab.basis[6] == Catch::Approx(0.0).margin(1e-15));
  CHECK(slab.basis[8] == Catch::Approx(1.0));
  auto wire = vanderfast::make_lattice(make(x, cell, {false, false, true}));
  const double dot = wire.basis[0] * cell[6] + wire.basis[1] * cell[7] + wire.basis[2] * cell[8];
  CHECK(dot == Catch::Approx(0.0).margin(1e-12));
  CHECK(wire.spacing[2] == Catch::Approx(std::sqrt(1.0 + 0.25 + 144.0)));
}

TEST_CASE("invalid geometries are rejected") {
  std::vector<double> bad{0, 0, std::nan("")};
  CHECK_THROWS_AS(vanderfast::make_lattice(make(bad, {}, {false, false, false})), Error);
  std::vector<double> x{0, 0, 0};
  CHECK_THROWS_AS(vanderfast::make_lattice(make(x, {}, {true, false, false})), Error);
  std::array<double, 9> parallel{1, 0, 0, 2, 0, 0, 0, 0, 1};
  CHECK_THROWS_AS(vanderfast::make_lattice(make(x, parallel, {true, true, false})), Error);
  std::array<double, 9> thin{1e-9, 0, 0, 0, 1, 0, 0, 0, 1};
  auto lattice = vanderfast::make_lattice(make(x, thin, {true, false, false}));
  CHECK_THROWS_AS(vanderfast::image_ranges(lattice, x.data(), 1, 60.0), Error);
}
