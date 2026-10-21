// Geometry validation, the neighbour-search basis, and lattice image bounds.
#pragma once

#include <array>
#include <cstddef>

#include "vanderfast/context.hpp"

namespace vanderfast {

struct Lattice {
  std::array<double, 9> cell{};     // user cell, rows
  std::array<bool, 3> pbc{};
  int n_active = 0;
  std::array<double, 9> basis{};    // active rows from the cell, inactive rows completed
  std::array<double, 9> inverse{};  // inverse of the basis (row-vector convention)
  std::array<double, 3> spacing{};  // plane spacing of each basis direction
};

// Validate positions, cell, and periodicity. Throws InvalidGeometry.
Lattice make_lattice(const Geometry& geometry);

// Per-direction image bounds complete for any wrapping of the input: every
// site within ``cutoff`` has ``|n_k| <= cutoff / h_k + spread_k``.
std::array<int, 3> image_ranges(const Lattice& lattice, const double* positions,
                                std::size_t n_atoms, double cutoff);

// Absolute determinant of the user cell (the 3D volume used for stress).
double cell_volume(const std::array<double, 9>& cell) noexcept;

}  // namespace vanderfast
