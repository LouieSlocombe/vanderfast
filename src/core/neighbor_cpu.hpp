// Brute-force CPU enumeration of directed lattice-site pairs (test oracle).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lattice.hpp"

namespace vanderfast {

struct SiteEdge {
  std::int32_t i;
  std::int32_t j;
  std::int32_t n0, n1, n2;
  double r2;
};

// Every directed pair ``(i -> j, n)`` except ``(i, i, 0)`` with r^2 <= cutoff^2,
// using the same displacement and r^2 arithmetic as the device code. Sorted by
// (i, j, n0, n1, n2).
std::vector<SiteEdge> enumerate_edges(const Lattice& lattice, const double* positions,
                                      std::size_t n_atoms, double cutoff);

}  // namespace vanderfast
