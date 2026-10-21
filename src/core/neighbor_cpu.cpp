#include "neighbor_cpu.hpp"

#include <algorithm>
#include <tuple>

#include "../common/d3_math.hpp"

namespace vanderfast {

std::vector<SiteEdge> enumerate_edges(const Lattice& lattice, const double* positions,
                                      std::size_t n_atoms, double cutoff) {
  const std::array<int, 3> range = image_ranges(lattice, positions, n_atoms, cutoff);
  const double cutoff2 = cutoff * cutoff;
  std::vector<SiteEdge> edges;
  for (std::size_t i = 0; i < n_atoms; ++i) {
    const math::Vec3 xi{positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]};
    for (int a = -range[0]; a <= range[0]; ++a) {
      for (int b = -range[1]; b <= range[1]; ++b) {
        for (int c = -range[2]; c <= range[2]; ++c) {
          for (std::size_t j = 0; j < n_atoms; ++j) {
            if (i == j && a == 0 && b == 0 && c == 0) continue;
            const math::Vec3 xj{positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]};
            const math::Vec3 d = math::site_displacement(xi, xj, a, b, c, lattice.cell.data());
            const double r2 = math::norm2(d);
            if (r2 <= cutoff2) {
              edges.push_back({static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), a, b, c, r2});
            }
          }
        }
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const SiteEdge& x, const SiteEdge& y) {
    return std::tie(x.i, x.j, x.n0, x.n1, x.n2) < std::tie(y.i, y.j, y.n0, y.n1, y.n2);
  });
  return edges;
}

}  // namespace vanderfast
