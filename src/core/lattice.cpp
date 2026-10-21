#include "lattice.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "vanderfast/error.hpp"

namespace vanderfast {

namespace {

using Vec3 = std::array<double, 3>;

Vec3 row(const std::array<double, 9>& m, int r) {
  const auto base = static_cast<std::size_t>(3 * r);
  return {m[base], m[base + 1], m[base + 2]};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

double det(const std::array<double, 9>& m) {
  return dot(row(m, 0), cross(row(m, 1), row(m, 2)));
}

std::array<double, 9> inverse(const std::array<double, 9>& m) {
  const double d = det(m);
  const Vec3 c0 = cross(row(m, 1), row(m, 2));
  const Vec3 c1 = cross(row(m, 2), row(m, 0));
  const Vec3 c2 = cross(row(m, 0), row(m, 1));
  // For row vectors f = x M^-1, the columns of M^-1 are the cofactor rows / det.
  return {c0[0] / d, c1[0] / d, c2[0] / d, c0[1] / d, c1[1] / d,
          c2[1] / d, c0[2] / d, c1[2] / d, c2[2] / d};
}

}  // namespace

double cell_volume(const std::array<double, 9>& cell) noexcept { return std::fabs(det(cell)); }

Lattice make_lattice(const Geometry& geometry) {
  for (std::size_t i = 0; i < 3 * geometry.n_atoms; ++i) {
    if (!std::isfinite(geometry.positions[i])) {
      fail(ErrorCode::InvalidGeometry, "positions must be finite");
    }
  }
  Lattice lattice;
  lattice.cell = geometry.cell;
  lattice.pbc = geometry.pbc;
  for (double value : geometry.cell) {
    if (!std::isfinite(value)) fail(ErrorCode::InvalidGeometry, "cell must be finite");
  }
  int active[3];
  int inactive[3];
  int n_inactive = 0;
  for (int k = 0; k < 3; ++k) {
    if (geometry.pbc[static_cast<std::size_t>(k)]) {
      active[lattice.n_active++] = k;
    } else {
      inactive[n_inactive++] = k;
    }
  }
  lattice.basis = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (lattice.n_active > 0) {
    // Gram determinant test for linear independence of the active vectors.
    double gram[3][3];
    double diagonal = 1.0;
    for (int a = 0; a < lattice.n_active; ++a) {
      for (int b = 0; b < lattice.n_active; ++b) {
        gram[a][b] = dot(row(geometry.cell, active[a]), row(geometry.cell, active[b]));
      }
      diagonal *= gram[a][a];
    }
    double g = gram[0][0];
    if (lattice.n_active == 2) g = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
    if (lattice.n_active == 3) g = det({gram[0][0], gram[0][1], gram[0][2], gram[1][0], gram[1][1],
                                        gram[1][2], gram[2][0], gram[2][1], gram[2][2]});
    if (!(diagonal > 0.0) || !(g > 1e-12 * diagonal)) {
      fail(ErrorCode::InvalidGeometry, "active lattice vectors must be linearly independent");
    }
    for (int a = 0; a < lattice.n_active; ++a) {
      for (int c = 0; c < 3; ++c) {
        lattice.basis[static_cast<std::size_t>(3 * active[a] + c)] =
            geometry.cell[static_cast<std::size_t>(3 * active[a] + c)];
      }
    }
    Vec3 completion[2];
    if (lattice.n_active == 2) {
      Vec3 normal = cross(row(geometry.cell, active[0]), row(geometry.cell, active[1]));
      completion[0] = scale(normal, 1.0 / norm(normal));
    } else if (lattice.n_active == 1) {
      Vec3 a = row(geometry.cell, active[0]);
      a = scale(a, 1.0 / norm(a));
      int smallest = 0;
      for (int c = 1; c < 3; ++c) {
        if (std::fabs(a[static_cast<std::size_t>(c)]) < std::fabs(a[static_cast<std::size_t>(smallest)])) smallest = c;
      }
      Vec3 trial{0, 0, 0};
      trial[static_cast<std::size_t>(smallest)] = 1.0;
      Vec3 u = cross(a, trial);
      u = scale(u, 1.0 / norm(u));
      Vec3 v = cross(a, u);
      completion[0] = u;
      completion[1] = scale(v, 1.0 / norm(v));
    }
    for (int k = 0; k < n_inactive; ++k) {
      for (int c = 0; c < 3; ++c) {
        lattice.basis[static_cast<std::size_t>(3 * inactive[k] + c)] =
            completion[k][static_cast<std::size_t>(c)];
      }
    }
  }
  lattice.inverse = inverse(lattice.basis);
  const double volume = std::fabs(det(lattice.basis));
  for (int k = 0; k < 3; ++k) {
    const double area = norm(cross(row(lattice.basis, (k + 1) % 3), row(lattice.basis, (k + 2) % 3)));
    lattice.spacing[static_cast<std::size_t>(k)] = volume / area;
  }
  return lattice;
}

std::array<int, 3> image_ranges(const Lattice& lattice, const double* positions,
                                std::size_t n_atoms, double cutoff) {
  std::array<int, 3> ranges{0, 0, 0};
  if (lattice.n_active == 0 || n_atoms == 0) return ranges;
  for (int k = 0; k < 3; ++k) {
    if (!lattice.pbc[static_cast<std::size_t>(k)]) continue;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    for (std::size_t i = 0; i < n_atoms; ++i) {
      double f = 0.0;
      for (int c = 0; c < 3; ++c) {
        f += positions[3 * i + static_cast<std::size_t>(c)] *
             lattice.inverse[static_cast<std::size_t>(3 * c + k)];
      }
      lo = std::min(lo, f);
      hi = std::max(hi, f);
    }
    const double bound = std::ceil(cutoff / lattice.spacing[static_cast<std::size_t>(k)]) + std::ceil(hi - lo);
    if (!(bound < (1 << 20))) {
      fail(ErrorCode::InvalidGeometry,
           "the cell is too thin for the cutoff (more than 2^20 images along a direction)");
    }
    ranges[static_cast<std::size_t>(k)] = static_cast<int>(bound);
  }
  return ranges;
}

}  // namespace vanderfast
