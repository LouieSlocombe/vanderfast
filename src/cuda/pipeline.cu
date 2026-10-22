#include "pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "../common/d3_math.hpp"
#include "vanderfast/error.hpp"

namespace vanderfast::cuda {

namespace {

constexpr long long kBruteForceLimit = 1LL << 24;  // candidate pairs for the O(N^2) builder

// Bin the search basis with bins of about half the largest radius, capped at
// max(64, 4N) bins. The stencil reach is conservative for every direction.
CellGrid make_grid(const Lattice& lattice, const double* x, std::size_t n, double rmax) {
  CellGrid grid;
  std::memcpy(grid.inverse, lattice.inverse.data(), sizeof(grid.inverse));
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double s = x[3 * i] * grid.inverse[k] + x[3 * i + 1] * grid.inverse[3 + k] +
                       x[3 * i + 2] * grid.inverse[6 + k];
      lo[k] = i == 0 ? s : std::min(lo[k], s);
      hi[k] = i == 0 ? s : std::max(hi[k], s);
    }
  }
  const double width = rmax / 2.0;
  double extent[3];
  for (int k = 0; k < 3; ++k) {
    grid.periodic[k] = lattice.pbc[static_cast<std::size_t>(k)] ? 1 : 0;
    if (grid.periodic[k]) {
      if (std::max(std::fabs(lo[k]), std::fabs(hi[k])) > 1.0e9) {
        fail(ErrorCode::InvalidGeometry, "positions are too far outside the periodic cell");
      }
      extent[k] = lattice.spacing[static_cast<std::size_t>(k)];
    } else {
      extent[k] = hi[k] - lo[k];
      grid.origin[k] = lo[k];
    }
    grid.nb[k] = extent[k] > 0.0 ? std::max(1, static_cast<int>(std::floor(extent[k] / width))) : 1;
  }
  const long long limit = std::max<long long>(64, 4 * static_cast<long long>(n));
  while (grid.bins() > limit) {
    int largest = 0;
    for (int k = 1; k < 3; ++k) {
      if (grid.nb[k] > grid.nb[largest]) largest = k;
    }
    grid.nb[largest] = std::max(1, grid.nb[largest] / 2);
  }
  for (int k = 0; k < 3; ++k) {
    if (grid.periodic[k]) {
      grid.scale[k] = grid.nb[k];
      grid.reach[k] = static_cast<int>(std::floor(rmax * grid.nb[k] / extent[k])) + 1;
    } else if (extent[k] > 0.0) {
      grid.scale[k] = grid.nb[k] / extent[k];
      grid.reach[k] = std::min(static_cast<int>(std::floor(rmax * grid.scale[k])) + 1, grid.nb[k] - 1);
    } else {
      grid.scale[k] = 0.0;
      grid.reach[k] = 0;
    }
    if (grid.reach[k] > (1 << 20)) {
      fail(ErrorCode::InvalidGeometry, "the cell is too thin for the cutoff");
    }
  }
  return grid;
}

}  // namespace

int checked_device(int device) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    cudaGetLastError();
    fail(ErrorCode::CudaUnavailable,
         std::string("no CUDA device is available (") + cudaGetErrorString(status) + ")");
  }
  if (device < 0 || device >= count) {
    fail(ErrorCode::InvalidArgument, "CUDA device " + std::to_string(device) + " does not exist (" +
                                         std::to_string(count) + " available)");
  }
  return device;
}

bool NeighborCache::needs_rebuild(const Geometry& geometry, const Lattice& lattice,
                                  const std::array<double, kViewCount>& radius, double skin) const {
  if (!valid || cell_ != geometry.cell || pbc_ != geometry.pbc || radius_ != radius ||
      !(skin > 0.0)) {
    return true;
  }
  for (int k = 0; k < 3; ++k) {
    // A lattice-vector wrap of an atom must exceed the skin to be detected.
    if (geometry.pbc[static_cast<std::size_t>(k)] &&
        skin >= 2.0 * lattice.spacing[static_cast<std::size_t>(k)]) {
      return true;
    }
  }
  double max_d2 = 0.0;
  for (std::size_t i = 0; i < 3 * geometry.n_atoms; i += 3) {
    const double dx = geometry.positions[i] - positions_[i];
    const double dy = geometry.positions[i + 1] - positions_[i + 1];
    const double dz = geometry.positions[i + 2] - positions_[i + 2];
    max_d2 = std::max(max_d2, dx * dx + dy * dy + dz * dz);
  }
  return 4.0 * max_d2 >= skin * skin;
}

void NeighborCache::ensure(const Geometry& geometry, const Lattice& lattice, const GeometryView& g,
                           const std::array<double, kViewCount>& radius,
                           const ContextOptions& options, bool canonical_atm, EvalInfo& info,
                           ContextStats& stats) {
  if (!needs_rebuild(geometry, lattice, radius, options.skin)) {
    info.builder = builder_;
    return;
  }
  const std::size_t n = geometry.n_atoms;
  valid = false;
  BandSpec bands = make_bands(radius);
  bands.canonical_atm = canonical_atm ? 1 : 0;
  const double rmax = std::max({radius[0], radius[1], radius[2]});
  ImageBox box;
  const std::array<int, 3> range = image_ranges(lattice, geometry.positions, n, rmax);
  box.images = 1;
  for (int k = 0; k < 3; ++k) {
    box.range[k] = range[static_cast<std::size_t>(k)];
    box.images = checked_mul<long long>(box.images, 2LL * box.range[k] + 1, "image count");
  }
  const long long candidates = checked_mul<long long>(
      checked_mul<long long>(box.images, static_cast<long long>(n), "candidate count"),
      static_cast<long long>(n), "candidate count");
  const bool brute = options.builder == NeighborBuilder::BruteForce ||
                     (options.builder == NeighborBuilder::Auto && candidates <= kBruteForceLimit);
  const double min_r2 =
      brute ? graph.build_brute_force(g, box, bands)
            : graph.build_cell_bins(g, make_grid(lattice, geometry.positions, n, rmax), bands);
  builder_ = brute ? "brute_force" : "cell_bins";
  stats.rebuilds += 1;
  if (options.reject_coincident && min_r2 < math::kCnExclusionR2) {
    fail(ErrorCode::CoincidentSites,
         "two distinct lattice sites are closer than 1e-6 bohr; the input geometry is unphysical");
  }
  positions_.assign(geometry.positions, geometry.positions + 3 * n);
  cell_ = geometry.cell;
  pbc_ = geometry.pbc;
  radius_ = radius;
  valid = true;
  info.rebuilt = true;
  info.builder = builder_;
}

}  // namespace vanderfast::cuda
