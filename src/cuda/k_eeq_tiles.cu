// Optimised EEQ Coulomb matrix and reverse pass (kernels="optimized").
//
// Pairs are processed in 32 x 32 tiles of the lower triangle (i > j), one
// block per tile. Real-space Ewald sums loop over the translations sorted by
// length and stop once no further image can contribute: beyond
// r = 6.5 / min(alpha, gamma) both error functions equal one (erfc below
// 4e-20), so the term vanishes. The reciprocal sum is separable,
// sum_G e_G cos(G.(r_j - r_i)) = sum_G e_G (c_i c_j + s_i s_j), and is the
// same for every Wigner-Seitz image because G.T is a multiple of 2 pi.
//
// Screening: upstream mode drops a translation once both erf arguments exceed
// 5.9, where the correctly rounded erf of the reference is exactly one, so
// the value term vanishes there too (the dropped derivative terms are below
// 1e-20). Converged mode keeps terms down to erfc(6.5) = 4e-20.
//
// The reverse pass writes per-tile row and column partials and reduces them
// per atom in a fixed order, so gradients and virials are deterministic.
//
// Small periodic systems have too few tiles to fill the device, so each
// tile's translations (and reciprocal chunks) can be split over ``slices``
// blocks whose partial results are combined in slice order.
#include <algorithm>
#include <cfloat>

#include "cuda_check.cuh"
#include "d4_kernels.cuh"
#include "device_util.cuh"
#include "eeq_device.cuh"

namespace vanderfast::cuda {

namespace {

using namespace eeq;  // NOLINT(google-build-using-namespace)

constexpr int kTile = 32;
constexpr int kTileWarps = 8;
constexpr int kTileThreads = kTileWarps * kWarpSize;
constexpr int kRowsPerWarp = kTile / kTileWarps;
constexpr int kChunk = 16;       // reciprocal vectors staged per pass
constexpr int kMaxSlices = 16;
constexpr double kErfOne = 5.9;        // erf(x) rounds to 1 beyond (erfc(5.9) = 3.7e-17 < 2^-54)
constexpr double kConvergedScreen = 6.5;  // erfc(x) < 4e-20 beyond
constexpr double kExpZero = 6.8;        // exp(-x^2) < 1e-20 beyond

__device__ __forceinline__ void tile_coords(int t, int& ti, int& tj) {
  int row = static_cast<int>((sqrt(8.0 * t + 1.0) - 1.0) * 0.5);
  while (row * (row + 1) / 2 > t) --row;
  while ((row + 1) * (row + 2) / 2 <= t) ++row;
  ti = row;
  tj = t - row * (row + 1) / 2;
}

__device__ __forceinline__ int tile_index(int ti, int tj) { return ti * (ti + 1) / 2 + tj; }

// The translations t = index, index + count, ... of one block.
struct Slice {
  int index;
  int count;
};

// Wigner-Seitz candidates T minimising |r_a - r_b - T|^2 within 0.01 bohr^2,
// as a bit mask over the 27 candidates (two passes, no stored distances).
__device__ unsigned ws_mask(const EwaldView& e, V3 xa, V3 xb, int& count) {
  const double rx = xa.x - xb.x;
  const double ry = xa.y - xb.y;
  const double rz = xa.z - xb.z;
  double best = DBL_MAX;
  for (int c = 0; c < kWsCandidates; ++c) {
    const double* t = e.ws_candidates + 3 * c;
    const double vx = rx - t[0];
    const double vy = ry - t[1];
    const double vz = rz - t[2];
    best = fmin(best, vx * vx + vy * vy + vz * vz);
  }
  unsigned mask = 0u;
  count = 0;
  for (int c = 0; c < kWsCandidates; ++c) {
    const double* t = e.ws_candidates + 3 * c;
    const double vx = rx - t[0];
    const double vy = ry - t[1];
    const double vz = rz - t[2];
    if (fabs((vx * vx + vy * vy + vz * vz) - best) <= kWsTol) {
      mask |= 1u << c;
      count += 1;
    }
  }
  return mask;
}

__device__ __forceinline__ double screen(const EwaldView& e) {
  return e.mode == kModeConverged ? kConvergedScreen : kErfOne;
}

// Largest image distance with a non-vanishing real-space term.
__device__ __forceinline__ double reach(const EwaldView& e, double gam) {
  const double screened = screen(e) / fmin(e.alpha, gam);
  return e.mode == kModeConverged ? fmin(screened, e.real_cutoff) : screened;
}

// Screened real-space sums over the sorted translations for one displacement:
// the value (when Value) and, with weight ``scale``, its derivative by the
// displacement (dg) and a homogeneous strain (ds: xx, yy, zz, yz, xz, xy)
// (when Deriv). Both share the error functions.
template <bool Value, bool Deriv>
__device__ __forceinline__ void real_sums(const EwaldView& e, V3 vec, double gam, double scale, Slice slice,
                                          double& value, double dg[3], double ds[6]) {
  const double limit = reach(e, gam);
  const double limit2 = limit * limit;
  const double cut = screen(e);
  const double vlen = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
  const double gam2 = gam * gam;
  const double alp2 = e.alpha * e.alpha;
  for (int t = slice.index; t < e.n_trans; t += slice.count) {
    const double* tr = e.trans_sorted + 4 * t;
    if (tr[3] - vlen > limit) break;  // |vec + T| >= |T| - |vec|
    const double vx = vec.x + tr[0];
    const double vy = vec.y + tr[1];
    const double vz = vec.z + tr[2];
    const double r2 = vx * vx + vy * vy + vz * vz;
    if (r2 > limit2 || r2 < kEwaldEps * kEwaldEps) continue;
    const double inv = rsqrt(r2);
    const double r1 = r2 * inv;
    const double xg = gam * r1;
    const double xa = e.alpha * r1;
    // f: erf (upstream) or erfc (converged) of both arguments, screened.
    double fg;
    double fa;
    if (e.mode == kModeUpstream) {
      fg = xg > cut ? 1.0 : erf(xg);
      fa = xa > cut ? 1.0 : erf(xa);
      if (Value) value += (fg - fa) * inv;
    } else {
      fa = xa > cut ? 0.0 : erfc(xa);
      fg = xg > cut ? 0.0 : erfc(xg);
      if (Value) value += (fa - fg) * inv;
    }
    if (Deriv) {
      const double inv2 = inv * inv;
      // Gaussian slopes 2 gam exp(-gam^2 r^2) and 2 alpha exp(-alpha^2 r^2).
      const double eg = xg > kExpZero ? 0.0 : 2 * gam * exp(-r2 * gam2);
      const double ea = xa > kExpZero ? 0.0 : 2 * e.alpha * exp(-r2 * alp2);
      double dtmp = e.mode == kModeUpstream ? (eg - ea) * inv2 / kSqrtPi + (fa - fg) * inv2 * inv
                                            : ((eg - ea) / kSqrtPi - (fa - fg) * inv) * inv2;
      dtmp *= scale;
      dg[0] += dtmp * vx;
      dg[1] += dtmp * vy;
      dg[2] += dtmp * vz;
      ds[0] += dtmp * vx * vx;
      ds[1] += dtmp * vy * vy;
      ds[2] += dtmp * vz * vz;
      ds[3] += dtmp * vy * vz;
      ds[4] += dtmp * vx * vz;
      ds[5] += dtmp * vx * vy;
    }
  }
}

__device__ __forceinline__ double pair_gamma(const D4View& d, int si, int sj) {
  return 1.0 / sqrt(d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
}

// Real-space sums of a periodic pair (Wigner-Seitz averaged in upstream mode):
// the matrix element (Value) and d(element)/d(r_j - r_i) with its strain
// derivative (Deriv), both zeroed first.
template <bool Value, bool Deriv>
__device__ __forceinline__ void pair_periodic(const GeometryView& g, const D4View& d, const EwaldView& e, int i,
                                              int j, Slice slice, double& value, double dg[3], double ds[6]) {
  value = 0.0;
  for (int c = 0; c < 3; ++c) dg[c] = 0.0;
  for (int c = 0; c < 6; ++c) ds[c] = 0.0;
  const V3 xi = pos3(g, i);
  const V3 xj = pos3(g, j);
  const V3 vec0{xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
  const double gam = pair_gamma(d, species(g, i), species(g, j));
  if (e.mode == kModeConverged) {
    real_sums<Value, Deriv>(e, vec0, gam, 1.0, slice, value, dg, ds);
    return;
  }
  int count = 0;
  const unsigned mask = ws_mask(e, xi, xj, count);
  const double wsw = 1.0 / static_cast<double>(count);
  for (int c = 0; c < kWsCandidates; ++c) {
    if (!((mask >> c) & 1u)) continue;
    const double* t = e.ws_candidates + 3 * c;
    double part = 0.0;
    real_sums<Value, Deriv>(e, {vec0.x + t[0], vec0.y + t[1], vec0.z + t[2]}, gam, wsw, slice, part, dg, ds);
    value += part * wsw;
  }
}

// Real-space (periodic) or complete (molecular) matrix element of a pair.
__device__ __forceinline__ double pair_real(const GeometryView& g, const D4View& d, const EwaldView& e, int i,
                                            int j, Slice slice) {
  if (e.mode == kModeMolecular) {  // molecular systems use one slice
    const int si = species(g, i);
    const int sj = species(g, j);
    const V3 xi = pos3(g, i);
    const V3 xj = pos3(g, j);
    const double dx = xj.x - xi.x;
    const double dy = xj.y - xi.y;
    const double dz = xj.z - xi.z;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double gam = 1.0 / (d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
    return erf(sqrt(r2 * gam)) / sqrt(r2);
  }
  double value;
  double dg[3];
  double ds[6];
  pair_periodic<true, false>(g, d, e, i, j, slice, value, dg, ds);
  return value;
}

// d(element)/d(r_j - r_i) and its strain derivative, real space only for
// periodic systems.
__device__ void pair_real_derivative(const GeometryView& g, const D4View& d, const EwaldView& e, int i, int j,
                                     Slice slice, double dg[3], double ds[6]) {
  if (e.mode != kModeMolecular) {
    double value;
    pair_periodic<false, true>(g, d, e, i, j, slice, value, dg, ds);
    return;
  }
  const V3 xi = pos3(g, i);
  const V3 xj = pos3(g, j);
  const V3 vec0{xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
  const double gam = pair_gamma(d, species(g, i), species(g, j));
  const double r2 = vec0.x * vec0.x + vec0.y * vec0.y + vec0.z * vec0.z;
  const double arg = gam * gam * r2;
  const double dtmp = 2.0 * gam * exp(-arg) / (kSqrtPi * r2) - erf(sqrt(arg)) / (r2 * sqrt(r2));
  dg[0] = dtmp * vec0.x;
  dg[1] = dtmp * vec0.y;
  dg[2] = dtmp * vec0.z;
  ds[0] = dtmp * vec0.x * vec0.x;
  ds[1] = dtmp * vec0.y * vec0.y;
  ds[2] = dtmp * vec0.z * vec0.z;
  ds[3] = dtmp * vec0.y * vec0.z;
  ds[4] = dtmp * vec0.x * vec0.z;
  ds[5] = dtmp * vec0.x * vec0.y;
}

__global__ void structure_factor_kernel(GeometryView g, EwaldView e, double* __restrict__ c,
                                        double* __restrict__ s) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(g.n) * e.n_recip;
  if (idx >= total) return;
  const int k = static_cast<int>(idx / g.n);
  const int i = static_cast<int>(idx % g.n);
  const double* G = e.rtrans + 3 * k;
  const V3 x = pos3(g, i);
  double sine;
  double cosine;
  sincos_phase(x.x * G[0] + x.y * G[1] + x.z * G[2], sine, cosine);
  c[idx] = cosine;
  s[idx] = sine;
}

// Write a lower-triangle tile held in ``out`` to K and mirror it through
// shared memory so the transposed writes stay coalesced.
__device__ __forceinline__ void store_tile(double (*out)[kTile + 1], int ti, int tj, int n, double* __restrict__ k) {
  const long long np = n + 1;
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int j = tj * kTile + lane;
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int i = ti * kTile + r;
    if (i < n && j < n && i > j) k[i * np + j] = out[r][lane];
  }
  __syncthreads();
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int jj = tj * kTile + r;
    const int ii = ti * kTile + lane;
    if (ii < n && jj < n && ii > jj) k[jj * np + ii] = out[lane][r];
  }
}

__device__ __forceinline__ double diagonal(const GeometryView& g, const D4View& d, const EwaldView& e, int i) {
  const int si = species(g, i);
  return e.mode == kModeMolecular ? d.eta[si] + kSqrt2Pi / d.rad[si] : e.self_diag[si];
}

// Packed storage for MINRES: lower-triangle tile (I, J), I >= J, as a
// row-major 32 x 32 block at tile_index(I, J) * 1024. Diagonal tiles hold
// both triangles and the diagonal; entries beyond n are zero; the border
// (all ones) is implicit.
__device__ __forceinline__ void store_packed(double (*out)[kTile + 1], int tile, int ti, int tj,
                                             const GeometryView& g, const D4View& d, const EwaldView& e,
                                             double* __restrict__ kp) {
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  double* dst = kp + static_cast<long long>(tile) * kTile * kTile;
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int i = ti * kTile + r;
    const int j = tj * kTile + lane;
    double value = 0.0;
    if (i < g.n && j < g.n) {
      if (ti != tj || r > lane) {
        value = out[r][lane];
      } else if (r < lane) {
        value = out[lane][r];
      } else {
        value = diagonal(g, d, e, i);
      }
    }
    dst[r * kTile + lane] = value;
  }
}

// Block b handles tile b % tiles and slice b / tiles. With one slice the
// tile goes straight to K (dense, or packed when ``packed``); otherwise to
// ``partial`` [slices][tiles][32 * 32]. With Store (periodic systems) the
// real-space derivatives of every pair go to ``deriv``
// [slices][tiles][9][32 * 32] (dg then ds) for the reverse pass.
// (Two resident blocks per SM: ptxas would otherwise spill to fit three.)
template <bool Store>
__global__ void __launch_bounds__(kTileThreads, 2)
    eeq_assemble_tiles_kernel(GeometryView g, D4View d, EwaldView e, const double* __restrict__ sf_c,
                              const double* __restrict__ sf_s, int tiles, int slices, bool packed,
                              double* __restrict__ k, double* __restrict__ partial, double* __restrict__ deriv) {
  __shared__ double out[kTile][kTile + 1];
  __shared__ double row_c[kChunk][kTile];
  __shared__ double row_s[kChunk][kTile];
  __shared__ double col_c[kChunk][kTile];
  __shared__ double col_s[kChunk][kTile];
  const int tile = static_cast<int>(blockIdx.x) % tiles;
  const Slice slice{static_cast<int>(blockIdx.x) / tiles, slices};
  int ti;
  int tj;
  tile_coords(tile, ti, tj);
  const int n = g.n;
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int j = tj * kTile + lane;
  // Each thread accumulates its four elements in the shared tile.
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int i = ti * kTile + r;
    const bool live = i < n && j < n && i > j && (e.mode != kModeMolecular || slice.index == 0);
    if (Store) {
      double value = 0.0;
      double dg[3] = {0.0, 0.0, 0.0};
      double ds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      if (live) pair_periodic<true, true>(g, d, e, i, j, slice, value, dg, ds);
      out[r][lane] = value;
      double* dst = deriv + (static_cast<long long>(slice.index) * tiles + tile) * 9 * kTile * kTile + r * kTile + lane;
      for (int c = 0; c < 3; ++c) dst[c * kTile * kTile] = dg[c];
      for (int c = 0; c < 6; ++c) dst[(3 + c) * kTile * kTile] = ds[c];
    } else {
      out[r][lane] = live ? pair_real(g, d, e, i, j, slice) : 0.0;
    }
  }
  if (e.mode != kModeMolecular) {
    for (int g0 = slice.index * kChunk; g0 < e.n_recip; g0 += slices * kChunk) {
      for (int idx = static_cast<int>(threadIdx.x); idx < kChunk * kTile; idx += kTileThreads) {
        const int gg = idx / kTile;
        const int a = idx % kTile;
        const int ri = ti * kTile + a;
        const int cj = tj * kTile + a;
        const bool live = g0 + gg < e.n_recip;
        const long long base = static_cast<long long>(g0 + gg) * n;
        row_c[gg][a] = live && ri < n ? sf_c[base + ri] : 0.0;
        row_s[gg][a] = live && ri < n ? sf_s[base + ri] : 0.0;
        col_c[gg][a] = live && cj < n ? sf_c[base + cj] : 0.0;
        col_s[gg][a] = live && cj < n ? sf_s[base + cj] : 0.0;
      }
      __syncthreads();
      const int count = min(kChunk, e.n_recip - g0);
      for (int gg = 0; gg < count; ++gg) {
        const double eg = e.recip_e[g0 + gg];
        const double cj = col_c[gg][lane];
        const double sj = col_s[gg][lane];
#pragma unroll
        for (int q = 0; q < kRowsPerWarp; ++q) {
          const int r = warp + q * kTileWarps;
          out[r][lane] += eg * (row_c[gg][r] * cj + row_s[gg][r] * sj);
        }
      }
      __syncthreads();
    }
  }
  if (slices == 1) {
    __syncthreads();
    if (packed) {
      store_packed(out, tile, ti, tj, g, d, e, k);
    } else {
      store_tile(out, ti, tj, n, k);
    }
    return;
  }
  double* dst = partial + (static_cast<long long>(slice.index) * tiles + tile) * kTile * kTile;
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    dst[r * kTile + lane] = out[r][lane];
  }
}

// Sum the slices of each tile in slice order and store it.
__global__ void __launch_bounds__(kTileThreads)
    eeq_assemble_slices_kernel(GeometryView g, D4View d, EwaldView e, int tiles, int slices, bool packed,
                               const double* __restrict__ partial, double* __restrict__ k) {
  __shared__ double out[kTile][kTile + 1];
  const int tile = static_cast<int>(blockIdx.x);
  int ti;
  int tj;
  tile_coords(tile, ti, tj);
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    double total = 0.0;
    for (int s = 0; s < slices; ++s) {
      total += partial[(static_cast<long long>(s) * tiles + tile) * kTile * kTile + r * kTile + lane];
    }
    out[r][lane] = total;
  }
  __syncthreads();
  if (packed) {
    store_packed(out, tile, ti, tj, g, d, e, k);
  } else {
    store_tile(out, ti, tj, g.n, k);
  }
}

__global__ void eeq_diagonal_kernel(GeometryView g, D4View d, EwaldView e, double* __restrict__ k) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = g.n;
  const long long np = n + 1;
  if (i > n) return;
  if (i == n) {
    k[n * np + n] = 0.0;
    return;
  }
  const int si = species(g, i);
  k[i * np + i] = e.mode == kModeMolecular ? d.eta[si] + kSqrt2Pi / d.rad[si] : e.self_diag[si];
  k[i * np + n] = 1.0;
  k[n * np + i] = 1.0;
}

// Reverse layout of the scratch buffer: per tile 32 row and 32 column
// gradients (3 each) and 6 virial components, then 5 structure sums per G.
constexpr int kTileRecord = 2 * kTile * 3 + 6;

// Block b handles tile b % tiles and slice b / tiles; its record is record b.
// With Stored the pair derivatives come from the assembly pass (summed over
// its ``deriv_slices`` in order) and a single slice is used.
template <bool Stored>
__global__ void __launch_bounds__(kTileThreads)
    eeq_reverse_tiles_kernel(GeometryView g, D4View d, EwaldView e, const double* __restrict__ q,
                             const double* __restrict__ nu, int tile_total, int slices,
                             const double* __restrict__ deriv, int deriv_slices, double* __restrict__ tiles) {
  __shared__ double col[kTileWarps][kTile][3];
  __shared__ double vir_s[kTileWarps][6];
  const Slice slice{static_cast<int>(blockIdx.x) / tile_total, slices};
  int ti;
  int tj;
  tile_coords(static_cast<int>(blockIdx.x) % tile_total, ti, tj);
  const int n = g.n;
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int j = tj * kTile + lane;
  double* record = tiles + static_cast<long long>(blockIdx.x) * kTileRecord;
  double cg[3] = {0.0, 0.0, 0.0};
  double vir[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int qq = 0; qq < kRowsPerWarp; ++qq) {
    const int r = warp + qq * kTileWarps;
    const int i = ti * kTile + r;
    double dg[3] = {0.0, 0.0, 0.0};
    double ds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double w = 0.0;
    if (i < n && j < n && i > j && (e.mode != kModeMolecular || slice.index == 0)) {
      if (Stored) {
        const int tile = static_cast<int>(blockIdx.x) % tile_total;
        for (int s = 0; s < deriv_slices; ++s) {
          const double* src =
              deriv + (static_cast<long long>(s) * tile_total + tile) * 9 * kTile * kTile + r * kTile + lane;
          for (int c = 0; c < 3; ++c) dg[c] += src[c * kTile * kTile];
          for (int c = 0; c < 6; ++c) ds[c] += src[(3 + c) * kTile * kTile];
        }
      } else {
        pair_real_derivative(g, d, e, i, j, slice, dg, ds);
      }
      w = nu[i] * q[j] + nu[j] * q[i];
    }
    // Row atom i (the larger index) takes +w dg, column atom j takes -w dg.
    for (int c = 0; c < 3; ++c) {
      const double total = warp_sum(w * dg[c]);
      if (lane == 0) record[r * 3 + c] = total;
      cg[c] -= w * dg[c];
    }
    for (int c = 0; c < 6; ++c) vir[c] -= w * ds[c];
  }
  for (int c = 0; c < 3; ++c) col[warp][lane][c] = cg[c];
  for (int c = 0; c < 6; ++c) {
    const double total = warp_sum(vir[c]);
    if (lane == 0) vir_s[warp][c] = total;
  }
  __syncthreads();
  if (warp == 0) {
    for (int c = 0; c < 3; ++c) {
      double total = 0.0;
      for (int w = 0; w < kTileWarps; ++w) total += col[w][lane][c];
      record[kTile * 3 + lane * 3 + c] = total;
    }
  }
  if (threadIdx.x < 6) {
    double total = 0.0;
    for (int w = 0; w < kTileWarps; ++w) total += vir_s[w][threadIdx.x];
    record[2 * kTile * 3 + threadIdx.x] = total;
  }
}

// Per reciprocal vector: sum nu c, nu s, q c, q s, and nu q (c^2 + s^2).
__global__ void structure_sums_kernel(int n, const double* __restrict__ sf_c, const double* __restrict__ sf_s,
                                      const double* __restrict__ q, const double* __restrict__ nu,
                                      double* __restrict__ sums) {
  __shared__ double scratch[5][kTileWarps];
  const int k = static_cast<int>(blockIdx.x);
  const double* c = sf_c + static_cast<long long>(k) * n;
  const double* s = sf_s + static_cast<long long>(k) * n;
  double part[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = static_cast<int>(threadIdx.x); i < n; i += kTileThreads) {
    const double ci = c[i];
    const double si = s[i];
    part[0] += nu[i] * ci;
    part[1] += nu[i] * si;
    part[2] += q[i] * ci;
    part[3] += q[i] * si;
    part[4] += nu[i] * q[i] * (ci * ci + si * si);
  }
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  for (int m = 0; m < 5; ++m) {
    const double total = warp_sum(part[m]);
    if (lane_id() == 0) scratch[m][warp] = total;
  }
  __syncthreads();
  if (threadIdx.x < 5) {
    double total = 0.0;
    for (int w = 0; w < kTileWarps; ++w) total += scratch[threadIdx.x][w];
    sums[5 * k + threadIdx.x] = total;
  }
}

// Per atom: the tile partials of its row and column, the reciprocal gradient,
// and the self-image virial.
__global__ void eeq_reverse_atoms_kernel(GeometryView g, EwaldView e, const double* __restrict__ sf_c,
                                         const double* __restrict__ sf_s, const double* __restrict__ q,
                                         const double* __restrict__ nu, const double* __restrict__ tiles,
                                         int tile_total, int slices, const double* __restrict__ sums,
                                         AccumView acc) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = g.n;
  if (i >= n) return;
  const int t_rows = (n + kTile - 1) / kTile;
  const int ti = i / kTile;
  const int slot = i % kTile;
  double grad[3] = {0.0, 0.0, 0.0};
  for (int s = 0; s < slices; ++s) {
    const double* base = tiles + static_cast<long long>(s) * tile_total * kTileRecord;
    for (int tj = 0; tj <= ti; ++tj) {
      const double* record = base + static_cast<long long>(tile_index(ti, tj)) * kTileRecord;
      for (int c = 0; c < 3; ++c) grad[c] += record[slot * 3 + c];
    }
    for (int t2 = ti; t2 < t_rows; ++t2) {
      const double* record = base + static_cast<long long>(tile_index(t2, ti)) * kTileRecord;
      for (int c = 0; c < 3; ++c) grad[c] += record[kTile * 3 + slot * 3 + c];
    }
  }
  if (e.mode != kModeMolecular) {
    // -d/dr_i of sum_{a != b} nu_a q_b sum_G e_G cos(G.(r_b - r_a)), as the
    // pair loop books it: grad_i = -sum_G e_G G [nu_i (c_i S_q - s_i C_q) + q_i (c_i S_nu - s_i C_nu)].
    const double qi = q[i];
    const double ni = nu[i];
    for (int k = 0; k < e.n_recip; ++k) {
      const double* G = e.rtrans + 3 * k;
      const double* sk = sums + 5 * k;
      const long long at = static_cast<long long>(k) * n + i;
      const double ci = sf_c[at];
      const double si = sf_s[at];
      const double f = -e.recip_e[k] * (ni * (ci * sk[3] - si * sk[2]) + qi * (ci * sk[1] - si * sk[0]));
      grad[0] += f * G[0];
      grad[1] += f * G[1];
      grad[2] += f * G[2];
    }
    const double scale = ni * qi;
    const int si = species(g, i);
    for (int c = 0; c < 6; ++c) acc.vir[c * n + i] -= scale * e.self_vir[6 * si + c];
  }
  for (int c = 0; c < 3; ++c) acc.grad[3 * i + c] += grad[c];
}

// The virial of every tile and of the reciprocal sum, added to atom 0's slot.
__global__ void eeq_reverse_virial_kernel(int n, int tiles_count, EwaldView e, const double* __restrict__ tiles,
                                          const double* __restrict__ sums, AccumView acc) {
  __shared__ double scratch[6][kTileWarps];
  double part[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int t = static_cast<int>(threadIdx.x); t < tiles_count; t += kTileThreads) {
    const double* record = tiles + static_cast<long long>(t) * kTileRecord + 2 * kTile * 3;
    for (int c = 0; c < 6; ++c) part[c] += record[c];
  }
  if (e.mode != kModeMolecular) {
    const double alp2 = e.alpha * e.alpha;
    for (int k = static_cast<int>(threadIdx.x); k < e.n_recip; k += kTileThreads) {
      const double* G = e.rtrans + 3 * k;
      const double* sk = sums + 5 * k;
      const double g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
      const double gfac = 2.0 / g2 + 0.5 / alp2;
      const double f = -e.recip_e[k] * (sk[0] * sk[2] + sk[1] * sk[3] - sk[4]);
      part[0] += f * (gfac * G[0] * G[0] - 1.0);
      part[1] += f * (gfac * G[1] * G[1] - 1.0);
      part[2] += f * (gfac * G[2] * G[2] - 1.0);
      part[3] += f * (gfac * G[1] * G[2]);
      part[4] += f * (gfac * G[0] * G[2]);
      part[5] += f * (gfac * G[0] * G[1]);
    }
  }
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  for (int c = 0; c < 6; ++c) {
    const double total = warp_sum(part[c]);
    if (lane_id() == 0) scratch[c][warp] = total;
  }
  __syncthreads();
  if (threadIdx.x < 6) {
    double total = 0.0;
    for (int w = 0; w < kTileWarps; ++w) total += scratch[threadIdx.x][w];
    acc.vir[threadIdx.x * n] += total;
  }
}

int tile_count(int n) {
  const int rows = (n + kTile - 1) / kTile;
  return rows * (rows + 1) / 2;
}

}  // namespace

void launch_structure_factors(const GeometryView& g, const EwaldView& e, double* c, double* s,
                              cudaStream_t stream) {
  const long long total = static_cast<long long>(g.n) * e.n_recip;
  if (total == 0) return;
  structure_factor_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(g, e, c, s);
  VF_LAUNCH_CHECK(stream);
}

int eeq_tile_slices(int n, bool periodic, int device) {
  if (!periodic || n < 2) return 1;
  int sms = 0;
  VF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const int tiles = tile_count(n);
  return std::clamp((4 * sms + tiles - 1) / tiles, 1, kMaxSlices);
}

std::size_t eeq_assemble_tiles_scratch(int n, int slices) {
  return slices > 1 ? static_cast<std::size_t>(slices) * tile_count(n) * kTile * kTile : 0;
}

int eeq_packed_tiles(int n) { return tile_count(n); }

std::size_t eeq_derivative_scratch(int n, int slices) {
  return static_cast<std::size_t>(slices) * tile_count(n) * 9 * kTile * kTile;
}

void launch_eeq_assemble_tiles(const GeometryView& g, const D4View& d, const EwaldView& e, const double* sf_c,
                               const double* sf_s, int slices, double* scratch, bool packed, double* k,
                               double* deriv, cudaStream_t stream) {
  // Packed storage needs every tile, including the diagonal of a single atom.
  if (g.n > 1 || (packed && g.n == 1)) {
    const int tiles = tile_count(g.n);
    const unsigned blocks = static_cast<unsigned>(tiles * slices);
    if (deriv != nullptr && e.mode != kModeMolecular) {
      eeq_assemble_tiles_kernel<true><<<blocks, kTileThreads, 0, stream>>>(g, d, e, sf_c, sf_s, tiles, slices,
                                                                          packed, k, scratch, deriv);
    } else {
      eeq_assemble_tiles_kernel<false><<<blocks, kTileThreads, 0, stream>>>(g, d, e, sf_c, sf_s, tiles, slices,
                                                                           packed, k, scratch, nullptr);
    }
    VF_LAUNCH_CHECK(stream);
    if (slices > 1) {
      eeq_assemble_slices_kernel<<<static_cast<unsigned>(tiles), kTileThreads, 0, stream>>>(g, d, e, tiles, slices,
                                                                                         packed, scratch, k);
      VF_LAUNCH_CHECK(stream);
    }
  }
  if (packed) return;
  eeq_diagonal_kernel<<<static_cast<unsigned>((g.n + 1 + 255) / 256), 256, 0, stream>>>(g, d, e, k);
  VF_LAUNCH_CHECK(stream);
}

std::size_t eeq_reverse_tiles_scratch(int n, int n_recip, int slices) {
  return static_cast<std::size_t>(slices) * tile_count(n) * kTileRecord + 5 * static_cast<std::size_t>(n_recip);
}

void launch_eeq_reverse_tiles(const GeometryView& g, const D4View& d, const EwaldView& e, const double* sf_c,
                              const double* sf_s, const double* q, const double* nu, const AccumView& acc,
                              int slices, const double* deriv, double* scratch, cudaStream_t stream) {
  const int n = g.n;
  if (n == 0) return;
  // Stored derivatives need no lattice sums here, so one slice suffices.
  const bool stored = deriv != nullptr && e.mode != kModeMolecular;
  const int deriv_slices = slices;
  if (stored) slices = 1;
  const int tiles = tile_count(n);
  double* sums = scratch + static_cast<std::size_t>(slices) * tiles * kTileRecord;
  const int recip = e.mode == kModeMolecular ? 0 : e.n_recip;
  if (recip > 0) {
    structure_sums_kernel<<<static_cast<unsigned>(recip), kTileThreads, 0, stream>>>(n, sf_c, sf_s, q, nu, sums);
    VF_LAUNCH_CHECK(stream);
  }
  const unsigned blocks = static_cast<unsigned>(tiles * slices);
  if (stored) {
    eeq_reverse_tiles_kernel<true><<<blocks, kTileThreads, 0, stream>>>(g, d, e, q, nu, tiles, slices, deriv,
                                                                       deriv_slices, scratch);
  } else {
    eeq_reverse_tiles_kernel<false><<<blocks, kTileThreads, 0, stream>>>(g, d, e, q, nu, tiles, slices, nullptr, 0,
                                                                        scratch);
  }
  VF_LAUNCH_CHECK(stream);
  eeq_reverse_atoms_kernel<<<static_cast<unsigned>((n + 127) / 128), 128, 0, stream>>>(
      g, e, sf_c, sf_s, q, nu, scratch, tiles, slices, sums, acc);
  VF_LAUNCH_CHECK(stream);
  eeq_reverse_virial_kernel<<<1, kTileThreads, 0, stream>>>(n, tiles * slices, e, scratch, sums, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
