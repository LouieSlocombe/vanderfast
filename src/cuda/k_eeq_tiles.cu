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

// Screened real-space sum over the sorted translations for one displacement.
__device__ double real_value(const EwaldView& e, V3 vec, double gam) {
  const double limit = reach(e, gam);
  const double limit2 = limit * limit;
  const double cut = screen(e);
  const double vlen = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
  double total = 0.0;
  for (int t = 0; t < e.n_trans; ++t) {
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
    if (e.mode == kModeUpstream) {
      const double fg = xg > cut ? 1.0 : erf(xg);
      const double fa = xa > cut ? 1.0 : erf(xa);
      total += (fg - fa) * inv;
    } else {
      const double ca = xa > cut ? 0.0 : erfc(xa);
      const double cg = xg > cut ? 0.0 : erfc(xg);
      total += (ca - cg) * inv;
    }
  }
  return total;
}

// Derivative of real_value by the displacement (dg) and a homogeneous strain
// (ds: xx, yy, zz, yz, xz, xy), accumulated with weight ``scale``.
__device__ void real_derivative(const EwaldView& e, V3 vec, double gam, double scale, double dg[3],
                                double ds[6]) {
  const double limit = reach(e, gam);
  const double limit2 = limit * limit;
  const double cut = screen(e);
  const double vlen = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
  const double gam2 = gam * gam;
  const double alp2 = e.alpha * e.alpha;
  for (int t = 0; t < e.n_trans; ++t) {
    const double* tr = e.trans_sorted + 4 * t;
    if (tr[3] - vlen > limit) break;
    const double vx = vec.x + tr[0];
    const double vy = vec.y + tr[1];
    const double vz = vec.z + tr[2];
    const double r2 = vx * vx + vy * vy + vz * vz;
    if (r2 > limit2 || r2 < kEwaldEps * kEwaldEps) continue;
    const double inv = rsqrt(r2);
    const double inv2 = inv * inv;
    const double r1 = r2 * inv;
    const double xg = gam * r1;
    const double xa = e.alpha * r1;
    // Gaussian slopes 2 gam exp(-gam^2 r^2) and 2 alpha exp(-alpha^2 r^2).
    const double eg = xg > kExpZero ? 0.0 : 2 * gam * exp(-r2 * gam2);
    const double ea = xa > kExpZero ? 0.0 : 2 * e.alpha * exp(-r2 * alp2);
    double dtmp;
    if (e.mode == kModeUpstream) {
      const double fg = xg > cut ? 1.0 : erf(xg);
      const double fa = xa > cut ? 1.0 : erf(xa);
      dtmp = (eg - ea) * inv2 / kSqrtPi + (fa - fg) * inv2 * inv;
    } else {
      const double ca = xa > cut ? 0.0 : erfc(xa);
      const double cg = xg > cut ? 0.0 : erfc(xg);
      dtmp = ((eg - ea) / kSqrtPi - (ca - cg) * inv) * inv2;
    }
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

__device__ __forceinline__ double pair_gamma(const D4View& d, int si, int sj) {
  return 1.0 / sqrt(d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
}

// Real-space (periodic) or complete (molecular) matrix element of a pair.
__device__ double pair_real(const GeometryView& g, const D4View& d, const EwaldView& e, int i, int j) {
  const int si = species(g, i);
  const int sj = species(g, j);
  const V3 xi = pos3(g, i);
  const V3 xj = pos3(g, j);
  const V3 vec0{xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
  if (e.mode == kModeMolecular) {
    const double r2 = vec0.x * vec0.x + vec0.y * vec0.y + vec0.z * vec0.z;
    const double gam = 1.0 / (d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
    return erf(sqrt(r2 * gam)) / sqrt(r2);
  }
  const double gam = pair_gamma(d, si, sj);
  if (e.mode == kModeConverged) return real_value(e, vec0, gam);
  int count = 0;
  const unsigned mask = ws_mask(e, xi, xj, count);
  const double wsw = 1.0 / static_cast<double>(count);
  double total = 0.0;
  for (int c = 0; c < kWsCandidates; ++c) {
    if (!((mask >> c) & 1u)) continue;
    const double* t = e.ws_candidates + 3 * c;
    total += real_value(e, {vec0.x + t[0], vec0.y + t[1], vec0.z + t[2]}, gam) * wsw;
  }
  return total;
}

// d(element)/d(r_j - r_i) and its strain derivative, real space only for
// periodic systems.
__device__ void pair_real_derivative(const GeometryView& g, const D4View& d, const EwaldView& e, int i, int j,
                                     double dg[3], double ds[6]) {
  for (int c = 0; c < 3; ++c) dg[c] = 0.0;
  for (int c = 0; c < 6; ++c) ds[c] = 0.0;
  const int si = species(g, i);
  const int sj = species(g, j);
  const V3 xi = pos3(g, i);
  const V3 xj = pos3(g, j);
  const V3 vec0{xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
  const double gam = pair_gamma(d, si, sj);
  if (e.mode == kModeMolecular) {
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
    return;
  }
  if (e.mode == kModeConverged) {
    real_derivative(e, vec0, gam, 1.0, dg, ds);
    return;
  }
  int count = 0;
  const unsigned mask = ws_mask(e, xi, xj, count);
  const double wsw = 1.0 / static_cast<double>(count);
  for (int c = 0; c < kWsCandidates; ++c) {
    if (!((mask >> c) & 1u)) continue;
    const double* t = e.ws_candidates + 3 * c;
    real_derivative(e, {vec0.x + t[0], vec0.y + t[1], vec0.z + t[2]}, gam, wsw, dg, ds);
  }
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

__global__ void __launch_bounds__(kTileThreads)
    eeq_assemble_tiles_kernel(GeometryView g, D4View d, EwaldView e, const double* __restrict__ sf_c,
                              const double* __restrict__ sf_s, double* __restrict__ k) {
  __shared__ double out[kTile][kTile + 1];
  __shared__ double row_c[kChunk][kTile];
  __shared__ double row_s[kChunk][kTile];
  __shared__ double col_c[kChunk][kTile];
  __shared__ double col_s[kChunk][kTile];
  int ti;
  int tj;
  tile_coords(static_cast<int>(blockIdx.x), ti, tj);
  const int n = g.n;
  const long long np = n + 1;
  const int lane = lane_id();
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int j = tj * kTile + lane;
  // Each thread accumulates its four elements in the shared tile.
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int i = ti * kTile + r;
    out[r][lane] = (i < n && j < n && i > j) ? pair_real(g, d, e, i, j) : 0.0;
  }
  if (e.mode != kModeMolecular) {
    for (int g0 = 0; g0 < e.n_recip; g0 += kChunk) {
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
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int i = ti * kTile + r;
    if (i < n && j < n && i > j) k[i * np + j] = out[r][lane];
  }
  __syncthreads();
  // Mirror through shared memory so the transposed writes stay coalesced.
#pragma unroll
  for (int q = 0; q < kRowsPerWarp; ++q) {
    const int r = warp + q * kTileWarps;
    const int jj = tj * kTile + r;
    const int ii = ti * kTile + lane;
    if (ii < n && jj < n && ii > jj) k[jj * np + ii] = out[lane][r];
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

__global__ void __launch_bounds__(kTileThreads)
    eeq_reverse_tiles_kernel(GeometryView g, D4View d, EwaldView e, const double* __restrict__ q,
                             const double* __restrict__ nu, double* __restrict__ tiles) {
  __shared__ double col[kTileWarps][kTile][3];
  __shared__ double vir_s[kTileWarps][6];
  int ti;
  int tj;
  tile_coords(static_cast<int>(blockIdx.x), ti, tj);
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
    if (i < n && j < n && i > j) {
      pair_real_derivative(g, d, e, i, j, dg, ds);
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
                                         const double* __restrict__ sums, AccumView acc) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = g.n;
  if (i >= n) return;
  const int t_rows = (n + kTile - 1) / kTile;
  const int ti = i / kTile;
  const int slot = i % kTile;
  double grad[3] = {0.0, 0.0, 0.0};
  for (int tj = 0; tj <= ti; ++tj) {
    const double* record = tiles + static_cast<long long>(tile_index(ti, tj)) * kTileRecord;
    for (int c = 0; c < 3; ++c) grad[c] += record[slot * 3 + c];
  }
  for (int t2 = ti; t2 < t_rows; ++t2) {
    const double* record = tiles + static_cast<long long>(tile_index(t2, ti)) * kTileRecord;
    for (int c = 0; c < 3; ++c) grad[c] += record[kTile * 3 + slot * 3 + c];
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

void launch_eeq_assemble_tiles(const GeometryView& g, const D4View& d, const EwaldView& e, const double* sf_c,
                               const double* sf_s, double* k, cudaStream_t stream) {
  if (g.n > 1) {
    eeq_assemble_tiles_kernel<<<static_cast<unsigned>(tile_count(g.n)), kTileThreads, 0, stream>>>(g, d, e, sf_c,
                                                                                                 sf_s, k);
    VF_LAUNCH_CHECK(stream);
  }
  eeq_diagonal_kernel<<<static_cast<unsigned>((g.n + 1 + 255) / 256), 256, 0, stream>>>(g, d, e, k);
  VF_LAUNCH_CHECK(stream);
}

std::size_t eeq_reverse_tiles_scratch(int n, int n_recip) {
  return static_cast<std::size_t>(tile_count(n)) * kTileRecord + 5 * static_cast<std::size_t>(n_recip);
}

void launch_eeq_reverse_tiles(const GeometryView& g, const D4View& d, const EwaldView& e, const double* sf_c,
                              const double* sf_s, const double* q, const double* nu, const AccumView& acc,
                              double* scratch, cudaStream_t stream) {
  const int n = g.n;
  if (n == 0) return;
  const int tiles = tile_count(n);
  double* sums = scratch + static_cast<std::size_t>(tiles) * kTileRecord;
  const int recip = e.mode == kModeMolecular ? 0 : e.n_recip;
  if (recip > 0) {
    structure_sums_kernel<<<static_cast<unsigned>(recip), kTileThreads, 0, stream>>>(n, sf_c, sf_s, q, nu, sums);
    VF_LAUNCH_CHECK(stream);
  }
  eeq_reverse_tiles_kernel<<<static_cast<unsigned>(tiles), kTileThreads, 0, stream>>>(g, d, e, q, nu, scratch);
  VF_LAUNCH_CHECK(stream);
  eeq_reverse_atoms_kernel<<<static_cast<unsigned>((n + 127) / 128), 128, 0, stream>>>(g, e, sf_c, sf_s, q, nu,
                                                                                       scratch, sums, acc);
  VF_LAUNCH_CHECK(stream);
  eeq_reverse_virial_kernel<<<1, kTileThreads, 0, stream>>>(n, tiles, e, scratch, sums, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
