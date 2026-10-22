// EEQ charge model: the bordered Coulomb matrix, a dense LU solve, and the
// derivative of -nu^T A q (multicharge v0.5.0, model/eeq.f90).
//
// Pairs are evaluated once with the larger index as the centre (multicharge's
// iat > jat loop) and mirrored, so the matrix is exactly symmetric. Periodic
// pairs either average multicharge's fixed Ewald sums over the Wigner-Seitz
// images (upstream mode) or use complete lattice sums (converged mode).
#include "cuda_check.cuh"
#include "d4_kernels.cuh"
#include "device_util.cuh"
#include "eeq_device.cuh"

namespace vanderfast::cuda {

namespace {

using namespace eeq;  // NOLINT(google-build-using-namespace)

// Real-space kernel (erf(gam r) - erf(alpha r)) / r summed over translations,
// plus the reciprocal sum, for one displacement.
__device__ double ewald_value(const EwaldView& e, V3 vec, double gam) {
  double real = 0.0;
  for (int t = 0; t < e.n_trans; ++t) {
    const double* tr = e.dtrans + 3 * t;
    const double vx = vec.x + tr[0];
    const double vy = vec.y + tr[1];
    const double vz = vec.z + tr[2];
    const double r1 = sqrt(vx * vx + vy * vy + vz * vz);
    if (r1 < kEwaldEps) continue;
    if (e.mode == kModeUpstream) {
      real += erf(gam * r1) / r1 - erf(e.alpha * r1) / r1;
    } else {
      if (r1 > e.real_cutoff) continue;
      real += (erfc(e.alpha * r1) - erfc(gam * r1)) / r1;
    }
  }
  double rec = 0.0;
  const double fac = 4 * kPi / e.volume;
  for (int k = 0; k < e.n_recip; ++k) {
    const double* G = e.rtrans + 3 * k;
    const double g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const double gv = vec.x * G[0] + vec.y * G[1] + vec.z * G[2];
    rec += cos_phase(gv) * fac * exp(-0.25 * g2 / (e.alpha * e.alpha)) / g2;
  }
  return real + rec;
}

__device__ void ewald_derivative(const EwaldView& e, V3 vec, double gam, double dg[3], double ds[6]) {
  const double gam2 = gam * gam;
  const double alp2 = e.alpha * e.alpha;
  for (int t = 0; t < e.n_trans; ++t) {
    const double* tr = e.dtrans + 3 * t;
    const double v[3] = {vec.x + tr[0], vec.y + tr[1], vec.z + tr[2]};
    const double r1 = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (r1 < kEwaldEps) continue;
    const double r2 = r1 * r1;
    double dtmp;
    if (e.mode == kModeUpstream) {
      const double gtmp = 2 * gam * exp(-r2 * gam2) / (kSqrtPi * r2) - erf(r1 * gam) / (r2 * r1);
      const double atmp = -2 * e.alpha * exp(-r2 * alp2) / (kSqrtPi * r2) + erf(r1 * e.alpha) / (r2 * r1);
      dtmp = gtmp + atmp;
    } else {
      if (r1 > e.real_cutoff) continue;
      const double value = (erfc(e.alpha * r1) - erfc(gam * r1)) / r1;
      const double slope = (2 * gam * exp(-r2 * gam2) - 2 * e.alpha * exp(-r2 * alp2)) / kSqrtPi;
      dtmp = (slope - value) / r2;
    }
    dg[0] += dtmp * v[0];
    dg[1] += dtmp * v[1];
    dg[2] += dtmp * v[2];
    ds[0] += dtmp * v[0] * v[0];
    ds[1] += dtmp * v[1] * v[1];
    ds[2] += dtmp * v[2] * v[2];
    ds[3] += dtmp * v[1] * v[2];
    ds[4] += dtmp * v[0] * v[2];
    ds[5] += dtmp * v[0] * v[1];
  }
  const double fac = 4 * kPi / e.volume;
  for (int k = 0; k < e.n_recip; ++k) {
    const double* G = e.rtrans + 3 * k;
    const double g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const double gv = vec.x * G[0] + vec.y * G[1] + vec.z * G[2];
    const double etmp = fac * exp(-0.25 * g2 / alp2) / g2;
    double sin_gv;
    double cos_gv;
    sincos_phase(gv, sin_gv, cos_gv);
    const double sine = -sin_gv * etmp;
    const double cosine = etmp * cos_gv;
    const double gfac = 2.0 / g2 + 0.5 / alp2;
    dg[0] += sine * G[0];
    dg[1] += sine * G[1];
    dg[2] += sine * G[2];
    ds[0] += cosine * (gfac * G[0] * G[0] - 1.0);
    ds[1] += cosine * (gfac * G[1] * G[1] - 1.0);
    ds[2] += cosine * (gfac * G[2] * G[2] - 1.0);
    ds[3] += cosine * (gfac * G[1] * G[2]);
    ds[4] += cosine * (gfac * G[0] * G[2]);
    ds[5] += cosine * (gfac * G[0] * G[1]);
  }
}

// Wigner-Seitz images of the pair (centre a, partner b) in upstream mode:
// candidates T minimising |r_a - r_b - T|^2, with ties within 0.01 bohr^2.
// Returns a bit mask over the 27 candidates and the image count.
__device__ unsigned ws_images(const EwaldView& e, V3 xa, V3 xb, int& count) {
  const double rx = xa.x - xb.x;
  const double ry = xa.y - xb.y;
  const double rz = xa.z - xb.z;
  double dist[kWsCandidates];
  double best = 0.0;
  for (int c = 0; c < kWsCandidates; ++c) {
    const double* t = e.ws_candidates + 3 * c;
    const double vx = rx - t[0];
    const double vy = ry - t[1];
    const double vz = rz - t[2];
    dist[c] = vx * vx + vy * vy + vz * vz;
    best = c == 0 ? dist[c] : fmin(best, dist[c]);
  }
  unsigned mask = 0u;
  count = 0;
  for (int c = 0; c < kWsCandidates; ++c) {
    if (fabs(dist[c] - best) <= kWsTol) {
      mask |= 1u << c;
      count += 1;
    }
  }
  return mask;
}

__device__ double pair_value(const EwaldView& e, V3 xa, V3 xb, double gam) {
  const V3 vec0{xb.x - xa.x, xb.y - xa.y, xb.z - xa.z};
  if (e.mode != kModeUpstream) return ewald_value(e, vec0, gam);
  int count = 0;
  const unsigned mask = ws_images(e, xa, xb, count);
  const double wsw = 1.0 / static_cast<double>(count);
  double total = 0.0;
  for (int c = 0; c < kWsCandidates; ++c) {
    if (!((mask >> c) & 1u)) continue;
    const double* t = e.ws_candidates + 3 * c;
    total += ewald_value(e, {vec0.x + t[0], vec0.y + t[1], vec0.z + t[2]}, gam) * wsw;
  }
  return total;
}

__device__ void pair_derivative(const EwaldView& e, V3 xa, V3 xb, double gam, double dg[3], double ds[6]) {
  const V3 vec0{xb.x - xa.x, xb.y - xa.y, xb.z - xa.z};
  for (int k = 0; k < 3; ++k) dg[k] = 0.0;
  for (int k = 0; k < 6; ++k) ds[k] = 0.0;
  if (e.mode != kModeUpstream) {
    ewald_derivative(e, vec0, gam, dg, ds);
    return;
  }
  int count = 0;
  const unsigned mask = ws_images(e, xa, xb, count);
  const double wsw = 1.0 / static_cast<double>(count);
  for (int c = 0; c < kWsCandidates; ++c) {
    if (!((mask >> c) & 1u)) continue;
    const double* t = e.ws_candidates + 3 * c;
    double g[3] = {0.0, 0.0, 0.0};
    double s[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ewald_derivative(e, {vec0.x + t[0], vec0.y + t[1], vec0.z + t[2]}, gam, g, s);
    for (int k = 0; k < 3; ++k) dg[k] += g[k] * wsw;
    for (int k = 0; k < 6; ++k) ds[k] += s[k] * wsw;
  }
}

__global__ void eeq_assemble_kernel(GeometryView g, D4View d, EwaldView e, double* __restrict__ k) {
  const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = g.n;
  const long long np = n + 1;
  if (i > n || j > n) return;
  if (i == n || j == n) {
    k[i * np + j] = i == j ? 0.0 : 1.0;
    return;
  }
  if (j > i) return;
  const int si = species(g, i);
  if (i == j) {
    k[i * np + i] = e.mode == kModeMolecular ? d.eta[si] + kSqrt2Pi / d.rad[si] : e.self_diag[si];
    return;
  }
  const int sj = species(g, j);
  const V3 xi = pos3(g, i);
  const V3 xj = pos3(g, j);
  double value;
  if (e.mode == kModeMolecular) {
    const double vx = xj.x - xi.x;
    const double vy = xj.y - xi.y;
    const double vz = xj.z - xi.z;
    const double r2 = vx * vx + vy * vy + vz * vz;
    const double gam = 1.0 / (d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
    value = erf(sqrt(r2 * gam)) / sqrt(r2);
  } else {
    const double gam = 1.0 / sqrt(d.rad[si] * d.rad[si] + d.rad[sj] * d.rad[sj]);
    value = pair_value(e, xi, xj, gam);
  }
  k[i * np + j] = value;
  k[j * np + i] = value;
}

// Find the pivot of column ``col`` (largest |value|, lowest row on ties) and
// swap it into place. One block.
__global__ void lu_pivot_kernel(double* __restrict__ k, int np, int col, int* __restrict__ pivots,
                                int* __restrict__ status) {
  __shared__ double best_value[256];
  __shared__ int best_row[256];
  const int tid = static_cast<int>(threadIdx.x);
  double value = -1.0;
  int row = np;
  for (int r = col + tid; r < np; r += blockDim.x) {
    const double candidate = fabs(k[static_cast<long long>(r) * np + col]);
    if (candidate > value) {
      value = candidate;
      row = r;
    }
  }
  best_value[tid] = value;
  best_row[tid] = row;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      const double other = best_value[tid + stride];
      const int other_row = best_row[tid + stride];
      if (other > best_value[tid] || (other == best_value[tid] && other_row < best_row[tid])) {
        best_value[tid] = other;
        best_row[tid] = other_row;
      }
    }
    __syncthreads();
  }
  const int pivot = best_row[0];
  if (tid == 0) {
    pivots[col] = pivot;
    if (!(best_value[0] > 0.0)) *status = 1;
  }
  if (pivot != col) {
    for (int c = tid; c < np; c += blockDim.x) {
      const long long a = static_cast<long long>(col) * np + c;
      const long long b = static_cast<long long>(pivot) * np + c;
      const double tmp = k[a];
      k[a] = k[b];
      k[b] = tmp;
    }
  }
}

__global__ void lu_scale_kernel(double* __restrict__ k, int np, int col) {
  const int r = col + 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= np) return;
  const double diag = k[static_cast<long long>(col) * np + col];
  if (diag != 0.0) k[static_cast<long long>(r) * np + col] /= diag;
}

__global__ void lu_update_kernel(double* __restrict__ k, int np, int col) {
  const int c = col + 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int r = col + 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (r >= np || c >= np) return;
  k[static_cast<long long>(r) * np + c] -= k[static_cast<long long>(r) * np + col] * k[static_cast<long long>(col) * np + c];
}

__device__ double block_sum(double value, double* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  const double total = scratch[0];
  __syncthreads();
  return total;
}

// Solve with the row-permuted LU factors in place. One block of 256 threads.
__global__ void lu_solve_kernel(const double* __restrict__ lu, int np, const int* __restrict__ pivots,
                                double* __restrict__ b) {
  __shared__ double scratch[256];
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    for (int r = 0; r < np; ++r) {
      const int p = pivots[r];
      if (p != r) {
        const double tmp = b[r];
        b[r] = b[p];
        b[p] = tmp;
      }
    }
  }
  __syncthreads();
  for (int r = 1; r < np; ++r) {
    double partial = 0.0;
    for (int c = tid; c < r; c += blockDim.x) partial += lu[static_cast<long long>(r) * np + c] * b[c];
    const double total = block_sum(partial, scratch);
    if (tid == 0) b[r] -= total;
    __syncthreads();
  }
  for (int r = np - 1; r >= 0; --r) {
    double partial = 0.0;
    for (int c = r + 1 + tid; c < np; c += blockDim.x) partial += lu[static_cast<long long>(r) * np + c] * b[c];
    const double total = block_sum(partial, scratch);
    if (tid == 0) b[r] = (b[r] - total) / lu[static_cast<long long>(r) * np + r];
    __syncthreads();
  }
}

// Owner rows: warp i accumulates the derivative of -nu^T A q with respect to
// r_i over every partner, and books the virial of its pairs with smaller partners.
__global__ void eeq_reverse_kernel(GeometryView g, D4View d, EwaldView e, const double* __restrict__ q,
                                   const double* __restrict__ nu, AccumView acc) {
  const int i = global_warp();
  if (i >= g.n) return;
  const int lane = lane_id();
  const int si = species(g, i);
  const V3 xi = pos3(g, i);
  math::Vec3 grad{0.0, 0.0, 0.0};
  Sym6 vir;
  for (int j = lane; j < g.n; j += kWarpSize) {
    if (j == i) continue;
    const int a = max(i, j);
    const int b = min(i, j);
    const int sa = species(g, a);
    const int sb = species(g, b);
    const V3 xa = a == i ? xi : pos3(g, j);
    const V3 xb = b == i ? xi : pos3(g, j);
    const double gam = 1.0 / sqrt(d.rad[sa] * d.rad[sa] + d.rad[sb] * d.rad[sb]);
    double dg[3];
    double ds[6];
    if (e.mode == kModeMolecular) {
      const double v[3] = {xb.x - xa.x, xb.y - xa.y, xb.z - xa.z};
      const double r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      const double arg = gam * gam * r2;
      const double dtmp = 2.0 * gam * exp(-arg) / (kSqrtPi * r2) - erf(sqrt(arg)) / (r2 * sqrt(r2));
      for (int c = 0; c < 3; ++c) dg[c] = dtmp * v[c];
      ds[0] = dtmp * v[0] * v[0];
      ds[1] = dtmp * v[1] * v[1];
      ds[2] = dtmp * v[2] * v[2];
      ds[3] = dtmp * v[1] * v[2];
      ds[4] = dtmp * v[0] * v[2];
      ds[5] = dtmp * v[0] * v[1];
    } else {
      pair_derivative(e, xa, xb, gam, dg, ds);
    }
    const double w = nu[a] * q[b] + nu[b] * q[a];
    if (a == i) {
      grad.x += w * dg[0];
      grad.y += w * dg[1];
      grad.z += w * dg[2];
      for (int c = 0; c < 6; ++c) vir.v[c] -= w * ds[c];
    } else {
      grad.x -= w * dg[0];
      grad.y -= w * dg[1];
      grad.z -= w * dg[2];
    }
  }
  if (e.mode != kModeMolecular && lane == 0) {
    const double scale = nu[i] * q[i];
    for (int c = 0; c < 6; ++c) vir.v[c] -= scale * e.self_vir[6 * si + c];
  }
  const double gx = warp_sum(grad.x);
  const double gy = warp_sum(grad.y);
  const double gz = warp_sum(grad.z);
  if (lane == 0) {
    acc.grad[3 * i] += gx;
    acc.grad[3 * i + 1] += gy;
    acc.grad[3 * i + 2] += gz;
  }
  vir.reduce_and_add(acc.vir, i, g.n);
}

}  // namespace

void launch_eeq_assemble(const GeometryView& g, const D4View& d, const EwaldView& e, double* k,
                         cudaStream_t stream) {
  const int np = g.n + 1;
  const dim3 threads(16, 16);
  const dim3 blocks(static_cast<unsigned>((np + 15) / 16), static_cast<unsigned>((np + 15) / 16));
  eeq_assemble_kernel<<<blocks, threads, 0, stream>>>(g, d, e, k);
  VF_LAUNCH_CHECK(stream);
}

void launch_lu_factor(double* k, int n, int* pivots, int* status, cudaStream_t stream) {
  const int np = n + 1;
  for (int col = 0; col < np; ++col) {
    lu_pivot_kernel<<<1, 256, 0, stream>>>(k, np, col, pivots, status);
    const int rest = np - col - 1;
    if (rest > 0) {
      lu_scale_kernel<<<static_cast<unsigned>((rest + 255) / 256), 256, 0, stream>>>(k, np, col);
      const dim3 threads(16, 16);
      const dim3 blocks(static_cast<unsigned>((rest + 15) / 16), static_cast<unsigned>((rest + 15) / 16));
      lu_update_kernel<<<blocks, threads, 0, stream>>>(k, np, col);
    }
  }
  VF_LAUNCH_CHECK(stream);
}

void launch_lu_solve(const double* lu, int n, const int* pivots, double* b, cudaStream_t stream) {
  lu_solve_kernel<<<1, 256, 0, stream>>>(lu, n + 1, pivots, b);
  VF_LAUNCH_CHECK(stream);
}

void launch_eeq_reverse(const GeometryView& g, const D4View& d, const EwaldView& e,
                        const double* q, const double* nu, const AccumView& acc,
                        cudaStream_t stream) {
  if (g.n == 0) return;
  eeq_reverse_kernel<<<static_cast<unsigned>(warp_grid(g.n)), kWarpBlock, 0, stream>>>(g, d, e, q, nu, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
