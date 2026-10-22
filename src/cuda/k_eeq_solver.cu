// EEQ charge solvers: LU in shared memory for small bordered systems and a
// cooperative preconditioned MINRES for the rest.
//
// MINRES follows Paige and Saunders (scipy.sparse.linalg.minres) with a
// diagonal positive-definite preconditioner. Each iteration is one matrix
// row sweep and two grid-wide reductions; every block evaluates the scalar
// recurrences itself from the same fixed-order partial sums, so the blocks
// agree bitwise and the result does not depend on scheduling. Vectors that
// other blocks write during the launch are read with ld.global.cg (L2), as
// L1 is not coherent across SMs.
#include <cooperative_groups.h>

#include <algorithm>
#include <cfloat>

#include "cuda_check.cuh"
#include "d4_kernels.cuh"
#include "device_util.cuh"

namespace vanderfast::cuda {

namespace cg = cooperative_groups;

namespace {

constexpr int kSmallThreads = 256;
constexpr int kSmallMaxRows = 128;
constexpr int kSolverWarps = 8;
constexpr int kSolverThreads = kSolverWarps * kWarpSize;
constexpr double kRefineTarget = 3.552713678800501e-15;  // 2^-48
constexpr double kInnerTolerance = 1e-15;                // phibar / |b|_M^-1 before a residual check
constexpr int kMaxRestarts = 3;
constexpr double kSingularCondition = 1e14;
constexpr double kPreconditionerFloor = 1e-12;  // relative to the largest |A_ii|

// Slots of the per-block partial buffer.
enum Slot : int {
  kSlotNorm,
  kSlotDiag,
  kSlotRhs,
  kSlotBorder,
  kSlotBeta1,
  kSlotAlpha,
  kSlotBeta,
  kSlotX,
  kSlotScale,
};
static_assert(kSlotScale < kMinresSlots, "MINRES partial slots");

// ---------------------------------------------------------------- small LU

// Forward and back substitution with row-permuted LU factors; one warp.
__device__ void warp_lu_solve(const double* lu, int np, const int* piv, double* b) {
  const int lane = lane_id();
  if (lane == 0) {
    for (int r = 0; r < np; ++r) {
      const int p = piv[r];
      if (p != r) {
        const double tmp = b[r];
        b[r] = b[p];
        b[p] = tmp;
      }
    }
  }
  __syncwarp();
  for (int r = 1; r < np; ++r) {
    double partial = 0.0;
    for (int c = lane; c < r; c += kWarpSize) partial += lu[r * np + c] * b[c];
    partial = warp_sum(partial);
    if (lane == 0) b[r] -= partial;
    __syncwarp();
  }
  for (int r = np - 1; r >= 0; --r) {
    double partial = 0.0;
    for (int c = r + 1 + lane; c < np; c += kWarpSize) partial += lu[r * np + c] * b[c];
    partial = warp_sum(partial);
    if (lane == 0) b[r] = (b[r] - partial) / lu[r * np + r];
    __syncwarp();
  }
}

// Partial pivoting as launch_lu_factor: the largest |value|, the lowest row on ties.
__global__ void __launch_bounds__(kSmallThreads)
    lu_small_kernel(double* __restrict__ k, int np, int* __restrict__ pivots, double* __restrict__ b,
                    int* __restrict__ status) {
  extern __shared__ double a[];
  __shared__ int piv[kSmallMaxRows];
  const int tid = static_cast<int>(threadIdx.x);
  const int count = np * np;
  for (int e = tid; e < count; e += kSmallThreads) a[e] = k[e];
  __syncthreads();
  for (int col = 0; col < np; ++col) {
    if (tid < kWarpSize) {
      double best = -1.0;
      int row = np;
      for (int r = col + tid; r < np; r += kWarpSize) {
        const double value = fabs(a[r * np + col]);
        if (value > best) {
          best = value;
          row = r;
        }
      }
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const double other = __shfl_xor_sync(kAllLanes, best, offset);
        const int other_row = __shfl_xor_sync(kAllLanes, row, offset);
        if (other > best || (other == best && other_row < row)) {
          best = other;
          row = other_row;
        }
      }
      if (tid == 0) {
        if (row >= np) row = col;  // a column of NaN
        piv[col] = row;
        pivots[col] = row;
        if (!(best > 0.0)) *status = 1;
      }
    }
    __syncthreads();
    const int p = piv[col];
    if (p != col) {
      for (int c = tid; c < np; c += kSmallThreads) {
        const double tmp = a[col * np + c];
        a[col * np + c] = a[p * np + c];
        a[p * np + c] = tmp;
      }
    }
    __syncthreads();
    const double diag = a[col * np + col];
    for (int r = col + 1 + tid; r < np; r += kSmallThreads) {
      if (diag != 0.0) a[r * np + col] /= diag;
    }
    __syncthreads();
    const int rest = np - col - 1;
    for (int e = tid; e < rest * rest; e += kSmallThreads) {
      const int r = col + 1 + e / rest;
      const int c = col + 1 + e % rest;
      a[r * np + c] -= a[r * np + col] * a[col * np + c];
    }
    __syncthreads();
  }
  for (int e = tid; e < count; e += kSmallThreads) k[e] = a[e];
  if (tid < kWarpSize) warp_lu_solve(a, np, piv, b);
}

__global__ void lu_small_solve_kernel(const double* __restrict__ lu, int np, const int* __restrict__ pivots,
                                      double* __restrict__ b) {
  warp_lu_solve(lu, np, pivots, b);
}

// ------------------------------------------------------------------ MINRES

// Sum (or maximum) of one value per warp, held in lane 0, over the block.
__device__ double block_reduce(double value, double* shared, bool maximum) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  if (lane_id() == 0) shared[warp] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = shared[0];
    for (int w = 1; w < kSolverWarps; ++w) total = maximum ? fmax(total, shared[w]) : total + shared[w];
    shared[kSolverWarps] = total;
  }
  __syncthreads();
  const double total = shared[kSolverWarps];
  __syncthreads();
  return total;
}

// Publish a block value to its slot; call grid.sync() before grid_reduce.
__device__ void publish(double* partials, int slot, double value) {
  if (threadIdx.x == 0) partials[slot * gridDim.x + blockIdx.x] = value;
}

// Combine the block values of a slot in a fixed order (identical in every block).
__device__ double grid_reduce(const double* partials, int slot, double* shared, bool maximum) {
  const int count = static_cast<int>(gridDim.x);
  const double* base = partials + slot * count;
  if (threadIdx.x < kWarpSize) {
    const int lane = lane_id();
    double value = maximum ? 0.0 : 0.0;
    for (int b = lane; b < count; b += kWarpSize) {
      const double part = __ldcg(base + b);
      value = maximum ? fmax(value, part) : value + part;
    }
    if (maximum) {
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmax(value, __shfl_down_sync(kAllLanes, value, offset));
      }
    } else {
      value = warp_sum(value);
    }
    if (lane == 0) shared[kSolverWarps] = value;
  }
  __syncthreads();
  const double total = shared[kSolverWarps];
  __syncthreads();
  return total;
}

// Row ``i`` of K times a vector that other blocks write (result in lane 0).
__device__ __forceinline__ double row_times(const double* __restrict__ row, const double* vec, int np) {
  double acc = 0.0;
  for (int j = lane_id(); j < np; j += kWarpSize) acc += __ldg(row + j) * __ldcg(vec + j);
  return warp_sum(acc);
}

struct Vectors {
  double *r1, *r2, *z, *v, *y, *w, *w2, *m, *r;
};

__device__ Vectors vectors(const MinresArgs& a) {
  const int np = a.np;
  double* base = a.work;
  return {base,          base + np,     base + 2 * np, base + 3 * np, base + 4 * np,
          base + 5 * np, base + 6 * np, base + 7 * np, base + 8 * np};
}

// (Re)start the Lanczos process from the residual in vec.r (own rows).
__device__ void begin(MinresState& st, const MinresArgs& a, const Vectors& vec, cg::grid_group& grid,
                      double* shared) {
  const int np = a.np;
  const int n = np - 1;
  const int gw = static_cast<int>(blockIdx.x) * kSolverWarps + static_cast<int>(threadIdx.x) / kWarpSize;
  const int tw = static_cast<int>(gridDim.x) * kSolverWarps;
  double part = 0.0;
  if (lane_id() == 0) {
    for (int i = gw; i < np; i += tw) {
      const double ri = vec.r[i];
      const double mi = i < n ? vec.m[i] : st.mborder;
      if (i == n) vec.m[i] = mi;
      const double zi = ri / mi;
      vec.r1[i] = ri;
      vec.r2[i] = ri;
      vec.z[i] = zi;
      vec.w[i] = 0.0;
      vec.w2[i] = 0.0;
      part += ri * zi;
    }
  }
  publish(a.partials, kSlotBeta1, block_reduce(part, shared, false));
  grid.sync();
  const double beta_sq = grid_reduce(a.partials, kSlotBeta1, shared, false);
  st.beta1 = sqrt(beta_sq);
  st.beta = st.beta1;
  st.oldb = 0.0;
  st.dbar = 0.0;
  st.epsln = 0.0;
  st.phibar = st.beta1;
  st.cs = -1.0;
  st.sn = 0.0;
  st.gmax = 0.0;
  st.gmin = DBL_MAX;
  st.inner = 0;
  if (!isfinite(beta_sq)) {
    st.status = 1;
    st.phase = 2;
  } else if (beta_sq == 0.0) {
    st.eta = 0.0;  // x solves the system exactly
    st.status = 0;
    st.phase = 2;
  } else {
    st.phase = 1;
  }
}

__global__ void __launch_bounds__(kSolverThreads) minres_kernel(MinresArgs a, int budget) {
  cg::grid_group grid = cg::this_grid();
  __shared__ double shared[kSolverWarps + 1];
  const int np = a.np;
  const int n = np - 1;
  const int lane = lane_id();
  const int gw = static_cast<int>(blockIdx.x) * kSolverWarps + static_cast<int>(threadIdx.x) / kWarpSize;
  const int tw = static_cast<int>(gridDim.x) * kSolverWarps;
  const Vectors vec = vectors(a);
  MinresState st = *a.state;

  if (st.phase == 0) {
    // Norms of K and b, the diagonal, and the initial residual.
    double norm = 0.0;
    double diag = 0.0;
    double rhs = 0.0;
    for (int i = gw; i < np; i += tw) {
      const double* row = a.k + static_cast<long long>(i) * np;
      double abs_sum = 0.0;
      double kx = 0.0;
      for (int j = lane; j < np; j += kWarpSize) {
        const double kij = __ldg(row + j);
        abs_sum += fabs(kij);
        if (a.warm) kx += kij * __ldcg(a.x + j);
      }
      abs_sum = warp_sum(abs_sum);
      kx = warp_sum(kx);
      if (lane == 0) {
        const double bi = a.b[i];
        norm = fmax(norm, abs_sum);
        rhs = fmax(rhs, fabs(bi));
        if (i < n) diag = fmax(diag, fabs(row[i]));
        vec.r[i] = a.warm ? bi - kx : bi;
        if (!a.warm) a.x[i] = 0.0;
      }
    }
    publish(a.partials, kSlotNorm, block_reduce(norm, shared, true));
    publish(a.partials, kSlotDiag, block_reduce(diag, shared, true));
    publish(a.partials, kSlotRhs, block_reduce(rhs, shared, true));
    grid.sync();
    st.knorm = grid_reduce(a.partials, kSlotNorm, shared, true);
    st.dmax = grid_reduce(a.partials, kSlotDiag, shared, true);
    st.bnorm = grid_reduce(a.partials, kSlotRhs, shared, true);
    st.itn = 0;
    st.restarts = 0;
    st.status = 0;
    st.eta = 0.0;
    // Jacobi preconditioner; the border gets the Schur estimate sum_i 1/m_i.
    const double floor = st.dmax > 0.0 ? kPreconditionerFloor * st.dmax : 1.0;
    double border = 0.0;
    if (lane == 0) {
      for (int i = gw; i < n; i += tw) {
        const double mi = fmax(fabs(a.k[static_cast<long long>(i) * np + i]), floor);
        vec.m[i] = mi;
        border += 1.0 / mi;
      }
    }
    publish(a.partials, kSlotBorder, block_reduce(border, shared, false));
    grid.sync();
    st.mborder = n > 0 ? grid_reduce(a.partials, kSlotBorder, shared, false) : 1.0;
    // The scale of the inner tolerance, |b|_M^-1 (beta1 of a cold start).
    double scale = 0.0;
    if (lane == 0) {
      for (int i = gw; i < np; i += tw) {
        const double bi = a.b[i];
        scale += bi * bi / (i < n ? vec.m[i] : st.mborder);
      }
    }
    publish(a.partials, kSlotScale, block_reduce(scale, shared, false));
    grid.sync();
    st.bscale = sqrt(grid_reduce(a.partials, kSlotScale, shared, false));
    if (!isfinite(st.knorm) || !isfinite(st.bnorm) || !isfinite(st.bscale)) {
      st.status = 1;
      st.phase = 2;
    } else if (st.bnorm == 0.0) {
      // b = 0: x = 0 is exact (also discards a warm-start guess).
      if (lane == 0) {
        for (int i = gw; i < np; i += tw) a.x[i] = 0.0;
      }
      st.phase = 2;
    } else {
      begin(st, a, vec, grid, shared);
    }
  }

  for (int done = 0; st.phase == 1 && done < budget; ++done) {
    // y = K v - (beta / oldb) r1 with v = z / beta; alpha = v . y.
    const double s = 1.0 / st.beta;
    const double back = st.inner > 0 ? st.beta / st.oldb : 0.0;
    double part = 0.0;
    for (int i = gw; i < np; i += tw) {
      const double kz = row_times(a.k + static_cast<long long>(i) * np, vec.z, np);
      if (lane == 0) {
        double yi = s * kz;
        if (st.inner > 0) yi -= back * vec.r1[i];
        const double vi = s * vec.z[i];
        vec.v[i] = vi;
        vec.y[i] = yi;
        part += vi * yi;
      }
    }
    publish(a.partials, kSlotAlpha, block_reduce(part, shared, false));
    grid.sync();
    const double alfa = grid_reduce(a.partials, kSlotAlpha, shared, false);

    // y -= (alpha / beta) r2; r1 <- r2 <- y; z = M^-1 y; beta' = sqrt(y . z).
    const double c = alfa / st.beta;
    double part2 = 0.0;
    if (lane == 0) {
      for (int i = gw; i < np; i += tw) {
        const double yi = vec.y[i] - c * vec.r2[i];
        vec.r1[i] = vec.r2[i];
        vec.r2[i] = yi;
        const double zi = yi / vec.m[i];
        vec.z[i] = zi;
        part2 += yi * zi;
      }
    }
    publish(a.partials, kSlotBeta, block_reduce(part2, shared, false));
    grid.sync();
    const double beta_sq = grid_reduce(a.partials, kSlotBeta, shared, false);
    if (!isfinite(beta_sq) || !isfinite(alfa)) {
      st.status = 1;
      st.phase = 2;
      break;
    }
    st.oldb = st.beta;
    st.beta = sqrt(beta_sq);

    // Plane rotation of the tridiagonal QR and the solution update.
    const double oldeps = st.epsln;
    const double delta = st.cs * st.dbar + st.sn * alfa;
    const double gbar = st.sn * st.dbar - st.cs * alfa;
    st.epsln = st.sn * st.beta;
    st.dbar = -st.cs * st.beta;
    const double gamma = fmax(hypot(gbar, st.beta), DBL_EPSILON);
    st.cs = gbar / gamma;
    st.sn = st.beta / gamma;
    const double phi = st.cs * st.phibar;
    st.phibar = st.sn * st.phibar;
    const double denom = 1.0 / gamma;
    st.gmax = fmax(st.gmax, gamma);
    st.gmin = fmin(st.gmin, gamma);
    if (lane == 0) {
      for (int i = gw; i < np; i += tw) {
        const double w1 = vec.w2[i];
        const double w2 = vec.w[i];
        const double wi = (vec.v[i] - oldeps * w1 - delta * w2) * denom;
        vec.w2[i] = w2;
        vec.w[i] = wi;
        a.x[i] += phi * wi;
      }
    }
    st.itn += 1;
    st.inner += 1;
    if (st.gmax > kSingularCondition * st.gmin) {
      st.status = 1;
      st.phase = 2;
      break;
    }
    const bool exhausted = st.itn >= a.max_iterations;
    if (!(st.phibar <= kInnerTolerance * st.bscale) && st.beta > 0.0 && !exhausted) continue;

    // True residual r = b - K x and the normwise backward error.
    grid.sync();
    double rmax = 0.0;
    double xmax = 0.0;
    for (int i = gw; i < np; i += tw) {
      const double kx = row_times(a.k + static_cast<long long>(i) * np, a.x, np);
      if (lane == 0) {
        const double ri = a.b[i] - kx;
        vec.r[i] = ri;
        rmax = fmax(rmax, fabs(ri));
        xmax = fmax(xmax, fabs(a.x[i]));
      }
    }
    publish(a.partials, kSlotNorm, block_reduce(rmax, shared, true));
    publish(a.partials, kSlotX, block_reduce(xmax, shared, true));
    grid.sync();
    const double residual = grid_reduce(a.partials, kSlotNorm, shared, true);
    const double scale = st.knorm * grid_reduce(a.partials, kSlotX, shared, true) + st.bnorm;
    st.eta = residual > 0.0 ? residual / scale : 0.0;
    if (!isfinite(st.eta)) {
      st.status = 1;
      st.phase = 2;
    } else if (st.eta <= kRefineTarget) {
      st.status = 0;
      st.phase = 2;
    } else if (st.restarts < kMaxRestarts && !exhausted) {
      st.restarts += 1;
      begin(st, a, vec, grid, shared);
    } else {
      st.status = st.eta <= a.tolerance ? 0 : 2;
      st.phase = 2;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *a.state = st;
}

}  // namespace

bool lu_small_fits(int n, int device) {
  const int np = n + 1;
  if (np > kSmallMaxRows) return false;
  int optin = 0;
  VF_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  return static_cast<long long>(np) * np * static_cast<long long>(sizeof(double)) <= optin;
}

void launch_lu_small_factor_solve(double* k, int n, int* pivots, double* b, int* status, cudaStream_t stream) {
  const int np = n + 1;
  const int bytes = np * np * static_cast<int>(sizeof(double));
  VF_CUDA_CHECK(cudaFuncSetAttribute(lu_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  lu_small_kernel<<<1, kSmallThreads, static_cast<std::size_t>(bytes), stream>>>(k, np, pivots, b, status);
  VF_LAUNCH_CHECK(stream);
}

void launch_lu_small_solve(const double* lu, int n, const int* pivots, double* b, cudaStream_t stream) {
  lu_small_solve_kernel<<<1, kWarpSize, 0, stream>>>(lu, n + 1, pivots, b);
  VF_LAUNCH_CHECK(stream);
}

int minres_blocks(int n, int device) {
  int cooperative = 0;
  VF_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  if (cooperative == 0) return 0;
  int sms = 0;
  VF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int per_sm = 0;
  VF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, minres_kernel, kSolverThreads, 0));
  const int rows = (n + 1 + kSolverWarps - 1) / kSolverWarps;
  return std::max(1, std::min(sms * per_sm, rows));
}

void launch_minres(const MinresArgs& args, int blocks, int iterations, cudaStream_t stream) {
  MinresArgs copy = args;
  void* params[] = {&copy, &iterations};
  VF_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(minres_kernel), dim3(static_cast<unsigned>(blocks)),
                                            dim3(kSolverThreads), params, 0, stream));
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
