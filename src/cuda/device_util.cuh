// Small device helpers shared by the kernels.
#pragma once

#include <cuda_runtime.h>

#include <type_traits>

#include "../common/d3_math.hpp"
#include "views.cuh"

namespace vanderfast::cuda {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr unsigned kAllLanes = 0xffffffffu;

__device__ __forceinline__ int global_warp() {
  return static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize);
}

__device__ __forceinline__ int lane_id() { return static_cast<int>(threadIdx.x % kWarpSize); }

// Sum over the warp; the result is valid in lane 0.
__device__ __forceinline__ double warp_sum(double value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kAllLanes, value, offset);
  }
  return value;
}

__device__ __forceinline__ math::Vec3 position(const GeometryView& g, int i) {
  const double4_32a p = g.pos[i];
  return {p.x, p.y, p.z};
}

__device__ __forceinline__ int species(const GeometryView& g, int i) {
  return static_cast<int>(g.pos[i].w);
}

__device__ __forceinline__ math::Vec3 edge_displacement(const GeometryView& g, math::Vec3 xi,
                                                        int4 edge) {
  return math::site_displacement(xi, position(g, edge.x), edge.y, edge.z, edge.w, g.cell);
}

// Symmetric 3x3 accumulator in the order xx, yy, zz, yz, xz, xy.
struct Sym6 {
  double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  __device__ __forceinline__ void add_outer(math::Vec3 a, math::Vec3 b) {
    v[0] += a.x * b.x;
    v[1] += a.y * b.y;
    v[2] += a.z * b.z;
    v[3] += a.y * b.z;
    v[4] += a.x * b.z;
    v[5] += a.x * b.y;
  }

  __device__ __forceinline__ void reduce_and_add(double* vir, int i, int n) {
    for (int c = 0; c < 6; ++c) {
      const double total = warp_sum(v[c]);
      if (lane_id() == 0) vir[c * n + i] += total;
    }
  }
};

__device__ __forceinline__ void atomic_add3(double* target, math::Vec3 value) {
  atomicAdd(target, value.x);
  atomicAdd(target + 1, value.y);
  atomicAdd(target + 2, value.z);
}

inline int warp_grid(int n) { return (n + kWarpsPerBlock - 1) / kWarpsPerBlock; }

// Call ``f(std::integral_constant<int, M>{})`` for the runtime maximum
// reference count M in 1..7, so kernels can unroll reference loops.
template <class F>
void dispatch_max_nref(int max_nref, F&& f) {
  switch (max_nref) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 5: f(std::integral_constant<int, 5>{}); break;
    case 6: f(std::integral_constant<int, 6>{}); break;
    default: f(std::integral_constant<int, 7>{}); break;
  }
}
constexpr int kWarpBlock = kWarpSize * kWarpsPerBlock;

}  // namespace vanderfast::cuda
