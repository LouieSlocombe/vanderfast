// Device helpers shared by the literal and optimised EEQ kernels.
#pragma once

#include <cuda_runtime.h>

#include "d4_kernels.cuh"
#include "views.cuh"

namespace vanderfast::cuda::eeq {

constexpr double kPi = 3.141592653589793;
constexpr double kInvPi = 0.31830988618379067;
constexpr double kSqrtPi = 1.7724538509055159;
constexpr double kSqrt2Pi = 0.79788456080286541;      // sqrt(2/pi)
constexpr double kEwaldEps = 1.4901161193847656e-08;  // sqrt(epsilon(1.0_wp))
constexpr double kWsTol = 0.01;
constexpr int kWsCandidates = 27;
constexpr int kModeMolecular = 0;
constexpr int kModeUpstream = 1;
constexpr int kModeConverged = 2;

struct V3 {
  double x, y, z;
};

__device__ __forceinline__ V3 pos3(const GeometryView& g, int i) {
  const double4_32a p = g.pos[i];
  return {p.x, p.y, p.z};
}

// cos and sin of a reciprocal-space phase through sinpi/cospi, whose exact
// argument reduction needs no local memory (the libdevice cos/sin slow path
// for huge arguments does). |G.r| stays below a few hundred here, so the
// extra rounding of gv/pi is within an ulp of the phase.
__device__ __forceinline__ double cos_phase(double gv) { return cospi(gv * kInvPi); }

__device__ __forceinline__ void sincos_phase(double gv, double& s, double& c) {
  sincospi(gv * kInvPi, &s, &c);
}

}  // namespace vanderfast::cuda::eeq
