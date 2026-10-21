// Formulas shared by host code and CUDA kernels.
//
// Every expression is written with explicit fused multiply-adds where the
// result feeds a cutoff decision, so the CPU enumerator, the neighbour
// builders, and the kernels make bitwise-identical inclusion decisions.
#pragma once

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define VF_HD __host__ __device__ __forceinline__
#else
#define VF_HD inline
#endif

namespace vanderfast::math {

struct Vec3 {
  double x, y, z;
};

VF_HD double fused(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
  return ::fma(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

// d = r_j + n H - r_i with lattice vectors as rows of H. Rounding is
// sign-symmetric, so d(j, i, -n) == -d(i, j, n) bit for bit.
VF_HD Vec3 site_displacement(Vec3 xi, Vec3 xj, int n0, int n1, int n2, const double* cell) {
  const double a = static_cast<double>(n0);
  const double b = static_cast<double>(n1);
  const double c = static_cast<double>(n2);
  const double tx = fused(a, cell[0], fused(b, cell[3], c * cell[6]));
  const double ty = fused(a, cell[1], fused(b, cell[4], c * cell[7]));
  const double tz = fused(a, cell[2], fused(b, cell[5], c * cell[8]));
  return {(xj.x - xi.x) + tx, (xj.y - xi.y) + ty, (xj.z - xi.z) + tz};
}

VF_HD double norm2(Vec3 d) { return fused(d.x, d.x, fused(d.y, d.y, d.z * d.z)); }

// D3 coordination-number counting function and its radial derivative.
constexpr double kKcn = 16.0;
constexpr double kCnExclusionR2 = 1.0e-12;
constexpr double kWf = 4.0;

VF_HD double cn_count(double rc, double r) {
#if defined(__CUDA_ARCH__)
  return 1.0 / (1.0 + ::exp(-kKcn * (rc / r - 1.0)));
#else
  return 1.0 / (1.0 + std::exp(-kKcn * (rc / r - 1.0)));
#endif
}

// dcf/dr for the counting function (upstream ``dcf``).
VF_HD double cn_count_derivative(double cf, double rc, double r2) {
  return -cf * (1.0 - cf) * kKcn * rc / r2;
}

VF_HD bool is_exceptional(double value) {
#if defined(__CUDA_ARCH__)
  return ::isnan(value) || ::fabs(value) > DBL_MAX;
#else
  return std::isnan(value) || std::fabs(value) > DBL_MAX;
#endif
}

}  // namespace vanderfast::math
