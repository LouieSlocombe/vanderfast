// D4 coordination numbers, charge-scaled reference weights, and the CN reverse pass.
//
// Both coordination numbers use the erf counting function of mctc-lib with
// kcn = 7.5 and the same covalent radii; the D4 CN multiplies each count by
// the electronegativity factor (PairConst::aux) within its cutoff, the EEQ CN
// takes the plain count within 25 bohr and is capped smoothly at 8. One erf
// per edge serves both.
#include "cuda_check.cuh"
#include "d4_kernels.cuh"
#include "device_util.cuh"
#include "row_items.cuh"

namespace vanderfast::cuda {

namespace {

constexpr double kKcn = 7.5;
constexpr double kCnMax = 8.0;
constexpr double kEeqReg = 1.0e-14;
constexpr double kSqrtPi = 1.7724538509055159;  // sqrt(pi)
constexpr double kEpsNorm = 1.4916681462400413e-154;  // sqrt(tiny(1.0_wp))

__device__ __forceinline__ double erf_count(double rc, double r) {
  return 0.5 * (1.0 + erf(-kKcn * (r - rc) / rc));
}

__device__ __forceinline__ double erf_dcount(double rc, double r) {
  const double exponent = kKcn * (r - rc) / rc;
  return -(kKcn * exp(-(exponent * exponent))) / (kSqrtPi * rc);
}

__global__ void d4_cn_items_kernel(GeometryView g, ModelView m, D4View d, GraphView graph,
                                   const long long* __restrict__ prefix, long long items,
                                   double* __restrict__ partials) {
  const long long item = global_warp();
  if (item >= items) return;
  const ItemRange r = item_range(prefix, g.n, item, graph.row_ptr, graph.len_cn);
  const int lane = lane_id();
  const math::Vec3 xi = position(g, r.i);
  const int si = species(g, r.i);
  double cn = 0.0;
  double cn_eeq = 0.0;
  for (long long e = r.begin + lane; e < r.end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const double r2 = math::norm2(edge_displacement(g, xi, edge));
    if (r2 < math::kCnExclusionR2) continue;
    const bool in_cn = r2 <= d.cn_cutoff2;
    const bool in_eeq = r2 <= d.eeq_cutoff2;
    if (!in_cn && !in_eeq) continue;
    const PairConst pc = m.pair[si * m.n_species + species(g, edge.x)];
    const double count = erf_count(pc.rc_cn, sqrt(r2));
    if (in_cn) cn += pc.aux * count;
    if (in_eeq) cn_eeq += count;
  }
  cn = warp_sum(cn);
  cn_eeq = warp_sum(cn_eeq);
  if (lane == 0) {
    partials[2 * item] = cn;
    partials[2 * item + 1] = cn_eeq;
  }
}

__global__ void d4_cn_items_reduce(GeometryView g, D4View d, const long long* __restrict__ prefix,
                                   const double* __restrict__ partials, double total_charge,
                                   double* __restrict__ cn, double* __restrict__ cn_eeq,
                                   double* __restrict__ rhs, double* __restrict__ dxfac) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) rhs[g.n] = total_charge;
  if (i >= g.n) return;
  double sum = 0.0;
  double raw = 0.0;
  for (long long item = prefix[i]; item < prefix[i + 1]; ++item) {
    sum += partials[2 * item];
    raw += partials[2 * item + 1];
  }
  cn[i] = sum;
  // mctc cut_coordination_number; the chain factor uses the uncapped value.
  const double capped = log(1.0 + exp(kCnMax)) - log(1.0 + exp(kCnMax - raw));
  const double dcap = exp(kCnMax) / (exp(kCnMax) + exp(raw));
  cn_eeq[i] = capped;
  const int s = species(g, i);
  // multicharge get_xvec; its derivative drops the regulariser.
  const double tmp = d.kcnchi[s] / sqrt(capped + kEeqReg);
  rhs[i] = -d.chi[s] + tmp * capped;
  dxfac[i] = 0.5 * tmp * dcap;
}

__device__ __forceinline__ double zeta(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) return exp(a);
  return exp(a * (1.0 - exp(c * (1.0 - qref / qmod))));
}

// dzeta/dqmod; zero for qmod <= 0 (upstream evaluates 0/0 at exactly zero).
__device__ __forceinline__ double dzeta(double a, double c, double qref, double qmod) {
  if (qmod <= 0.0) return 0.0;
  return -a * c * exp(c * (1.0 - qref / qmod)) * zeta(a, c, qref, qmod) * qref / (qmod * qmod);
}

template <bool Derivatives, bool Neutral>
__global__ void d4_weights_kernel(D4View d, GeometryView g, const double* __restrict__ cn,
                                  const double* __restrict__ q, double* __restrict__ w,
                                  double* __restrict__ dw_cn, double* __restrict__ dw_q,
                                  double* __restrict__ w0, double* __restrict__ dw0) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= g.n) return;
  const int s = species(g, i);
  const int* ngw = d.ngw + s * kRefPad;
  const double* cnref = d.cnref + s * kRefPad;
  const double c = cn[i];
  double expw[kRefPad];
  double expd[kRefPad];
  double norm = 0.0;
  double dnorm = 0.0;
  // dftd4 weight_references: the normalisation sums every Gaussian of every
  // reference in order; each reference keeps its own sums. Both reference
  // loops are unrolled so expw/expd stay in registers.
#pragma unroll
  for (int a = 0; a < kRefPad; ++a) {
    expw[a] = 0.0;
    expd[a] = 0.0;
    for (int k = 1; k <= ngw[a]; ++k) {
      const double wf = k * d.wf;
      const double delta = c - cnref[a];
      const double gw = exp(-wf * (delta * delta));
      norm += gw;
      expw[a] += gw;
      if (Derivatives) {
        const double dgw = 2 * wf * (cnref[a] - c) * gw;
        dnorm += dgw;
        expd[a] += dgw;
      }
    }
  }
  norm = fabs(norm) > kEpsNorm ? 1.0 / norm : 0.0;
  const unsigned maxmask = d.maxmask[s];
  const double zi = d.zeff[s];
  const double gi = d.gi[s];
  const double qi = q[i] + zi;
#pragma unroll
  for (int a = 0; a < kRefPad; ++a) {
    // Padding entries have ngw = 0, so expw = expd = 0 and their weights vanish.
    double gwk = expw[a] * norm;
    if (math::is_exceptional(gwk) || norm == 0.0) gwk = ((maxmask >> a) & 1u) ? 1.0 : 0.0;
    double dgwk = 0.0;
    if (Derivatives) {
      dgwk = norm * (expd[a] - expw[a] * dnorm * norm);
      if (math::is_exceptional(dgwk) || norm == 0.0) dgwk = 0.0;
    }
    if (ngw[a] == 0) {
      gwk = 0.0;
      dgwk = 0.0;
    }
    const double qref = d.qref[s * kRefPad + a];
    const double scale = zeta(d.ga, gi, qref, qi);
    w[i * kRefPad + a] = gwk * scale;
    if (Derivatives) {
      dw_cn[i * kRefPad + a] = dgwk * scale;
      dw_q[i * kRefPad + a] = gwk * dzeta(d.ga, gi, qref, qi);
    }
    if (Neutral) {
      const double scale0 = zeta(d.ga, gi, qref, zi);
      w0[i * kRefPad + a] = gwk * scale0;
      if (Derivatives) dw0[i * kRefPad + a] = dgwk * scale0;
    }
  }
}

constexpr int kReversePartials = 9;  // grad[3], vir[6]

__global__ void d4_cn_reverse_items_kernel(GeometryView g, ModelView m, D4View d, GraphView graph,
                                           const long long* __restrict__ prefix, long long items,
                                           const double* __restrict__ dEdcn,
                                           const double* __restrict__ nu,
                                           const double* __restrict__ dxfac,
                                           double* __restrict__ partials) {
  const long long item = global_warp();
  if (item >= items) return;
  const ItemRange r = item_range(prefix, g.n, item, graph.row_ptr, graph.len_cn);
  const int i = r.i;
  const int lane = lane_id();
  const double gi = dEdcn[i];
  const double hi = nu != nullptr ? nu[i] * dxfac[i] : 0.0;
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  for (long long e = r.begin + lane; e < r.end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const math::Vec3 dv = edge_displacement(g, xi, edge);
    const double r2 = math::norm2(dv);
    if (r2 < math::kCnExclusionR2) continue;
    const bool in_cn = r2 <= d.cn_cutoff2;
    const bool in_eeq = nu != nullptr && r2 <= d.eeq_cutoff2;
    if (!in_cn && !in_eeq) continue;
    const int j = edge.x;
    const PairConst pc = m.pair[si * m.n_species + species(g, j)];
    const double rr = sqrt(r2);
    const double dcount = erf_dcount(pc.rc_cn, rr);
    double weight = 0.0;
    if (in_cn) weight += pc.aux * (gi + dEdcn[j]);
    if (in_eeq) weight += hi + nu[j] * dxfac[j];
    // Edge (i -> j) and its reverse both act on r_i; the pair's virial is
    // split evenly between the two rows.
    const double scale = weight * dcount / rr;
    const math::Vec3 de{scale * dv.x, scale * dv.y, scale * dv.z};
    grad_i.x -= de.x;
    grad_i.y -= de.y;
    grad_i.z -= de.z;
    vir.add_outer(dv, {0.5 * de.x, 0.5 * de.y, 0.5 * de.z});
  }
  double* out = partials + kReversePartials * item;
  const double totals[kReversePartials] = {grad_i.x, grad_i.y, grad_i.z, vir.v[0], vir.v[1],
                                           vir.v[2], vir.v[3], vir.v[4], vir.v[5]};
#pragma unroll
  for (int c = 0; c < kReversePartials; ++c) {
    const double total = warp_sum(totals[c]);
    if (lane == 0) out[c] = total;
  }
}

__global__ void d4_cn_reverse_items_reduce(const long long* __restrict__ prefix, int n,
                                           const double* __restrict__ partials, AccumView acc) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  double sums[kReversePartials] = {};
  for (long long item = prefix[i]; item < prefix[i + 1]; ++item) {
#pragma unroll
    for (int c = 0; c < kReversePartials; ++c) sums[c] += partials[kReversePartials * item + c];
  }
  acc.grad[3 * i] += sums[0];
  acc.grad[3 * i + 1] += sums[1];
  acc.grad[3 * i + 2] += sums[2];
#pragma unroll
  for (int c = 0; c < 6; ++c) acc.vir[c * n + i] += sums[3 + c];
}

}  // namespace

void launch_d4_cn_items(const GeometryView& g, const ModelView& m, const D4View& d,
                        const GraphView& graph, const long long* prefix, long long items,
                        double total_charge, double* partials, double* cn, double* cn_eeq,
                        double* rhs, double* dxfac, cudaStream_t stream) {
  if (g.n == 0) return;
  if (items > 0) {
    d4_cn_items_kernel<<<item_blocks(items), kWarpBlock, 0, stream>>>(g, m, d, graph, prefix, items,
                                                                      partials);
    VF_LAUNCH_CHECK(stream);
  }
  d4_cn_items_reduce<<<centre_blocks(g.n), 128, 0, stream>>>(g, d, prefix, partials, total_charge, cn,
                                                             cn_eeq, rhs, dxfac);
  VF_LAUNCH_CHECK(stream);
}

void launch_d4_weights(const D4View& d, const GeometryView& g, const double* cn, const double* q,
                       double* w, double* dw_cn, double* dw_q, double* w0, double* dw0,
                       cudaStream_t stream) {
  if (g.n == 0) return;
  constexpr int threads = 128;
  const int blocks = (g.n + threads - 1) / threads;
  const bool derivatives = dw_cn != nullptr;
  const bool neutral = w0 != nullptr;
#define VF_D4_WEIGHTS(D, N) \
  d4_weights_kernel<D, N><<<blocks, threads, 0, stream>>>(d, g, cn, q, w, dw_cn, dw_q, w0, dw0)
  if (derivatives && neutral) {
    VF_D4_WEIGHTS(true, true);
  } else if (derivatives) {
    VF_D4_WEIGHTS(true, false);
  } else if (neutral) {
    VF_D4_WEIGHTS(false, true);
  } else {
    VF_D4_WEIGHTS(false, false);
  }
#undef VF_D4_WEIGHTS
  VF_LAUNCH_CHECK(stream);
}

void launch_d4_cn_reverse_items(const GeometryView& g, const ModelView& m, const D4View& d,
                                const GraphView& graph, const long long* prefix, long long items,
                                const AccumView& acc, const double* nu, const double* dxfac,
                                double* partials, cudaStream_t stream) {
  if (g.n == 0 || items == 0) return;
  d4_cn_reverse_items_kernel<<<item_blocks(items), kWarpBlock, 0, stream>>>(
      g, m, d, graph, prefix, items, acc.g, nu, dxfac, partials);
  VF_LAUNCH_CHECK(stream);
  d4_cn_reverse_items_reduce<<<centre_blocks(g.n), 128, 0, stream>>>(prefix, g.n, partials, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
