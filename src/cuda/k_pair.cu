// K3: fused C6 interpolation and BJ pair term over directed edges.
//
// Each directed edge (i -> j, n) carries half of the pair energy. The literal
// variant scatters its derivatives to both endpoints; the owner variant gives
// the centre the full-weight derivative of the pair, relying on the exact
// reverse edge (j -> i, -n) in row j for the other endpoint. Both reproduce
// the upstream unordered sum, including self-images of the same atom.
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

template <int M, bool Derivatives, bool Owner>
__global__ void pair_kernel(GeometryView g, ModelView m, GraphView graph,
                            const double* __restrict__ w, const double* __restrict__ dw,
                            ContractedView cv, AccumView acc) {
  const int i = global_warp();
  if (i >= g.n) return;
  const int lane = lane_id();
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const int nref_i = m.nref[si];
  double wi[M];
  double dwi[M];
#pragma unroll
  for (int a = 0; a < M; ++a) {
    wi[a] = w[i * kRefPad + a];
    dwi[a] = Derivatives ? dw[i * kRefPad + a] : 0.0;
  }
  const long long begin = graph.row_ptr[i];
  const long long end = begin + graph.len_pair[i];
  double energy = 0.0;
  double g_i = 0.0;
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  for (long long e = begin + lane; e < end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const int j = edge.x;
    const math::Vec3 d = edge_displacement(g, xi, edge);
    const double r2 = math::norm2(d);
    if (r2 > m.disp2_cutoff2 || r2 < DBL_EPSILON) continue;
    const int sj = species(g, j);
    const int pair_index = si * m.n_species + sj;
    const PairConst pc = m.pair[pair_index];
    const double* block = m.c6ref + pair_index * kC6Block;
    const int nref_j = m.nref[sj];
    const double* wj = w + j * kRefPad;
    const double* dwj = Derivatives ? dw + j * kRefPad : nullptr;
    double c6 = 0.0;
    double dc6_i = 0.0;
    double dc6_j = 0.0;
    if constexpr (Owner) {
      const long long base = (static_cast<long long>(i) * m.n_species + sj) * kRefPad;
      const double* ui = cv.u + base;
#pragma unroll
      for (int b = 0; b < M; ++b) {
        if (b >= nref_j) break;
        c6 += ui[b] * wj[b];
        if constexpr (Derivatives) dc6_i += cv.du[base + b] * wj[b];
      }
    } else {
#pragma unroll
      for (int a = 0; a < M; ++a) {
        if (a >= nref_i) break;
#pragma unroll
        for (int b = 0; b < M; ++b) {
          if (b >= nref_j) break;
          const double k = block[a * kRefPad + b];
          c6 += wi[a] * wj[b] * k;
          if constexpr (Derivatives) {
            dc6_i += dwi[a] * wj[b] * k;
            dc6_j += wi[a] * dwj[b] * k;
          }
        }
      }
    }
    const double r4 = r2 * r2;
    const double t6 = 1.0 / (r4 * r2 + pc.r0_6);
    const double t8 = 1.0 / (r4 * r4 + pc.r0_8);
    const double edisp = m.s6 * t6 + m.s8 * pc.kappa * t8;
    energy += -0.5 * c6 * edisp;
    if (Derivatives) {
      const double d6 = -6.0 * r4 * t6 * t6;
      const double d8 = -8.0 * r4 * r2 * t8 * t8;
      const double gdisp = m.s6 * d6 + m.s8 * pc.kappa * d8;
      const double scale = -0.5 * c6 * gdisp;
      const math::Vec3 de{scale * d.x, scale * d.y, scale * d.z};
      vir.add_outer(d, de);
      if (Owner) {
        // Full-weight centre partials: the reverse edge (j -> i, -n) in
        // row j supplies the other endpoint, so no scatter is needed.
        if (j != i) {
          grad_i.x -= 2.0 * de.x;
          grad_i.y -= 2.0 * de.y;
          grad_i.z -= 2.0 * de.z;
        }
        g_i += -dc6_i * edisp;
      } else {
        if (j != i) {
          atomic_add3(acc.grad + 3 * j, de);
          grad_i.x -= de.x;
          grad_i.y -= de.y;
          grad_i.z -= de.z;
        }
        g_i += -0.5 * dc6_i * edisp;
        atomicAdd(acc.g + j, -0.5 * dc6_j * edisp);
      }
    }
  }
  energy = warp_sum(energy);
  if (lane == 0) acc.e_pair[i] = energy;
  if (Derivatives) {
    g_i = warp_sum(g_i);
    const math::Vec3 total{warp_sum(grad_i.x), warp_sum(grad_i.y), warp_sum(grad_i.z)};
    if (lane == 0) {
      if (Owner) {  // the only writer of row i's centre terms in this kernel
        acc.g[i] += g_i;
        acc.grad[3 * i] += total.x;
        acc.grad[3 * i + 1] += total.y;
        acc.grad[3 * i + 2] += total.z;
      } else {
        atomicAdd(acc.g + i, g_i);
        atomic_add3(acc.grad + 3 * i, total);
      }
    }
    vir.reduce_and_add(acc.vir, i, acc.n);
  }
}

template <int M>
void launch_pair_m(const GeometryView& g, const ModelView& m, const GraphView& graph,
                   const double* w, const double* dw, const ContractedView& cv,
                   const AccumView& acc, cudaStream_t stream) {
  const auto blocks = warp_grid(g.n);
  const bool owner = cv.u != nullptr;
  if (dw == nullptr && owner) {
    pair_kernel<M, false, true><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, w, dw, cv, acc);
  } else if (dw == nullptr) {
    pair_kernel<M, false, false><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, w, dw, cv, acc);
  } else if (owner) {
    pair_kernel<M, true, true><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, w, dw, cv, acc);
  } else {
    pair_kernel<M, true, false><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, w, dw, cv, acc);
  }
}

}  // namespace

void launch_pair(const GeometryView& g, const ModelView& m, const GraphView& graph, const double* w,
                 const double* dw, const ContractedView& cv, const AccumView& acc,
                 cudaStream_t stream) {
  if (g.n == 0) return;
  dispatch_max_nref(m.max_nref, [&](auto max_nref) {
    launch_pair_m<decltype(max_nref)::value>(g, m, graph, w, dw, cv, acc, stream);
  });
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
