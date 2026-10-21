// K4: streamed Axilrod-Teller-Muto three-body term.
//
// Literal anchored baseline: for every primary-cell centre i and every
// unordered pair of distinct neighbour sites u = (j, n1), v = (k, n2) of its
// ATM row, the triangle contributes with weight 1/3 (each translational
// triangle class is met once from each of its three vertices). The third edge
// is d_uv = d_iv - d_iu, never a fresh minimum image. All three edges are
// screened against the ATM cutoff.
//
// Work items are (centre, chunk of 32 outer sites); a warp owns one item and
// lane l owns the outer site 32 c + l. Energy and virial partials are stored
// per item and reduced per centre in a fixed order, so energies are
// deterministic; forces and dE/dCN are scattered with atomics.
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

constexpr double kThird = 1.0 / 3.0;

template <int M>
struct Weights {
  double w[M];
  double dw[M];
};

template <int M, bool Derivatives>
__device__ __forceinline__ void load_weights(const double* w, const double* dw, int atom,
                                             Weights<M>& out) {
#pragma unroll
  for (int a = 0; a < M; ++a) {
    out.w[a] = w[atom * kRefPad + a];
    out.dw[a] = Derivatives ? dw[atom * kRefPad + a] : 0.0;
  }
}

// C6 between a register-resident atom (x) and an atom read from memory (y),
// with the derivatives with respect to each atom's CN.
template <int M, bool Derivatives>
__device__ __forceinline__ void contract(const Weights<M>& x, int nref_x, const double* wy,
                                         const double* dwy, int nref_y, const double* block,
                                         double& c6, double& dc6_x, double& dc6_y) {
  c6 = 0.0;
  dc6_x = 0.0;
  dc6_y = 0.0;
#pragma unroll
  for (int a = 0; a < M; ++a) {
    if (a >= nref_x) break;
#pragma unroll
    for (int b = 0; b < M; ++b) {
      if (b >= nref_y) break;
      const double k = block[a * kRefPad + b];
      c6 += x.w[a] * wy[b] * k;
      if (Derivatives) {
        dc6_x += x.dw[a] * wy[b] * k;
        dc6_y += x.w[a] * dwy[b] * k;
      }
    }
  }
}

__device__ __forceinline__ bool screened(double r2, double cutoff2) {
  return r2 <= cutoff2 && r2 >= DBL_EPSILON;
}

__device__ __forceinline__ unsigned long long warp_sum_u64(unsigned long long value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kAllLanes, value, offset);
  }
  return value;
}

template <int M, bool Derivatives>
__global__ void atm_kernel(GeometryView g, ModelView m, GraphView graph,
                           const long long* __restrict__ item_prefix, long long item_begin,
                           long long item_count, const double* __restrict__ w,
                           const double* __restrict__ dw, AccumView acc,
                           double* __restrict__ partials, unsigned long long* __restrict__ counters) {
  const long long local_item = global_warp();
  if (local_item >= item_count) return;
  const long long item = item_begin + local_item;
  // Centre owning this item: the last i with item_prefix[i] <= item.
  int lo = 0;
  int hi = g.n;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (item_prefix[mid] <= item) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int i = lo;
  const int lane = lane_id();
  const int a = static_cast<int>(item - item_prefix[i]) * kWarpSize + lane;
  const long long row = graph.row_ptr[i];
  const int k = graph.len_atm[i];
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const int nref_i = m.nref[si];
  const int S = m.n_species;
  Weights<M> wi;
  load_weights<M, Derivatives>(w, dw, i, wi);

  double energy = 0.0;
  double g_i = 0.0;
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  unsigned long long candidates = 0;
  unsigned long long accepted = 0;
  unsigned long long c6_skipped = 0;

  bool valid_u = false;
  int ju = 0;
  int su = 0;
  int nref_u = 0;
  math::Vec3 diu{0.0, 0.0, 0.0};
  double r2iu = 0.0;
  if (a < k) {
    const int4 edge = graph.edges[row + a];
    ju = edge.x;
    diu = edge_displacement(g, xi, edge);
    r2iu = math::norm2(diu);
    valid_u = screened(r2iu, m.disp3_cutoff2);
    su = species(g, ju);
    nref_u = m.nref[su];
  }
  double g_u = 0.0;
  math::Vec3 grad_u{0.0, 0.0, 0.0};
  if (valid_u) {
    Weights<M> wu;
    load_weights<M, Derivatives>(w, dw, ju, wu);
    double c6iu, dc6iu_i, dc6iu_u;
    contract<M, Derivatives>(wi, nref_i, w + ju * kRefPad, Derivatives ? dw + ju * kRefPad : nullptr, nref_u,
                             m.c6ref + (si * S + su) * kC6Block, c6iu, dc6iu_i, dc6iu_u);
    const double r0iu = m.pair[si * S + su].r0_atm;
    for (int t = a + 1; t < k; ++t) {
      const int4 edge = graph.edges[row + t];
      const int kv = edge.x;
      const math::Vec3 div = edge_displacement(g, xi, edge);
      const double r2iv = math::norm2(div);
      if (!screened(r2iv, m.disp3_cutoff2)) continue;
      candidates += 1;
      const math::Vec3 duv{div.x - diu.x, div.y - diu.y, div.z - diu.z};
      const double r2uv = math::norm2(duv);
      if (!screened(r2uv, m.disp3_cutoff2)) continue;
      const int sv = species(g, kv);
      const int nref_v = m.nref[sv];
      const double* wv = w + kv * kRefPad;
      const double* dwv = Derivatives ? dw + kv * kRefPad : nullptr;
      double c6iv, dc6iv_i, dc6iv_v, c6uv, dc6uv_u, dc6uv_v;
      contract<M, Derivatives>(wi, nref_i, wv, dwv, nref_v, m.c6ref + (si * S + sv) * kC6Block,
                               c6iv, dc6iv_i, dc6iv_v);
      contract<M, Derivatives>(wu, nref_u, wv, dwv, nref_v, m.c6ref + (su * S + sv) * kC6Block,
                               c6uv, dc6uv_u, dc6uv_v);
      if (fabs(c6iu) < DBL_EPSILON || fabs(c6iv) < DBL_EPSILON || fabs(c6uv) < DBL_EPSILON) {
        c6_skipped += 1;
        continue;
      }
      accepted += 1;
      const double r0 = r0iu * m.pair[si * S + sv].r0_atm * m.pair[su * S + sv].r0_atm;
      const double r2 = r2iu * r2iv * r2uv;
      const double r1 = sqrt(r2);
      const double r3 = r2 * r1;
      const double r5 = r3 * r2;
      const double q = 6.0 * pow(r0 / r1, m.alp3);
      const double fdmp = 1.0 / (1.0 + q);
      const double pa = r2iu + r2uv - r2iv;
      const double pb = r2iu - r2uv + r2iv;
      const double pc = -r2iu + r2uv + r2iv;
      const double abc = pa * pb * pc;
      const double ang = 0.375 * abc / r5 + 1.0 / r3;
      const double pre = kThird * m.s9 * sqrt(fabs(c6iu * c6iv * c6uv));
      const double e = pre * ang * fdmp;
      energy += e;
      if (Derivatives) {
        const double dfac = fdmp * fdmp * (0.5 * m.alp3) * q;
        auto partial = [&](double x, double dp) {
          const double dang = 0.375 * (dp / r5 - 2.5 * abc / (r5 * x)) - 1.5 / (r3 * x);
          return 2.0 * pre * (dang * fdmp + ang * dfac / x);
        };
        const double s_iu = partial(r2iu, pb * pc + pa * pc - pa * pb);
        const double s_iv = partial(r2iv, -pb * pc + pa * pc + pa * pb);
        const double s_uv = partial(r2uv, pb * pc - pa * pc + pa * pb);
        const math::Vec3 g_iu{s_iu * diu.x, s_iu * diu.y, s_iu * diu.z};
        const math::Vec3 g_iv{s_iv * div.x, s_iv * div.y, s_iv * div.z};
        const math::Vec3 g_uv{s_uv * duv.x, s_uv * duv.y, s_uv * duv.z};
        grad_i.x -= g_iu.x + g_iv.x;
        grad_i.y -= g_iu.y + g_iv.y;
        grad_i.z -= g_iu.z + g_iv.z;
        grad_u.x += g_iu.x - g_uv.x;
        grad_u.y += g_iu.y - g_uv.y;
        grad_u.z += g_iu.z - g_uv.z;
        atomic_add3(acc.grad + 3 * kv, {g_iv.x + g_uv.x, g_iv.y + g_uv.y, g_iv.z + g_uv.z});
        vir.add_outer(diu, g_iu);
        vir.add_outer(div, g_iv);
        vir.add_outer(duv, g_uv);
        const double half = 0.5 * e;
        g_i += half * (dc6iu_i / c6iu + dc6iv_i / c6iv);
        g_u += half * (dc6iu_u / c6iu + dc6uv_u / c6uv);
        atomicAdd(acc.g + kv, half * (dc6iv_v / c6iv + dc6uv_v / c6uv));
      }
    }
  }
  if (Derivatives && valid_u) {
    atomic_add3(acc.grad + 3 * ju, grad_u);
    atomicAdd(acc.g + ju, g_u);
  }

  double* out = partials + 7 * local_item;
  energy = warp_sum(energy);
  if (lane == 0) out[0] = energy;
  if (Derivatives) {
    for (int c = 0; c < 6; ++c) {
      const double total = warp_sum(vir.v[c]);
      if (lane == 0) out[1 + c] = total;
    }
    g_i = warp_sum(g_i);
    const math::Vec3 total{warp_sum(grad_i.x), warp_sum(grad_i.y), warp_sum(grad_i.z)};
    if (lane == 0) {
      atomicAdd(acc.g + i, g_i);
      atomic_add3(acc.grad + 3 * i, total);
    }
  } else if (lane == 0) {
    for (int c = 0; c < 6; ++c) out[1 + c] = 0.0;
  }
  candidates = warp_sum_u64(candidates);
  accepted = warp_sum_u64(accepted);
  c6_skipped = warp_sum_u64(c6_skipped);
  if (lane == 0) {
    if (candidates != 0) atomicAdd(counters, candidates);
    if (accepted != 0) atomicAdd(counters + 1, accepted);
    if (c6_skipped != 0) atomicAdd(counters + 2, c6_skipped);
  }
}


template <int M>
__device__ __forceinline__ double dot(const double* x, const double* y, int n) {
  double sum = 0.0;
#pragma unroll
  for (int b = 0; b < M; ++b) {
    if (b >= n) break;
    sum += x[b] * y[b];
  }
  return sum;
}

// (r0 / r1)^(alp_eff / 3) from y = r0^2 / r1^2. For the default exponent 16 it
// is y^(8/3) = y^2 cbrt(y^2), which avoids pow.
__device__ __forceinline__ double atm_power(double y, const ModelView& m) {
  if (m.alp_eff == 16.0) {
    const double y2 = y * y;
    return y2 * cbrt(y2);
  }
  return pow(y, 0.5 * m.alp3);
}

// Inner sites of one tile, prepared once per warp.
struct VSlot {
  double dx, dy, dz, r2iv;
  double c6iv, dc6iv_i, dc6iv_v, r0iv;
  double w[kRefPad];
  double dw[kRefPad];
  int kv, sv, nref, valid;
};

// Per-tile accumulators for the inner sites: dE/dx (3), dE/dCN, energy split.
struct VAccum {
  double gx, gy, gz, g, e;
};

template <int M, bool Derivatives, bool Split>
__global__ void __launch_bounds__(kWarpBlock)
atm_canonical_kernel(GeometryView g, ModelView m, GraphView graph,
                     const long long* __restrict__ item_prefix, long long item_begin,
                     long long item_count, const double* __restrict__ w,
                     const double* __restrict__ dw, ContractedView cv, AccumView acc,
                     double* __restrict__ partials, unsigned long long* __restrict__ counters) {
  __shared__ VSlot tiles[kWarpsPerBlock][kWarpSize];
  __shared__ VAccum vacc[kWarpsPerBlock][kWarpSize];
  const long long local_item = global_warp();
  if (local_item >= item_count) return;
  const int warp_in_block = static_cast<int>(threadIdx.x / kWarpSize);
  VSlot* tile = tiles[warp_in_block];
  VAccum* va = vacc[warp_in_block];
  const long long item = item_begin + local_item;
  int lo = 0;
  int hi = g.n;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (item_prefix[mid] <= item) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int i = lo;
  const int lane = lane_id();
  const int first = static_cast<int>(item - item_prefix[i]) * kWarpSize;
  const int a = first + lane;
  const long long row = graph.row_ptr[i];
  const int k = graph.counts[4 * i + 3];
  const int* upper = graph.upper + row;
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const int S = m.n_species;
  const double* ui = cv.u + static_cast<long long>(i) * S * kRefPad;
  const double* dui = Derivatives ? cv.du + static_cast<long long>(i) * S * kRefPad : nullptr;

  double energy = 0.0;
  double g_i = 0.0;
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  unsigned long long candidates = 0;
  unsigned long long accepted = 0;
  unsigned long long c6_skipped = 0;

  // Outer site u owned by this lane.
  bool valid_u = false;
  int ju = 0;
  int su = 0;
  int nref_u = 0;
  math::Vec3 diu{0.0, 0.0, 0.0};
  double r2iu = 0.0;
  double c6iu = 0.0, dc6iu_i = 0.0, dc6iu_u = 0.0, inv_c6iu = 0.0, r0iu = 0.0;
  const double* uu = nullptr;
  const double* duu = nullptr;
  if (a < k) {
    const int4 edge = graph.edges[row + upper[a]];
    ju = edge.x;
    diu = edge_displacement(g, xi, edge);
    r2iu = math::norm2(diu);
    valid_u = screened(r2iu, m.disp3_cutoff2);
    su = species(g, ju);
    nref_u = m.nref[su];
  }
  if (valid_u) {
    const double* wu = w + ju * kRefPad;
    c6iu = dot<M>(ui + su * kRefPad, wu, nref_u);
    if (Derivatives) {
      dc6iu_i = dot<M>(dui + su * kRefPad, wu, nref_u);
      dc6iu_u = dot<M>(ui + su * kRefPad, dw + ju * kRefPad, nref_u);
    }
    inv_c6iu = 1.0 / c6iu;
    r0iu = m.pair[si * S + su].r0_atm;
    uu = cv.u + static_cast<long long>(ju) * S * kRefPad;
    duu = Derivatives ? cv.du + static_cast<long long>(ju) * S * kRefPad : nullptr;
  }
  double g_u = 0.0;
  double e_u = 0.0;
  math::Vec3 grad_u{0.0, 0.0, 0.0};

  for (int base = first; base < k; base += kWarpSize) {
    // Prepare the inner sites of this tile (one per lane).
    {
      VSlot& slot = tile[lane];
      const int t = base + lane;
      slot.valid = 0;
      if (t < k) {
        const int4 edge = graph.edges[row + upper[t]];
        const math::Vec3 div = edge_displacement(g, xi, edge);
        const double r2iv = math::norm2(div);
        if (screened(r2iv, m.disp3_cutoff2)) {
          const int kv = edge.x;
          const int sv = species(g, kv);
          const int nref_v = m.nref[sv];
          slot.dx = div.x;
          slot.dy = div.y;
          slot.dz = div.z;
          slot.r2iv = r2iv;
          slot.kv = kv;
          slot.sv = sv;
          slot.nref = nref_v;
          slot.valid = 1;
          slot.r0iv = m.pair[si * S + sv].r0_atm;
#pragma unroll
          for (int b = 0; b < kRefPad; ++b) {
            slot.w[b] = w[kv * kRefPad + b];
            slot.dw[b] = Derivatives ? dw[kv * kRefPad + b] : 0.0;
          }
          slot.c6iv = dot<M>(ui + sv * kRefPad, slot.w, nref_v);
          slot.dc6iv_i = Derivatives ? dot<M>(dui + sv * kRefPad, slot.w, nref_v) : 0.0;
          slot.dc6iv_v = Derivatives ? dot<M>(ui + sv * kRefPad, slot.dw, nref_v) : 0.0;
        }
      }
      va[lane] = {0.0, 0.0, 0.0, 0.0, 0.0};
    }
    __syncwarp();
    // Rotate: at step s, lane l works on slot (l + s) mod 32, so the lanes of
    // a step touch distinct inner sites and can accumulate without atomics.
    for (int step = 0; step < kWarpSize; ++step) {
      const int t = (lane + step) & (kWarpSize - 1);
      const VSlot& v = tile[t];
      if (valid_u && v.valid && base + t > a) {
        candidates += 1;
        const math::Vec3 div{v.dx, v.dy, v.dz};
        const math::Vec3 duv{div.x - diu.x, div.y - diu.y, div.z - diu.z};
        const double r2uv = math::norm2(duv);
        if (screened(r2uv, m.disp3_cutoff2)) {
          const double c6iv = v.c6iv;
          const double c6uv = dot<M>(uu + v.sv * kRefPad, v.w, v.nref);
          if (fabs(c6iu) < DBL_EPSILON || fabs(c6iv) < DBL_EPSILON || fabs(c6uv) < DBL_EPSILON) {
            c6_skipped += 1;
          } else {
            accepted += 1;
            const double r2iv = v.r2iv;
            const double r0 = r0iu * v.r0iv * m.pair[su * S + v.sv].r0_atm;
            const double r2 = r2iu * r2iv * r2uv;
            const double ir1 = rsqrt(r2);
            const double ir2 = ir1 * ir1;
            const double ir3 = ir2 * ir1;
            const double ir5 = ir3 * ir2;
            const double q = 6.0 * atm_power(r0 * r0 * ir2, m);
            const double fdmp = 1.0 / (1.0 + q);
            const double pa = r2iu + r2uv - r2iv;
            const double pb = r2iu - r2uv + r2iv;
            const double pc = -r2iu + r2uv + r2iv;
            const double abc = pa * pb * pc;
            const double ang = 0.375 * abc * ir5 + ir3;
            const double pre = m.s9 * sqrt(fabs(c6iu * c6iv * c6uv));
            const double e = pre * ang * fdmp;
            energy += e;
            if (Split) {
              const double third = e / 3.0;
              e_u += third;
              va[t].e += third;
            }
            if (Derivatives) {
              const double dc6uv_u = dot<M>(duu + v.sv * kRefPad, v.w, v.nref);
              const double dc6uv_v = dot<M>(uu + v.sv * kRefPad, v.dw, v.nref);
              const double dfac = fdmp * fdmp * (0.5 * m.alp3) * q;
              const double base_ang = 0.375 * ir5;
              const double tail = -2.5 * abc * base_ang - 1.5 * ir3;
              auto partial = [&](double inv_x, double dp) {
                return 2.0 * pre * ((base_ang * dp + tail * inv_x) * fdmp + ang * dfac * inv_x);
              };
              const double s_iu = partial(r2iv * r2uv * ir2, pb * pc + pa * pc - pa * pb);
              const double s_iv = partial(r2iu * r2uv * ir2, -pb * pc + pa * pc + pa * pb);
              const double s_uv = partial(r2iu * r2iv * ir2, pb * pc - pa * pc + pa * pb);
              const math::Vec3 g_iu{s_iu * diu.x, s_iu * diu.y, s_iu * diu.z};
              const math::Vec3 g_iv{s_iv * div.x, s_iv * div.y, s_iv * div.z};
              const math::Vec3 g_uv{s_uv * duv.x, s_uv * duv.y, s_uv * duv.z};
              grad_i.x -= g_iu.x + g_iv.x;
              grad_i.y -= g_iu.y + g_iv.y;
              grad_i.z -= g_iu.z + g_iv.z;
              grad_u.x += g_iu.x - g_uv.x;
              grad_u.y += g_iu.y - g_uv.y;
              grad_u.z += g_iu.z - g_uv.z;
              vir.add_outer(diu, g_iu);
              vir.add_outer(div, g_iv);
              vir.add_outer(duv, g_uv);
              const double inv_pair = 1.0 / (c6iv * c6uv);
              const double half = 0.5 * e;
              g_i += half * (dc6iu_i * inv_c6iu + v.dc6iv_i * c6uv * inv_pair);
              g_u += half * (dc6iu_u * inv_c6iu + dc6uv_u * c6iv * inv_pair);
              VAccum& out = va[t];
              out.gx += g_iv.x + g_uv.x;
              out.gy += g_iv.y + g_uv.y;
              out.gz += g_iv.z + g_uv.z;
              out.g += half * (v.dc6iv_v * c6uv + dc6uv_v * c6iv) * inv_pair;
            }
          }
        }
      }
      __syncwarp();
    }
    // Flush the inner-site accumulators: one atomic per site and tile.
    {
      const VSlot& v = tile[lane];
      const VAccum& out = va[lane];
      if (v.valid && base + lane < k) {
        if (Derivatives && (out.gx != 0.0 || out.gy != 0.0 || out.gz != 0.0 || out.g != 0.0)) {
          atomic_add3(acc.grad + 3 * v.kv, {out.gx, out.gy, out.gz});
          atomicAdd(acc.g + v.kv, out.g);
        }
        if (Split && out.e != 0.0) atomicAdd(acc.atm_split + v.kv, out.e);
      }
    }
    __syncwarp();
  }
  if (Derivatives && valid_u) {
    atomic_add3(acc.grad + 3 * ju, grad_u);
    atomicAdd(acc.g + ju, g_u);
  }
  if (Split && valid_u) atomicAdd(acc.atm_split + ju, e_u);

  double* out = partials + 7 * local_item;
  energy = warp_sum(energy);
  if (lane == 0) {
    out[0] = energy;
    if (Split) atomicAdd(acc.atm_split + i, energy / 3.0);
  }
  if (Derivatives) {
    for (int c = 0; c < 6; ++c) {
      const double total = warp_sum(vir.v[c]);
      if (lane == 0) out[1 + c] = total;
    }
    g_i = warp_sum(g_i);
    const math::Vec3 total{warp_sum(grad_i.x), warp_sum(grad_i.y), warp_sum(grad_i.z)};
    if (lane == 0) {
      atomicAdd(acc.g + i, g_i);
      atomic_add3(acc.grad + 3 * i, total);
    }
  } else if (lane == 0) {
    for (int c = 0; c < 6; ++c) out[1 + c] = 0.0;
  }
  candidates = warp_sum_u64(candidates);
  accepted = warp_sum_u64(accepted);
  c6_skipped = warp_sum_u64(c6_skipped);
  if (lane == 0) {
    if (candidates != 0) atomicAdd(counters, candidates);
    if (accepted != 0) atomicAdd(counters + 1, accepted);
    if (c6_skipped != 0) atomicAdd(counters + 2, c6_skipped);
  }
}

// Add the item partials of one launch to the per-centre accumulators in item
// order (deterministic).
__global__ void atm_reduce_kernel(const long long* __restrict__ item_prefix, int n,
                                  long long item_begin, long long item_count,
                                  const double* __restrict__ partials, AccumView acc) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const long long first = max(item_prefix[i], item_begin);
  const long long last = min(item_prefix[i + 1], item_begin + item_count);
  if (first >= last) return;
  double sums[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (long long item = first; item < last; ++item) {
    const double* p = partials + 7 * (item - item_begin);
#pragma unroll
    for (int c = 0; c < 7; ++c) sums[c] += p[c];
  }
  acc.e_atm[i] += sums[0];
#pragma unroll
  for (int c = 0; c < 6; ++c) acc.vir[c * n + i] += sums[1 + c];
}

template <int M>
void launch_atm_m(const GeometryView& g, const ModelView& m, const GraphView& graph,
                  const long long* item_prefix, long long item_begin, long long item_count,
                  const double* w, const double* dw, const AccumView& acc, double* partials,
                  unsigned long long* counters, cudaStream_t stream) {
  const auto blocks = static_cast<unsigned>((item_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (dw != nullptr) {
    atm_kernel<M, true><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, item_prefix, item_begin,
                                                           item_count, w, dw, acc, partials, counters);
  } else {
    atm_kernel<M, false><<<blocks, kWarpBlock, 0, stream>>>(g, m, graph, item_prefix, item_begin,
                                                            item_count, w, dw, acc, partials, counters);
  }
}

template <int M>
void launch_atm_canonical_m(const GeometryView& g, const ModelView& m, const GraphView& graph,
                            const long long* item_prefix, long long item_begin, long long item_count,
                            const double* w, const double* dw, const ContractedView& cv,
                            const AccumView& acc, double* partials, unsigned long long* counters,
                            bool split, cudaStream_t stream) {
  const auto blocks = static_cast<unsigned>((item_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
#define VF_ATM_CANONICAL(D, SPLIT)                                                         \
  atm_canonical_kernel<M, D, SPLIT><<<blocks, kWarpBlock, 0, stream>>>(                    \
      g, m, graph, item_prefix, item_begin, item_count, w, dw, cv, acc, partials, counters)
  if (dw != nullptr && split) {
    VF_ATM_CANONICAL(true, true);
  } else if (dw != nullptr) {
    VF_ATM_CANONICAL(true, false);
  } else if (split) {
    VF_ATM_CANONICAL(false, true);
  } else {
    VF_ATM_CANONICAL(false, false);
  }
#undef VF_ATM_CANONICAL
}

}  // namespace

void launch_atm_canonical(const GeometryView& g, const ModelView& m, const GraphView& graph,
                          const long long* item_prefix, long long item_begin, long long item_count,
                          const double* w, const double* dw, const ContractedView& cv,
                          const AccumView& acc, double* partials, unsigned long long* counters,
                          bool split_atoms, cudaStream_t stream) {
  if (g.n == 0 || item_count == 0) return;
  dispatch_max_nref(m.max_nref, [&](auto max_nref) {
    launch_atm_canonical_m<decltype(max_nref)::value>(g, m, graph, item_prefix, item_begin,
                                                      item_count, w, dw, cv, acc, partials,
                                                      counters, split_atoms, stream);
  });
  VF_LAUNCH_CHECK(stream);
  atm_reduce_kernel<<<(g.n + 127) / 128, 128, 0, stream>>>(item_prefix, g.n, item_begin, item_count,
                                                           partials, acc);
  VF_LAUNCH_CHECK(stream);
}

void launch_atm(const GeometryView& g, const ModelView& m, const GraphView& graph,
                const long long* item_prefix, long long item_begin, long long item_count,
                const double* w, const double* dw, const AccumView& acc, double* partials,
                unsigned long long* counters, cudaStream_t stream) {
  if (g.n == 0 || item_count == 0) return;
  dispatch_max_nref(m.max_nref, [&](auto max_nref) {
    launch_atm_m<decltype(max_nref)::value>(g, m, graph, item_prefix, item_begin, item_count, w, dw,
                                            acc, partials, counters, stream);
  });
  VF_LAUNCH_CHECK(stream);
  atm_reduce_kernel<<<(g.n + 127) / 128, 128, 0, stream>>>(item_prefix, g.n, item_begin, item_count,
                                                           partials, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
