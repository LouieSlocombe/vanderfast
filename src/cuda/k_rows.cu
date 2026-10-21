// Row-chunked owner kernels for the CN, pair, and CN-reverse stages.
//
// A row (one centre's edges in a view) is split into work items of
// kRowChunk edges. Each warp processes one item and stores its partial sums;
// a reduction kernel then adds the partials of each centre in item order.
// This balances rows of very different length (a two-atom cell has thousands
// of edges per row) while keeping every result deterministic.
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

__device__ __forceinline__ int centre_of(const long long* __restrict__ prefix, int n, long long item) {
  int lo = 0;
  int hi = n;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (prefix[mid] <= item) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

struct ItemRange {
  int i;
  long long begin;
  long long end;
};

__device__ __forceinline__ ItemRange item_range(const long long* prefix, int n, long long item,
                                                const long long* row_ptr, const int* len) {
  const int i = centre_of(prefix, n, item);
  const long long chunk = item - prefix[i];
  const long long row = row_ptr[i];
  const long long begin = row + chunk * kRowChunk;
  const long long end = min(begin + kRowChunk, row + len[i]);
  return {i, begin, end};
}

__global__ void cn_items_kernel(GeometryView g, ModelView m, GraphView graph,
                                const long long* __restrict__ prefix, long long items,
                                double* __restrict__ partials) {
  const long long item = global_warp();
  if (item >= items) return;
  const ItemRange r = item_range(prefix, g.n, item, graph.row_ptr, graph.len_cn);
  const int lane = lane_id();
  const math::Vec3 xi = position(g, r.i);
  const int si = species(g, r.i);
  double sum = 0.0;
  for (long long e = r.begin + lane; e < r.end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const double r2 = math::norm2(edge_displacement(g, xi, edge));
    if (r2 > m.cn_cutoff2 || r2 < math::kCnExclusionR2) continue;
    const double rc = m.pair[si * m.n_species + species(g, edge.x)].rc_cn;
    sum += math::cn_count(rc, sqrt(r2));
  }
  sum = warp_sum(sum);
  if (lane == 0) partials[item] = sum;
}

__global__ void cn_items_reduce(const long long* __restrict__ prefix, int n,
                                const double* __restrict__ partials, double* __restrict__ cn) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  double sum = 0.0;
  for (long long item = prefix[i]; item < prefix[i + 1]; ++item) sum += partials[item];
  cn[i] = sum;
}

constexpr int kPairPartials = 11;  // energy, g, grad[3], vir[6]

template <int M, bool Derivatives>
__global__ void pair_items_kernel(GeometryView g, ModelView m, GraphView graph,
                                  const long long* __restrict__ prefix, long long items,
                                  const double* __restrict__ w, const double* __restrict__ dw,
                                  ContractedView cv, double* __restrict__ partials) {
  const long long item = global_warp();
  if (item >= items) return;
  const ItemRange r = item_range(prefix, g.n, item, graph.row_ptr, graph.len_pair);
  const int i = r.i;
  const int lane = lane_id();
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const long long ubase = static_cast<long long>(i) * m.n_species * kRefPad;
  double energy = 0.0;
  double g_i = 0.0;
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  for (long long e = r.begin + lane; e < r.end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const int j = edge.x;
    const math::Vec3 d = edge_displacement(g, xi, edge);
    const double r2 = math::norm2(d);
    if (r2 > m.disp2_cutoff2 || r2 < DBL_EPSILON) continue;
    const int sj = species(g, j);
    const PairConst pc = m.pair[si * m.n_species + sj];
    const int nref_j = m.nref[sj];
    const double* wj = w + j * kRefPad;
    const double* ui = cv.u + ubase + sj * kRefPad;
    double c6 = 0.0;
    double dc6_i = 0.0;
#pragma unroll
    for (int b = 0; b < M; ++b) {
      if (b >= nref_j) break;
      c6 += ui[b] * wj[b];
      if constexpr (Derivatives) dc6_i += cv.du[ubase + sj * kRefPad + b] * wj[b];
    }
    const double r4 = r2 * r2;
    const double t6 = 1.0 / (r4 * r2 + pc.r0_6);
    const double t8 = 1.0 / (r4 * r4 + pc.r0_8);
    const double edisp = m.s6 * t6 + m.s8 * pc.kappa * t8;
    energy += -0.5 * c6 * edisp;
    if constexpr (Derivatives) {
      const double d6 = -6.0 * r4 * t6 * t6;
      const double d8 = -8.0 * r4 * r2 * t8 * t8;
      const double gdisp = m.s6 * d6 + m.s8 * pc.kappa * d8;
      const double scale = -0.5 * c6 * gdisp;
      const math::Vec3 de{scale * d.x, scale * d.y, scale * d.z};
      vir.add_outer(d, de);
      if (j != i) {  // the reverse edge in row j supplies the other endpoint
        grad_i.x -= 2.0 * de.x;
        grad_i.y -= 2.0 * de.y;
        grad_i.z -= 2.0 * de.z;
      }
      g_i += -dc6_i * edisp;
    }
  }
  double* out = partials + kPairPartials * item;
  const double totals[kPairPartials] = {
      energy, g_i, grad_i.x, grad_i.y, grad_i.z, vir.v[0], vir.v[1], vir.v[2], vir.v[3], vir.v[4], vir.v[5]};
#pragma unroll
  for (int c = 0; c < kPairPartials; ++c) {
    const double total = (Derivatives || c == 0) ? warp_sum(totals[c]) : 0.0;
    if (lane == 0) out[c] = total;
  }
}

__global__ void pair_items_reduce(const long long* __restrict__ prefix, int n,
                                  const double* __restrict__ partials, AccumView acc) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  double sums[kPairPartials] = {};
  for (long long item = prefix[i]; item < prefix[i + 1]; ++item) {
#pragma unroll
    for (int c = 0; c < kPairPartials; ++c) sums[c] += partials[kPairPartials * item + c];
  }
  acc.e_pair[i] += sums[0];
  acc.g[i] += sums[1];
  acc.grad[3 * i] += sums[2];
  acc.grad[3 * i + 1] += sums[3];
  acc.grad[3 * i + 2] += sums[4];
#pragma unroll
  for (int c = 0; c < 6; ++c) acc.vir[c * n + i] += sums[5 + c];
}

constexpr int kReversePartials = 9;  // grad[3], vir[6]

__global__ void cn_reverse_items_kernel(GeometryView g, ModelView m, GraphView graph,
                                        const long long* __restrict__ prefix, long long items,
                                        const double* __restrict__ dEdcn,
                                        double* __restrict__ partials) {
  const long long item = global_warp();
  if (item >= items) return;
  const ItemRange r = item_range(prefix, g.n, item, graph.row_ptr, graph.len_cn);
  const int i = r.i;
  const int lane = lane_id();
  const double gi = dEdcn[i];
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  for (long long e = r.begin + lane; e < r.end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const math::Vec3 d = edge_displacement(g, xi, edge);
    const double r2 = math::norm2(d);
    if (r2 > m.cn_cutoff2 || r2 < math::kCnExclusionR2) continue;
    const double rc = m.pair[si * m.n_species + species(g, edge.x)].rc_cn;
    const double rr = sqrt(r2);
    const double cf = math::cn_count(rc, rr);
    // Edge (i -> j) carries g_i and its reverse edge carries g_j; both act on
    // r_i, and the pair's virial is split evenly between the two rows.
    const double scale = (gi + dEdcn[edge.x]) * math::cn_count_derivative(cf, rc, r2) / rr;
    const math::Vec3 de{scale * d.x, scale * d.y, scale * d.z};
    grad_i.x -= de.x;
    grad_i.y -= de.y;
    grad_i.z -= de.z;
    vir.add_outer(d, {0.5 * de.x, 0.5 * de.y, 0.5 * de.z});
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

__global__ void cn_reverse_items_reduce(const long long* __restrict__ prefix, int n,
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

unsigned item_blocks(long long items) {
  return static_cast<unsigned>((items + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

unsigned centre_blocks(int n) { return static_cast<unsigned>((n + 127) / 128); }

}  // namespace

void launch_cn_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                     const long long* prefix, long long items, double* partials, double* cn,
                     cudaStream_t stream) {
  if (g.n == 0) return;
  if (items > 0) {
    cn_items_kernel<<<item_blocks(items), kWarpBlock, 0, stream>>>(g, m, graph, prefix, items, partials);
    VF_LAUNCH_CHECK(stream);
  }
  cn_items_reduce<<<centre_blocks(g.n), 128, 0, stream>>>(prefix, g.n, partials, cn);
  VF_LAUNCH_CHECK(stream);
}

void launch_pair_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                       const long long* prefix, long long items, const double* w, const double* dw,
                       const ContractedView& cv, const AccumView& acc, double* partials,
                       cudaStream_t stream) {
  if (g.n == 0 || items == 0) return;
  dispatch_max_nref(m.max_nref, [&](auto max_nref) {
    constexpr int M = decltype(max_nref)::value;
    if (dw != nullptr) {
      pair_items_kernel<M, true><<<item_blocks(items), kWarpBlock, 0, stream>>>(
          g, m, graph, prefix, items, w, dw, cv, partials);
    } else {
      pair_items_kernel<M, false><<<item_blocks(items), kWarpBlock, 0, stream>>>(
          g, m, graph, prefix, items, w, dw, cv, partials);
    }
  });
  VF_LAUNCH_CHECK(stream);
  pair_items_reduce<<<centre_blocks(g.n), 128, 0, stream>>>(prefix, g.n, partials, acc);
  VF_LAUNCH_CHECK(stream);
}

void launch_cn_reverse_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                             const long long* prefix, long long items, const AccumView& acc,
                             double* partials, cudaStream_t stream) {
  if (g.n == 0 || items == 0) return;
  cn_reverse_items_kernel<<<item_blocks(items), kWarpBlock, 0, stream>>>(g, m, graph, prefix, items,
                                                                         acc.g, partials);
  VF_LAUNCH_CHECK(stream);
  cn_reverse_items_reduce<<<centre_blocks(g.n), 128, 0, stream>>>(prefix, g.n, partials, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
