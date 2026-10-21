// K1: coordination numbers. K5: CN-mediated forces and virial.
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

__global__ void cn_kernel(GeometryView g, ModelView m, GraphView graph, double* __restrict__ cn) {
  const int i = global_warp();
  if (i >= g.n) return;
  const int lane = lane_id();
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const long long begin = graph.row_ptr[i];
  const long long end = begin + graph.len_cn[i];
  double sum = 0.0;
  for (long long e = begin + lane; e < end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const double r2 = math::norm2(edge_displacement(g, xi, edge));
    if (r2 > m.cn_cutoff2 || r2 < math::kCnExclusionR2) continue;
    const double rc = m.pair[si * m.n_species + species(g, edge.x)].rc_cn;
    sum += math::cn_count(rc, sqrt(r2));
  }
  sum = warp_sum(sum);
  if (lane == 0) cn[i] = sum;
}

__global__ void cn_reverse_kernel(GeometryView g, ModelView m, GraphView graph, AccumView acc) {
  const int i = global_warp();
  if (i >= g.n) return;
  const double gi = acc.g[i];
  if (gi == 0.0) return;
  const int lane = lane_id();
  const math::Vec3 xi = position(g, i);
  const int si = species(g, i);
  const long long begin = graph.row_ptr[i];
  const long long end = begin + graph.len_cn[i];
  math::Vec3 grad_i{0.0, 0.0, 0.0};
  Sym6 vir;
  for (long long e = begin + lane; e < end; e += kWarpSize) {
    const int4 edge = graph.edges[e];
    const math::Vec3 d = edge_displacement(g, xi, edge);
    const double r2 = math::norm2(d);
    if (r2 > m.cn_cutoff2 || r2 < math::kCnExclusionR2) continue;
    const double rc = m.pair[si * m.n_species + species(g, edge.x)].rc_cn;
    const double r = sqrt(r2);
    const double cf = math::cn_count(rc, r);
    const double scale = gi * math::cn_count_derivative(cf, rc, r2) / r;
    const math::Vec3 de{scale * d.x, scale * d.y, scale * d.z};
    atomic_add3(acc.grad + 3 * edge.x, de);
    grad_i.x -= de.x;
    grad_i.y -= de.y;
    grad_i.z -= de.z;
    vir.add_outer(d, de);
  }
  const math::Vec3 total{warp_sum(grad_i.x), warp_sum(grad_i.y), warp_sum(grad_i.z)};
  if (lane == 0) atomic_add3(acc.grad + 3 * i, total);
  vir.reduce_and_add(acc.vir, i, acc.n);
}

}  // namespace

void launch_cn(const GeometryView& g, const ModelView& m, const GraphView& graph, double* cn,
               cudaStream_t stream) {
  if (g.n == 0) return;
  cn_kernel<<<warp_grid(g.n), kWarpBlock, 0, stream>>>(g, m, graph, cn);
  VF_LAUNCH_CHECK(stream);
}

void launch_cn_reverse(const GeometryView& g, const ModelView& m, const GraphView& graph,
                       const AccumView& acc, cudaStream_t stream) {
  if (g.n == 0) return;
  cn_reverse_kernel<<<warp_grid(g.n), kWarpBlock, 0, stream>>>(g, m, graph, acc);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
