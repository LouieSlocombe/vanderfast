#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/d3_math.hpp"
#include "cuda_check.cuh"
#include "neighbor.cuh"

namespace vanderfast::cuda {

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kAtmItems = 32;  // outer sites per ATM work item (kAtmItemWidth)

__device__ __forceinline__ math::Vec3 load(const double4_32a* pos, int i) {
  const double4_32a p = pos[i];
  return {p.x, p.y, p.z};
}

__device__ __forceinline__ void decode_image(long long m, const ImageBox& box, int& a, int& b, int& c) {
  const long long s1 = 2 * box.range[1] + 1;
  const long long s2 = 2 * box.range[2] + 1;
  const long long m0 = m / (s1 * s2);
  const long long rest = m - m0 * s1 * s2;
  const long long m1 = rest / s2;
  a = static_cast<int>(m0) - box.range[0];
  b = static_cast<int>(m1) - box.range[1];
  c = static_cast<int>(rest - m1 * s2) - box.range[2];
}

__device__ __forceinline__ int band_of(double r2, const BandSpec& bands) {
  if (bands.n_bands > 0 && r2 <= bands.radius2[0]) return 0;
  if (bands.n_bands > 1 && r2 <= bands.radius2[1]) return 1;
  if (bands.n_bands > 2 && r2 <= bands.radius2[2]) return 2;
  return -1;
}

// Visit candidate (j, n) number t of centre i; returns its band or -1.
__device__ __forceinline__ int classify(const GeometryView& g, const ImageBox& box,
                                        const BandSpec& bands, int i, math::Vec3 xi, long long t,
                                        int& j, int& a, int& b, int& c, double& r2) {
  const long long m = t / g.n;
  j = static_cast<int>(t - m * g.n);
  decode_image(m, box, a, b, c);
  r2 = DBL_MAX;
  if (j == i && a == 0 && b == 0 && c == 0) return -1;
  r2 = math::norm2(math::site_displacement(xi, load(g.pos, j), a, b, c, g.cell));
  return band_of(r2, bands);
}

__global__ void count_kernel(GeometryView g, ImageBox box, BandSpec bands, int* __restrict__ counts,
                             unsigned long long* __restrict__ scalars) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarp);
  const int lane = static_cast<int>(threadIdx.x % kWarp);
  if (warp >= g.n) return;
  const int i = warp;
  const math::Vec3 xi = load(g.pos, i);
  const long long total = box.images * g.n;
  int count0 = 0, count1 = 0, count2 = 0;
  double min_r2 = DBL_MAX;
  for (long long t = lane; t < total; t += kWarp) {
    int j, a, b, c;
    double r2;
    const int band = classify(g, box, bands, i, xi, t, j, a, b, c, r2);
    min_r2 = fmin(min_r2, r2);
    count0 += band == 0;
    count1 += band == 1;
    count2 += band == 2;
  }
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    count0 += __shfl_down_sync(kFullMask, count0, offset);
    count1 += __shfl_down_sync(kFullMask, count1, offset);
    count2 += __shfl_down_sync(kFullMask, count2, offset);
    min_r2 = fmin(min_r2, __shfl_down_sync(kFullMask, min_r2, offset));
  }
  if (lane == 0) {
    counts[3 * i] = count0;
    counts[3 * i + 1] = count1;
    counts[3 * i + 2] = count2;
    atomicMin(scalars + 3, static_cast<unsigned long long>(__double_as_longlong(min_r2)));
  }
}

__global__ void totals_kernel(const int* __restrict__ counts, int n, BandSpec bands,
                              long long* __restrict__ totals, int* __restrict__ len_cn,
                              int* __restrict__ len_pair, int* __restrict__ len_atm,
                              long long* __restrict__ items,
                              unsigned long long* __restrict__ scalars) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i > n) return;
  if (i == n) {
    totals[n] = 0;
    items[n] = 0;
    return;
  }
  int prefix[4] = {0, 0, 0, 0};
  for (int band = 0; band < 3; ++band) prefix[band + 1] = prefix[band] + counts[3 * i + band];
  totals[i] = prefix[3];
  int* lens[3] = {len_cn, len_pair, len_atm};
  for (int view = 0; view < 3; ++view) {
    lens[view][i] = bands.view_band[view] < 0 ? 0 : prefix[bands.view_band[view] + 1];
  }
  items[i] = (len_atm[i] + kAtmItems - 1) / kAtmItems;
  for (int band = 0; band < 3; ++band) {
    if (counts[3 * i + band] != 0) {
      atomicAdd(scalars + band, static_cast<unsigned long long>(counts[3 * i + band]));
    }
  }
}

__global__ void fill_kernel(GeometryView g, ImageBox box, BandSpec bands,
                            const long long* __restrict__ row_ptr, const int* __restrict__ counts,
                            int4* __restrict__ edges) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarp);
  const int lane = static_cast<int>(threadIdx.x % kWarp);
  if (warp >= g.n) return;
  const int i = warp;
  const math::Vec3 xi = load(g.pos, i);
  long long cursor0 = row_ptr[i];
  long long cursor1 = cursor0 + counts[3 * i];
  long long cursor2 = cursor1 + counts[3 * i + 1];
  const long long total = box.images * g.n;
  const unsigned below = (1u << lane) - 1u;
  for (long long base = 0; base < total; base += kWarp) {
    const long long t = base + lane;
    int j = 0, a = 0, b = 0, c = 0;
    double r2;
    const int band = t < total ? classify(g, box, bands, i, xi, t, j, a, b, c, r2) : -1;
    const unsigned mask0 = __ballot_sync(kFullMask, band == 0);
    const unsigned mask1 = __ballot_sync(kFullMask, band == 1);
    const unsigned mask2 = __ballot_sync(kFullMask, band == 2);
    const int4 record = make_int4(j, a, b, c);
    if (band == 0) edges[cursor0 + __popc(mask0 & below)] = record;
    if (band == 1) edges[cursor1 + __popc(mask1 & below)] = record;
    if (band == 2) edges[cursor2 + __popc(mask2 & below)] = record;
    cursor0 += __popc(mask0);
    cursor1 += __popc(mask1);
    cursor2 += __popc(mask2);
  }
}


__device__ __forceinline__ int floor_div(int a, int b) {
  const int q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

__global__ void bin_kernel(GeometryView g, CellGrid grid, int* __restrict__ key,
                           int* __restrict__ atom, int4* __restrict__ wrap) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= g.n) return;
  const double4_32a p = g.pos[i];
  int bin[3];
  int shift[3];
  for (int k = 0; k < 3; ++k) {
    const double s = p.x * grid.inverse[k] + p.y * grid.inverse[3 + k] + p.z * grid.inverse[6 + k];
    double f;
    if (grid.periodic[k]) {
      const double w = floor(s);
      shift[k] = static_cast<int>(w);
      f = (s - w) * grid.scale[k];
    } else {
      shift[k] = 0;
      f = (s - grid.origin[k]) * grid.scale[k];
    }
    const int b = static_cast<int>(floor(f));
    bin[k] = b < 0 ? 0 : (b >= grid.nb[k] ? grid.nb[k] - 1 : b);
  }
  key[i] = (bin[0] * grid.nb[1] + bin[1]) * grid.nb[2] + bin[2];
  atom[i] = i;
  wrap[i] = make_int4(shift[0], shift[1], shift[2], 0);
}

__global__ void bin_start_kernel(const int* __restrict__ sorted_key, int n, long long bins,
                                 int* __restrict__ bin_start) {
  const long long b = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b > bins) return;
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (sorted_key[mid] < b) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  bin_start[b] = lo;
}

// Walk the stencil of centre i and call visit(t, j, a, b, c) for every
// candidate site in a fixed, warp-uniform order (t indexes sorted_atom).
template <class Visit>
__device__ __forceinline__ void walk_stencil(const CellGrid& grid, int key_i, Visit&& visit) {
  const int bi2 = key_i % grid.nb[2];
  const int bi1 = (key_i / grid.nb[2]) % grid.nb[1];
  const int bi0 = key_i / (grid.nb[2] * grid.nb[1]);
  for (int d0 = -grid.reach[0]; d0 <= grid.reach[0]; ++d0) {
    const int c0 = bi0 + d0;
    if (!grid.periodic[0] && (c0 < 0 || c0 >= grid.nb[0])) continue;
    const int s0 = grid.periodic[0] ? floor_div(c0, grid.nb[0]) : 0;
    const int w0 = c0 - s0 * grid.nb[0];
    for (int d1 = -grid.reach[1]; d1 <= grid.reach[1]; ++d1) {
      const int c1 = bi1 + d1;
      if (!grid.periodic[1] && (c1 < 0 || c1 >= grid.nb[1])) continue;
      const int s1 = grid.periodic[1] ? floor_div(c1, grid.nb[1]) : 0;
      const int w1 = c1 - s1 * grid.nb[1];
      for (int d2 = -grid.reach[2]; d2 <= grid.reach[2]; ++d2) {
        const int c2 = bi2 + d2;
        if (!grid.periodic[2] && (c2 < 0 || c2 >= grid.nb[2])) continue;
        const int s2 = grid.periodic[2] ? floor_div(c2, grid.nb[2]) : 0;
        const int w2 = c2 - s2 * grid.nb[2];
        visit((w0 * grid.nb[1] + w1) * grid.nb[2] + w2, s0, s1, s2);
      }
    }
  }
}

__global__ void cell_count_kernel(GeometryView g, CellGrid grid, BandSpec bands,
                                  const int* __restrict__ key, const int* __restrict__ sorted_atom,
                                  const int4* __restrict__ wrap, const int* __restrict__ bin_start,
                                  int* __restrict__ counts, unsigned long long* __restrict__ scalars) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarp);
  const int lane = static_cast<int>(threadIdx.x % kWarp);
  if (warp >= g.n) return;
  const int i = warp;
  const math::Vec3 xi = load(g.pos, i);
  const int4 wi = wrap[i];
  int count0 = 0, count1 = 0, count2 = 0;
  double min_r2 = DBL_MAX;
  walk_stencil(grid, key[i], [&](int bin, int s0, int s1, int s2) {
    for (int t = bin_start[bin] + lane; t < bin_start[bin + 1]; t += kWarp) {
      const int j = sorted_atom[t];
      const int4 wj = wrap[j];
      const int a = s0 + wi.x - wj.x;
      const int b = s1 + wi.y - wj.y;
      const int c = s2 + wi.z - wj.z;
      if (j == i && a == 0 && b == 0 && c == 0) continue;
      const double r2 = math::norm2(math::site_displacement(xi, load(g.pos, j), a, b, c, g.cell));
      min_r2 = fmin(min_r2, r2);
      const int band = band_of(r2, bands);
      count0 += band == 0;
      count1 += band == 1;
      count2 += band == 2;
    }
  });
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    count0 += __shfl_down_sync(kFullMask, count0, offset);
    count1 += __shfl_down_sync(kFullMask, count1, offset);
    count2 += __shfl_down_sync(kFullMask, count2, offset);
    min_r2 = fmin(min_r2, __shfl_down_sync(kFullMask, min_r2, offset));
  }
  if (lane == 0) {
    counts[3 * i] = count0;
    counts[3 * i + 1] = count1;
    counts[3 * i + 2] = count2;
    atomicMin(scalars + 3, static_cast<unsigned long long>(__double_as_longlong(min_r2)));
  }
}

__global__ void cell_fill_kernel(GeometryView g, CellGrid grid, BandSpec bands,
                                 const int* __restrict__ key, const int* __restrict__ sorted_atom,
                                 const int4* __restrict__ wrap, const int* __restrict__ bin_start,
                                 const long long* __restrict__ row_ptr,
                                 const int* __restrict__ counts, int4* __restrict__ edges) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarp);
  const int lane = static_cast<int>(threadIdx.x % kWarp);
  if (warp >= g.n) return;
  const int i = warp;
  const math::Vec3 xi = load(g.pos, i);
  const int4 wi = wrap[i];
  long long cursor0 = row_ptr[i];
  long long cursor1 = cursor0 + counts[3 * i];
  long long cursor2 = cursor1 + counts[3 * i + 1];
  const unsigned below = (1u << lane) - 1u;
  walk_stencil(grid, key[i], [&](int bin, int s0, int s1, int s2) {
    const int start = bin_start[bin];
    const int stop = bin_start[bin + 1];
    for (int base = start; base < stop; base += kWarp) {
      const int t = base + lane;
      int band = -1;
      int4 record = make_int4(0, 0, 0, 0);
      if (t < stop) {
        const int j = sorted_atom[t];
        const int4 wj = wrap[j];
        record = make_int4(j, s0 + wi.x - wj.x, s1 + wi.y - wj.y, s2 + wi.z - wj.z);
        if (!(j == i && record.y == 0 && record.z == 0 && record.w == 0)) {
          const double r2 = math::norm2(
              math::site_displacement(xi, load(g.pos, j), record.y, record.z, record.w, g.cell));
          band = band_of(r2, bands);
        }
      }
      const unsigned mask0 = __ballot_sync(kFullMask, band == 0);
      const unsigned mask1 = __ballot_sync(kFullMask, band == 1);
      const unsigned mask2 = __ballot_sync(kFullMask, band == 2);
      if (band == 0) edges[cursor0 + __popc(mask0 & below)] = record;
      if (band == 1) edges[cursor1 + __popc(mask1 & below)] = record;
      if (band == 2) edges[cursor2 + __popc(mask2 & below)] = record;
      cursor0 += __popc(mask0);
      cursor1 += __popc(mask1);
      cursor2 += __popc(mask2);
    }
  });
}

int warp_blocks(int n) { return (n + kWarpsPerBlock - 1) / kWarpsPerBlock; }

}  // namespace

BandSpec make_bands(const std::array<double, kViewCount>& radius) {
  BandSpec bands;
  std::vector<double> distinct;
  for (double r : radius) {
    if (r > 0.0) distinct.push_back(r * r);
  }
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  bands.n_bands = static_cast<int>(distinct.size());
  for (int b = 0; b < bands.n_bands; ++b) bands.radius2[b] = distinct[static_cast<std::size_t>(b)];
  for (int v = 0; v < kViewCount; ++v) {
    const double r = radius[static_cast<std::size_t>(v)];
    if (r <= 0.0) continue;
    bands.view_band[v] = static_cast<int>(
        std::lower_bound(distinct.begin(), distinct.end(), r * r) - distinct.begin());
  }
  return bands;
}

NeighborGraph::NeighborGraph(cudaMemPool_t pool, cudaStream_t stream, AllocationCounters* counters)
    : stream_(stream),
      counts_(pool, stream, counters),
      totals_(pool, stream, counters),
      row_ptr_(pool, stream, counters),
      items_(pool, stream, counters),
      item_prefix_(pool, stream, counters),
      edges_(pool, stream, counters),
      len_{DeviceBuffer<int>(pool, stream, counters), DeviceBuffer<int>(pool, stream, counters),
           DeviceBuffer<int>(pool, stream, counters)},
      scalars_(pool, stream, counters),
      scan_temp_(pool, stream, counters),
      bin_key_(pool, stream, counters),
      sorted_key_(pool, stream, counters),
      atom_index_(pool, stream, counters),
      sorted_atom_(pool, stream, counters),
      wrap_(pool, stream, counters),
      bin_start_(pool, stream, counters),
      sort_temp_(pool, stream, counters) {
  scalars_.resize(4);
  host_scalars_.reserve(6);
}

void NeighborGraph::begin_build(const GeometryView& g, const BandSpec& bands) {
  bands_ = bands;
  const auto n = static_cast<std::size_t>(g.n);
  counts_.resize(3 * n);
  totals_.resize(n + 1);
  row_ptr_.resize(n + 1);
  items_.resize(n + 1);
  item_prefix_.resize(n + 1);
  for (auto& len : len_) len.resize(n);
  long long max_bits;
  const double max_r2 = DBL_MAX;
  std::memcpy(&max_bits, &max_r2, sizeof(max_bits));
  const unsigned long long init[4] = {0, 0, 0, static_cast<unsigned long long>(max_bits)};
  std::memcpy(host_scalars_.data(), init, sizeof(init));
  scalars_.upload_async(host_scalars_.data(), 4);
}

double NeighborGraph::finish_build(const GeometryView& g, const BandSpec& bands) {
  const auto n = static_cast<std::size_t>(g.n);
  totals_kernel<<<(g.n + 1 + 255) / 256, 256, 0, stream_>>>(
      counts_.data(), g.n, bands, totals_.data(), len_[0].data(), len_[1].data(), len_[2].data(),
      items_.data(), scalars_.data());
  VF_LAUNCH_CHECK(stream_);
  std::size_t temp_bytes = 0;
  VF_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, totals_.data(), row_ptr_.data(),
                                              static_cast<long long>(n + 1), stream_));
  scan_temp_.resize(temp_bytes);
  VF_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_temp_.data(), temp_bytes, totals_.data(),
                                              row_ptr_.data(), static_cast<long long>(n + 1), stream_));
  VF_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_temp_.data(), temp_bytes, items_.data(),
                                              item_prefix_.data(), static_cast<long long>(n + 1),
                                              stream_));
  scalars_.download_async(host_scalars_.data(), 4);
  copy_to_host_async(reinterpret_cast<const unsigned long long*>(row_ptr_.data() + n),
                     host_scalars_.data() + 4, 1, stream_);
  copy_to_host_async(reinterpret_cast<const unsigned long long*>(item_prefix_.data() + n),
                     host_scalars_.data() + 5, 1, stream_);
  VF_CUDA_CHECK(cudaStreamSynchronize(stream_));

  const unsigned long long* scalars = host_scalars_.data();
  total_edges_ = static_cast<std::int64_t>(scalars[4]);
  atm_items_total_ = static_cast<std::int64_t>(scalars[5]);
  for (int v = 0; v < kViewCount; ++v) {
    view_edges_[v] = 0;
    for (int band = 0; band <= bands.view_band[v]; ++band) {
      view_edges_[v] += static_cast<std::int64_t>(scalars[band]);
    }
  }
  double min_r2;
  const long long bits = static_cast<long long>(scalars[3]);
  std::memcpy(&min_r2, &bits, sizeof(min_r2));
  edges_.resize(static_cast<std::size_t>(total_edges_));
  return min_r2;
}

double NeighborGraph::build_brute_force(const GeometryView& g, const ImageBox& box,
                                        const BandSpec& bands) {
  begin_build(g, bands);
  const int threads = kWarp * kWarpsPerBlock;
  count_kernel<<<warp_blocks(g.n), threads, 0, stream_>>>(g, box, bands, counts_.data(), scalars_.data());
  VF_LAUNCH_CHECK(stream_);
  const double min_r2 = finish_build(g, bands);
  if (total_edges_ > 0) {
    fill_kernel<<<warp_blocks(g.n), threads, 0, stream_>>>(g, box, bands, row_ptr_.data(),
                                                           counts_.data(), edges_.data());
    VF_LAUNCH_CHECK(stream_);
  }
  return min_r2;
}

double NeighborGraph::build_cell_bins(const GeometryView& g, const CellGrid& grid,
                                      const BandSpec& bands) {
  begin_build(g, bands);
  const auto n = static_cast<std::size_t>(g.n);
  const long long bins = grid.bins();
  bin_key_.resize(n);
  sorted_key_.resize(n);
  atom_index_.resize(n);
  sorted_atom_.resize(n);
  wrap_.resize(n);
  bin_start_.resize(static_cast<std::size_t>(bins) + 1);
  bin_kernel<<<(g.n + 255) / 256, 256, 0, stream_>>>(g, grid, bin_key_.data(), atom_index_.data(),
                                                    wrap_.data());
  VF_LAUNCH_CHECK(stream_);
  int end_bit = 1;
  while ((1LL << end_bit) <= bins && end_bit < 31) ++end_bit;
  std::size_t temp_bytes = 0;
  VF_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, bin_key_.data(), sorted_key_.data(),
                                                atom_index_.data(), sorted_atom_.data(), g.n, 0, end_bit,
                                                stream_));
  sort_temp_.resize(temp_bytes);
  VF_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp_.data(), temp_bytes, bin_key_.data(),
                                                sorted_key_.data(), atom_index_.data(),
                                                sorted_atom_.data(), g.n, 0, end_bit, stream_));
  bin_start_kernel<<<static_cast<unsigned>((bins + 1 + 255) / 256), 256, 0, stream_>>>(
      sorted_key_.data(), g.n, bins, bin_start_.data());
  VF_LAUNCH_CHECK(stream_);

  const int threads = kWarp * kWarpsPerBlock;
  cell_count_kernel<<<warp_blocks(g.n), threads, 0, stream_>>>(
      g, grid, bands, bin_key_.data(), sorted_atom_.data(), wrap_.data(), bin_start_.data(),
      counts_.data(), scalars_.data());
  VF_LAUNCH_CHECK(stream_);
  const double min_r2 = finish_build(g, bands);
  if (total_edges_ > 0) {
    cell_fill_kernel<<<warp_blocks(g.n), threads, 0, stream_>>>(
        g, grid, bands, bin_key_.data(), sorted_atom_.data(), wrap_.data(), bin_start_.data(),
        row_ptr_.data(), counts_.data(), edges_.data());
    VF_LAUNCH_CHECK(stream_);
  }
  return min_r2;
}

GraphView NeighborGraph::view() const noexcept {
  return {row_ptr_.data(), edges_.data(), len_[kViewCn].data(), len_[kViewPair].data(),
          len_[kViewAtm].data()};
}

void NeighborGraph::download(std::vector<long long>& row_ptr, std::vector<int4>& edges) const {
  row_ptr.resize(row_ptr_.size());
  edges.resize(edges_.size());
  row_ptr_.download_async(row_ptr.data(), row_ptr.size());
  edges_.download_async(edges.data(), edges.size());
  VF_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace vanderfast::cuda
