// Directed, image-aware neighbour graph in banded CSR form.
#pragma once

#include <array>
#include <cstdint>

#include "resources.cuh"
#include "views.cuh"

namespace vanderfast::cuda {

enum View : int { kViewCn = 0, kViewPair = 1, kViewAtm = 2, kViewCount = 3 };

// Nested radius bands. Each row stores band 0 first, then band 1, ...; a
// view is the prefix of bands up to and including ``view_band[v]``.
struct BandSpec {
  int n_bands = 0;
  double radius2[3] = {0.0, 0.0, 0.0};
  int view_band[kViewCount] = {-1, -1, -1};  // -1: view inactive
};

struct ImageBox {
  int range[3] = {0, 0, 0};
  long long images = 1;  // product of (2 range + 1)
};

// Build bands from per-view radii (bohr); non-positive radii disable a view.
BandSpec make_bands(const std::array<double, kViewCount>& radius);

class NeighborGraph {
 public:
  NeighborGraph() = default;
  NeighborGraph(cudaMemPool_t pool, cudaStream_t stream, AllocationCounters* counters);

  // Brute-force build over all atoms and images. Synchronises once to size
  // the edge array. Returns the minimum squared distance between distinct sites.
  double build_brute_force(const GeometryView& geometry, const ImageBox& box, const BandSpec& bands);

  GraphView view() const noexcept;
  std::int64_t edges(View view) const noexcept { return view_edges_[view]; }
  std::int64_t total_edges() const noexcept { return total_edges_; }
  std::int64_t capacity() const noexcept { return static_cast<std::int64_t>(edges_.capacity()); }
  const BandSpec& bands() const noexcept { return bands_; }

  // Copy the full graph to the host (debugging and tests).
  void download(std::vector<long long>& row_ptr, std::vector<int4>& edges) const;

 private:
  cudaStream_t stream_ = nullptr;
  BandSpec bands_{};
  std::int64_t total_edges_ = 0;
  std::int64_t view_edges_[kViewCount] = {0, 0, 0};
  DeviceBuffer<int> counts_;
  DeviceBuffer<long long> totals_;
  DeviceBuffer<long long> row_ptr_;
  DeviceBuffer<int4> edges_;
  DeviceBuffer<int> len_[kViewCount];
  DeviceBuffer<unsigned long long> scalars_;  // band totals[3], min r2 bits
  DeviceBuffer<unsigned char> scan_temp_;
  PinnedBuffer<unsigned long long> host_scalars_;
};

}  // namespace vanderfast::cuda
