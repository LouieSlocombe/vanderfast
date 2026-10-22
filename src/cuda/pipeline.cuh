// Context plumbing shared by the D3 and D4 pipelines: device selection, stage
// timings, and the neighbour graph with its skin-based reuse.
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include "../core/lattice.hpp"
#include "neighbor.cuh"
#include "resources.cuh"
#include "vanderfast/context.hpp"

namespace vanderfast::cuda {

// Validate a device index against the driver; throws CudaUnavailable or
// InvalidArgument.
int checked_device(int device);

// Optional CUDA-event timings of the pipeline stages (stream time between marks).
class StageTimer {
 public:
  StageTimer(bool enabled, cudaStream_t stream, std::vector<Event>& pool)
      : enabled_(enabled), stream_(stream), pool_(pool) {}
  void mark(const char* name) {
    if (!enabled_) return;
    if (pool_.size() <= names_.size()) pool_.push_back(Event::create());
    pool_[names_.size()].record(stream_);
    names_.push_back(name);
  }
  void collect(std::map<std::string, double>& out) const {
    for (std::size_t k = 1; k < names_.size(); ++k) {
      float ms = 0.0f;
      VF_CUDA_CHECK(cudaEventElapsedTime(&ms, pool_[k - 1].get(), pool_[k].get()));
      out[names_[k]] = static_cast<double>(ms);
    }
  }

 private:
  bool enabled_;
  cudaStream_t stream_;
  std::vector<Event>& pool_;
  std::vector<const char*> names_;
};

// The neighbour graph of one context and the state that decides when it can
// be reused: it is rebuilt when the cell, periodicity, or radii change, the
// skin is zero, or any atom moved by half the skin since the last build.
class NeighborCache {
 public:
  NeighborCache() = default;
  NeighborCache(cudaMemPool_t pool, cudaStream_t stream, AllocationCounters* counters)
      : graph(pool, stream, counters) {}

  // Rebuild the graph if needed and record the builder in ``info``.
  void ensure(const Geometry& geometry, const Lattice& lattice, const GeometryView& g,
              const std::array<double, kViewCount>& radius, const ContextOptions& options,
              bool canonical_atm, EvalInfo& info, ContextStats& stats);

  void invalidate() noexcept { valid = false; }

  NeighborGraph graph;
  bool valid = false;

 private:
  bool needs_rebuild(const Geometry& geometry, const Lattice& lattice,
                     const std::array<double, kViewCount>& radius, double skin) const;

  std::vector<double> positions_;
  std::array<double, 9> cell_{};
  std::array<bool, 3> pbc_{};
  std::array<double, kViewCount> radius_{};
  const char* builder_ = "none";
};

}  // namespace vanderfast::cuda
