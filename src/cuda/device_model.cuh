// Device-resident reference tables restricted to the species of a context.
#pragma once

#include <memory>
#include <vector>

#include "resources.cuh"
#include "vanderfast/parameters.hpp"
#include "vanderfast/reference_data.hpp"
#include "views.cuh"

namespace vanderfast::cuda {

class DeviceModel {
 public:
  // Host-side copies (used by the CPU reference enumerator and debugging).
  struct Host {
    std::vector<int> species_z;     // atomic number per species
    std::vector<int> nref;          // [S]
    std::vector<double> cnref;      // [S * kRefPad]
    std::vector<unsigned> maxmask;  // [S]
    std::vector<double> c6ref;      // [S * S * kC6Block]
    std::vector<PairConst> pair;    // [S * S]
    int max_nref = 1;
  };

  DeviceModel() = default;
  DeviceModel(const ReferenceData& reference, const std::vector<int>& species_z,
              const D3Parameters& parameters, cudaMemPool_t pool, cudaStream_t stream,
              AllocationCounters* counters);

  // Recompute parameter-dependent constants (BJ radii) and upload them.
  void update_parameters(const ReferenceData& reference, const D3Parameters& parameters);

  ModelView view(const D3Parameters& parameters) const;
  const Host& host() const noexcept { return host_; }

 private:
  Host host_;
  DeviceBuffer<int> nref_;
  DeviceBuffer<double> cnref_;
  DeviceBuffer<unsigned> maxmask_;
  DeviceBuffer<double> c6ref_;
  DeviceBuffer<PairConst> pair_;
};

}  // namespace vanderfast::cuda
