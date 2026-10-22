// Device-resident D4 tables restricted to the species of a context.
#pragma once

#include <vector>

#include "../core/d4_model.hpp"
#include "d4_kernels.cuh"
#include "resources.cuh"
#include "vanderfast/d4_parameters.hpp"
#include "vanderfast/d4_reference_data.hpp"
#include "views.cuh"

namespace vanderfast::cuda {

class D4DeviceModel {
 public:
  D4DeviceModel() = default;
  D4DeviceModel(const D4ReferenceData& reference, const std::vector<int>& species_z,
                const D4Parameters& parameters, cudaMemPool_t pool, cudaStream_t stream,
                AllocationCounters* counters);

  // Rebuild the parameter-dependent tables (reference C6 for ga/gc, BJ radii).
  void update_parameters(const D4ReferenceData& reference, const D4Parameters& parameters);

  // The view consumed by the kernels shared with D3 (contraction, pair, ATM).
  ModelView view(const D4Parameters& parameters) const;
  D4View d4_view(const D4Parameters& parameters) const;
  const D4SpeciesTables& host() const noexcept { return host_; }

 private:
  std::vector<int> species_z_;
  D4SpeciesTables host_;
  std::vector<PairConst> pair_host_;
  DeviceBuffer<int> nref_;
  DeviceBuffer<double> cnref_;
  DeviceBuffer<unsigned> maxmask_;
  DeviceBuffer<double> c6ref_;
  DeviceBuffer<PairConst> pair_;
  DeviceBuffer<int> ngw_;
  DeviceBuffer<double> qref_;
  DeviceBuffer<double> per_species_;  // zeff, gi, chi, eta, kcnchi, rad: [6 * S]
};

}  // namespace vanderfast::cuda
