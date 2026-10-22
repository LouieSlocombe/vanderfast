#include <cfloat>
#include <cmath>

#include "device_model.cuh"

namespace vanderfast::cuda {

namespace {

std::vector<PairConst> pair_constants(const ReferenceData& reference,
                                      const std::vector<int>& species_z,
                                      const D3Parameters& parameters) {
  const std::size_t n = species_z.size();
  std::vector<PairConst> pair(n * n);
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t t = 0; t <= s; ++t) {
      const int zi = species_z[s];
      const int zj = species_z[t];
      PairConst value{};
      value.rc_cn = reference.rcov(zi) + reference.rcov(zj);
      value.kappa = 3 * reference.r4r2(zi) * reference.r4r2(zj);
      const double r0 = parameters.a1 * std::sqrt(value.kappa) + parameters.a2;
      const double r0_2 = r0 * r0;
      value.r0_6 = r0_2 * r0_2 * r0_2;
      value.r0_8 = value.r0_6 * r0_2;
      value.r0_atm = D3Parameters::kRs9 * reference.rvdw(zi, zj);
      pair[s * n + t] = value;
      pair[t * n + s] = value;
    }
  }
  return pair;
}

}  // namespace

DeviceModel::DeviceModel(const ReferenceData& reference, const std::vector<int>& species_z,
                         const D3Parameters& parameters, cudaMemPool_t pool, cudaStream_t stream,
                         AllocationCounters* counters)
    : nref_(pool, stream, counters),
      cnref_(pool, stream, counters),
      maxmask_(pool, stream, counters),
      c6ref_(pool, stream, counters),
      pair_(pool, stream, counters) {
  const std::size_t n = species_z.size();
  host_.species_z = species_z;
  host_.nref.resize(n);
  host_.cnref.assign(n * kRefPad, 0.0);
  host_.maxmask.assign(n, 0u);
  host_.c6ref.assign(n * n * kC6Block, 0.0);
  host_.max_nref = 1;
  for (std::size_t s = 0; s < n; ++s) {
    const int z = species_z[s];
    const int nref = reference.nref(z);
    host_.nref[s] = nref;
    host_.max_nref = std::max(host_.max_nref, nref);
    double maximum = reference.reference_cn(z, 0);
    for (int a = 0; a < nref; ++a) {
      host_.cnref[s * kRefPad + static_cast<std::size_t>(a)] = reference.reference_cn(z, a);
      maximum = std::max(maximum, reference.reference_cn(z, a));
    }
    for (int a = 0; a < nref; ++a) {
      if (reference.reference_cn(z, a) == maximum) host_.maxmask[s] |= 1u << a;
    }
  }
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t t = 0; t < n; ++t) {
      double* block = host_.c6ref.data() + (s * n + t) * kC6Block;
      for (int a = 0; a < host_.nref[s]; ++a) {
        for (int b = 0; b < host_.nref[t]; ++b) {
          block[a * kRefPad + b] = reference.c6(species_z[s], species_z[t], a, b);
        }
      }
    }
  }
  nref_.upload_async(host_.nref.data(), host_.nref.size());
  cnref_.upload_async(host_.cnref.data(), host_.cnref.size());
  maxmask_.upload_async(host_.maxmask.data(), host_.maxmask.size());
  c6ref_.upload_async(host_.c6ref.data(), host_.c6ref.size());
  update_parameters(reference, parameters);
}

void DeviceModel::update_parameters(const ReferenceData& reference, const D3Parameters& parameters) {
  host_.pair = pair_constants(reference, host_.species_z, parameters);
  pair_.upload_async(host_.pair.data(), host_.pair.size());
}

ModelView DeviceModel::view(const D3Parameters& parameters) const {
  ModelView view{};
  view.n_species = static_cast<int>(host_.species_z.size());
  view.max_nref = host_.max_nref;
  view.nref = nref_.data();
  view.cnref = cnref_.data();
  view.maxmask = maxmask_.data();
  view.c6ref = c6ref_.data();
  view.pair = pair_.data();
  view.s6 = parameters.s6;
  view.s8 = parameters.s8;
  view.s9 = parameters.s9;
  view.alp_eff = parameters.atm_exponent();
  view.alp3 = parameters.atm_exponent() / 3.0;
  view.cn_cutoff2 = parameters.cutoffs.cn * parameters.cutoffs.cn;
  view.disp2_cutoff2 = parameters.cutoffs.disp2 * parameters.cutoffs.disp2;
  view.disp3_cutoff2 = parameters.cutoffs.disp3 * parameters.cutoffs.disp3;
  view.atm_c6_floor = DBL_EPSILON;  // upstream skips triangles with any |C6| < eps
  return view;
}

}  // namespace vanderfast::cuda
