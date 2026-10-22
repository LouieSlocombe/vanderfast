#include "d4_model.cuh"

#include <cfloat>
#include <cmath>

namespace vanderfast::cuda {

namespace {

// mctc-lib's D4 electronegativity factor of the coordination number.
constexpr double kK4 = 4.10451;
constexpr double kK5 = 19.08857;
const double kK6 = 2 * std::pow(11.28174, 2.0);

std::vector<PairConst> d4_pair_constants(const D4SpeciesTables& t, const D4Parameters& parameters) {
  const std::size_t n = t.z.size();
  std::vector<PairConst> pair(n * n);
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t u = 0; u <= s; ++u) {
      PairConst value{};
      value.rc_cn = t.rcov[s] + t.rcov[u];
      value.kappa = 3 * t.r4r2[s] * t.r4r2[u];
      const double r0 = parameters.a1 * std::sqrt(value.kappa) + parameters.a2;
      const double r0_2 = r0 * r0;
      value.r0_6 = r0_2 * r0_2 * r0_2;
      value.r0_8 = value.r0_6 * r0_2;
      value.r0_atm = r0;
      const double x = std::fabs(t.en[s] - t.en[u]) + kK5;
      value.aux = kK4 * std::exp(-(x * x) / kK6);
      pair[s * n + u] = value;
      pair[u * n + s] = value;
    }
  }
  return pair;
}

}  // namespace

D4DeviceModel::D4DeviceModel(const D4ReferenceData& reference, const std::vector<int>& species_z,
                             const D4Parameters& parameters, cudaMemPool_t pool, cudaStream_t stream,
                             AllocationCounters* counters)
    : species_z_(species_z),
      nref_(pool, stream, counters),
      cnref_(pool, stream, counters),
      maxmask_(pool, stream, counters),
      c6ref_(pool, stream, counters),
      pair_(pool, stream, counters),
      ngw_(pool, stream, counters),
      qref_(pool, stream, counters),
      per_species_(pool, stream, counters) {
  update_parameters(reference, parameters);
}

void D4DeviceModel::update_parameters(const D4ReferenceData& reference, const D4Parameters& parameters) {
  host_ = make_d4_species(reference, species_z_, parameters.ga, parameters.gc);
  pair_host_ = d4_pair_constants(host_, parameters);
  const std::size_t n = species_z_.size();
  std::vector<double> per_species;
  per_species.reserve(6 * n);
  for (const std::vector<double>* column :
       {&host_.zeff, &host_.gi, &host_.chi, &host_.eta, &host_.kcnchi, &host_.rad}) {
    per_species.insert(per_species.end(), column->begin(), column->end());
  }
  nref_.upload_async(host_.nref.data(), host_.nref.size());
  cnref_.upload_async(host_.cnref.data(), host_.cnref.size());
  maxmask_.upload_async(host_.maxmask.data(), host_.maxmask.size());
  c6ref_.upload_async(host_.c6ref.data(), host_.c6ref.size());
  pair_.upload_async(pair_host_.data(), pair_host_.size());
  ngw_.upload_async(host_.ngw.data(), host_.ngw.size());
  qref_.upload_async(host_.qref.data(), host_.qref.size());
  per_species_.upload_async(per_species.data(), per_species.size());
}

ModelView D4DeviceModel::view(const D4Parameters& parameters) const {
  ModelView view{};
  view.n_species = static_cast<int>(species_z_.size());
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
  // dftd4 has no small-C6 rule; only exactly vanishing C6 (whose CN
  // derivative upstream would divide by zero) is skipped.
  view.atm_c6_floor = DBL_MIN;
  return view;
}

D4View D4DeviceModel::d4_view(const D4Parameters& parameters) const {
  const std::size_t n = species_z_.size();
  const double* base = per_species_.data();
  D4View view{};
  view.n_species = static_cast<int>(n);
  view.ngw = ngw_.data();
  view.cnref = cnref_.data();
  view.qref = qref_.data();
  view.maxmask = maxmask_.data();
  view.zeff = base;
  view.gi = base + n;
  view.chi = base + 2 * n;
  view.eta = base + 3 * n;
  view.kcnchi = base + 4 * n;
  view.rad = base + 5 * n;
  view.ga = parameters.ga;
  view.wf = parameters.wf;
  view.cn_cutoff2 = parameters.cutoffs.cn * parameters.cutoffs.cn;
  view.eeq_cutoff2 = D4Parameters::kEeqCnCutoff * D4Parameters::kEeqCnCutoff;
  return view;
}

}  // namespace vanderfast::cuda
