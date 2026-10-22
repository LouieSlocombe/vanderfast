#include "d4_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "vanderfast/error.hpp"

namespace vanderfast {

namespace {

constexpr double kPi = 3.141592653589793;  // mctc_io_constants pi
constexpr int kFreq = D4ReferenceData::kFreq;
using R = D4ReferenceData;

}  // namespace

double d4_zeta(double a, double c, double qref, double qmod) noexcept {
  if (qmod < 0.0) return std::exp(a);
  return std::exp(a * (1.0 - std::exp(c * (1.0 - qref / qmod))));
}

std::vector<double> d4_reference_alpha(const D4ReferenceData& reference, int z, double ga, double gc) {
  const int nref = reference.nref(z);
  const std::size_t first = reference.first_ref(z);
  std::vector<double> alpha(static_cast<std::size_t>(nref) * kFreq, 0.0);
  for (int r = 0; r < nref; ++r) {
    const std::size_t ref = first + static_cast<std::size_t>(r);
    const int system = reference.refsys(ref);
    if (system == 0) continue;  // uninitialised reference: zero polarizability
    // The secondary-system index doubles as an atomic number upstream.
    const double iz = reference.element(system, R::kZeff);
    const double scale = d4_zeta(ga, reference.element(system, R::kHardness) * gc, iz,
                                 reference.ref_value(ref, R::kClsh) + iz);
    const double sscale = reference.sscale(system);
    const double* secaiw = reference.secaiw(system);
    const double* alphaiw = reference.alphaiw(ref);
    const double ascale = reference.ref_value(ref, R::kAscale);
    const double hcount = reference.ref_value(ref, R::kHcount);
    for (int k = 0; k < kFreq; ++k) {
      const double aiw = (sscale * secaiw[k]) * scale;
      alpha[static_cast<std::size_t>(r) * kFreq + static_cast<std::size_t>(k)] =
          std::max(ascale * (alphaiw[k] - hcount * aiw), 0.0);
    }
  }
  return alpha;
}

std::vector<double> d4_reference_c6(const D4ReferenceData& reference, const std::vector<double>& a,
                                    int nref_a, const std::vector<double>& b, int nref_b) {
  const double* weights = reference.weights();
  const double thopi = 3.0 / kPi;
  std::vector<double> c6(static_cast<std::size_t>(nref_a) * static_cast<std::size_t>(nref_b));
  for (int i = 0; i < nref_a; ++i) {
    for (int j = 0; j < nref_b; ++j) {
      double total = 0.0;
      for (int k = 0; k < kFreq; ++k) {
        total += (a[static_cast<std::size_t>(i * kFreq + k)] * b[static_cast<std::size_t>(j * kFreq + k)]) *
                 weights[k];
      }
      c6[static_cast<std::size_t>(i * nref_b + j)] = thopi * total;
    }
  }
  return c6;
}

D4SpeciesTables make_d4_species(const D4ReferenceData& reference, const std::vector<int>& species_z,
                                double ga, double gc) {
  constexpr int P = D4SpeciesTables::kPad;
  const std::size_t n = species_z.size();
  D4SpeciesTables t;
  t.z = species_z;
  t.nref.resize(n);
  t.ngw.assign(n * P, 0);
  t.cnref.assign(n * P, 0.0);
  t.qref.assign(n * P, 0.0);
  t.maxmask.assign(n, 0u);
  t.alpha0.assign(n * P, 0.0);
  t.c6ref.assign(n * n * P * P, 0.0);
  std::vector<std::vector<double>> alpha(n);
  for (std::size_t s = 0; s < n; ++s) {
    const int z = species_z[s];
    const int nref = reference.nref(z);
    const std::size_t first = reference.first_ref(z);
    t.nref[s] = nref;
    t.max_nref = std::max(t.max_nref, nref);
    const double zeff = reference.element(z, R::kZeff);
    t.zeff.push_back(zeff);
    t.gi.push_back(reference.element(z, R::kHardness) * gc);
    t.rcov.push_back(reference.element(z, R::kRcovD4));
    t.en.push_back(reference.element(z, R::kEn));
    t.r4r2.push_back(reference.element(z, R::kR4r2));
    t.chi.push_back(reference.element(z, R::kEeqChi));
    t.eta.push_back(reference.element(z, R::kEeqEta));
    t.kcnchi.push_back(reference.element(z, R::kEeqKcnchi));
    t.rad.push_back(reference.element(z, R::kEeqRad));
    if (reference.element(z, R::kEeqRcov) != reference.element(z, R::kRcovD4)) {
      fail(ErrorCode::InvalidArgument,
           "D4 reference data: the D4 and EEQ covalent radii of Z=" + std::to_string(z) + " differ");
    }
    alpha[s] = d4_reference_alpha(reference, z, ga, gc);
    double maxcn = reference.ref_value(first, R::kRefCovCn);
    for (int r = 0; r < nref; ++r) {
      const std::size_t ref = first + static_cast<std::size_t>(r);
      const std::size_t slot = s * P + static_cast<std::size_t>(r);
      t.ngw[slot] = reference.ngw(ref);
      t.cnref[slot] = reference.ref_value(ref, R::kRefCovCn);
      t.qref[slot] = reference.ref_value(ref, R::kClsq) + zeff;
      t.alpha0[slot] = alpha[s][static_cast<std::size_t>(r) * kFreq];
      maxcn = std::max(maxcn, t.cnref[slot]);
    }
    for (int r = 0; r < nref; ++r) {
      if (std::fabs(maxcn - t.cnref[s * P + static_cast<std::size_t>(r)]) < 1e-12) t.maxmask[s] |= 1u << r;
    }
  }
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t u = 0; u <= s; ++u) {
      const std::vector<double> block = d4_reference_c6(reference, alpha[s], t.nref[s], alpha[u], t.nref[u]);
      for (int a = 0; a < t.nref[s]; ++a) {
        for (int b = 0; b < t.nref[u]; ++b) {
          const double value = block[static_cast<std::size_t>(a * t.nref[u] + b)];
          t.c6ref[(s * n + u) * P * P + static_cast<std::size_t>(a * P + b)] = value;
          t.c6ref[(u * n + s) * P * P + static_cast<std::size_t>(b * P + a)] = value;
        }
      }
    }
  }
  return t;
}

}  // namespace vanderfast
