// Host precomputation of the species tables of a D4 context.
#pragma once

#include <vector>

#include "vanderfast/d4_reference_data.hpp"

namespace vanderfast {

// Upstream dftd4 charge scaling: exp(a) for negative effective charge.
double d4_zeta(double a, double c, double qref, double qmod) noexcept;

// set_refalpha_eeq for every reference of ``z``: nref(z) x 23 values.
std::vector<double> d4_reference_alpha(const D4ReferenceData& reference, int z, double ga, double gc);

// Casimir-Polder C6 of every reference pair of two alpha sets (rows ``a``):
// (3/pi) sum_k (a[k] b[k]) w[k], summed sequentially as dftd4's trapzd.
std::vector<double> d4_reference_c6(const D4ReferenceData& reference, const std::vector<double>& a,
                                    int nref_a, const std::vector<double>& b, int nref_b);

// Species-restricted tables; reference arrays are padded to kPad entries.
struct D4SpeciesTables {
  static constexpr int kPad = 8;

  std::vector<int> z;            // [S] atomic numbers in ascending order
  std::vector<int> nref;         // [S]
  int max_nref = 1;
  std::vector<int> ngw;          // [S * kPad]
  std::vector<double> cnref;     // [S * kPad] weighting CNs (refcovcn)
  std::vector<double> qref;      // [S * kPad] reference charge + zeff
  std::vector<unsigned> maxmask; // [S] references whose CN is within 1e-12 of the maximum
  std::vector<double> alpha0;    // [S * kPad] static reference polarizabilities
  std::vector<double> c6ref;     // [S * S * kPad * kPad], block (s, t) rows for s
  // Per species.
  std::vector<double> zeff, gi, rcov, en, r4r2;
  std::vector<double> chi, eta, kcnchi, rad;
};

// Build the tables for ``species_z`` (ascending atomic numbers). Fails if the
// D4 and EEQ covalent radii differ (the fused CN kernels rely on equality).
D4SpeciesTables make_d4_species(const D4ReferenceData& reference, const std::vector<int>& species_z,
                                double ga, double gc);

}  // namespace vanderfast
