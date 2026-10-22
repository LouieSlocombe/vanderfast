// Host-side lattice setup for the periodic EEQ Coulomb matrix.
//
// ``Upstream`` replicates multicharge v0.5.0: Wigner-Seitz image averaging
// over 27 candidate translations, a fixed 125-translation real-space and
// 124-vector reciprocal sum, and the Ewald parameter from its bisection.
// ``Converged`` keeps every real-space image and reciprocal vector above
// 1e-20, so the result does not depend on the Ewald parameter or the cell.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "vanderfast/d4_parameters.hpp"

namespace vanderfast {

struct EwaldSetup {
  EwaldMode mode = EwaldMode::Upstream;
  double alpha = 0.0;
  double volume = 0.0;
  double real_cutoff = 0.0;              // converged mode only
  std::vector<double> dtrans;            // [T * 3] real-space translations
  std::vector<double> rtrans;            // [G * 3] reciprocal vectors (no origin)
  std::vector<double> ws_candidates;     // [27 * 3] upstream Wigner-Seitz candidates
  std::vector<double> self_images;       // [I * 3] upstream self images (with the quirk)
  std::size_t n_trans() const noexcept { return dtrans.size() / 3; }
  std::size_t n_recip() const noexcept { return rtrans.size() / 3; }
};

// mctc matdet_3x3 / matinv_3x3 on a Fortran-ordered matrix ``a`` (a[i + 3 j] = a(i+1, j+1)).
double mctc_det(const std::array<double, 9>& a) noexcept;
std::array<double, 9> mctc_inverse(const std::array<double, 9>& a) noexcept;

// mctc wrap_to_central_cell on row-major positions (bohr) for cell rows ``cell``.
void wrap_to_central_cell(std::vector<double>& positions, const std::array<double, 9>& cell);

// multicharge get_alpha for cell rows ``cell``.
double upstream_alpha(const std::array<double, 9>& cell);

EwaldSetup upstream_ewald(const std::array<double, 9>& cell);

// ``span`` bounds every pair displacement (bohr); ``slowest`` is the smallest
// Gaussian exponent (min of alpha and the narrowest charge width) that
// decides the real-space cutoff.
EwaldSetup converged_ewald(const std::array<double, 9>& cell, double alpha, double gam_min, double span);

// Real-space plus reciprocal Ewald sum for one displacement and its derivative
// with respect to the displacement (dg) and a homogeneous strain (ds, 3x3 row-major).
double ewald_value(const EwaldSetup& setup, const double vec[3], double gam);
void ewald_derivative(const EwaldSetup& setup, const double vec[3], double gam, double dg[3],
                      double ds[9]);

// The diagonal self term of a species with self-interaction exponent ``gam``
// (images of the atom itself plus the Gaussian self energy minus the Ewald
// self term) and its strain derivative; excludes eta and sqrt(2/pi)/rad.
double ewald_self_value(const EwaldSetup& setup, double gam);
void ewald_self_derivative(const EwaldSetup& setup, double gam, double ds[9]);

}  // namespace vanderfast
