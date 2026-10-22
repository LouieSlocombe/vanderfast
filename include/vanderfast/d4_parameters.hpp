// Resolved DFT-D4 parameters, cutoffs, and charge-model options (atomic units).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vanderfast/parameters.hpp"

namespace vanderfast {

struct D4Parameters {
  double s6 = 1.0;
  double s8 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;   // bohr
  double s9 = 1.0;   // zero disables ATM
  double alp = 16.0; // ATM exponent, used as is (alp / 3 on r0 / r)
  double ga = 3.0;   // charge scaling height
  double gc = 2.0;   // charge scaling steepness (times the chemical hardness)
  double wf = 6.0;   // Gaussian weighting factor
  Cutoffs cutoffs{30.0, 60.0, 40.0};

  static constexpr double kEeqCnCutoff = 25.0;  // bohr, fixed by multicharge

  void validate() const;
  bool pair_active() const noexcept;
  bool atm_active() const noexcept;
  double atm_exponent() const noexcept { return alp; }
  // Radius covering the D4 and EEQ coordination numbers (bohr).
  double cn_radius() const noexcept { return cutoffs.cn > kEeqCnCutoff ? cutoffs.cn : kEeqCnCutoff; }
  std::uint64_t fingerprint() const noexcept;
};

// Periodic EEQ: dftd4's fixed Ewald truncation, or complete lattice sums.
enum class EwaldMode { Upstream, Converged };

// How the bordered EEQ system is solved.
enum class ChargeSolver { Auto, Iterative, Direct };

struct EeqOptions {
  EwaldMode ewald = EwaldMode::Upstream;
  ChargeSolver solver = ChargeSolver::Auto;
  double tolerance = 1e-13;     // accepted normwise backward error
  int max_iterations = 2000;    // Krylov iterations including restarts
  bool warm_start = false;      // start from the previous solution
  std::size_t max_matrix_bytes = 0;  // 0: half of the device memory

  void validate() const;
};

}  // namespace vanderfast
