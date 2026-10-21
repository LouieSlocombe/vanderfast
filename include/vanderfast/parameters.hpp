// Resolved D3(BJ) damping parameters and real-space cutoffs (atomic units).
#pragma once

#include <cstdint>

namespace vanderfast {

struct Cutoffs {
  double cn = 40.0;     // bohr
  double disp2 = 60.0;  // bohr
  double disp3 = 40.0;  // bohr
};

struct D3Parameters {
  double s6 = 1.0;
  double s8 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;   // bohr
  double s9 = 0.0;   // zero disables ATM
  double alp = 14.0; // public exponent; the BJ ATM kernel uses alp + 2
  Cutoffs cutoffs{};

  static constexpr double kRs9 = 4.0 / 3.0;
  static constexpr double kWf = 4.0;
  static constexpr double kKcn = 16.0;

  void validate() const;
  bool pair_active() const noexcept;
  bool atm_active() const noexcept;
  double atm_exponent() const noexcept { return alp + 2.0; }
  std::uint64_t fingerprint() const noexcept;
};

}  // namespace vanderfast
