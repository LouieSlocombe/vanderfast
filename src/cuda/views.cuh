// Plain-old-data views passed by value to kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vanderfast::cuda {

constexpr int kRefPad = 8;  // reference arrays are padded to 8 entries
constexpr int kC6Block = kRefPad * kRefPad;

// Species-pair constants, computed once on the host in upstream operation order.
struct alignas(16) PairConst {
  double rc_cn;   // rcov_i + rcov_j (bohr)
  double kappa;   // 3 r4r2_i r4r2_j
  double r0_6;    // (a1 sqrt(kappa) + a2)^6
  double r0_8;    // (a1 sqrt(kappa) + a2)^8
  double r0_atm;  // rs9 rvdw_ij
  double pad;
};

struct ModelView {
  int n_species;
  int max_nref;
  const int* nref;             // [S]
  const double* cnref;         // [S * kRefPad]
  const unsigned* maxmask;     // [S] bit a set iff cnref[a] equals the maximum valid cnref
  const double* c6ref;         // [S * S * kC6Block], block (s, t) rows for s
  const PairConst* pair;       // [S * S]
  double s6, s8, s9;
  double alp_eff;              // alp + 2
  double alp3;                 // (alp + 2) / 3
  double cn_cutoff2, disp2_cutoff2, disp3_cutoff2;
};

struct GeometryView {
  std::int32_t n;
  const double4_32a* pos;  // x, y, z (bohr) and species index in w
  double cell[9];      // lattice vectors as rows (bohr)
};

// Directed edges ``(i -> j, n)`` in CSR form. Each row is ordered by radius
// band so that the CN and ATM views are prefixes of the row.
struct GraphView {
  const long long* row_ptr;  // [N + 1]
  const int4* edges;         // {j, n0, n1, n2}
  const int* len_cn;         // [N] row prefix length of the CN view
  const int* len_pair;       // [N] row prefix length of the pair view
  const int* len_atm;        // [N] row prefix length of the ATM view
};

// Per-atom accumulation arena, reset once per evaluation.
struct AccumView {
  double* grad;    // [3N] dE/dx
  double* g;       // [N] dE/dCN
  double* e_pair;  // [N]
  double* e_atm;   // [N]
  double* vir;     // [6N] SoA: xx, yy, zz, yz, xz, xy
  std::int32_t n;
};

}  // namespace vanderfast::cuda
