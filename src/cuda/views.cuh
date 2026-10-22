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
  double r0_atm;  // ATM radius: rs9 rvdw_ij (D3) or a1 sqrt(kappa) + a2 (D4)
  double aux;     // model-specific: the D4 electronegativity factor of the CN; unused by D3
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
  double alp_eff;              // ATM exponent: alp + 2 (D3) or alp (D4)
  double alp3;                 // alp_eff / 3
  double cn_cutoff2, disp2_cutoff2, disp3_cutoff2;
  double atm_c6_floor;         // ATM triangles with any |C6| below this are skipped
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
  const int* upper;          // [P] row-relative offsets of canonical upper ATM sites
  const int* counts;         // [4N] per-row band counts; counts[4 i + 3] upper ATM sites
};

// Per-atom accumulation arena, reset once per evaluation.
struct AccumView {
  double* atm_split;  // [N] ATM energy split 1/3 per vertex (atomic energies only)
  double* grad;    // [3N] dE/dx
  double* g;       // [N] dE/dCN
  double* e_pair;  // [N]
  double* e_atm;   // [N]
  double* vir;     // [6N] SoA: xx, yy, zz, yz, xz, xy
  std::int32_t n;
  double* gq;      // [N] dE/dq (D4 only; null for D3)
};

// Species-contracted weights (optional; null in the literal variant).
struct ContractedView {
  const double* u;   // [N * S * kRefPad]
  const double* du;  // [N * S * kRefPad], null for energy-only evaluations
  const double* dq = nullptr;  // [N * S * kRefPad] charge derivative (D4 only)
};

}  // namespace vanderfast::cuda
