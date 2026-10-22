// D4-specific views and host-callable launchers: coordination numbers,
// charge-scaled weights, and the EEQ charge model.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "views.cuh"

namespace vanderfast::cuda {

// Per-species D4 and EEQ tables (padded reference arrays of kRefPad entries).
struct D4View {
  int n_species;
  const int* ngw;          // [S * kRefPad] Gaussians per reference
  const double* cnref;     // [S * kRefPad] weighting CNs
  const double* qref;      // [S * kRefPad] reference charge + zeff
  const unsigned* maxmask; // [S] references at the maximum weighting CN
  const double* zeff;      // [S]
  const double* gi;        // [S] chemical hardness * gc
  const double* chi;       // [S] EEQ electronegativity
  const double* eta;       // [S] EEQ hardness
  const double* kcnchi;    // [S] EEQ CN scaling
  const double* rad;       // [S] EEQ charge width
  double ga, wf;
  double cn_cutoff2;       // D4 CN cutoff squared
  double eeq_cutoff2;      // EEQ CN cutoff squared (25 bohr)
};

// Lattice data of the periodic Coulomb matrix (device copies of EwaldSetup).
struct EwaldView {
  int mode;                     // 0 molecular, 1 upstream, 2 converged
  double alpha;
  double volume;
  double real_cutoff;           // converged mode
  int n_trans;
  int n_recip;
  const double* dtrans;         // [T * 3]
  const double* rtrans;         // [G * 3]
  const double* ws_candidates;  // [27 * 3] upstream mode
  const double* self_diag;      // [S] periodic diagonal (self images, Gaussian self, Ewald self)
  const double* self_vir;       // [S * 6] strain derivative of the self images
};

// K1 (D4): fused D4 and EEQ coordination numbers over the CN view; the
// reduction also forms the EEQ right-hand side X (length n + 1, last entry
// the total charge) and dXfac = dX/dcn_raw. ``partials`` needs 2 per item.
void launch_d4_cn_items(const GeometryView& g, const ModelView& m, const D4View& d,
                        const GraphView& graph, const long long* prefix, long long items,
                        double total_charge, double* partials, double* cn, double* cn_eeq,
                        double* rhs, double* dxfac, cudaStream_t stream);

// K2 (D4): charge-scaled weights w, dw/dCN, dw/dq and, when ``w0`` is
// non-null, the neutral (q = 0) weights w0 and dw0/dCN for ATM. The
// derivative arrays may be null for energy-only evaluations.
void launch_d4_weights(const D4View& d, const GeometryView& g, const double* cn, const double* q,
                       double* w, double* dw_cn, double* dw_q, double* w0, double* dw0,
                       cudaStream_t stream);

// K5 (D4): CN-mediated gradient and virial from dE/dCN (acc.g) and from the
// EEQ adjoint, whose dE/dcn_raw is nu_i * dxfac_i.
void launch_d4_cn_reverse_items(const GeometryView& g, const ModelView& m, const D4View& d,
                                const GraphView& graph, const long long* prefix, long long items,
                                const AccumView& acc, const double* nu, const double* dxfac,
                                double* partials, cudaStream_t stream);

// EEQ: dense bordered matrix K (row-major, (n + 1)^2) from ``pos`` (wrapped
// positions for periodic systems, species in w).
void launch_eeq_assemble(const GeometryView& g, const D4View& d, const EwaldView& e, double* k,
                         cudaStream_t stream);

// EEQ: LU factorisation with partial pivoting of the dense (n + 1)^2 matrix in
// place; ``status`` receives 1 if a pivot vanishes.
void launch_lu_factor(double* k, int n, int* pivots, int* status, cudaStream_t stream);

// EEQ: solve K x = b in place with the factors of launch_lu_factor.
void launch_lu_solve(const double* lu, int n, const int* pivots, double* b, cudaStream_t stream);

// EEQ reverse: gradient and virial of -nu^T A q (owner rows, deterministic).
void launch_eeq_reverse(const GeometryView& g, const D4View& d, const EwaldView& e,
                        const double* q, const double* nu, const AccumView& acc,
                        cudaStream_t stream);

}  // namespace vanderfast::cuda
