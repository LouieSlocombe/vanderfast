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
  // Optimised kernels: reciprocal weights 4 pi / V exp(-G^2 / 4 alpha^2) / G^2
  // and the translations sorted by length as (x, y, z, |T|).
  const double* recip_e;        // [G]
  const double* trans_sorted;   // [T * 4]
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

// EEQ small systems: LU with partial pivoting of K in shared memory, factors
// written back to ``k`` (for later solves) and ``b`` solved in place. One block;
// needs (n + 1)^2 doubles of shared memory (lu_small_fits).
bool lu_small_fits(int n, int device);
void launch_lu_small_factor_solve(double* k, int n, int* pivots, double* b, int* status,
                                  cudaStream_t stream);
// Solve with the factors of launch_lu_small_factor_solve (one warp).
void launch_lu_small_solve(const double* lu, int n, const int* pivots, double* b, cudaStream_t stream);

// Preconditioned MINRES (Paige-Saunders) on the bordered system K x = b with
// the Jacobi preconditioner diag(|A_ii|, sum_i 1/|A_ii|). Every block of one
// cooperative launch evaluates the scalar recurrences from the same
// fixed-order partial sums, so results are bitwise reproducible. Converged
// inner iterations are checked against the true residual, eta = |b - Kx|_inf /
// (|K|_inf |x|_inf + |b|_inf), and restarted from x until eta <= 2^-48.
constexpr int kMinresSlots = 10;  // per-block partial sums held at once

struct MinresState {
  double beta, oldb, dbar, epsln, phibar, cs, sn, beta1, gmax, gmin;
  double knorm, bnorm, dmax, mborder, eta;
  double bscale;  // |b| in the M^-1 norm: the inner tolerance is relative to it
  int itn;       // iterations over all restarts
  int inner;     // iterations since the last (re)start
  int restarts;
  int phase;     // 0 start, 1 iterating, 2 finished
  int status;    // 0 converged, 1 singular, 2 not converged
  int pad;
};

struct MinresArgs {
  const double* k;   // [(n + 1)^2], or packed tiles (eeq_packed_tiles)
  const double* b;   // [n + 1]
  double* x;         // [n + 1] solution; the initial guess when ``warm``
  double* work;      // [9 * (n + 1)] r1, r2, z, v, y, w, w2, m, r
  double* partials;  // [kMinresSlots * blocks]
  double* tile_partials;  // packed: [tiles * 64] row and column sums per tile
  MinresState* state;
  int np;
  int tiles;         // packed tiles, 0 for dense storage
  int max_iterations;
  int warm;
  double tolerance;  // accepted eta
};

// Blocks of the cooperative MINRES launch on the current device.
int minres_blocks(int n, int device);
// Run up to ``iterations`` further iterations (state->phase must be 0 or 1).
void launch_minres(const MinresArgs& args, int blocks, int iterations, cudaStream_t stream);

// EEQ (optimised): cos and sin of G . r_i for every reciprocal vector, [G * n] each.
void launch_structure_factors(const GeometryView& g, const EwaldView& e, double* c, double* s,
                              cudaStream_t stream);

// EEQ (optimised): blocks per tile, splitting the lattice sums of small
// periodic systems so the tiles fill the device (1 for molecules).
int eeq_tile_slices(int n, bool periodic, int device);
std::size_t eeq_assemble_tiles_scratch(int n, int slices);

// EEQ (optimised): K from 32 x 32 lower-triangle tiles (screened, length-sorted
// real-space sums and the reciprocal part from structure factors);
// ``scratch`` holds eeq_assemble_tiles_scratch doubles. Dense output is the
// bordered (n + 1)^2 matrix; packed output is the eeq_packed_tiles(n)
// lower-triangle 32 x 32 tiles of A (diagonal tiles in full, border implicit).
// With ``deriv`` (periodic systems, eeq_derivative_scratch doubles) the
// real-space derivatives of every pair are stored for the reverse pass.
int eeq_packed_tiles(int n);
std::size_t eeq_derivative_scratch(int n, int slices);
void launch_eeq_assemble_tiles(const GeometryView& g, const D4View& d, const EwaldView& e,
                               const double* sf_c, const double* sf_s, int slices, double* scratch,
                               bool packed, double* k, double* deriv, cudaStream_t stream);

// Partial buffers of the tiled reverse pass (doubles).
std::size_t eeq_reverse_tiles_scratch(int n, int n_recip, int slices);

// EEQ reverse (optimised): each pair once per lower-triangle tile with
// fixed-order per-tile partials, the reciprocal part through structure sums,
// and the self images; deterministic.
// ``deriv``: the derivatives stored by launch_eeq_assemble_tiles, or null to
// recompute the lattice sums.
void launch_eeq_reverse_tiles(const GeometryView& g, const D4View& d, const EwaldView& e,
                              const double* sf_c, const double* sf_s, const double* q, const double* nu,
                              const AccumView& acc, int slices, const double* deriv, double* scratch,
                              cudaStream_t stream);

// EEQ reverse: gradient and virial of -nu^T A q (owner rows, deterministic).
void launch_eeq_reverse(const GeometryView& g, const D4View& d, const EwaldView& e,
                        const double* q, const double* nu, const AccumView& acc,
                        cudaStream_t stream);

}  // namespace vanderfast::cuda
