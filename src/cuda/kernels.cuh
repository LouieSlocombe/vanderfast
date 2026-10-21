// Host-callable launchers for the evaluation pipeline kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "views.cuh"

namespace vanderfast::cuda {

// K0: pack row-major positions (bohr) and species indices into double4_32a.
void launch_pack_positions(const double* positions, const int* species, std::int32_t n,
                           double4_32a* packed, cudaStream_t stream);

// K6: deterministic column sums. ``columns`` holds ``n_columns`` arrays of
// ``n`` doubles at stride ``stride``; ``partials`` needs kReduceBlocks *
// n_columns doubles and ``sums`` n_columns doubles.
constexpr int kReduceBlocks = 128;
constexpr int kReduceThreads = 256;
void launch_column_sums(const double* columns, int n_columns, std::size_t stride, std::size_t n,
                        double* partials, double* sums, cudaStream_t stream);

// K1: coordination numbers over the CN view.
void launch_cn(const GeometryView& g, const ModelView& m, const GraphView& graph, double* cn,
               cudaStream_t stream);

// K2: reference weights; ``dw`` may be null for energy-only evaluations.
void launch_weights(const ModelView& m, const GeometryView& g, const double* cn, double* w,
                    double* dw, cudaStream_t stream);

// K3: pair energy and, when ``dw`` is non-null, direct derivatives and dE/dCN.
// The owner-only variant is used when ``cv`` carries contracted weights.
void launch_pair(const GeometryView& g, const ModelView& m, const GraphView& graph, const double* w,
                 const double* dw, const ContractedView& cv, const AccumView& acc,
                 cudaStream_t stream);

// K2b: species-contracted weights; ``dw``/``du`` may be null (energy only).
void launch_contract(const ModelView& m, const GeometryView& g, const double* w, const double* dw,
                     double* u, double* du, cudaStream_t stream);

// Row-chunked owner kernels (optimised variant). Items are kRowChunk edges of
// one centre's row; ``partials`` needs 11 (pair), 9 (CN reverse), or 1 (CN)
// doubles per item.
constexpr int kRowChunk = 256;
void launch_cn_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                     const long long* prefix, long long items, double* partials, double* cn,
                     cudaStream_t stream);
void launch_pair_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                       const long long* prefix, long long items, const double* w, const double* dw,
                       const ContractedView& cv, const AccumView& acc, double* partials,
                       cudaStream_t stream);
void launch_cn_reverse_items(const GeometryView& g, const ModelView& m, const GraphView& graph,
                             const long long* prefix, long long items, const AccumView& acc,
                             double* partials, cudaStream_t stream);

// K4: ATM triangles for the work items [item_begin, item_begin + item_count).
// ``partials`` holds 7 doubles per item of the launch; ``counters`` three
// uint64 (candidates, accepted, C6-skipped).
constexpr int kAtmItemWidth = 32;  // outer sites per work item
void launch_atm(const GeometryView& g, const ModelView& m, const GraphView& graph,
                const long long* item_prefix, long long item_begin, long long item_count,
                const double* w, const double* dw, const AccumView& acc, double* partials,
                unsigned long long* counters, cudaStream_t stream);

// K4 canonical variant: every triangle once, at its minimal vertex, with C6
// from contracted weights. With ``split_atoms`` the energy is also split 1/3
// per vertex into acc.atm_split (for per-atom energies).
void launch_atm_canonical(const GeometryView& g, const ModelView& m, const GraphView& graph,
                          const long long* item_prefix, long long item_begin, long long item_count,
                          const double* w, const double* dw, const ContractedView& cv,
                          const AccumView& acc, double* partials, unsigned long long* counters,
                          bool split_atoms, cudaStream_t stream);

// K5: CN-mediated gradient and virial from the accumulated dE/dCN.
void launch_cn_reverse(const GeometryView& g, const ModelView& m, const GraphView& graph,
                       const AccumView& acc, bool owner, cudaStream_t stream);

}  // namespace vanderfast::cuda
