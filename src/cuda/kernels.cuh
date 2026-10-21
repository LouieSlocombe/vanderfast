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
void launch_pair(const GeometryView& g, const ModelView& m, const GraphView& graph, const double* w,
                 const double* dw, const AccumView& acc, cudaStream_t stream);

// K5: CN-mediated gradient and virial from the accumulated dE/dCN.
void launch_cn_reverse(const GeometryView& g, const ModelView& m, const GraphView& graph,
                       const AccumView& acc, cudaStream_t stream);

}  // namespace vanderfast::cuda
