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

}  // namespace vanderfast::cuda
