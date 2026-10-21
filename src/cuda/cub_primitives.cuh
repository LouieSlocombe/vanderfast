// Device-wide CUB primitives behind a plain interface.
//
// CUB's kernels are compiled in their own translation unit (cub_primitives.cu)
// so that the fatal ptxas spill warnings applied to vanderfast's kernels do not
// apply to NVIDIA's tuning policies; see cmake/VanderfastCuda.cmake.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace vanderfast::cuda {

// Exclusive prefix sum of n values. With temp == nullptr, only sets temp_bytes.
cudaError_t exclusive_sum(void* temp, std::size_t& temp_bytes, const long long* in,
                          long long* out, long long n, cudaStream_t stream);

// Stable radix sort of (key, value) pairs on key bits [begin_bit, end_bit).
// With temp == nullptr, only sets temp_bytes.
cudaError_t sort_pairs(void* temp, std::size_t& temp_bytes, const int* keys_in, int* keys_out,
                       const int* values_in, int* values_out, int n, int begin_bit, int end_bit,
                       cudaStream_t stream);

}  // namespace vanderfast::cuda
