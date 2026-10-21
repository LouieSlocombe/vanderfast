// CUDA error checking that maps runtime failures onto vanderfast::Error.
#pragma once

#include <cuda_runtime.h>

#include "vanderfast/error.hpp"

namespace vanderfast::cuda {

[[noreturn]] void throw_cuda_error(cudaError_t status, const char* expression, const char* file,
                                   int line);

// True for errors that leave the CUDA context unusable.
bool is_sticky(cudaError_t status) noexcept;

inline void check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) throw_cuda_error(status, expression, file, line);
}

}  // namespace vanderfast::cuda

#define VF_CUDA_CHECK(expression) \
  ::vanderfast::cuda::check((expression), #expression, __FILE__, __LINE__)

#ifdef VF_DEBUG_SYNC
#define VF_LAUNCH_CHECK(stream)                                                            \
  do {                                                                                     \
    ::vanderfast::cuda::check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__);    \
    ::vanderfast::cuda::check(cudaStreamSynchronize(stream), "kernel", __FILE__, __LINE__); \
  } while (0)
#else
#define VF_LAUNCH_CHECK(stream) \
  ::vanderfast::cuda::check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
#endif
