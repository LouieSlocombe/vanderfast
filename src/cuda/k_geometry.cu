#include "cuda_check.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

__global__ void pack_positions_kernel(const double* __restrict__ positions,
                                      const int* __restrict__ species, std::int32_t n,
                                      double4_32a* __restrict__ packed) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  packed[i] = make_double4_32a(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2],
                           static_cast<double>(species[i]));
}

}  // namespace

void launch_pack_positions(const double* positions, const int* species, std::int32_t n,
                           double4_32a* packed, cudaStream_t stream) {
  if (n == 0) return;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  pack_positions_kernel<<<blocks, threads, 0, stream>>>(positions, species, n, packed);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
