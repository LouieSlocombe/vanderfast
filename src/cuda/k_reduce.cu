#include "cuda_check.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

// Fixed-shape block reduction: the summation order depends only on the
// launch configuration and n, so results are bitwise reproducible.
__device__ double block_sum(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) shared[threadIdx.x] += shared[threadIdx.x + width];
    __syncthreads();
  }
  return shared[0];
}

__global__ void column_partials_kernel(const double* __restrict__ columns, std::size_t stride,
                                       std::size_t n, double* __restrict__ partials) {
  __shared__ double shared[kReduceThreads];
  const int column = static_cast<int>(blockIdx.y);
  const double* data = columns + static_cast<std::size_t>(column) * stride;
  double sum = 0.0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum += data[i];
  }
  const double total = block_sum(sum, shared);
  if (threadIdx.x == 0) partials[column * gridDim.x + blockIdx.x] = total;
}

__global__ void column_finish_kernel(const double* __restrict__ partials, int n_partials,
                                     double* __restrict__ sums) {
  __shared__ double shared[kReduceThreads];
  const int column = static_cast<int>(blockIdx.x);
  double sum = 0.0;
  for (int i = static_cast<int>(threadIdx.x); i < n_partials; i += static_cast<int>(blockDim.x)) {
    sum += partials[column * n_partials + i];
  }
  const double total = block_sum(sum, shared);
  if (threadIdx.x == 0) sums[column] = total;
}

}  // namespace

void launch_column_sums(const double* columns, int n_columns, std::size_t stride, std::size_t n,
                        double* partials, double* sums, cudaStream_t stream) {
  if (n_columns == 0) return;
  const dim3 grid(kReduceBlocks, static_cast<unsigned>(n_columns));
  column_partials_kernel<<<grid, kReduceThreads, 0, stream>>>(columns, stride, n, partials);
  VF_LAUNCH_CHECK(stream);
  column_finish_kernel<<<n_columns, kReduceThreads, 0, stream>>>(partials, kReduceBlocks, sums);
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
