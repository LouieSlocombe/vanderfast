#include "cub_primitives.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

namespace vanderfast::cuda {

cudaError_t exclusive_sum(void* temp, std::size_t& temp_bytes, const long long* in,
                          long long* out, long long n, cudaStream_t stream) {
  return cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream);
}

cudaError_t sort_pairs(void* temp, std::size_t& temp_bytes, const int* keys_in, int* keys_out,
                       const int* values_in, int* values_out, int n, int begin_bit, int end_bit,
                       cudaStream_t stream) {
  return cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys_in, keys_out, values_in,
                                         values_out, n, begin_bit, end_bit, stream);
}

}  // namespace vanderfast::cuda
