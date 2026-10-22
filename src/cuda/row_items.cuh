// Work items of the row-chunked owner kernels.
//
// A row (one centre's edges in a view) is split into items of kRowChunk edges.
// ``prefix[i]`` is the first item of centre i, so the owning centre of an item
// is found by binary search and its edge range follows from the row length.
#pragma once

#include "kernels.cuh"

namespace vanderfast::cuda {

__device__ __forceinline__ int centre_of(const long long* __restrict__ prefix, int n, long long item) {
  int lo = 0;
  int hi = n;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (prefix[mid] <= item) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

struct ItemRange {
  int i;
  long long begin;
  long long end;
};

__device__ __forceinline__ ItemRange item_range(const long long* prefix, int n, long long item,
                                                const long long* row_ptr, const int* len) {
  const int i = centre_of(prefix, n, item);
  const long long chunk = item - prefix[i];
  const long long row = row_ptr[i];
  const long long begin = row + chunk * kRowChunk;
  const long long end = min(begin + kRowChunk, row + len[i]);
  return {i, begin, end};
}

inline unsigned item_blocks(long long items) {
  return static_cast<unsigned>((items + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

inline unsigned centre_blocks(int n) { return static_cast<unsigned>((n + 127) / 128); }

}  // namespace vanderfast::cuda
