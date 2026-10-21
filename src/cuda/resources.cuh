// RAII ownership of CUDA streams, events, memory pools, and buffers.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "../common/checked_math.hpp"
#include "cuda_check.cuh"

namespace vanderfast::cuda {

// Sets the current device for the lifetime of the guard and restores the
// previous one afterwards.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    VF_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) VF_CUDA_CHECK(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
};

class Stream {
 public:
  Stream() = default;
  static Stream create() {
    Stream stream;
    VF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream.handle_, cudaStreamNonBlocking));
    return stream;
  }
  ~Stream() {
    if (handle_ != nullptr) cudaStreamDestroy(handle_);
  }
  Stream(Stream&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Stream& operator=(Stream&& other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const noexcept { return handle_; }
  void synchronize() const { VF_CUDA_CHECK(cudaStreamSynchronize(handle_)); }

 private:
  cudaStream_t handle_ = nullptr;
};

class Event {
 public:
  Event() = default;
  static Event create() {
    Event event;
    VF_CUDA_CHECK(cudaEventCreate(&event.handle_));
    return event;
  }
  ~Event() {
    if (handle_ != nullptr) cudaEventDestroy(handle_);
  }
  Event(Event&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Event& operator=(Event&& other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const noexcept { return handle_; }
  void record(cudaStream_t stream) const { VF_CUDA_CHECK(cudaEventRecord(handle_, stream)); }

 private:
  cudaEvent_t handle_ = nullptr;
};

// A device memory pool owned by one context, so stream-ordered allocations
// never interfere with other libraries using the device's default pool.
class MemoryPool {
 public:
  MemoryPool() = default;
  static MemoryPool create(int device) {
    MemoryPool pool;
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    VF_CUDA_CHECK(cudaMemPoolCreate(&pool.handle_, &props));
    std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
    VF_CUDA_CHECK(cudaMemPoolSetAttribute(pool.handle_, cudaMemPoolAttrReleaseThreshold, &threshold));
    return pool;
  }
  ~MemoryPool() {
    if (handle_ != nullptr) cudaMemPoolDestroy(handle_);
  }
  MemoryPool(MemoryPool&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  MemoryPool& operator=(MemoryPool&& other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;
  cudaMemPool_t get() const noexcept { return handle_; }

 private:
  cudaMemPool_t handle_ = nullptr;
};

// Allocation bookkeeping shared by the buffers of one context.
struct AllocationCounters {
  std::int64_t allocations = 0;
  std::size_t device_bytes = 0;
};

// Move-only device array with stream-ordered allocation from a context pool.
// Capacity grows geometrically; growth discards the previous contents.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(cudaMemPool_t pool, cudaStream_t stream, AllocationCounters* counters)
      : pool_(pool), stream_(stream), counters_(counters) {}
  ~DeviceBuffer() { release(); }
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    swap(other);
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Ensure capacity for ``count`` elements; returns true when reallocated.
  bool reserve(std::size_t count) {
    if (count <= capacity_) return false;
    std::size_t grown = capacity_ + capacity_ / 2;
    std::size_t target = count > grown ? count : grown;
    std::size_t bytes = checked_mul(target, sizeof(T), "device buffer size");
    bytes = checked_add(bytes, std::size_t{255}, "device buffer size") & ~std::size_t{255};
    release();
    void* pointer = nullptr;
    VF_CUDA_CHECK(cudaMallocFromPoolAsync(&pointer, bytes, pool_, stream_));
    ptr_ = static_cast<T*>(pointer);
    capacity_ = bytes / sizeof(T);
    if (counters_ != nullptr) {
      counters_->allocations += 1;
      counters_->device_bytes += capacity_ * sizeof(T);
    }
    return true;
  }

  bool resize(std::size_t count) {
    bool reallocated = reserve(count);
    size_ = count;
    return reallocated;
  }

  void zero_async() {
    if (size_ != 0) VF_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream_));
  }
  void upload_async(const T* host, std::size_t count) {
    resize(count);
    if (count != 0) {
      VF_CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream_));
    }
  }
  void download_async(T* host, std::size_t count) const {
    if (count != 0) {
      VF_CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream_));
    }
  }

  void release() noexcept {
    if (ptr_ != nullptr) {
      cudaFreeAsync(ptr_, stream_);
      if (counters_ != nullptr) counters_->device_bytes -= capacity_ * sizeof(T);
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  void swap(DeviceBuffer& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(pool_, other.pool_);
    std::swap(stream_, other.stream_);
    std::swap(counters_, other.counters_);
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
  cudaMemPool_t pool_ = nullptr;
  cudaStream_t stream_ = nullptr;
  AllocationCounters* counters_ = nullptr;
};

// Asynchronous device-to-host copy of ``count`` elements.
template <class T>
void copy_to_host_async(const T* device, T* host, std::size_t count, cudaStream_t stream) {
  if (count != 0) {
    VF_CUDA_CHECK(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
}

// Page-locked host staging memory for asynchronous transfers.
template <class T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { release(); }
  PinnedBuffer(PinnedBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), capacity_(std::exchange(other.capacity_, 0)) {}
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(capacity_, other.capacity_);
    return *this;
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void reserve(std::size_t count) {
    if (count <= capacity_) return;
    std::size_t target = count > 2 * capacity_ ? count : 2 * capacity_;
    std::size_t bytes = checked_mul(target, sizeof(T), "pinned buffer size");
    release();
    void* pointer = nullptr;
    VF_CUDA_CHECK(cudaMallocHost(&pointer, bytes));
    ptr_ = static_cast<T*>(pointer);
    capacity_ = target;
  }
  void release() noexcept {
    if (ptr_ != nullptr) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
  }
  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  T* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace vanderfast::cuda
