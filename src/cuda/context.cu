// D3Context: device resources, persistent state, and the evaluation pipeline.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "../core/lattice.hpp"
#include "device_model.cuh"
#include "kernels.cuh"
#include "resources.cuh"
#include "vanderfast/context.hpp"
#include "vanderfast/error.hpp"

namespace vanderfast {

using cuda::DeviceBuffer;
using cuda::DeviceGuard;
using cuda::PinnedBuffer;

namespace {

constexpr int kSumColumns = 8;  // e_pair, e_atm, six virial components

int checked_device(int device) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    cudaGetLastError();
    fail(ErrorCode::CudaUnavailable,
         std::string("no CUDA device is available (") + cudaGetErrorString(status) + ")");
  }
  if (device < 0 || device >= count) {
    fail(ErrorCode::InvalidArgument, "CUDA device " + std::to_string(device) + " does not exist (" +
                                         std::to_string(count) + " available)");
  }
  return device;
}

}  // namespace

struct D3Context::Impl {
  Impl(const std::int32_t* numbers_in, std::size_t n, const D3Parameters& params,
       const ContextOptions& opts, std::shared_ptr<const ReferenceData> ref)
      : device(checked_device(opts.device)), options(opts), parameters(params), reference(std::move(ref)) {
    if (!reference) fail(ErrorCode::InvalidArgument, "reference data must not be null");
    parameters.validate();
    if (!(options.skin >= 0.0) || !std::isfinite(options.skin)) {
      fail(ErrorCode::InvalidArgument, "skin must be finite and non-negative");
    }
    checked_int32(static_cast<std::int64_t>(n), "atom count");
    numbers.assign(numbers_in, numbers_in + n);
    for (std::int32_t z : numbers) {
      if (!reference->supports(z)) {
        fail(ErrorCode::UnsupportedElement, "atomic number " + std::to_string(z) +
                                                " is not supported (1.." +
                                                std::to_string(reference->max_z()) + ")");
      }
    }
    std::vector<int> unique(numbers.begin(), numbers.end());
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    species.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      species[i] = static_cast<int>(std::lower_bound(unique.begin(), unique.end(), numbers[i]) -
                                    unique.begin());
    }
    DeviceGuard guard(device);
    stream = cuda::Stream::create();
    pool = cuda::MemoryPool::create(device);
    model = cuda::DeviceModel(*reference, unique, parameters, pool.get(), stream.get(), &counters);
    auto make = [&](auto& buffer) {
      using Buffer = std::decay_t<decltype(buffer)>;
      buffer = Buffer(pool.get(), stream.get(), &counters);
    };
    make(species_dev);
    make(positions_dev);
    make(packed);
    make(arena);
    make(partials);
    make(sums);
    species_dev.upload_async(species.data(), species.size());
    partials.resize(static_cast<std::size_t>(cuda::kReduceBlocks) * kSumColumns);
    sums.resize(kSumColumns);
    host_sums.reserve(kSumColumns);
    stream.synchronize();
  }

  ~Impl() {
    if (stream.get() != nullptr) {
      cudaSetDevice(device);
      cudaStreamSynchronize(stream.get());
    }
  }

  cuda::AccumView accum_view() {
    const std::size_t n = numbers.size();
    double* base = arena.data();
    cuda::AccumView view{};
    view.e_pair = base;
    view.e_atm = base + n;
    view.vir = base + 2 * n;
    view.g = base + 8 * n;
    view.grad = base + 9 * n;
    view.n = static_cast<std::int32_t>(n);
    return view;
  }

  HostResult evaluate(const Geometry& geometry, unsigned properties) {
    if (geometry.n_atoms != numbers.size()) {
      fail(ErrorCode::InvalidArgument, "expected " + std::to_string(numbers.size()) +
                                           " positions, got " + std::to_string(geometry.n_atoms));
    }
    make_lattice(geometry);
    const bool want_gradient = (properties & kGradient) != 0;
    const bool want_virial = (properties & kVirial) != 0;
    const bool want_atoms = (properties & kAtomEnergies) != 0;
    HostResult result;
    const std::size_t n = numbers.size();
    result.has_gradient = want_gradient;
    result.has_virial = want_virial;
    if (want_gradient) result.gradient.assign(3 * n, 0.0);
    result.has_atom_energies = want_atoms;
    if (want_atoms) result.atom_energies.assign(n, 0.0);
    stats.evaluations += 1;
    if (n == 0) return result;

    DeviceGuard guard(device);
    cudaStream_t s = stream.get();

    // K0: upload and pack the geometry.
    staging.reserve(3 * n);
    std::memcpy(staging.data(), geometry.positions, 3 * n * sizeof(double));
    positions_dev.upload_async(staging.data(), 3 * n);
    packed.resize(n);
    cuda::launch_pack_positions(positions_dev.data(), species_dev.data(),
                                static_cast<std::int32_t>(n), packed.data(), s);

    // K0b: reset every accumulator of this evaluation in one call.
    arena.resize(12 * n);
    arena.zero_async();

    // K6: reductions and transfers.
    const cuda::AccumView accum = accum_view();
    cuda::launch_column_sums(accum.e_pair, kSumColumns, n, n, partials.data(), sums.data(), s);
    sums.download_async(host_sums.data(), kSumColumns);
    if (want_gradient) {
      host_gradient.reserve(3 * n);
      cuda::copy_to_host_async(accum.grad, host_gradient.data(), 3 * n, s);
    }
    if (want_atoms) {
      host_atoms.reserve(2 * n);
      cuda::copy_to_host_async(accum.e_pair, host_atoms.data(), 2 * n, s);
    }
    stream.synchronize();

    result.energy_pair = host_sums.data()[0];
    result.energy_atm = host_sums.data()[1];
    result.energy = result.energy_pair + result.energy_atm;
    if (want_virial) {
      const double* v = host_sums.data() + 2;
      result.virial = {v[0], v[5], v[4], v[5], v[1], v[3], v[4], v[3], v[2]};
    }
    if (want_gradient) std::memcpy(result.gradient.data(), host_gradient.data(), 3 * n * sizeof(double));
    if (want_atoms) {
      for (std::size_t i = 0; i < n; ++i) result.atom_energies[i] = host_atoms.data()[i] + host_atoms.data()[n + i];
    }
    return result;
  }

  std::mutex mutex;
  bool poisoned = false;
  int device;
  ContextOptions options;
  D3Parameters parameters;
  std::shared_ptr<const ReferenceData> reference;
  std::vector<std::int32_t> numbers;
  std::vector<int> species;
  cuda::AllocationCounters counters;
  ContextStats stats;
  cuda::Stream stream;
  cuda::MemoryPool pool;
  cuda::DeviceModel model;
  DeviceBuffer<int> species_dev;
  DeviceBuffer<double> positions_dev;
  DeviceBuffer<double4_32a> packed;
  DeviceBuffer<double> arena;
  DeviceBuffer<double> partials;
  DeviceBuffer<double> sums;
  PinnedBuffer<double> staging;
  PinnedBuffer<double> host_sums;
  PinnedBuffer<double> host_gradient;
  PinnedBuffer<double> host_atoms;
};

D3Context::D3Context(const std::int32_t* numbers, std::size_t n_atoms, const D3Parameters& parameters,
                     const ContextOptions& options, std::shared_ptr<const ReferenceData> reference)
    : impl_(std::make_unique<Impl>(numbers, n_atoms, parameters, options, std::move(reference))) {}

D3Context::~D3Context() = default;
D3Context::D3Context(D3Context&&) noexcept = default;
D3Context& D3Context::operator=(D3Context&&) noexcept = default;

HostResult D3Context::evaluate(const Geometry& geometry, unsigned properties) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    fail(ErrorCode::ContextBusy, "this context is already evaluating on another thread");
  }
  if (impl_->poisoned) {
    fail(ErrorCode::ContextPoisoned, "an earlier CUDA error invalidated this context");
  }
  try {
    return impl_->evaluate(geometry, properties);
  } catch (const Error& error) {
    if (error.code() == ErrorCode::CudaRuntime) impl_->poisoned = true;
    throw;
  }
}

DebugResult D3Context::debug(const Geometry& geometry, bool derivatives) {
  (void)geometry;
  (void)derivatives;
  fail(ErrorCode::UnsupportedFeature, "debug output is not implemented yet");
}

void D3Context::set_parameters(const D3Parameters& parameters) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  parameters.validate();
  DeviceGuard guard(impl_->device);
  impl_->parameters = parameters;
  impl_->model.update_parameters(*impl_->reference, parameters);
  impl_->stream.synchronize();
}

void D3Context::invalidate_neighbors() noexcept {}

std::size_t D3Context::n_atoms() const noexcept { return impl_->numbers.size(); }
const D3Parameters& D3Context::parameters() const noexcept { return impl_->parameters; }
const ContextOptions& D3Context::options() const noexcept { return impl_->options; }

ContextStats D3Context::stats() const {
  ContextStats stats = impl_->stats;
  stats.allocations = impl_->counters.allocations;
  stats.device_bytes = impl_->counters.device_bytes;
  return stats;
}

int D3Context::device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return count;
}

std::vector<DeviceInfo> D3Context::devices() {
  std::vector<DeviceInfo> result;
  const int count = device_count();
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp props{};
    VF_CUDA_CHECK(cudaGetDeviceProperties(&props, index));
    int timeout = 0;
    VF_CUDA_CHECK(cudaDeviceGetAttribute(&timeout, cudaDevAttrKernelExecTimeout, index));
    DeviceInfo info;
    info.index = index;
    info.name = props.name;
    info.compute_major = props.major;
    info.compute_minor = props.minor;
    info.total_memory = props.totalGlobalMem;
    info.multiprocessors = props.multiProcessorCount;
    info.kernel_timeout = timeout != 0;
    result.push_back(info);
  }
  return result;
}

}  // namespace vanderfast
