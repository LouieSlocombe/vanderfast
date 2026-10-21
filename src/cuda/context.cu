// D3Context: device resources, persistent state, and the evaluation pipeline.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "../core/lattice.hpp"
#include "../common/d3_math.hpp"
#include "device_model.cuh"
#include "neighbor.cuh"
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
constexpr long long kBruteForceLimit = 1LL << 24;  // candidate pairs for the O(N^2) builder

// Bin the search basis with bins of about half the largest radius, capped at
// max(64, 4N) bins. The stencil reach is conservative for every direction.
cuda::CellGrid make_grid(const Lattice& lattice, const double* x, std::size_t n, double rmax) {
  cuda::CellGrid grid;
  std::memcpy(grid.inverse, lattice.inverse.data(), sizeof(grid.inverse));
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double s = x[3 * i] * grid.inverse[k] + x[3 * i + 1] * grid.inverse[3 + k] +
                       x[3 * i + 2] * grid.inverse[6 + k];
      lo[k] = i == 0 ? s : std::min(lo[k], s);
      hi[k] = i == 0 ? s : std::max(hi[k], s);
    }
  }
  const double width = rmax / 2.0;
  double extent[3];
  for (int k = 0; k < 3; ++k) {
    grid.periodic[k] = lattice.pbc[static_cast<std::size_t>(k)] ? 1 : 0;
    if (grid.periodic[k]) {
      if (std::max(std::fabs(lo[k]), std::fabs(hi[k])) > 1.0e9) {
        fail(ErrorCode::InvalidGeometry, "positions are too far outside the periodic cell");
      }
      extent[k] = lattice.spacing[static_cast<std::size_t>(k)];
    } else {
      extent[k] = hi[k] - lo[k];
      grid.origin[k] = lo[k];
    }
    grid.nb[k] = extent[k] > 0.0 ? std::max(1, static_cast<int>(std::floor(extent[k] / width))) : 1;
  }
  const long long limit = std::max<long long>(64, 4 * static_cast<long long>(n));
  while (grid.bins() > limit) {
    int largest = 0;
    for (int k = 1; k < 3; ++k) {
      if (grid.nb[k] > grid.nb[largest]) largest = k;
    }
    grid.nb[largest] = std::max(1, grid.nb[largest] / 2);
  }
  for (int k = 0; k < 3; ++k) {
    if (grid.periodic[k]) {
      grid.scale[k] = grid.nb[k];
      grid.reach[k] = static_cast<int>(std::floor(rmax * grid.nb[k] / extent[k])) + 1;
    } else if (extent[k] > 0.0) {
      grid.scale[k] = grid.nb[k] / extent[k];
      grid.reach[k] = std::min(static_cast<int>(std::floor(rmax * grid.scale[k])) + 1, grid.nb[k] - 1);
    } else {
      grid.scale[k] = 0.0;
      grid.reach[k] = 0;
    }
    if (grid.reach[k] > (1 << 20)) {
      fail(ErrorCode::InvalidGeometry, "the cell is too thin for the cutoff");
    }
  }
  return grid;
}

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

// Optional CUDA-event timings of the pipeline stages (stream time between marks).
class StageTimer {
 public:
  StageTimer(bool enabled, cudaStream_t stream, std::vector<cuda::Event>& pool)
      : enabled_(enabled), stream_(stream), pool_(pool) {}
  void mark(const char* name) {
    if (!enabled_) return;
    if (pool_.size() <= names_.size()) pool_.push_back(cuda::Event::create());
    pool_[names_.size()].record(stream_);
    names_.push_back(name);
  }
  void collect(std::map<std::string, double>& out) const {
    for (std::size_t k = 1; k < names_.size(); ++k) {
      float ms = 0.0f;
      VF_CUDA_CHECK(cudaEventElapsedTime(&ms, pool_[k - 1].get(), pool_[k].get()));
      out[names_[k]] = static_cast<double>(ms);
    }
  }

 private:
  bool enabled_;
  cudaStream_t stream_;
  std::vector<cuda::Event>& pool_;
  std::vector<const char*> names_;
};

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
    graph = cuda::NeighborGraph(pool.get(), stream.get(), &counters);
    make(cn);
    make(weights);
    make(dweights);
    make(atm_partials);
    make(contracted);
    make(row_partials);
    make(dcontracted);
    make(atm_counters);
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
    view.atm_split = base + 12 * n;
    view.n = static_cast<std::int32_t>(n);
    return view;
  }

  cuda::GeometryView geometry_view(const Lattice& lattice) const {
    cuda::GeometryView view{};
    view.n = static_cast<std::int32_t>(numbers.size());
    view.pos = packed.data();
    std::memcpy(view.cell, lattice.cell.data(), sizeof(view.cell));
    return view;
  }

  // Run the pipeline. With ``debug`` non-null, intermediates are downloaded.
  HostResult run(const Geometry& geometry, unsigned properties, DebugResult* debug) {
    if (geometry.n_atoms != numbers.size()) {
      fail(ErrorCode::InvalidArgument, "expected " + std::to_string(numbers.size()) +
                                           " positions, got " + std::to_string(geometry.n_atoms));
    }
    const Lattice lattice = make_lattice(geometry);
    const bool want_gradient = (properties & kGradient) != 0;
    const bool want_virial = (properties & kVirial) != 0;
    const bool want_atoms = (properties & kAtomEnergies) != 0;
    const bool derivatives = want_gradient || want_virial || (debug != nullptr && debug_derivatives);
    HostResult result;
    const std::size_t n = numbers.size();
    result.has_gradient = want_gradient;
    result.has_virial = want_virial;
    result.has_atom_energies = want_atoms;
    if (want_gradient) result.gradient.assign(3 * n, 0.0);
    if (want_atoms) result.atom_energies.assign(n, 0.0);
    stats.evaluations += 1;
    if (n == 0) return result;

    DeviceGuard guard(device);
    cudaStream_t s = stream.get();

    StageTimer timer(options.collect_timings, s, events);
    timer.mark("start");
    // K0: upload and pack the geometry.
    staging.reserve(3 * n);
    std::memcpy(staging.data(), geometry.positions, 3 * n * sizeof(double));
    positions_dev.upload_async(staging.data(), 3 * n);
    packed.resize(n);
    cuda::launch_pack_positions(positions_dev.data(), species_dev.data(),
                                static_cast<std::int32_t>(n), packed.data(), s);
    const cuda::GeometryView g = geometry_view(lattice);

    // Neighbour graph: reuse while every atom stays within half the skin of
    // its position at the last build (and nothing else relevant changed).
    const std::array<double, cuda::kViewCount> radius{
        parameters.cutoffs.cn + options.skin,
        parameters.pair_active() ? parameters.cutoffs.disp2 + options.skin : 0.0,
        parameters.atm_active() ? parameters.cutoffs.disp3 + options.skin : 0.0};
    timer.mark("upload");
    if (needs_rebuild(geometry, lattice, radius)) {
      build_graph(geometry, lattice, g, radius, result.info);
    } else {
      result.info.builder = last_builder;
    }
    result.info.cn_edges = graph.edges(cuda::kViewCn);
    result.info.pair_edges = graph.edges(cuda::kViewPair);
    result.info.atm_edges = graph.edges(cuda::kViewAtm);
    const cuda::GraphView gv = graph.view();
    const cuda::ModelView mv = model.view(parameters);


    timer.mark("neighbors");
    // K0b: reset every accumulator of this evaluation in one call.
    arena.resize(13 * n);
    arena.zero_async();
    const cuda::AccumView accum = accum_view();

    // K1, K2: coordination numbers and reference weights.
    cn.resize(n);
    weights.resize(cuda::kRefPad * n);
    dweights.resize(cuda::kRefPad * n);
    if (optimized()) {
      row_partials.resize(static_cast<std::size_t>(
          std::max({graph.cn_items() * 9, graph.pair_items() * 11, std::int64_t{1}})));
      cuda::launch_cn_items(g, mv, gv, graph.cn_item_prefix(), graph.cn_items(),
                            row_partials.data(), cn.data(), s);
    } else {
      cuda::launch_cn(g, mv, gv, cn.data(), s);
    }
    cuda::launch_weights(mv, g, cn.data(), weights.data(), derivatives ? dweights.data() : nullptr, s);
    cuda::ContractedView cv{nullptr, nullptr};
    if (optimized()) {
      const std::size_t size = n * static_cast<std::size_t>(mv.n_species) * cuda::kRefPad;
      contracted.resize(size);
      dcontracted.resize(derivatives ? size : 0);
      cuda::launch_contract(mv, g, weights.data(), derivatives ? dweights.data() : nullptr,
                            contracted.data(), derivatives ? dcontracted.data() : nullptr, s);
      cv = {contracted.data(), derivatives ? dcontracted.data() : nullptr};
    }

    timer.mark("cn_weights");
    // K3: pair term.
    if (parameters.pair_active()) {
      if (optimized()) {
        cuda::launch_pair_items(g, mv, gv, graph.pair_item_prefix(), graph.pair_items(),
                                weights.data(), derivatives ? dweights.data() : nullptr, cv, accum,
                                row_partials.data(), s);
      } else {
        cuda::launch_pair(g, mv, gv, weights.data(), derivatives ? dweights.data() : nullptr, cv,
                          accum, s);
      }
    }
    timer.mark("pair");
    // K4: ATM triangles in bounded launches.
    if (parameters.atm_active()) run_atm(g, mv, gv, cv, derivatives, want_atoms, accum, result.info);
    if (debug != nullptr) download_debug(*debug, derivatives, accum);
    timer.mark("atm");
    // K5: CN-mediated derivatives.
    if (derivatives && optimized()) {
      cuda::launch_cn_reverse_items(g, mv, gv, graph.cn_item_prefix(), graph.cn_items(), accum,
                                    row_partials.data(), s);
    } else if (derivatives) {
      cuda::launch_cn_reverse(g, mv, gv, accum, false, s);
    }

    timer.mark("cn_reverse");
    // K6: reductions and transfers.
    cuda::launch_column_sums(accum.e_pair, kSumColumns, n, n, partials.data(), sums.data(), s);
    sums.download_async(host_sums.data(), kSumColumns);
    if (want_gradient) {
      host_gradient.reserve(3 * n);
      cuda::copy_to_host_async(accum.grad, host_gradient.data(), 3 * n, s);
    }
    if (want_atoms) {
      host_atoms.reserve(2 * n);
      cuda::copy_to_host_async(accum.e_pair, host_atoms.data(), n, s);
      // The canonical ATM kernel books each triangle to its anchor; the
      // per-atom split (1/3 per vertex, as upstream) is accumulated separately.
      cuda::copy_to_host_async(optimized() ? accum.atm_split : accum.e_atm, host_atoms.data() + n, n, s);
    }
    timer.mark("reduce");
    stream.synchronize();
    timer.collect(result.info.stage_ms);

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

  bool needs_rebuild(const Geometry& geometry, const Lattice& lattice,
                     const std::array<double, cuda::kViewCount>& radius) const {
    if (!graph_valid || graph_cell != geometry.cell || graph_pbc != geometry.pbc ||
        graph_radius != radius || !(options.skin > 0.0)) {
      return true;
    }
    for (int k = 0; k < 3; ++k) {
      // A lattice-vector wrap of an atom must exceed the skin to be detected.
      if (geometry.pbc[static_cast<std::size_t>(k)] &&
          options.skin >= 2.0 * lattice.spacing[static_cast<std::size_t>(k)]) {
        return true;
      }
    }
    double max_d2 = 0.0;
    for (std::size_t i = 0; i < 3 * geometry.n_atoms; i += 3) {
      const double dx = geometry.positions[i] - graph_positions[i];
      const double dy = geometry.positions[i + 1] - graph_positions[i + 1];
      const double dz = geometry.positions[i + 2] - graph_positions[i + 2];
      max_d2 = std::max(max_d2, dx * dx + dy * dy + dz * dz);
    }
    return 4.0 * max_d2 >= options.skin * options.skin;
  }

  void build_graph(const Geometry& geometry, const Lattice& lattice, const cuda::GeometryView& g,
                   const std::array<double, cuda::kViewCount>& radius, EvalInfo& info) {
    const std::size_t n = numbers.size();
    graph_valid = false;
    cuda::BandSpec bands = cuda::make_bands(radius);
    bands.canonical_atm = optimized() ? 1 : 0;
    const double rmax = std::max({radius[0], radius[1], radius[2]});
    cuda::ImageBox box;
    const std::array<int, 3> range = image_ranges(lattice, geometry.positions, n, rmax);
    box.images = 1;
    for (int k = 0; k < 3; ++k) {
      box.range[k] = range[static_cast<std::size_t>(k)];
      box.images = checked_mul<long long>(box.images, 2LL * box.range[k] + 1, "image count");
    }
    const long long candidates = checked_mul<long long>(
        checked_mul<long long>(box.images, static_cast<long long>(n), "candidate count"),
        static_cast<long long>(n), "candidate count");
    const bool brute = options.builder == NeighborBuilder::BruteForce ||
                       (options.builder == NeighborBuilder::Auto && candidates <= kBruteForceLimit);
    const double min_r2 =
        brute ? graph.build_brute_force(g, box, bands)
              : graph.build_cell_bins(g, make_grid(lattice, geometry.positions, n, rmax), bands);
    last_builder = brute ? "brute_force" : "cell_bins";
    stats.rebuilds += 1;
    if (options.reject_coincident && min_r2 < math::kCnExclusionR2) {
      fail(ErrorCode::CoincidentSites,
           "two distinct lattice sites are closer than 1e-6 bohr; the input geometry is unphysical");
    }
    graph_positions.assign(geometry.positions, geometry.positions + 3 * n);
    graph_cell = geometry.cell;
    graph_pbc = geometry.pbc;
    graph_radius = radius;
    graph_valid = true;
    info.rebuilt = true;
    info.builder = last_builder;
  }

  void run_atm(const cuda::GeometryView& g, const cuda::ModelView& mv, const cuda::GraphView& gv,
               const cuda::ContractedView& cv, bool derivatives, bool split_atoms,
               const cuda::AccumView& accum, EvalInfo& info) {
    const long long items = graph.atm_items();
    if (items == 0) return;
    cudaStream_t s = stream.get();
    // Bound the candidate triangles per launch (about 2^30) so a single
    // launch stays short even on display GPUs with a watchdog.
    const double mean_row = static_cast<double>(graph.edges(cuda::kViewAtm)) /
                            static_cast<double>(std::max<std::size_t>(1, numbers.size()));
    const double work_per_item = 0.5 * cuda::kAtmItemWidth * std::max(1.0, mean_row);
    const long long per_launch = std::clamp<long long>(
        static_cast<long long>(1073741824.0 / work_per_item), 1024, 1 << 20);
    const long long chunk =
        std::min(items, options.atm_items_per_launch > 0 ? options.atm_items_per_launch : per_launch);
    atm_partials.resize(static_cast<std::size_t>(7 * chunk));
    atm_counters.resize(3);
    atm_counters.zero_async();
    for (long long begin = 0; begin < items; begin += chunk) {
      const long long count = std::min(chunk, items - begin);
      const double* dw = derivatives ? dweights.data() : nullptr;
      if (optimized()) {
        cuda::launch_atm_canonical(g, mv, gv, graph.atm_item_prefix(), begin, count, weights.data(),
                                   dw, cv, accum, atm_partials.data(), atm_counters.data(),
                                   split_atoms, s);
      } else {
        cuda::launch_atm(g, mv, gv, graph.atm_item_prefix(), begin, count, weights.data(), dw,
                         accum, atm_partials.data(), atm_counters.data(), s);
      }
    }
    host_counters.reserve(3);
    atm_counters.download_async(host_counters.data(), 3);
    stream.synchronize();
    info.atm_candidates = host_counters.data()[0];
    info.atm_accepted = host_counters.data()[1];
    info.atm_c6_skipped = host_counters.data()[2];
  }

  void download_debug(DebugResult& debug, bool derivatives, const cuda::AccumView& accum) {
    const std::size_t n = numbers.size();
    cudaStream_t s = stream.get();
    debug.cn.resize(n);
    debug.weights.resize(cuda::kRefPad * n);
    cn.download_async(debug.cn.data(), n);
    weights.download_async(debug.weights.data(), debug.weights.size());
    if (derivatives) {
      debug.dweights.resize(cuda::kRefPad * n);
      debug.dEdcn.resize(n);
      dweights.download_async(debug.dweights.data(), debug.dweights.size());
      cuda::copy_to_host_async(accum.g, debug.dEdcn.data(), n, s);
    }
    std::vector<long long> row_ptr;
    std::vector<int4> edges;
    graph.download(row_ptr, edges);
    debug.edge_i.clear();
    debug.edge_j.clear();
    debug.edge_shift.clear();
    for (std::size_t i = 0; i < n; ++i) {
      for (long long e = row_ptr[i]; e < row_ptr[i + 1]; ++e) {
        const int4 edge = edges[static_cast<std::size_t>(e)];
        debug.edge_i.push_back(static_cast<std::int32_t>(i));
        debug.edge_j.push_back(edge.x);
        debug.edge_shift.insert(debug.edge_shift.end(), {edge.y, edge.z, edge.w});
      }
    }
    stream.synchronize();
  }

  bool debug_derivatives = false;
  bool optimized() const noexcept { return options.kernels == KernelVariant::Optimized; }
  std::vector<cuda::Event> events;
  bool graph_valid = false;
  std::vector<double> graph_positions;
  std::array<double, 9> graph_cell{};
  std::array<bool, 3> graph_pbc{};
  std::array<double, cuda::kViewCount> graph_radius{};
  const char* last_builder = "none";
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
  cuda::NeighborGraph graph;
  DeviceBuffer<double> cn;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> dweights;
  DeviceBuffer<double> atm_partials;
  DeviceBuffer<double> contracted;
  DeviceBuffer<double> row_partials;
  DeviceBuffer<double> dcontracted;
  DeviceBuffer<unsigned long long> atm_counters;
  PinnedBuffer<unsigned long long> host_counters;
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
    return impl_->run(geometry, properties, nullptr);
  } catch (const Error& error) {
    impl_->graph_valid = false;
    if (error.code() == ErrorCode::CudaRuntime) impl_->poisoned = true;
    throw;
  }
}

DebugResult D3Context::debug(const Geometry& geometry, bool derivatives) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  if (impl_->poisoned) fail(ErrorCode::ContextPoisoned, "an earlier CUDA error invalidated this context");
  DebugResult debug;
  impl_->debug_derivatives = derivatives;
  try {
    impl_->run(geometry, derivatives ? kEnergy | kGradient : kEnergy, &debug);
  } catch (const Error& error) {
    impl_->graph_valid = false;
    if (error.code() == ErrorCode::CudaRuntime) impl_->poisoned = true;
    throw;
  }
  return debug;
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

void D3Context::invalidate_neighbors() noexcept { impl_->graph_valid = false; }

std::size_t D3Context::n_atoms() const noexcept { return impl_->numbers.size(); }
const D3Parameters& D3Context::parameters() const noexcept { return impl_->parameters; }
const ContextOptions& D3Context::options() const noexcept { return impl_->options; }

ContextStats D3Context::stats() const {
  ContextStats stats = impl_->stats;
  stats.allocations = impl_->counters.allocations;
  stats.device_bytes = impl_->counters.device_bytes;
  stats.capacity_edges = impl_->graph.capacity();
  return stats;
}

void D3Context::testing_set_allocation_limit(std::size_t bytes) { impl_->counters.limit = bytes; }

void D3Context::testing_hold(double seconds) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
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
