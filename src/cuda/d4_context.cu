// D4Context: EEQ charges, D4 dispersion, and ATM on the GPU.
//
// Stage order (as dftd4 get_dispersion): D4 and EEQ coordination numbers ->
// EEQ charges -> charge-scaled weights and C6 -> pair term -> adjoint of the
// charge equations -> ATM with neutral C6 -> charge and CN reverse passes ->
// reductions. Any periodic flag makes the system fully periodic over the cell.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

#include "../common/d3_math.hpp"
#include "../core/ewald.hpp"
#include "../core/lattice.hpp"
#include "d4_kernels.cuh"
#include "d4_model.cuh"
#include "kernels.cuh"
#include "neighbor.cuh"
#include "pipeline.cuh"
#include "resources.cuh"
#include "vanderfast/d4_context.hpp"
#include "vanderfast/error.hpp"

namespace vanderfast {

using cuda::DeviceBuffer;
using cuda::DeviceGuard;
using cuda::PinnedBuffer;
using cuda::StageTimer;

namespace {

constexpr int kSumColumns = 8;  // e_pair, e_atm, six virial components
constexpr int kArenaColumns = 15;  // e_pair, e_atm, vir[6], g, grad[3], atm_split, gq, spare
// MINRES launches are sized to stream about this many matrix bytes, so a
// single launch stays well below display watchdog limits.
constexpr double kMinresBytesPerLaunch = 1.4e11;

enum class SolveMethod { SmallLu, Lu, Minres };

Geometry promoted(const Geometry& geometry) {
  Geometry out = geometry;
  const bool periodic = geometry.pbc[0] || geometry.pbc[1] || geometry.pbc[2];
  out.pbc = {periodic, periodic, periodic};
  return out;
}

}  // namespace

struct D4Context::Impl {
  Impl(const std::int32_t* numbers_in, std::size_t n, const D4Parameters& params,
       const ContextOptions& opts, double total_charge, const EeqOptions& eeq_options,
       std::shared_ptr<const D4ReferenceData> ref)
      : device(cuda::checked_device(opts.device)),
        options(opts),
        eeq(eeq_options),
        parameters(params),
        charge(total_charge),
        reference(std::move(ref)) {
    if (!reference) fail(ErrorCode::InvalidArgument, "reference data must not be null");
    parameters.validate();
    eeq.validate();
    if (!std::isfinite(charge)) fail(ErrorCode::InvalidArgument, "the total charge must be finite");
    if (!(options.skin >= 0.0) || !std::isfinite(options.skin)) {
      fail(ErrorCode::InvalidArgument, "skin must be finite and non-negative");
    }
    checked_int32(static_cast<std::int64_t>(n), "atom count");
    numbers.assign(numbers_in, numbers_in + n);
    for (std::int32_t z : numbers) {
      if (!reference->supports(z)) {
        fail(ErrorCode::UnsupportedElement, "atomic number " + std::to_string(z) +
                                                " is not supported by D4 (1.." +
                                                std::to_string(reference->max_z()) + ")");
      }
    }
    unique.assign(numbers.begin(), numbers.end());
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
    model = cuda::D4DeviceModel(*reference, unique, parameters, pool.get(), stream.get(), &counters);
    neighbors = cuda::NeighborCache(pool.get(), stream.get(), &counters);
    auto make = [&](auto& buffer) {
      using Buffer = std::decay_t<decltype(buffer)>;
      buffer = Buffer(pool.get(), stream.get(), &counters);
    };
    for (DeviceBuffer<double>* buffer :
         {&cn, &cn_eeq, &dxfac, &rhs, &nu, &kmat, &qx, &adjoint_rhs, &minres_work, &minres_partials,
          &w, &dw_cn, &dw_q, &w0, &dw0, &u, &du, &duq, &u0, &du0,
          &row_partials, &atm_partials, &positions_dev, &eeq_positions_dev, &arena, &partials, &sums,
          &dtrans, &rtrans, &ws_candidates, &self_diag, &self_vir, &recip_e, &trans_sorted, &sf_c, &sf_s,
          &reverse_scratch}) {
      make(*buffer);
    }
    make(atm_counters);
    make(species_dev);
    make(packed);
    make(eeq_packed);
    make(pivots);
    make(status);
    make(minres_state);
    minres_state.resize(2);
    minres_block_count = cuda::minres_blocks(static_cast<int>(n), device);
    host_minres.reserve(2);
    species_dev.upload_async(species.data(), species.size());
    partials.resize(static_cast<std::size_t>(cuda::kReduceBlocks) * kSumColumns);
    sums.resize(kSumColumns);
    host_sums.reserve(kSumColumns);
    host_status.reserve(1);
    stream.synchronize();
  }

  ~Impl() {
    if (stream.get() != nullptr) {
      cudaSetDevice(device);
      cudaStreamSynchronize(stream.get());
    }
  }

  bool optimized() const noexcept { return options.kernels == KernelVariant::Optimized; }

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
    view.gq = base + 13 * n;
    view.n = static_cast<std::int32_t>(n);
    return view;
  }

  cuda::GeometryView geometry_view(const Lattice& lattice, const double4_32a* pos) const {
    cuda::GeometryView view{};
    view.n = static_cast<std::int32_t>(numbers.size());
    view.pos = pos;
    std::memcpy(view.cell, lattice.cell.data(), sizeof(view.cell));
    return view;
  }

  // Lattice tables of the periodic Coulomb matrix, rebuilt when the cell changes.
  cuda::EwaldView ewald_view(const Geometry& geometry, EeqInfo& info) {
    cuda::EwaldView view{};
    const bool periodic = geometry.pbc[0];
    info.ewald = periodic ? (eeq.ewald == EwaldMode::Upstream ? "upstream" : "converged") : "none";
    if (!periodic) {
      view.mode = 0;
      return view;
    }
    if (!ewald_valid || ewald_cell != geometry.cell || ewald_mode != eeq.ewald) build_ewald(geometry);
    view.mode = eeq.ewald == EwaldMode::Upstream ? 1 : 2;
    view.alpha = setup.alpha;
    view.volume = setup.volume;
    view.real_cutoff = setup.real_cutoff;
    view.n_trans = static_cast<int>(setup.n_trans());
    view.n_recip = static_cast<int>(setup.n_recip());
    view.dtrans = dtrans.data();
    view.rtrans = rtrans.data();
    view.ws_candidates = ws_candidates.data();
    view.self_diag = self_diag.data();
    view.self_vir = self_vir.data();
    view.recip_e = recip_e.data();
    view.trans_sorted = trans_sorted.data();
    info.alpha = setup.alpha;
    info.n_trans = static_cast<std::int64_t>(setup.n_trans());
    info.n_recip = static_cast<std::int64_t>(setup.n_recip());
    return view;
  }

  void build_ewald(const Geometry& geometry) {
    const D4SpeciesTables& t = model.host();
    if (eeq.ewald == EwaldMode::Upstream) {
      setup = upstream_ewald(geometry.cell);
    } else {
      // Wrapped positions lie in the cell, so the span of any pair displacement
      // is bounded by the cell diagonals.
      double span = 0.0;
      for (int a = -1; a <= 1; a += 2) {
        for (int b = -1; b <= 1; b += 2) {
          double d[3];
          for (int c = 0; c < 3; ++c) {
            d[c] = geometry.cell[static_cast<std::size_t>(c)] + a * geometry.cell[static_cast<std::size_t>(3 + c)] +
                   b * geometry.cell[static_cast<std::size_t>(6 + c)];
          }
          span = std::max(span, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        }
      }
      const double rad_max = *std::max_element(t.rad.begin(), t.rad.end());
      setup = converged_ewald(geometry.cell, 0.0, 1.0 / std::sqrt(2.0 * rad_max * rad_max), span);
    }
    const std::size_t s_count = t.z.size();
    std::vector<double> diag(s_count);
    std::vector<double> vir(6 * s_count);
    const double sqrt2pi = std::sqrt(2.0 / 3.141592653589793);
    const double sqrtpi = std::sqrt(3.141592653589793);
    for (std::size_t s = 0; s < s_count; ++s) {
      const double gam = 1.0 / std::sqrt(2.0 * t.rad[s] * t.rad[s]);
      const double images = ewald_self_value(setup, gam);
      diag[s] = images + (t.eta[s] + sqrt2pi / t.rad[s] - 2 * setup.alpha / sqrtpi);
      double ds[9];
      ewald_self_derivative(setup, gam, ds);
      const double voigt[6] = {ds[0], ds[4], ds[8], ds[5], ds[2], ds[1]};
      std::copy(voigt, voigt + 6, vir.begin() + static_cast<std::ptrdiff_t>(6 * s));
    }
    // Optimised kernels: reciprocal weights and translations sorted by length.
    const std::size_t n_recip = setup.n_recip();
    std::vector<double> weights(n_recip);
    const double fac = 4 * 3.141592653589793 / setup.volume;
    for (std::size_t k = 0; k < n_recip; ++k) {
      const double* gk = setup.rtrans.data() + 3 * k;
      const double g2 = gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2];
      weights[k] = fac * std::exp(-0.25 * g2 / (setup.alpha * setup.alpha)) / g2;
    }
    const std::size_t n_trans = setup.n_trans();
    std::vector<double> lengths(n_trans);
    std::vector<std::size_t> order(n_trans);
    for (std::size_t k = 0; k < n_trans; ++k) {
      const double* tk = setup.dtrans.data() + 3 * k;
      lengths[k] = std::sqrt(tk[0] * tk[0] + tk[1] * tk[1] + tk[2] * tk[2]);
      order[k] = k;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return lengths[a] < lengths[b]; });
    std::vector<double> sorted(4 * n_trans);
    for (std::size_t k = 0; k < n_trans; ++k) {
      std::copy_n(setup.dtrans.data() + 3 * order[k], 3, sorted.data() + 4 * k);
      sorted[4 * k + 3] = lengths[order[k]];
    }
    recip_e.upload_async(weights.data(), weights.size());
    trans_sorted.upload_async(sorted.data(), sorted.size());
    dtrans.upload_async(setup.dtrans.data(), setup.dtrans.size());
    rtrans.upload_async(setup.rtrans.data(), setup.rtrans.size());
    if (!setup.ws_candidates.empty()) {
      ws_candidates.upload_async(setup.ws_candidates.data(), setup.ws_candidates.size());
    }
    self_diag.upload_async(diag.data(), diag.size());
    self_vir.upload_async(vir.data(), vir.size());
    stream.synchronize();  // host vectors above go out of scope
    ewald_valid = true;
    ewald_cell = geometry.cell;
    ewald_mode = eeq.ewald;
  }

  std::size_t matrix_limit() const {
    if (eeq.max_matrix_bytes > 0) return eeq.max_matrix_bytes;
    std::size_t free_bytes = 0;
    std::size_t total = 0;
    VF_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
    return total / 2;
  }

  SolveMethod choose_method(std::size_t n) const {
    const bool small = cuda::lu_small_fits(static_cast<int>(n), device);
    if (eeq.solver == ChargeSolver::Direct) {
      // The literal kernels keep the multi-kernel LU as an independent check.
      return small && optimized() ? SolveMethod::SmallLu : SolveMethod::Lu;
    }
    if (eeq.solver == ChargeSolver::Auto && small) return SolveMethod::SmallLu;
    if (minres_block_count == 0) return small ? SolveMethod::SmallLu : SolveMethod::Lu;
    return SolveMethod::Minres;
  }

  // MINRES on K x = b into slot ``slot`` of the solver state (0 forward, 1 adjoint).
  void solve_minres(int slot, const double* b, double* x, bool warm, std::size_t matrix_bytes) {
    const std::size_t np = numbers.size() + 1;
    cudaStream_t s = stream.get();
    minres_work.resize(9 * np);
    minres_partials.resize(static_cast<std::size_t>(cuda::kMinresSlots * minres_block_count));
    VF_CUDA_CHECK(cudaMemsetAsync(minres_state.data() + slot, 0, sizeof(cuda::MinresState), s));
    cuda::MinresArgs args{};
    args.k = kmat.data();
    args.b = b;
    args.x = x;
    args.work = minres_work.data();
    args.partials = minres_partials.data();
    args.state = minres_state.data() + slot;
    args.np = static_cast<int>(np);
    args.max_iterations = eeq.max_iterations;
    args.warm = warm ? 1 : 0;
    args.tolerance = eeq.tolerance;
    const double per_launch_estimate = kMinresBytesPerLaunch / static_cast<double>(matrix_bytes);
    const int per_launch = per_launch_estimate >= eeq.max_iterations
                               ? eeq.max_iterations
                               : std::max(16, static_cast<int>(per_launch_estimate));
    if (per_launch >= eeq.max_iterations) {
      cuda::launch_minres(args, minres_block_count, eeq.max_iterations, s);
      return;
    }
    for (int done = 0; done < eeq.max_iterations; done += per_launch) {
      cuda::launch_minres(args, minres_block_count, per_launch, s);
      cuda::copy_to_host_async(minres_state.data() + slot, host_minres.data() + slot, 1, s);
      stream.synchronize();
      if (host_minres.data()[slot].phase == 2) break;
    }
  }

  D4Result run(const Geometry& input, unsigned properties, D4Debug* debug) {
    if (input.n_atoms != numbers.size()) {
      fail(ErrorCode::InvalidArgument, "expected " + std::to_string(numbers.size()) +
                                           " positions, got " + std::to_string(input.n_atoms));
    }
    const Geometry geometry = promoted(input);
    const Lattice lattice = make_lattice(geometry);
    const bool periodic = geometry.pbc[0];
    const bool want_gradient = (properties & kGradient) != 0;
    const bool want_virial = (properties & kVirial) != 0;
    const bool want_atoms = (properties & kAtomEnergies) != 0;
    const bool derivatives = want_gradient || want_virial || (debug != nullptr && debug_derivatives);
    const bool atm = parameters.atm_active();
    D4Result result;
    const std::size_t n = numbers.size();
    result.has_gradient = want_gradient;
    result.has_virial = want_virial;
    result.has_atom_energies = want_atoms;
    if (want_gradient) result.gradient.assign(3 * n, 0.0);
    if (want_atoms) result.atom_energies.assign(n, 0.0);
    stats.evaluations += 1;
    if (n == 0) return result;

    const std::size_t np = n + 1;
    const std::size_t matrix_bytes = checked_mul(checked_mul(np, np, "EEQ matrix size"), sizeof(double),
                                                 "EEQ matrix size");
    if (matrix_bytes > matrix_limit()) {
      fail(ErrorCode::OutOfDeviceMemory,
           "the EEQ matrix of " + std::to_string(n) + " atoms needs " + std::to_string(matrix_bytes) +
               " bytes, above the limit of " + std::to_string(matrix_limit()) + " bytes");
    }
    result.eeq.matrix_bytes = matrix_bytes;
    const SolveMethod method = choose_method(n);
    result.eeq.solver = method == SolveMethod::Minres ? "minres" : "direct";

    DeviceGuard guard(device);
    cudaStream_t s = stream.get();
    StageTimer timer(options.collect_timings, s, events);
    timer.mark("start");
    // K0: upload and pack the geometry; periodic charges use wrapped positions.
    staging.reserve(3 * n);
    std::memcpy(staging.data(), geometry.positions, 3 * n * sizeof(double));
    positions_dev.upload_async(staging.data(), 3 * n);
    packed.resize(n);
    cuda::launch_pack_positions(positions_dev.data(), species_dev.data(), static_cast<std::int32_t>(n),
                                packed.data(), s);
    std::vector<double> wrapped;
    const double4_32a* eeq_pos = packed.data();
    if (periodic) {
      wrapped.assign(geometry.positions, geometry.positions + 3 * n);
      wrap_to_central_cell(wrapped, geometry.cell);
      eeq_staging.reserve(3 * n);
      std::memcpy(eeq_staging.data(), wrapped.data(), 3 * n * sizeof(double));
      eeq_positions_dev.upload_async(eeq_staging.data(), 3 * n);
      eeq_packed.resize(n);
      cuda::launch_pack_positions(eeq_positions_dev.data(), species_dev.data(),
                                  static_cast<std::int32_t>(n), eeq_packed.data(), s);
      eeq_pos = eeq_packed.data();
    }
    const cuda::GeometryView g = geometry_view(lattice, packed.data());
    const cuda::GeometryView ge = geometry_view(lattice, eeq_pos);
    const cuda::EwaldView ev = ewald_view(geometry, result.eeq);

    const std::array<double, cuda::kViewCount> radius{
        parameters.cn_radius() + options.skin,
        parameters.pair_active() ? parameters.cutoffs.disp2 + options.skin : 0.0,
        atm ? parameters.cutoffs.disp3 + options.skin : 0.0};
    timer.mark("upload");
    neighbors.ensure(geometry, lattice, g, radius, options, optimized(), result.info, stats);
    const cuda::NeighborGraph& graph = neighbors.graph;
    result.info.cn_edges = graph.edges(cuda::kViewCn);
    result.info.pair_edges = graph.edges(cuda::kViewPair);
    result.info.atm_edges = graph.edges(cuda::kViewAtm);
    const cuda::GraphView gv = graph.view();
    const cuda::ModelView mv = model.view(parameters);
    const cuda::D4View dv = model.d4_view(parameters);
    timer.mark("neighbors");

    arena.resize(kArenaColumns * n);
    arena.zero_async();
    const cuda::AccumView accum = accum_view();
    status.resize(1);
    status.zero_async();

    // K1: coordination numbers and the EEQ right-hand side.
    cn.resize(n);
    cn_eeq.resize(n);
    dxfac.resize(n);
    rhs.resize(np);
    row_partials.resize(static_cast<std::size_t>(
        std::max({graph.cn_items() * 9, graph.pair_items() * 12, std::int64_t{1}})));
    cuda::launch_d4_cn_items(g, mv, dv, gv, graph.cn_item_prefix(), graph.cn_items(), charge,
                             row_partials.data(), cn.data(), cn_eeq.data(), rhs.data(), dxfac.data(), s);
    timer.mark("cn");

    // K2: the EEQ charges q (the first n entries of the solution).
    kmat.resize(np * np);
    pivots.resize(np);
    const bool structure = optimized() && periodic;
    if (structure) {
      const std::size_t sf_size = n * static_cast<std::size_t>(ev.n_recip);
      sf_c.resize(sf_size);
      sf_s.resize(sf_size);
      cuda::launch_structure_factors(ge, ev, sf_c.data(), sf_s.data(), s);
    }
    if (optimized()) {
      cuda::launch_eeq_assemble_tiles(ge, dv, ev, structure ? sf_c.data() : nullptr,
                                      structure ? sf_s.data() : nullptr, kmat.data(), s);
    } else {
      cuda::launch_eeq_assemble(ge, dv, ev, kmat.data(), s);
    }
    timer.mark("eeq_setup");
    const double* q = rhs.data();
    bool warm_adjoint = false;
    if (method == SolveMethod::Minres) {
      qx.resize(np);
      const bool warm = eeq.warm_start && warm_charges;
      warm_adjoint = eeq.warm_start && warm_nu;
      result.eeq.warm_started = warm;
      warm_charges = false;
      warm_nu = false;
      solve_minres(0, rhs.data(), qx.data(), warm, matrix_bytes);
      q = qx.data();
    } else if (method == SolveMethod::SmallLu) {
      cuda::launch_lu_small_factor_solve(kmat.data(), static_cast<int>(n), pivots.data(), rhs.data(),
                                         status.data(), s);
    } else {
      cuda::launch_lu_factor(kmat.data(), static_cast<int>(n), pivots.data(), status.data(), s);
      cuda::launch_lu_solve(kmat.data(), static_cast<int>(n), pivots.data(), rhs.data(), s);
    }
    timer.mark("eeq_solve");

    // K3: weights and species-contracted weights (charged and neutral).
    const std::size_t wsize = cuda::kRefPad * n;
    w.resize(wsize);
    dw_cn.resize(derivatives ? wsize : 0);
    dw_q.resize(derivatives ? wsize : 0);
    w0.resize(atm ? wsize : 0);
    dw0.resize(atm && derivatives ? wsize : 0);
    cuda::launch_d4_weights(dv, g, cn.data(), q, w.data(), derivatives ? dw_cn.data() : nullptr,
                            derivatives ? dw_q.data() : nullptr, atm ? w0.data() : nullptr,
                            atm && derivatives ? dw0.data() : nullptr, s);
    const std::size_t usize = n * static_cast<std::size_t>(mv.n_species) * cuda::kRefPad;
    u.resize(usize);
    du.resize(derivatives ? usize : 0);
    duq.resize(derivatives ? usize : 0);
    cuda::launch_contract(mv, g, w.data(), derivatives ? dw_cn.data() : nullptr, u.data(),
                          derivatives ? du.data() : nullptr, s);
    if (derivatives) cuda::launch_contract(mv, g, dw_q.data(), nullptr, duq.data(), nullptr, s);
    cuda::ContractedView cv{u.data(), derivatives ? du.data() : nullptr, derivatives ? duq.data() : nullptr};
    cuda::ContractedView cv0{nullptr, nullptr};
    if (atm) {
      u0.resize(usize);
      du0.resize(derivatives ? usize : 0);
      cuda::launch_contract(mv, g, w0.data(), derivatives ? dw0.data() : nullptr, u0.data(),
                            derivatives ? du0.data() : nullptr, s);
      cv0 = {u0.data(), derivatives ? du0.data() : nullptr};
    }
    timer.mark("weights");

    // K4: pair term with dE/dCN and dE/dq.
    if (parameters.pair_active()) {
      cuda::launch_pair_items(g, mv, gv, graph.pair_item_prefix(), graph.pair_items(), w.data(),
                              derivatives ? dw_cn.data() : nullptr, cv, accum, row_partials.data(), s);
    }
    timer.mark("pair");

    // K5: adjoint of the charge equations, K [nu; lambda] = [dE/dq; 0].
    if (derivatives) {
      nu.resize(np);
      if (method == SolveMethod::Minres) {
        adjoint_rhs.resize(np);
        adjoint_rhs.zero_async();
        VF_CUDA_CHECK(
            cudaMemcpyAsync(adjoint_rhs.data(), accum.gq, n * sizeof(double), cudaMemcpyDeviceToDevice, s));
        solve_minres(1, adjoint_rhs.data(), nu.data(), warm_adjoint, matrix_bytes);
      } else {
        nu.zero_async();
        VF_CUDA_CHECK(cudaMemcpyAsync(nu.data(), accum.gq, n * sizeof(double), cudaMemcpyDeviceToDevice, s));
        if (method == SolveMethod::SmallLu) {
          cuda::launch_lu_small_solve(kmat.data(), static_cast<int>(n), pivots.data(), nu.data(), s);
        } else {
          cuda::launch_lu_solve(kmat.data(), static_cast<int>(n), pivots.data(), nu.data(), s);
        }
      }
    }
    timer.mark("eeq_adjoint");

    // K6: ATM with neutral C6 (its charge derivative does not enter dftd4's gradient).
    if (atm) run_atm(g, mv, gv, cv0, derivatives, want_atoms, accum, result.info);
    if (debug != nullptr) download_debug(*debug, derivatives, accum, q);
    timer.mark("atm");

    // K7: charge and CN reverse passes.
    if (derivatives) {
      if (optimized()) {
        reverse_scratch.resize(cuda::eeq_reverse_tiles_scratch(static_cast<int>(n), ev.n_recip));
        cuda::launch_eeq_reverse_tiles(ge, dv, ev, structure ? sf_c.data() : nullptr,
                                       structure ? sf_s.data() : nullptr, q, nu.data(), accum,
                                       reverse_scratch.data(), s);
      } else {
        cuda::launch_eeq_reverse(ge, dv, ev, q, nu.data(), accum, s);
      }
      timer.mark("eeq_reverse");
      cuda::launch_d4_cn_reverse_items(g, mv, dv, gv, graph.cn_item_prefix(), graph.cn_items(), accum,
                                       nu.data(), dxfac.data(), row_partials.data(), s);
    }
    timer.mark("cn_reverse");

    // K8: reductions and transfers.
    cuda::launch_column_sums(accum.e_pair, kSumColumns, n, n, partials.data(), sums.data(), s);
    sums.download_async(host_sums.data(), kSumColumns);
    status.download_async(host_status.data(), 1);
    if (method == SolveMethod::Minres) minres_state.download_async(host_minres.data(), derivatives ? 2 : 1);
    host_charges.reserve(n);
    cuda::copy_to_host_async(q, host_charges.data(), n, s);
    if (want_gradient) {
      host_gradient.reserve(3 * n);
      cuda::copy_to_host_async(accum.grad, host_gradient.data(), 3 * n, s);
    }
    if (want_atoms) {
      host_atoms.reserve(2 * n);
      cuda::copy_to_host_async(accum.e_pair, host_atoms.data(), n, s);
      cuda::copy_to_host_async(optimized() ? accum.atm_split : accum.e_atm, host_atoms.data() + n, n, s);
    }
    timer.mark("reduce");
    stream.synchronize();
    timer.collect(result.info.stage_ms);
    if (host_status.data()[0] != 0) {
      fail(ErrorCode::ChargeSystemSingular, "the EEQ charge equations are singular");
    }
    if (method == SolveMethod::Minres) check_minres(result.eeq, derivatives);

    result.charges.assign(host_charges.data(), host_charges.data() + n);
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

  // Record the MINRES diagnostics and raise on singular or unconverged solves.
  void check_minres(EeqInfo& info, bool adjoint) {
    const cuda::MinresState& forward = host_minres.data()[0];
    info.forward_iterations = forward.itn;
    info.restarts = forward.restarts;
    info.backward_error = forward.eta;
    const int solves = adjoint ? 2 : 1;
    for (int slot = 0; slot < solves; ++slot) {
      const cuda::MinresState& st = host_minres.data()[slot];
      const char* which = slot == 0 ? "charge" : "adjoint charge";
      if (st.status == 1) {
        fail(ErrorCode::ChargeSystemSingular,
             std::string("the EEQ ") + which + " equations are singular or ill-conditioned");
      }
      if (st.status == 2 || st.phase != 2) {
        fail(ErrorCode::ChargeSolverNotConverged,
             std::string("MINRES did not converge for the EEQ ") + which + " equations after " +
                 std::to_string(st.itn) + " iterations (backward error " + std::to_string(st.eta) + ")");
      }
    }
    if (adjoint) {
      info.adjoint_iterations = host_minres.data()[1].itn;
      info.restarts += host_minres.data()[1].restarts;
    }
    warm_charges = true;
    warm_nu = adjoint;
  }

  void run_atm(const cuda::GeometryView& g, const cuda::ModelView& mv, const cuda::GraphView& gv,
               const cuda::ContractedView& cv0, bool derivatives, bool split_atoms,
               const cuda::AccumView& accum, EvalInfo& info) {
    const cuda::NeighborGraph& graph = neighbors.graph;
    const long long items = graph.atm_items();
    if (items == 0) return;
    cudaStream_t s = stream.get();
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
      const double* d0 = derivatives ? dw0.data() : nullptr;
      if (optimized()) {
        cuda::launch_atm_canonical(g, mv, gv, graph.atm_item_prefix(), begin, count, w0.data(), d0, cv0,
                                   accum, atm_partials.data(), atm_counters.data(), split_atoms, s);
      } else {
        cuda::launch_atm(g, mv, gv, graph.atm_item_prefix(), begin, count, w0.data(), d0, accum,
                         atm_partials.data(), atm_counters.data(), s);
      }
    }
    host_counters.reserve(3);
    atm_counters.download_async(host_counters.data(), 3);
    stream.synchronize();
    info.atm_candidates = host_counters.data()[0];
    info.atm_accepted = host_counters.data()[1];
    info.atm_c6_skipped = host_counters.data()[2];
  }

  void download_debug(D4Debug& debug, bool derivatives, const cuda::AccumView& accum, const double* q) {
    const std::size_t n = numbers.size();
    const std::size_t wsize = cuda::kRefPad * n;
    cudaStream_t s = stream.get();
    debug.cn.resize(n);
    debug.cn_eeq.resize(n);
    debug.charges.resize(n);
    debug.weights.resize(wsize);
    cn.download_async(debug.cn.data(), n);
    cn_eeq.download_async(debug.cn_eeq.data(), n);
    cuda::copy_to_host_async(q, debug.charges.data(), n, s);
    w.download_async(debug.weights.data(), wsize);
    if (parameters.atm_active()) {
      debug.weights_q0.resize(wsize);
      w0.download_async(debug.weights_q0.data(), wsize);
    }
    if (derivatives) {
      debug.dweights_dcn.resize(wsize);
      debug.dweights_dq.resize(wsize);
      debug.dEdcn.resize(n);
      debug.dEdq.resize(n);
      dw_cn.download_async(debug.dweights_dcn.data(), wsize);
      dw_q.download_async(debug.dweights_dq.data(), wsize);
      cuda::copy_to_host_async(accum.g, debug.dEdcn.data(), n, s);
      cuda::copy_to_host_async(accum.gq, debug.dEdq.data(), n, s);
    }
    std::vector<long long> row_ptr;
    std::vector<int4> edges;
    neighbors.graph.download(row_ptr, edges);
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
  std::vector<cuda::Event> events;
  std::mutex mutex;
  bool poisoned = false;
  int device;
  ContextOptions options;
  EeqOptions eeq;
  D4Parameters parameters;
  double charge;
  std::shared_ptr<const D4ReferenceData> reference;
  std::vector<std::int32_t> numbers;
  std::vector<int> unique;
  std::vector<int> species;
  cuda::AllocationCounters counters;
  ContextStats stats;
  cuda::Stream stream;
  cuda::MemoryPool pool;
  cuda::D4DeviceModel model;
  cuda::NeighborCache neighbors;
  EwaldSetup setup;
  bool ewald_valid = false;
  std::array<double, 9> ewald_cell{};
  EwaldMode ewald_mode = EwaldMode::Upstream;
  DeviceBuffer<double> cn, cn_eeq, dxfac, rhs, nu, kmat;
  DeviceBuffer<double> w, dw_cn, dw_q, w0, dw0;
  DeviceBuffer<double> u, du, duq, u0, du0;
  DeviceBuffer<double> row_partials, atm_partials;
  DeviceBuffer<double> positions_dev, eeq_positions_dev;
  DeviceBuffer<double> arena, partials, sums;
  DeviceBuffer<double> dtrans, rtrans, ws_candidates, self_diag, self_vir, recip_e, trans_sorted;
  DeviceBuffer<double> sf_c, sf_s, reverse_scratch;
  DeviceBuffer<unsigned long long> atm_counters;
  DeviceBuffer<int> species_dev;
  DeviceBuffer<double4_32a> packed, eeq_packed;
  DeviceBuffer<int> pivots, status;
  DeviceBuffer<double> qx, adjoint_rhs, minres_work, minres_partials;
  DeviceBuffer<cuda::MinresState> minres_state;
  PinnedBuffer<cuda::MinresState> host_minres;
  int minres_block_count = 0;
  bool warm_charges = false;  // qx holds a converged solution for a warm start
  bool warm_nu = false;       // nu likewise
  PinnedBuffer<unsigned long long> host_counters;
  PinnedBuffer<double> staging, eeq_staging, host_sums, host_gradient, host_atoms, host_charges;
  PinnedBuffer<int> host_status;
};

D4Context::D4Context(const std::int32_t* numbers, std::size_t n_atoms, const D4Parameters& parameters,
                     const ContextOptions& options, double total_charge, const EeqOptions& eeq,
                     std::shared_ptr<const D4ReferenceData> reference)
    : impl_(std::make_unique<Impl>(numbers, n_atoms, parameters, options, total_charge, eeq,
                                   std::move(reference))) {}

D4Context::~D4Context() = default;
D4Context::D4Context(D4Context&&) noexcept = default;
D4Context& D4Context::operator=(D4Context&&) noexcept = default;

D4Result D4Context::evaluate(const Geometry& geometry, unsigned properties) {
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
    impl_->neighbors.invalidate();
    if (error.code() == ErrorCode::CudaRuntime) impl_->poisoned = true;
    throw;
  }
}

D4Debug D4Context::debug(const Geometry& geometry, bool derivatives) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  if (impl_->poisoned) fail(ErrorCode::ContextPoisoned, "an earlier CUDA error invalidated this context");
  D4Debug debug;
  impl_->debug_derivatives = derivatives;
  try {
    impl_->run(geometry, derivatives ? kEnergy | kGradient : kEnergy, &debug);
  } catch (const Error& error) {
    impl_->neighbors.invalidate();
    if (error.code() == ErrorCode::CudaRuntime) impl_->poisoned = true;
    throw;
  }
  return debug;
}

void D4Context::set_parameters(const D4Parameters& parameters) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  parameters.validate();
  DeviceGuard guard(impl_->device);
  impl_->parameters = parameters;
  impl_->model.update_parameters(*impl_->reference, parameters);
  impl_->ewald_valid = false;  // the self terms depend on the species tables
  impl_->stream.synchronize();
}

void D4Context::set_charge(double total_charge) {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) fail(ErrorCode::ContextBusy, "this context is busy");
  if (!std::isfinite(total_charge)) fail(ErrorCode::InvalidArgument, "the total charge must be finite");
  impl_->charge = total_charge;
}

void D4Context::invalidate_neighbors() noexcept { impl_->neighbors.invalidate(); }

std::size_t D4Context::n_atoms() const noexcept { return impl_->numbers.size(); }
const D4Parameters& D4Context::parameters() const noexcept { return impl_->parameters; }
double D4Context::charge() const noexcept { return impl_->charge; }

ContextStats D4Context::stats() const {
  ContextStats stats = impl_->stats;
  stats.allocations = impl_->counters.allocations;
  stats.device_bytes = impl_->counters.device_bytes;
  stats.capacity_edges = impl_->neighbors.graph.capacity();
  return stats;
}

void D4Context::testing_set_allocation_limit(std::size_t bytes) { impl_->counters.limit = bytes; }

}  // namespace vanderfast
