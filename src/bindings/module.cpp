// pybind11 bindings: the vanderfast._native extension (atomic units only).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../core/build_metadata.hpp"
#include "../core/d4_model.hpp"
#include "../core/lattice.hpp"
#include "../core/neighbor_cpu.hpp"
#include "vanderfast/context.hpp"
#include "vanderfast/d4_context.hpp"
#include "vanderfast/error.hpp"
#include "vanderfast/reference_data.hpp"

namespace py = pybind11;
using namespace vanderfast;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

void raise_python(const Error& error) {
  static const char* const names[] = {
      "InputError",            // InvalidArgument
      "ParameterError",        // InvalidParameter
      "InputError",            // UnsupportedElement
      "UnsupportedFeatureError",
      "InputError",            // InvalidGeometry
      "InputError",            // CoincidentSites
      "CudaUnavailableError",
      "CudaError",             // CudaRuntime
      "DeviceOutOfMemoryError",
      "InputError",            // SizeOverflow
      "ContextBusyError",
      "CudaError",             // ContextPoisoned
      "VanderfastError",       // Internal
      "ChargeSolverError",     // ChargeSystemSingular
      "ChargeSolverError",     // ChargeSolverNotConverged
  };
  const auto index = static_cast<std::size_t>(error.code());
  py::object errors = py::module_::import("vanderfast.errors");
  py::object type = errors.attr(names[index]);
  PyErr_SetString(type.ptr(), error.what());
}

template <class Context, class Result = decltype(std::declval<Context&>().evaluate(Geometry{}, 0u))>
Result run_evaluate(Context& context, const DoubleArray& positions, const DoubleArray& cell,
                    std::array<bool, 3> pbc, unsigned properties) {
  if (positions.ndim() != 2 || positions.shape(1) != 3) {
    throw Error(ErrorCode::InvalidArgument, "positions must have shape (N, 3)");
  }
  if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
    throw Error(ErrorCode::InvalidArgument, "cell must have shape (3, 3)");
  }
  // Copy the inputs so the GIL can be released without racing Python writers.
  std::vector<double> owned(positions.data(), positions.data() + positions.size());
  Geometry geometry;
  geometry.positions = owned.data();
  geometry.n_atoms = static_cast<std::size_t>(positions.shape(0));
  std::memcpy(geometry.cell.data(), cell.data(), 9 * sizeof(double));
  geometry.pbc = pbc;
  py::gil_scoped_release release;
  return context.evaluate(geometry, properties);
}

Geometry debug_geometry(const DoubleArray& positions, const DoubleArray& cell, std::array<bool, 3> pbc,
                        std::vector<double>& owned) {
  if (positions.ndim() != 2 || positions.shape(1) != 3 || cell.size() != 9) {
    throw Error(ErrorCode::InvalidArgument, "positions must be (N, 3) and cell (3, 3)");
  }
  owned.assign(positions.data(), positions.data() + positions.size());
  Geometry geometry;
  geometry.positions = owned.data();
  geometry.n_atoms = static_cast<std::size_t>(positions.shape(0));
  std::memcpy(geometry.cell.data(), cell.data(), 9 * sizeof(double));
  geometry.pbc = pbc;
  return geometry;
}

py::array_t<double> to_array(const std::vector<double>& values, std::vector<py::ssize_t> shape) {
  py::array_t<double> out(shape);
  if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(double));
  return out;
}

py::dict result_dict(const HostResult& result, std::size_t n) {
  py::dict out;
  out["energy"] = result.energy;
  out["energy_pair"] = result.energy_pair;
  out["energy_atm"] = result.energy_atm;
  out["gradient"] = result.has_gradient
                        ? py::object(to_array(result.gradient, {static_cast<py::ssize_t>(n), 3}))
                        : py::none();
  if (result.has_virial) {
    std::vector<double> virial(result.virial.begin(), result.virial.end());
    out["virial"] = to_array(virial, {3, 3});
  } else {
    out["virial"] = py::none();
  }
  out["atom_energies"] =
      result.has_atom_energies
          ? py::object(to_array(result.atom_energies, {static_cast<py::ssize_t>(n)}))
          : py::none();
  py::dict info;
  info["rebuilt"] = result.info.rebuilt;
  info["builder"] = result.info.builder;
  info["cn_edges"] = result.info.cn_edges;
  info["pair_edges"] = result.info.pair_edges;
  info["atm_edges"] = result.info.atm_edges;
  info["atm_candidates"] = result.info.atm_candidates;
  info["atm_accepted"] = result.info.atm_accepted;
  info["atm_c6_skipped"] = result.info.atm_c6_skipped;
  info["stage_ms"] = result.info.stage_ms;
  out["info"] = info;
  return out;
}

D3Parameters make_parameters(double s6, double s8, double a1, double a2, double s9, double alp,
                             double cutoff_cn, double cutoff_disp2, double cutoff_disp3) {
  D3Parameters p;
  p.s6 = s6;
  p.s8 = s8;
  p.a1 = a1;
  p.a2 = a2;
  p.s9 = s9;
  p.alp = alp;
  p.cutoffs.cn = cutoff_cn;
  p.cutoffs.disp2 = cutoff_disp2;
  p.cutoffs.disp3 = cutoff_disp3;
  return p;
}

D4Parameters make_d4_parameters(double s6, double s8, double a1, double a2, double s9, double alp,
                                double ga, double gc, double wf, double cutoff_cn, double cutoff_disp2,
                                double cutoff_disp3) {
  D4Parameters p;
  p.s6 = s6;
  p.s8 = s8;
  p.a1 = a1;
  p.a2 = a2;
  p.s9 = s9;
  p.alp = alp;
  p.ga = ga;
  p.gc = gc;
  p.wf = wf;
  p.cutoffs.cn = cutoff_cn;
  p.cutoffs.disp2 = cutoff_disp2;
  p.cutoffs.disp3 = cutoff_disp3;
  return p;
}

KernelVariant parse_kernels(const std::string& kernels) {
  if (kernels == "optimized") return KernelVariant::Optimized;
  if (kernels == "literal") return KernelVariant::Literal;
  throw Error(ErrorCode::InvalidArgument, "kernels must be 'optimized' or 'literal'");
}

NeighborBuilder parse_builder(const std::string& name) {
  if (name == "auto") return NeighborBuilder::Auto;
  if (name == "brute_force") return NeighborBuilder::BruteForce;
  if (name == "cell_bins") return NeighborBuilder::CellBins;
  throw Error(ErrorCode::InvalidArgument, "unknown neighbour builder '" + name + "'");
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "vanderfast CUDA backend (atomic units: bohr, hartree)";

  py::register_exception_translator([](std::exception_ptr pointer) {
    try {
      if (pointer) std::rethrow_exception(pointer);
    } catch (const Error& error) {
      raise_python(error);
    }
  });

  m.def("build_info", [] { return build_info(); });
  m.def("device_count", [] { return D3Context::device_count(); });
  m.def("devices", [] {
    py::list out;
    for (const DeviceInfo& info : D3Context::devices()) {
      py::dict d;
      d["index"] = info.index;
      d["name"] = info.name;
      d["compute_capability"] = std::to_string(info.compute_major) + "." + std::to_string(info.compute_minor);
      d["total_memory"] = info.total_memory;
      d["multiprocessors"] = info.multiprocessors;
      d["kernel_timeout"] = info.kernel_timeout;
      out.append(d);
    }
    return out;
  });
  m.def("reference_summary", [] {
    auto tables = ReferenceData::embedded();
    py::dict d;
    d["source"] = tables->source_id();
    d["sha256"] = tables->identity();
    d["max_z"] = tables->max_z();
    d["max_ref"] = tables->max_ref();
    return d;
  });
  m.def(
      "reference_value",
      [](const std::string& table, int zi, int zj, int a, int b) {
        auto tables = ReferenceData::embedded();
        if (table == "nref") return static_cast<double>(tables->nref(zi));
        if (table == "reference_cn") return tables->reference_cn(zi, a);
        if (table == "c6") return tables->c6(zi, zj, a, b);
        if (table == "rcov") return tables->rcov(zi);
        if (table == "r4r2") return tables->r4r2(zi);
        if (table == "rvdw") return tables->rvdw(zi, zj);
        throw Error(ErrorCode::InvalidArgument, "unknown table '" + table + "'");
      },
      py::arg("table"), py::arg("zi"), py::arg("zj") = 1, py::arg("a") = 0, py::arg("b") = 0);

  m.def(
      "enumerate_edges",
      [](const DoubleArray& positions, const DoubleArray& cell, std::array<bool, 3> pbc,
         double cutoff) {
        if (positions.ndim() != 2 || positions.shape(1) != 3 || cell.size() != 9) {
          throw Error(ErrorCode::InvalidArgument, "positions must be (N, 3) and cell (3, 3)");
        }
        Geometry geometry;
        geometry.positions = positions.data();
        geometry.n_atoms = static_cast<std::size_t>(positions.shape(0));
        std::memcpy(geometry.cell.data(), cell.data(), 9 * sizeof(double));
        geometry.pbc = pbc;
        const Lattice lattice = make_lattice(geometry);
        const std::vector<SiteEdge> edges =
            enumerate_edges(lattice, geometry.positions, geometry.n_atoms, cutoff);
        const auto count = static_cast<py::ssize_t>(edges.size());
        py::array_t<std::int32_t> i(count), j(count), shift({count, py::ssize_t{3}});
        py::array_t<double> r2(count);
        for (py::ssize_t e = 0; e < count; ++e) {
          const SiteEdge& edge = edges[static_cast<std::size_t>(e)];
          i.mutable_at(e) = edge.i;
          j.mutable_at(e) = edge.j;
          shift.mutable_at(e, 0) = edge.n0;
          shift.mutable_at(e, 1) = edge.n1;
          shift.mutable_at(e, 2) = edge.n2;
          r2.mutable_at(e) = edge.r2;
        }
        py::dict out;
        out["i"] = i;
        out["j"] = j;
        out["shift"] = shift;
        out["r2"] = r2;
        return out;
      },
      py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::arg("cutoff"));

  py::class_<D3Context>(m, "Context")
      .def(py::init([](const IntArray& numbers, double s6, double s8, double a1, double a2, double s9,
                       double alp, double cutoff_cn, double cutoff_disp2, double cutoff_disp3,
                       int device, double skin, const std::string& builder,
                       const py::object& reference_data, const std::string& reference_identity,
                       bool collect_timings, long long atm_items_per_launch,
                       const std::string& kernels) {
             if (numbers.ndim() != 1) {
               throw Error(ErrorCode::InvalidArgument, "numbers must be one-dimensional");
             }
             const D3Parameters parameters =
                 make_parameters(s6, s8, a1, a2, s9, alp, cutoff_cn, cutoff_disp2, cutoff_disp3);
             ContextOptions options;
             options.device = device;
             options.skin = skin;
             options.builder = parse_builder(builder);
             options.collect_timings = collect_timings;
             if (atm_items_per_launch < 0) {
               throw Error(ErrorCode::InvalidArgument, "atm_items_per_launch must be >= 0");
             }
             options.atm_items_per_launch = atm_items_per_launch;
             if (kernels == "optimized") {
               options.kernels = KernelVariant::Optimized;
             } else if (kernels == "literal") {
               options.kernels = KernelVariant::Literal;
             } else {
               throw Error(ErrorCode::InvalidArgument, "kernels must be 'optimized' or 'literal'");
             }
             std::shared_ptr<const ReferenceData> reference = ReferenceData::embedded();
             if (!reference_data.is_none()) {
               const std::string blob = reference_data.cast<py::bytes>();
               reference = ReferenceData::from_bytes(blob.data(), blob.size(), reference_identity);
             }
             std::vector<std::int32_t> z(numbers.data(), numbers.data() + numbers.size());
             py::gil_scoped_release release;
             return D3Context(z.data(), z.size(), parameters, options, std::move(reference));
           }),
           py::arg("numbers"), py::kw_only(), py::arg("s6"), py::arg("s8"), py::arg("a1"),
           py::arg("a2"), py::arg("s9"), py::arg("alp"), py::arg("cutoff_cn"),
           py::arg("cutoff_disp2"), py::arg("cutoff_disp3"), py::arg("device") = 0,
           py::arg("skin") = 0.0, py::arg("builder") = "auto",
           py::arg("reference_data") = py::none(), py::arg("reference_identity") = "",
           py::arg("collect_timings") = false, py::arg("atm_items_per_launch") = 0,
           py::arg("kernels") = "optimized")
      .def(
          "evaluate",
          [](D3Context& context, const DoubleArray& positions, const DoubleArray& cell,
             std::array<bool, 3> pbc, bool gradient, bool virial, bool atom_energies) {
            unsigned properties = kEnergy;
            if (gradient) properties |= kGradient;
            if (virial) properties |= kVirial;
            if (atom_energies) properties |= kAtomEnergies;
            const HostResult result = run_evaluate(context, positions, cell, pbc, properties);
            return result_dict(result, context.n_atoms());
          },
          py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::kw_only(),
          py::arg("gradient") = false, py::arg("virial") = false, py::arg("atom_energies") = false)
      .def(
          "set_parameters",
          [](D3Context& context, double s6, double s8, double a1, double a2, double s9, double alp,
             double cutoff_cn, double cutoff_disp2, double cutoff_disp3) {
            const D3Parameters parameters =
                make_parameters(s6, s8, a1, a2, s9, alp, cutoff_cn, cutoff_disp2, cutoff_disp3);
            py::gil_scoped_release release;
            context.set_parameters(parameters);
          },
          py::kw_only(), py::arg("s6"), py::arg("s8"), py::arg("a1"), py::arg("a2"), py::arg("s9"),
          py::arg("alp"), py::arg("cutoff_cn"), py::arg("cutoff_disp2"), py::arg("cutoff_disp3"))
      .def(
          "debug",
          [](D3Context& context, const DoubleArray& positions, const DoubleArray& cell,
             std::array<bool, 3> pbc, bool derivatives) {
            if (positions.ndim() != 2 || positions.shape(1) != 3 || cell.size() != 9) {
              throw Error(ErrorCode::InvalidArgument, "positions must be (N, 3) and cell (3, 3)");
            }
            std::vector<double> owned(positions.data(), positions.data() + positions.size());
            Geometry geometry;
            geometry.positions = owned.data();
            geometry.n_atoms = static_cast<std::size_t>(positions.shape(0));
            std::memcpy(geometry.cell.data(), cell.data(), 9 * sizeof(double));
            geometry.pbc = pbc;
            DebugResult debug;
            {
              py::gil_scoped_release release;
              debug = context.debug(geometry, derivatives);
            }
            const auto n = static_cast<py::ssize_t>(debug.cn.size());
            const auto edges = static_cast<py::ssize_t>(debug.edge_i.size());
            py::dict out;
            out["cn"] = to_array(debug.cn, {n});
            out["weights"] = to_array(debug.weights, {n, 8});
            out["dweights"] = derivatives ? py::object(to_array(debug.dweights, {n, 8})) : py::none();
            out["dEdcn"] = derivatives ? py::object(to_array(debug.dEdcn, {n})) : py::none();
            py::array_t<std::int32_t> i(edges), j(edges), shift({edges, py::ssize_t{3}});
            if (edges > 0) {
              std::memcpy(i.mutable_data(), debug.edge_i.data(), debug.edge_i.size() * 4);
              std::memcpy(j.mutable_data(), debug.edge_j.data(), debug.edge_j.size() * 4);
              std::memcpy(shift.mutable_data(), debug.edge_shift.data(), debug.edge_shift.size() * 4);
            }
            out["edge_i"] = i;
            out["edge_j"] = j;
            out["edge_shift"] = shift;
            return out;
          },
          py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::kw_only(),
          py::arg("derivatives") = true)
      .def("invalidate_neighbors", &D3Context::invalidate_neighbors)
      .def(
          "_testing_set_allocation_limit",
          [](D3Context& context, std::size_t bytes) { context.testing_set_allocation_limit(bytes); },
          py::arg("bytes"))
      .def(
          "_testing_hold",
          [](D3Context& context, double seconds) {
            py::gil_scoped_release release;
            context.testing_hold(seconds);
          },
          py::arg("seconds"))
      .def_property_readonly("n_atoms", &D3Context::n_atoms)
      .def("stats", [](const D3Context& context) {
        const ContextStats s = context.stats();
        py::dict d;
        d["evaluations"] = s.evaluations;
        d["rebuilds"] = s.rebuilds;
        d["allocations"] = s.allocations;
        d["device_bytes"] = s.device_bytes;
        d["capacity_edges"] = s.capacity_edges;
        return d;
      });

  m.def("d4_reference_summary", [] {
    auto tables = D4ReferenceData::embedded();
    py::dict d;
    d["source"] = tables->source_id();
    d["sha256"] = tables->identity();
    d["max_z"] = tables->max_z();
    return d;
  });
  m.def(
      "d4_reference_alpha",
      [](int z, double ga, double gc) {
        auto tables = D4ReferenceData::embedded();
        const std::vector<double> alpha = d4_reference_alpha(*tables, z, ga, gc);
        return to_array(alpha, {static_cast<py::ssize_t>(tables->nref(z)), D4ReferenceData::kFreq});
      },
      py::arg("z"), py::arg("ga") = 3.0, py::arg("gc") = 2.0);
  m.def(
      "d4_reference_c6",
      [](int zi, int zj, double ga, double gc) {
        auto tables = D4ReferenceData::embedded();
        const std::vector<double> a = d4_reference_alpha(*tables, zi, ga, gc);
        const std::vector<double> b = d4_reference_alpha(*tables, zj, ga, gc);
        const int ni = tables->nref(zi);
        const int nj = tables->nref(zj);
        return to_array(d4_reference_c6(*tables, a, ni, b, nj), {ni, nj});
      },
      py::arg("zi"), py::arg("zj"), py::arg("ga") = 3.0, py::arg("gc") = 2.0);

  py::class_<D4Context>(m, "D4Context")
      .def(py::init([](const IntArray& numbers, double s6, double s8, double a1, double a2, double s9,
                       double alp, double ga, double gc, double wf, double cutoff_cn,
                       double cutoff_disp2, double cutoff_disp3, double charge,
                       const std::string& ewald, const std::string& solver, double solver_tolerance,
                       int solver_max_iterations, bool warm_start, int device, double skin,
                       const std::string& builder, const py::object& reference_data,
                       const std::string& reference_identity, bool collect_timings,
                       long long atm_items_per_launch, const std::string& kernels) {
             if (numbers.ndim() != 1) {
               throw Error(ErrorCode::InvalidArgument, "numbers must be one-dimensional");
             }
             const D4Parameters parameters = make_d4_parameters(s6, s8, a1, a2, s9, alp, ga, gc, wf,
                                                                cutoff_cn, cutoff_disp2, cutoff_disp3);
             ContextOptions options;
             options.device = device;
             options.skin = skin;
             options.builder = parse_builder(builder);
             options.collect_timings = collect_timings;
             if (atm_items_per_launch < 0) {
               throw Error(ErrorCode::InvalidArgument, "atm_items_per_launch must be >= 0");
             }
             options.atm_items_per_launch = atm_items_per_launch;
             options.kernels = parse_kernels(kernels);
             EeqOptions eeq;
             if (ewald == "upstream") {
               eeq.ewald = EwaldMode::Upstream;
             } else if (ewald == "converged") {
               eeq.ewald = EwaldMode::Converged;
             } else {
               throw Error(ErrorCode::InvalidArgument, "ewald must be 'upstream' or 'converged'");
             }
             if (solver == "auto") {
               eeq.solver = ChargeSolver::Auto;
             } else if (solver == "iterative") {
               eeq.solver = ChargeSolver::Iterative;
             } else if (solver == "direct") {
               eeq.solver = ChargeSolver::Direct;
             } else {
               throw Error(ErrorCode::InvalidArgument, "solver must be 'auto', 'iterative', or 'direct'");
             }
             if (solver_tolerance > 0.0) eeq.tolerance = solver_tolerance;
             if (solver_max_iterations > 0) eeq.max_iterations = solver_max_iterations;
             eeq.warm_start = warm_start;
             std::shared_ptr<const D4ReferenceData> reference = D4ReferenceData::embedded();
             if (!reference_data.is_none()) {
               const std::string blob = reference_data.cast<py::bytes>();
               reference = D4ReferenceData::from_bytes(blob.data(), blob.size(), reference_identity);
             }
             std::vector<std::int32_t> z(numbers.data(), numbers.data() + numbers.size());
             py::gil_scoped_release release;
             return D4Context(z.data(), z.size(), parameters, options, charge, eeq, std::move(reference));
           }),
           py::arg("numbers"), py::kw_only(), py::arg("s6"), py::arg("s8"), py::arg("a1"),
           py::arg("a2"), py::arg("s9"), py::arg("alp"), py::arg("ga"), py::arg("gc"), py::arg("wf"),
           py::arg("cutoff_cn"), py::arg("cutoff_disp2"), py::arg("cutoff_disp3"),
           py::arg("charge") = 0.0, py::arg("ewald") = "upstream", py::arg("solver") = "auto",
           py::arg("solver_tolerance") = 0.0, py::arg("solver_max_iterations") = 0,
           py::arg("warm_start") = false, py::arg("device") = 0, py::arg("skin") = 0.0,
           py::arg("builder") = "auto", py::arg("reference_data") = py::none(),
           py::arg("reference_identity") = "", py::arg("collect_timings") = false,
           py::arg("atm_items_per_launch") = 0, py::arg("kernels") = "optimized")
      .def(
          "evaluate",
          [](D4Context& context, const DoubleArray& positions, const DoubleArray& cell,
             std::array<bool, 3> pbc, bool gradient, bool virial, bool atom_energies) {
            unsigned properties = kEnergy;
            if (gradient) properties |= kGradient;
            if (virial) properties |= kVirial;
            if (atom_energies) properties |= kAtomEnergies;
            const D4Result result = run_evaluate(context, positions, cell, pbc, properties);
            const std::size_t n = context.n_atoms();
            py::dict out = result_dict(result, n);
            out["charges"] = to_array(result.charges, {static_cast<py::ssize_t>(n)});
            py::dict info = out["info"];
            info["eeq_solver"] = result.eeq.solver;
            info["ewald_mode"] = result.eeq.ewald;
            info["eeq_iterations"] = result.eeq.forward_iterations;
            info["eeq_adjoint_iterations"] = result.eeq.adjoint_iterations;
            info["eeq_restarts"] = result.eeq.restarts;
            info["eeq_backward_error"] = result.eeq.backward_error;
            info["ewald_alpha"] = result.eeq.alpha;
            info["ewald_translations"] = result.eeq.n_trans;
            info["ewald_reciprocal"] = result.eeq.n_recip;
            info["eeq_matrix_bytes"] = result.eeq.matrix_bytes;
            info["eeq_warm_started"] = result.eeq.warm_started;
            return out;
          },
          py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::kw_only(),
          py::arg("gradient") = false, py::arg("virial") = false, py::arg("atom_energies") = false)
      .def(
          "set_parameters",
          [](D4Context& context, double s6, double s8, double a1, double a2, double s9, double alp,
             double ga, double gc, double wf, double cutoff_cn, double cutoff_disp2,
             double cutoff_disp3) {
            const D4Parameters parameters = make_d4_parameters(s6, s8, a1, a2, s9, alp, ga, gc, wf,
                                                               cutoff_cn, cutoff_disp2, cutoff_disp3);
            py::gil_scoped_release release;
            context.set_parameters(parameters);
          },
          py::kw_only(), py::arg("s6"), py::arg("s8"), py::arg("a1"), py::arg("a2"), py::arg("s9"),
          py::arg("alp"), py::arg("ga"), py::arg("gc"), py::arg("wf"), py::arg("cutoff_cn"),
          py::arg("cutoff_disp2"), py::arg("cutoff_disp3"))
      .def(
          "set_charge", [](D4Context& context, double charge) { context.set_charge(charge); },
          py::arg("charge"))
      .def(
          "debug",
          [](D4Context& context, const DoubleArray& positions, const DoubleArray& cell,
             std::array<bool, 3> pbc, bool derivatives) {
            std::vector<double> owned;
            const Geometry geometry = debug_geometry(positions, cell, pbc, owned);
            D4Debug debug;
            {
              py::gil_scoped_release release;
              debug = context.debug(geometry, derivatives);
            }
            const auto n = static_cast<py::ssize_t>(debug.cn.size());
            const auto edges = static_cast<py::ssize_t>(debug.edge_i.size());
            auto padded = [&](const std::vector<double>& values) {
              return values.empty() ? py::object(py::none()) : py::object(to_array(values, {n, 8}));
            };
            auto vector = [&](const std::vector<double>& values) {
              return values.empty() ? py::object(py::none()) : py::object(to_array(values, {n}));
            };
            py::dict out;
            out["cn"] = to_array(debug.cn, {n});
            out["cn_eeq"] = to_array(debug.cn_eeq, {n});
            out["charges"] = to_array(debug.charges, {n});
            out["weights"] = padded(debug.weights);
            out["weights_q0"] = padded(debug.weights_q0);
            out["dweights_dcn"] = padded(debug.dweights_dcn);
            out["dweights_dq"] = padded(debug.dweights_dq);
            out["dEdcn"] = vector(debug.dEdcn);
            out["dEdq"] = vector(debug.dEdq);
            py::array_t<std::int32_t> i(edges), j(edges), shift({edges, py::ssize_t{3}});
            if (edges > 0) {
              std::memcpy(i.mutable_data(), debug.edge_i.data(), debug.edge_i.size() * 4);
              std::memcpy(j.mutable_data(), debug.edge_j.data(), debug.edge_j.size() * 4);
              std::memcpy(shift.mutable_data(), debug.edge_shift.data(), debug.edge_shift.size() * 4);
            }
            out["edge_i"] = i;
            out["edge_j"] = j;
            out["edge_shift"] = shift;
            return out;
          },
          py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::kw_only(),
          py::arg("derivatives") = true)
      .def("invalidate_neighbors", &D4Context::invalidate_neighbors)
      .def(
          "_testing_set_allocation_limit",
          [](D4Context& context, std::size_t bytes) { context.testing_set_allocation_limit(bytes); },
          py::arg("bytes"))
      .def(
          "_testing_set_packed_min_atoms",
          [](D4Context& context, std::size_t atoms) { context.testing_set_packed_min_atoms(atoms); },
          py::arg("atoms"))
      .def(
          "_testing_set_matrix_limit",
          [](D4Context& context, std::size_t bytes) { context.testing_set_matrix_limit(bytes); },
          py::arg("bytes"))
      .def_property_readonly("n_atoms", &D4Context::n_atoms)
      .def("stats", [](const D4Context& context) {
        const ContextStats s = context.stats();
        py::dict d;
        d["evaluations"] = s.evaluations;
        d["rebuilds"] = s.rebuilds;
        d["allocations"] = s.allocations;
        d["device_bytes"] = s.device_bytes;
        d["capacity_edges"] = s.capacity_edges;
        return d;
      });
}
