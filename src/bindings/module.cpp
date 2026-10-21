// pybind11 bindings: the vanderfast._native extension (atomic units only).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "../core/build_metadata.hpp"
#include "vanderfast/context.hpp"
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
  };
  const auto index = static_cast<std::size_t>(error.code());
  py::object errors = py::module_::import("vanderfast.errors");
  py::object type = errors.attr(names[index]);
  PyErr_SetString(type.ptr(), error.what());
}

HostResult run_evaluate(D3Context& context, const DoubleArray& positions, const DoubleArray& cell,
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

  py::class_<D3Context>(m, "Context")
      .def(py::init([](const IntArray& numbers, double s6, double s8, double a1, double a2, double s9,
                       double alp, double cutoff_cn, double cutoff_disp2, double cutoff_disp3,
                       int device, double skin, const std::string& builder,
                       const py::object& reference_data, const std::string& reference_identity,
                       bool collect_timings) {
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
           py::arg("collect_timings") = false)
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
      .def("invalidate_neighbors", &D3Context::invalidate_neighbors)
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
}
