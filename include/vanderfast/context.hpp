// The vanderfast evaluation context: device resources and persistent state.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vanderfast/parameters.hpp"
#include "vanderfast/reference_data.hpp"

namespace vanderfast {

enum class NeighborBuilder { Auto, BruteForce, CellBins };

struct ContextOptions {
  int device = 0;
  double skin = 0.0;  // bohr; enlarges candidate lists only
  NeighborBuilder builder = NeighborBuilder::Auto;
  bool reject_coincident = true;
  bool collect_timings = false;
};

enum Property : unsigned {
  kEnergy = 1u << 0,
  kGradient = 1u << 1,
  kVirial = 1u << 2,
  kAtomEnergies = 1u << 3,
};

struct Geometry {
  const double* positions = nullptr;  // n_atoms x 3, bohr, row-major
  std::size_t n_atoms = 0;
  std::array<double, 9> cell{};  // lattice vectors as rows, bohr
  std::array<bool, 3> pbc{};
};

struct EvalInfo {
  bool rebuilt = false;
  std::string builder;
  std::int64_t cn_edges = 0;
  std::int64_t pair_edges = 0;
  std::int64_t atm_edges = 0;
  std::uint64_t atm_candidates = 0;
  std::uint64_t atm_accepted = 0;
  std::uint64_t atm_c6_skipped = 0;
  std::map<std::string, double> stage_ms;
};

struct HostResult {
  double energy = 0.0;       // hartree
  double energy_pair = 0.0;  // hartree
  double energy_atm = 0.0;   // hartree
  bool has_gradient = false;
  std::vector<double> gradient;  // dE/dx, n_atoms x 3, hartree/bohr
  bool has_virial = false;
  std::array<double, 9> virial{};  // dE/dstrain, hartree
  bool has_atom_energies = false;
  std::vector<double> atom_energies;
  EvalInfo info;
};

struct DebugResult {
  std::vector<double> cn;            // n_atoms
  std::vector<double> weights;       // n_atoms x 7, zero padded
  std::vector<double> dweights;      // n_atoms x 7, zero padded
  std::vector<double> dEdcn;         // n_atoms (when derivatives were requested)
  std::vector<std::int32_t> edge_i;  // directed edges of the largest view
  std::vector<std::int32_t> edge_j;
  std::vector<std::int32_t> edge_shift;  // n_edges x 3
};

struct ContextStats {
  std::int64_t evaluations = 0;
  std::int64_t rebuilds = 0;
  std::int64_t allocations = 0;
  std::size_t device_bytes = 0;
  std::int64_t capacity_edges = 0;
};

struct DeviceInfo {
  int index = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_memory = 0;
  int multiprocessors = 0;
  bool kernel_timeout = false;
};

class D3Context {
 public:
  D3Context(const std::int32_t* numbers, std::size_t n_atoms, const D3Parameters& parameters,
            const ContextOptions& options = {},
            std::shared_ptr<const ReferenceData> reference = ReferenceData::embedded());
  ~D3Context();
  D3Context(D3Context&&) noexcept;
  D3Context& operator=(D3Context&&) noexcept;
  D3Context(const D3Context&) = delete;
  D3Context& operator=(const D3Context&) = delete;

  // Evaluate the requested properties. Throws Error(ContextBusy) if another
  // thread is evaluating with the same context.
  HostResult evaluate(const Geometry& geometry, unsigned properties);

  // Run the stages eagerly and return intermediate quantities.
  DebugResult debug(const Geometry& geometry, bool derivatives);

  void set_parameters(const D3Parameters& parameters);
  void invalidate_neighbors() noexcept;

  std::size_t n_atoms() const noexcept;
  const D3Parameters& parameters() const noexcept;
  const ContextOptions& options() const noexcept;
  ContextStats stats() const;

  static int device_count() noexcept;
  static std::vector<DeviceInfo> devices();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vanderfast
