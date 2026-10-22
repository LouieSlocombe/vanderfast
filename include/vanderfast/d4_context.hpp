// The DFT-D4 evaluation context: EEQ charges, D4 dispersion, and ATM on the GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vanderfast/context.hpp"
#include "vanderfast/d4_parameters.hpp"
#include "vanderfast/d4_reference_data.hpp"

namespace vanderfast {

// Charge-solver diagnostics of one evaluation.
struct EeqInfo {
  std::string solver;               // "direct" or "minres"
  std::string ewald;                // "none", "upstream", or "converged"
  int forward_iterations = 0;
  int adjoint_iterations = 0;
  int restarts = 0;
  double backward_error = 0.0;      // normwise backward error of the forward solve
  double alpha = 0.0;               // Ewald parameter (periodic systems)
  std::int64_t n_trans = 0;         // real-space translations
  std::int64_t n_recip = 0;         // reciprocal vectors
  std::size_t matrix_bytes = 0;
  bool warm_started = false;
};

struct D4Result : HostResult {
  std::vector<double> charges;  // EEQ partial charges, e
  EeqInfo eeq;
};

struct D4Debug {
  std::vector<double> cn;            // D4 coordination numbers
  std::vector<double> cn_eeq;        // capped EEQ coordination numbers
  std::vector<double> charges;
  std::vector<double> weights;       // n_atoms x 8, charge scaled
  std::vector<double> weights_q0;    // n_atoms x 8, neutral (ATM)
  std::vector<double> dweights_dcn;  // n_atoms x 8
  std::vector<double> dweights_dq;   // n_atoms x 8
  std::vector<double> dEdcn;         // before the CN reverse pass
  std::vector<double> dEdq;
  std::vector<std::int32_t> edge_i;
  std::vector<std::int32_t> edge_j;
  std::vector<std::int32_t> edge_shift;
};

class D4Context {
 public:
  D4Context(const std::int32_t* numbers, std::size_t n_atoms, const D4Parameters& parameters,
            const ContextOptions& options = {}, double total_charge = 0.0, const EeqOptions& eeq = {},
            std::shared_ptr<const D4ReferenceData> reference = D4ReferenceData::embedded());
  ~D4Context();
  D4Context(D4Context&&) noexcept;
  D4Context& operator=(D4Context&&) noexcept;
  D4Context(const D4Context&) = delete;
  D4Context& operator=(const D4Context&) = delete;

  // Evaluate the requested properties. Any periodic flag makes the system
  // fully periodic over the cell, as in dftd4. Throws Error(ContextBusy) if
  // another thread is evaluating with the same context.
  D4Result evaluate(const Geometry& geometry, unsigned properties);

  // Run the stages eagerly and return intermediate quantities.
  D4Debug debug(const Geometry& geometry, bool derivatives);

  void set_parameters(const D4Parameters& parameters);
  void set_charge(double total_charge);
  void invalidate_neighbors() noexcept;

  std::size_t n_atoms() const noexcept;
  const D4Parameters& parameters() const noexcept;
  double charge() const noexcept;
  ContextStats stats() const;

  void testing_set_allocation_limit(std::size_t bytes);
  // Testing hooks: the atom count from which MINRES uses packed storage, and
  // the EEQ memory limit (0 restores the defaults).
  void testing_set_packed_min_atoms(std::size_t atoms);
  void testing_set_matrix_limit(std::size_t bytes);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vanderfast
