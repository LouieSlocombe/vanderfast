// Error reporting for the vanderfast native core.
#pragma once

#include <stdexcept>
#include <string>

namespace vanderfast {

enum class ErrorCode {
  InvalidArgument,     // malformed input arrays or options
  InvalidParameter,    // invalid damping parameters or cutoffs
  UnsupportedElement,  // atomic number without reference data
  UnsupportedFeature,  // a request the implementation does not provide
  InvalidGeometry,     // non-finite positions, degenerate lattice, ...
  CoincidentSites,     // distinct lattice sites closer than the input threshold
  CudaUnavailable,     // no usable driver or device
  CudaRuntime,         // any other CUDA runtime failure
  OutOfDeviceMemory,   // a device allocation failed
  SizeOverflow,        // a size or index computation overflowed
  ContextBusy,         // a context was used concurrently
  ContextPoisoned,     // an earlier sticky CUDA error invalidated the context
  Internal,            // a broken internal invariant
  ChargeSystemSingular,       // the EEQ charge equations are singular
  ChargeSolverNotConverged,   // the iterative charge solver did not converge
};

const char* to_string(ErrorCode code) noexcept;

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& message);
  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] void fail(ErrorCode code, const std::string& message);

}  // namespace vanderfast
