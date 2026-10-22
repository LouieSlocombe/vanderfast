#include "vanderfast/error.hpp"

namespace vanderfast {

const char* to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::InvalidArgument:
      return "InvalidArgument";
    case ErrorCode::InvalidParameter:
      return "InvalidParameter";
    case ErrorCode::UnsupportedElement:
      return "UnsupportedElement";
    case ErrorCode::UnsupportedFeature:
      return "UnsupportedFeature";
    case ErrorCode::InvalidGeometry:
      return "InvalidGeometry";
    case ErrorCode::CoincidentSites:
      return "CoincidentSites";
    case ErrorCode::CudaUnavailable:
      return "CudaUnavailable";
    case ErrorCode::CudaRuntime:
      return "CudaRuntime";
    case ErrorCode::OutOfDeviceMemory:
      return "OutOfDeviceMemory";
    case ErrorCode::SizeOverflow:
      return "SizeOverflow";
    case ErrorCode::ContextBusy:
      return "ContextBusy";
    case ErrorCode::ContextPoisoned:
      return "ContextPoisoned";
    case ErrorCode::Internal:
      return "Internal";
    case ErrorCode::ChargeSystemSingular:
      return "ChargeSystemSingular";
    case ErrorCode::ChargeSolverNotConverged:
      return "ChargeSolverNotConverged";
  }
  return "Unknown";
}

Error::Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}

void fail(ErrorCode code, const std::string& message) { throw Error(code, message); }

}  // namespace vanderfast
