#include <string>

#include "cuda_check.cuh"

namespace vanderfast::cuda {

bool is_sticky(cudaError_t status) noexcept {
  switch (status) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorMisalignedAddress:
    case cudaErrorIllegalInstruction:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorHardwareStackError:
    case cudaErrorAssert:
    case cudaErrorLaunchTimeout:
    case cudaErrorECCUncorrectable:
      return true;
    default:
      return false;
  }
}

void throw_cuda_error(cudaError_t status, const char* expression, const char* file, int line) {
  std::string message = std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status) +
                        " (" + expression + " at " + file + ":" + std::to_string(line) + ")";
  switch (status) {
    case cudaErrorMemoryAllocation:
      throw Error(ErrorCode::OutOfDeviceMemory, message);
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorSystemDriverMismatch:
    case cudaErrorCompatNotSupportedOnDevice:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorUnsupportedPtxVersion:
    case cudaErrorDevicesUnavailable:
      throw Error(ErrorCode::CudaUnavailable, message);
    default:
      throw Error(ErrorCode::CudaRuntime, message);
  }
}

}  // namespace vanderfast::cuda
