#include "build_metadata.hpp"

#include "vanderfast/build_info.hpp"

namespace vanderfast {

std::map<std::string, std::string> build_info() {
  return {
      {"version", VANDERFAST_VERSION},
      {"reference_sha256", VANDERFAST_REFERENCE_SHA256},
      {"build_type", VANDERFAST_BUILD_TYPE},
      {"cxx_compiler", VANDERFAST_CXX_COMPILER},
      {"cuda_compiler", VANDERFAST_CUDA_COMPILER},
      {"cuda_host_compiler", VANDERFAST_CUDA_HOST_COMPILER},
      {"cuda_architectures", VANDERFAST_CUDA_ARCHITECTURES},
      {"cuda_fmad", VANDERFAST_CUDA_FMAD ? "true" : "false"},
      {"cuda_runtime", VANDERFAST_CUDA_RUNTIME},
      {"fast_math", "false"},
      {"precision", "float64"},
  };
}

}  // namespace vanderfast
