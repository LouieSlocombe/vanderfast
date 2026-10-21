# CUDA toolchain guard, architecture policy, and device compile flags.
include_guard(GLOBAL)

enable_language(CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR
    "vanderfast requires CUDA >= 13.0, but ${CMAKE_CUDA_COMPILER} is "
    "${CMAKE_CUDA_COMPILER_VERSION}. Activate the vanderfast-dev conda environment "
    "or set CUDACXX to a CUDA 13 nvcc (for example /usr/local/cuda-13.1/bin/nvcc).")
endif()

find_package(CUDAToolkit 13.0 REQUIRED)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _vf_nvcc_version "${CMAKE_CUDA_COMPILER_VERSION}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _vf_toolkit_version "${CUDAToolkit_VERSION}")
if(NOT _vf_nvcc_version VERSION_EQUAL _vf_toolkit_version)
  message(FATAL_ERROR
    "CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION} and toolkit ${CUDAToolkit_VERSION} "
    "(${CUDAToolkit_INCLUDE_DIRS}) differ; mixed CUDA installations are not supported.")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(VANDERFAST_RELEASE_BUILD AND VANDERFAST_CUDA_ARCHITECTURES MATCHES "^(native|all|all-major)$")
  message(FATAL_ERROR
    "Release builds need an explicit CUDA architecture list, not '${VANDERFAST_CUDA_ARCHITECTURES}'.")
endif()

set(VANDERFAST_CUDA_ARCHITECTURES_RESOLVED "${VANDERFAST_CUDA_ARCHITECTURES}")
if(VANDERFAST_CUDA_ARCHITECTURES STREQUAL "native" AND DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
  set(VANDERFAST_CUDA_ARCHITECTURES_RESOLVED "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
endif()
message(STATUS "vanderfast: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures "
               "${VANDERFAST_CUDA_ARCHITECTURES_RESOLVED}")

if(VANDERFAST_CUDA_FMAD)
  set(_vf_fmad true)
else()
  set(_vf_fmad false)
endif()

# Apply the vanderfast device policy to a target: explicit architectures, a
# statically linked CUDA runtime, IEEE-conforming division and square root,
# no flush-to-zero, and never fast-math (the reference-weight fallback relies
# on NaN/Inf detection).
function(vanderfast_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${VANDERFAST_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --prec-div=true --prec-sqrt=true --ftz=false
      --fmad=${_vf_fmad} -Xptxas=-warn-spills,-warn-lmem-usage -Xcompiler=-fno-fast-math>)
  if(VANDERFAST_CUDA_DEVICE_DEBUG)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
  if(VANDERFAST_WERROR)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  endif()
endfunction()
