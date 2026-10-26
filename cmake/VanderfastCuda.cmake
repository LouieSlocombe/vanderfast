# CUDA toolchain guard, architecture policy, and device compile flags.
include_guard(GLOBAL)

enable_language(CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6)
  message(FATAL_ERROR
    "vanderfast requires CUDA >= 12.6, but ${CMAKE_CUDA_COMPILER} is "
    "${CMAKE_CUDA_COMPILER_VERSION}. Activate the vanderfast-dev conda environment "
    "or set CUDACXX to a CUDA 12.6 or newer nvcc (for example /usr/local/cuda-12.9/bin/nvcc).")
endif()

find_package(CUDAToolkit 12.6 REQUIRED)
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

# "release" names the architecture list every published wheel carries: SASS for
# each supported generation plus PTX for the newest, so later GPUs can JIT.
# Blackwell (sm_100, sm_120) needs CUDA >= 12.8.
set(VANDERFAST_RELEASE_CUDA_ARCHITECTURES
  "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual")
if(VANDERFAST_CUDA_ARCHITECTURES STREQUAL "release")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
      "The release architecture list needs CUDA >= 12.8 for sm_100 and sm_120, but "
      "${CMAKE_CUDA_COMPILER} is ${CMAKE_CUDA_COMPILER_VERSION}. Pass an explicit list "
      "instead, for example 75-real;80-real;86-real;89-real;90-real;90-virtual.")
  endif()
  set(VANDERFAST_CUDA_ARCHITECTURES "${VANDERFAST_RELEASE_CUDA_ARCHITECTURES}")
endif()

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
#
# THIRD_PARTY targets wrap library kernels (CUB): they keep every flag except
# the ptxas spill and local-memory warnings, which -Werror would make fatal.
function(vanderfast_cuda_target target)
  cmake_parse_arguments(PARSE_ARGV 1 _vf "THIRD_PARTY" "" "")
  if(_vf_THIRD_PARTY)
    set(_vf_ptxas_warnings "")
  else()
    set(_vf_ptxas_warnings "-Xptxas=-warn-spills,-warn-lmem-usage")
  endif()
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${VANDERFAST_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --prec-div=true --prec-sqrt=true --ftz=false
      --fmad=${_vf_fmad} ${_vf_ptxas_warnings} -Xcompiler=-fno-fast-math>)
  if(VANDERFAST_CUDA_DEVICE_DEBUG)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
  if(VANDERFAST_WERROR)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  endif()
endfunction()
