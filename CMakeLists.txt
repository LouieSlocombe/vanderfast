cmake_minimum_required(VERSION 3.30...4.4)

if(DEFINED SKBUILD_PROJECT_VERSION)
  set(VANDERFAST_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(VANDERFAST_VERSION "0.1.0")
endif()

project(vanderfast VERSION ${VANDERFAST_VERSION} LANGUAGES NONE)

option(VANDERFAST_ENABLE_CUDA "Build the CUDA core and the _native extension" ON)
option(VANDERFAST_BUILD_TESTS "Build the C++ tests (Catch2 v3, CTest)" OFF)
option(VANDERFAST_WERROR "Treat warnings as errors" OFF)
option(VANDERFAST_RELEASE_BUILD "Enforce release build invariants" OFF)
option(VANDERFAST_CUDA_FMAD "Allow FMA contraction in device code" ON)
option(VANDERFAST_CUDA_DEVICE_DEBUG "Compile device code with -G" OFF)
set(VANDERFAST_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for the kernels")

if(VANDERFAST_RELEASE_BUILD AND NOT VANDERFAST_ENABLE_CUDA)
  message(FATAL_ERROR "vanderfast: release builds require VANDERFAST_ENABLE_CUDA=ON")
endif()

if(NOT VANDERFAST_ENABLE_CUDA AND NOT VANDERFAST_BUILD_TESTS)
  message(STATUS "vanderfast: CUDA disabled; installing the pure-Python package only")
  return()
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(cmake/VanderfastWarnings.cmake)

# --- Embedded reference data (the same file is Python package data) ---------
set(VANDERFAST_REFERENCE_BLOB "${PROJECT_SOURCE_DIR}/src/vanderfast/data/d3_reference.bin")
file(SHA256 "${VANDERFAST_REFERENCE_BLOB}" VANDERFAST_REFERENCE_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VANDERFAST_REFERENCE_BLOB}")
set(VANDERFAST_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(
  OUTPUT "${VANDERFAST_GENERATED}/embedded_reference.cpp"
  COMMAND "${CMAKE_COMMAND}" -DINPUT=${VANDERFAST_REFERENCE_BLOB}
          -DOUTPUT=${VANDERFAST_GENERATED}/embedded_reference.cpp
          -DSYMBOL=d3_reference_blob -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
  DEPENDS "${VANDERFAST_REFERENCE_BLOB}" "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
  COMMENT "Embedding the D3 reference tables"
  VERBATIM)
set(VANDERFAST_D4_REFERENCE_BLOB "${PROJECT_SOURCE_DIR}/src/vanderfast/data/d4_reference.bin")
file(SHA256 "${VANDERFAST_D4_REFERENCE_BLOB}" VANDERFAST_D4_REFERENCE_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VANDERFAST_D4_REFERENCE_BLOB}")
add_custom_command(
  OUTPUT "${VANDERFAST_GENERATED}/embedded_d4_reference.cpp"
  COMMAND "${CMAKE_COMMAND}" -DINPUT=${VANDERFAST_D4_REFERENCE_BLOB}
          -DOUTPUT=${VANDERFAST_GENERATED}/embedded_d4_reference.cpp
          -DSYMBOL=d4_reference_blob -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
  DEPENDS "${VANDERFAST_D4_REFERENCE_BLOB}" "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
  COMMENT "Embedding the D4 reference tables"
  VERBATIM)

if(VANDERFAST_ENABLE_CUDA)
  include(cmake/VanderfastCuda.cmake)
endif()
if(VANDERFAST_CUDA_FMAD)
  set(VANDERFAST_CUDA_FMAD_VALUE 1)
else()
  set(VANDERFAST_CUDA_FMAD_VALUE 0)
endif()
if(CMAKE_CUDA_HOST_COMPILER)
  set(VANDERFAST_CUDA_HOST_COMPILER "${CMAKE_CUDA_HOST_COMPILER}")
elseif(DEFINED ENV{NVCC_PREPEND_FLAGS})
  string(STRIP "$ENV{NVCC_PREPEND_FLAGS}" VANDERFAST_CUDA_HOST_COMPILER)
else()
  set(VANDERFAST_CUDA_HOST_COMPILER "nvcc default")
endif()
configure_file(cmake/build_info.hpp.in "${VANDERFAST_GENERATED}/vanderfast/build_info.hpp" @ONLY)

# --- Host library: reference data, parameters, lattice ----------------------
add_library(vanderfast_host STATIC
  src/core/error.cpp
  src/core/reference_data.cpp
  src/core/parameters.cpp
  src/core/d4_reference_data.cpp
  src/core/d4_parameters.cpp
  src/core/d4_model.cpp
  src/core/ewald.cpp
  src/core/lattice.cpp
  src/core/neighbor_cpu.cpp
  src/core/build_metadata.cpp
  "${VANDERFAST_GENERATED}/embedded_reference.cpp"
  "${VANDERFAST_GENERATED}/embedded_d4_reference.cpp")
target_include_directories(vanderfast_host
  PUBLIC "${PROJECT_SOURCE_DIR}/include" "${VANDERFAST_GENERATED}"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")
vanderfast_host_warnings(vanderfast_host)

# --- CUDA core and the Python extension --------------------------------------
if(VANDERFAST_ENABLE_CUDA)
  add_library(vanderfast_cuda STATIC
    src/cuda/cuda_check.cu
    src/cuda/device_model.cu
    src/cuda/neighbor.cu
    src/cuda/k_geometry.cu
    src/cuda/k_cn.cu
    src/cuda/k_weights.cu
    src/cuda/k_contract.cu
    src/cuda/k_pair.cu
    src/cuda/k_atm.cu
    src/cuda/k_rows.cu
    src/cuda/k_reduce.cu
    src/cuda/pipeline.cu
    src/cuda/context.cu
    src/cuda/d4_model.cu
    src/cuda/k_d4.cu
    src/cuda/k_eeq.cu
    src/cuda/k_eeq_solver.cu
    src/cuda/k_eeq_tiles.cu
    src/cuda/d4_context.cu)
  target_include_directories(vanderfast_cuda PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(vanderfast_cuda PUBLIC vanderfast_host vanderfast_cub CUDA::cudart_static)
  vanderfast_cuda_target(vanderfast_cuda)

  # CUB's device-wide scan and sort, kept apart so that its kernels are exempt
  # from the fatal spill warnings (NVIDIA's tuning policies spill a few bytes
  # on some architectures).
  add_library(vanderfast_cub STATIC src/cuda/cub_primitives.cu)
  target_link_libraries(vanderfast_cub PUBLIC CUDA::cudart_static)
  vanderfast_cuda_target(vanderfast_cub THIRD_PARTY)

  if(DEFINED SKBUILD_PROJECT_NAME OR NOT VANDERFAST_BUILD_TESTS)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 3.0 CONFIG REQUIRED)
    pybind11_add_module(_native MODULE src/bindings/module.cpp)
    target_include_directories(_native PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(_native PRIVATE vanderfast_cuda)
    install(TARGETS _native LIBRARY DESTINATION vanderfast COMPONENT python)
  endif()
endif()

if(VANDERFAST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
