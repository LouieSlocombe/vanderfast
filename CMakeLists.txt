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

if(VANDERFAST_RELEASE_BUILD AND NOT VANDERFAST_ENABLE_CUDA)
  message(FATAL_ERROR "vanderfast: release builds require VANDERFAST_ENABLE_CUDA=ON")
endif()

if(NOT VANDERFAST_ENABLE_CUDA AND NOT VANDERFAST_BUILD_TESTS)
  message(STATUS "vanderfast: CUDA disabled; installing the pure-Python package only")
  return()
endif()

message(FATAL_ERROR "vanderfast: the native core is not implemented yet; "
                    "set VANDERFAST_ENABLE_CUDA=OFF")
