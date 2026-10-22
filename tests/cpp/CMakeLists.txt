find_package(Catch2 3 CONFIG REQUIRED)

add_executable(vanderfast_host_tests
  test_checked_math.cpp
  test_reference_data.cpp
  test_parameters.cpp
  test_lattice.cpp
  test_d4.cpp)
target_include_directories(vanderfast_host_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(vanderfast_host_tests PRIVATE vanderfast_host Catch2::Catch2WithMain)
target_compile_definitions(vanderfast_host_tests PRIVATE
  VANDERFAST_REFERENCE_BLOB_PATH="${VANDERFAST_REFERENCE_BLOB}"
  VANDERFAST_D4_REFERENCE_BLOB_PATH="${VANDERFAST_D4_REFERENCE_BLOB}")
vanderfast_host_warnings(vanderfast_host_tests)
add_test(NAME host COMMAND vanderfast_host_tests)
set_tests_properties(host PROPERTIES LABELS host)

if(VANDERFAST_ENABLE_CUDA)
  add_executable(vanderfast_cuda_tests test_device.cpp)
  target_include_directories(vanderfast_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(vanderfast_cuda_tests PRIVATE vanderfast_cuda Catch2::Catch2WithMain)
  add_test(NAME cuda COMMAND vanderfast_cuda_tests)
  set_tests_properties(cuda PROPERTIES LABELS cuda)
endif()
