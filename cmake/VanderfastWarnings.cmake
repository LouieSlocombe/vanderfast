# Host compiler warnings for vanderfast's own C++ sources.
include_guard(GLOBAL)

function(vanderfast_host_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion
      -Wsign-conversion -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual
      -Wcast-align -Wnull-dereference -Wdouble-promotion -Wformat=2
      -Wimplicit-fallthrough -fno-fast-math>)
  if(VANDERFAST_WERROR)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endfunction()
