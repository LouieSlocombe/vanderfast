# Script mode: embed a binary file as a C++ byte array.
#   cmake -DINPUT=<file> -DOUTPUT=<cpp> -DSYMBOL=<name> -P EmbedBinary.cmake
file(READ "${INPUT}" _hex HEX)
string(LENGTH "${_hex}" _hex_length)
math(EXPR _size "${_hex_length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n" _bytes "${_bytes}")
get_filename_component(_name "${INPUT}" NAME)
file(WRITE "${OUTPUT}.tmp"
  "// Generated from ${_name} by cmake/EmbedBinary.cmake. Do not edit.\n"
  "#include <cstddef>\n\n"
  "namespace vanderfast::detail {\n"
  "alignas(64) extern const unsigned char ${SYMBOL}[] = {\n${_bytes}\n};\n"
  "extern const std::size_t ${SYMBOL}_size = ${_size};\n"
  "}  // namespace vanderfast::detail\n")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
