# Building vanderfast

## Toolchain

vanderfast's CUDA core is built with CMake (>= 3.30) through scikit-build-core
and pybind11. The supported toolchain is the conda-forge environment in
`build_tools/environment.yml`:

| Component | Version |
| --- | --- |
| CUDA compiler | `cuda-nvcc` 13.1 (conda-forge) |
| Host compiler | GCC 15 (`gxx_linux-64=15`; CUDA 13.1 accepts GCC <= 15) |
| CUDA runtime | statically linked (`CUDA::cudart_static`) |
| Python | 3.14 |

The build refuses CUDA < 13.0 and mixed compiler/toolkit versions. To use a
system CUDA toolkit instead, point `CUDACXX` at its `nvcc` (for example
`/usr/local/cuda-13.1/bin/nvcc`).

## Commands

```bash
make dev        # editable CUDA build for this machine's GPU (build/dev)
make dev-cpu    # pure-Python install (VANDERFAST_ENABLE_CUDA=OFF)
make test-gpu   # full test suite; fails instead of skipping GPU or reference tests
make cpp-test   # Catch2 host and device tests with -Werror
make stubtest   # check src/vanderfast/_native.pyi against the extension
make wheel      # release-architecture wheel + device-code and dependency checks
```

## Build options

| Variable | Default | Meaning |
| --- | --- | --- |
| `VANDERFAST_ENABLE_CUDA` | `ON` | Build the CUDA core and `vanderfast._native` |
| `VANDERFAST_CUDA_ARCHITECTURES` | `native` | CMake `CUDA_ARCHITECTURES` for the kernels |
| `VANDERFAST_RELEASE_BUILD` | `OFF` | Require an explicit architecture list and CUDA |
| `VANDERFAST_WERROR` | `OFF` | Treat host and device warnings as errors |

Device code is always compiled with `--prec-div=true --prec-sqrt=true
--ftz=false` and never with fast-math: the reference-weight fallback depends on
IEEE NaN and infinity handling. `-lineinfo` is always on for profiler and
compute-sanitizer line mapping.

The release architecture list is
`75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual`: SASS
for Turing through Blackwell (sm_103 runs sm_100 code, sm_121 runs sm_120 code)
and compute_120 PTX for newer GPUs. CUDA 13 no longer supports Maxwell, Pascal,
or Volta.

## Support matrix

Only combinations that have been exercised are listed.

| GPU | Compute capability | Driver | CUDA (build) | OS | Status |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4080 SUPER | 8.9 | 615.71.09 (CUDA 13.4 UMD) | 13.1.115 | Linux x86_64 | tested locally |

The statically linked runtime needs a driver that supports CUDA 13.1 for SASS
execution, or a newer one for PTX just-in-time compilation. Importing
`vanderfast._native` does not require a GPU; creating a context does.
