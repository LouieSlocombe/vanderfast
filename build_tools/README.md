# Build environment

vanderfast is developed in a conda-forge environment that provides Python, the
runtime dependencies, the pinned `dftd3` 1.6.0 reference, and a self-contained
CUDA 12.9 + GCC 14 toolchain (CUDA 12.9 accepts host GCC <= 14).

| File | Purpose |
| --- | --- |
| `environment.yml` | Development environment specification (source of truth) |
| `conda-linux-64.lock` | Explicit lock of a solved `environment.yml`, used by CI |
| `environment-oldest.yml` | Oldest allowed dependency versions and the oldest supported CUDA (12.6 + GCC 13): CPU tests and a no-GPU CUDA build |
| `environment-cuda13.yml` | Newest CUDA 13 release + GCC 15, unlocked: a no-GPU CUDA build |

Create and activate the environment, then install vanderfast in editable mode:

```bash
conda env create -f build_tools/environment.yml   # or: conda create -n vanderfast-dev --file build_tools/conda-linux-64.lock
conda activate vanderfast-dev
python -m pip install --no-build-isolation --no-deps -e .
```

Set `VANDERFAST_ENABLE_CUDA=OFF` to install only the pure-Python parts (oracle,
reference adapter, parameter tables) on machines without CUDA.

Refresh the lock after changing `environment.yml`:

```bash
conda env update -n vanderfast-dev -f build_tools/environment.yml --prune
conda list -n vanderfast-dev --explicit --md5 > build_tools/conda-linux-64.lock
```

The machine's `/usr/bin/nvcc` may be an older distribution CUDA. The CMake build
refuses CUDA < 12.6, and the conda toolchain keeps system CUDA headers out of
the include path. To build with a system toolkit instead, set `CUDACXX` to its
nvcc, for example `CUDACXX=/usr/local/cuda-13.1/bin/nvcc`. Profilers and
`compute-sanitizer` from a system CUDA 13.1 installation work with binaries built
by the conda toolchain.
