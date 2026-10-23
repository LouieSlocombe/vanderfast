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
make sdist      # source distribution + contents and metadata (twine) checks
make wheel      # release-architecture wheel + device-code and dependency checks
make gpu-verify # install dist/*.whl in a fresh venv and run the whole suite on the GPU
```

## Build options

| Variable | Default | Meaning |
| --- | --- | --- |
| `VANDERFAST_ENABLE_CUDA` | `ON` | Build the CUDA core and `vanderfast._native` |
| `VANDERFAST_CUDA_ARCHITECTURES` | `native` | CMake `CUDA_ARCHITECTURES` for the kernels, or `release` for the published list |
| `VANDERFAST_RELEASE_BUILD` | `OFF` | Require an explicit architecture list and CUDA |
| `VANDERFAST_WERROR` | `OFF` | Treat host and device warnings as errors |

Device code is always compiled with `--prec-div=true --prec-sqrt=true
--ftz=false` and never with fast-math: the reference-weight fallback depends on
IEEE NaN and infinity handling. `-lineinfo` is always on for profiler and
compute-sanitizer line mapping.

vanderfast's kernels are compiled with `-Xptxas=-warn-spills,-warn-lmem-usage`,
so with `VANDERFAST_WERROR=ON` any register spill on any listed architecture
fails the build. `tools/check_wheel.py` checks the same thing again on the
built binary with `cuobjdump --dump-resource-usage`. CUB's scan and sort are
the one exception: they live in their own translation unit
(`src/cuda/cub_primitives.cu`), without those warnings, because NVIDIA's
tuning policies spill a few bytes on some architectures.

`release` expands to the release architecture list,
`75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual`: SASS
for Turing through Blackwell (sm_103 runs sm_100 code, sm_121 runs sm_120 code)
and compute_120 PTX for newer GPUs. CUDA 13 no longer supports Maxwell, Pascal,
or Volta.

## Support matrix

Only combinations that have been exercised are listed.

| GPU | Compute capability | Driver | Build | OS | Status |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4080 SUPER | 8.9 | 615.71.09 (CUDA 13.4 UMD) | editable, `native` (sm_89), CUDA 13.1.115 | Linux x86_64 | full suite, sanitizers, benchmarks |
| NVIDIA GeForce RTX 4080 SUPER | 8.9 | 615.71.09 (CUDA 13.4 UMD) | release-architecture wheel, CUDA 13.1.115 (`make wheel gpu-verify`) | Linux x86_64 | full suite in a fresh venv, benchmarks |

Not yet exercised:
- **Other architectures.** The release wheel contains SASS for sm_75, sm_80,
  sm_86, sm_90, sm_100 and sm_120 and PTX for compute_120. Those binaries are
  compiled and checked for spills but have never run on a GPU, and neither
  has the PTX just-in-time path. A second architecture, ideally a data-centre
  part with full-rate FP64, is the main open validation gap.
- **The manylinux_2_28 build.** The cibuildwheel container build has not run
  yet; it needs Docker and runs in the release workflow.
- **Drivers.** No driver older than the one listed has been tried.

The statically linked runtime needs a driver that supports CUDA 13.1 for SASS
execution, or a newer one for PTX just-in-time compilation. Importing
`vanderfast._native` does not require a GPU; creating a context does.

## Releasing

Publishing a GitHub release runs `.github/workflows/release.yml`:

1. **sdist**: checks that the tag is `v<project.version>`, then builds the
   sdist and runs `tools/check_sdist.py` and `twine check --strict` on it.
2. **wheel**: cibuildwheel builds `cp314-manylinux_x86_64` in the
   `manylinux_2_28` image (configured in `[tool.cibuildwheel]` in
   `pyproject.toml`):
   - CUDA 13.1 comes from NVIDIA's RHEL 8 repository, at the same nvcc and
     runtime versions as the conda lock.
   - The wheel is compiled for the `release` architecture list with `-Werror`.
   - `auditwheel repair --exclude libcuda.so.1` repairs it, then
     `tools/check_wheel.py` checks it: SASS for every listed architecture,
     compute_120 PTX, and no CUDA libraries among its dynamic dependencies.
   - Finally it runs the tests that need no GPU.
3. **gpu-verify**: a self-hosted runner labelled `cuda`, behind the `gpu`
   environment, installs the downloaded wheel into a fresh virtual
   environment that has PyPI dependencies only (`dftd3==1.6.0` and
   `dftd4==4.2.0` for the reference tests).
   - It runs the full suite with `--require-native --require-cuda
     --require-reference --run-slow` in a scrubbed environment.
   - `tools/gpu_report.py record` writes `reports/gpu-<commit>.json`. The
     report holds the wheel's sha256, the JUnit totals, the device list, and
     the build info.
4. **publish**: `tools/gpu_report.py verify` refuses to continue unless the
   report covers every wheel byte for byte and the run passed. PyPI trusted
   publishing then uploads the sdist and wheel with attestations.

`make sdist wheel gpu-verify` runs steps 1 to 3 locally, apart from the
manylinux container.

A wheel built locally with the conda toolchain is tagged `linux_x86_64`. It
needs the libstdc++ and glibc of the machine that built it; auditwheel rates
it `manylinux_2_39`. Only the cibuildwheel wheel targets `manylinux_2_28`: in
that image, gcc-toolset links newer libstdc++ symbols statically.

Before the first release, the repository needs four things set up:
- a PyPI trusted publisher for `release.yml` and the `pypi` environment;
- a `gpu` environment with required reviewers;
- a self-hosted runner with the `cuda` label, a driver that supports CUDA 13.1,
  and network access to PyPI;
- protected `v*` tags.

Only maintainers can trigger the workflow, because only they can publish a
release. The self-hosted runner never runs pull-request code.

### Release checklist

1. Set the new version in `project.version` (`pyproject.toml`), the
   fallback `VANDERFAST_VERSION` in `CMakeLists.txt`, and `version` and
   `date-released` in `CITATION.cff`; add the release's section to
   `CHANGELOG.md`. The version follows semantic versioning: a change that
   alters results, the public API, or the bundled tables is a major release.
2. `make lint test-gpu cpp-test sanitize data-check fixtures-check` on a GPU
   machine, then `make sdist wheel gpu-verify` (which writes
   `reports/gpu-<commit>.json`); update the support matrix above and the
   release-artifact counts in `docs/validation.md` from that run.
3. Commit, tag the commit `v<version>`, and push the tag. Publish a GitHub
   release from the tag with the `CHANGELOG.md` section as its notes; that
   triggers `release.yml`, whose `gpu` environment waits for a reviewer
   before the self-hosted job runs.
4. After `publish` succeeds, check `pip install vanderfast==<version>` in a
   fresh virtual environment and `vanderfast info --cuda` on a GPU machine.
