# Developer entry points. Run inside the activated vanderfast-dev environment.
PYTHON ?= python
UPSTREAM ?=
UPSTREAM_D4 ?=

.PHONY: help env lock dev dev-cpu lint test test-gpu cpp-test sanitize stubtest sdist wheel gpu-verify bench-smoke bench data data-check fixtures fixtures-check

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1

env:
	conda env create -f build_tools/environment.yml

lock:
	conda list --explicit --md5 > build_tools/conda-linux-64.lock

dev:
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e . \
	  -Ccmake.build-type=RelWithDebInfo -Cbuild-dir=build/dev

dev-cpu:
	VANDERFAST_ENABLE_CUDA=OFF $(PYTHON) -m pip install --no-build-isolation --no-deps -e .

lint:
	prek run --all-files --show-diff-on-failure
	mypy

test:
	$(PYTHON) -m pytest

test-gpu:
	$(PYTHON) -m pytest --require-native --require-cuda --require-reference --run-slow

cpp-test:
	cmake -S . -B build/cpp -G Ninja -DVANDERFAST_BUILD_TESTS=ON -DVANDERFAST_WERROR=ON \
	  -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build/cpp
	ctest --test-dir build/cpp --output-on-failure

SANITIZER ?= compute-sanitizer
sanitize:
	$(SANITIZER) --tool memcheck --leak-check full --error-exitcode 99 $(PYTHON) tools/sanitizer_workload.py --atm
	$(SANITIZER) --tool racecheck --error-exitcode 99 $(PYTHON) tools/sanitizer_workload.py --atm
	$(SANITIZER) --tool synccheck --error-exitcode 99 $(PYTHON) tools/sanitizer_workload.py --atm
	$(SANITIZER) --tool initcheck --error-exitcode 99 $(PYTHON) tools/sanitizer_workload.py --atm

bench-smoke:
	$(PYTHON) benchmarks/run.py --suite smoke --output build/bench-smoke.json

bench:
	$(PYTHON) benchmarks/run.py --suite standard

stubtest:
	cd /tmp && $(PYTHON) -m mypy.stubtest vanderfast._native --allowlist $(CURDIR)/tests/stubtest_allowlist.txt

sdist:
	$(PYTHON) -m build --sdist --no-isolation
	$(PYTHON) tools/check_sdist.py dist/*.tar.gz
	$(PYTHON) -m twine check --strict dist/*.tar.gz

wheel:
	VANDERFAST_RELEASE_BUILD=ON VANDERFAST_WERROR=ON \
	VANDERFAST_CUDA_ARCHITECTURES=release \
	  $(PYTHON) -m build --wheel --no-isolation
	$(PYTHON) tools/check_wheel.py dist/*.whl

# The release workflow's gpu-verify job, locally: install dist/*.whl into a
# fresh virtual environment with PyPI dependencies only, run the whole suite on
# the GPU with a scrubbed environment, and write reports/gpu-<commit>.json.
GPU_VENV ?= build/gpu-venv
gpu-verify:
	rm -rf $(GPU_VENV)
	$(PYTHON) -m venv $(GPU_VENV)
	$(GPU_VENV)/bin/python -m pip install --quiet dist/*.whl "dftd3==1.6.0" "dftd4==4.2.0" pytest pytest-cov
	cd build && env -i HOME="$$HOME" PATH=/usr/bin:/bin $(CURDIR)/$(GPU_VENV)/bin/python -m pytest \
	  $(CURDIR)/tests --rootdir $(CURDIR) --require-native --require-cuda --require-reference \
	  --run-slow --no-cov -p no:cacheprovider --junitxml $(CURDIR)/build/gpu-junit.xml
	env -i HOME="$$HOME" PATH=/usr/bin:/bin $(GPU_VENV)/bin/python tools/gpu_report.py record \
	  --dist dist --junit build/gpu-junit.xml --output reports/gpu-$$(git rev-parse HEAD).json

data:
	$(PYTHON) -I tools/generate_d3_data.py $(UPSTREAM)
	$(PYTHON) -I tools/generate_d4_data.py $(UPSTREAM_D4)

data-check:
	$(PYTHON) -I tools/generate_d3_data.py --check --verify-binary "$$CONDA_PREFIX/lib" $(UPSTREAM)
	$(PYTHON) -I tools/generate_d4_data.py --check --verify-binary "$$CONDA_PREFIX/lib" $(UPSTREAM_D4)

fixtures:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) tools/generate_d4_fixtures.py

fixtures-check:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py --check
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) tools/generate_d4_fixtures.py --check
