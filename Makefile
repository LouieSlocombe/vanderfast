# Developer entry points. Run inside the activated vanderfast-dev environment.
PYTHON ?= python
UPSTREAM ?=

.PHONY: help env lock dev dev-cpu lint test test-gpu cpp-test sanitize stubtest wheel data data-check fixtures fixtures-check

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
	prek run --all-files
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

stubtest:
	cd /tmp && $(PYTHON) -m mypy.stubtest vanderfast._native --allowlist $(CURDIR)/tests/stubtest_allowlist.txt

wheel:
	VANDERFAST_RELEASE_BUILD=ON VANDERFAST_WERROR=ON \
	VANDERFAST_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" \
	  $(PYTHON) -m build --wheel --no-isolation
	$(PYTHON) tools/check_wheel.py dist/*.whl

data:
	$(PYTHON) -I tools/generate_d3_data.py $(UPSTREAM)

data-check:
	$(PYTHON) -I tools/generate_d3_data.py --check --verify-binary "$$CONDA_PREFIX/lib" $(UPSTREAM)

fixtures:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py

fixtures-check:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py --check
