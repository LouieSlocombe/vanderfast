# Developer entry points. Run inside the activated vanderfast-dev environment.
PYTHON ?= python
UPSTREAM ?=

.PHONY: help env lock dev-cpu lint test data data-check fixtures fixtures-check

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1

env:
	conda env create -f build_tools/environment.yml

lock:
	conda list --explicit --md5 > build_tools/conda-linux-64.lock

dev-cpu:
	VANDERFAST_ENABLE_CUDA=OFF $(PYTHON) -m pip install --no-build-isolation --no-deps -e .

lint:
	prek run --all-files
	mypy

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -I tools/generate_d3_data.py $(UPSTREAM)

data-check:
	$(PYTHON) -I tools/generate_d3_data.py --check --verify-binary "$$CONDA_PREFIX/lib" $(UPSTREAM)

fixtures:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py

fixtures-check:
	OMP_NUM_THREADS=1 $(PYTHON) tools/generate_fixtures.py --check
