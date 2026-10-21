# Performance

Every number here was measured on one machine and is reproduced by the
commands below; the raw `vanderfast.benchmark/1` JSON files are not committed
(`benchmarks/results/` is ignored), but each records the commit, build,
hardware, parameters, and per-sample timings.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER (AD103, sm_89, 16 GB), driver 615.71.09. Consumer Ada: FP64 runs at 1/64 of the FP32 rate, about 0.8 TFLOP/s peak |
| CPU | Intel Core i7-14700KF, 28 logical CPUs |
| vanderfast | release-architecture wheel built at commit 7ee4352 (`make wheel`), installed in a fresh virtual environment (`make gpu-verify`) |
| Toolchain | CUDA 13.1.115, GCC 15.3, CPython 3.14.8, NumPy 2.5.4 |
| Reference | simple-dftd3 1.6.0, conda-forge build (OpenMP), `OMP_NUM_THREADS=28` |
| Model | PBE D3(BJ); "pair" is `s9 = 0`, "atm" is `s9 = 1` with identical pair coefficients |
| Cutoffs | defaults: 40 bohr (CN), 60 bohr (pair), 40 bohr (ATM); skin 1 Å |

```bash
# vanderfast: the release wheel in the fresh venv that `make wheel gpu-verify` creates
OMP_NUM_THREADS=28 build/gpu-venv/bin/python benchmarks/run.py \
  --workloads benzene-cluster,water-box,mgo-rocksalt,skewed-argon,small-cell \
  --sizes 64,256,1024,4096 --atm both --repeats 5 --warmup 1 \
  --modes cold,warm-native,warm-ase-ef,warm-ase-efs,rebuild,memory \
  --reference-max-atoms 1024 --output benchmarks/results/v0.1.0-wheel.json
# dftd3: the OpenMP conda-forge build in the vanderfast-dev environment
OMP_NUM_THREADS=28 python benchmarks/run.py \
  --workloads benzene-cluster,water-box,mgo-rocksalt,skewed-argon,small-cell \
  --sizes 64,256,1024 --atm both --repeats 1 --warmup 0 --modes memory \
  --reference-max-atoms 1024 --output benchmarks/results/v0.1.0-dftd3-omp.json
python benchmarks/report.py benchmarks/results/v0.1.0-wheel.json \
  --reference-timings benchmarks/results/v0.1.0-dftd3-omp.json
```

### Workloads

| Workload | Structure |
| --- | --- |
| `benzene-cluster` | benzene molecules, randomly rotated, on a jittered 7 Å grid; non-periodic and sparse |
| `water-box` | randomly oriented water at 1 g/cm³ in a cubic periodic box |
| `mgo-rocksalt` | cubic MgO supercells (a = 4.21 Å): a dense mixed-element solid |
| `skewed-argon` | primitive fcc argon supercells (60° cell angles) |
| `small-cell` | the 2-atom primitive NaCl cell: thousands of periodic images per atom |

### What is timed

- **Warm native.** `D3Context.evaluate` for energy and forces. Positions are
  perturbed by up to 1/20 of the skin before every call, so nothing is cached,
  but the neighbour list is reused.
- **ASE.** The same through `D3Calculator`. The harness fails the case if any
  call was served from ASE's result cache. E+F+S also requests the stress; on a
  3D cell the force evaluation computes it as well.
- **Rebuild.** A warm call after `invalidate_neighbors()`.
- **Cold.** A fresh process: `import vanderfast`, context creation (CUDA
  initialisation, table upload), and the first evaluation, which includes the
  first neighbour build.
- **dftd3.** The median of 3 energy, gradient, and virial calls. It runs
  after every vanderfast measurement, so its OpenMP threads never share the
  CPU with a timed call.

All vanderfast times are wall-clock medians of 5 calls after 1 warm-up. They
include host-device transfers and the Python layer. GPU stage times come from
CUDA events (`profile=True`), which add no synchronisation.

## Results

### Warm evaluation

| case | edges (pair / ATM) | ATM triangles | warm native ms (p10-p90) | ASE E+F ms | ASE E+F+S ms | rebuild ms | dftd3 ms | speedup | max ΔF eV/Å |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benzene-cluster/N=60/pair | 3,540 / 0 | 0 | 0.131 (0.124-0.382) | 0.189 | n/a | 0.141 | 0.415 | 3x | 4.0e-17 |
| benzene-cluster/N=60/atm | 3,540 / 3,540 | 34,220 | 0.408 (0.401-0.566) | 0.486 | n/a | 0.417 | 0.517 | 1x | 4.0e-17 |
| benzene-cluster/N=252/pair | 63,252 / 0 | 0 | 0.222 (0.201-0.26) | 0.244 | n/a | 0.189 | 0.929 | 4x | 8.8e-17 |
| benzene-cluster/N=252/atm | 63,252 / 61,350 | 2,287,269 | 3.41 (3.05-3.43) | 3.35 | n/a | 3.31 | 9.03 | 3x | 8.5e-17 |
| benzene-cluster/N=1020/pair | 962,018 / 0 | 0 | 0.78 (0.746-0.975) | 0.807 | n/a | 0.969 | 9.82 | 13x | 2.2e-16 |
| benzene-cluster/N=1020/atm | 962,018 / 583,654 | 35,793,036 | 39 (37.8-39.3) | 37 | n/a | 35.8 | 147 | 4x | 2.0e-16 |
| benzene-cluster/N=4092/pair | 8,667,586 / 0 | 0 | 4.49 (4.17-4.67) | 4.36 | n/a | 6.46 | n/a | n/a | n/a |
| benzene-cluster/N=4092/atm | 8,667,586 / 3,722,932 | 281,472,057 | 309 (300-312) | 309 | n/a | 301 | n/a | n/a | n/a |
| water-box/N=63/pair | 724,896 / 0 | 0 | 0.45 (0.399-0.51) | 0.582 | 0.504 | 4.98 | 1.52 | 3x | 1.4e-16 |
| water-box/N=63/atm | 724,896 / 224,346 | 47,416,472 | 59.8 (59.3-60.3) | 59.6 | 60.4 | 63.7 | 345 | 6x | 3.8e-16 |
| water-box/N=255/pair | 2,554,768 / 0 | 0 | 1.19 (1.17-1.23) | 1.24 | 1.24 | 5.48 | 5.34 | 4x | 1.9e-16 |
| water-box/N=255/atm | 2,554,768 / 799,666 | 148,112,795 | 179 (178-180) | 181 | 174 | 184 | 1034 | 6x | 3.0e-16 |
| water-box/N=1023/pair | 15,033,830 / 0 | 0 | 6.31 (5.89-6.33) | 6.31 | 6.32 | 31.1 | 43.6 | 7x | 2.3e-16 |
| water-box/N=1023/atm | 15,033,830 / 4,641,326 | 1,252,839,294 | 1505 (1492-1520) | 1517 | 1509 | 1538 | 6756 | 4x | 2.8e-16 |
| water-box/N=4095/pair | 47,332,024 / 0 | 0 | 19.6 (19.4-20.3) | 19.8 | 19.6 | 62.6 | n/a | n/a | n/a |
| water-box/N=4095/atm | 47,332,024 / 14,998,206 | 3,456,525,607 | 3729 (3700-3958) | 3622 | 3520 | 3563 | n/a | n/a | n/a |
| mgo-rocksalt/N=64/pair | 1,010,560 / 0 | 0 | 0.578 (0.559-0.608) | 0.593 | 0.621 | 4.37 | 1.89 | 3x | 2.4e-15 |
| mgo-rocksalt/N=64/atm | 1,010,560 / 316,416 | 93,879,360 | 102 (102-103) | 102 | 102 | 107 | 611 | 6x | 2.2e-15 |
| mgo-rocksalt/N=512/pair | 8,084,480 / 0 | 0 | 3.19 (3.18-3.21) | 3.29 | 3.35 | 8.76 | 14 | 4x | 3.3e-15 |
| mgo-rocksalt/N=512/atm | 8,084,480 / 2,531,328 | 751,034,880 | 817 (814-893) | 867 | 865 | 826 | 5651 | 7x | 3.7e-15 |
| mgo-rocksalt/N=1728/pair | 27,285,120 / 0 | 0 | 10.4 (10.3-10.5) | 10.4 | 10.5 | 66.4 | n/a | n/a | n/a |
| mgo-rocksalt/N=1728/atm | 27,285,120 / 8,543,232 | 2,534,742,720 | 2918 (2875-2940) | 3093 | 3044 | 2996 | n/a | n/a | n/a |
| mgo-rocksalt/N=4096/pair | 64,675,840 / 0 | 0 | 25.5 (25.4-25.6) | 25.8 | 25.8 | 63.4 | n/a | n/a | n/a |
| mgo-rocksalt/N=4096/atm | 64,675,840 / 20,250,624 | 6,008,279,040 | 6682 (6648-6686) | 6911 | 7120 | 6722 | n/a | n/a | n/a |
| skewed-argon/N=64/pair | 261,888 / 0 | 0 | 0.204 (0.199-0.444) | 0.288 | 0.323 | 2.43 | 0.804 | 4x | 7.2e-17 |
| skewed-argon/N=64/atm | 261,888 / 80,128 | 5,595,008 | 13.3 (13.3-13.4) | 13.5 | 13.5 | 15.5 | 54.3 | 4x | 7.1e-17 |
| skewed-argon/N=343/pair | 1,403,556 / 0 | 0 | 0.661 (0.64-0.783) | 0.752 | 0.799 | 4.74 | 3.34 | 5x | 1.0e-16 |
| skewed-argon/N=343/atm | 1,403,556 / 429,436 | 29,985,746 | 43 (42.8-43.4) | 42.8 | 42.8 | 46.6 | 194 | 5x | 1.0e-16 |
| skewed-argon/N=1331/pair | 5,446,452 / 0 | 0 | 2.14 (2.06-2.36) | 2.27 | 2.34 | 7.61 | n/a | n/a | n/a |
| skewed-argon/N=1331/atm | 5,446,452 / 1,666,412 | 116,358,682 | 155 (155-156) | 153 | 156 | 161 | n/a | n/a | n/a |
| skewed-argon/N=4096/pair | 16,760,832 / 0 | 0 | 6.24 (6.22-6.43) | 6.48 | 6.37 | 45 | n/a | n/a | n/a |
| skewed-argon/N=4096/atm | 16,760,832 / 5,128,192 | 358,080,512 | 472 (466-475) | 476 | 477 | 514 | n/a | n/a | n/a |
| small-cell/N=2/pair | 13,236 / 0 | 0 | 0.152 (0.147-0.315) | 0.223 | 0.252 | 1.44 | 0.411 | 3x | 3.5e-18 |
| small-cell/N=2/atm | 13,236 / 4,012 | 499,566 | 5.65 (5.63-5.74) | 5.74 | 5.82 | 6.89 | 79.9 | 14x | 3.8e-18 |

### GPU stages (warm native, ms)

| case | upload | neighbors | cn_weights | pair | atm | cn_reverse | reduce | GPU total | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benzene-cluster/N=60/pair | 0.00598 | 0.00205 | 0.0236 | 0.0133 | 0.00102 | 0.0143 | 0.0196 | 0.0798 | 0.131 |
| benzene-cluster/N=60/atm | 0.0125 | 0.0017 | 0.0236 | 0.0133 | 0.288 | 0.0144 | 0.0178 | 0.372 | 0.408 |
| benzene-cluster/N=252/pair | 0.00902 | 0.00102 | 0.0389 | 0.0287 | 0.00102 | 0.0317 | 0.0189 | 0.129 | 0.222 |
| benzene-cluster/N=252/atm | 0.0115 | 0.00102 | 0.0399 | 0.0287 | 2.79 | 0.0392 | 0.0191 | 2.93 | 3.41 |
| benzene-cluster/N=1020/pair | 0.0169 | 0.00288 | 0.15 | 0.256 | 0.00102 | 0.209 | 0.0183 | 0.654 | 0.78 |
| benzene-cluster/N=1020/atm | 0.0108 | 0.00102 | 0.148 | 0.256 | 38 | 0.228 | 0.0204 | 38.7 | 39 |
| benzene-cluster/N=4092/pair | 0.0243 | 0.00102 | 0.717 | 2.05 | 0.00102 | 1.13 | 0.0249 | 3.95 | 4.49 |
| benzene-cluster/N=4092/atm | 0.024 | 0.0017 | 0.716 | 2.38 | 305 | 1.14 | 0.0247 | 309 | 309 |
| water-box/N=63/pair | 0.00922 | 0.00182 | 0.0604 | 0.18 | 0.00102 | 0.0789 | 0.0165 | 0.348 | 0.45 |
| water-box/N=63/atm | 0.0057 | 0.00102 | 0.0595 | 0.18 | 58.9 | 0.0799 | 0.0204 | 59.2 | 59.8 |
| water-box/N=255/pair | 0.0113 | 0.00102 | 0.176 | 0.593 | 0.00102 | 0.262 | 0.0171 | 1.06 | 1.19 |
| water-box/N=255/atm | 0.00947 | 0.00102 | 0.175 | 0.594 | 177 | 0.266 | 0.02 | 178 | 179 |
| water-box/N=1023/pair | 0.282 | 0.00186 | 0.846 | 3.65 | 0.00102 | 1.35 | 0.0186 | 6.15 | 6.31 |
| water-box/N=1023/atm | 0.159 | 0.00205 | 0.847 | 3.35 | 1499 | 1.37 | 0.0186 | 1505 | 1505 |
| water-box/N=4095/pair | 0.0289 | 0.00198 | 2.69 | 11.5 | 0.00102 | 5.15 | 0.0249 | 19.4 | 19.6 |
| water-box/N=4095/atm | 0.0258 | 0.00102 | 2.91 | 11 | 3711 | 4.66 | 0.0258 | 3729 | 3729 |
| mgo-rocksalt/N=64/pair | 0.0106 | 0.00102 | 0.0768 | 0.254 | 0.00102 | 0.114 | 0.0146 | 0.472 | 0.578 |
| mgo-rocksalt/N=64/atm | 0.0101 | 0.00176 | 0.0768 | 0.254 | 102 | 0.116 | 0.0186 | 102 | 102 |
| mgo-rocksalt/N=512/pair | 0.00749 | 0.00102 | 0.476 | 1.82 | 0.00102 | 0.805 | 0.0204 | 3.13 | 3.19 |
| mgo-rocksalt/N=512/atm | 0.0104 | 0.00102 | 0.482 | 1.82 | 814 | 0.81 | 0.0174 | 817 | 817 |
| mgo-rocksalt/N=1728/pair | 0.0177 | 0.00224 | 1.55 | 6.02 | 0.00102 | 2.63 | 0.0195 | 10.2 | 10.4 |
| mgo-rocksalt/N=1728/atm | 0.0811 | 0.00205 | 1.55 | 6.21 | 2907 | 2.65 | 0.019 | 2917 | 2918 |
| mgo-rocksalt/N=4096/pair | 0.0246 | 0.00102 | 3.97 | 14.9 | 0.00102 | 6.48 | 0.0251 | 25.4 | 25.5 |
| mgo-rocksalt/N=4096/atm | 0.0248 | 0.00614 | 3.99 | 14.9 | 6656 | 6.5 | 0.0247 | 6681 | 6682 |
| skewed-argon/N=64/pair | 0.00563 | 0.00102 | 0.0287 | 0.0797 | 0.00102 | 0.0307 | 0.0192 | 0.166 | 0.204 |
| skewed-argon/N=64/atm | 0.00592 | 0.00102 | 0.0287 | 0.0799 | 13.1 | 0.0313 | 0.0186 | 13.3 | 13.3 |
| skewed-argon/N=343/pair | 0.00835 | 0.00102 | 0.102 | 0.32 | 0.00102 | 0.147 | 0.0193 | 0.599 | 0.661 |
| skewed-argon/N=343/atm | 0.00813 | 0.00102 | 0.103 | 0.32 | 42.3 | 0.149 | 0.0192 | 42.9 | 43 |
| skewed-argon/N=1331/pair | 0.0674 | 0.00102 | 0.316 | 1.15 | 0.00102 | 0.495 | 0.0211 | 2.05 | 2.14 |
| skewed-argon/N=1331/atm | 0.0158 | 0.00227 | 0.322 | 1.15 | 153 | 0.5 | 0.0174 | 155 | 155 |
| skewed-argon/N=4096/pair | 0.124 | 0.00102 | 0.934 | 3.49 | 0.00176 | 1.49 | 0.0247 | 6.07 | 6.24 |
| skewed-argon/N=4096/atm | 0.0281 | 0.00102 | 0.937 | 3.7 | 466 | 1.5 | 0.0236 | 472 | 472 |
| small-cell/N=2/pair | 0.00512 | 0.00102 | 0.0276 | 0.0287 | 0.00176 | 0.0297 | 0.0186 | 0.112 | 0.152 |
| small-cell/N=2/atm | 0.0048 | 0.00102 | 0.0276 | 0.0297 | 5.49 | 0.0304 | 0.0186 | 5.61 | 5.65 |

### Cold start

| case | import ms | context ms | first evaluation ms | device MB |
| --- | ---: | ---: | ---: | ---: |
| benzene-cluster/N=60/pair | 53.7 | 88.4 | 1.82 | 0.1 |
| benzene-cluster/N=60/atm | 55.9 | 86.6 | 5.02 | 0.1 |
| benzene-cluster/N=252/pair | 84.8 | 85.7 | 2.08 | 1.4 |
| benzene-cluster/N=252/atm | 61.4 | 90.5 | 7.57 | 1.5 |
| benzene-cluster/N=1020/pair | 92.2 | 80.7 | 2.74 | 19.3 |
| benzene-cluster/N=1020/atm | 70.8 | 98.4 | 43.2 | 19.8 |
| benzene-cluster/N=4092/pair | 107 | 83.6 | 8.26 | 170.9 |
| benzene-cluster/N=4092/atm | 91.9 | 80.6 | 315 | 174.1 |
| water-box/N=63/pair | 54.1 | 92.1 | 5.17 | 14.1 |
| water-box/N=63/atm | 53.6 | 90.8 | 68.8 | 14.3 |
| water-box/N=255/pair | 54.2 | 85.7 | 6.46 | 49.8 |
| water-box/N=255/atm | 52.9 | 123 | 179 | 50.4 |
| water-box/N=1023/pair | 54.1 | 87.5 | 32.4 | 292.4 |
| water-box/N=1023/atm | 94 | 84.8 | 1548 | 293.2 |
| water-box/N=4095/pair | 99.3 | 129 | 63.8 | 921.1 |
| water-box/N=4095/atm | 77.5 | 124 | 3869 | 922.1 |
| mgo-rocksalt/N=64/pair | 52 | 80.9 | 5.06 | 19.7 |
| mgo-rocksalt/N=64/atm | 52.5 | 78.1 | 109 | 19.9 |
| mgo-rocksalt/N=512/pair | 52 | 80.4 | 9.95 | 157.2 |
| mgo-rocksalt/N=512/atm | 52.2 | 80.1 | 829 | 157.9 |
| mgo-rocksalt/N=1728/pair | 52.4 | 78.9 | 68 | 530.5 |
| mgo-rocksalt/N=1728/atm | 91.1 | 82.6 | 3033 | 531.3 |
| mgo-rocksalt/N=4096/pair | 102 | 120 | 65.3 | 1257.6 |
| mgo-rocksalt/N=4096/atm | 54 | 94.9 | 6775 | 1258.3 |
| skewed-argon/N=64/pair | 53.6 | 99.1 | 3.13 | 5.1 |
| skewed-argon/N=64/atm | 53.7 | 86 | 19 | 5.2 |
| skewed-argon/N=343/pair | 52.9 | 88.2 | 5.63 | 27.4 |
| skewed-argon/N=343/atm | 53.4 | 89.2 | 51.4 | 27.8 |
| skewed-argon/N=1331/pair | 53.1 | 82.1 | 8.82 | 106.4 |
| skewed-argon/N=1331/atm | 75 | 93.1 | 164 | 107.8 |
| skewed-argon/N=4096/pair | 57.3 | 83.5 | 49.4 | 327.4 |
| skewed-argon/N=4096/atm | 53.5 | 93.7 | 518 | 330.2 |
| small-cell/N=2/pair | 74 | 80.2 | 2.48 | 0.3 |
| small-cell/N=2/atm | 66 | 84.9 | 10.7 | 0.3 |

The dftd3 column is the conda-forge build (OpenMP) on 28 threads, median of 3
calls, from a separate pass (`v0.1.0-dftd3-omp.json`, `report.py
--reference-timings`). The PyPI `dftd3` 1.6.0 wheel is built without OpenMP; with it
the reference is 1–15x slower again (ATM cases about 10–15x), so speedups
quoted against it would overstate vanderfast's advantage. The force
differences are against the PyPI wheel in the timed run (the same 1.6.0
source).

Agreement with dftd3 across the 22 compared cases is at the level of FP64
rounding for sums of 10⁶–10⁹ terms taken in a different order:

| Quantity | Largest difference |
| --- | --- |
| energy per atom | 6.9e-15 eV |
| total energy | 2.5e-12 eV |
| force component | 3.7e-15 eV/Å |
| stress (16 periodic cases) | 8.0e-14 eV/Å³ |

The targets are 1e-8 eV + N·1e-10 eV, 1e-7 eV/Å and 1e-8 eV/Å³.

In the stage table, values of about 0.001 ms are the CUDA-event resolution
floor for stages that did not run (ATM with `s9 = 0`, `neighbors` without a
rebuild).

## Optimisation history

Warm-native medians in ms at each M6 step. The M5–M6c columns are the
development build (sm_89 only, RelWithDebInfo), measured while each step was
developed. The last column is the release wheel from the table above. M5 is the literal reference implementation:
scatter-everything pair and CN-reverse kernels, and ATM over every anchored
triangle with weight 1/3. Each step was checked against that literal path to
1e-12 relative (`tests/cuda/test_variants.py`; the literal kernels remain
available as `kernels="literal"`).

| case | M5 literal | M6a | M6b | M6c | release wheel | M5 / wheel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| benzene-cluster/N=1020/atm | 355 | 66.3 | – | 37.6 | 39 | 9.1x |
| benzene-cluster/N=1020/pair | 1.2 | 0.811 | 0.666 | – | 0.78 | 1.5x |
| benzene-cluster/N=252/atm | 21.2 | 5.52 | – | 2.93 | 3.41 | 6.2x |
| benzene-cluster/N=252/pair | 0.18 | 0.153 | 0.153 | – | 0.222 | 0.8x |
| benzene-cluster/N=60/atm | 0.957 | 0.725 | – | 0.451 | 0.408 | 2.3x |
| benzene-cluster/N=60/pair | 0.109 | 0.11 | 0.109 | – | 0.131 | 0.8x |
| mgo-rocksalt/N=1728/atm | 25380 | 6420 | – | 3081 | 2918 | 8.7x |
| mgo-rocksalt/N=1728/pair | 13.3 | 11.6 | 11.8 | – | 10.4 | 1.3x |
| mgo-rocksalt/N=512/atm | 7068 | 1832 | – | 905 | 817 | 8.6x |
| mgo-rocksalt/N=512/pair | 4.49 | 3.81 | 3.55 | – | 3.19 | 1.4x |
| mgo-rocksalt/N=64/atm | 887 | 238 | – | 112 | 102 | 8.7x |
| mgo-rocksalt/N=64/pair | 2.31 | 1.91 | 0.505 | – | 0.578 | 4.0x |
| small-cell/N=2/atm | 18.1 | 11.1 | – | 6.15 | 5.65 | 3.2x |
| small-cell/N=2/pair | 0.593 | 0.53 | 0.142 | – | 0.152 | 3.9x |
| water-box/N=1023/atm | 11581 | 2895 | – | 1561 | 1505 | 7.7x |
| water-box/N=1023/pair | 9.24 | 7.4 | 6.61 | – | 6.31 | 1.5x |
| water-box/N=255/atm | 1348 | 346 | – | 180 | 179 | 7.6x |
| water-box/N=255/pair | 1.57 | 1.21 | 1.14 | – | 1.19 | 1.3x |
| water-box/N=63/atm | 438 | 112 | – | 61.1 | 59.8 | 7.3x |
| water-box/N=63/pair | 1.8 | 1.41 | 0.395 | – | 0.45 | 4.0x |

- **M6a.** Owner-only pair and CN-reverse kernels (no atomics, full-weight
  partials). Species-contracted C6 coefficients `U = wᵀK`, which make each
  edge's C6 and its derivatives cost O(M) instead of O(M²). A canonical ATM
  anchor, so each triangle is visited once instead of three times.
- **M6b.** Row-chunked work items of 256 edges for CN, pair and CN reverse,
  with deterministic in-order reduction. This balances rows of very different
  length (the small cell has 6600 edges per atom) and is what makes small
  periodic systems 3–4× faster.
- **M6c.** Shared-memory tiles in the ATM kernel: each warp prepares 32 inner
  sites once and rotates them through its lanes. Also `rsqrt`, `cbrt` for the
  default exponent 16 in place of `pow`, and fewer divisions.

Pair-only calls on small sparse molecules (benzene 60 and 252) take
0.1–0.25 ms. That is within the session-to-session spread of launch-bound
calls: the focused M6b run measured 0.109 and 0.153 ms, faster than M5. The
release-wheel session, at 0.131 and 0.222 ms, came out 20–40 µs slower than
M5. Nothing in M6 targets these cases, and no change is claimed for them.
The other rows are GPU-bound and their gains are well outside the noise.

## Where the time goes

Above about 1000 atoms the ATM stage is over 98% of GPU time whenever ATM
is on (99.6% for MgO 4096). Pair-only evaluations split between the pair
kernel (about 60%), the CN reverse pass (25%), and CN plus weights (15%).

- **ATM is FP64-bound.** At the default 40 bohr cutoff a dense solid has
  about 1.5 million accepted triangles per atom (MgO). The triangle loop is
  pure FP64 arithmetic on register and shared-memory operands. It reaches
  0.8–0.9 billion accepted triangles per second (with derivatives) on this
  card, whose FP64 rate is 1/64 of FP32. A data-centre GPU with full-rate FP64
  (A100, H100) should be much faster, but none has been measured. The ATM
  kernels use 150–154 registers on sm_89 and 179–183 on sm_100/sm_120, with
  no spills on any release architecture (`tools/check_wheel.py`).
- **The pair term streams edges.** Each 16-byte edge record is read by CN,
  pair, and CN reverse. MgO 4096 has 64.7 million edges and its three pair-side
  stages take 25 ms: about 3 GB of edge reads plus position
  gathers. That is bandwidth-bound, not arithmetic-bound.
- **Small systems are latency-bound.** Below about 100 atoms (and for the
  2-atom cell, pair only) a warm call is 11 small kernels, each running only a
  few warps. See the next section.
- **Rebuilds.** A neighbour rebuild adds up to about 60 ms to a call (pair-only
  rebuild minus warm medians above: 0–5 ms below 300 atoms, 25–60 ms for
  1000–4000-atom periodic cells). Counting and filling the edge list dominate.
  With a 1 Å skin a rebuild happens only after some atom has moved 0.5 Å
  since the last one.

## CUDA Graphs: measured, deferred

Capturing the steady-state launch sequence in a CUDA Graph removes per-kernel
launch overhead, but nothing else. The small-system numbers (median of 2000
calls, energy and forces, no profiling):

| System (pair only) | native call | `D3Context.evaluate` | ASE `get_forces` | GPU kernel span | GPU busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-atom NaCl cell (13 236 edges) | 108 µs | 117 µs | 189 µs | 99 µs | 92 µs |
| benzene (12 atoms) | 59 µs | 67 µs | 138 µs | 50 µs | 42 µs |

The span and busy columns come from an Nsight Systems trace of 250 warm calls
each. Every evaluation is 11 kernel launches, one memset, three small copies,
and one stream synchronisation. Kernel launches are asynchronous, so the GPU
is busy for 85–92% of the span; the gaps between kernels total under 8 µs per
call. What remains is the kernels themselves: with 52 work items (2-atom
cell) or a dozen rows (benzene), each kernel runs a handful of warps and is
latency-bound.

The GPU-side pipeline for these systems is tens of microseconds. A graph
would save at most the launch gaps between about a dozen kernels, which is
small next to the Python and ASE overhead per call. It would also add capture
and invalidation logic on every neighbour rebuild, capacity change, and
property set. It is deferred until a workload that needs it (for example
batched small molecules) exists.

## Profiling notes

- **Nsight Compute.** Not used: GPU performance counters are restricted to
  administrators on this machine (`ERR_NVGPUCTRPERM`), and that setting was
  left alone.
- **What was used instead.** Kernel-level evidence comes from CUDA-event
  stage timings, `cuobjdump --dump-resource-usage` (registers, spills), SASS
  inspection, Nsight Systems timelines, and the benchmark harness.
- **Variability.** The same build measured in different sessions varies by
  up to about 8% for GPU-bound ATM cases (GPU boost clocks and temperature).
  Launch-bound cases under 0.5 ms vary by up to 2× (CPU frequency and
  scheduling). The p10–p90 spread within a session is given in the tables.
