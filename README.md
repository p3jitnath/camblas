# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 kernels, packed operand panels and shape-dependent task grids. Bounded paths use twelve-row FP32 or six-row FP64 layouts; transposed FP32 inputs can be packed directly into the kernel layout without an intermediate panel. Bounded square and transposed products also use one-level Strassen: seven half-size products, with operand sums formed during packing. Every call repacks its inputs, and the conservative finite-range preflight rides on the pack pass itself (per-worker abs-maxima reduced after packing) so no separate operand scan runs before the packed route; unsafe ranges still fall back to the classical path. At 64 cores a persistent spinning worker-pool executor replaces per-call OpenMP fork/join (workers spin briefly after each dispatch and then sleep on futex; below 64 threads the OpenMP executor is retained). Runtime capability, bounds, workspace and finite-range checks retain classical fallbacks. Strassen changes rounding behaviour; the range check limits overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot: **60/60 wins against OpenBLAS; 48/60 against NVPL**. The table contains all 60 cases: seven NumPy and eight PyTorch workloads, each at two precisions and 16/64 cores. **† marks 38 cases measured with this source configuration on an idle whole-node allocation:** all 30 64-core cases, whose executor the spinning worker pool replaces, plus the marked 16-core cases from the previous snapshot, whose execution paths are unchanged below 64 threads. The remaining unmarked 16-core cases use measurements from separate allocations and earlier CAMBLAS builds whose corresponding execution paths are unchanged; they have not been remeasured with this build, so the table remains a combined snapshot, not one complete run. Small median differences do not establish statistical significance, so entries within roughly two percent of 1× can move between sessions; and the PyTorch 64-core OpenBLAS medians vary several-fold between sessions on this allocation, so those entries are conservative rather than best-case for OpenBLAS.

FP32 and FP64 denote single and double precision, respectively. Latency is reported in milliseconds, with lower values indicating better performance. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1× favour CAMBLAS**. Each entry is the median of three fresh-process medians after three warm-ups. The eight marked 16-core square and transpose cases use 1001 timed calls per NumPy round and 201 per PyTorch round; the 30 marked 64-core cases use at least five calls per NumPy round and 25 per PyTorch round. Unmarked cases use at least five calls per round; some use the longer 1001/201-call protocol.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Square 1024† | FP32 | 16 | 1.352 | 1.622 | 1.435 | 1.199× | 1.061× |
| NumPy | Square 1024† | FP32 | 64 | 0.409 | 0.501 | 0.385 | 1.227× | 0.943× |
| NumPy | Square 1024† | FP64 | 16 | 2.799 | 3.351 | 2.867 | 1.197× | 1.024× |
| NumPy | Square 1024† | FP64 | 64 | 0.786 | 0.965 | 0.773 | 1.229× | 0.984× |
| NumPy | Square 4096 | FP32 | 16 | 84.093 | 104.127 | 93.619 | 1.238× | 1.113× |
| NumPy | Square 4096† | FP32 | 64 | 25.664 | 34.470 | 36.795 | 1.343× | 1.434× |
| NumPy | Square 4096 | FP64 | 16 | 177.555 | 219.378 | 193.596 | 1.236× | 1.090× |
| NumPy | Square 4096† | FP64 | 64 | 54.481 | 73.629 | 79.259 | 1.351× | 1.455× |
| NumPy | Square 8192 | FP32 | 16 | 653.509 | 822.125 | 720.597 | 1.258× | 1.103× |
| NumPy | Square 8192† | FP32 | 64 | 183.261 | 251.429 | 240.683 | 1.372× | 1.313× |
| NumPy | Square 8192 | FP64 | 16 | 1387.948 | 1736.488 | 1493.413 | 1.251× | 1.076× |
| NumPy | Square 8192† | FP64 | 64 | 385.468 | 530.695 | 484.353 | 1.377× | 1.257× |
| NumPy | Transposed GEMM† | FP32 | 16 | 1.296 | 1.621 | 1.426 | 1.251× | 1.100× |
| NumPy | Transposed GEMM† | FP32 | 64 | 0.394 | 0.505 | 0.377 | 1.281× | 0.955× |
| NumPy | Transposed GEMM† | FP64 | 16 | 2.731 | 3.349 | 2.865 | 1.226× | 1.049× |
| NumPy | Transposed GEMM† | FP64 | 64 | 0.751 | 0.960 | 0.775 | 1.279× | 1.033× |
| NumPy | Gram | FP32 | 16 | 1.102 | 1.746 | 1.160 | 1.584× | 1.053× |
| NumPy | Gram† | FP32 | 64 | 0.420 | 13.342 | 0.574 | 31.778× | 1.368× |
| NumPy | Gram | FP64 | 16 | 2.152 | 2.896 | 2.336 | 1.345× | 1.085× |
| NumPy | Gram† | FP64 | 64 | 0.818 | 27.398 | 1.021 | 33.504× | 1.249× |
| NumPy | MLP forward | FP32 | 16 | 10.692 | 10.730 | 9.965 | 1.004× | 0.932× |
| NumPy | MLP forward† | FP32 | 64 | 4.063 | 4.139 | 3.777 | 1.019× | 0.930× |
| NumPy | MLP forward | FP64 | 16 | 22.569 | 26.578 | 21.522 | 1.178× | 0.954× |
| NumPy | MLP forward† | FP64 | 64 | 8.785 | 20.201 | 12.000 | 2.299× | 1.366× |
| NumPy | Attention | FP32 | 16 | 4.081 | 4.249 | 4.154 | 1.041× | 1.018× |
| NumPy | Attention† | FP32 | 64 | 3.552 | 4.026 | 4.330 | 1.134× | 1.219× |
| NumPy | Attention | FP64 | 16 | 6.956 | 7.394 | 6.768 | 1.063× | 0.973× |
| NumPy | Attention† | FP64 | 64 | 6.100 | 6.413 | 6.207 | 1.051× | 1.018× |
| PyTorch | Square 1024† | FP32 | 16 | 1.366 | 1.628 | 1.442 | 1.192× | 1.055× |
| PyTorch | Square 1024† | FP32 | 64 | 0.421 | 0.508 | 0.400 | 1.205× | 0.950× |
| PyTorch | Square 1024† | FP64 | 16 | 2.823 | 3.378 | 2.884 | 1.197× | 1.021× |
| PyTorch | Square 1024† | FP64 | 64 | 0.807 | 0.968 | 0.785 | 1.200× | 0.972× |
| PyTorch | Square 4096 | FP32 | 16 | 84.131 | 103.732 | 92.992 | 1.233× | 1.105× |
| PyTorch | Square 4096† | FP32 | 64 | 26.073 | 34.484 | 36.703 | 1.323× | 1.408× |
| PyTorch | Square 4096 | FP64 | 16 | 177.810 | 217.491 | 192.401 | 1.223× | 1.082× |
| PyTorch | Square 4096† | FP64 | 64 | 54.685 | 72.256 | 78.260 | 1.321× | 1.431× |
| PyTorch | Square 8192 | FP32 | 16 | 653.124 | 814.144 | 712.992 | 1.247× | 1.092× |
| PyTorch | Square 8192† | FP32 | 64 | 183.209 | 241.877 | 233.113 | 1.320× | 1.272× |
| PyTorch | Square 8192 | FP64 | 16 | 1386.473 | 1709.889 | 1467.706 | 1.233× | 1.059× |
| PyTorch | Square 8192† | FP64 | 64 | 384.730 | 519.914 | 457.687 | 1.351× | 1.190× |
| PyTorch | Transposed GEMM† | FP32 | 16 | 1.420 | 2.477 | 1.597 | 1.745× | 1.125× |
| PyTorch | Transposed GEMM† | FP32 | 64 | 0.510 | 1.873 | 0.540 | 3.674× | 1.060× |
| PyTorch | Transposed GEMM† | FP64 | 16 | 2.977 | 4.969 | 4.038 | 1.669× | 1.356× |
| PyTorch | Transposed GEMM† | FP64 | 64 | 0.999 | 4.011 | 1.542 | 4.014× | 1.543× |
| PyTorch | Gram | FP32 | 16 | 0.984 | 1.682 | 1.512 | 1.710× | 1.537× |
| PyTorch | Gram† | FP32 | 64 | 0.299 | 0.909 | 0.484 | 3.044× | 1.621× |
| PyTorch | Gram | FP64 | 16 | 1.916 | 3.596 | 3.115 | 1.877× | 1.626× |
| PyTorch | Gram† | FP64 | 64 | 0.561 | 2.344 | 0.880 | 4.181× | 1.569× |
| PyTorch | MLP forward | FP32 | 16 | 9.128 | 15.642 | 8.727 | 1.714× | 0.956× |
| PyTorch | MLP forward† | FP32 | 64 | 3.694 | 8.770 | 2.463 | 2.374× | 0.667× |
| PyTorch | MLP forward | FP64 | 16 | 20.420 | 32.244 | 21.466 | 1.579× | 1.051× |
| PyTorch | MLP forward† | FP64 | 64 | 7.366 | 44.668 | 12.172 | 6.064× | 1.652× |
| PyTorch | Attention | FP32 | 16 | 1.202 | 9.479 | 1.290 | 7.887× | 1.073× |
| PyTorch | Attention† | FP32 | 64 | 0.916 | 2.357 | 1.500 | 2.572× | 1.637× |
| PyTorch | Attention | FP64 | 16 | 2.487 | 14.507 | 2.356 | 5.834× | 0.948× |
| PyTorch | Attention† | FP64 | 64 | 1.379 | 31.936 | 2.278 | 23.152× | 1.652× |
| PyTorch | Forward + backward | FP32 | 16 | 23.236 | 52.699 | 25.239 | 2.268× | 1.086× |
| PyTorch | Forward + backward† | FP32 | 64 | 9.429 | 44.957 | 17.762 | 4.768× | 1.884× |
| PyTorch | Forward + backward | FP64 | 16 | 47.952 | 106.468 | 51.584 | 2.220× | 1.076× |
| PyTorch | Forward + backward† | FP64 | 64 | 17.800 | 69.995 | 35.548 | 3.932× | 1.997× |

Measurements used Grace, GCC 14.3, NumPy 2.3.2, PyTorch 2.8.0, OpenBLAS 0.3.33 (pthreads) and NVPL from HPC SDK 24.11 (LP64 GNU OpenMP). All libraries used matching inputs, affinity and requested thread counts. The [multi-layer perceptron workload](bench/workload.py) (MLP) uses batch size 512 and widths 2048 → 4096 → 1024. PyTorch uses the BLAS-focused build below, rather than default wheels or GPUs.

## Build

Requirements: Linux, Python 3.11+, Make and a C11 compiler; GCC 14.3 was tested. Framework builds also need matching C++/Fortran compilers, Git, pkg-config and Python development headers.

```bash
git clone git@github.com:p3jitnath/camblas.git
cd camblas
make test CC=gcc-14 PYTHON=python3.11
make grace CC=gcc-14 PYTHON=python3.11
```

The Grace library is `build/grace/libcamblas_sve_nr4.so`; this target requires Neoverse V2 with SVE. On x86_64, use `make reference CC=cc PYTHON=python3.11`. The default native context is serial; use the framework adapters to reproduce the table. Native API and workspace contracts are in [include/](include/).

## NumPy and PyTorch

Supply shared LP64 OpenBLAS with CBLAS headers and LAPACK symbols; LP64 uses 32-bit BLAS integers. Set the dependency paths below, adjusting NVPL filenames to your installation while retaining the LP64 GNU OpenMP interface.

```bash
export CAMBLAS_OPENBLAS_PREFIX=/path/to/openblas
export CAMBLAS_NVPL_PREFIX=/path/to/nvpl
python3.11 scripts/build.py --target grace --cc gcc-14 --backend all \
  --openblas-lib "$CAMBLAS_OPENBLAS_PREFIX/lib/libopenblas.so" \
  --openblas-include "$CAMBLAS_OPENBLAS_PREFIX/include" \
  --nvpl-blas "$CAMBLAS_NVPL_PREFIX/lib/libnvpl_blas_lp64_gomp.so.0.3.0" \
  --nvpl-lapack "$CAMBLAS_NVPL_PREFIX/lib/libnvpl_lapack_lp64_gomp.so.0.2.3"
```

For CAMBLAS alone, use `--backend camblas` and omit the NVPL arguments. Compile the frameworks on allocated compute resources, adjusting `--jobs` to available CPUs and memory; add `--dry-run` to inspect commands first.

```bash
python3.11 scripts/frameworks.py prepare --jobs 8
for backend in camblas openblas nvpl; do
  python3.11 scripts/frameworks.py numpy --backend "$backend" --jobs 8
  python3.11 scripts/frameworks.py pytorch --backend "$backend" --jobs 8
  python3.11 scripts/frameworks.py install --backend "$backend"
done
```

The [build helper](scripts/frameworks.py) pins sources and dependencies, applies the [PyTorch build patch](patches/pytorch-2.8.0.patch) and installs separate environments under `.frameworks/envs/<backend>`. All variants disable GPUs, alternative acceleration backends and SVE256 uniformly; OpenMP and CAMBLAS SVE128 remain enabled. Installation is project-local, with absolute dependency paths: rebuild after moving the checkout or dependencies.

## Test and reproduce

Run the adapter and framework correctness checks before collecting performance measurements:

```bash
for backend in camblas openblas nvpl; do
  CAMBLAS_FRAMEWORK_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3.11 tests/test_framework_bridge.py "$backend"
  CAMBLAS_FRAMEWORK_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .frameworks/envs/"$backend"/bin/python tests/test_framework_smoke.py "$backend"
done
```

On allocated Grace CPUs, check the batched route against full-output scalar oracles, concurrent calls and exceptional-input fallbacks:

```bash
.frameworks/envs/camblas/bin/python tests/check_batch.py \
  .frameworks/prefix/camblas/lib/libframework_blas.so
# Also run tests/test_mlp_packing.py with --profile batch for dispatch boundaries.
```

For timing, bind the parent process to idle, allocated Grace CPUs within one NUMA domain. The driver uses matching CPU sets for all libraries; leave allocator, preload and OpenMP wait-policy overrides unset.

```bash
# Example targeted comparison; select other workloads and precisions as needed.
python3.11 bench/compare.py --threads 16 64 --workloads mlp backward

# Confirm the marked 16-core square and transpose cases.
python3.11 bench/compare.py --frameworks numpy --threads 16 --dtypes float32 float64 \
  --workloads square1024 transpose --repetitions 1001
python3.11 bench/compare.py --frameworks pytorch --threads 16 --dtypes float32 float64 \
  --workloads square1024 transpose --repetitions 201

# Inspect the complete 60-case matrix without running it.
python3.11 bench/compare.py --full --dry-run

# Request a fresh complete suite only when needed.
python3.11 bench/compare.py --full
```

The driver rotates all three libraries over at least three fresh-process rounds and saves timings, library identities, affinity, call counters and numerical checks under ignored `results/`. It rejects mismatched bridges, native cores or CPU affinity, incomplete rounds and non-finite timings or output signatures. The full matrix has seven NumPy and eight PyTorch workloads at two precisions and two core counts. Application checks use samples and norms; packing changes also require the independent full-output checks in [tests/test_mlp_packing.py](tests/test_mlp_packing.py).

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and [LICENSE](LICENSE) for the MIT licence, copyright Pritthijit Nath. External dependencies retain their own licences; generated artefacts and internal documents are excluded from this repository.
