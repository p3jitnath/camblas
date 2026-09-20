# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 kernels, packed operand panels and shape-dependent task grids. Bounded paths use twelve-row FP32 or six-row FP64 layouts; transposed FP32 inputs can be packed directly into the kernel layout without an intermediate panel. Bounded square and transposed products also use one-level Strassen: seven half-size products, with operand sums formed during packing. Every call repacks its inputs, and the conservative finite-range preflight rides on the pack pass itself (per-worker abs-maxima reduced after packing) so no separate operand scan runs before the packed route; unsafe ranges still fall back to the classical path. At 64 cores a persistent spinning worker-pool executor replaces per-call OpenMP fork/join (workers spin briefly after each dispatch and then sleep on futex; below 64 threads the OpenMP executor is retained). Runtime capability, bounds, workspace and finite-range checks retain classical fallbacks. Strassen changes rounding behaviour; the range check limits overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot: **59/60 wins against OpenBLAS; 27/60 against NVPL**. The table contains all 60 cases from one complete run of this source configuration on an idle whole-node allocation (Grace, exclusive 288-core node): five NumPy and five PyTorch workloads (attention, gram, MLP, square 1024, transpose), each at two precisions and 4/16/64 cores. Small median differences do not establish statistical significance, so entries within roughly two percent of 1x can move between sessions.

FP32 and FP64 denote single and double precision, respectively. Latency is reported in milliseconds, with lower values indicating better performance. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. Each entry is the median of three fresh-process medians after warm-up, with all four libraries using matching inputs, affinity and requested thread counts.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 4 | 3.860 | 3.851 | 3.508 | 0.998x | 0.909x |
| NumPy | Transpose | FP32 | 4 | 5.683 | 6.396 | 5.575 | 1.125x | 0.981x |
| NumPy | Attention | FP32 | 4 | 6.440 | 6.614 | 6.304 | 1.027x | 0.979x |
| NumPy | Square 1024 | FP32 | 4 | 5.774 | 6.359 | 5.594 | 1.101x | 0.969x |
| NumPy | MLP | FP32 | 4 | 37.255 | 39.261 | 36.269 | 1.054x | 0.974x |
| NumPy | Gram | FP64 | 4 | 8.582 | 8.622 | 7.158 | 1.005x | 0.834x |
| NumPy | Transpose | FP64 | 4 | 12.163 | 13.312 | 11.450 | 1.094x | 0.941x |
| NumPy | Attention | FP64 | 4 | 11.284 | 12.051 | 10.928 | 1.068x | 0.969x |
| NumPy | Square 1024 | FP64 | 4 | 12.011 | 13.368 | 11.554 | 1.113x | 0.962x |
| NumPy | MLP | FP64 | 4 | 80.402 | 83.129 | 73.463 | 1.034x | 0.914x |
| NumPy | Gram | FP32 | 16 | 1.108 | 1.754 | 1.167 | 1.582x | 1.053x |
| NumPy | Transpose | FP32 | 16 | 1.330 | 1.630 | 1.441 | 1.225x | 1.083x |
| NumPy | Attention | FP32 | 16 | 4.117 | 4.276 | 4.087 | 1.039x | 0.993x |
| NumPy | Square 1024 | FP32 | 16 | 1.362 | 1.635 | 1.437 | 1.201x | 1.055x |
| NumPy | MLP | FP32 | 16 | 10.580 | 10.931 | 10.018 | 1.033x | 0.947x |
| NumPy | Gram | FP64 | 16 | 2.141 | 2.902 | 2.352 | 1.356x | 1.099x |
| NumPy | Transpose | FP64 | 16 | 2.782 | 3.386 | 2.896 | 1.217x | 1.041x |
| NumPy | Attention | FP64 | 16 | 6.947 | 7.335 | 6.766 | 1.056x | 0.974x |
| NumPy | Square 1024 | FP64 | 16 | 2.853 | 3.383 | 2.905 | 1.186x | 1.018x |
| NumPy | MLP | FP64 | 16 | 22.294 | 26.062 | 21.497 | 1.169x | 0.964x |
| NumPy | Gram | FP32 | 64 | 0.417 | 13.334 | 0.586 | 31.943x | 1.404x |
| NumPy | Transpose | FP32 | 64 | 0.397 | 0.507 | 0.381 | 1.278x | 0.961x |
| NumPy | Attention | FP32 | 64 | 3.574 | 4.036 | 4.198 | 1.130x | 1.175x |
| NumPy | Square 1024 | FP32 | 64 | 0.412 | 0.505 | 0.390 | 1.225x | 0.946x |
| NumPy | MLP | FP32 | 64 | 4.026 | 4.065 | 3.781 | 1.010x | 0.939x |
| NumPy | Gram | FP64 | 64 | 0.818 | 27.725 | 1.038 | 33.880x | 1.268x |
| NumPy | Transpose | FP64 | 64 | 0.766 | 0.967 | 0.780 | 1.263x | 1.019x |
| NumPy | Attention | FP64 | 64 | 6.070 | 6.362 | 6.252 | 1.048x | 1.030x |
| NumPy | Square 1024 | FP64 | 64 | 0.782 | 0.959 | 0.782 | 1.226x | 1.000x |
| NumPy | MLP | FP64 | 64 | 8.729 | 20.479 | 10.405 | 2.346x | 1.192x |
| PyTorch | Gram | FP32 | 4 | 6.255 | 6.434 | 5.726 | 1.029x | 0.916x |
| PyTorch | Transpose | FP32 | 4 | 5.973 | 6.600 | 5.823 | 1.105x | 0.975x |
| PyTorch | Attention | FP32 | 4 | 3.998 | 7.116 | 3.843 | 1.780x | 0.961x |
| PyTorch | Square 1024 | FP32 | 4 | 5.755 | 6.321 | 5.545 | 1.098x | 0.963x |
| PyTorch | MLP | FP32 | 4 | 36.159 | 39.496 | 34.290 | 1.092x | 0.948x |
| PyTorch | Gram | FP64 | 4 | 12.791 | 13.425 | 11.754 | 1.050x | 0.919x |
| PyTorch | Transpose | FP64 | 4 | 12.793 | 13.757 | 11.829 | 1.075x | 0.925x |
| PyTorch | Attention | FP64 | 4 | 8.314 | 14.912 | 7.988 | 1.794x | 0.961x |
| PyTorch | Square 1024 | FP64 | 4 | 11.988 | 13.220 | 11.454 | 1.103x | 0.955x |
| PyTorch | MLP | FP64 | 4 | 77.959 | 84.808 | 72.520 | 1.088x | 0.930x |
| PyTorch | Gram | FP32 | 16 | 0.996 | 1.695 | 1.512 | 1.702x | 1.518x |
| PyTorch | Transpose | FP32 | 16 | 1.447 | 2.475 | 1.587 | 1.710x | 1.097x |
| PyTorch | Attention | FP32 | 16 | 1.264 | 9.469 | 1.306 | 7.490x | 1.033x |
| PyTorch | Square 1024 | FP32 | 16 | 1.398 | 1.635 | 1.449 | 1.169x | 1.037x |
| PyTorch | MLP | FP32 | 16 | 9.051 | 9.956 | 8.819 | 1.100x | 0.974x |
| PyTorch | Gram | FP64 | 16 | 1.917 | 3.680 | 3.116 | 1.919x | 1.626x |
| PyTorch | Transpose | FP64 | 16 | 2.998 | 5.031 | 3.496 | 1.678x | 1.166x |
| PyTorch | Attention | FP64 | 16 | 2.534 | 14.484 | 2.525 | 5.715x | 0.996x |
| PyTorch | Square 1024 | FP64 | 16 | 2.869 | 3.378 | 2.908 | 1.178x | 1.014x |
| PyTorch | MLP | FP64 | 16 | 20.217 | 26.820 | 20.598 | 1.327x | 1.019x |
| PyTorch | Gram | FP32 | 64 | 0.309 | 0.917 | 0.492 | 2.968x | 1.593x |
| PyTorch | Transpose | FP32 | 64 | 0.534 | 1.817 | 0.522 | 3.404x | 0.978x |
| PyTorch | Attention | FP32 | 64 | 0.929 | 2.400 | 1.496 | 2.584x | 1.611x |
| PyTorch | Square 1024 | FP32 | 64 | 0.429 | 0.490 | 0.400 | 1.142x | 0.932x |
| PyTorch | MLP | FP32 | 64 | 3.901 | 6.584 | 2.494 | 1.688x | 0.639x |
| PyTorch | Gram | FP64 | 64 | 0.575 | 2.334 | 0.892 | 4.058x | 1.551x |
| PyTorch | Transpose | FP64 | 64 | 1.012 | 4.038 | 1.518 | 3.991x | 1.500x |
| PyTorch | Attention | FP64 | 64 | 1.431 | 15.479 | 2.328 | 10.816x | 1.627x |
| PyTorch | Square 1024 | FP64 | 64 | 0.817 | 0.974 | 0.793 | 1.192x | 0.970x |
| PyTorch | MLP | FP64 | 64 | 7.440 | 22.304 | 12.144 | 2.998x | 1.632x |
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
