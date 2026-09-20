# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 kernels, packed operand panels and shape-dependent task grids. Bounded paths use twelve-row FP32 or six-row FP64 layouts; transposed FP32 inputs can be packed directly into the kernel layout without an intermediate panel. Bounded square and transposed products also use one-level Strassen: seven half-size products, with operand sums formed during packing. Every call repacks its inputs, and the conservative finite-range preflight rides on the pack pass itself (per-worker abs-maxima reduced after packing) so no separate operand scan runs before the packed route; unsafe ranges still fall back to the classical path. At 64 cores a persistent spinning worker-pool executor replaces per-call OpenMP fork/join (workers spin briefly after each dispatch and then sleep on futex; below 64 threads the OpenMP executor is retained). Runtime capability, bounds, workspace and finite-range checks retain classical fallbacks. Strassen changes rounding behaviour; the range check limits overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot: **60/60 wins against OpenBLAS; 50/60 against NVPL**. The table contains all 60 cases from one complete run of this source configuration on an idle whole-node allocation (Grace, exclusive 288-core node): seven NumPy and eight PyTorch workloads, each at two precisions and 16/64 cores. Small median differences do not establish statistical significance, so entries within roughly two percent of 1x can move between sessions.

FP32 and FP64 denote single and double precision, respectively. Latency is reported in milliseconds, with lower values indicating better performance. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. Each entry is the median of three fresh-process medians after warm-up, with all libraries using matching inputs, affinity and requested thread counts.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 16 | 1.120 | 1.782 | 1.172 | 1.591x | 1.047x |
| NumPy | Transposed GEMM | FP32 | 16 | 1.344 | 1.659 | 1.451 | 1.235x | 1.080x |
| NumPy | Attention | FP32 | 16 | 4.155 | 4.354 | 4.171 | 1.048x | 1.004x |
| NumPy | Square 1024 | FP32 | 16 | 1.395 | 1.666 | 1.457 | 1.194x | 1.045x |
| NumPy | Square 4096 | FP32 | 16 | 85.124 | 105.415 | 94.005 | 1.238x | 1.104x |
| NumPy | Square 8192 | FP32 | 16 | 657.702 | 824.885 | 719.099 | 1.254x | 1.093x |
| NumPy | MLP forward | FP32 | 16 | 10.630 | 11.058 | 10.137 | 1.040x | 0.954x |
| NumPy | Gram | FP64 | 16 | 2.182 | 2.934 | 2.364 | 1.345x | 1.084x |
| NumPy | Transposed GEMM | FP64 | 16 | 2.822 | 3.420 | 2.931 | 1.212x | 1.039x |
| NumPy | Attention | FP64 | 16 | 6.934 | 7.387 | 6.728 | 1.065x | 0.970x |
| NumPy | Square 1024 | FP64 | 16 | 2.850 | 3.425 | 2.928 | 1.202x | 1.027x |
| NumPy | Square 4096 | FP64 | 16 | 178.357 | 220.985 | 197.411 | 1.239x | 1.107x |
| NumPy | Square 8192 | FP64 | 16 | 1403.409 | 1737.491 | 1478.621 | 1.238x | 1.054x |
| NumPy | MLP forward | FP64 | 16 | 22.331 | 26.080 | 21.817 | 1.168x | 0.977x |
| NumPy | Gram | FP32 | 64 | 0.423 | 13.517 | 0.588 | 31.941x | 1.390x |
| NumPy | Transposed GEMM | FP32 | 64 | 0.398 | 0.511 | 0.384 | 1.284x | 0.964x |
| NumPy | Attention | FP32 | 64 | 3.598 | 4.093 | 4.309 | 1.138x | 1.198x |
| NumPy | Square 1024 | FP32 | 64 | 0.415 | 0.508 | 0.390 | 1.223x | 0.941x |
| NumPy | Square 4096 | FP32 | 64 | 26.065 | 35.009 | 37.907 | 1.343x | 1.454x |
| NumPy | Square 8192 | FP32 | 64 | 184.796 | 251.287 | 234.081 | 1.360x | 1.267x |
| NumPy | MLP forward | FP32 | 64 | 4.046 | 4.086 | 3.817 | 1.010x | 0.943x |
| NumPy | Gram | FP64 | 64 | 0.836 | 28.018 | 1.042 | 33.508x | 1.246x |
| NumPy | Transposed GEMM | FP64 | 64 | 0.758 | 1.050 | 0.787 | 1.385x | 1.038x |
| NumPy | Attention | FP64 | 64 | 6.034 | 6.440 | 6.285 | 1.067x | 1.042x |
| NumPy | Square 1024 | FP64 | 64 | 0.784 | 0.973 | 0.788 | 1.241x | 1.005x |
| NumPy | Square 4096 | FP64 | 64 | 55.524 | 72.127 | 79.419 | 1.299x | 1.430x |
| NumPy | Square 8192 | FP64 | 64 | 392.901 | 538.881 | 483.167 | 1.372x | 1.230x |
| NumPy | MLP forward | FP64 | 64 | 8.742 | 21.338 | 10.517 | 2.441x | 1.203x |
| PyTorch | Gram | FP32 | 16 | 0.999 | 1.705 | 1.538 | 1.708x | 1.540x |
| PyTorch | Transposed GEMM | FP32 | 16 | 1.482 | 2.446 | 1.620 | 1.650x | 1.093x |
| PyTorch | Attention | FP32 | 16 | 1.205 | 9.528 | 1.287 | 7.904x | 1.068x |
| PyTorch | Square 1024 | FP32 | 16 | 1.406 | 1.659 | 1.466 | 1.179x | 1.042x |
| PyTorch | Square 4096 | FP32 | 16 | 84.970 | 105.200 | 94.155 | 1.238x | 1.108x |
| PyTorch | Square 8192 | FP32 | 16 | 656.905 | 810.444 | 712.405 | 1.234x | 1.084x |
| PyTorch | MLP forward | FP32 | 16 | 9.130 | 11.202 | 8.939 | 1.227x | 0.979x |
| PyTorch | Forward + backward | FP32 | 16 | 23.366 | 41.408 | 24.107 | 1.772x | 1.032x |
| PyTorch | Gram | FP64 | 16 | 1.965 | 3.642 | 3.162 | 1.854x | 1.609x |
| PyTorch | Transposed GEMM | FP64 | 16 | 3.020 | 4.970 | 3.579 | 1.646x | 1.185x |
| PyTorch | Attention | FP64 | 16 | 2.488 | 14.702 | 2.509 | 5.908x | 1.008x |
| PyTorch | Square 1024 | FP64 | 16 | 2.876 | 3.426 | 2.940 | 1.191x | 1.022x |
| PyTorch | Square 4096 | FP64 | 16 | 179.015 | 218.657 | 195.211 | 1.221x | 1.090x |
| PyTorch | Square 8192 | FP64 | 16 | 1402.786 | 1721.596 | 1464.075 | 1.227x | 1.044x |
| PyTorch | MLP forward | FP64 | 16 | 20.440 | 27.344 | 20.540 | 1.338x | 1.005x |
| PyTorch | Forward + backward | FP64 | 16 | 48.364 | 79.676 | 52.129 | 1.647x | 1.078x |
| PyTorch | Gram | FP32 | 64 | 0.300 | 0.918 | 0.493 | 3.057x | 1.641x |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.501 | 1.929 | 0.536 | 3.848x | 1.070x |
| PyTorch | Attention | FP32 | 64 | 0.910 | 2.572 | 1.463 | 2.826x | 1.607x |
| PyTorch | Square 1024 | FP32 | 64 | 0.429 | 0.499 | 0.402 | 1.163x | 0.938x |
| PyTorch | Square 4096 | FP32 | 64 | 26.223 | 34.622 | 37.103 | 1.320x | 1.415x |
| PyTorch | Square 8192 | FP32 | 64 | 184.856 | 241.107 | 245.800 | 1.304x | 1.330x |
| PyTorch | MLP forward | FP32 | 64 | 3.742 | 6.665 | 2.503 | 1.781x | 0.669x |
| PyTorch | Forward + backward | FP32 | 64 | 9.291 | 39.890 | 14.184 | 4.293x | 1.527x |
| PyTorch | Gram | FP64 | 64 | 0.584 | 2.339 | 0.897 | 4.009x | 1.537x |
| PyTorch | Transposed GEMM | FP64 | 64 | 1.007 | 4.189 | 1.563 | 4.160x | 1.552x |
| PyTorch | Attention | FP64 | 64 | 1.395 | 4.282 | 2.456 | 3.070x | 1.761x |
| PyTorch | Square 1024 | FP64 | 64 | 0.809 | 0.971 | 0.793 | 1.201x | 0.981x |
| PyTorch | Square 4096 | FP64 | 64 | 55.281 | 74.035 | 67.750 | 1.339x | 1.226x |
| PyTorch | Square 8192 | FP64 | 64 | 392.655 | 528.356 | 486.421 | 1.346x | 1.239x |
| PyTorch | MLP forward | FP64 | 64 | 7.476 | 26.981 | 12.240 | 3.609x | 1.637x |
| PyTorch | Forward + backward | FP64 | 64 | 17.473 | 74.905 | 37.465 | 4.287x | 2.144x |

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
