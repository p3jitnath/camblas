# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 kernels, packed operand panels and shape-dependent task grids. Bounded paths use twelve-row FP32 or six-row FP64 layouts; transposed FP32 inputs can be packed directly into the kernel layout without an intermediate panel. Bounded square and transposed products also use one-level Strassen: seven half-size products, with operand sums formed during packing. Every call repacks its inputs. Runtime capability, bounds, workspace and finite-range checks retain classical fallbacks. Strassen changes rounding behaviour; the range check limits overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot: **59/60 wins against OpenBLAS; 46/60 against NVPL**. The table contains all 60 cases: seven NumPy and eight PyTorch workloads, each at two precisions and 16/64 cores. **† marks 12 cases measured with this source configuration on an idle whole-node allocation.** The 48 unmarked cases use measurements from separate allocations and CAMBLAS builds whose corresponding execution paths are unchanged. They have not been remeasured with this build, so the table is a combined snapshot, not one complete run. Small median differences do not establish statistical significance; the unmarked NumPy FP32 attention win reverses in one of three paired NVPL comparisons.

FP32 and FP64 denote single and double precision, respectively. Latency is reported in milliseconds, with lower values indicating better performance. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1× favour CAMBLAS**. Each entry is the median of three fresh-process medians after three warm-ups. Marked cases use 1001 timed calls per NumPy round and 201 per PyTorch round. Unmarked cases use at least five calls per round; some use the longer 1001/201-call protocol.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Square 1024† | FP32 | 16 | 1.384 | 1.632 | 1.441 | 1.179× | 1.041× |
| NumPy | Square 1024 | FP32 | 64 | 0.441 | 0.516 | 0.396 | 1.170× | 0.899× |
| NumPy | Square 1024† | FP64 | 16 | 2.812 | 3.379 | 2.898 | 1.202× | 1.031× |
| NumPy | Square 1024† | FP64 | 64 | 0.858 | 1.043 | 0.819 | 1.216× | 0.955× |
| NumPy | Square 4096 | FP32 | 16 | 84.093 | 104.127 | 93.619 | 1.238× | 1.113× |
| NumPy | Square 4096 | FP32 | 64 | 25.865 | 34.125 | 36.185 | 1.319× | 1.399× |
| NumPy | Square 4096 | FP64 | 16 | 177.555 | 219.378 | 193.596 | 1.236× | 1.090× |
| NumPy | Square 4096 | FP64 | 64 | 53.745 | 72.402 | 76.985 | 1.347× | 1.432× |
| NumPy | Square 8192 | FP32 | 16 | 653.509 | 822.125 | 720.597 | 1.258× | 1.103× |
| NumPy | Square 8192 | FP32 | 64 | 182.385 | 248.564 | 240.594 | 1.363× | 1.319× |
| NumPy | Square 8192 | FP64 | 16 | 1387.948 | 1736.488 | 1493.413 | 1.251× | 1.076× |
| NumPy | Square 8192 | FP64 | 64 | 382.314 | 542.601 | 486.776 | 1.419× | 1.273× |
| NumPy | Transposed GEMM† | FP32 | 16 | 1.330 | 1.633 | 1.438 | 1.227× | 1.081× |
| NumPy | Transposed GEMM | FP32 | 64 | 0.424 | 0.521 | 0.386 | 1.230× | 0.909× |
| NumPy | Transposed GEMM† | FP64 | 16 | 2.784 | 3.374 | 2.887 | 1.212× | 1.037× |
| NumPy | Transposed GEMM† | FP64 | 64 | 0.829 | 1.010 | 0.822 | 1.218× | 0.991× |
| NumPy | Gram | FP32 | 16 | 1.102 | 1.746 | 1.160 | 1.584× | 1.053× |
| NumPy | Gram | FP32 | 64 | 0.458 | 13.258 | 0.570 | 28.927× | 1.244× |
| NumPy | Gram | FP64 | 16 | 2.152 | 2.896 | 2.336 | 1.345× | 1.085× |
| NumPy | Gram | FP64 | 64 | 0.827 | 27.510 | 1.024 | 33.261× | 1.239× |
| NumPy | MLP forward | FP32 | 16 | 10.692 | 10.730 | 9.965 | 1.004× | 0.932× |
| NumPy | MLP forward | FP32 | 64 | 4.088 | 4.003 | 3.728 | 0.979× | 0.912× |
| NumPy | MLP forward | FP64 | 16 | 22.569 | 26.578 | 21.522 | 1.178× | 0.954× |
| NumPy | MLP forward | FP64 | 64 | 8.962 | 20.462 | 11.951 | 2.283× | 1.334× |
| NumPy | Attention | FP32 | 16 | 4.081 | 4.249 | 4.154 | 1.041× | 1.018× |
| NumPy | Attention | FP32 | 64 | 3.569 | 4.032 | 4.215 | 1.130× | 1.181× |
| NumPy | Attention | FP64 | 16 | 6.956 | 7.394 | 6.768 | 1.063× | 0.973× |
| NumPy | Attention | FP64 | 64 | 6.083 | 6.465 | 6.236 | 1.063× | 1.025× |
| PyTorch | Square 1024† | FP32 | 16 | 1.403 | 1.636 | 1.452 | 1.166× | 1.034× |
| PyTorch | Square 1024 | FP32 | 64 | 0.447 | 0.503 | 0.403 | 1.124× | 0.901× |
| PyTorch | Square 1024† | FP64 | 16 | 2.825 | 3.388 | 2.906 | 1.199× | 1.029× |
| PyTorch | Square 1024† | FP64 | 64 | 0.872 | 1.015 | 0.828 | 1.164× | 0.949× |
| PyTorch | Square 4096 | FP32 | 16 | 84.131 | 103.732 | 92.992 | 1.233× | 1.105× |
| PyTorch | Square 4096 | FP32 | 64 | 25.846 | 34.195 | 36.309 | 1.323× | 1.405× |
| PyTorch | Square 4096 | FP64 | 16 | 177.810 | 217.491 | 192.401 | 1.223× | 1.082× |
| PyTorch | Square 4096 | FP64 | 64 | 54.201 | 71.729 | 77.170 | 1.323× | 1.424× |
| PyTorch | Square 8192 | FP32 | 16 | 653.124 | 814.144 | 712.992 | 1.247× | 1.092× |
| PyTorch | Square 8192 | FP32 | 64 | 182.080 | 244.074 | 210.217 | 1.340× | 1.155× |
| PyTorch | Square 8192 | FP64 | 16 | 1386.473 | 1709.889 | 1467.706 | 1.233× | 1.059× |
| PyTorch | Square 8192 | FP64 | 64 | 381.077 | 524.470 | 455.757 | 1.376× | 1.196× |
| PyTorch | Transposed GEMM† | FP32 | 16 | 1.459 | 2.462 | 1.611 | 1.687× | 1.104× |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.526 | 1.891 | 0.482 | 3.598× | 0.918× |
| PyTorch | Transposed GEMM† | FP64 | 16 | 3.034 | 5.146 | 4.039 | 1.696× | 1.331× |
| PyTorch | Transposed GEMM† | FP64 | 64 | 1.025 | 3.978 | 1.733 | 3.881× | 1.691× |
| PyTorch | Gram | FP32 | 16 | 0.984 | 1.682 | 1.512 | 1.710× | 1.537× |
| PyTorch | Gram | FP32 | 64 | 0.315 | 0.910 | 0.487 | 2.887× | 1.546× |
| PyTorch | Gram | FP64 | 16 | 1.916 | 3.596 | 3.115 | 1.877× | 1.626× |
| PyTorch | Gram | FP64 | 64 | 0.580 | 2.313 | 0.880 | 3.987× | 1.517× |
| PyTorch | MLP forward | FP32 | 16 | 9.128 | 15.642 | 8.727 | 1.714× | 0.956× |
| PyTorch | MLP forward | FP32 | 64 | 2.595 | 9.823 | 2.519 | 3.785× | 0.971× |
| PyTorch | MLP forward | FP64 | 16 | 20.420 | 32.244 | 21.466 | 1.579× | 1.051× |
| PyTorch | MLP forward | FP64 | 64 | 6.911 | 24.223 | 12.536 | 3.505× | 1.814× |
| PyTorch | Attention | FP32 | 16 | 1.202 | 9.479 | 1.290 | 7.887× | 1.073× |
| PyTorch | Attention | FP32 | 64 | 0.664 | 2.380 | 1.465 | 3.587× | 2.208× |
| PyTorch | Attention | FP64 | 16 | 2.487 | 14.507 | 2.356 | 5.834× | 0.948× |
| PyTorch | Attention | FP64 | 64 | 1.094 | 8.409 | 2.271 | 7.687× | 2.076× |
| PyTorch | Forward + backward | FP32 | 16 | 23.236 | 52.699 | 25.239 | 2.268× | 1.086× |
| PyTorch | Forward + backward | FP32 | 64 | 8.115 | 56.632 | 17.859 | 6.979× | 2.201× |
| PyTorch | Forward + backward | FP64 | 16 | 47.952 | 106.468 | 51.584 | 2.220× | 1.076× |
| PyTorch | Forward + backward | FP64 | 64 | 16.822 | 60.935 | 32.264 | 3.622× | 1.918× |

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
