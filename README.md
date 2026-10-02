# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 and NEON kernels, packed operand panels and shape-dependent task grids. Bounded square, transposed and rectangular products use one-level Strassen, with all seven operand transforms formed during packing. Deep products stream through small depth panels and accumulate private products before writing C. FP64 packing forms the seven transforms directly from the four original quadrants, including transposed A. Short transposed-A FP32 products use a parallel four-by-four transpose before the guarded packed route. Every call repacks its inputs. A finite-range check during packing preserves a classical fallback, including when an unsafe value appears in a later depth panel.

Selected FP32 products at 64 threads use an eight-row, twelve-column NEON kernel and shared packed operands. Short products use the private worker pool. Eligible deep products stream through bounded panels with row and column group barriers, using either the private pool or the application's existing OpenMP executor. Deep FP32 Strassen and the streaming grids allocate only their live scratch and release it after the synchronous call; other paths retain allocation capacity. Bounded 64-worker FP64 squares use a wider depth panel to reduce partial-product traffic, with a six-row, eight-column NEON kernel for the smaller squares. Workers in the private pool poll for up to 256 microseconds before sleeping on a futex; a timeout without new work leaves the previous task untouched. Runtime capability, matrix bounds and workspace checks retain classical fallbacks. Strassen changes rounding behaviour; its range check bounds overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot (2 October 2026): **60/60 wins against OpenBLAS; 59/60 against NVPL**. **60/60 cases meet the NVPL 2% target**, defined as CAMBLAS latency at most 1.02 times NVPL latency, including wins. All 60 cases were measured on one idle, exclusive Grace node (`nid010267`, allocation `7002579`), with matching inputs and 16/64 CPUs on one Grace chip. Small differences near 1x can move between sessions and are not tests of statistical significance.

The versions are GCC 14.3.0, OpenBLAS 0.3.33 and NVPL BLAS 0.3.0 from HPC SDK 24.11. Each entry is the median of three fresh-process medians, with backend order rotated between rounds. After warm-up, each round uses 1,001 NumPy MLP calls, 201 PyTorch MLP/backward calls and five calls for other workloads. Vendor thread counts, loaded-library hashes, native-core identity, affinity, routing counters and output samples/norms are checked. Separate full-output scalar, packing, exceptional-input, concurrent-caller and framework gradient tests passed.

Latency is in milliseconds; lower is better. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. [All rounds, source hashes, build commands and correctness checks](bench/reports/grace_20261002.json) are recorded alongside this table. The public [NVPL/BLIS Grace presentation](https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Evarist_BLIS_Retreat_2024.pdf) informed the packing and register-blocking experiments; the retained kernels and algorithms are independently implemented.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 16 | 1.097 | 1.740 | 1.158 | 1.587x | 1.056x |
| NumPy | Transposed GEMM | FP32 | 16 | 1.318 | 1.626 | 1.425 | 1.233x | 1.081x |
| NumPy | Attention | FP32 | 16 | 3.763 | 4.335 | 4.106 | 1.152x | 1.091x |
| NumPy | Square 1024 | FP32 | 16 | 1.379 | 1.628 | 1.436 | 1.181x | 1.042x |
| NumPy | Square 4096 | FP32 | 16 | 83.785 | 103.604 | 92.979 | 1.237x | 1.110x |
| NumPy | Square 8192 | FP32 | 16 | 648.910 | 812.147 | 706.748 | 1.252x | 1.089x |
| NumPy | MLP forward | FP32 | 16 | 9.827 | 10.674 | 9.926 | 1.086x | 1.010x |
| NumPy | Gram | FP64 | 16 | 2.158 | 2.903 | 2.318 | 1.345x | 1.074x |
| NumPy | Transposed GEMM | FP64 | 16 | 2.779 | 3.372 | 2.878 | 1.213x | 1.036x |
| NumPy | Attention | FP64 | 16 | 6.379 | 7.343 | 6.730 | 1.151x | 1.055x |
| NumPy | Square 1024 | FP64 | 16 | 2.846 | 3.362 | 2.891 | 1.181x | 1.016x |
| NumPy | Square 4096 | FP64 | 16 | 176.587 | 218.387 | 192.915 | 1.237x | 1.092x |
| NumPy | Square 8192 | FP64 | 16 | 1387.156 | 1716.486 | 1475.935 | 1.237x | 1.064x |
| NumPy | MLP forward | FP64 | 16 | 20.846 | 26.302 | 20.931 | 1.262x | 1.004x |
| NumPy | Gram | FP32 | 64 | 0.413 | 13.257 | 0.574 | 32.125x | 1.392x |
| NumPy | Transposed GEMM | FP32 | 64 | 0.368 | 0.502 | 0.376 | 1.365x | 1.021x |
| NumPy | Attention | FP32 | 64 | 3.553 | 4.054 | 4.325 | 1.141x | 1.217x |
| NumPy | Square 1024 | FP32 | 64 | 0.383 | 0.499 | 0.386 | 1.303x | 1.008x |
| NumPy | Square 4096 | FP32 | 64 | 25.642 | 34.335 | 35.856 | 1.339x | 1.398x |
| NumPy | Square 8192 | FP32 | 64 | 181.089 | 236.921 | 240.615 | 1.308x | 1.329x |
| NumPy | MLP forward | FP32 | 64 | 3.569 | 3.874 | 3.707 | 1.086x | 1.039x |
| NumPy | Gram | FP64 | 64 | 0.827 | 27.341 | 1.025 | 33.040x | 1.238x |
| NumPy | Transposed GEMM | FP64 | 64 | 0.759 | 0.961 | 0.807 | 1.265x | 1.063x |
| NumPy | Attention | FP64 | 64 | 6.181 | 6.326 | 6.212 | 1.024x | 1.005x |
| NumPy | Square 1024 | FP64 | 64 | 0.733 | 1.085 | 0.771 | 1.481x | 1.053x |
| NumPy | Square 4096 | FP64 | 64 | 54.512 | 72.889 | 78.513 | 1.337x | 1.440x |
| NumPy | Square 8192 | FP64 | 64 | 387.521 | 533.237 | 497.985 | 1.376x | 1.285x |
| NumPy | MLP forward | FP64 | 64 | 9.074 | 23.877 | 11.826 | 2.631x | 1.303x |
| PyTorch | Gram | FP32 | 16 | 0.997 | 1.680 | 1.505 | 1.686x | 1.509x |
| PyTorch | Transposed GEMM | FP32 | 16 | 1.484 | 2.519 | 1.585 | 1.697x | 1.068x |
| PyTorch | Attention | FP32 | 16 | 1.019 | 9.356 | 1.234 | 9.180x | 1.211x |
| PyTorch | Square 1024 | FP32 | 16 | 1.383 | 1.631 | 1.438 | 1.179x | 1.040x |
| PyTorch | Square 4096 | FP32 | 16 | 83.692 | 103.271 | 92.542 | 1.234x | 1.106x |
| PyTorch | Square 8192 | FP32 | 16 | 643.210 | 800.561 | 709.222 | 1.245x | 1.103x |
| PyTorch | MLP forward | FP32 | 16 | 8.681 | 15.448 | 8.733 | 1.779x | 1.006x |
| PyTorch | Forward + backward | FP32 | 16 | 22.147 | 51.887 | 25.116 | 2.343x | 1.134x |
| PyTorch | Gram | FP64 | 16 | 1.961 | 3.592 | 3.105 | 1.831x | 1.583x |
| PyTorch | Transposed GEMM | FP64 | 16 | 3.021 | 5.111 | 4.100 | 1.692x | 1.357x |
| PyTorch | Attention | FP64 | 16 | 2.036 | 8.286 | 2.519 | 4.069x | 1.237x |
| PyTorch | Square 1024 | FP64 | 16 | 2.873 | 3.366 | 2.894 | 1.172x | 1.007x |
| PyTorch | Square 4096 | FP64 | 16 | 176.575 | 216.536 | 193.054 | 1.226x | 1.093x |
| PyTorch | Square 8192 | FP64 | 16 | 1385.714 | 1697.751 | 1441.750 | 1.225x | 1.040x |
| PyTorch | MLP forward | FP64 | 16 | 18.631 | 32.526 | 20.183 | 1.746x | 1.083x |
| PyTorch | Forward + backward | FP64 | 16 | 47.012 | 102.384 | 51.070 | 2.178x | 1.086x |
| PyTorch | Gram | FP32 | 64 | 0.310 | 0.906 | 0.488 | 2.927x | 1.575x |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.499 | 1.872 | 0.536 | 3.749x | 1.074x |
| PyTorch | Attention | FP32 | 64 | 0.852 | 2.430 | 1.412 | 2.852x | 1.657x |
| PyTorch | Square 1024 | FP32 | 64 | 0.397 | 0.503 | 0.393 | 1.266x | 0.990x |
| PyTorch | Square 4096 | FP32 | 64 | 25.889 | 34.393 | 37.311 | 1.328x | 1.441x |
| PyTorch | Square 8192 | FP32 | 64 | 182.694 | 245.060 | 242.264 | 1.341x | 1.326x |
| PyTorch | MLP forward | FP32 | 64 | 2.391 | 8.875 | 2.455 | 3.711x | 1.027x |
| PyTorch | Forward + backward | FP32 | 64 | 8.624 | 53.659 | 17.708 | 6.222x | 2.053x |
| PyTorch | Gram | FP64 | 64 | 0.572 | 2.310 | 0.874 | 4.041x | 1.530x |
| PyTorch | Transposed GEMM | FP64 | 64 | 1.014 | 4.176 | 1.872 | 4.120x | 1.846x |
| PyTorch | Attention | FP64 | 64 | 1.588 | 6.627 | 2.364 | 4.172x | 1.488x |
| PyTorch | Square 1024 | FP64 | 64 | 0.759 | 0.964 | 0.784 | 1.269x | 1.032x |
| PyTorch | Square 4096 | FP64 | 64 | 55.074 | 72.932 | 78.903 | 1.324x | 1.433x |
| PyTorch | Square 8192 | FP64 | 64 | 386.430 | 527.024 | 475.564 | 1.364x | 1.231x |
| PyTorch | MLP forward | FP64 | 64 | 7.364 | 30.134 | 10.092 | 4.092x | 1.371x |
| PyTorch | Forward + backward | FP64 | 64 | 17.815 | 80.405 | 31.889 | 4.513x | 1.790x |

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
.frameworks/envs/camblas/bin/python tests/check_shared_grid32.py \
  .frameworks/prefix/camblas/lib/libframework_blas.so --torch-context
.frameworks/envs/camblas/bin/python tests/check_pool_idle.py \
  .frameworks/prefix/camblas/lib/libframework_blas.so --threads 64
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
python3.11 bench/compare.py --full --output bench/results/full60
```

The driver rotates all three libraries over at least three fresh-process rounds and saves timings, library identities, affinity, call counters and numerical checks in the requested output directory; `bench/results/` is ignored by Git. It rejects mismatched bridges, native cores or CPU affinity, incomplete rounds and non-finite timings or output signatures. The full matrix has seven NumPy and eight PyTorch workloads at two precisions and two core counts. Application checks use samples and norms; packing changes also require the independent full-output checks in [tests/test_mlp_packing.py](tests/test_mlp_packing.py).

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and [LICENSE](LICENSE) for the MIT licence, copyright Pritthijit Nath. External dependencies retain their own licences; generated artefacts and internal documents are excluded from this repository.
