# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 and NEON kernels, packed operand panels and shape-dependent task grids. Bounded square, transposed and rectangular products use one-level Strassen, with all seven operand transforms formed during packing. Deep products stream through small depth panels and accumulate private products before writing C. FP64 packing forms the seven transforms directly from the four original quadrants, including transposed A. Short transposed-A FP32 products use a parallel four-by-four transpose before the guarded packed route. Every call repacks its inputs. A finite-range check during packing preserves a classical fallback, including when an unsafe value appears in a later depth panel.

Selected FP32 products at 64 threads use an eight-row, twelve-column NEON kernel and shared packed operands. Short products use the private worker pool. Eligible deep products stream through bounded panels with row and column group barriers, using either the private pool or the application's existing OpenMP executor. Deep FP32 Strassen and the streaming grids allocate only their live scratch and release it after the synchronous call; other paths retain allocation capacity. Bounded 64-worker FP64 squares use a wider depth panel to reduce partial-product traffic. Workers in the private pool poll for up to 256 microseconds before sleeping on a futex; a timeout without new work leaves the previous task untouched. Runtime capability, matrix bounds and workspace checks retain classical fallbacks. Strassen changes rounding behaviour; its range check bounds overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot (2 October 2026): **60/60 wins against OpenBLAS; 57/60 against NVPL**. **59/60 cases meet the NVPL 2% target**, defined as CAMBLAS latency at most 1.02 times NVPL latency, including wins. All 60 cases were measured on one idle, exclusive Grace node (`nid010267`, allocation `7002579`), with matching inputs and 16/64 CPUs on one Grace chip. Small differences near 1x can move between sessions and are not tests of statistical significance.

The versions are GCC 14.3.0, OpenBLAS 0.3.33 and NVPL BLAS 0.3.0 from HPC SDK 24.11. Each entry is the median of three fresh-process medians, with backend order rotated between rounds. After warm-up, each round uses 1,001 NumPy MLP calls, 201 PyTorch MLP/backward calls and five calls for other workloads. Vendor thread counts, loaded-library hashes, native-core identity, affinity, routing counters and output samples/norms are checked. Separate full-output scalar, packing, exceptional-input, concurrent-caller and framework gradient tests passed.

Latency is in milliseconds; lower is better. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. [All rounds, source hashes, build commands and correctness checks](bench/reports/grace_20261002.json) are recorded alongside this table. The public [NVPL/BLIS Grace presentation](https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Evarist_BLIS_Retreat_2024.pdf) informed the packing and register-blocking experiments; the retained kernels and algorithms are independently implemented.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 16 | 1.114 | 1.751 | 1.158 | 1.572x | 1.039x |
| NumPy | Transposed GEMM | FP32 | 16 | 1.333 | 1.620 | 1.433 | 1.215x | 1.075x |
| NumPy | Attention | FP32 | 16 | 3.737 | 4.271 | 4.168 | 1.143x | 1.115x |
| NumPy | Square 1024 | FP32 | 16 | 1.371 | 1.632 | 1.443 | 1.191x | 1.053x |
| NumPy | Square 4096 | FP32 | 16 | 83.947 | 103.455 | 92.865 | 1.232x | 1.106x |
| NumPy | Square 8192 | FP32 | 16 | 649.927 | 807.094 | 716.911 | 1.242x | 1.103x |
| NumPy | MLP forward | FP32 | 16 | 9.850 | 10.673 | 9.921 | 1.084x | 1.007x |
| NumPy | Gram | FP64 | 16 | 2.146 | 2.887 | 2.319 | 1.345x | 1.081x |
| NumPy | Transposed GEMM | FP64 | 16 | 2.744 | 3.361 | 2.879 | 1.225x | 1.049x |
| NumPy | Attention | FP64 | 16 | 6.296 | 7.302 | 6.799 | 1.160x | 1.080x |
| NumPy | Square 1024 | FP64 | 16 | 2.834 | 3.355 | 2.903 | 1.184x | 1.024x |
| NumPy | Square 4096 | FP64 | 16 | 176.446 | 217.513 | 190.606 | 1.233x | 1.080x |
| NumPy | Square 8192 | FP64 | 16 | 1387.255 | 1717.026 | 1469.451 | 1.238x | 1.059x |
| NumPy | MLP forward | FP64 | 16 | 20.846 | 26.508 | 21.389 | 1.272x | 1.026x |
| NumPy | Gram | FP32 | 64 | 0.409 | 13.061 | 0.582 | 31.948x | 1.424x |
| NumPy | Transposed GEMM | FP32 | 64 | 0.367 | 0.502 | 0.375 | 1.367x | 1.021x |
| NumPy | Attention | FP32 | 64 | 3.577 | 3.942 | 4.203 | 1.102x | 1.175x |
| NumPy | Square 1024 | FP32 | 64 | 0.384 | 0.498 | 0.385 | 1.297x | 1.003x |
| NumPy | Square 4096 | FP32 | 64 | 25.716 | 34.340 | 36.659 | 1.335x | 1.426x |
| NumPy | Square 8192 | FP32 | 64 | 181.455 | 238.443 | 216.949 | 1.314x | 1.196x |
| NumPy | MLP forward | FP32 | 64 | 3.575 | 3.914 | 3.690 | 1.095x | 1.032x |
| NumPy | Gram | FP64 | 64 | 0.804 | 27.162 | 1.033 | 33.793x | 1.285x |
| NumPy | Transposed GEMM | FP64 | 64 | 0.758 | 0.949 | 0.772 | 1.252x | 1.018x |
| NumPy | Attention | FP64 | 64 | 6.175 | 6.266 | 6.134 | 1.015x | 0.993x |
| NumPy | Square 1024 | FP64 | 64 | 0.767 | 0.959 | 0.772 | 1.249x | 1.006x |
| NumPy | Square 4096 | FP64 | 64 | 54.147 | 72.721 | 79.490 | 1.343x | 1.468x |
| NumPy | Square 8192 | FP64 | 64 | 386.075 | 530.570 | 479.368 | 1.374x | 1.242x |
| NumPy | MLP forward | FP64 | 64 | 8.993 | 23.484 | 11.765 | 2.611x | 1.308x |
| PyTorch | Gram | FP32 | 16 | 1.005 | 1.675 | 1.507 | 1.667x | 1.500x |
| PyTorch | Transposed GEMM | FP32 | 16 | 1.469 | 2.426 | 1.607 | 1.652x | 1.094x |
| PyTorch | Attention | FP32 | 16 | 1.027 | 9.321 | 1.211 | 9.079x | 1.180x |
| PyTorch | Square 1024 | FP32 | 16 | 1.389 | 1.626 | 1.437 | 1.170x | 1.034x |
| PyTorch | Square 4096 | FP32 | 16 | 83.979 | 103.166 | 92.166 | 1.228x | 1.097x |
| PyTorch | Square 8192 | FP32 | 16 | 644.841 | 799.487 | 703.787 | 1.240x | 1.091x |
| PyTorch | MLP forward | FP32 | 16 | 8.673 | 14.222 | 8.730 | 1.640x | 1.007x |
| PyTorch | Forward + backward | FP32 | 16 | 22.124 | 47.335 | 24.979 | 2.140x | 1.129x |
| PyTorch | Gram | FP64 | 16 | 1.930 | 3.569 | 3.111 | 1.850x | 1.612x |
| PyTorch | Transposed GEMM | FP64 | 16 | 2.994 | 5.111 | 4.030 | 1.707x | 1.346x |
| PyTorch | Attention | FP64 | 16 | 2.048 | 7.354 | 2.409 | 3.592x | 1.177x |
| PyTorch | Square 1024 | FP64 | 16 | 2.881 | 3.367 | 2.891 | 1.169x | 1.004x |
| PyTorch | Square 4096 | FP64 | 16 | 176.459 | 216.131 | 192.096 | 1.225x | 1.089x |
| PyTorch | Square 8192 | FP64 | 16 | 1383.313 | 1703.660 | 1443.017 | 1.232x | 1.043x |
| PyTorch | MLP forward | FP64 | 16 | 18.599 | 30.832 | 20.378 | 1.658x | 1.096x |
| PyTorch | Forward + backward | FP64 | 16 | 47.276 | 94.428 | 50.696 | 1.997x | 1.072x |
| PyTorch | Gram | FP32 | 64 | 0.307 | 0.910 | 0.487 | 2.961x | 1.583x |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.504 | 1.880 | 0.535 | 3.731x | 1.061x |
| PyTorch | Attention | FP32 | 64 | 1.195 | 2.346 | 1.467 | 1.963x | 1.227x |
| PyTorch | Square 1024 | FP32 | 64 | 0.398 | 0.503 | 0.395 | 1.263x | 0.991x |
| PyTorch | Square 4096 | FP32 | 64 | 26.145 | 34.626 | 36.837 | 1.324x | 1.409x |
| PyTorch | Square 8192 | FP32 | 64 | 182.343 | 244.445 | 231.492 | 1.341x | 1.270x |
| PyTorch | MLP forward | FP32 | 64 | 2.394 | 8.777 | 2.456 | 3.666x | 1.026x |
| PyTorch | Forward + backward | FP32 | 64 | 8.661 | 55.590 | 17.855 | 6.418x | 2.062x |
| PyTorch | Gram | FP64 | 64 | 0.573 | 2.337 | 0.875 | 4.082x | 1.528x |
| PyTorch | Transposed GEMM | FP64 | 64 | 1.008 | 3.981 | 1.979 | 3.950x | 1.964x |
| PyTorch | Attention | FP64 | 64 | 1.405 | 7.746 | 2.278 | 5.514x | 1.622x |
| PyTorch | Square 1024 | FP64 | 64 | 0.794 | 0.965 | 0.776 | 1.216x | 0.978x |
| PyTorch | Square 4096 | FP64 | 64 | 55.338 | 73.178 | 79.296 | 1.322x | 1.433x |
| PyTorch | Square 8192 | FP64 | 64 | 388.191 | 527.762 | 463.836 | 1.360x | 1.195x |
| PyTorch | MLP forward | FP64 | 64 | 7.009 | 26.443 | 10.122 | 3.773x | 1.444x |
| PyTorch | Forward + backward | FP64 | 64 | 17.533 | 83.013 | 32.301 | 4.735x | 1.842x |

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
