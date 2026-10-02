# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 and NEON kernels, packed operand panels and shape-dependent task grids. Bounded square, transposed and rectangular products use one-level Strassen, with all seven operand transforms formed during packing. Deep products stream through small depth panels and accumulate private products before writing C. Eligible deep FP32 rectangles at 16/32 workers use a twelve-row, eight-column NEON kernel, with product/tile assignments rotated to spread partial micro-panels across workers. FP64 packing forms the seven transforms directly from the four original quadrants. Eligible short transposed-A products use this fused packing at 16/32 threads and in NumPy at 64 threads. Short transposed-A FP32 products use a parallel four-by-four transpose before the guarded packed route. Every call repacks its inputs. A finite-range check during packing preserves a classical fallback, including when an unsafe value appears in a later depth panel.

Selected FP32 products at 64 threads use an eight-row, twelve-column NEON kernel and shared packed operands. Short products use the private worker pool. Aligned short NN squares use eight row groups and eight column groups; other short products retain sixteen row groups and four column groups. Eligible deep products stream through bounded panels with row and column group barriers, using either the private pool or the application's existing OpenMP executor. Deep FP32 Strassen and the streaming grids allocate only their live scratch and release it after the synchronous call; other paths retain allocation capacity. Bounded 64-worker FP64 squares use a wider depth panel to reduce partial-product traffic. Aligned NN squares from 512 through 1024 use a six-row, eight-column NEON kernel at 16 through 64 workers. Eligible transposed-B FP64 rectangles use the same NEON kernel at 64 workers, with the existing packed operand layout. Workers in the private pool poll for up to 256 microseconds before sleeping on a futex; a timeout without new work leaves the previous task untouched. Runtime capability, matrix bounds and workspace checks retain classical fallbacks. Strassen changes rounding behaviour; its range check bounds overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot (2 October 2026): **60/60 wins against OpenBLAS; 60/60 against NVPL**. **60/60 cases meet the NVPL 2% target**, defined as CAMBLAS latency at most 1.02 times NVPL latency, including wins. All 60 cases were measured on one idle, exclusive Grace node (`nid010099`, allocation `7005823`), with matching inputs and 16/64 CPUs on one Grace chip. Small differences near 1x can move between sessions and are not tests of statistical significance.

The versions are GCC 14.3.0, OpenBLAS 0.3.33 and NVPL BLAS 0.3.0 from HPC SDK 24.11. Each entry is the median of three fresh-process medians, with backend order rotated between rounds. After warm-up, each round uses 1,001 NumPy MLP calls, 201 PyTorch MLP/backward calls and five calls for other workloads. Vendor thread counts, loaded-library hashes, native-core identity, affinity, routing counters and output samples/norms are checked. Separate full-output scalar, packing, exceptional-input, concurrent-caller and framework gradient tests passed.

Latency is in milliseconds; lower is better. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. [All rounds, source hashes, build commands and correctness checks](bench/reports/grace_20261002.json) are recorded alongside this table. The public [NVPL/BLIS Grace presentation](https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Evarist_BLIS_Retreat_2024.pdf) informed the packing and register-blocking experiments; the retained kernels and algorithms are independently implemented.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 16 | 1.114 | 1.782 | 1.170 | 1.600x | 1.050x |
| NumPy | Transposed GEMM | FP32 | 16 | 1.330 | 1.649 | 1.447 | 1.240x | 1.088x |
| NumPy | Attention | FP32 | 16 | 3.795 | 4.349 | 4.158 | 1.146x | 1.096x |
| NumPy | Square 1024 | FP32 | 16 | 1.397 | 1.647 | 1.459 | 1.179x | 1.044x |
| NumPy | Square 4096 | FP32 | 16 | 84.551 | 104.641 | 94.190 | 1.238x | 1.114x |
| NumPy | Square 8192 | FP32 | 16 | 655.372 | 825.186 | 723.144 | 1.259x | 1.103x |
| NumPy | MLP forward | FP32 | 16 | 9.848 | 10.804 | 10.045 | 1.097x | 1.020x |
| NumPy | Gram | FP64 | 16 | 2.190 | 2.940 | 2.351 | 1.343x | 1.074x |
| NumPy | Transposed GEMM | FP64 | 16 | 2.806 | 3.415 | 2.943 | 1.217x | 1.049x |
| NumPy | Attention | FP64 | 16 | 6.483 | 7.390 | 6.836 | 1.140x | 1.054x |
| NumPy | Square 1024 | FP64 | 16 | 2.832 | 3.442 | 2.934 | 1.215x | 1.036x |
| NumPy | Square 4096 | FP64 | 16 | 178.224 | 219.887 | 195.888 | 1.234x | 1.099x |
| NumPy | Square 8192 | FP64 | 16 | 1397.545 | 1739.120 | 1509.593 | 1.244x | 1.080x |
| NumPy | MLP forward | FP64 | 16 | 20.949 | 26.646 | 21.139 | 1.272x | 1.009x |
| NumPy | Gram | FP32 | 64 | 0.417 | 13.310 | 0.581 | 31.924x | 1.392x |
| NumPy | Transposed GEMM | FP32 | 64 | 0.370 | 0.512 | 0.379 | 1.386x | 1.026x |
| NumPy | Attention | FP32 | 64 | 3.604 | 4.029 | 4.289 | 1.118x | 1.190x |
| NumPy | Square 1024 | FP32 | 64 | 0.382 | 0.507 | 0.390 | 1.327x | 1.022x |
| NumPy | Square 4096 | FP32 | 64 | 27.399 | 35.450 | 38.516 | 1.294x | 1.406x |
| NumPy | Square 8192 | FP32 | 64 | 195.561 | 264.138 | 254.628 | 1.351x | 1.302x |
| NumPy | MLP forward | FP32 | 64 | 3.629 | 3.992 | 3.725 | 1.100x | 1.026x |
| NumPy | Gram | FP64 | 64 | 0.825 | 27.502 | 1.032 | 33.320x | 1.251x |
| NumPy | Transposed GEMM | FP64 | 64 | 0.743 | 0.983 | 0.789 | 1.322x | 1.061x |
| NumPy | Attention | FP64 | 64 | 6.171 | 6.422 | 6.300 | 1.041x | 1.021x |
| NumPy | Square 1024 | FP64 | 64 | 0.753 | 0.993 | 0.786 | 1.318x | 1.044x |
| NumPy | Square 4096 | FP64 | 64 | 59.145 | 78.672 | 83.913 | 1.330x | 1.419x |
| NumPy | Square 8192 | FP64 | 64 | 412.167 | 571.006 | 513.263 | 1.385x | 1.245x |
| NumPy | MLP forward | FP64 | 64 | 8.933 | 23.594 | 12.102 | 2.641x | 1.355x |
| PyTorch | Gram | FP32 | 16 | 0.994 | 1.700 | 1.528 | 1.710x | 1.537x |
| PyTorch | Transposed GEMM | FP32 | 16 | 1.463 | 2.472 | 1.612 | 1.690x | 1.102x |
| PyTorch | Attention | FP32 | 16 | 1.031 | 9.777 | 1.230 | 9.487x | 1.193x |
| PyTorch | Square 1024 | FP32 | 16 | 1.403 | 1.637 | 1.466 | 1.167x | 1.045x |
| PyTorch | Square 4096 | FP32 | 16 | 84.483 | 104.162 | 93.279 | 1.233x | 1.104x |
| PyTorch | Square 8192 | FP32 | 16 | 653.788 | 812.927 | 717.535 | 1.243x | 1.098x |
| PyTorch | MLP forward | FP32 | 16 | 8.606 | 15.381 | 8.824 | 1.787x | 1.025x |
| PyTorch | Forward + backward | FP32 | 16 | 22.126 | 52.883 | 25.001 | 2.390x | 1.130x |
| PyTorch | Gram | FP64 | 16 | 1.957 | 3.661 | 3.142 | 1.870x | 1.605x |
| PyTorch | Transposed GEMM | FP64 | 16 | 3.004 | 5.011 | 4.099 | 1.668x | 1.365x |
| PyTorch | Attention | FP64 | 16 | 2.054 | 9.028 | 2.394 | 4.396x | 1.166x |
| PyTorch | Square 1024 | FP64 | 16 | 2.862 | 3.414 | 2.937 | 1.193x | 1.026x |
| PyTorch | Square 4096 | FP64 | 16 | 178.672 | 218.799 | 194.099 | 1.225x | 1.086x |
| PyTorch | Square 8192 | FP64 | 16 | 1396.580 | 1726.001 | 1479.843 | 1.236x | 1.060x |
| PyTorch | MLP forward | FP64 | 16 | 18.742 | 33.458 | 20.403 | 1.785x | 1.089x |
| PyTorch | Forward + backward | FP64 | 16 | 46.941 | 100.711 | 51.791 | 2.146x | 1.103x |
| PyTorch | Gram | FP32 | 64 | 0.310 | 0.916 | 0.487 | 2.959x | 1.572x |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.507 | 1.840 | 0.534 | 3.626x | 1.052x |
| PyTorch | Attention | FP32 | 64 | 0.993 | 2.478 | 1.473 | 2.494x | 1.483x |
| PyTorch | Square 1024 | FP32 | 64 | 0.391 | 0.487 | 0.396 | 1.248x | 1.013x |
| PyTorch | Square 4096 | FP32 | 64 | 28.346 | 37.242 | 39.217 | 1.314x | 1.384x |
| PyTorch | Square 8192 | FP32 | 64 | 198.366 | 259.604 | 258.619 | 1.309x | 1.304x |
| PyTorch | MLP forward | FP32 | 64 | 2.477 | 8.815 | 2.573 | 3.558x | 1.039x |
| PyTorch | Forward + backward | FP32 | 64 | 8.658 | 54.371 | 18.085 | 6.280x | 2.089x |
| PyTorch | Gram | FP64 | 64 | 0.570 | 2.334 | 0.885 | 4.095x | 1.553x |
| PyTorch | Transposed GEMM | FP64 | 64 | 0.981 | 4.018 | 1.950 | 4.097x | 1.989x |
| PyTorch | Attention | FP64 | 64 | 1.596 | 5.950 | 2.217 | 3.727x | 1.389x |
| PyTorch | Square 1024 | FP64 | 64 | 0.767 | 0.963 | 0.790 | 1.256x | 1.030x |
| PyTorch | Square 4096 | FP64 | 64 | 60.076 | 78.119 | 84.161 | 1.300x | 1.401x |
| PyTorch | Square 8192 | FP64 | 64 | 413.104 | 562.449 | 497.939 | 1.362x | 1.205x |
| PyTorch | MLP forward | FP64 | 64 | 7.189 | 29.679 | 10.308 | 4.128x | 1.434x |
| PyTorch | Forward + backward | FP64 | 64 | 17.975 | 87.763 | 32.521 | 4.883x | 1.809x |

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
