# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 kernels, packed operand panels and shape-dependent task grids. Bounded paths use twelve-row FP32 or six-row FP64 layouts; transposed FP32 inputs can be packed directly into the kernel layout without an intermediate panel. Bounded square and transposed products also use one-level Strassen: seven half-size products, with operand sums formed during packing. Every call repacks its inputs, and the conservative finite-range preflight rides on the pack pass itself (per-worker abs-maxima reduced after packing) so no separate operand scan runs before the packed route; unsafe ranges still fall back to the classical path. At 64 cores a persistent spinning worker-pool executor replaces per-call OpenMP fork/join (workers spin only while a GEMM call is open and sleep on a futex once it is fenced, so idle cores stay available to the application's own thread pools); deep PyTorch chains at 64 cores, and every call below 64 threads, keep the OpenMP executor. Runtime capability, bounds, workspace and finite-range checks retain classical fallbacks. Strassen changes rounding behaviour; the range check limits overflow growth, not general relative error.

## Benchmarks

Application benchmark snapshot: **59/60 wins against OpenBLAS; 49/60 against NVPL**. The table contains all 60 cases from one complete run of this source configuration on an idle whole-node allocation (Grace, exclusive 288-core node): seven NumPy and eight PyTorch workloads, each at two precisions and 16/64 cores. Small median differences do not establish statistical significance, so entries within roughly two percent of 1x can move between sessions.

FP32 and FP64 denote single and double precision, respectively. Latency is reported in milliseconds, with lower values indicating better performance. Speed-up is vendor latency divided by CAMBLAS latency; **values above 1x favour CAMBLAS**. Each entry is the median of three fresh-process medians after warm-up, with all libraries using matching inputs, affinity and requested thread counts.

| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms | vs OpenBLAS | vs NVPL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NumPy | Gram | FP32 | 16 | 1.132 | 1.759 | 1.170 | 1.554x | 1.033x |
| NumPy | Transposed GEMM | FP32 | 16 | 1.351 | 1.618 | 1.447 | 1.198x | 1.072x |
| NumPy | Attention | FP32 | 16 | 4.142 | 4.337 | 4.177 | 1.047x | 1.008x |
| NumPy | Square 1024 | FP32 | 16 | 1.374 | 1.636 | 1.443 | 1.191x | 1.050x |
| NumPy | Square 4096 | FP32 | 16 | 86.516 | 105.033 | 96.108 | 1.214x | 1.111x |
| NumPy | Square 8192 | FP32 | 16 | 654.216 | 821.296 | 726.663 | 1.255x | 1.111x |
| NumPy | MLP forward | FP32 | 16 | 10.592 | 10.952 | 10.118 | 1.034x | 0.955x |
| NumPy | Gram | FP64 | 16 | 2.164 | 2.951 | 2.353 | 1.363x | 1.087x |
| NumPy | Transposed GEMM | FP64 | 16 | 2.811 | 3.406 | 2.910 | 1.212x | 1.035x |
| NumPy | Attention | FP64 | 16 | 7.106 | 7.391 | 6.827 | 1.040x | 0.961x |
| NumPy | Square 1024 | FP64 | 16 | 2.846 | 3.396 | 2.919 | 1.193x | 1.026x |
| NumPy | Square 4096 | FP64 | 16 | 181.587 | 219.547 | 200.061 | 1.209x | 1.102x |
| NumPy | Square 8192 | FP64 | 16 | 1395.351 | 1737.422 | 1506.758 | 1.245x | 1.080x |
| NumPy | MLP forward | FP64 | 16 | 22.648 | 26.613 | 21.935 | 1.175x | 0.969x |
| NumPy | Gram | FP32 | 64 | 0.419 | 13.412 | 0.596 | 32.041x | 1.425x |
| NumPy | Transposed GEMM | FP32 | 64 | 0.397 | 0.511 | 0.383 | 1.287x | 0.965x |
| NumPy | Attention | FP32 | 64 | 3.610 | 4.052 | 4.425 | 1.122x | 1.226x |
| NumPy | Square 1024 | FP32 | 64 | 0.417 | 0.508 | 0.390 | 1.219x | 0.935x |
| NumPy | Square 4096 | FP32 | 64 | 26.994 | 35.778 | 37.553 | 1.325x | 1.391x |
| NumPy | Square 8192 | FP32 | 64 | 192.389 | 258.342 | 237.560 | 1.343x | 1.235x |
| NumPy | MLP forward | FP32 | 64 | 4.099 | 4.080 | 3.821 | 0.995x | 0.932x |
| NumPy | Gram | FP64 | 64 | 0.834 | 27.591 | 1.034 | 33.093x | 1.240x |
| NumPy | Transposed GEMM | FP64 | 64 | 0.767 | 0.965 | 0.806 | 1.258x | 1.050x |
| NumPy | Attention | FP64 | 64 | 6.206 | 6.404 | 6.237 | 1.032x | 1.005x |
| NumPy | Square 1024 | FP64 | 64 | 0.784 | 0.983 | 0.785 | 1.254x | 1.000x |
| NumPy | Square 4096 | FP64 | 64 | 57.014 | 76.119 | 80.576 | 1.335x | 1.413x |
| NumPy | Square 8192 | FP64 | 64 | 403.930 | 556.581 | 498.430 | 1.378x | 1.234x |
| NumPy | MLP forward | FP64 | 64 | 9.012 | 20.136 | 11.968 | 2.234x | 1.328x |
| PyTorch | Gram | FP32 | 16 | 1.005 | 1.698 | 1.544 | 1.689x | 1.536x |
| PyTorch | Transposed GEMM | FP32 | 16 | 1.492 | 2.463 | 1.633 | 1.650x | 1.095x |
| PyTorch | Attention | FP32 | 16 | 1.260 | 9.463 | 1.304 | 7.512x | 1.035x |
| PyTorch | Square 1024 | FP32 | 16 | 1.396 | 1.633 | 1.456 | 1.170x | 1.043x |
| PyTorch | Square 4096 | FP32 | 16 | 86.387 | 104.243 | 95.158 | 1.207x | 1.102x |
| PyTorch | Square 8192 | FP32 | 16 | 653.878 | 813.068 | 721.760 | 1.243x | 1.104x |
| PyTorch | MLP forward | FP32 | 16 | 9.146 | 19.408 | 8.895 | 2.122x | 0.973x |
| PyTorch | Forward + backward | FP32 | 16 | 23.836 | 68.347 | 25.234 | 2.867x | 1.059x |
| PyTorch | Gram | FP64 | 16 | 1.975 | 3.652 | 3.131 | 1.850x | 1.585x |
| PyTorch | Transposed GEMM | FP64 | 16 | 3.004 | 5.029 | 3.534 | 1.674x | 1.177x |
| PyTorch | Attention | FP64 | 16 | 2.492 | 14.897 | 2.566 | 5.977x | 1.029x |
| PyTorch | Square 1024 | FP64 | 16 | 2.880 | 3.385 | 2.920 | 1.175x | 1.014x |
| PyTorch | Square 4096 | FP64 | 16 | 181.923 | 218.760 | 193.710 | 1.202x | 1.065x |
| PyTorch | Square 8192 | FP64 | 16 | 1395.662 | 1718.591 | 1495.500 | 1.231x | 1.072x |
| PyTorch | MLP forward | FP64 | 16 | 21.173 | 31.134 | 20.722 | 1.470x | 0.979x |
| PyTorch | Forward + backward | FP64 | 16 | 49.577 | 95.593 | 53.442 | 1.928x | 1.078x |
| PyTorch | Gram | FP32 | 64 | 0.307 | 0.916 | 0.492 | 2.986x | 1.603x |
| PyTorch | Transposed GEMM | FP32 | 64 | 0.510 | 1.949 | 1.077 | 3.822x | 2.113x |
| PyTorch | Attention | FP32 | 64 | 0.943 | 2.378 | 1.395 | 2.521x | 1.480x |
| PyTorch | Square 1024 | FP32 | 64 | 0.430 | 0.509 | 0.404 | 1.184x | 0.940x |
| PyTorch | Square 4096 | FP32 | 64 | 27.479 | 36.024 | 38.440 | 1.311x | 1.399x |
| PyTorch | Square 8192 | FP32 | 64 | 194.149 | 252.719 | 253.077 | 1.302x | 1.304x |
| PyTorch | MLP forward | FP32 | 64 | 2.618 | 7.272 | 2.557 | 2.777x | 0.976x |
| PyTorch | Forward + backward | FP32 | 64 | 8.796 | 91.539 | 17.889 | 10.407x | 2.034x |
| PyTorch | Gram | FP64 | 64 | 0.577 | 2.368 | 0.895 | 4.102x | 1.549x |
| PyTorch | Transposed GEMM | FP64 | 64 | 1.009 | 4.049 | 1.538 | 4.011x | 1.524x |
| PyTorch | Attention | FP64 | 64 | 1.737 | 32.382 | 2.434 | 18.644x | 1.401x |
| PyTorch | Square 1024 | FP64 | 64 | 0.810 | 0.970 | 0.794 | 1.197x | 0.980x |
| PyTorch | Square 4096 | FP64 | 64 | 56.981 | 74.299 | 82.525 | 1.304x | 1.448x |
| PyTorch | Square 8192 | FP64 | 64 | 407.521 | 549.940 | 474.220 | 1.349x | 1.164x |
| PyTorch | MLP forward | FP64 | 64 | 7.036 | 32.315 | 12.420 | 4.593x | 1.765x |
| PyTorch | Forward + backward | FP64 | 64 | 17.483 | 71.370 | 35.922 | 4.082x | 2.055x |

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
