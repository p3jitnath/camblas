# CAMBLAS

CAMBLAS is an experimental CPU BLAS library for real single- and double-precision matrix multiplication. The primary target is NVIDIA Grace (AArch64, Neoverse V2, SVE128), with a portable reference build for other Linux CPUs.

An optional [CUDA backend](#experimental-camblas-cuda-backend) provides GPU matrix multiplication, affine/MLP operations and attention through PyTorch tensors. CPU-versus-GPU and CUDA-backend comparisons below include host transfers.

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

### CPU versus CUDA PyTorch

NVPL runs on the CPU. CUDA PyTorch uses cuBLAS on the GPU. The separate [CUDA comparison](bench/compare_cuda.py) measures CAMBLAS and NVPL CPU PyTorch against one GPU, using the same eight application workloads and FP32/FP64 inputs. Existing CPU environments are retained; install CUDA PyTorch 2.8.0 and NumPy 2.3.2 in `.frameworks/envs/cuda`, or supply its interpreter with `--cuda-python`. Select a CUDA wheel compatible with the node architecture and driver, and verify an actual GPU matrix multiplication before benchmarking.

```bash
# Run inside an allocated GPU node, with the CPU affinity on its attached Grace chip.
python3.11 bench/compare_cuda.py --threads 64 --output bench/results/cpu_gpu
# Use a plotting environment with NumPy and Matplotlib 3.7 or later.
.venv-paper/bin/python bench/plot_cuda.py bench/results/cpu_gpu
```

The comparison rotates CPU/CUDA order across three fresh-process rounds. GPU-resident wall-clock timings include synchronous eager execution; transfer timings additionally copy every input and weight to the GPU and return the output and all gradients to pageable CPU memory on every call. FP32 uses highest precision with TF32 disabled. CUDA-event timings are saved separately. Outputs and gradients are checked against NVPL using the existing samples and norms, with independent sampled dot-product checks for GEMM and Gram. The figures show medians and ranges of process medians; the report, raw observations, library identities, source snapshots and CSV remain in the ignored results directory. Plotting uses the graph-plotting skill's local style helper; specify `--style-dir` if it is installed elsewhere.

<!-- cpu-cuda-results start -->

CPU snapshot, 2 October 2026: 64 Grace cores versus one GH200 on `nid010961` (allocation `7004991`), three rotated fresh-process rounds. The GPU time includes every input/weight and output/gradient transfer. Slowdown = CPU time / CUDA PyTorch time; values above 1 mean the CPU is slower.

| Workload | CAMBLAS CPU FP32 slowdown | NVPL CPU FP32 slowdown | CAMBLAS CPU FP64 slowdown | NVPL CPU FP64 slowdown |
|---|---:|---:|---:|---:|
| Square 1024 | 2.13× | 2.09× | 1.41× | 1.46× |
| Square 4096 | 6.42× | 9.02× | 7.70× | 10.92× |
| Square 8192 | 6.84× | 9.07× | 12.40× | 15.72× |
| Transposed GEMM | 2.32× | 2.57× | 1.17× | 2.30× |
| Gram | 2.04× | 3.22× | 2.98× | 4.55× |
| MLP forward | 4.13× | 4.26× | 9.85× | 14.00× |
| Attention | 5.66× | 6.55× | 6.80× | 9.53× |
| Forward + backward | 2.87× | 5.95× | 2.92× | 5.40× |

<details>
<summary>CPU/GPU latency medians and fresh-process ranges (milliseconds)</summary>

Each cell is median [minimum–maximum] of three process medians.

| Workload | Precision | CAMBLAS CPU ms | NVPL CPU ms | PyTorch CUDA with transfers ms |
|---|---|---:|---:|---:|
| Square 1024 | FP32 | 0.4087 [0.3972–0.4340] | 0.4008 [0.3983–0.4037] | 0.1920 [0.1920–0.2035] |
| Square 4096 | FP32 | 26.4051 [26.3249–26.4120] | 37.0779 [36.5323–37.4589] | 4.1124 [4.0890–4.1169] |
| Square 8192 | FP32 | 183.5027 [183.1093–184.0763] | 243.3235 [214.8697–247.0135] | 26.8325 [26.8273–27.1727] |
| Transposed GEMM | FP32 | 0.4873 [0.4822–0.4886] | 0.5393 [0.5184–0.5420] | 0.2098 [0.2083–0.5197] |
| Gram | FP32 | 0.3100 [0.3062–0.3229] | 0.4903 [0.4878–0.5142] | 0.1523 [0.1499–0.1532] |
| MLP forward | FP32 | 2.4416 [2.4412–2.4488] | 2.5139 [2.5114–2.5169] | 0.5907 [0.5873–0.6005] |
| Attention | FP32 | 1.2461 [1.2347–1.3340] | 1.4422 [1.4231–1.4487] | 0.2201 [0.2136–0.2265] |
| Forward + backward | FP32 | 8.7331 [8.7158–8.7843] | 18.0713 [17.9629–18.1240] | 3.0384 [2.0943–3.1691] |
| Square 1024 | FP64 | 0.7638 [0.7583–0.7715] | 0.7920 [0.7905–0.7971] | 0.5432 [0.5208–0.5460] |
| Square 4096 | FP64 | 55.1820 [54.9802–55.4411] | 78.2460 [65.7531–78.8034] | 7.1658 [6.9774–7.1839] |
| Square 8192 | FP64 | 386.9441 [386.7062–387.4346] | 490.4928 [460.3166–491.4267] | 31.1952 [30.9000–31.7111] |
| Transposed GEMM | FP64 | 1.0030 [0.9842–1.0184] | 1.9679 [1.9544–2.0253] | 0.8562 [0.2783–0.8604] |
| Gram | FP64 | 0.5809 [0.5730–0.5869] | 0.8869 [0.8808–0.9008] | 0.1950 [0.1936–0.1967] |
| MLP forward | FP64 | 7.1746 [7.1320–7.2019] | 10.1995 [10.0897–10.4657] | 0.7286 [0.7168–0.7286] |
| Attention | FP64 | 1.6367 [1.5753–1.7214] | 2.2943 [2.2919–2.3498] | 0.2407 [0.2342–0.2504] |
| Forward + backward | FP64 | 17.2359 [17.2357–17.3243] | 31.8487 [31.3563–34.6956] | 5.8987 [5.8899–6.0717] |

</details>

<!-- cpu-cuda-results end -->

### Experimental CAMBLAS CUDA backend

The optional [CUDA backend](src/cuda/backend.cu) operates on GPU memory and interoperates with PyTorch CUDA tensors through [camblas_gpu](camblas_gpu/__init__.py). It provides matrix multiplication, affine transforms, a two-layer ReLU MLP and dense single-head attention in FP32 and FP64. The CPU implementation remains available separately. NVPL is a CPU library; this backend uses cuBLAS/cuBLASLt and independently implemented CUDA packing, recombination, softmax and gradient kernels.

Build from the repository root with a CUDA toolkit and a compatible CUDA PyTorch environment. The default architecture is Hopper `sm_90`; pass `--architecture` for another supported device. `--torch` builds a thin tensor binding against the active PyTorch installation. Without it, the Python adapter uses ctypes with greater host overhead.

```bash
.frameworks/envs/cuda/bin/python scripts/build_cuda.py \
  --cuda-root /path/to/cuda --cxx g++-14 --torch
.frameworks/envs/cuda/bin/python -m unittest discover -s tests -p test_cuda.py -v
python3.11 bench/compare_gpu.py --threads 64 --output bench/results/camblas_gpu
```

From the repository root, use `import camblas_gpu as cb` and `cb.matmul(a, b)`, `cb.affine(x, weight, bias, relu=True)`, `cb.mlp(x, w1, b1, w2, b2)` or `cb.attention(q, k, v)`. Operands must be CUDA FP32/FP64 tensors on the same device. Matrix multiplication accepts transpose views and padded row/column input storage; other irregular layouts are materialised. Supplied `out` tensors must have contiguous row storage and their mutations update PyTorch's version counter. Affine, MLP and attention use contiguous storage. Operations follow the active PyTorch stream; ordering and operand lifetimes across streams follow PyTorch's ordinary CUDA rules. Matrix multiplication and affine support autograd; MLP supports first derivatives, and attention uses differentiable matrix products and PyTorch softmax when gradients are requested.

Automatic dispatch uses guarded Strassen for large even square products, with fused packing and recombination for two levels, and classical/cuBLASLt kernels elsewhere. It never enables TF32 or reduces the compute dtype. Strassen changes summation order and can worsen relative error when dot products cancel; range checks address overflow, rather than guaranteeing relative accuracy. Request `with cb.algorithm("classical")` to use classical multiplication, including backward. The explicit `lt`, `strassen`, `strassen2` and `symmetric` modes aid diagnosis. Symmetric Gram multiplication is experimental and is excluded from automatic dispatch after measured regressions. Guarded Strassen performs a stream synchronisation for its input-range check and falls back to classical GEMM if scratch allocation fails.

Warm up operations on their intended stream before CUDA graph capture. Guarded Strassen falls back to classical multiplication during capture; warmed FP32 affine operations retain their cuBLASLt bias epilogue. Captured private scratch remains allocated until its context is closed, even after later growth; destroy graphs before closing their contexts. Replay on the capture stream or explicitly serialise replay with other work using that context.

On GPUs reporting coherent pageable memory through host page tables, `cb.matmul_host`, `cb.mlp_host` and `cb.attention_host` provide a separate inference experiment using ordinary CPU tensors and completed CPU output. They synchronise before returning and issue no explicit tensor copies; GPU kernels access CPU storage directly, with GPU scratch for hidden MLP values and attention scores. This can benefit small workloads and slow large GEMMs. It is opt-in, requires the native tensor binding and rejects gradient-enabled inputs. The main comparison always includes explicit copies of every operand and output. Pass `--coherent-host` to `bench/compare_gpu.py` to report this separate CPU-to-CPU experiment alongside the copy pipelines; backward is excluded.

The GPU comparison rotates eager PyTorch, automatic CAMBLAS and a classical CAMBLAS control across at least three fresh processes per case. It includes every input/weight and output/gradient transfer on every transfer-mode call, verifies numerical signatures and scalar-dot oracles, and records source/binary identities, launch counters, affinity and process-median ranges. `cb.stats(all_threads=True)` includes launches from PyTorch autograd workers. Build records and measurements remain under ignored `build/` and `bench/results/` directories.

`--pytorch-baseline fused` selects PyTorch's `linear` and [scaled-dot-product attention](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) APIs. Dedicated linear checks prepare contiguous PyTorch weights once before timing; mathematical values and transfer bytes match CAMBLAS, which keeps its own natural weight layout. `compiled` compiles the eager expressions and `compiled-fused` compiles the dedicated APIs. Choose [compiler modes](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html) with `--compile-mode`; compilation and tuning occur during warm-up, while execution, graph replay and any input staging remain timed.

```bash
python3.11 bench/compare_gpu.py --threads 64 --rounds 9 \
  --output bench/results/gpu_final
python3.11 bench/compare_gpu.py --threads 64 --workloads mlp attention backward \
  --pytorch-baseline fused --output bench/results/gpu_fused
python3.11 bench/compare_gpu.py --threads 64 --pytorch-baseline compiled \
  --output bench/results/gpu_compiled
python3.11 bench/compare_gpu.py --threads 64 --workloads square8192 mlp attention backward \
  --pytorch-baseline compiled --compile-mode max-autotune-no-cudagraphs \
  --output bench/results/gpu_autotuned
python3.11 bench/compare_gpu.py --threads 64 --workloads mlp attention backward \
  --pytorch-baseline compiled-fused --output bench/results/gpu_compiled_fused
python3.11 bench/compare_gpu.py --threads 64 --coherent-host \
  --output bench/results/gpu_coherent
```

<!-- cuda-results start -->

#### CUDA performance snapshot

Nine-round comparison, 3 October 2026: one GH200, 64 attached Grace host cores (CPUs 0–63), PyTorch 2.8.0+cu129, NumPy 2.3.2, CUDA toolkit 12.9.1 (nvcc 12.9.86), GCC 14.3.0 and `sm_90`. 9 rotated fresh processes per backend/case compare eager PyTorch, automatic CAMBLAS CUDA and its classical control. TF32 is disabled; storage and compute remain FP32/FP64. No other GPU processes were present at the suite boundaries. Timings include allocation, dispatch and completion; transfer mode also copies every operand/weight and every output/parameter gradient on every call. Speed-up = PyTorch / CAMBLAS; values above 1 favour CAMBLAS.

Square GEMMs use N=1024/4096/8192. Transposed GEMM multiplies (512×1024)ᵀ by (512×2048); Gram computes XᵀX for X=(4096×512). MLP uses batch 512 and widths 2048→4096→1024, with ReLU after the first affine transform. Attention uses 1024 query/key/value rows of width 256, scaled by 1/16, without a mask. Forward + backward uses the same MLP with a mean-squared-output loss and gradients for both weights and biases. Inputs share seed 20260906. Ten warm-up calls precede 21 timed calls per mode, or 201 for MLP/backward.

This suite predates the final widening of softmax column indices and backward-bias grid arithmetic at the LP64 limit. GEMM dispatch and packing were unchanged. The changed neural paths and 8192² GEMM were checked again with the final binary in the verification tables below; the other GEMM shapes were not remeasured after that final change.

| Workload | FP32 resident speed-up | FP32 with transfers | FP64 resident speed-up | FP64 with transfers |
|---|---:|---:|---:|---:|
| Square 1024 | 1.017× | 0.990× | 1.023× | 0.991× |
| Square 4096 | 1.079× | 1.038× | 1.041× | 1.002× |
| Square 8192 | 1.272× | 1.212× | 1.252× | 1.139× |
| Transposed GEMM | 0.994× | 0.964× | 0.967× | 1.008× |
| Gram | 0.977× | 0.966× | 1.018× | 0.997× |
| MLP forward | 1.003× | 0.987× | 0.989× | 1.022× |
| Attention | 1.516× | 1.240× | 1.737× | 1.272× |
| Forward + backward | 1.023× | 1.010× | 1.053× | 1.013× |

<details>
<summary>Latency medians and fresh-process ranges (milliseconds)</summary>

Each cell is median [minimum–maximum] of the 9 process medians.

| Workload | Precision | PyTorch resident ms | CAMBLAS resident ms | PyTorch with transfers ms | CAMBLAS with transfers ms |
|---|---|---:|---:|---:|---:|
| Square 1024 | FP32 | 0.0857 [0.0841–0.0865] | 0.0842 [0.0832–0.0863] | 0.1929 [0.1877–0.1971] | 0.1948 [0.1930–0.1985] |
| Square 1024 | FP64 | 0.0745 [0.0733–0.0767] | 0.0728 [0.0719–0.0739] | 0.5370 [0.5222–0.5450] | 0.5417 [0.5213–0.5486] |
| Square 4096 | FP32 | 2.8035 [2.7540–2.8584] | 2.5987 [2.5718–2.6547] | 4.0811 [4.0375–4.1148] | 3.9312 [3.8789–3.9558] |
| Square 4096 | FP64 | 3.0883 [2.9778–3.2794] | 2.9654 [2.5246–3.1733] | 6.7816 [6.5681–7.2906] | 6.7694 [6.6654–6.9987] |
| Square 8192 | FP32 | 22.9782 [22.8900–23.0250] | 18.0647 [17.9570–18.1506] | 26.9189 [26.7328–27.0583] | 22.2020 [22.1746–22.4970] |
| Square 8192 | FP64 | 24.7292 [24.6104–25.0005] | 19.7456 [19.6551–19.8173] | 30.7527 [30.2009–31.0882] | 26.9982 [26.7725–29.1802] |
| Transposed GEMM | FP32 | 0.0856 [0.0842–0.0878] | 0.0861 [0.0852–0.0872] | 0.2135 [0.2067–0.5073] | 0.2215 [0.2140–0.5100] |
| Transposed GEMM | FP64 | 0.0817 [0.0792–0.0844] | 0.0845 [0.0801–0.0861] | 0.8468 [0.2788–0.8725] | 0.8397 [0.2821–0.8680] |
| Gram | FP32 | 0.0831 [0.0812–0.0838] | 0.0851 [0.0830–0.0869] | 0.1559 [0.1495–0.1627] | 0.1614 [0.1531–0.1635] |
| Gram | FP64 | 0.0873 [0.0851–0.0889] | 0.0857 [0.0840–0.0863] | 0.1903 [0.1828–0.1933] | 0.1910 [0.1863–0.1963] |
| MLP forward | FP32 | 0.3297 [0.3239–0.3336] | 0.3286 [0.3268–0.3340] | 0.5929 [0.5894–0.5985] | 0.6007 [0.5869–0.6036] |
| MLP forward | FP64 | 0.3236 [0.3069–0.3543] | 0.3271 [0.3010–0.3409] | 0.7270 [0.7163–0.7400] | 0.7113 [0.7094–0.7221] |
| Attention | FP32 | 0.1239 [0.1177–0.1278] | 0.0817 [0.0792–0.0829] | 0.2241 [0.2117–0.2328] | 0.1807 [0.1750–0.1895] |
| Attention | FP64 | 0.1312 [0.1218–0.1348] | 0.0755 [0.0736–0.0772] | 0.2458 [0.2366–0.2567] | 0.1933 [0.1872–0.2053] |
| Forward + backward | FP32 | 0.7989 [0.7916–0.8078] | 0.7806 [0.7729–0.7872] | 3.0414 [2.0712–3.1807] | 3.0123 [2.9287–3.2230] |
| Forward + backward | FP64 | 0.8310 [0.8006–0.8404] | 0.7891 [0.7834–0.8127] | 5.8460 [5.8157–6.0828] | 5.7732 [5.7521–6.0284] |

The classical control retains the same native fusions but disables Strassen and cuBLASLt dispatch. For 8192² GEMM:

| Precision | Classical control resident ms | Automatic CAMBLAS resident ms | Speed-up over control |
|---|---:|---:|---:|
| FP32 | 22.9892 | 18.0647 | 1.273× |
| FP64 | 24.7845 | 19.7456 | 1.255× |

Nine-round native library SHA-256: `5813faec3092be16c75cb0af00b90f579a396d8bcb9f6464f927c04f2a8dd93e`. Tensor binding SHA-256: `f996ee5eb5553d09932eb263d0c05db051a6825b2b9e24e97b4acf2d657e4039`. Source snapshots, build commands, input/vendor hashes, counters, raw rounds and scalar-dot checks accompany the local report under `bench/results/`; generated artefacts are excluded from Git.

</details>

Differences near 1× can lie within the process ranges. Transfer timings include CPU-output allocation and varied substantially on transposed GEMM in earlier rounds; these are pipeline timings, rather than pure kernel speed-ups. Strassen changes rounding and does not guarantee relative accuracy under cancellation. An additional three-round resident shape sweep found a roughly 3% FP32 regression at N=4098 and 1.20–1.30× speed-ups at N=12288/16384. This backend is not uniformly faster for every shape.

Fused two-level packing/recombination reduces private Strassen scratch by 27.6%: 8192² needs about 2.30 GiB in FP32 or 4.59 GiB in FP64, plus operands, outputs and cuBLAS workspace. Against an unchanged pre-fusion CUDA control, three rotated fresh-process rounds improved resident throughput by 3.7%/7.1% (FP32/FP64).

<details>
<summary>Dedicated/compiled PyTorch and coherent-memory checks</summary>

These separate suites used three rotated fresh processes per backend/case on 2 October. They were collected before the final oversized-attention scratch validation and LP64 index hardening; those variants were not remeasured with the final binary. GEMM dispatch and packing were retained. Each cell below is CAMBLAS speed-up over that PyTorch variant, resident / including transfers. Compiled checks use reduce-overhead; autotuned checks use max-autotune-no-cudagraphs. All preserve FP32/FP64 and disable TF32. Compilation, tuning and one-time weight preparation are excluded, while graph replay and staging are included.

| Workload | Precision | Dedicated APIs | Compiled eager | Autotuned eager | Compiled dedicated |
|---|---|---:|---:|---:|---:|
| MLP forward | FP32 | 1.247× / 1.125× | 1.326× / 1.181× | 1.215× / 1.130× | 1.452× / 1.231× |
| MLP forward | FP64 | 1.794× / 1.294× | 1.503× / 1.267× | 1.085× / 1.051× | 1.461× / 1.262× |
| Attention | FP32 | 2.320× / 1.609× | 1.696× / 1.496× | 1.317× / 1.159× | 3.131× / 2.046× |
| Attention | FP64 | 2.813× / 1.755× | 1.828× / 1.453× | 1.756× / 1.340× | 1.883× / 1.466× |
| Forward + backward | FP32 | 1.137× / 1.011× | 1.176× / 1.004× | 1.127× / 1.040× | 1.198× / 1.094× |
| Forward + backward | FP64 | 1.330× / 1.023× | 1.236× / 1.068× | 1.154× / 1.027× | 1.203× / 1.026× |

PyTorch SDPA selected its efficient-attention implementation in FP32 and its math implementation in FP64, verified from operator profiles. The full 16-case compiled-eager run also recorded transfer regressions for CAMBLAS at FP64 square 1024 (0.597×; PyTorch 0.309–0.645 ms, CAMBLAS 0.518–0.545 ms) and square 4096 (0.989×); resident CAMBLAS was faster in those cases. The autotuned 8192² comparison retained 1.284× / 1.214× FP32 and 1.273× / 1.181× FP64 speed-ups.

The coherent-host experiment starts with ordinary CPU tensors and returns completed CPU tensors. It includes hardware memory traffic but issues no explicit bulk tensor copies. It is opt-in, supports inference only and includes substantial regressions. Latency cells show median [minimum–maximum] process medians in milliseconds; speed-up compares against PyTorch's explicit-copy CPU-to-CPU pipeline.

| Workload | FP32 coherent ms | FP32 speed-up | FP64 coherent ms | FP64 speed-up |
|---|---:|---:|---:|---:|
| Square 1024 | 0.0944 [0.0910–0.0971] | 2.095× | 1.5428 [1.1386–2.1850] | 0.340× |
| Square 4096 | 13.6672 [13.5131–13.8091] | 0.300× | 24.3839 [24.0748–24.4125] | 0.289× |
| Square 8192 | 66.3676 [66.2141–66.8679] | 0.406× | 41.7480 [40.0182–42.2734] | 0.734× |
| Transposed GEMM | 2.2550 [2.2415–2.3832] | 0.097× | 0.1058 [0.1019–3.6230] | 7.814× |
| Gram | 0.0981 [0.0972–0.0996] | 1.552× | 0.1051 [0.1002–0.1068] | 1.822× |
| MLP forward | 0.3533 [0.3479–0.3596] | 1.664× | 0.3880 [0.3762–0.4251] | 1.872× |
| Attention | 0.0923 [0.0905–0.0946] | 2.437× | 0.1011 [0.0898–0.1034] | 2.376× |

</details>

<!-- cuda-results end -->

<!-- final-verification start -->

#### Final verification

On 3 October 2026, 01:02:25–01:06:32 UTC, in the last five minutes of allocation `7004991`, CPU `make test` and all 38 native Python tests (31 CUDA and seven tool tests) passed. The ctypes CUDA path passed 29 tests with two native-only skips. Compute Sanitizer reported zero errors for CAMBLAS kernels in the 30 ordinary CUDA regressions under memcheck and the two graph scratch-growth tests under synccheck. The separate LP64 regression requires 24 GiB free GPU memory and checks every bias-gradient element at INT_MAX width in FP32/FP64 outside instrumentation. Formatting, Python compilation and whitespace checks passed. The source and binary hashes matched the final neural and final-window runs.

After index hardening, all six neural cases were remeasured with three rotated fresh-process rounds, five timed calls per mode and ten warm-up calls. Speed-up = PyTorch / CAMBLAS.

| Workload | FP32 resident | FP32 with transfers | FP64 resident | FP64 with transfers |
|---|---:|---:|---:|---:|
| MLP forward | 1.007× | 0.990× | 1.029× | 1.003× |
| Attention | 1.481× | 1.160× | 1.704× | 1.173× |
| Forward + backward | 1.026× | 1.120× | 1.087× | 0.988× |

Final native library SHA-256: `0a88747a19396d2f94b97497e990cdea0358700e60c506adc48e307795dd886d`. Tensor binding SHA-256: `f996ee5eb5553d09932eb263d0c05db051a6825b2b9e24e97b4acf2d657e4039`.

A fresh final-window comparison checked square 8192 and attention in both precisions using three rotated process rounds, five timed calls per mode and ten warm-up calls. The following speed-ups use PyTorch / CAMBLAS; the larger nine-round suite above remains the broader performance measurement before index hardening.

| Workload | FP32 resident | FP32 with transfers | FP64 resident | FP64 with transfers |
|---|---:|---:|---:|---:|
| Square 8192 | 1.271× | 1.204× | 1.240× | 1.136× |
| Attention | 1.515× | 1.167× | 1.678× | 1.298× |

<details>
<summary>Final-binary latency medians and process ranges (milliseconds)</summary>

Each cell is median [minimum–maximum] of three process medians.

| Suite | Workload | Precision | PyTorch resident ms | CAMBLAS resident ms | PyTorch with transfers ms | CAMBLAS with transfers ms |
|---|---|---|---:|---:|---:|---:|
| Neural | MLP forward | FP32 | 0.3392 [0.3336–0.3408] | 0.3369 [0.3331–0.3401] | 0.6010 [0.5909–0.6066] | 0.6073 [0.5966–0.6199] |
| Neural | MLP forward | FP64 | 0.3132 [0.3106–0.3198] | 0.3042 [0.3030–0.3055] | 0.7254 [0.7221–0.7422] | 0.7234 [0.7232–0.7247] |
| Neural | Attention | FP32 | 0.1253 [0.1199–0.1290] | 0.0846 [0.0838–0.0846] | 0.2193 [0.2178–0.2266] | 0.1891 [0.1835–0.1923] |
| Neural | Attention | FP64 | 0.1283 [0.1280–0.1340] | 0.0753 [0.0747–0.0769] | 0.2390 [0.2256–0.2593] | 0.2038 [0.1986–0.2102] |
| Neural | Forward + backward | FP32 | 0.8085 [0.8043–0.8122] | 0.7879 [0.7850–0.7898] | 3.1844 [3.0406–3.3374] | 2.8421 [2.2333–2.8792] |
| Neural | Forward + backward | FP64 | 0.7744 [0.7505–0.7812] | 0.7126 [0.7091–0.7144] | 5.8430 [5.8032–6.1057] | 5.9141 [5.8198–5.9820] |
| Final window | Square 8192 | FP32 | 22.8115 [22.7771–22.9027] | 17.9486 [17.8296–18.0875] | 26.8176 [26.7485–27.0811] | 22.2649 [22.1290–22.3144] |
| Final window | Square 8192 | FP64 | 24.5732 [24.2450–24.7886] | 19.8214 [19.6960–19.9198] | 30.7459 [30.4849–31.5198] | 27.0657 [26.5348–27.3418] |
| Final window | Attention | FP32 | 0.1262 [0.1256–0.1262] | 0.0833 [0.0816–0.0836] | 0.2171 [0.2166–0.2492] | 0.1861 [0.1800–0.1932] |
| Final window | Attention | FP64 | 0.1306 [0.1190–0.1406] | 0.0778 [0.0724–0.0780] | 0.2494 [0.2277–0.2540] | 0.1922 [0.1832–0.2036] |

</details>


```bash
python3.11 bench/compare_gpu.py --threads 64 --rounds 3 \
  --workloads square8192 attention --repetitions 5 \
  --output bench/results/gpu_final_window
python3.11 bench/compare_gpu.py --threads 64 --rounds 3 \
  --workloads mlp attention backward --repetitions 5 \
  --output bench/results/gpu_final_neural
make test CC=gcc-14 PYTHON=python3.11
.frameworks/envs/cuda/bin/python -m unittest discover -s tests -v
# Instrument CAMBLAS kernels; CUDA vendor kernels are excluded.
compute-sanitizer --tool memcheck --error-exitcode 1 --target-processes all \
  --kernel-name 'regex=.*(pack_strassen|recombine_strassen|softmax_rows|backward_bias|finish_bias|bias_activation|scale_output|mirror_triangle).*' \
  .frameworks/envs/cuda/bin/python -m unittest discover -s tests -p test_cuda.py \
  -k CudaTests -v
compute-sanitizer --tool synccheck --error-exitcode 1 --target-processes all \
  --kernel-name 'regex=.*(softmax_rows|backward_bias|finish_bias|bias_activation).*' \
  .frameworks/envs/cuda/bin/python -m unittest discover -s tests -p test_cuda.py \
  -k graph_scratch_growth -v
.frameworks/style-env/bin/python scripts/style.py
python3.11 -m compileall -q scripts bench tests camblas_gpu
git diff --check
```

<!-- final-verification end -->

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and [LICENSE](LICENSE) for the MIT licence, copyright Pritthijit Nath. External dependencies retain their own licences; generated artefacts and internal documents are excluded from this repository.
