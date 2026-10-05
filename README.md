# CAMBLAS

CAMBLAS provides CPU matrix multiplication for NVIDIA Grace and an optional CUDA backend for PyTorch and LLM inference. CPU products support FP32 and FP64; CUDA linear products also support BF16 storage with FP32 accumulation. The project is experimental, so use matching headers and libraries.

## Install

For the CPU library, use Linux, Python 3.11+, Make and a C11 compiler:

```bash
git clone git@github.com:p3jitnath/camblas.git
cd camblas
make reference CC=cc PYTHON=python3.11
```

On NVIDIA Grace with SVE, use `make grace CC=gcc-14 PYTHON=python3.11` instead. The library is written to `build/reference/` or `build/grace/`; the [headers](include/) define the native API.

For CUDA, first install a compatible CUDA toolkit and PyTorch in your Python environment, then build the tensor binding:

```bash
python -m pip install --no-deps -e .
python scripts/build_cuda.py --torch --architecture sm_90a --cxx g++-14 \
  --cuda-root "$CUDA_HOME"
```

Use `sm_90a` for Hopper inference kernels, or select an architecture supported by your GPU. The [pinned SGLang runtime](configs/sglang.json) and [requirements](configs/sglang-requirements.txt) describe the measured GH200 environment.

## Use

Run Python from the checkout, with CUDA tensors on the same device:

```python
import torch
import camblas_gpu as cb

x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(4096, 2048, device="cuda", dtype=x.dtype)
y = cb.linear(x, weight)
```

The [Python API](camblas_gpu/__init__.py) also provides matrix multiplication, MLP, attention, normalisation and quantised inference operations. Operations use the active PyTorch stream; warm them on that stream before CUDA graph capture. CPU framework adapters require LP64 OpenBLAS for unsupported BLAS and LAPACK operations.

To enable the SGLang plugin, set `SGLANG_PLUGINS=camblas`, `CAMBLAS_SGLANG_OPS=linear,fp8` and `CAMBLAS_CUDA_LIBRARY` to the built `libcamblas_cuda.so`. The expanded instructions below describe model verification and the complete benchmark protocol.

## Details and results

<details>
<summary>SGLang setup and benchmark commands</summary>

### Full-model SGLang comparisons

[bench/compare_llm.py](bench/compare_llm.py) compares the same pinned SGLang engine with and without the opt-in CAMBLAS linear and FP8 plugin. Both routes use tensor parallelism across four GH200 GPUs, batch one, ordinary greedy decoding and the original checkpoint files. The [runtime settings](configs/sglang.json) and [dependencies](configs/sglang-requirements.txt) pin SGLang's Python source, CUDA PyTorch and native source revisions. SGLang's native extensions must be rebuilt against the selected PyTorch/CUDA ABI; the driver records the libraries actually mapped into every model rank.

The complete Llama 3.1 70B checkpoint is pinned to revision 349b2ddb53ce8f2849a6c168a81980ab25258dac, with 70,553,706,496 BF16 parameters in 30 original shards. DeepSeek V4.1 Flash is pinned to 2cba9e42aa026125f3ed06c6d98c1db82f7ca027, with all 48 original shards verified. The [Llama manifest](configs/llama31-70b-weights.json) and [DeepSeek manifest](configs/deepseek-v41-flash-weights.json) include configuration, tokenizer and index hashes. The verifier counts stored elements; packed FP4 bytes and scaling tensors are not a logical parameter count. DeepSeek serves text through all 40 backbone layers; vision and speculative prediction are outside this comparison.

Llama uses BF16 weights and activations with FP32 accumulation. DeepSeek retains FP4 routed-expert weights with BF16 expert activations, original block-scaled FP8 dense weights with dynamic FP8 activation scaling, FP32 accumulation and SGLang's FP8 E4M3 KV storage. The pinned engine dequantises the checkpoint's 32×32-scaled attention `wo_a` weights to BF16 during loading; request timings use these prepared weights. Its Engram lookup tables remain in prepared host memory and are gathered by the engine on each request. TF32 and BF16 reduced-precision GEMM reduction are disabled in every rank.

The CAMBLAS plugin selects contiguous, unquantised BF16/FP32 CUDA linear products through cuBLASLt. DeepSeek also uses native SM90a WGMMA for seven single-token block32 FP8 weight shapes. Six shapes fuse BF16-to-FP8 activation quantisation with the product; the largest shape retains SGLang's separate quantisation. The fused path uses the same power-of-two activation scales, E4M3 rounding and 32-element products as SGLang. Small, deep weights compute independent K32 products in parallel and finish with FP32 FMAs in the original order. Each call uses fresh output and scratch storage, including graph-owned storage during capture; changed inputs, weights and scales remain visible on replay. FP4 experts, attention, routing, cache management and unsupported inputs use SGLang's existing implementation. Dispatch counters record host calls, excluding CUDA graph replay.

Both measured routes use the same scheduler, graphs, attention backend, quantisation and GH200 tile settings. DeepSeek uses Marlin for the FP4 experts. Prefill token choices are ordered within each expert, preserving membership and padding, to make Marlin's BF16 reductions repeatable. Single-token alignment uses SGLang's existing path. The shared FP8 settings tune seven shapes while retaining 32-element scaling blocks, FP32 accumulation and the original reduction order. Tests compare every tuned and native output bit with the upstream tiles, quantised bytes and scales with an independent CPU implementation, and selected outputs with independent FP64 products.

Each measured request starts with CPU token IDs and ends with completed streamed CPU token IDs. Prefill latency is time to the first token; decode throughput counts the remaining 255 tokens of a 256-token generation. Request throughput counts all generated tokens and includes prefill. Timings include scheduling, GPU execution, communication and host-table gathers. Model loading, graph capture, warm-ups, tokenisation and untimed vocabulary downloads are excluded; weights and KV storage remain prepared. Prefix reuse is disabled.

Three fresh processes per route and prompt length run serially with rotated backend order on a quiet exclusive node. The driver hashes benchmark sources, native binaries and their compiled source records, checks checkpoint identities before and after each worker, and enforces a cutoff at least five minutes before the allocation ends. Accuracy checks capture every raw vocabulary logit for 16 predictions in each rank: the original prompt twice and a changed prompt. Repeated captures must match bitwise, rank outputs must agree, changed input must change the vectors, and all greedy generations must match. The paired Llama tolerance remains atol=rtol=0.02; DeepSeek requires bitwise equality.

Use Python 3.11, CUDA 12.9.1 and GCC 14.3.0 for the pinned runtime. This measured CUDA 12.9 profile overrides SGLang's default CUDA 13 dependency metadata: install the Python serving dependencies separately, apply the listed critical pins with `--no-deps` using PyTorch's cu129 index, and rebuild the native extensions. Build SGLang's AOT kernels from the native revision in the runtime settings using Release mode, C++20 and SM90-only architecture flags; the measured targets are `common_ops_sm90_build`, `infllm_ops` and `spatial_ops`, with `ENABLE_BELOW_SM90=OFF` and `SGL_KERNEL_ENABLE_FA3=OFF`. Build FlashMLA with its pinned upstream CMake definition. Rebuild DeepGEMM's `csrc/tvm_ffi_api.cpp` with `tvm_ffi.cpp.build`, C++20, SM90a and the matching installed Python API and headers. All extensions link against this environment's PyTorch and CUDA libraries.

Build CAMBLAS in that environment and register the plugin with an editable install. Keep downloaded sources, models and build outputs under ignored directories. With the checkpoint and source checkout available, run inside the exclusive allocation:

```bash
python -m pip install --no-deps -e .
python scripts/build_cuda.py --torch --architecture sm_90a --cxx g++-14 \
  --cuda-root "$CUDA_HOME" --output build/cuda/sglang-native
python bench/verify_llm_weights.py \
  --model-directory .frameworks/models/Llama-3.1-70B \
  --manifest configs/llama31-70b-weights.json \
  --output build/cuda/llama_verified.json
python bench/compare_llm.py \
  --model-directory .frameworks/models/Llama-3.1-70B \
  --weights-manifest configs/llama31-70b-weights.json \
  --weights-verification build/cuda/llama_verified.json \
  --runtime configs/sglang.json --sglang-source .frameworks/src/sglang-dsv41 \
  --camblas-library build/cuda/sglang-native/libcamblas_cuda.so \
  --prompt-lengths 128 512 --generated-tokens 256 --accuracy-tokens 16 \
  --rounds 3 --warmups 2 --repetitions 3 \
  --gpu-cutoff "$CAMBLAS_GPU_CUTOFF" --output bench/results/llama-sglang
```

For DeepSeek, substitute its model directory, manifest and verification record. Request the allocation's full host memory in the worker step; a smaller inherited task memory limit cannot hold the model and Engram tables. Model loading is recorded separately.

</details>

<details>
<summary>LLM inference results — 5 October 2026</summary>

<!-- sglang-benchmarks -->
Verified on 5 October 2026 on quiet exclusive node `nid010900`, allocation `7072882`, with four NVIDIA GH200 120GB GPUs and the full 288-CPU worker step. These are fresh full-model measurements; the separate CPU and CUDA tables retain their stated measurement dates. Values are medians of three fresh-process medians; brackets give their minimum and maximum. The SGLang reference includes the shared GH200 tuning described above. Speed-up is SGLang latency divided by CAMBLAS latency.

**Llama 3.1 70B: batch 1, 256 generated tokens**

| Prompt tokens | Phase | SGLang ms [range] | CAMBLAS ms [range] | Tokens/sec, SGLang / CAMBLAS | Speed-up |
|---:|---|---:|---:|---:|---:|
| 128 | Prefill / first token | 33.167 [32.346–33.857] | 32.304 [32.290–32.605] | — | 1.027× |
| 128 | Decode | 3256.188 [3252.137–3262.475] | 3256.696 [3245.127–3263.252] | 78.31 / 78.30 | 1.000× |
| 128 | Request | 3288.427 [3285.595–3296.332] | 3289.410 [3277.417–3295.556] | 77.85 / 77.83 | 1.000× |
| 512 | Prefill / first token | 54.820 [54.506–54.988] | 54.659 [54.646–54.911] | — | 1.003× |
| 512 | Decode | 3267.383 [3265.607–3268.524] | 3265.860 [3264.394–3268.081] | 78.04 / 78.08 | 1.000× |
| 512 | Request | 3321.636 [3320.451–3323.720] | 3320.712 [3319.032–3322.405] | 77.07 / 77.09 | 1.000× |

**DeepSeek V4.1 Flash: batch 1, 256 generated tokens**

| Prompt tokens | Phase | SGLang ms [range] | CAMBLAS ms [range] | Tokens/sec, SGLang / CAMBLAS | Speed-up |
|---:|---|---:|---:|---:|---:|
| 128 | Prefill / first token | 193.193 [184.858–194.857] | 185.800 [184.704–187.667] | — | 1.040× |
| 128 | Decode | 3304.535 [3303.413–3307.437] | 1665.198 [1664.422–1666.069] | 77.17 / 153.13 | 1.984× |
| 128 | Request | 3498.165 [3489.393–3500.819] | 1851.905 [1848.855–1852.865] | 73.18 / 138.24 | 1.889× |
| 512 | Prefill / first token | 198.717 [191.661–202.289] | 189.344 [187.367–190.195] | — | 1.050× |
| 512 | Decode | 3313.934 [3313.316–3317.481] | 1672.803 [1669.744–1753.911] | 76.95 / 152.44 | 1.981× |
| 512 | Request | 3513.544 [3509.142–3515.605] | 1860.265 [1859.088–1944.106] | 72.86 / 137.61 | 1.889× |

Both models exceed 50 decode tokens/sec on both routes and prompt lengths; the slowest of the 72 timed requests decoded at 73.78 tokens/sec. All 12 paired comparisons passed with bitwise-identical raw logits and zero maximum absolute error across 296,681,472 vocabulary values. Separate fresh-process captures also matched bitwise on each route. The new native FP8 products and fused activation quantisation apply to DeepSeek; Llama continues to use cuBLASLt for eligible unquantised linear products.

The full native suite ran 103 tests: 100 passed and three existing tests were skipped. The FP8 graph, stream and guard checks also passed against a fresh Torch 2.8 build. Compute Sanitizer reported zero errors or hazards under memcheck, synccheck and racecheck. A generic SM90 build correctly reported the SM90a path unavailable. CPU tests, formatting, Python compilation and whitespace checks passed.

Raw timings, per-rank dispatch counts, complete vocabulary captures, source snapshots and validation receipts are retained in the ignored `build/cuda/sglang_20261004/formal_llama_final/` and `formal_deepseek_final/` directories.

Measured CAMBLAS and key SGLang native SHA-256 identities:

```text
CAMBLAS core    e2f73c1b9c36ed6242039ae83730e030481b7e0f06182a1491d0812fbe17bcf9
CAMBLAS binding 20bb223b3114742e5070bb21e70831d5b09b6bd6188468de857267c73e4c2451
SGLang common   be68b73554a1430f3cd02dea24d56f6633be5eeb5a07ae4d02ddab70bc266a89
FlashMLA        a825ccb62284c83ccb2715bda934c2797f82cf547bd715e920df3b356cb096b3
DeepGEMM        c07dc26bc9bc946c62dd9d95bf193b5869b167acb68060d94568923207ea89b0
```

<!-- sglang-benchmarks end -->

</details>

<details>
<summary>CUDA API and numerical contracts</summary>

### Experimental CAMBLAS CUDA backend

The optional [CUDA backend](src/cuda/backend.cu) operates on GPU memory and interoperates with PyTorch CUDA tensors through [camblas_gpu](camblas_gpu/__init__.py). It provides matrix multiplication, affine transforms, a two-layer ReLU MLP and dense single-head attention in FP32 and FP64. The CPU implementation remains available separately. NVPL is a CPU library; this backend uses cuBLAS/cuBLASLt and independently implemented CUDA packing, recombination, softmax and gradient kernels.

Build from the repository root with a CUDA toolkit and a compatible CUDA PyTorch environment. The default architecture is Hopper `sm_90`; pass `--architecture` for another supported device. `--torch` builds a thin tensor binding against the active PyTorch installation. Without it, the Python adapter uses ctypes with greater host overhead.

```bash
.frameworks/envs/cuda/bin/python scripts/build_cuda.py \
  --cuda-root /path/to/cuda --cxx g++-14 --torch
.frameworks/envs/cuda/bin/python -m unittest discover -s tests -p test_cuda.py -v
python3.11 bench/compare_gpu.py --threads 64 --output bench/results/camblas_gpu
```

From the repository root, use `import camblas_gpu as cb` and `cb.matmul(a, b)`, `cb.affine(x, weight, bias, relu=True)`, `cb.mlp(x, w1, b1, w2, b2)` or `cb.attention(q, k, v)`. Affine, MLP and attention operands must be CUDA FP32/FP64 tensors on the same device. Matrix multiplication also supports BF16 storage with FP32 accumulation. Matrix multiplication accepts transpose views and padded row/column input storage; other irregular layouts are materialised. Supplied `out` tensors must have contiguous row storage and their mutations update PyTorch's version counter. Affine, MLP and attention use contiguous storage. Inference-only FP32/BF16 operations also include `cb.linear(x, weight, bias=None)`, `cb.rms_norm(x, weight)`, `cb.add_rms_norm(x, residual, weight)`, `cb.gated_mlp(x, gate_weight, up_weight, down_weight)` and `cb.qkv_linear(x, q_weight, k_weight, v_weight)`. Linear projection weights have PyTorch’s `[outputs, inputs]` layout. Residual RMS returns both the fresh residual sum and its normalised output. Operations follow the active PyTorch stream; ordering and operand lifetimes across streams follow PyTorch's ordinary CUDA rules. Matrix multiplication and affine support autograd; MLP supports first derivatives, and attention uses differentiable matrix products and PyTorch softmax when gradients are requested.

Automatic dispatch uses guarded Strassen for large even square products, with fused packing and recombination for up to four levels, and classical/cuBLASLt kernels elsewhere. Four levels apply to FP32 squares of size at least 12288 and FP64 squares of size at least 24576, with sizes divisible by 256. Three levels apply to the remaining FP64 squares of size at least 16384 and FP32 squares of size at least 32768, with sizes divisible by 128. It never enables TF32 or reduces the compute dtype. Strassen changes summation order and can worsen relative error when dot products cancel; range checks address overflow, rather than guaranteeing relative accuracy. Request `with cb.algorithm("classical")` to use classical multiplication, including backward. The explicit `lt`, `strassen`, `strassen2`, `strassen3`, `strassen4` and `symmetric` modes aid diagnosis. Recursive modes use fewer levels when dimensions do not divide evenly. Symmetric Gram multiplication is experimental and is excluded from automatic dispatch after measured regressions. Guarded Strassen performs a stream synchronisation for its input-range check. If third-level scratch allocation fails, it tries two levels before falling back to classical GEMM; fourth-level allocation failure falls back to classical GEMM.

Four levels stream seven outer products through a reusable three-level leaf arena. With beta zero, private scratch is `3 × 343 × (N / 16)² × sizeof(dtype)`: 16.08 GiB in FP32 or 32.16 GiB in FP64 at 32768. Non-zero beta needs an additional `N² × sizeof(dtype)` bytes to preserve the existing output until the final alpha/beta update: 20.08 / 40.16 GiB at that size. Three levels require `3 × 343 × (N / 8)² × sizeof(dtype)` bytes: 32.16 GiB for FP64 at 16384 and 72.35 GiB at 24576. These figures exclude tensors and cuBLAS workspace. Scratch remains reusable within its context; growth retains the previous arena until allocation succeeds, so an existing large arena or other live tensors can cause an earlier fallback.

Short FP64 attention has a fused path for width 256, 32 or 64 keys, and up to 1024 query rows divisible by eight. It uses full FP64 matrix instructions and keeps a normalised running weighted result to avoid overflow from an unnormalised sum. It requires an `sm_80` or newer build and enough shared memory; other sizes use matrix products and softmax. Gradient-enabled attention retains the differentiable PyTorch path.

Warm up operations on their intended stream before CUDA graph capture. Guarded Strassen falls back to classical multiplication during capture; warmed FP32 affine operations retain their cuBLASLt bias epilogue. Captured private scratch remains allocated until its context is closed, even after later growth; destroy graphs before closing their contexts. Replay on the capture stream or explicitly serialise replay with other work using that context.

On GPUs reporting coherent pageable memory through host page tables, `cb.matmul_host`, `cb.mlp_host` and `cb.attention_host` provide a separate inference experiment using ordinary CPU tensors and completed CPU output. They synchronise before returning and issue no explicit tensor copies; GPU kernels access CPU storage directly, with GPU scratch for hidden MLP values and attention scores. This can benefit small workloads and slow large GEMMs. It is opt-in, requires the native tensor binding and rejects gradient-enabled inputs. The main comparison always includes explicit copies of every operand and output. Pass `--coherent-host` to `bench/compare_gpu.py` to report this separate CPU-to-CPU experiment alongside the copy pipelines; backward is excluded.

The GPU comparison rotates eager PyTorch, automatic CAMBLAS and a classical CAMBLAS control across at least three fresh processes per case. It includes every input/weight and output/gradient transfer on every transfer-mode call, verifies numerical signatures and scalar-dot oracles, and records source/binary identities, launch counters, affinity and process-median ranges. `cb.stats(all_threads=True)` includes launches from PyTorch autograd workers. Build records and measurements remain under ignored `build/` and `bench/results/` directories.

Pass `--control-library /path/to/saved/libcamblas_cuda.so` to add an unchanged automatic-dispatch control. Save its sibling tensor binding, build record and source snapshots before editing; each process checks the libraries it actually loaded. The saved core and binding use the current common Python adapter. With four backends, twelve rounds cover every ordering position three times. Optional large cases are `square12288`, `square16384`, `square24576` and `square32768`; attention cases include `attention32`, `attention64`, `attention256`, `attention512`, `attention2048`, `attention4096`, `attention8192`, `attention64x1024`, `attention1024x64`, `attention64x32` and `attention1024x32`. The default eight workloads remain the same. `--camblas-algorithm strassen4` forces the candidate policy for a diagnostic sweep; saved controls keep automatic dispatch. Records include device free memory after warm-up and PyTorch allocated/reserved memory; PyTorch's allocator peak excludes the native scratch arena.

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

### Generic inference APIs

`cb.fp8_decode(input, input_scale, weight, weight_scale)` accepts one contiguous E4M3 activation row and E4M3 weights with FP32 block32 scales. `cb.fp8_linear(input, weight, weight_scale)` accepts one finite BF16 row and performs the same dynamic power-of-two block32 activation quantisation internally. Both return fresh BF16 output and preserve ordered FP32 accumulation. These inference-only operations require four-byte-aligned storage, outputs divisible by 16, inner dimensions divisible by 32 in the range 32–8192, and an SM90a build on a compute-capability 9.0 device. Positive FP32 scale strides are supported. `cb.fp8_decode_supported()` reports availability on the current device; the SGLang plugin keeps its original operator when this capability or an input contract does not match. The C interface reports the required scratch bytes through `camblas_cuda_fp8_workspace_size(quantise, outputs, inner)`, where `quantise` is 0 for E4M3 input and 1 for BF16 input. Supply separate, four-byte-aligned scratch on the context's stream and retain it through graph replay. A plain SM90 build retains the other APIs and reports this WGMMA path unavailable.

The generic inference APIs now also include `cb.quantized_matmul(input, input_scale, weight, weight_scale, activation_block=32)`, `cb.swiglu(gate, up, routing=None, limit=0)` and `cb.rms_norm(x, weight, round_before_weight=False)`. Quantised multiplication requires contiguous CUDA FP8 inputs, FP8 or packed FP4 weights, E8M0 scales and an SM90+ binary; it returns fresh BF16 output. SwiGLU computes FP32 intermediates, optionally clips the gate above and up on both sides, applies one FP32 routing value per row, then rounds once to the input storage type. The RMS option multiplies the weight before the final storage rounding; the default retains Llama's existing rounding policy. All use the active stream and support inference only.

`cb.QuantizedGroups(weights, scales)` retains matching quantised weight storages and their device address descriptors, without caching dequantised values or products. Its `matmul(input, input_scale, active, counts)` accepts `[groups, rows, inner]` FP8 input and CUDA int32 selectors/counts. Invalid selectors and padded rows return zero; counts clip to the available rows. In-place weight changes remain visible. Recreate the group when replacing or relocating weight storage, and retain it while captured graphs can replay. `cb.route_groups(experts, active, first_expert=0, rows=...)` produces slot/reverse maps and counts; `cb.reduce_groups(input, experts, reverse)` sums BF16 contributions into FP32 in ascending expert/choice order, including duplicate choices. These primitives use the active stream and caller-supplied shapes rather than model-specific constants.

</details>

<details>
<summary>CPU framework installation</summary>

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

The [build helper](scripts/frameworks.py) pins sources and dependencies and installs separate environments under `.frameworks/envs/<backend>`. All variants disable GPUs, alternative acceleration backends and SVE256 uniformly; OpenMP and CAMBLAS SVE128 remain enabled. Installation is project-local, with absolute dependency paths: rebuild after moving the checkout or dependencies.

</details>

<details>
<summary>Tests and benchmark reproduction</summary>

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

</details>

<details>
<summary>CPU results — 2 October 2026</summary>

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

</details>

<details>
<summary>CPU versus GPU results — 3 October 2026</summary>

<!-- cpu-cuda-results start -->

CPU snapshot, 3 October 2026: 64 attached Grace cores versus one GH200 on `nid011082` (allocation `7032547`), three rotated fresh-process rounds. The GPU pipeline copies every input/weight and every output/gradient on each call. CPU slowdown = CPU time / PyTorch CUDA time including transfers; values above 1 mean the CPU is slower. NVPL is a CPU library.

| Workload | CAMBLAS CPU FP32 slowdown | NVPL CPU FP32 slowdown | CAMBLAS CPU FP64 slowdown | NVPL CPU FP64 slowdown |
|---|---:|---:|---:|---:|
| Square 1024 | 2.04× | 2.07× | 1.58× | 1.62× |
| Square 4096 | 6.53× | 9.03× | 8.66× | 12.31× |
| Square 8192 | 6.75× | 8.90× | 12.50× | 15.24× |
| Transposed GEMM | 2.31× | 2.50× | 3.48× | 6.92× |
| Gram | 2.00× | 3.24× | 3.13× | 4.91× |
| MLP forward | 4.14× | 4.26× | 9.48× | 13.55× |
| Attention | 5.66× | 6.33× | 7.23× | 10.07× |
| Forward + backward | 2.77× | 6.03× | 2.94× | 5.37× |

<details>
<summary>CPU/GPU latency medians and fresh-process ranges (milliseconds)</summary>

Each cell is median [minimum–maximum] of three process medians. Three CPU warm-up calls and ten GPU warm-up calls precede 21 timed CPU calls and 101 timed GPU calls per mode. TF32 is disabled. The validated raw observations, hashes, source snapshots and commands are saved in `bench/results/cpu_cuda_current_20261003/` (ignored by Git).

| Workload | Precision | CAMBLAS CPU ms | NVPL CPU ms | PyTorch CUDA with transfers ms |
|---|---|---:|---:|---:|
| Square 1024 | FP32 | 0.3921 [0.3882–0.3924] | 0.3976 [0.3944–0.4014] | 0.1925 [0.1877–0.1965] |
| Square 4096 | FP32 | 26.4660 [26.4057–26.4747] | 36.5858 [36.3448–36.8611] | 4.0500 [4.0153–4.0741] |
| Square 8192 | FP32 | 182.9684 [182.8445–185.3230] | 241.2854 [226.4402–241.6973] | 27.1112 [26.9708–27.1695] |
| Transposed GEMM | FP32 | 0.4737 [0.4722–0.4829] | 0.5139 [0.5074–0.5182] | 0.2052 [0.2051–0.4908] |
| Gram | FP32 | 0.2977 [0.2953–0.3008] | 0.4815 [0.4808–0.4836] | 0.1488 [0.1477–0.1504] |
| MLP forward | FP32 | 2.4691 [2.4682–2.4801] | 2.5401 [2.5381–2.5516] | 0.5967 [0.5859–0.5971] |
| Attention | FP32 | 1.2563 [1.2063–1.2790] | 1.4050 [1.3987–1.4100] | 0.2220 [0.2152–0.2234] |
| Forward + backward | FP32 | 8.1334 [8.0724–8.1871] | 17.7261 [15.9746–17.9411] | 2.9407 [2.0347–2.9834] |
| Square 1024 | FP64 | 0.7985 [0.7953–0.8006] | 0.8204 [0.8197–0.8227] | 0.5069 [0.5067–0.5139] |
| Square 4096 | FP64 | 55.4432 [55.2364–55.4694] | 78.8126 [77.8894–78.8506] | 6.4016 [6.2307–6.5435] |
| Square 8192 | FP64 | 388.7647 [388.0201–388.8676] | 474.2740 [464.1378–477.8876] | 31.1131 [30.9583–31.1216] |
| Transposed GEMM | FP64 | 0.9550 [0.9399–0.9580] | 1.9008 [1.8844–1.9717] | 0.2746 [0.2681–0.8075] |
| Gram | FP64 | 0.5833 [0.5828–0.5846] | 0.9140 [0.9113–0.9144] | 0.1862 [0.1841–0.1905] |
| MLP forward | FP64 | 6.9389 [6.9161–6.9724] | 9.9143 [9.9122–10.1237] | 0.7318 [0.7192–0.7614] |
| Attention | FP64 | 1.6807 [1.6408–1.7297] | 2.3404 [2.3288–2.3549] | 0.2324 [0.2299–0.2413] |
| Forward + backward | FP64 | 16.7728 [16.7261–16.7911] | 30.6128 [29.9310–30.9346] | 5.7017 [5.7005–5.7037] |

</details>

<!-- cpu-cuda-results end -->

</details>

<details>
<summary>CUDA results and validation</summary>

<!-- cuda-results start -->

#### CUDA performance snapshot

Allocation `7032547`, `nid011082`: one GH200 with about 95.6 GiB exposed memory, 64 attached Grace host cores (CPUs 0–63), PyTorch 2.8.0+cu129, NumPy 2.3.2, CUDA toolkit 12.9.1 (nvcc 12.9.86), GCC 14.3.0 and `sm_90`. The main suite uses 4 rotated fresh processes per backend/case. Eager PyTorch, automatic CAMBLAS, its classical control and unchanged main (`0ddcdf9461628594fc7be0049396d42beee5e158`) use matching inputs and full FP32/FP64 with TF32 disabled.

Resident latency includes allocation, dispatch and synchronous completion. Transfer latency additionally copies every input/weight to the GPU and every output/parameter gradient back to pageable CPU memory on every call. Context creation and tuning occur during warm-up. Speed-up = PyTorch / CAMBLAS; values above 1 favour CAMBLAS.

Square GEMMs use the stated N. Transposed GEMM multiplies (512×1024)ᵀ by (512×2048); Gram uses X=(4096×512). MLP uses batch 512 and widths 2048→4096→1024, with ReLU after the first affine. Attention uses 1024 query/key/value rows of width 256 and scale 1/16. Forward + backward uses the same MLP and returns gradients for both weights and biases. All comparisons use seed 20260906.

| Workload | FP32 resident | FP32 incl. transfers | FP64 resident | FP64 incl. transfers |
|---|---:|---:|---:|---:|
| Square 1024 | 1.036× | 0.998× | 1.024× | 0.988× |
| Square 4096 | 1.077× | 1.037× | 1.001× | 1.004× |
| Square 8192 | 1.269× | 1.220× | 1.232× | 1.115× |
| Transposed GEMM | 0.999× | 0.999× | 0.977× | 1.007× |
| Gram | 0.970× | 0.953× | 1.019× | 0.992× |
| MLP forward | 1.020× | 1.008× | 1.039× | 1.016× |
| Attention | 1.514× | 1.218× | 1.762× | 1.268× |
| Forward + backward | 1.041× | 1.031× | 1.061× | 1.010× |

<details>
<summary>Latency medians and fresh-process ranges (milliseconds)</summary>

| Workload | Precision | PyTorch resident ms | CAMBLAS resident ms | PyTorch incl. transfers ms | CAMBLAS incl. transfers ms |
|---|---|---:|---:|---:|---:|
| Square 1024 | FP32 | 0.0856 [0.0835–0.0860] | 0.0827 [0.0810–0.0847] | 0.1905 [0.1894–0.1950] | 0.1909 [0.1901–0.1978] |
| Square 1024 | FP64 | 0.0722 [0.0719–0.0731] | 0.0706 [0.0705–0.0709] | 0.5095 [0.5075–0.5112] | 0.5156 [0.5080–0.5209] |
| Square 4096 | FP32 | 2.8773 [2.8687–2.8932] | 2.6720 [2.6527–2.6784] | 4.0486 [4.0354–4.0664] | 3.9033 [3.8859–3.9224] |
| Square 4096 | FP64 | 3.2050 [3.1599–3.2099] | 3.2005 [3.1931–3.2120] | 6.5668 [6.4244–6.8133] | 6.5386 [6.4943–6.9717] |
| Square 8192 | FP32 | 23.3512 [23.3287–23.3733] | 18.4074 [18.3576–18.4480] | 26.9114 [26.8584–26.9437] | 22.0619 [22.0086–22.1226] |
| Square 8192 | FP64 | 24.6810 [24.6391–24.7404] | 20.0305 [20.0146–20.0347] | 30.9216 [30.7223–31.0159] | 27.7364 [27.5193–28.3872] |
| Transposed GEMM | FP32 | 0.0846 [0.0829–0.0851] | 0.0847 [0.0845–0.0848] | 0.3517 [0.2038–0.4952] | 0.3518 [0.2122–0.4950] |
| Transposed GEMM | FP64 | 0.0811 [0.0800–0.0830] | 0.0830 [0.0809–0.0851] | 0.8199 [0.2672–0.8206] | 0.8144 [0.2788–0.8195] |
| Gram | FP32 | 0.0811 [0.0793–0.0813] | 0.0836 [0.0831–0.0847] | 0.1482 [0.1471–0.1518] | 0.1555 [0.1512–0.1601] |
| Gram | FP64 | 0.0871 [0.0843–0.0875] | 0.0854 [0.0847–0.0863] | 0.1895 [0.1812–0.1911] | 0.1910 [0.1860–0.1925] |
| MLP forward | FP32 | 0.3289 [0.3277–0.3314] | 0.3226 [0.3203–0.3277] | 0.5928 [0.5858–0.5939] | 0.5883 [0.5838–0.5938] |
| MLP forward | FP64 | 0.3152 [0.3120–0.3209] | 0.3032 [0.3011–0.3218] | 0.7306 [0.7245–0.7398] | 0.7194 [0.7159–0.7246] |
| Attention | FP32 | 0.1185 [0.1130–0.1251] | 0.0783 [0.0772–0.0794] | 0.2098 [0.2066–0.2243] | 0.1722 [0.1689–0.1775] |
| Attention | FP64 | 0.1260 [0.1137–0.1292] | 0.0715 [0.0704–0.0730] | 0.2391 [0.2271–0.2458] | 0.1885 [0.1870–0.1967] |
| Forward + backward | FP32 | 0.8058 [0.7976–0.8093] | 0.7740 [0.7718–0.7772] | 3.0069 [2.9766–3.0237] | 2.9152 [2.8603–2.9649] |
| Forward + backward | FP64 | 0.8296 [0.8038–0.8537] | 0.7823 [0.7622–0.8057] | 5.7566 [5.7442–5.7674] | 5.6969 [5.6677–5.7223] |

</details>

Against unchanged main, the following ratios are main / candidate. Values below 1 retain measured regressions; differences near 1 can fall within the process ranges.

| Workload | FP32 resident | FP32 incl. transfers | FP64 resident | FP64 incl. transfers |
|---|---:|---:|---:|---:|
| Square 1024 | 1.025× | 1.023× | 1.012× | 0.998× |
| Square 4096 | 1.000× | 0.998× | 1.007× | 0.999× |
| Square 8192 | 0.998× | 0.997× | 1.001× | 1.022× |
| Transposed GEMM | 0.991× | 0.997× | 0.965× | 1.001× |
| Gram | 1.008× | 1.003× | 0.983× | 0.978× |
| MLP forward | 1.034× | 1.017× | 0.986× | 1.007× |
| Attention | 1.026× | 1.048× | 1.026× | 1.004× |
| Forward + backward | 1.009× | 1.017× | 0.996× | 1.001× |

Additional GEMM/attention shapes use the same final binary and transfer contract, with three rotated fresh processes each for PyTorch, automatic CAMBLAS and unchanged main. Each process uses ten warm-up calls, then five timed calls per mode for large GEMMs or 101 for short attention. Attention dimensions state query/key rows; depth and value width remain 256.

| Workload | Precision | PyTorch / CAMBLAS resident | PyTorch / CAMBLAS incl. transfers | Main / candidate resident | Main / candidate incl. transfers |
|---|---|---:|---:|---:|---:|
| Square 12288 | FP32 | 1.420× | 1.197× | 1.073× | 1.033× |
| Square 12288 | FP64 | 1.180× | 1.032× | 1.005× | 1.006× |
| Square 16384 | FP32 | 1.445× | 1.123× | 1.103× | 1.026× |
| Square 16384 | FP64 | 1.331× | 1.041× | 1.078× | 1.008× |
| Square 24576 | FP32 | 1.626× | 1.113× | 1.241× | 1.063× |
| Square 24576 | FP64 | 1.440× | 1.390× | 1.224× | 1.198× |
| Square 32768 | FP32 | 1.462× | 1.078× | 1.187× | 0.384× |
| Square 32768 | FP64 | 1.513× | 0.292× | 1.729× | 0.338× |
| Attention 32 | FP32 | 2.007× | 1.287× | 1.016× | 0.996× |
| Attention 32 | FP64 | 5.584× | 1.954× | 2.749× | 1.472× |
| Attention 64 | FP32 | 1.802× | 1.247× | 1.053× | 1.060× |
| Attention 64 | FP64 | 4.893× | 2.024× | 2.234× | 1.479× |
| Attention 1024 queries, 32 keys | FP32 | 2.207× | 1.328× | 1.063× | 1.039× |
| Attention 1024 queries, 32 keys | FP64 | 5.145× | 1.713× | 2.419× | 1.286× |
| Attention 1024 queries, 64 keys | FP32 | 2.308× | 1.387× | 0.944× | 0.987× |
| Attention 1024 queries, 64 keys | FP64 | 4.563× | 1.777× | 2.048× | 1.419× |

<details>
<summary>Additional-shape latency ranges (milliseconds)</summary>

| Workload | Precision | PyTorch resident ms | CAMBLAS resident ms | PyTorch incl. transfers ms | CAMBLAS incl. transfers ms |
|---|---|---:|---:|---:|---:|
| Square 12288 | FP32 | 81.7133 [80.8903–81.9142] | 57.5524 [57.5212–57.9884] | 126.1722 [125.8765–126.4090] | 105.4480 [105.1658–105.6252] |
| Square 12288 | FP64 | 85.2143 [84.5264–85.2152] | 72.1931 [71.2920–72.4354] | 503.5149 [494.2677–507.0770] | 487.7890 [487.5449–489.6183] |
| Square 16384 | FP32 | 184.8577 [183.3851–185.7163] | 127.9715 [127.7905–128.3844] | 519.7224 [518.6400–520.6428] | 462.7853 [461.7597–465.5710] |
| Square 16384 | FP64 | 200.7538 [199.7742–201.2421] | 150.8343 [150.7065–151.3965] | 1174.6837 [224.2363–1179.3176] | 1127.8911 [178.1891–1130.1255] |
| Square 24576 | FP32 | 663.1273 [662.0924–663.3652] | 407.8594 [407.5427–408.6444] | 1718.7120 [689.0506–1800.3284] | 1543.5654 [1528.3481–1592.1137] |
| Square 24576 | FP64 | 685.1774 [683.0202–687.9574] | 475.9200 [474.8487–477.2642] | 738.3586 [737.1997–3304.8102] | 531.3256 [530.1224–532.9532] |
| Square 32768 | FP32 | 1483.5475 [1482.1543–1483.7682] | 1015.0131 [1014.0896–1016.9451] | 3533.4816 [1531.7947–3734.1788] | 3276.9776 [3255.5431–3277.9683] |
| Square 32768 | FP64 | 1631.5309 [1605.6222–1633.3676] | 1078.6756 [1078.6518–1079.0143] | 1731.4479 [1729.3027–1733.5635] | 5931.0586 [1189.1844–5956.9284] |
| Attention 32 | FP32 | 0.1043 [0.1029–0.1062] | 0.0520 [0.0509–0.0535] | 0.1859 [0.1780–0.1870] | 0.1444 [0.1424–0.1497] |
| Attention 32 | FP64 | 0.1172 [0.1156–0.1265] | 0.0210 [0.0203–0.0229] | 0.2157 [0.2036–0.2184] | 0.1104 [0.1027–0.1299] |
| Attention 64 | FP32 | 0.0962 [0.0949–0.1075] | 0.0534 [0.0528–0.0544] | 0.1920 [0.1836–0.2002] | 0.1540 [0.1533–0.1579] |
| Attention 64 | FP64 | 0.1290 [0.1148–0.1354] | 0.0264 [0.0256–0.0268] | 0.2087 [0.1886–0.2101] | 0.1031 [0.0953–0.1045] |
| Attention 1024 queries, 32 keys | FP32 | 0.1083 [0.1080–0.1159] | 0.0491 [0.0461–0.0499] | 0.1964 [0.1926–0.2091] | 0.1478 [0.1472–0.1480] |
| Attention 1024 queries, 32 keys | FP64 | 0.1187 [0.1178–0.1284] | 0.0231 [0.0219–0.0247] | 0.2305 [0.2299–0.2343] | 0.1346 [0.1332–0.1415] |
| Attention 1024 queries, 64 keys | FP32 | 0.1203 [0.1162–0.1234] | 0.0521 [0.0461–0.0526] | 0.2214 [0.2111–0.2265] | 0.1596 [0.1561–0.1612] |
| Attention 1024 queries, 64 keys | FP64 | 0.1257 [0.1235–0.1259] | 0.0276 [0.0273–0.0285] | 0.2161 [0.2130–0.2274] | 0.1216 [0.1177–0.1341] |

</details>

The FP64 32768² transfer case regressed: 0.292× versus PyTorch and 0.338× versus unchanged main. Its candidate process medians ranged from 1.189 to 5.957 seconds; two processes alternated fast and slow transfer samples despite stable resident compute times. These observations remain in the aggregate.

A separate one-process diagnostic isolated the large FP64 stall to downloading into newly allocated pageable CPU output: median 4831.0 ms for that stage, versus 53.8 ms with a warmed, reusable CPU output buffer. Input uploads and GPU computation were timed separately, and previous-output retirement was much shorter than the stall. The buffer-reuse contract excludes CPU output allocation and therefore does not replace the aggregate above. The underlying driver/first-touch cause remains unresolved; staged samples and a trace are saved in `build/cuda/large_transfer_diagnosis_20261004.json` and its companion trace.

FP32 32768² also regressed versus unchanged main with transfers (0.384×), while remaining 1.078× versus PyTorch. Fresh-process ranges above show the substantial variation in this large-output transfer workload.

Dedicated and compiled PyTorch baselines retain full FP32/FP64. Compilation, autotuning and one-time CPU weight-layout preparation are excluded; graph replay and input staging remain timed. Cells are CAMBLAS speed-up over that baseline, resident / including transfers.

| Workload | Precision | Fused | Compiled fused |
|---|---|---:|---:|
| MLP forward | FP32 | 1.260× / 1.144× | 1.491× / 1.253× |
| MLP forward | FP64 | 1.706× / 1.285× | 1.467× / 1.223× |
| Attention | FP32 | 2.368× / 1.695× | 3.037× / 1.892× |
| Attention | FP64 | 2.762× / 1.718× | 1.845× / 1.323× |
| Forward + backward | FP32 | 1.152× / 1.014× | 1.211× / 1.102× |
| Forward + backward | FP64 | 1.345× / 1.049× | 1.208× / 1.021× |

Transfer pipelines can vary with CPU-output allocation and process state. The earlier neural study recorded an FP32 backward transfer regression of 0.694×; unchanged main was similarly affected. The tables and raw rounds retain process ranges and measured regressions. Strassen rounding can increase relative error under cancellation; it does not guarantee accuracy for every input or a speed-up for every shape.

The four-level arena allows FP64 32768² multiplication to avoid the low-memory classical fallback observed with unchanged main on this GPU. Allocator peaks reported by PyTorch exclude native scratch. The analytical arena sizes and beta-dependent extra storage are documented above.

Raw records, minimum/maximum process medians, source snapshots, actual loaded-library hashes, commands, counters and oracles are retained in the ignored result directories: `bench/results/gpu_allocation_final_20261004/`, `bench/results/gpu_large_allocation_final_20261004/`, `bench/results/gpu_short_allocation_final_20261004/`, `bench/results/gpu_fused_allocation_final_20261004/`, `bench/results/gpu_compiled_fused_allocation_final_20261004/`.

```bash
python3.11 bench/compare_gpu.py --threads 64 --rounds 4 \
  --repetitions 101 --control-library /path/to/saved/libcamblas_cuda.so \
  --output bench/results/gpu_allocation_final
CAMBLAS_TEST_LARGE_CUDA=1 .frameworks/envs/cuda/bin/python \
  -m unittest tests.test_cuda.CudaLargeTests -v
```

<!-- cuda-results end -->

<!-- final-verification start -->

#### Final verification

Final checks ran 2026-10-04T00:39:16.016380+00:00–2026-10-04T00:42:43.341902+00:00 UTC before the user’s four-hour deadline on allocation `7032547`. CPU `make test` passed. Native Python tests passed 52/52 (0 skips); ctypes CUDA tests passed 40/43 (3 native-only skips). The optional 32768² analytic check compared every output element in both precisions and recomputed changed inputs.

Compute Sanitizer preflight completed at 2026-10-03T23:53:24.734031+00:00 with zero CAMBLAS-kernel errors under memcheck and zero errors in the three graph tests under synccheck. The LP64 bias-gradient regression ran separately from instrumentation. Formatting, Python compilation and whitespace checks passed. Final source and binary hashes matched the broader measurements and final-window run.

The final-window comparison uses 3 rotated fresh processes per backend/case, ten warm-up calls and five timed calls per mode. Values below are PyTorch / CAMBLAS.

| Workload | Precision | Resident speed-up | Incl. transfers speed-up |
|---|---|---:|---:|
| Square 4096 | FP32 | 1.068× | 1.043× |
| Square 4096 | FP64 | 0.982× | 1.005× |
| Attention 64 | FP32 | 1.953× | 1.277× |
| Attention 64 | FP64 | 4.690× | 1.902× |

Core and tensor-binding SHA-256:

```text
core 1f1a86215af9c899303d9bfb589e6a3fb047ff63604eb524f0bc021a529cb48e
binding 4ae26e933336a245065175cc878bca2937ec2852d4ba432d2ffb2a72a2d6e44b
```

<!-- final-verification end -->

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and [LICENSE](LICENSE) for the MIT licence, copyright Pritthijit Nath. External dependencies retain their own licences; generated artefacts and internal documents are excluded from this repository.

</details>

<details>
<summary>Earlier FP64 transfer results</summary>

### Earlier FP64 transfer measurements

The FP64 transfer table uses fresh logical pinned CPU output allocations inside every timed call, with the same policy for PyTorch and CAMBLAS. All inputs/weights and outputs/parameter gradients are copied on every call; previous-result retirement is timed. PyTorch’s warmed pinned allocator may recycle freed physical storage. Small-case inference transfer operations use one native dispatch; backward retains its gradient path.

| FP64 workload | Resident speed-up | PyTorch transfer ms [range] | CAMBLAS transfer ms [range] | Transfer speed-up |
|---|---:|---:|---:|---:|
| square1024 | 1.034× | 0.235 [0.232–0.239] | 0.231 [0.222–0.235] | 1.017× |
| square4096 | 0.964× | 4.212 [4.122–4.229] | 4.215 [4.095–4.247] | 0.999× |
| square8192 | 1.230× | 29.65 [29.38–29.68] | 25.29 [24.91–25.45] | 1.172× |
| transpose | 0.950× | 0.275 [0.275–0.280] | 0.285 [0.279–0.289] | 0.967× |
| gram | 1.006× | 0.195 [0.194–0.195] | 0.194 [0.193–0.196] | 1.006× |
| mlp | 1.025× | 0.739 [0.724–0.742] | 0.720 [0.716–0.722] | 1.027× |
| attention | 1.914× | 0.259 [0.248–0.261] | 0.185 [0.182–0.193] | 1.398× |
| backward | 1.077× | 1.862 [1.831–1.886] | 1.802 [1.796–1.815] | 1.033× |
| square32768 | — | 1883.02 [1869.34–1900.70] | 1322.34 [1318.50–1325.16] | 1.424× |

The 32768² case uploads 16 GiB and downloads 8 GiB on every call. Every output element is checked against an independent analytic reference before and after input changes; 64 random entries are also checked with CPU FP64. The earlier pageable-output regression remains above because it uses a different allocation policy. Near-1× rows with overlapping process ranges do not establish a reliable win.

<!-- fp64-direct-followup -->
### FP64 direct-transfer follow-up

Three fresh rotated processes compare PyTorch, unchanged main and a direct-copy candidate with identical inputs and fresh logical pinned CPU output allocations inside every timed call. Each process uses ten warm-ups and 21 timed calls, 64 threads on CPU cores 0–63, and the native transfer entry for both CAMBLAS builds. Dense CPU storage, including transposes, uploads directly on the active stream; irregular views retain the normal copy path. The result downloads directly and the stream finishes before CPU output returns, including empty results and exception paths. No operand values are cached.

| FP64 workload | Resident speed-up | PyTorch transfer ms [range] | CAMBLAS transfer ms [range] | Transfer speed-up | Main / candidate transfer |
|---|---:|---:|---:|---:|---:|
| transpose | 0.949× | 0.276 [0.275–0.278] | 0.273 [0.271–0.284] | 1.012× | 1.050× |
| square4096 | 1.109× | 4.108 [3.982–4.118] | 4.143 [3.995–4.187] | 0.991× | 1.004× |
| gram | 0.940× | 0.194 [0.193–0.194] | 0.192 [0.178–0.194] | 1.011× | 1.011× |
| mlp | 1.038× | 0.731 [0.726–0.732] | 0.679 [0.670–0.694] | 1.076× | 1.050× |
| attention | 1.746× | 0.245 [0.244–0.248] | 0.173 [0.169–0.174] | 1.416× | 1.106× |
| square1024 | 0.996× | 0.236 [0.236–0.237] | 0.217 [0.212–0.222] | 1.089× | 1.046× |
| square8192 | 1.246× | 29.590 [29.544–29.714] | 25.039 [24.554–25.056] | 1.182× | 0.992× |
| backward | 1.041× | 1.905 [1.834–1.913] | 1.806 [1.792–1.817] | 1.055× | 0.994× |

Seven of eight transfer medians exceed 1×; 4096² GEMM remains slightly slower. Overlapping ranges for transposed GEMM and Gram do not establish reliable wins. Clear gains against unchanged main occur for 1024² GEMM, MLP and attention. A separate 4096² Strassen trial improved resident compute but regressed transfer latency and was rejected.

These timings use the preceding direct-copy core `bb7f3a3a7561c4a9668424ee2d86fc1ab43835c98394ab4b4d0138ab897006be`. The final core adds grouped quantised inference and removes the unreachable non-contiguous transfer-output branch; these eight rows were not remeasured with that final binary. The earlier 32768² three-process table above uses its original Python transfer entry and is retained with that contract.
<!-- fp64-direct-followup end -->

</details>

<details>
<summary>CPU implementation and compatibility</summary>

Unsupported BLAS/LAPACK operations use an explicit OpenBLAS compatibility dependency. The binary interface remains unstable, so use matching headers and libraries. The framework adapter serialises concurrent GEMMs and requires spawn/exec after initialisation rather than fork; fast-mode results are not guaranteed to match vendor libraries bitwise.

The Grace implementation combines SVE128 and NEON kernels, packed operand panels and shape-dependent task grids. Bounded square, transposed and rectangular products use one-level Strassen, with all seven operand transforms formed during packing. Deep products stream through small depth panels and accumulate private products before writing C. Eligible deep FP32 rectangles at 16/32 workers use a twelve-row, eight-column NEON kernel, with product/tile assignments rotated to spread partial micro-panels across workers. FP64 packing forms the seven transforms directly from the four original quadrants. Eligible short transposed-A products use this fused packing at 16/32 threads and in NumPy at 64 threads. Short transposed-A FP32 products use a parallel four-by-four transpose before the guarded packed route. Every call repacks its inputs. A finite-range check during packing preserves a classical fallback, including when an unsafe value appears in a later depth panel.

Selected FP32 products at 64 threads use an eight-row, twelve-column NEON kernel and shared packed operands. Short products use the private worker pool. Aligned short NN squares use eight row groups and eight column groups; other short products retain sixteen row groups and four column groups. Eligible deep products stream through bounded panels with row and column group barriers, using either the private pool or the application's existing OpenMP executor. Deep FP32 Strassen and the streaming grids allocate only their live scratch and release it after the synchronous call; other paths retain allocation capacity. Bounded 64-worker FP64 squares use a wider depth panel to reduce partial-product traffic. Aligned NN squares from 512 through 1024 use a six-row, eight-column NEON kernel at 16 through 64 workers. Eligible transposed-B FP64 rectangles use the same NEON kernel at 64 workers, with the existing packed operand layout. Workers in the private pool poll for up to 256 microseconds before sleeping on a futex; a timeout without new work leaves the previous task untouched. Runtime capability, matrix bounds and workspace checks retain classical fallbacks. Strassen changes rounding behaviour; its range check bounds overflow growth, not general relative error.

</details>
