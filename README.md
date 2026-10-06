# CAMBLAS

CAMBLAS provides CPU matrix multiplication for NVIDIA Grace and an optional CUDA backend for PyTorch and LLM inference. CPU products support FP32 and FP64; CUDA products also support BF16 storage with FP32 accumulation. The project is experimental, so use matching headers and libraries.

## Install

Clone the repository, then select a CPU or CUDA build:

```bash
git clone git@github.com:p3jitnath/camblas.git
cd camblas
```

For CPU, use Linux, Python 3.11+, Make and a C11 compiler. Build with `make reference CC=cc PYTHON=python3.11`, or use `make grace CC=gcc-14 PYTHON=python3.11` on Grace with SVE. The [headers](include/) define the native interface; CPU framework adapters also require LP64 OpenBLAS.

For CUDA, install a compatible CUDA toolkit and PyTorch, then build in that Python environment:

```bash
python -m pip install --no-deps -e .
python scripts/build_cuda.py --torch --architecture sm_90a --cxx g++-14 \
  --cuda-root "$CUDA_HOME"
```

Use `sm_90a` for Hopper inference kernels, or select the architecture for your GPU. Rebuild the tensor binding when the PyTorch or CUDA ABI changes.

## Use

Enable the backend before starting Python:

```bash
CAMBLAS_ENABLE=1 python your_model.py
```

Keep ordinary PyTorch code, including `torch.matmul`, `torch.mm`, `torch.addmm` and `torch.nn.functional.linear`:

```python
import torch

x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(4096, 2048, device="cuda", dtype=x.dtype)
y = torch.nn.functional.linear(x, weight)
```

PyTorch loads CAMBLAS through backend discovery. Eligible CUDA products use CAMBLAS; unsupported types, layouts and requested TF32 use PyTorch's existing kernels. Autograd, tensor arguments and return values retain the PyTorch interface. Operations use the active stream; warm that stream before CUDA graph capture.

For SGLang, enable the plugin when you launch the server:

```bash
CAMBLAS_ENABLE=1 SGLANG_PLUGINS=camblas CAMBLAS_SGLANG_OPS=linear,fp8 \
  python -m sglang.launch_server --model-path /path/to/model
```

See [contribution and validation guidance](CONTRIBUTING.md) and the [MIT licence](LICENSE).

<details>
<summary>Results</summary>

- [LLM inference](results/llm.md)
- [CPU matrix multiplication](results/cpu.md)
- [GPU matrix multiplication and attention](results/gpu.md)

</details>

<details>
<summary>Runtime settings and reproduction</summary>

The [SGLang runtime](configs/sglang.json) and [requirements](configs/sglang-requirements.txt) pin the measured GH200 environment. Enable its plugin with `SGLANG_PLUGINS=camblas` and `CAMBLAS_SGLANG_OPS=linear,fp8`; set `CAMBLAS_CUDA_LIBRARY` when the native library is outside `build/cuda/`. The plugin uses standard PyTorch linear calls and private GPU kernels for the checkpoint's existing FP8 operations.

Verify the original checkpoint against the [Llama manifest](configs/llama31-70b-weights.json), [DeepSeek manifest](configs/deepseek-v41-flash-weights.json) or [GLM manifest](configs/glm53-flash-weights.json), then compare the same pinned engine with and without CAMBLAS:

```bash
python bench/verify_llm_weights.py \
  --model-directory .frameworks/models/Llama-3.1-70B \
  --manifest configs/llama31-70b-weights.json --output build/cuda/llama_verified.json
python bench/compare_llm.py \
  --model-directory .frameworks/models/Llama-3.1-70B \
  --weights-manifest configs/llama31-70b-weights.json \
  --weights-verification build/cuda/llama_verified.json \
  --runtime configs/sglang.json --sglang-source .frameworks/src/sglang-dsv41 \
  --camblas-library build/cuda/libcamblas_cuda.so \
  --prompt-lengths 128 512 --generated-tokens 256 --accuracy-tokens 16 \
  --rounds 3 --warmups 2 --repetitions 3 \
  --gpu-cutoff "$CAMBLAS_GPU_CUTOFF" --output bench/results/llama-sglang
```

Run measurements on quiet exclusive nodes, with a cutoff at least five minutes before the allocation ends. Request the full host memory for DeepSeek's Engram tables, and rebuild SGLang's native extensions against the pinned PyTorch/CUDA environment. Both routes must use the same precision, hardware, graphs and shared tuning. For GLM, use its manifest and model directory with a 128-token prompt; the pinned profile selects text-only inference and BF16 KV storage.

The CUDA backend defaults to cuBLASLt. `CAMBLAS_CUDA_ALGORITHM=classical` selects classical cuBLAS; `auto` can select guarded Strassen for large products. Strassen changes rounding, and its finite-range guard bounds overflow growth rather than relative error. Keep full precision for comparisons by disabling TF32 and BF16 reduced-precision reduction.

`make test CC=gcc-14 PYTHON=python3.11` validates the CPU library. CUDA tests require a compatible built backend and a CUDA-enabled PyTorch environment. Benchmark drivers record source and library hashes, dispatch evidence, fresh-process ranges and correctness checks; keep their generated outputs under ignored directories.

</details>
