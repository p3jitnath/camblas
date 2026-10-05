# Experimental results

Historical measurements at [ab838f2](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md), before the PyTorch interface change. Speed-up is reference latency divided by CAMBLAS latency.

## LLM inference — 5 October 2026

Four GH200 GPUs on a quiet exclusive node; batch one, 256 greedy tokens, prompts of 128/512 tokens. Each value is the median of three fresh-process medians.

| Model | Prompt | SGLang tokens/sec | CAMBLAS tokens/sec | Decode speed-up |
|---|---:|---:|---:|---:|
| Llama 3.1 70B | 128 | 78.31 | 78.30 | 1.000× |
| Llama 3.1 70B | 512 | 78.04 | 78.08 | 1.000× |
| DeepSeek V4.1 Flash | 128 | 77.17 | 153.13 | 1.984× |
| DeepSeek V4.1 Flash | 512 | 76.95 | 152.44 | 1.981× |

DeepSeek's request speed-up was 1.889×. Its slower 512-token process, 145.39 tokens/sec, remains included. Logits and greedy outputs matched bitwise.

Llama retains BF16. DeepSeek retains FP4 experts, FP8 dense weights/KV and FP32 accumulation. The [runtime settings](configs/sglang.json) and archived measurements retain the protocol and identities.

## CPU and CUDA — 2–3 October 2026

CAMBLAS beat OpenBLAS and NVPL in all 60 FP32/FP64 CPU cases at 16/64 Grace cores; see the [CPU report](bench/reports/grace_20261002.json).

The 32,768-square FP64 CUDA case, including transfers, reached 0.292× against PyTorch and 0.338× against unchanged CAMBLAS.
