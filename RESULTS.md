# Experimental results

Historical benchmarks at [ab838f2](https://github.com/p3jitnath/camblas/tree/ab838f2); these tables precede the PyTorch interface change. Speed-up is reference latency divided by CAMBLAS latency.

## LLM inference — 5 October 2026

Four GH200 120GB GPUs on quiet exclusive node `nid010900`; batch one, original checkpoints, 256 greedy tokens (255 decode tokens). Medians of three fresh-process medians, with rotated backend order.

| Model | Prompt tokens | SGLang tokens/sec | CAMBLAS tokens/sec | Decode speed-up |
|---|---:|---:|---:|---:|
| Llama 3.1 70B | 128 | 78.31 | 78.30 | 1.000× |
| Llama 3.1 70B | 512 | 78.04 | 78.08 | 1.000× |
| DeepSeek V4.1 Flash | 128 | 77.17 | 153.13 | 1.984× |
| DeepSeek V4.1 Flash | 512 | 76.95 | 152.44 | 1.981× |

DeepSeek's complete-request speed-up was 1.889× at both prompts. The slower 512-token process, 145.39 tokens/sec, remains included. Llama's decode improvement was negligible.

Llama retains BF16. DeepSeek retains checkpoint FP4 experts, FP8 dense weights, FP32 accumulation and FP8 KV storage. Timings include scheduling, communication and host-table gathers; they exclude loading, graph capture, warm-ups and tokenisation. Full-vocabulary logits and greedy outputs matched bitwise. See the [runtime settings](configs/sglang.json).

## CPU and CUDA — 2–3 October 2026

CAMBLAS beat OpenBLAS and NVPL in all 60 FP32/FP64 NumPy/PyTorch cases at 16/64 Grace cores. Differences near 1× can vary. The [CPU report](bench/reports/grace_20261002.json) retains reproduction evidence.

CUDA gains depend on transfers: the 32,768-square FP64 case, including every input/output transfer, reached 0.292× against PyTorch and 0.338× against unchanged CAMBLAS.

The [archived tables](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md) retain all results, process ranges, protocols and binary identities.
