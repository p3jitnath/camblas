# Experimental results

Four GH200 GPUs on a quiet exclusive node; batch one, 128 prompt tokens and 256 greedy output tokens. Rates use median decode latency.

| Model | SGLang tokens/s | CAMBLAS tokens/s | Gain | Evidence |
|---|---:|---:|---:|---|
| Llama 3.1 70B | 78.31 | 78.30 | 0.0% | Historical |
| DeepSeek V4.1 Flash | 145.08 | 153.63 | 5.9% | Single-process pilots |
| GLM 5.3 Flash | 152.14 | 161.30 | 6.02% | Three fresh pairs |

GLM shares the tuned SGLang MoE tiles between backends. All 89.2 million measured logits matched bitwise, including repeated and changed inputs. Original FP8 weights/inputs, BF16 activation/KV storage and FP32 accumulation remain unchanged.

DeepSeek retains FP4 experts, FP8 dense weights/KV and FP32 accumulation; 24.8 million measured logits matched bitwise. Llama retains BF16 and showed no meaningful gain.

The [runtime](configs/sglang.json) pins reproduction. [Archived results](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md), including Llama, precede the PyTorch interface change.

CPU: CAMBLAS won all 60 Grace FP32/FP64 cases against OpenBLAS and NVPL at 16/64 cores ([report](bench/reports/grace_20261002.json)). CUDA: transfer-inclusive 32,768-square FP64 reached 0.292× against PyTorch and 0.338× against unchanged CAMBLAS.
