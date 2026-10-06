# Experimental results

Quiet exclusive GH200 nodes; batch one and 256 greedy output tokens. Rates use median decode latency across fresh processes.

| Model | GPUs | Prompt | SGLang tokens/s | CAMBLAS tokens/s | Gain | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Llama 3.1 70B | 4 | 128 | 78.31 | 78.30 | 0.0% | Historical |
| DeepSeek V4.1 Flash | 4 | 128 | 156.82 | 138.96 | −11.4% | Three fresh pairs; tuned native SGLang |
| GLM 5.3 Flash | 4 | 128 | 152.14 | 161.30 | 6.02% | Three fresh pairs |
| MiMo V2.6 Flash RL | 4 | 128 | 179.60 | 181.12 | 0.85% | Three fresh pairs; FP8 plugin only |
| MiMo V2.6 Pro RL | 8 | 128 | 73.59 | 73.96 | 0.50% | Three fresh pairs; FP8 plugin only |
| MiMo V2.6 Pro RL | 8 | 8192 | 64.28 | 64.50 | 0.34% | Three fresh pairs; FP8 plugin only |

GLM shares the tuned SGLang MoE tiles between backends. All 89.2 million measured logits matched bitwise, including repeated and changed inputs. Original FP8 weights/inputs, BF16 activation/KV storage and FP32 accumulation remain unchanged.

DeepSeek's reference uses native SGLang tuning without CAMBLAS. Its fresh-process range is 149.19–158.04 tokens/s; CAMBLAS ranges from 137.03 to 156.84. All 148.9 million measured logits matched bitwise with the original FP4 experts, FP8 dense weights/KV and FP32 accumulation. Llama retains BF16 and showed no meaningful gain.

MiMo retains original MXFP4 expert weights, block128 FP8 dense weights, BF16 activations/KV and FP32 accumulation. All reported MiMo pairs matched every raw logit bitwise; the BF16 PyTorch path stays unchanged. Pro profiles: 128 tokens: FlashInfer, with overlap; 8192 tokens: FlashInfer, without overlap. Both disable FlashInfer autotuning and prefill CUDA graphs. The Flash row is provisional: three pairs passed on one node, but later fresh-process comparisons also differed between reference-only runs.

The [runtime](../configs/sglang.json) pins reproduction. [Archived results](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md), including Llama, precede the PyTorch interface change.
