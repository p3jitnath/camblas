# Experimental results

Four GH200 GPUs on a quiet exclusive node; batch one, 128 prompt tokens and 256 greedy output tokens. Rates use median decode latency.

| Model | SGLang tokens/s | CAMBLAS tokens/s | Gain | Evidence |
|---|---:|---:|---:|---|
| Llama 3.1 70B | 78.31 | 78.30 | 0.0% | Historical |
| DeepSeek V4.1 Flash | 156.82 | 138.96 | −11.4% | Three fresh pairs; tuned native SGLang |
| GLM 5.3 Flash | 152.14 | 161.30 | 6.02% | Three fresh pairs |

GLM shares the tuned SGLang MoE tiles between backends. All 89.2 million measured logits matched bitwise, including repeated and changed inputs. Original FP8 weights/inputs, BF16 activation/KV storage and FP32 accumulation remain unchanged.

DeepSeek's reference uses native SGLang tuning without CAMBLAS. Its fresh-process range is 149.19–158.04 tokens/s; CAMBLAS ranges from 137.03 to 156.84. All 148.9 million measured logits matched bitwise with the original FP4 experts, FP8 dense weights/KV and FP32 accumulation. Llama retains BF16 and showed no meaningful gain.

The [runtime](../configs/sglang.json) pins reproduction. [Archived results](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md), including Llama, precede the PyTorch interface change.
