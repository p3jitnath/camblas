# LLM results

Quiet exclusive GH200 nodes; batch one, 256 greedy output tokens and median decode latency across fresh processes.

| Model | GPUs | Prompt | SGLang tokens/s | CAMBLAS tokens/s | Gain | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Llama 3.1 70B | 4 | 128 | 78.31 | 78.30 | 0.0% | Historical |
| DeepSeek V4.1 Flash | 4 | 128 | 156.82 | 138.96 | −11.4% | Three fresh pairs |
| GLM 5.3 Flash | 4 | 128 | 152.14 | 161.30 | 6.02% | Three fresh pairs |
| MiMo V2.6 Flash RL | 4 | 128 | 172.08 | 173.79 | 0.99% | Three fresh pairs; FP8 only |
| MiMo V2.6 Pro RL | 8 | 128 | 73.93 | 74.25 | 0.44% | Three fresh pairs; FP8 only |
| MiMo V2.6 Pro RL | 8 | 8192 | 69.04 | 69.69 | 0.94% | Three fresh pairs; FP8 only |

GLM's 89.2 million checked logits matched bitwise, with shared native MoE tuning. DeepSeek's 148.9 million checked logits matched bitwise; its reference-only range is 149.19–158.04 tokens/s. Llama showed no meaningful gain.

MiMo retains original MXFP4 experts, block128 FP8 dense weights, BF16 activation/KV storage and FP32 accumulation. All listed pairs passed bitwise raw-logit checks. MiMo disables FlashInfer autotuning and prefill CUDA graphs.

Profiles: Flash/128: FlashInfer, overlap on, KV splits 8; Pro/128: FlashInfer, overlap on, KV splits 8; Pro/8192: FlashInfer, overlap on, KV splits 8.

The [runtime](../configs/sglang.json) pins reproduction. [Archived measurements](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md), including Llama, precede the PyTorch interface change.
