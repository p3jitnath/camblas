# CPU results

Grace, 2026-10-02; 60 application cases, three fresh processes per backend. Speed-ups are geometric means of baseline/CAMBLAS latency ratios.

| Precision | Cores | Cases won | vs OpenBLAS | vs NVPL |
|---|---:|---:|---:|---:|
| FP32 | 16 | 15/15 | 1.580× | 1.111× |
| FP32 | 64 | 15/15 | 2.225× | 1.259× |
| FP64 | 16 | 15/15 | 1.496× | 1.118× |
| FP64 | 64 | 15/15 | 2.480× | 1.322× |

GCC 14.3.0, OpenBLAS 0.3.33 and NVPL 0.3.0. [Raw timings and correctness checks](data/grace_20261002.json) cover NumPy, SciPy and PyTorch workloads.
