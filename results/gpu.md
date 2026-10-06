# GPU results

GH200, 2026-10-04; three rotated fresh processes per backend. Ratios are baseline/CAMBLAS latency; values below 1 indicate a regression. These historical measurements use the earlier interface and automatic Strassen selection.

| Workload | Precision | vs PyTorch resident | vs PyTorch with transfers | vs previous CAMBLAS resident | vs previous CAMBLAS with transfers |
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

FP64 32768² transfer medians varied from 1.189 to 5.957 seconds. Resident gains did not prevent the transfer regression. [Complete historical tables and protocol](https://github.com/p3jitnath/camblas/blob/ab838f2/README.md#cuda-backend) include the remaining workloads. Current LLM measurements are in [LLM results](llm.md).
