#!/usr/bin/env python3
"""Probe resident CUDA performance while developing the CAMBLAS backend."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camblas_gpu as cb


def main():
    """Compare every algorithm with strict eager PyTorch on matched tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("bench/results/gpu_probe.json")
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=[
            "square1024",
            "square4096",
            "square8192",
            "transpose",
            "gram",
            "mlp",
            "attention",
            "backward",
        ],
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=["classical", "lt", "strassen", "auto"]
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument(
        "--rotate", type=int, default=0, help="Rotate backend order across fresh probes"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for name in args.dtypes:
        dtype = getattr(torch, name)
        for workload in args.workloads:
            torch.manual_seed(20260906)

            def rand(shape, scale):
                return torch.randn(shape, device="cuda", dtype=dtype) * scale

            if workload.startswith("square"):
                n = int(workload[6:])
                inputs = [rand((n, n), n**-0.5), rand((n, n), n**-0.5)]
            elif workload == "transpose":
                inputs = [rand((512, 1024), 512**-0.5), rand((512, 2048), 512**-0.5)]
            elif workload == "gram":
                inputs = [rand((4096, 512), 4096**-0.5)]
            elif workload in ("mlp", "backward"):
                inputs = [
                    rand((512, 2048), 2048**-0.5),
                    rand((2048, 4096), 2048**-0.5),
                    rand((4096,), 0.01),
                    rand((4096, 1024), 4096**-0.5),
                    rand((1024,), 0.01),
                ]
                if workload == "backward":
                    for tensor in inputs[1:]:
                        tensor.requires_grad_(True)
            else:
                inputs = [rand((1024, 256), 256**-0.5) for _ in range(3)]

            def call(engine, inputs=inputs):
                mm = cb.matmul if engine == "camblas" else torch.matmul
                if workload.startswith("square") or workload == "transpose":
                    x, y = inputs
                    return mm(x.T if workload == "transpose" else x, y)
                if workload == "gram":
                    return mm(inputs[0].T, inputs[0])
                if workload in ("mlp", "backward"):
                    x, w1, b1, w2, b2 = inputs
                    for tensor in inputs[1:]:
                        tensor.grad = None
                    output = (
                        cb.mlp(*inputs)
                        if engine == "camblas"
                        else torch.relu(x @ w1 + b1) @ w2 + b2
                    )
                    if workload == "backward":
                        (output * output).mean().backward()
                    return output
                q, k, v = inputs
                if engine == "camblas":
                    return cb.attention(q, k, v, scale=1 / 16)
                return mm(torch.softmax(mm(q, k.T) / 16, dim=-1), v)

            reference = call("pytorch").detach()
            engines = [("pytorch", "auto")] + [("camblas", a) for a in args.algorithms]
            shift = args.rotate % len(engines)
            for engine, policy in engines[shift:] + engines[:shift]:
                with cb.algorithm(policy):
                    for _ in range(5):
                        output = call(engine)
                    torch.cuda.synchronize()
                    error = (output.detach() - reference).abs().max().item()
                    maximum = reference.abs().max().item()
                    assert error / max(maximum, 1e-15) < (
                        3e-4 if dtype == torch.float32 else 3e-11
                    )
                    samples = []
                    for _ in range(args.repetitions):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        output = call(engine)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    record = dict(
                        dtype=name,
                        workload=workload,
                        engine=engine,
                        algorithm=policy,
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        max_absolute_error=error,
                        reference_max_abs=maximum,
                        shapes=[list(value.shape) for value in inputs],
                        stats=cb.stats(reset=True, all_threads=True),
                    )
                    records.append(record)
                    print(
                        f"{name:7s} {workload:10s} {engine:7s} {policy:9s} {record['median_ms']:9.4f} ms error={error:.3g}",
                        flush=True,
                    )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(records, indent=2) + "\n")
            del inputs, output, reference


if __name__ == "__main__":
    main()
