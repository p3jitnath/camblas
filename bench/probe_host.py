#!/usr/bin/env python3
"""Probe explicit-copy and coherent-host CPU-to-CPU pipelines on Grace Hopper."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camblas_gpu as cb
from camblas_gpu._native import tensor_module


def main():
    """Measure three inference pipelines on matching seeded pageable CPU inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
        ],
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--repetitions", type=int, default=41)
    args = parser.parse_args()
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    native = tensor_module()
    if native is None:
        parser.error("Build with scripts/build_cuda.py --torch")
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.dtypes:
        for workload in args.workloads:
            rng = np.random.default_rng(20260906)

            def array(shape, scale):
                return torch.from_numpy(
                    (rng.standard_normal(shape) * scale).astype(name)
                )

            if workload.startswith("square"):
                n = int(workload[6:])
                inputs = [array((n, n), n**-0.5), array((n, n), n**-0.5)]
            elif workload == "transpose":
                inputs = [array((512, 1024), 512**-0.5), array((512, 2048), 512**-0.5)]
            elif workload == "gram":
                inputs = [array((4096, 512), 4096**-0.5)]
            elif workload == "mlp":
                inputs = [
                    array((512, 2048), 2048**-0.5),
                    array((2048, 4096), 2048**-0.5),
                    array((4096,), 0.01),
                    array((4096, 1024), 4096**-0.5),
                    array((1024,), 0.01),
                ]
            elif workload == "attention":
                inputs = [array((1024, 256), 256**-0.5) for _ in range(3)]
            else:
                parser.error(f"Unsupported inference workload {workload}")

            def call(engine, inputs=inputs):
                host = engine == "coherent"
                values = inputs if host else [value.cuda() for value in inputs]
                mm = (
                    native.matmul_host
                    if host
                    else cb.matmul
                    if engine == "camblas"
                    else torch.matmul
                )
                if workload.startswith("square") or workload == "transpose":
                    a, b = values
                    output = mm(a.T if workload == "transpose" else a, b)
                elif workload == "gram":
                    output = mm(values[0].T, values[0])
                elif workload == "mlp":
                    x, w1, b1, w2, b2 = values
                    output = (
                        native.mlp_host(*values)
                        if host
                        else cb.mlp(*values)
                        if engine == "camblas"
                        else torch.relu(x @ w1 + b1) @ w2 + b2
                    )
                else:
                    q, k, v = values
                    output = (
                        native.attention_host(q, k, v, scale=1 / 16)
                        if host
                        else cb.attention(q, k, v, scale=1 / 16)
                        if engine == "camblas"
                        else torch.softmax((q @ k.T) / 16, dim=-1) @ v
                    )
                return output if host else output.cpu()

            reference = call("pytorch")
            for engine in ("pytorch", "camblas", "coherent"):
                for _ in range(10):
                    output = call(engine)
                difference = (output - reference).double().abs()
                error = difference.max().item() / max(
                    reference.abs().max().item(), 1e-15
                )
                assert error < (3e-4 if name == "float32" else 3e-11)
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
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                    max_scaled_error=error,
                    stats=cb.stats(all_threads=True, reset=True),
                )
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                print(
                    f"{name:7s} {workload:10s} {engine:8s} {record['median_ms']:9.4f} ms error={error:.3g}",
                    flush=True,
                )
            del inputs, output, reference, difference


if __name__ == "__main__":
    main()
