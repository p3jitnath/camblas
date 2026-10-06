#!/usr/bin/env python3
"""Measure matching PyTorch CUDA workloads with and without host transfers."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    """Measure synchronous eager calls and retain numerical and device evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        required=True,
        choices=[
            "square1024",
            "square4096",
            "square8192",
            "square12288",
            "square16384",
            "square24576",
            "square32768",
            "transpose",
            "gram",
            "mlp",
            "attention",
            "attention32",
            "attention64",
            "attention256",
            "attention512",
            "attention2048",
            "attention4096",
            "attention8192",
            "attention64x1024",
            "attention1024x64",
            "attention64x32",
            "attention1024x32",
            "backward",
        ],
    )
    parser.add_argument("--dtype", required=True, choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--transfer-first", action="store_true")
    parser.add_argument(
        "--output-memory",
        choices=["pageable", "prefault", "pinned"],
        default="pageable",
        help="Allocate each CPU output inside transfer timing; prefault includes zero-fill",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["pytorch", "camblas"], default="pytorch")
    parser.add_argument(
        "--pytorch-baseline",
        choices=["eager", "fused", "compiled", "compiled-fused"],
        default="eager",
        help="Use eager expressions or linear/SDPA APIs, optionally compiled",
    )
    parser.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="reduce-overhead",
        help="PyTorch compiler mode; compilation is excluded from timed calls",
    )
    parser.add_argument(
        "--algorithm",
        default="auto",
        choices=[
            "auto",
            "classical",
            "lt",
            "strassen",
            "strassen2",
            "strassen3",
            "strassen4",
            "symmetric",
        ],
    )
    args = parser.parse_args()
    os.environ["CAMBLAS_ENABLE"] = "1" if args.engine == "camblas" else "0"
    os.environ["CAMBLAS_CUDA_ALGORITHM"] = args.algorithm
    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch build and accessible GPU are required")
    if args.repetitions < 5 or args.warmups < 3:
        parser.error("Use at least five timed calls and three warm-ups")
    assert len(os.sched_getaffinity(0)) == args.threads
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    cb = None
    if args.engine == "camblas":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import camblas._kernels as cb
        from camblas import _autoload

        _autoload()
    rng = np.random.default_rng(20260906)
    dtype = np.dtype(args.dtype)
    host_inputs = []
    gradient_indices = []
    dedicated_api = args.pytorch_baseline in ("fused", "compiled-fused")
    compiled = args.pytorch_baseline in ("compiled", "compiled-fused")

    def array(shape, scale=1.0, gradient=False):
        """Append the same seeded input used by the existing CPU benchmark.

        Parameters
        ----------
        shape : tuple of int
            Input dimensions.
        scale : float, optional
            Standard-normal scaling applied before dtype conversion.
        gradient : bool, optional
            Whether to compute a parameter gradient.

        Returns
        -------
        torch.Tensor
            Pageable CPU tensor owning the generated input.
        """
        value = (rng.standard_normal(shape) * scale).astype(dtype)
        if gradient:
            gradient_indices.append(len(host_inputs))
        tensor = torch.from_numpy(value)
        host_inputs.append(tensor)
        return tensor

    if args.workload.startswith("square"):
        n = int(args.workload[6:])
        array((n, n), n**-0.5)
        array((n, n), n**-0.5)
    elif args.workload == "transpose":
        array((512, 1024), 512**-0.5)
        array((512, 2048), 512**-0.5)
    elif args.workload == "gram":
        array((4096, 512), 4096**-0.5)
    elif args.workload in ("mlp", "backward"):
        gradient = args.workload == "backward"
        array((512, 2048), 2048**-0.5)
        array((2048, 4096), 2048**-0.5, gradient)
        array((4096,), 0.01, gradient)
        array((4096, 1024), 4096**-0.5, gradient)
        array((1024,), 0.01, gradient)
    else:
        dimensions = args.workload.removeprefix("attention") or "1024"
        queries, _, keys = dimensions.partition("x")
        queries, keys = int(queries), int(keys or queries)
        array((queries, 256), 256**-0.5)
        array((keys, 256), 256**-0.5)
        array((keys, 256), 256**-0.5)

    # Dedicated linear APIs normally own contiguous [outputs, inputs] weights.
    # Preserve the mathematical arrays while storing those transposes densely.
    # This one-time CPU preparation is outside timing; each transfer call still
    # copies every weight. Both backends use the same prepared layouts.
    if dedicated_api and args.workload in ("mlp", "backward"):
        for index in (1, 3):
            host_inputs[index] = host_inputs[index].T.contiguous().T

    pytorch_function = None
    if args.pytorch_baseline != "eager":
        from torch.nn import functional

        def pytorch_expression(*inputs):
            """Evaluate the selected PyTorch expression with matching values.

            Parameters
            ----------
            *inputs : torch.Tensor
                Prepared operands in the selected workload's order.

            Returns
            -------
            torch.Tensor
                Matrix, MLP or attention output before gradient calculation.
            """
            if args.workload.startswith("square") or args.workload == "transpose":
                x, y = inputs
                return (x.T if args.workload == "transpose" else x) @ y
            if args.workload == "gram":
                return inputs[0].T @ inputs[0]
            if args.workload in ("mlp", "backward"):
                x, w1, b1, w2, b2 = inputs
                if dedicated_api:
                    hidden = functional.relu(functional.linear(x, w1.T, b1))
                    return functional.linear(hidden, w2.T, b2)
                return torch.relu(x @ w1 + b1) @ w2 + b2
            q, k, v = inputs
            if dedicated_api:
                return functional.scaled_dot_product_attention(
                    q[None, None],
                    k[None, None],
                    v[None, None],
                    dropout_p=0.0,
                    scale=1 / 16,
                )[0, 0]
            return torch.softmax((q @ k.T) / 16, dim=-1) @ v

        pytorch_function = (
            torch.compile(pytorch_expression, fullgraph=True, mode=args.compile_mode)
            if compiled
            else pytorch_expression
        )

    def device_inputs():
        """Copy every input and create leaf parameters for the backward workload.

        Returns
        -------
        list of torch.Tensor
            Fresh CUDA inputs, with gradient tracking enabled for selected parameters.
        """
        return [
            value.to("cuda").requires_grad_(i in gradient_indices)
            for i, value in enumerate(host_inputs)
        ]

    def operation(inputs):
        """Evaluate the same PyTorch expression for every selected backend.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Prepared CUDA tensors, with leaf parameters for gradients.

        Returns
        -------
        list of torch.Tensor
            Output followed by requested parameter gradients.
        """
        for index in gradient_indices:
            inputs[index].grad = None
        if pytorch_function is not None:
            output = pytorch_call(inputs)
        elif args.workload.startswith("square") or args.workload == "transpose":
            x, y = inputs
            output = torch.matmul(x.T if args.workload == "transpose" else x, y)
        elif args.workload == "gram":
            output = torch.matmul(inputs[0].T, inputs[0])
        elif args.workload in ("mlp", "backward"):
            x, w1, b1, w2, b2 = inputs
            output = torch.relu(x @ w1 + b1) @ w2 + b2
        else:
            q, k, v = inputs
            output = torch.softmax((q @ k.T) / 16, dim=-1) @ v
        if gradient_indices:
            (output * output).mean().backward()
        return [output] + [inputs[index].grad for index in gradient_indices]

    def pytorch_call(inputs):
        """Start a compiled iteration and include all execution overhead in timing.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Prepared CUDA operands for the selected expression.

        Returns
        -------
        torch.Tensor
            Expression output, with compiler and graph-dispatch overhead included.
        """
        if compiled and args.compile_mode in ("reduce-overhead", "max-autotune"):
            torch.compiler.cudagraph_mark_step_begin()
        return pytorch_function(*inputs)

    def signature(value):
        """Retain the same deterministic samples and norms as the CPU records.

        Parameters
        ----------
        value : torch.Tensor
            CPU or CUDA result or gradient.

        Returns
        -------
        dict
            Finite element count, samples, Euclidean norm and maximum magnitude.
        """
        values = value.detach().cpu().numpy().reshape(-1)
        assert np.isfinite(values).all()
        indices = np.random.default_rng(991).choice(
            values.size, min(4096, values.size), replace=False
        )
        return dict(
            size=values.size,
            samples=values[indices].astype(np.float64).tolist(),
            norm=float(np.linalg.norm(values.astype(np.float64))),
            max_abs=float(np.max(np.abs(values))),
        )

    measurements = {}
    modes = (
        ["transfer", "resident"] if args.transfer_first else ["resident", "transfer"]
    )
    for mode in modes:
        resident = device_inputs() if mode == "resident" else None

        def call():
            """Run one complete operation, copying results when transfers are timed.

            Returns
            -------
            list of torch.Tensor
                Output and requested gradients, on CPU for transfer timing or CUDA for resident timing.
            """
            inputs = resident if resident is not None else device_inputs()
            results = operation(inputs)
            if mode == "transfer":
                outputs = []
                for value in results:
                    value = value.detach()
                    if args.output_memory == "pageable":
                        host_output = value.cpu()
                    else:
                        host_output = torch.empty_like(
                            value,
                            device="cpu",
                            pin_memory=args.output_memory == "pinned",
                        )
                        if args.output_memory == "prefault":
                            host_output.zero_()
                        host_output.copy_(value)
                    outputs.append(host_output)
                return outputs
            return results

        for _ in range(args.warmups):
            output = call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_after_warmup, total_device_bytes = torch.cuda.mem_get_info()
        reserved_after_warmup = torch.cuda.memory_reserved()
        allocated_after_warmup = torch.cuda.memory_allocated()
        seconds = []
        for _ in range(args.repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = call()
            torch.cuda.synchronize()
            seconds.append((time.perf_counter_ns() - start) * 1e-9)
        measurements[mode] = dict(
            seconds=seconds,
            outputs=[signature(value) for value in output],
            output_layouts=[
                dict(
                    shape=list(value.shape),
                    stride=list(value.stride()),
                    dtype=str(value.dtype),
                    device=str(value.device),
                )
                for value in output
            ],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            device_free_bytes_after_warmup=free_after_warmup,
            device_total_bytes=total_device_bytes,
            torch_reserved_bytes_after_warmup=reserved_after_warmup,
            torch_allocated_bytes_after_warmup=allocated_after_warmup,
            input_transfer_bytes=sum(x.numel() * x.element_size() for x in host_inputs)
            if mode == "transfer"
            else 0,
            output_transfer_bytes=sum(x.numel() * x.element_size() for x in output)
            if mode == "transfer"
            else 0,
            host_input_bytes=sum(x.numel() * x.element_size() for x in host_inputs)
            if mode == "transfer"
            else 0,
            host_output_bytes=sum(x.numel() * x.element_size() for x in output)
            if mode == "transfer"
            else 0,
        )
        if mode == "resident":
            device_seconds = []
            for _ in range(args.repetitions):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = call()
                end_event.record()
                end_event.synchronize()
                device_seconds.append(start_event.elapsed_time(end_event) / 1000)
            measurements[mode]["device_seconds"] = device_seconds
        resident = None
        del output

    # Independent scalar dot-product checks use the already prepared host inputs.
    oracle = None
    if args.workload.startswith("square") or args.workload in ("transpose", "gram"):
        result = operation(device_inputs())[0].detach().cpu().numpy()
        left = host_inputs[0].numpy()
        right = host_inputs[1].numpy() if len(host_inputs) == 2 else left
        if args.workload in ("transpose", "gram"):
            left = left.T
        errors = []
        for _ in range(64):
            i = int(rng.integers(left.shape[0]))
            j = int(rng.integers(right.shape[1]))
            products = left[i, :].astype(np.float64) * right[:, j].astype(np.float64)
            error = abs(float(result[i, j]) - float(np.sum(products)))
            limit = (3e-5 if args.dtype == "float32" else 2e-12) * max(
                1.0, float(np.sum(np.abs(products)))
            )
            assert error <= limit, (i, j, error, limit)
            errors.append(error)
        oracle = dict(sampled_entries=64, max_absolute_error=max(errors))

    operator_names = None
    if args.pytorch_baseline != "eager":
        # CPU operator names identify SDPA's selected implementation without
        # requiring CUPTI. Profiling happens after every timed measurement.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            operation(device_inputs())
            torch.cuda.synchronize()
        operator_names = sorted(event.key for event in profile.key_averages())

    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and "/" in line
        }
    )
    assert any("libcublas.so" in path for path in libraries)
    properties = torch.cuda.get_device_properties(0)
    record = dict(
        backend="camblas_cuda" if cb is not None else "cuda",
        engine=args.engine,
        memory_mode="explicit_copy",
        output_memory=args.output_memory,
        transfer_entry="python",
        algorithm=args.algorithm if cb is not None else "pytorch",
        pytorch_baseline=args.pytorch_baseline,
        compile_mode=args.compile_mode if compiled else None,
        pytorch_operator_names=operator_names,
        camblas_cuda_stats=cb.stats(all_threads=True) if cb is not None else None,
        framework="pytorch",
        workload=args.workload,
        dtype=args.dtype,
        threads=args.threads,
        repetitions=args.repetitions,
        warmups=args.warmups,
        affinity=sorted(os.sched_getaffinity(0)),
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        matmul_precision=torch.get_float32_matmul_precision(),
        tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
        device=dict(
            name=properties.name,
            total_memory=properties.total_memory,
            capability=list(torch.cuda.get_device_capability(0)),
        ),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_sha256=[
            hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in host_inputs
        ],
        input_strides=[list(value.stride()) for value in host_inputs],
        measurements=measurements,
        oracle=oracle,
        libraries=libraries,
        library_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in libraries
            if any(
                name in path
                for name in (
                    "libcublas.so",
                    "libcublasLt.so",
                    "libcamblas_cuda",
                    "_camblas_cuda_torch",
                )
            )
        },
    )
    args.output.write_text(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                "workload": args.workload,
                "dtype": args.dtype,
                "device": record["device"],
                "torch_version": torch.__version__,
                "matmul_precision": record["matmul_precision"],
                "medians_ms": {
                    name: float(np.median(value["seconds"])) * 1000
                    for name, value in measurements.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
