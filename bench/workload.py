#!/usr/bin/env python3
"""One fresh-process application measurement; routing and output samples saved."""

import argparse
import ctypes as ct
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    """Run one fresh-process workload and save timing, routing and output checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True, choices=["camblas", "openblas", "nvpl"]
    )
    parser.add_argument("--framework", required=True, choices=["numpy", "pytorch"])
    parser.add_argument(
        "--workload",
        required=True,
        choices=[
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
    parser.add_argument("--dtype", required=True, choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    root = Path(
        os.environ.get("CAMBLAS_FRAMEWORK_ROOT", Path(__file__).resolve().parents[1])
    )
    bridge_path = Path(
        os.environ.get(
            "CAMBLAS_FRAMEWORK_BRIDGE",
            root / ".frameworks/prefix" / args.backend / "lib/libframework_blas.so",
        )
    ).resolve()
    bridge = ct.CDLL(str(bridge_path))
    core_identity = None
    if args.backend == "camblas":
        info = (ct.c_void_p * 4)()
        assert ct.CDLL(None).dladdr(
            ct.cast(bridge.camblas_dgemm_plan_workspace, ct.c_void_p), ct.byref(info)
        )
        core_path = Path(ct.cast(info[0], ct.c_char_p).value.decode()).resolve()
        assert core_path == bridge_path.parent / "libcamblas_sve_nr4.so", (
            "Wrong resolved core",
            core_path,
            bridge_path,
        )
        core_identity = dict(
            path=str(core_path),
            sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
        )
    bridge.framework_blas_backend.restype = ct.c_char_p
    assert bridge.framework_blas_threads() == args.threads
    assert len(os.sched_getaffinity(0)) == args.threads
    torch = None
    if args.framework == "pytorch":
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        assert (
            not torch.backends.mkldnn.enabled
            or not torch.backends.mkldnn.is_available()
        )
    rng = np.random.default_rng(20260906)
    dtype = np.dtype(args.dtype)

    def array(shape, scale=1.0, gradient=False):
        """Create an input using the shared seed and selected framework precision.

        Parameters
        ----------
        shape : tuple of int
            Dimensions of the input array or CPU tensor.
        scale : float, optional
            Factor applied to standard-normal samples before the dtype conversion.
        gradient : bool, optional
            Whether PyTorch should track gradients; ignored for NumPy arrays.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Input in the selected dtype, with identical sampled values for both
            frameworks when calls are made in the same order.
        """
        x = (rng.standard_normal(shape) * scale).astype(dtype)
        return torch.tensor(x, requires_grad=gradient) if torch else x

    def relu(x):
        """Apply the selected framework's elementwise rectified linear activation.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            Input using the framework selected for this process.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Elementwise maximum of the input and zero.
        """
        return torch.relu(x) if torch else np.maximum(x, 0)

    def host(x):
        """Expose a CPU result as a NumPy array for validation.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            CPU value using the selected framework.

        Returns
        -------
        numpy.ndarray
            Original array or detached tensor view sharing the input storage.
            The caller must not modify this view during validation.
        """
        return x.detach().numpy() if torch else x

    flops = None
    parameters = []
    if args.workload.startswith("square"):
        n = int(args.workload[6:])
        x = array((n, n), n**-0.5)
        y = array((n, n), n**-0.5)

        def operation():
            """Return the square matrix product using the prepared input arrays."""
            return x @ y

        flops = 2 * n**3
    elif args.workload == "transpose":
        x = array((512, 1024), 512**-0.5)
        y = array((512, 2048), 512**-0.5)

        def operation():
            """Return the matrix product with a transposed left input view."""
            return x.T @ y

        flops = 2 * 1024 * 2048 * 512
    elif args.workload == "gram":
        x = array((4096, 512), 4096**-0.5)

        def operation():
            """Return the Gram matrix of the prepared input array."""
            return x.T @ x
    elif args.workload in ("mlp", "backward"):
        grad = args.workload == "backward"
        if grad and not torch:
            raise ValueError("Backward requires PyTorch")
        x = array((512, 2048), 2048**-0.5)
        w1 = array((2048, 4096), 2048**-0.5, grad)
        b1 = array((4096,), 0.01, grad)
        w2 = array((4096, 1024), 4096**-0.5, grad)
        b2 = array((1024,), 0.01, grad)
        parameters = [w1, b1, w2, b2] if grad else []

        def operation():
            """Evaluate the two-layer MLP and, when requested, its parameter gradients.

            Returns
            -------
            numpy.ndarray or torch.Tensor
                Output of the second affine layer, before any output activation.

            Notes
            -----
            Gradients are cleared on every call. Backward timing includes the
            squared-output mean loss and differentiation, not gradient accumulation
            from earlier warm-up or measured calls.
            """
            for v in parameters:
                v.grad = None
            z = relu(x @ w1 + b1) @ w2 + b2
            if grad:
                (z * z).mean().backward()
            return z
    elif args.workload == "attention":
        q = array((1024, 256), 256**-0.5)
        k = array((1024, 256), 256**-0.5)
        v = array((1024, 256), 256**-0.5)

        def operation():
            """Return scaled dot-product attention for the prepared query, key and value.

            Returns
            -------
            numpy.ndarray or torch.Tensor
                Attention-weighted values, with shape ``(1024, 256)``.
            """
            scores = (q @ k.T) / 16
            if torch:
                weights = torch.softmax(scores, dim=-1)
            else:
                weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
                weights /= weights.sum(axis=-1, keepdims=True)
            return weights @ v

    # Warm up allocation and dispatch before resetting the counters. Each timed
    # region still contains the complete application operation, including backward
    # work when selected; validation and JSON serialisation remain outside it.
    for _ in range(3):
        output = operation()
    bridge.framework_blas_reset_stats()
    times = []
    for _ in range(args.repetitions):
        start = time.perf_counter_ns()
        output = operation()
        end = time.perf_counter_ns()
        times.append((end - start) * 1e-9)
    stats = (ct.c_uint64 * 12)()
    assert bridge.framework_blas_stats(stats, 12) == 12 and stats[9] == 0
    small_pool_calls = None
    if hasattr(bridge, "framework_blas_small_pool_calls"):
        bridge.framework_blas_small_pool_calls.restype = ct.c_uint64
        small_pool_calls = int(bridge.framework_blas_small_pool_calls())
    symmetric_calls = None
    if hasattr(bridge, "framework_blas_symmetric_calls"):
        bridge.framework_blas_symmetric_calls.restype = ct.c_uint64
        symmetric_calls = int(bridge.framework_blas_symmetric_calls())
    output_touch_calls = None
    if hasattr(bridge, "framework_blas_output_touch_calls"):
        bridge.framework_blas_output_touch_calls.restype = ct.c_uint64
        output_touch_calls = int(bridge.framework_blas_output_touch_calls())
    optional_counters = {}
    for name in [
        "output_resident_skips",
        "dot_calls",
        "compact_calls",
        "bilinear_calls",
    ]:
        symbol = "framework_blas_" + name
        if hasattr(bridge, symbol):
            function = getattr(bridge, symbol)
            function.restype = ct.c_uint64
            optional_counters[name] = int(function())

    def signature(value):
        """Summarise a finite output using deterministic samples and global norms.

        Parameters
        ----------
        value : numpy.ndarray or torch.Tensor
            Non-empty CPU output or gradient from the selected framework.

        Returns
        -------
        dict
            Element count, up to 4096 samples, Euclidean norm and largest absolute
            value. Samples and norm calculations use float64 for both precisions.

        Raises
        ------
        AssertionError
            If any output element is non-finite.

        Notes
        -----
        The sampling seed is independent of input generation. Signatures detect
        discrepancies across processes but are not a full-output correctness proof.
        """
        value = host(value).reshape(-1)
        assert np.isfinite(value).all()
        indices = np.random.default_rng(991).choice(
            value.size, min(4096, value.size), replace=False
        )
        return dict(
            size=value.size,
            samples=value[indices].astype(np.float64).tolist(),
            norm=float(np.linalg.norm(value.astype(np.float64))),
            max_abs=float(np.max(np.abs(value))),
        )

    outputs = [signature(output)] + [signature(v.grad) for v in parameters]
    # Check sampled GEMM/Gram entries independently of BLAS. Scaling the tolerance
    # by the sum of absolute products also handles cancellation near zero.
    oracle = None
    if args.workload.startswith("square") or args.workload in ("transpose", "gram"):
        left, right = (
            (host(x), host(y))
            if args.workload.startswith("square")
            else (host(x).T, host(y) if args.workload == "transpose" else host(x))
        )
        result = host(output)
        errors = []
        for _ in range(64):
            i = int(rng.integers(left.shape[0]))
            j = int(rng.integers(right.shape[1]))
            products = left[i, :].astype(np.float64) * right[:, j].astype(np.float64)
            ref = float(np.sum(products))
            error = abs(float(result[i, j]) - ref)
            limit = (3e-5 if args.dtype == "float32" else 2e-12) * max(
                1.0, float(np.sum(np.abs(products)))
            )
            assert error <= limit, (i, j, error, limit)
            errors.append(error)
        oracle = dict(sampled_entries=64, max_absolute_error=max(errors))
    maps = Path("/proc/self/maps").read_text()
    libraries = sorted(
        set(
            line.split()[-1]
            for line in maps.splitlines()
            if ".so" in line and "/" in line
        )
    )
    assert str(bridge_path) in libraries
    record = dict(
        backend=args.backend,
        framework=args.framework,
        workload=args.workload,
        dtype=args.dtype,
        threads=args.threads,
        core_identity=core_identity,
        small_pool_calls=small_pool_calls,
        symmetric_calls=symmetric_calls,
        output_touch_calls=output_touch_calls,
        runtime_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_WAIT_POLICY",
                "GOMP_SPINCOUNT",
                "MALLOC_TRIM_THRESHOLD_",
                "MALLOC_MMAP_THRESHOLD_",
                "MALLOC_ARENA_MAX",
                "GLIBC_TUNABLES",
            )
        },
        optional_counters=optional_counters,
        repetitions=args.repetitions,
        warmups=3,
        seconds=times,
        flops=flops,
        counters=list(stats),
        outputs=outputs,
        oracle=oracle,
        bridge_sha256=hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
        observed_backend=bridge.framework_blas_backend().decode(),
        libraries=libraries,
        numpy_version=np.__version__,
        torch_version=torch.__version__ if torch else None,
        affinity=sorted(os.sched_getaffinity(0)),
    )
    args.output.write_text(json.dumps(record))
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("outputs", "libraries")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
