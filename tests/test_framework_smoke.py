"""Small full-output NumPy/PyTorch routing and gradient checks, without timing."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np


def main():
    """Check full outputs, gradients and backend routing with bounded CPU inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("camblas", "openblas", "nvpl"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bridge_path = Path(
        os.environ.get(
            "CAMBLAS_FRAMEWORK_BRIDGE",
            root / ".frameworks/prefix" / args.backend / "lib/libframework_blas.so",
        )
    ).resolve()
    bridge = ctypes.CDLL(str(bridge_path))
    bridge.framework_blas_backend.restype = ctypes.c_char_p
    observed = bridge.framework_blas_backend().decode()
    assert observed == (
        "camblas+openblas-compatibility" if args.backend == "camblas" else args.backend
    )
    assert bridge.framework_blas_threads() == 1, (
        "Run this bounded smoke test with CAMBLAS_FRAMEWORK_THREADS=1"
    )
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.backends.mkldnn.enabled or not torch.backends.mkldnn.is_available()

    if args.backend == "camblas":
        info = (ctypes.c_void_p * 4)()
        assert ctypes.CDLL(None).dladdr(
            ctypes.cast(bridge.camblas_dgemm_plan_workspace, ctypes.c_void_p),
            ctypes.byref(info),
        )
        core_path = Path(ctypes.cast(info[0], ctypes.c_char_p).value.decode()).resolve()
        assert core_path == bridge_path.parent / "libcamblas_sve_nr4.so", core_path

    def oracle(left, right):
        """Multiply two small matrices using BLAS-independent scalar accumulation.

        Parameters
        ----------
        left : numpy.ndarray
            Two-dimensional left operand, with shape ``(m, k)``.
        right : numpy.ndarray
            Two-dimensional right operand, with shape ``(k, n)``.

        Returns
        -------
        numpy.ndarray
            Product with shape ``(m, n)``, accumulated using Python floats and
            returned in float64 regardless of the input precision.
        """
        return np.array(
            [
                [
                    sum(
                        float(left[i, q]) * float(right[q, j])
                        for q in range(left.shape[1])
                    )
                    for j in range(right.shape[1])
                ]
                for i in range(left.shape[0])
            ]
        )

    checks = []
    for dtype in (np.float32, np.float64):
        tolerance = 2e-5 if dtype == np.float32 else 2e-12
        for changed in (0, 1):
            left = (np.sin(np.arange(7 * 11) + changed).reshape(7, 11) / 3).astype(
                dtype
            )
            right = (np.cos(np.arange(11 * 9) - changed).reshape(9, 11).T / 7).astype(
                dtype
            )
            expected = oracle(left, right)
            for framework in ("numpy", "pytorch"):
                bridge.framework_blas_reset_stats()
                if framework == "numpy":
                    result = left @ right
                else:
                    a = torch.tensor(left, requires_grad=True)
                    b = torch.tensor(right, requires_grad=True)
                    value = a @ b
                    value.sum().backward()
                    result = value.detach().numpy()
                    np.testing.assert_allclose(
                        a.grad.numpy(),
                        oracle(np.ones((7, 9)), right.T),
                        atol=tolerance,
                        rtol=tolerance,
                    )
                    np.testing.assert_allclose(
                        b.grad.numpy(),
                        oracle(left.T, np.ones((7, 9))),
                        atol=tolerance,
                        rtol=tolerance,
                    )
                np.testing.assert_allclose(
                    result, expected, atol=tolerance, rtol=tolerance
                )
                stats = (ctypes.c_uint64 * 12)()
                assert bridge.framework_blas_stats(stats, 12) == 12
                assert stats[9] == 0 and stats[0] + stats[1] > 0, (
                    "Expected bridge did not receive GEMM"
                )
                checks.append(
                    {
                        "framework": framework,
                        "dtype": np.dtype(dtype).name,
                        "changed_input": changed,
                        "counters": list(stats),
                    }
                )
    assert str(bridge_path) in Path("/proc/self/maps").read_text()
    print(
        json.dumps(
            {"status": "passed", "backend": observed, "checks": checks}, indent=2
        )
    )


if __name__ == "__main__":
    main()
