#!/usr/bin/env python3
"""Check that private workers leave completed GEMM buffers untouched while idle."""

import argparse
import ctypes as ct
import os
import resource
import time
from pathlib import Path

import numpy as np


def main():
    """Verify output preservation across timed-out futex waits and later calls.

    Notes
    -----
    Run on allocated compute CPUs with matching bridge thread environment.
    Both precisions use exact rank-one products with an independent outer
    product oracle. The caller overwrites C after GEMM returns, waits through
    two worker timeout intervals, then verifies that no stale task ran.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--threads", type=int, choices=(16, 64), default=64)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < args.threads:
        parser.error("Run on allocated compute CPUs")
    os.sched_setaffinity(0, cpus[: args.threads])
    for key in ("CAMBLAS_FRAMEWORK_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    lib = ct.CDLL(str(args.library.resolve(strict=True)))
    assert lib.framework_blas_threads() == args.threads
    m, n, k = (264, 128, 1025) if args.threads == 16 else (128, 128, 128)
    for dtype, scalar, symbol in (
        (np.float32, ct.c_float, "cblas_sgemm"),
        (np.float64, ct.c_double, "cblas_dgemm"),
    ):
        ptr = ct.POINTER(scalar)
        gemm = getattr(lib, symbol)
        gemm.argtypes = [ct.c_int] * 6 + [
            scalar,
            ptr,
            ct.c_int,
            ptr,
            ct.c_int,
            scalar,
            ptr,
            ct.c_int,
        ]
        u = (np.arange(m) % 7 - 3).astype(dtype)
        v = (np.arange(n) % 11 - 5).astype(dtype)
        a = np.empty((m, k), dtype=dtype, order="F")
        b = np.empty((k, n), dtype=dtype, order="F")
        c = np.empty((m, n), dtype=dtype, order="F")
        for change in range(2):
            a[:] = u[:, None] + change
            b[:] = v[None, :] - change
            c[:] = np.nan
            gemm(
                102,
                111,
                111,
                m,
                n,
                k,
                1,
                a.ctypes.data_as(ptr),
                m,
                b.ctypes.data_as(ptr),
                k,
                0,
                c.ctypes.data_as(ptr),
                m,
            )
            expected = k * ((u + change)[:, None] * (v - change)[None, :])
            assert np.array_equal(c, expected), (symbol, change)
            c[:] = -731
            time.sleep(2.2)
            assert np.all(c == -731), (symbol, change, "idle worker changed caller C")
    print(f"Idle timeout and changed-input checks passed at {args.threads} threads")


if __name__ == "__main__":
    main()
