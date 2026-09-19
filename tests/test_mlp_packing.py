"""Full-output, BLAS-independent checks for deep and shallow packing policies."""

import argparse
import ctypes as ct
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    """Run the full-output packing checks on allocated compute CPUs.

    Notes
    -----
    The first positional argument is the framework bridge shared library.
    These deliberately large cases are correctness tests, not timing runs;
    use an allocation instead of running them on a login node.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--profile", choices=("deep", "shallow", "batch"), default="deep"
    )
    args = parser.parse_args()
    bridge = args.library.resolve()
    dtype = np.dtype(args.dtype)
    scalar = ct.c_float if args.dtype == "float32" else ct.c_double
    lib = ct.CDLL(str(bridge))
    ptr = ct.POINTER(scalar)
    fn = getattr(lib, "cblas_sgemm" if args.dtype == "float32" else "cblas_dgemm")
    fn.argtypes = [ct.c_int] * 6 + [
        scalar,
        ptr,
        ct.c_int,
        ptr,
        ct.c_int,
        scalar,
        ptr,
        ct.c_int,
    ]
    count = 0
    # Include selected deep-GEMM shapes and neighbours of the M/N/K dispatch
    # limits. Padding and changed inputs exercise the packing contract separately.
    shapes = [
        (4096, 512, 2048),
        (1024, 512, 4096),
        (4097, 513, 1025),
        (2049, 511, 4095),
        (1024, 512, 4097),
        (8193, 129, 1025),
        (2056, 2049, 1025),
        (1024, 512, 1024),
    ]
    if args.profile == "deep" and args.dtype == "float64":
        # Exercise both sides of the short-rectangle aspect-ratio bound,
        # its largest supported dimensions and the depth cut-offs.
        shapes = [
            (1024, 512, 4096),
            (2048, 512, 2048),
            (2049, 512, 2048),
            (1025, 513, 1025),
            (4096, 1024, 2048),
            (8192, 2048, 1025),
            (1024, 512, 1024),
            (1024, 512, 4097),
        ]
    if args.profile == "shallow":
        # Active square/transposed/attention products and policy neighbours.
        shapes = [
            (1024, 1024, 1024),
            (2048, 1024, 512),
            (256, 1024, 1024),
            (1024, 1024, 256),
            (127, 513, 257),
            (128, 129, 128),
            (2049, 2049, 1025),
            (513, 513, 127),
        ]
    if args.profile == "batch":
        # Check both sides of the batched square/transpose eligibility limits,
        # including even dimensions with incomplete micro-panels, and the
        # rectangular NN extension boundaries.
        shapes = [
            (510, 510, 510),
            (512, 512, 512),
            (514, 514, 514),
            (2046, 2046, 2046),
            (2048, 2048, 2048),
            (2050, 2050, 2050),
            (2048, 512, 254),
            (2048, 512, 256),
            (2048, 512, 258),
            (2048, 512, 1024),
            (2048, 512, 1026),
            (2050, 512, 512),
            (4096, 512, 2048),
            (4094, 512, 2048),
            (4098, 512, 2048),
            (1024, 512, 4096),
            (1024, 512, 4094),
            (1024, 512, 4098),
            (4096, 256, 512),
            (4096, 1026, 512),
        ]
    for m, n, k in shapes:
        i = np.arange(m, dtype=np.int64)
        j = np.arange(n, dtype=np.int64)
        depth = np.arange(k, dtype=np.int64)
        v = depth % 5 - 2
        q = depth % 11 - 5
        r = depth % 13 - 6
        t = depth % 3 - 1
        products = [int(np.sum(x * y)) for x, y in [(v, r), (v, t), (q, r), (q, t)]]
        for order, ta, tb in itertools.product([101, 102], [111, 112], [111, 112]):

            def storage(rows, cols, pad):
                """Allocate a padded matrix in the selected precision and CBLAS layout.

                Parameters
                ----------
                rows, cols : int
                    Physical matrix dimensions, before any BLAS transposition.
                pad : int
                    Extra scalar elements in each leading dimension.

                Returns
                -------
                raw : numpy.ndarray
                    Flat allocation initialised to the padding sentinel, -73.
                view : numpy.ndarray
                    Writable ``(rows, cols)`` view excluding the padding.
                ld : int
                    Leading dimension in scalar elements for the current order.
                """
                ld = (rows if order == 102 else cols) + pad
                raw = np.full(ld * (cols if order == 102 else rows), -73.0, dtype=dtype)
                view = (
                    raw.reshape(cols, ld).T[:rows, :]
                    if order == 102
                    else raw.reshape(rows, ld)[:, :cols]
                )
                return raw, view, ld

            ar, ac = (m, k) if ta == 111 else (k, m)
            br, bc = (k, n) if tb == 111 else (n, k)
            a, av, lda = storage(ar, ac, 3)
            b, bv, ldb = storage(br, bc, 5)
            c, cv, ldc = storage(m, n, 7)
            for change, (alpha, beta) in enumerate(
                [(1.0, 0.0), (0.5, -0.25), (-1.0, 1.0)]
            ):
                u = i % 7 - 3 + change
                p = i % 3 - 1 - change
                s = j % 5 - 2 - change
                w = j % 7 - 3 + change
                left = u[:, None] * v + p[:, None] * q
                right = r[:, None] * s + t[:, None] * w
                av[:] = left if ta == 111 else left.T
                bv[:] = right if tb == 111 else right.T
                before_a = a.copy()
                before_b = b.copy()
                cv[:] = np.nan if beta == 0 else 2.0
                fn(
                    order,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    alpha,
                    a.ctypes.data_as(ptr),
                    lda,
                    b.ctypes.data_as(ptr),
                    ldb,
                    beta,
                    c.ctypes.data_as(ptr),
                    ldc,
                )
                # A = u v^T + p q^T and B = r s^T + t w^T reduce AB to
                # four outer products. Integer dot products provide an exact,
                # BLAS-independent reference without an expensive scalar GEMM.
                expected = (
                    alpha
                    * (
                        products[0] * u[:, None] * s
                        + products[1] * u[:, None] * w
                        + products[2] * p[:, None] * s
                        + products[3] * p[:, None] * w
                    )
                    + beta * 2.0
                )
                assert np.array_equal(cv, expected), (
                    m,
                    n,
                    k,
                    order,
                    ta,
                    tb,
                    change,
                    float(np.max(np.abs(cv - expected))),
                )
                padding = (
                    c.reshape(n, ldc)[:, m:]
                    if order == 102
                    else c.reshape(m, ldc)[:, n:]
                )
                assert np.all(padding == -73.0)
                assert np.array_equal(a, before_a) and np.array_equal(b, before_b)
                count += 1
        print(json.dumps(dict(shape=[m, n, k], status="passed")), flush=True)
    print(
        json.dumps(
            dict(
                status="passed",
                cases=count,
                dtype=args.dtype,
                bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                checks="Full exact rank-two products without BLAS reference; changed values, both orders, all transposes, alpha/beta, NaN beta-zero C, padding/input preservation and dispatch boundaries",
            )
        )
    )


if __name__ == "__main__":
    main()
