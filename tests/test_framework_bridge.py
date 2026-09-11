#!/usr/bin/env python3
"""Independent small GEMM oracle for the research framework bridge."""

import argparse
import ctypes as c
import itertools
import json
import math
import os
from pathlib import Path


def main():
    """Validate both GEMM ABIs against small independent scalar reference products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["camblas", "openblas", "nvpl"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lib = c.CDLL(
        os.environ.get(
            "CAMBLAS_FRAMEWORK_BRIDGE",
            str(
                root / ".frameworks/prefix" / args.backend / "lib/libframework_blas.so"
            ),
        )
    )
    cases = 0
    fortran_cases = 0
    for kind, typ, tol in [("s", c.c_float, 3e-5), ("d", c.c_double, 2e-13)]:
        ptr = c.POINTER(typ)
        fn = getattr(lib, "cblas_" + kind + "gemm")
        fn.argtypes = [c.c_int] * 6 + [
            typ,
            ptr,
            c.c_int,
            ptr,
            c.c_int,
            typ,
            ptr,
            c.c_int,
        ]
        for order, ta, tb, alpha, beta in itertools.product(
            [101, 102],
            [111, 112, 113],
            [111, 112, 113],
            [0.0, 0.7, 1.0],
            [0.0, -0.2, 1.0],
        ):
            m, n, k = 7, 9, 11
            ar, ac = (m, k) if ta == 111 else (k, m)
            br, bc = (k, n) if tb == 111 else (n, k)
            lda = (ac if order == 101 else ar) + 3
            ldb = (bc if order == 101 else br) + 2
            ldc = (n if order == 101 else m) + 4

            def ix(i, j, ld):
                """Locate one stored element in the current CBLAS layout.

                Parameters
                ----------
                i, j : int
                    Zero-based physical row and column, before transposition.
                ld : int
                    Leading dimension in scalar elements, including padding.

                Returns
                -------
                int
                    Flat offset for row-major (101) or column-major (102) storage.
                """
                return i * ld + j if order == 101 else j * ld + i

            av = (typ * (lda * max(ar, ac)))(
                *[math.sin(i * 0.17) for i in range(lda * max(ar, ac))]
            )
            bv = (typ * (ldb * max(br, bc)))(
                *[math.cos(i * 0.13) for i in range(ldb * max(br, bc))]
            )
            size = ldc * (m if order == 101 else n)
            # A beta-zero GEMM must not read the old C values. NaNs expose an
            # accidental read even when the alpha/beta formula otherwise looks right.
            cv = (typ * size)(
                *[float("nan") if beta == 0 else 0.3 for _ in range(size)]
            )
            expected = []
            for i in range(m):
                for j in range(n):
                    dot = sum(
                        av[ix(i, q, lda) if ta == 111 else ix(q, i, lda)]
                        * bv[ix(q, j, ldb) if tb == 111 else ix(j, q, ldb)]
                        for q in range(k)
                    )
                    expected.append(
                        (
                            ix(i, j, ldc),
                            typ(alpha).value * dot
                            + typ(beta).value * (0.3 if beta else 0),
                        )
                    )
            fn(order, ta, tb, m, n, k, alpha, av, lda, bv, ldb, beta, cv, ldc)
            for index, ref in expected:
                assert math.isfinite(cv[index]) and abs(cv[index] - ref) < tol * max(
                    1.0, abs(ref)
                ), (
                    args.backend,
                    kind,
                    order,
                    ta,
                    tb,
                    alpha,
                    beta,
                    index,
                    cv[index],
                    ref,
                )
            cases += 1
            # The Fortran ABI is column-major and passes scalar arguments by
            # address. Reuse the independent scalar reference for the same case.
            if order == 102:
                cv = (typ * size)(
                    *[float("nan") if beta == 0 else 0.3 for _ in range(size)]
                )
                ff = getattr(lib, kind + "gemm_")
                ff.argtypes = (
                    [c.c_char_p, c.c_char_p]
                    + [c.POINTER(c.c_int)] * 3
                    + [
                        ptr,
                        ptr,
                        c.POINTER(c.c_int),
                        ptr,
                        c.POINTER(c.c_int),
                        ptr,
                        ptr,
                        c.POINTER(c.c_int),
                    ]
                )
                ints = [c.c_int(v) for v in (m, n, k, lda, ldb, ldc)]
                al, be = typ(alpha), typ(beta)
                ff(
                    {111: b"N", 112: b"T", 113: b"C"}[ta],
                    {111: b"N", 112: b"T", 113: b"C"}[tb],
                    *[c.byref(v) for v in ints[:3]],
                    c.byref(al),
                    av,
                    c.byref(ints[3]),
                    bv,
                    c.byref(ints[4]),
                    c.byref(be),
                    cv,
                    c.byref(ints[5]),
                )
                for index, ref in expected:
                    assert math.isfinite(cv[index]) and abs(
                        cv[index] - ref
                    ) < tol * max(1.0, abs(ref)), (
                        "Fortran",
                        args.backend,
                        kind,
                        ta,
                        tb,
                        index,
                    )
                fortran_cases += 1
    stats = (c.c_uint64 * 12)()
    assert lib.framework_blas_stats(stats, 12) == 12
    assert (
        stats[0] == (cases + fortran_cases) // 2
        and stats[1] == (cases + fortran_cases) // 2
        and stats[9] == 0
    )
    assert stats[10] == cases and stats[11] == fortran_cases
    print(
        json.dumps(
            dict(
                backend=args.backend,
                cases=cases,
                fortran_cases=fortran_cases,
                counters=list(stats),
                status="passed",
            )
        )
    )


if __name__ == "__main__":
    main()
