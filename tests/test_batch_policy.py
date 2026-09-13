"""Check batched-route selection and exceptional-input classical fallbacks."""

import argparse
import ctypes as ct
import json
from pathlib import Path


def check_precision(library, dtype_name):
    """Check full outputs and routing for finite, overflowing and nonfinite inputs.

    Parameters
    ----------
    library : ctypes.CDLL
        Isolated CAMBLAS bridge loaded by the validation runner.
    dtype_name : str
        Either ``float32`` or ``float64``.

    Returns
    -------
    int
        Number of complete matrix products checked without a BLAS oracle.
    """
    # Import after main loads the candidate: a BLAS-linked NumPy must not
    # resolve the core SONAME to an installed control before this check.
    import numpy as np

    dtype = np.dtype(dtype_name)
    scalar = ct.c_float if dtype_name == "float32" else ct.c_double
    pointer = ct.POINTER(scalar)
    gemm = getattr(library, "cblas_sgemm" if dtype_name == "float32" else "cblas_dgemm")
    gemm.argtypes = [ct.c_int] * 6 + [
        scalar,
        pointer,
        ct.c_int,
        pointer,
        ct.c_int,
        scalar,
        pointer,
        ct.c_int,
    ]
    gemm.restype = None
    library.framework_blas_stats.argtypes = [ct.POINTER(ct.c_uint64), ct.c_size_t]
    library.framework_blas_stats.restype = ct.c_int
    library.framework_blas_reset_stats.restype = None
    n = 512
    cases = 0
    for transpose in (111, 112):
        for scenario in ("finite", "large_a", "large_b", "nan_a", "infinite_b"):
            a = np.eye(n, dtype=dtype, order="F")
            logical_b = np.eye(n, dtype=dtype, order="F")
            expected = np.eye(n, dtype=dtype)
            if scenario == "finite":
                a *= 2
                logical_b.fill(3)
                expected.fill(6)
            elif scenario == "large_a":
                a *= np.finfo(dtype).max / 2
                expected *= np.finfo(dtype).max / 2
            elif scenario == "large_b":
                logical_b *= np.finfo(dtype).max / 2
                expected *= np.finfo(dtype).max / 2
            elif scenario == "nan_a":
                a[0, 0] = np.nan
                expected[0, :] = np.nan
            else:
                logical_b[0, 0] = np.inf
                expected[:, 0] = np.nan
                expected[0, 0] = np.inf
            b = np.array(logical_b if transpose == 111 else logical_b.T, order="F")
            saved_a, saved_b = a.copy(), b.copy()
            c = np.full((n, n), np.nan, dtype=dtype, order="F")
            library.framework_blas_reset_stats()
            gemm(
                102,
                111,
                transpose,
                n,
                n,
                n,
                1,
                a.ctypes.data_as(pointer),
                n,
                b.ctypes.data_as(pointer),
                n,
                0,
                c.ctypes.data_as(pointer),
                n,
            )
            np.testing.assert_array_equal(c, expected)
            np.testing.assert_array_equal(a, saved_a)
            np.testing.assert_array_equal(b, saved_b)
            counters = (ct.c_uint64 * 12)()
            assert library.framework_blas_stats(counters, 12) == 12
            route = 4 if dtype_name == "float32" else 5
            assert counters[route] == (scenario == "finite"), (scenario, list(counters))
            cases += 1
    return cases


def main():
    """Run on allocated CPUs; this is a correctness test, never a timing source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    library = ct.CDLL(str(args.library.resolve(strict=True)), mode=ct.RTLD_GLOBAL)
    count = sum(check_precision(library, dtype) for dtype in ("float32", "float64"))
    print(json.dumps({"full_output_policy_cases": count, "status": "passed"}))


if __name__ == "__main__":
    main()
