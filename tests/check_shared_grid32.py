#!/usr/bin/env python3
"""Check the retained FP32 packed grid with full scalar output oracles."""

import argparse
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Exercise partial tiles and concurrent callers on 64 allocated Grace CPUs.

    Notes
    -----
    Supply a CAMBLAS bridge built with the private spin executor and
    ``CAMBLAS_FRAMEWORK_SHARED_GRID32=1``. The mathematical checks compare every
    output with a compensated double-precision scalar dot product. A separate
    outer-OpenMP test verifies concurrent callers use the nested fallback.
    These checks use the first 64 available CPUs and collect no timings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc-14"))
    parser.add_argument(
        "--torch-context", action="store_true", help="Also check deep PyTorch packing"
    )
    args = parser.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        parser.error("These checks require Grace")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 64:
        parser.error("At least 64 allocated, idle CPUs are required")
    os.sched_setaffinity(0, cpus[:64])
    bridge = args.library.resolve(strict=True)
    env = dict(
        os.environ,
        CAMBLAS_FRAMEWORK_THREADS="64",
        OMP_NUM_THREADS="64",
        OPENBLAS_NUM_THREADS="64",
        OMP_DYNAMIC="FALSE",
    )
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shared-grid-check-", dir=build) as temp:
        for name in ("shared_grid32", "batch_nested"):
            executable = Path(temp) / name
            subprocess.run(
                [
                    *shlex.split(args.cc),
                    "-O2",
                    "-std=gnu11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fopenmp",
                    ROOT / f"tests/{name}_test.c",
                    "-ldl",
                    "-lm",
                    "-o",
                    executable,
                ],
                env=env,
                check=True,
                timeout=120,
            )
            subprocess.run([executable, bridge], env=env, check=True, timeout=300)
            if name == "shared_grid32":
                subprocess.run(
                    [executable, bridge, "--deep"], env=env, check=True, timeout=300
                )
        if args.torch_context:
            oracle = Path(temp) / "deep_oracle.so"
            subprocess.run(
                [
                    *shlex.split(args.cc),
                    "-O2",
                    "-std=gnu11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fPIC",
                    "-shared",
                    "-fopenmp",
                    ROOT / "tests/shared_grid32_test.c",
                    "-ldl",
                    "-lm",
                    "-o",
                    oracle,
                ],
                env=env,
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes as c, sys, torch; "
                    "torch.set_num_threads(64); torch.set_num_interop_threads(1); "
                    "f = c.CDLL(sys.argv[1]).shared_grid32_check; "
                    "f.argtypes = [c.c_char_p, c.c_int]; f.restype = c.c_int; "
                    "sys.exit(f(sys.argv[2].encode(), 1))",
                    oracle,
                    bridge,
                ],
                env=dict(env, LD_PRELOAD=str(bridge)),
                check=True,
                timeout=300,
            )
    if args.torch_context:
        subprocess.run(
            [
                sys.executable,
                ROOT / "tests/test_mlp_packing.py",
                bridge,
                "--dtype",
                "float32",
                "--profile",
                "deep",
                "--torch-context",
            ],
            env=dict(env, LD_PRELOAD=str(bridge)),
            check=True,
            timeout=300,
        )


if __name__ == "__main__":
    main()
