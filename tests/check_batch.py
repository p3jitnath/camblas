#!/usr/bin/env python3
"""Run independent batched-GEMM correctness checks on allocated Grace CPUs."""

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
    """Check scalar oracles, concurrent callers and exceptional-input fallbacks.

    Notes
    -----
    Supply a CAMBLAS framework bridge with its matching native core beside it.
    The active Python environment must contain NumPy. At least 16 allocated,
    idle CPUs must be available in the inherited affinity mask; this runner
    uses the first 16 and never collects performance measurements. Boundary
    sweeps remain available through ``test_mlp_packing.py --profile batch``.
    This runner checks the supplied binaries; it does not instrument them.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc-14"))
    args = parser.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        parser.error("These checks require Grace/SVE128")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 16:
        parser.error("Bind this process to at least 16 allocated, idle CPUs")
    bridge = args.library.resolve(strict=True)
    core = bridge.parent / "libcamblas_sve_nr4.so"
    if not core.is_file():
        parser.error(f"Missing matching native core: {core}")
    os.sched_setaffinity(0, cpus[:16])
    env = dict(
        os.environ,
        CAMBLAS_FRAMEWORK_THREADS="16",
        OMP_NUM_THREADS="16",
        OPENBLAS_NUM_THREADS="16",
        OMP_DYNAMIC="FALSE",
    )
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    cc = shlex.split(args.cc)
    with tempfile.TemporaryDirectory(prefix="batch-check-", dir=build) as temporary:
        directory = Path(temporary)
        for precision in (32, 64):
            executable = directory / f"rectangular{precision}"
            subprocess.run(
                [
                    *cc,
                    "-O2",
                    "-std=gnu11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-mcpu=neoverse-v2",
                    f"-I{ROOT / 'include'}",
                    f"-I{ROOT / 'src'}",
                    f"-I{ROOT / 'framework'}",
                    ROOT / f"tests/rectangular{precision}_test.c",
                    core,
                    f"-Wl,-rpath,{core.parent}",
                    "-lm",
                    "-o",
                    executable,
                ],
                check=True,
                env=env,
                timeout=120,
            )
            subprocess.run([executable], check=True, env=env, timeout=180)
        nested = directory / "batch_nested"
        subprocess.run(
            [
                *cc,
                "-O2",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fopenmp",
                ROOT / "tests/batch_nested_test.c",
                "-ldl",
                "-lm",
                "-o",
                nested,
            ],
            check=True,
            env=env,
            timeout=120,
        )
        subprocess.run([nested, bridge], check=True, env=env, timeout=180)
        subprocess.run(
            [sys.executable, ROOT / "tests/test_batch_policy.py", bridge],
            check=True,
            env=env,
            timeout=180,
        )


if __name__ == "__main__":
    main()
