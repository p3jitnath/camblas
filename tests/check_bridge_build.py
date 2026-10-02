#!/usr/bin/env python3
"""Compile the framework bridge's supported executor configurations."""

import argparse
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Check disabled and enabled executors without loading vendor libraries.

    Notes
    -----
    Object compilation exercises feature guards and unused-function warnings.
    A placeholder vendor path is sufficient because these objects are never
    linked or loaded. Generated objects live in a temporary build directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        print("Framework bridge executor compile checks require AArch64; skipped")
        return
    compiler = shlex.split(args.cc)
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    configurations = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
    with tempfile.TemporaryDirectory(
        prefix="bridge-build-check-", dir=build
    ) as temporary:
        for index, (openmp, spin, reuse) in enumerate(configurations):
            flags = [
                "-O2",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                f"-I{ROOT / 'include'}",
                f"-I{ROOT / 'framework'}",
                "-DFRAMEWORK_BACKEND=0",
                '-DFRAMEWORK_VENDOR_LIBRARY="/unused/compile-check.so"',
                f"-DCAMBLAS_FRAMEWORK_OPENMP={openmp}",
                f"-DCAMBLAS_FRAMEWORK_SPIN_POOL={spin}",
                f"-DCAMBLAS_FRAMEWORK_TEAM_REUSE={reuse}",
                "-DCAMBLAS_FRAMEWORK_SHARED_GRID32=1",
            ]
            if openmp:
                flags.append("-fopenmp")
            subprocess.run(
                [
                    *compiler,
                    *flags,
                    "-c",
                    ROOT / "framework/framework_blas_bridge.c",
                    "-o",
                    Path(temporary) / f"executor-{index}.o",
                ],
                check=True,
                timeout=120,
            )
    print(f"Framework bridge executor configurations passed: {len(configurations)}")


if __name__ == "__main__":
    main()
