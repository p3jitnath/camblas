#!/usr/bin/env python3
"""Build the reference core or the selected Grace core and framework adapters."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ("scalar", "cblas_shim", "topology", "executor", "packing", "packed", "planner")


def main():
    """Build the requested libraries and record their exact compiler commands.

    Notes
    -----
    Command-line arguments select the target and dependency paths. Generated
    libraries stay in project-local directories; dependency installations are
    never modified. The Grace configuration supplies per-file compiler flags
    so that each kernel retains its measured dispatch and arithmetic policy.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("reference", "grace"), default="reference")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--backend", choices=("camblas", "openblas", "nvpl", "all"))
    parser.add_argument("--prefix-root", type=Path, default=ROOT / ".frameworks/prefix")
    parser.add_argument(
        "--openblas-lib",
        type=Path,
        help="Exact LP64 shared library; also supplies CAMBLAS compatibility",
    )
    parser.add_argument(
        "--openblas-include",
        type=Path,
        help="Directory containing cblas.h and openblas_config.h",
    )
    parser.add_argument(
        "--nvpl-blas", type=Path, help="Exact NVPL LP64 GNU OpenMP BLAS shared library"
    )
    parser.add_argument(
        "--nvpl-lapack",
        type=Path,
        help="Exact NVPL LP64 GNU OpenMP LAPACK shared library",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run bounded core correctness checks after building",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Use Python 3.11 or later")
    if args.target == "grace" and platform.machine() not in ("aarch64", "arm64"):
        parser.error(
            "The Grace build requires an AArch64 host; use --target reference elsewhere"
        )
    if args.backend and args.target != "grace":
        parser.error("Framework adapters currently require --target grace")
    if args.test and args.target != "reference":
        parser.error(
            "--test is for --target reference; use tests/test_framework_bridge.py for Grace"
        )
    backends = (
        ("camblas", "openblas", "nvpl")
        if args.backend == "all"
        else ((args.backend,) if args.backend else ())
    )
    if backends:
        required = ["openblas_lib", "openblas_include"]
        if "nvpl" in backends:
            required += ["nvpl_blas", "nvpl_lapack"]
        for name in required:
            path = getattr(args, name)
            if path is None or not path.exists():
                parser.error(
                    f"--{name.replace('_', '-')} must name an existing dependency"
                )
        for name in ("cblas.h", "openblas_config.h"):
            if not (args.openblas_include / name).is_file():
                parser.error(f"Missing {args.openblas_include / name}")
    cc = shlex.split(args.cc)
    build = ROOT / "build" / args.target
    build.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, **kwargs):
        """Record and execute one build command, stopping on failure.

        Parameters
        ----------
        command : sequence of str or pathlib.Path
            Executable and arguments, passed directly without a shell.
        **kwargs : dict
            Additional keyword arguments for ``subprocess.run``.

        Raises
        ------
        subprocess.CalledProcessError
            If compilation, linking or a correctness check fails.
        """
        commands.append([str(x) for x in command])
        print(shlex.join(commands[-1]), flush=True)
        subprocess.run(commands[-1], check=True, **kwargs)

    config = json.loads((ROOT / "configs/grace.json").read_text())
    units = list(CORE) + (
        [
            "sve",
            "deep_amicro",
            "deep_amicro64",
            "rectangular64",
            "rectangular32",
            "batch_amicro64",
        ]
        if args.target == "grace"
        else []
    )
    includes = [f"-I{ROOT / 'include'}", f"-I{ROOT / 'src'}"]
    objects = []
    for unit in units:
        flags = (
            config["units"][unit]
            if args.target == "grace"
            else ["-O3", "-std=gnu11", "-Wall", "-Wextra", "-Wpedantic", "-fPIC"]
        )
        obj = build / f"{unit}.o"
        run(cc + flags + includes + ["-c", ROOT / "src" / f"{unit}.c", "-o", obj])
        objects.append(obj)
    name = "camblas_sve_nr4" if args.target == "grace" else "camblas"
    library = build / f"lib{name}.so"
    run(
        cc
        + [
            "-shared",
            *objects,
            "-Wl,-z,defs",
            "-pthread",
            "-lm",
            f"-Wl,-soname,{library.name}",
            "-o",
            library,
        ]
    )
    for backend in backends:
        prefix = args.prefix_root.resolve() / backend
        libdir, incdir = prefix / "lib", prefix / "include"
        pcdir = libdir / "pkgconfig"
        for directory in (libdir, incdir, pcdir):
            directory.mkdir(parents=True, exist_ok=True)
        for header in ("cblas.h", "openblas_config.h"):
            shutil.copy2(args.openblas_include / header, incdir / header)
        vendor = (args.nvpl_blas if backend == "nvpl" else args.openblas_lib).absolute()
        lapack = (
            args.nvpl_lapack if backend == "nvpl" else args.openblas_lib
        ).absolute()
        # NumPy and PyTorch need LAPACK as well as BLAS. Keep that dependency
        # explicit, and refuse to replace an unrelated library or symlink.
        lapack_link = libdir / "liblapack.so"
        if lapack_link.is_symlink() and lapack_link.resolve() != lapack.resolve():
            raise RuntimeError(f"Refusing to replace a different LAPACK: {lapack_link}")
        if not lapack_link.exists():
            lapack_link.symlink_to(lapack)
        elif not lapack_link.is_symlink():
            raise RuntimeError(f"Refusing to replace {lapack_link}")
        backend_id = {"camblas": 0, "openblas": 1, "nvpl": 2}[backend]
        extra = []
        if backend == "camblas":
            shutil.copy2(library, libdir / library.name)
            symmetric = build / "symmetric.o"
            run(
                cc
                + config["symmetric"]
                + [
                    f"-I{ROOT / 'include'}",
                    "-c",
                    ROOT / "src/symmetric/symmetric.c",
                    "-o",
                    symmetric,
                ]
            )
            flags = config["bridge"]
            extra = [symmetric, f"-L{libdir}", "-lcamblas_sve_nr4"]
        else:
            flags = [
                "-O3",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-mcpu=neoverse-v2",
                "-fPIC",
                f"-DFRAMEWORK_BACKEND={backend_id}",
                "-DCAMBLAS_BENCH_MEMORY_POLICY=1",
            ]
        bridge = libdir / "libframework_blas.so"
        run(
            cc
            + flags
            + [
                f'-DFRAMEWORK_VENDOR_LIBRARY="{vendor}"',
                *includes,
                f"-I{ROOT / 'framework'}",
                f"-I{ROOT / 'src/symmetric'}",
                "-shared",
                ROOT / "framework/framework_blas_bridge.c",
                ROOT / "src/pthread_pool.c",
                *extra,
                "-Wl,--no-as-needed",
                vendor,
                "-Wl,--as-needed",
                "-Wl,-soname,libframework_blas.so",
                "-Wl,-rpath,$ORIGIN",
                f"-Wl,-rpath,{vendor.parent}",
                "-Wl,-z,defs",
                "-pthread",
                "-fopenmp",
                "-ldl",
                "-lm",
                "-o",
                bridge,
            ]
        )
        for package, libs in (
            ("framework_blas", f"-L{libdir} -lframework_blas {vendor}"),
            ("framework_lapack", f"{lapack} {vendor}"),
        ):
            (pcdir / f"{package}.pc").write_text(
                f"Name: {package}\nDescription: {backend} LP64 framework adapter\nVersion: 0.1.0\n"
                f"Libs: {libs} -Wl,-rpath,{libdir} -Wl,-rpath,{vendor.parent} -Wl,-rpath,{lapack.parent}\n"
                f"Cflags: -I{incdir}\n"
            )
        (prefix / "bridge.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "bridge": str(bridge),
                    "vendor": str(vendor),
                    "lapack": str(lapack),
                    "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
                    "core_sha256": hashlib.sha256(library.read_bytes()).hexdigest()
                    if backend == "camblas"
                    else None,
                },
                indent=2,
            )
            + "\n"
        )
    if args.test:
        # Reference-policy tests must use the reference target: planner assertions
        # intentionally describe that configuration rather than Grace tuning.
        names = (
            "packing",
            "amicro_transpose",
            "packed_gemm",
            "cblas_semantics",
            "counter_contract",
            "external_input_contract",
            "topology",
            "planner",
            "guardpage",
        )
        for test in names:
            executable = build / f"{test}_test"
            extra_objects = [build / "packing.o"] if test == "packing" else []
            run(
                cc
                + [
                    "-O2",
                    "-std=gnu11",
                    *includes,
                    f"-I{ROOT / 'tests'}",
                    ROOT / "tests" / f"{test}_test.c",
                    *extra_objects,
                    f"-L{build}",
                    "-lcamblas",
                    f"-Wl,-rpath,{build}",
                    "-Wl,-E",
                    "-pthread",
                    "-ldl",
                    "-lm",
                    "-o",
                    executable,
                ]
            )
            run([executable])
    (build / "build.json").write_text(
        json.dumps(
            {
                "compiler": subprocess.check_output(
                    cc + ["--version"], text=True
                ).splitlines()[0],
                "target": args.target,
                "commands": commands,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "source_sha256": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for directory in ("src", "include", "framework", "configs")
                    for p in sorted((ROOT / directory).rglob("*"))
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built {library}")


if __name__ == "__main__":
    main()
