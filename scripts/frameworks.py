#!/usr/bin/env python3
"""Prepare, compile and install project-local NumPy/PyTorch variants."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "numpy": (
        "https://github.com/numpy/numpy.git",
        "v2.3.2",
        "bc5e4f811db9487a9ea1618ffb77a33b3919bb8e",
    ),
    "pytorch": (
        "https://github.com/pytorch/pytorch.git",
        "v2.8.0",
        "ba56102387ef21a3b04b357e5b183d48f0afefc7",
    ),
}


def main():
    """Run one project-local framework preparation, build or installation step.

    Notes
    -----
    Source revisions and dependency versions are pinned for reproducibility.
    Each backend has separate build, wheel and runtime directories. Dry runs
    print commands without changing files or starting downloads or builds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "numpy", "pytorch", "install"))
    parser.add_argument(
        "--backend", choices=("camblas", "openblas", "nvpl"), default="camblas"
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc-14"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++-14"))
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran-14"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without writing files or downloading/building anything",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Use Python 3.11 or later (3.11 was used for the reported builds)")
    if args.jobs < 1 or args.jobs > len(os.sched_getaffinity(0)):
        parser.error("--jobs must fit the current CPU affinity/allocation")
    base = ROOT / ".frameworks"
    python = base / "build-env/bin/python"
    prefix = base / "prefix" / args.backend
    wheel_dir = base / "wheels" / args.backend
    commands = []

    def run(command, cwd=None, env=None):
        """Record a command and execute it unless dry-run mode is selected.

        Parameters
        ----------
        command : sequence of str or pathlib.Path
            Executable and arguments, passed directly without a shell.
        cwd : pathlib.Path or None, optional
            Working directory; ``None`` preserves the current directory.
        env : dict of str to str or None, optional
            Complete child environment, or ``None`` to inherit it.

        Raises
        ------
        subprocess.CalledProcessError
            If an executed command returns a non-zero status.
        """
        command = [str(x) for x in command]
        commands.append({"command": command, "cwd": str(cwd or ROOT)})
        print(
            (f"cd {shlex.quote(str(cwd))}: " if cwd else "") + shlex.join(command),
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(command, cwd=cwd, env=env, check=True)

    def verify_source(name):
        """Locate a source checkout and verify its pinned revision.

        Parameters
        ----------
        name : {'numpy', 'pytorch'}
            Framework whose checkout is required.

        Returns
        -------
        pathlib.Path
            Project-local checkout path, without verification during dry runs.

        Raises
        ------
        RuntimeError
            If the checkout revision differs from the configured source pin.
        """
        source = base / "src" / name
        if not args.dry_run:
            revision = subprocess.check_output(
                ["git", "-C", source, "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != SOURCES[name][2]:
                raise RuntimeError(
                    f"{name}: expected {SOURCES[name][2]}, found {revision}; refusing to alter this checkout"
                )
        return source

    if args.action == "prepare":
        if not args.dry_run:
            (base / "src").mkdir(parents=True, exist_ok=True)
        for name, (url, tag, _) in SOURCES.items():
            source = base / "src" / name
            if not source.exists():
                run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        tag,
                        "--depth",
                        "1",
                        "--recursive",
                        url,
                        source,
                    ]
                )
            verify_source(name)
            run(
                [
                    "git",
                    "-C",
                    source,
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--jobs",
                    args.jobs,
                ]
            )
        patch = ROOT / "patches/pytorch-2.8.0.patch"
        source = base / "src/pytorch"
        if args.dry_run:
            run(["git", "-C", source, "apply", "--check", patch])
            run(["git", "-C", source, "apply", patch])
        else:
            applied = (
                subprocess.run(
                    ["git", "-C", source, "apply", "--reverse", "--check", patch],
                    capture_output=True,
                ).returncode
                == 0
            )
            if not applied:
                run(["git", "-C", source, "apply", "--check", patch])
                run(["git", "-C", source, "apply", patch])
        if not python.exists():
            run([sys.executable, "-m", "venv", base / "build-env"])
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-r",
                ROOT / "configs/build-requirements.txt",
            ]
        )
        return
    if not args.dry_run and not (prefix / "bridge.json").is_file():
        parser.error(f"Build the {args.backend} adapter first with scripts/build.py")
    if args.action == "install":
        runtime = base / "envs" / args.backend
        runtime_python = runtime / "bin/python"
        if not runtime_python.exists():
            run([sys.executable, "-m", "venv", runtime])
        run(
            [
                runtime_python,
                "-m",
                "pip",
                "install",
                "-r",
                ROOT / "configs/runtime-requirements.txt",
            ]
        )
        for name, version in (("numpy", "2.3.2"), ("torch", "2.8.0")):
            wheels = sorted(wheel_dir.glob(f"{name}-{version}-*.whl"))
            if args.dry_run and not wheels:
                wheels = [wheel_dir / f"{name}-{version}-<python>-<abi>-<platform>.whl"]
            if len(wheels) != 1:
                parser.error(
                    f"Expected exactly one {name} wheel in {wheel_dir}, found {len(wheels)}"
                )
            run(
                [
                    runtime_python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    wheels[0],
                ]
            )
        return
    source = verify_source(args.action)
    if not args.dry_run:
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (base / "build").mkdir(exist_ok=True)
    settings = dict(
        CC=args.cc,
        CXX=args.cxx,
        FC=args.fc,
        CAMBLAS_FRAMEWORK_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CAMBLAS_FRAMEWORK_TRACE="0",
    )
    if args.action == "numpy":
        settings.update(
            CFLAGS="-O3 -mcpu=neoverse-v2",
            CXXFLAGS="-O3 -mcpu=neoverse-v2",
            PKG_CONFIG_PATH=str(prefix / "lib/pkgconfig"),
        )
        command = [
            python,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            wheel_dir,
            "-Csetup-args=-Dblas=framework_blas",
            "-Csetup-args=-Dlapack=framework_lapack",
            "-Csetup-args=-Dallow-noblas=false",
            f"-Ccompile-args=-j{args.jobs}",
            f"-Cbuild-dir={base / 'build' / ('numpy-' + args.backend)}",
        ]
    else:
        # Disable alternate matrix engines uniformly so that the selected
        # BLAS bridge receives the compared work. Grace uses 128-bit SVE;
        # the pinned PyTorch patch lets us omit its SVE256 translation units.
        settings.update(
            CMAKE_C_FLAGS="-O3 -mcpu=neoverse-v2",
            CMAKE_CXX_FLAGS="-O3 -mcpu=neoverse-v2",
            CMAKE_BUILD_TYPE="Release",
            CMAKE_PREFIX_PATH=str(prefix),
            CMAKE_LIBRARY_PATH=str(prefix / "lib"),
            CMAKE_BUILD_RPATH=str(prefix / "lib"),
            CMAKE_INSTALL_RPATH=str(prefix / "lib"),
            CAMBLAS_FRAMEWORK_BUILD_DIR=str(base / "build" / ("torch-" + args.backend)),
            BLAS="Generic",
            GENERIC_BLAS_LIBRARIES="framework_blas",
            USE_CUDA="0",
            USE_ROCM="0",
            USE_XPU="0",
            USE_MKLDNN="0",
            USE_KLEIDIAI="0",
            USE_XNNPACK="0",
            USE_NNPACK="0",
            USE_PYTORCH_QNNPACK="0",
            USE_DISTRIBUTED="0",
            USE_KINETO="0",
            BUILD_TEST="0",
            USE_NUMPY="1",
            USE_OPENMP="1",
            USE_SVE256="0",
            MAX_JOBS=str(args.jobs),
            CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
            PYTORCH_BUILD_VERSION="2.8.0",
            PYTORCH_BUILD_NUMBER="1",
        )
        if not args.dry_run:
            subprocess.run(
                [
                    "git",
                    "-C",
                    source,
                    "apply",
                    "--reverse",
                    "--check",
                    ROOT / "patches/pytorch-2.8.0.patch",
                ],
                check=True,
            )
        command = [python, "setup.py", "bdist_wheel", "--dist-dir", wheel_dir]
    env = dict(os.environ, **settings)
    env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
    print("Build settings: " + json.dumps(settings, sort_keys=True), flush=True)
    run(command, cwd=source, env=env)
    if not args.dry_run:
        record = {
            "commands": commands,
            "settings": settings,
            "source_commit": SOURCES[args.action][2],
            "build_dependencies": subprocess.check_output(
                [python, "-m", "pip", "freeze"], text=True
            ),
            "submodules": subprocess.check_output(
                ["git", "-C", source, "submodule", "status", "--recursive"], text=True
            ),
            "bridge": json.loads((prefix / "bridge.json").read_text()),
            "wheels": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in wheel_dir.glob("*.whl")
            },
        }
        (wheel_dir / f"{args.action}-build.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
