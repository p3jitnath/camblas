#!/usr/bin/env python3
"""Build the standalone CAMBLAS CUDA backend without rebuilding PyTorch."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compile_library(command, destination):
    """Compile beside the destination and replace it after successful linking.

    Parameters
    ----------
    command : list of str
        Compiler command with its final argument naming the output library.
    destination : pathlib.Path
        Final library path.

    Notes
    -----
    Replacing the file preserves existing processes' mapped library contents.
    A failed compiler leaves the previous library intact.
    """
    with tempfile.TemporaryDirectory(
        prefix="." + destination.name + ".", dir=destination.parent
    ) as directory:
        staged = Path(directory) / destination.name
        actual = [*command[:-1], str(staged)]
        subprocess.run(actual, check=True)
        os.replace(staged, destination)


def main():
    """Compile CUDA device kernels and record compiler, source and library identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, default=os.environ.get("CUDA_HOME"))
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--architecture", default="sm_90")
    parser.add_argument("--output", type=Path, default=ROOT / "build/cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Build the optional native PyTorch tensor binding",
    )
    args = parser.parse_args()
    nvcc = args.nvcc or (args.cuda_root / "bin/nvcc" if args.cuda_root else None)
    if nvcc is None:
        found = shutil.which("nvcc")
        if not found:
            parser.error("Supply --cuda-root or --nvcc, or put nvcc on PATH")
        nvcc = Path(found)
    nvcc = nvcc.resolve()
    cuda_root = args.cuda_root or nvcc.parent.parent
    output = args.output.resolve()
    source = ROOT / "src/cuda/backend.cu"
    library = output / "libcamblas_cuda.so"
    command = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        "-arch=" + args.architecture,
        "-ccbin",
        args.cxx,
        "-Xcompiler=-fPIC,-Wall,-Wextra",
        "-shared",
        "--cudart=shared",
        "-I" + str(ROOT / "include"),
        str(source),
        "-L" + str(cuda_root / "lib64"),
        "-lcublas",
        "-lcublasLt",
        "-Xlinker=-soname",
        "-Xlinker=libcamblas_cuda.so",
        "-Xlinker=-rpath",
        "-Xlinker=" + str(cuda_root / "lib64"),
        "-o",
        str(library),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    compile_library(command, library)
    sources = [
        source,
        ROOT / "src/cuda/fusion.cuh",
        ROOT / "include/camblas_cuda.h",
        Path(__file__).resolve(),
    ]
    binding_command = None
    binding_library = None
    torch_version = None
    if args.torch:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths

        if torch.version.cuda is None:
            parser.error("--torch requires a CUDA-enabled PyTorch environment")
        binding_source = ROOT / "src/cuda/torch_bindings.cpp"
        binding_library = output / (
            "_camblas_cuda_torch" + sysconfig.get_config_var("EXT_SUFFIX")
        )
        binding_command = [
            args.cxx,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        ]
        for path in include_paths() + [
            sysconfig.get_path("include"),
            str(cuda_root / "include"),
            str(ROOT / "include"),
        ]:
            binding_command.append("-I" + path)
        for path in library_paths() + [str(output), str(cuda_root / "lib64")]:
            binding_command.extend(["-L" + path, "-Wl,-rpath," + path])
        binding_command.extend(
            [
                str(binding_source),
                "-lcamblas_cuda",
                "-ltorch_python",
                "-ltorch_cpu",
                "-ltorch",
                "-lc10",
                "-lc10_cuda",
                "-lcudart",
                "-o",
                str(binding_library),
            ]
        )
        print(shlex.join(binding_command), flush=True)
        compile_library(binding_command, binding_library)
        sources.append(binding_source)
        torch_version = torch.__version__
    record = dict(
        command=command,
        compiler=subprocess.check_output([str(nvcc), "--version"], text=True),
        host_compiler=subprocess.check_output([args.cxx, "--version"], text=True),
        architecture=args.architecture,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        cuda_root=str(cuda_root),
        tensor_binding_command=binding_command,
        tensor_binding_sha256=hashlib.sha256(binding_library.read_bytes()).hexdigest()
        if binding_library
        else None,
        torch_version=torch_version,
    )
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built {library}", flush=True)


if __name__ == "__main__":
    main()
