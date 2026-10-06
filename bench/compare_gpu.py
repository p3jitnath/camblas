#!/usr/bin/env python3
"""Compare native CAMBLAS CUDA with full-precision PyTorch, including transfers."""

import argparse
import csv
import datetime
import fcntl
import hashlib
import itertools
import json
import math
import os
import shlex
import socket
import statistics
import subprocess
from pathlib import Path

from compare import WORKLOADS
from compare_cuda import check_outputs, validate_cuda_record

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("pytorch", "camblas", "classical")
GPU_WORKLOADS = (
    *WORKLOADS,
    "square12288",
    "square16384",
    "square24576",
    "square32768",
    "attention32",
    "attention64",
    "attention256",
    "attention512",
    "attention2048",
    "attention4096",
    "attention8192",
    "attention64x1024",
    "attention1024x64",
    "attention64x32",
    "attention1024x32",
)


def digest(path):
    """Return a content identity for a source or binary artefact.

    Parameters
    ----------
    path : str or pathlib.Path
        Source or native-library file to hash.

    Returns
    -------
    str
        SHA256 hexadecimal digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_identity(library):
    """Identify a saved core and its optional sibling tensor binding.

    Parameters
    ----------
    library : pathlib.Path
        Saved CUDA core to load in separate control processes.

    Returns
    -------
    dict
        Resolved binary paths, content hashes and an optional build record.
    """
    library = Path(library).resolve(strict=True)
    bindings = sorted(library.parent.glob("_camblas_cuda_torch*.so"))
    if len(bindings) > 1:
        raise ValueError("Keep exactly one tensor binding beside the control core")
    binding = bindings[0] if bindings else None
    build = library.parent / "build.json"
    source = library.parent / "source"
    return dict(
        library=str(library),
        library_sha256=digest(library),
        tensor_binding=str(binding) if binding else None,
        tensor_binding_sha256=digest(binding) if binding else None,
        build=json.loads(build.read_text()) if build.is_file() else None,
        source_sha256={
            str(path.relative_to(source)): digest(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        },
    )


def validate_native_libraries(record, identity, *, control=False):
    """Reject missing, mismatched or incorrectly routed native binaries.

    Parameters
    ----------
    record : dict
        Worker's observed loaded-library hashes.
    identity : dict
        Expected CUDA core and tensor-binding identities.
    control : bool, optional
        Require the exact saved paths in addition to content hashes.
    """
    hashes = record["library_sha256"]
    for name, expected in (
        ("libcamblas_cuda", identity["library_sha256"]),
        ("_camblas_cuda_torch", identity["tensor_binding_sha256"]),
    ):
        actual = [sha for path, sha in hashes.items() if name in Path(path).name]
        if expected is not None and actual != [expected]:
            raise ValueError(f"Unexpected native binary: {name}")
    if control:
        for field in ("library", "tensor_binding"):
            path = identity[field]
            if path and hashes.get(path) != identity[field + "_sha256"]:
                raise ValueError(f"Control binary was not loaded: {path}")


def report(directory, rows, manifest):
    """Write medians, ranges and speed ratios without omitting regressions.

    Parameters
    ----------
    directory : pathlib.Path
        Output directory for the Markdown comparison.
    rows : list of dict
        Validated workload records with per-backend medians and ranges.
    manifest : dict
        Hardware, precision, timing and control settings for these records.
    """
    lines = [
        "# CAMBLAS CUDA versus PyTorch CUDA",
        "",
        f"{manifest['hostname']}, {manifest['timestamp']}; one GPU, full FP32/FP64, "
        f"TF32 disabled, {manifest['rounds']} fresh processes per backend/case.",
        "",
        f"PyTorch expression: {manifest.get('pytorch_baseline', 'eager')}. "
        "Both routes use the same expressions and physical input layouts. "
        "The fused expression uses standard linear and scaled-dot-product attention; "
        "compiled modes apply torch.compile to both routes. Compilation is excluded, "
        "while execution, input staging and graph replay remain timed.",
        "",
        "Speedup = PyTorch time / CAMBLAS time; values above 1 favour CAMBLAS.",
        f"CPU output allocation: {manifest.get('output_memory', 'pageable')}. "
        "Fresh allocation remains inside each timed call for every backend. "
        "Prefault includes a full CPU zero-fill; pinned uses PyTorch's warmed host allocator.",
        f"Candidate algorithm: {manifest.get('camblas_algorithm', 'auto')}. "
        "Resident calls include tensor allocation, dispatch and stream completion. "
        "Transfer calls also copy every input, weight, output and gradient. "
        "The classical control selects cuBLAS behind the same PyTorch interface.",
        "",
        "| Workload | Precision | PyTorch resident ms | CAMBLAS resident ms | Speedup | PyTorch transfers ms | CAMBLAS transfers ms | Speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        medians = row["medians_ms"]
        lines.append(
            f"| {row['workload']} | {row['dtype']} | {medians['pytorch_resident']:.4f} | {medians['camblas_resident']:.4f} | {row['speedup_resident']:.3f}× | {medians['pytorch_transfer']:.4f} | {medians['camblas_transfer']:.4f} | {row['speedup_transfer']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "Ranges are the minimum and maximum fresh-process medians; all are retained in summary.json and comparison.csv.",
            "",
            "| Workload | Precision | PyTorch resident range ms | CAMBLAS resident range ms | PyTorch transfer range ms | CAMBLAS transfer range ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ranges = row["ranges_ms"]
        values = [
            f"{ranges[key][0]:.4f}–{ranges[key][1]:.4f}"
            for key in (
                "pytorch_resident",
                "camblas_resident",
                "pytorch_transfer",
                "camblas_transfer",
            )
        ]
        lines.append(
            f"| {row['workload']} | {row['dtype']} | " + " | ".join(values) + " |"
        )
    for dtype in ("float32", "float64"):
        selected = [r for r in rows if r["dtype"] == dtype]
        if not selected:
            continue
        for mode in ("resident", "transfer"):
            ratios = [r["speedup_" + mode] for r in selected]
            geomean = math.exp(statistics.mean(math.log(value) for value in ratios))
            wins = sum(value > 1 for value in ratios)
            lines.extend(
                [
                    "",
                    f"{dtype}, {mode}: geometric-mean speedup {geomean:.3f}×; {wins}/{len(ratios)} median wins.",
                ]
            )
    if manifest.get("control"):
        lines.extend(
            [
                "",
                "## Saved native control",
                "",
                "The saved core and sibling binding run with automatic dispatch in "
                "separate fresh processes, using the same current Python adapter. "
                "Loaded binary paths and hashes are checked on every process. "
                "Speedup = saved control / candidate; values below 1 report regressions.",
                "",
                "| Workload | Precision | Control resident ms | Candidate resident ms | Speedup | Control transfers ms | Candidate transfers ms | Speedup |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            value = row["medians_ms"]
            lines.append(
                f"| {row['workload']} | {row['dtype']} | {value['control_resident']:.4f} | {value['camblas_resident']:.4f} | {row['speedup_control_resident']:.3f}× | {value['control_transfer']:.4f} | {value['camblas_transfer']:.4f} | {row['speedup_control_transfer']:.3f}× |"
            )
    if manifest.get("coherent_host"):
        lines.extend(
            [
                "",
                "## Separate coherent-host experiment",
                "",
                "All pipelines start with the same pageable CPU inputs and finish with CPU outputs. "
                "PyTorch and CAMBLAS copy pipelines explicitly copy every operand and result. "
                "The coherent pipeline lets GPU kernels access the original CPU allocations through "
                "host page tables and synchronises before returning CPU output. Hardware memory traffic "
                "is included, but no bulk tensor copies are issued. Hidden MLP and attention storage "
                "is on the GPU. Inputs are unchanged across timed calls. There is no host autograd.",
                "",
                "| Workload | Precision | PyTorch copies ms | CAMBLAS copies ms | CAMBLAS coherent ms | Coherent range ms | Coherent speedup |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            value = row["medians_ms"]
            low, high = row["ranges_ms"]["coherent_host"]
            lines.append(
                f"| {row['workload']} | {row['dtype']} | {value['pytorch_transfer']:.4f} | {value['camblas_transfer']:.4f} | {value['coherent_host']:.4f} | {low:.4f}–{high:.4f} | {row['speedup_host']:.3f}× |"
            )
        for dtype in ("float32", "float64"):
            ratios = [row["speedup_host"] for row in rows if row["dtype"] == dtype]
            if ratios:
                geomean = math.exp(statistics.mean(math.log(value) for value in ratios))
                lines.extend(
                    [
                        "",
                        f"{dtype}: coherent-host geometric-mean speedup {geomean:.3f}×.",
                    ]
                )
    lines.extend(
        [
            "",
            "Strassen changes summation order and may worsen relative error under cancellation. "
            "Its range check selects classical GEMM for unsafe inputs. Native "
            "PyTorch expressions retain FP32/FP64 compute "
            "types. No TF32 or reduced-precision inputs are used. "
            + (
                "Both routes use the selected PyTorch compilation mode. "
                + (
                    "The selected compiler mode may use internal CUDA graphs."
                    if (manifest.get("compile_mode") or "reduce-overhead")
                    in ("reduce-overhead", "max-autotune")
                    else "The selected compiler mode runs without CUDA graphs."
                )
                if manifest.get("pytorch_baseline", "eager").startswith("compiled")
                else "No CUDA graphs or torch.compile are used in this comparison."
            ),
            "",
            "Commands, binary/source identities, node-sharing checks, raw samples, gradients and scalar-dot oracles accompany this report.",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    flat = []
    for row in rows:
        value = {
            key: row[key]
            for key in (
                "workload",
                "dtype",
                "speedup_resident",
                "speedup_transfer",
                "max_scaled_sample_error",
            )
        }
        value.update(row["medians_ms"])
        if "speedup_host" in row:
            value["speedup_host"] = row["speedup_host"]
        for mode in ("resident", "transfer"):
            key = "speedup_control_" + mode
            if key in row:
                value[key] = row[key]
        for key, bounds in row["ranges_ms"].items():
            value[key + "_min"] = bounds[0]
            value[key + "_max"] = bounds[1]
        flat.append(value)
    with (directory / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main():
    """Run serial rotated measurements with locked provenance and correctness checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--output-memory",
        choices=["pageable", "prefault", "pinned"],
        default="pageable",
        help="CPU output allocation policy applied identically to every backend",
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=GPU_WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".frameworks/envs/cuda/bin/python"
    )
    parser.add_argument(
        "--camblas-library", type=Path, default=ROOT / "build/cuda/libcamblas_cuda.so"
    )
    parser.add_argument(
        "--camblas-algorithm",
        choices=[
            "auto",
            "classical",
            "symmetric",
            "lt",
            "strassen",
            "strassen2",
            "strassen3",
            "strassen4",
        ],
        default="auto",
        help="Select the candidate algorithm; saved controls retain automatic dispatch",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--control-library",
        type=Path,
        help="Add an unchanged saved CUDA core and its sibling binding as an automatic-dispatch control",
    )
    parser.add_argument(
        "--pytorch-baseline",
        choices=["eager", "fused", "compiled", "compiled-fused"],
        default="eager",
        help="Select eager expressions or linear/SDPA APIs, optionally compiled",
    )
    parser.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="reduce-overhead",
        help="PyTorch compiler mode, excluding compilation from timing",
    )
    args = parser.parse_args()
    policy_overrides = (
        "LD_PRELOAD",
        "OMP_WAIT_POLICY",
        "GOMP_SPINCOUNT",
        "MALLOC_TRIM_THRESHOLD_",
        "MALLOC_MMAP_THRESHOLD_",
        "MALLOC_ARENA_MAX",
        "GLIBC_TUNABLES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUBLAS_EMULATION_STRATEGY",
        "NVIDIA_TF32_OVERRIDE",
    )
    if any(os.environ.get(name) for name in policy_overrides):
        parser.error(
            "Unset preload, allocator, wait-policy and CUDA compute overrides before default comparisons"
        )
    backends = BACKENDS
    if args.control_library:
        backends = (*backends, "control")
    if args.rounds < 3:
        parser.error("Use at least three fresh-process rounds")
    cpus = sorted(os.sched_getaffinity(0))[: args.threads]
    if len(cpus) != args.threads:
        parser.error("Requested host cores are not available")
    directory = args.output.resolve()
    library = args.camblas_library.resolve()
    manifest = dict(
        camblas_algorithm=args.camblas_algorithm,
        output_memory=args.output_memory,
        transfer_entry="python",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        command=os.sys.argv,
        rounds=args.rounds,
        threads=args.threads,
        affinity=cpus,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        backends=list(backends),
        coherent_host=False,
        pytorch_baseline=args.pytorch_baseline,
        compile_mode=args.compile_mode
        if args.pytorch_baseline.startswith("compiled")
        else None,
        native_library=str(library),
        build=json.loads((library.parent / "build.json").read_text()),
        control=control_identity(args.control_library)
        if args.control_library
        else None,
    )
    for name, expected in manifest["build"]["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Rebuild CUDA after changing {name}")
    if manifest["slurm_job_id"]:
        manifest["allocation_before"] = subprocess.check_output(
            ["scontrol", "show", "job", "-o", manifest["slurm_job_id"]], text=True
        )
    manifest["environment"] = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_TF32_OVERRIDE",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_EMULATION_STRATEGY",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            *policy_overrides,
        )
    }
    sources = [
        ROOT / "bench/cuda_workload.py",
        ROOT / "bench/compare_cuda.py",
        ROOT / "bench/compare.py",
        Path(__file__).resolve(),
        ROOT / "scripts/build_cuda.py",
        ROOT / "include/camblas_cuda.h",
        ROOT / "tests/test_cuda.py",
        ROOT / "tests/cuda_fail_alloc.c",
        ROOT / "tests/test_tools.py",
        *sorted((ROOT / "src/cuda").glob("*")),
        *sorted((ROOT / "camblas").glob("*.py")),
    ]
    sources = [p for p in sources if p.is_file()]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    manifest["gpu_before"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,utilization.gpu,clocks.sm,temperature.gpu",
            "--format=csv",
        ],
        text=True,
    )
    manifest["processes_before"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv",
        ],
        text=True,
    )
    with (ROOT / ".frameworks/measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory.mkdir(parents=True, exist_ok=False)
        for source in sources:
            target = directory / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        if manifest["control"]:
            control_source = Path(manifest["control"]["library"]).parent / "source"
            for name in manifest["control"]["source_sha256"]:
                target = directory / "control_source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((control_source / name).read_bytes())
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        rows = []
        for workload, dtype in itertools.product(args.workloads, args.dtypes):
            repetitions = args.repetitions or (
                201 if workload in ("mlp", "backward") else 21
            )
            observations = {backend: [] for backend in backends}
            error = 0.0
            identities = set()
            vendor_identities = set()
            for round_id in range(args.rounds):
                shift = round_id % len(backends)
                for backend in backends[shift:] + backends[:shift]:
                    key = f"{workload}_{dtype}_r{round_id + 1}_{backend}"
                    output = directory / (key + ".json")
                    engine = "pytorch" if backend == "pytorch" else "camblas"
                    algorithm = (
                        "classical"
                        if backend == "classical"
                        else args.camblas_algorithm
                        if backend in ("camblas", "coherent")
                        else "auto"
                    )
                    command = [
                        str(args.python),
                        str(ROOT / "bench/cuda_workload.py"),
                        "--workload",
                        workload,
                        "--dtype",
                        dtype,
                        "--threads",
                        str(args.threads),
                        "--repetitions",
                        str(repetitions),
                        "--warmups",
                        "10",
                        "--engine",
                        engine,
                        "--pytorch-baseline",
                        args.pytorch_baseline,
                        "--compile-mode",
                        args.compile_mode,
                        "--algorithm",
                        algorithm,
                        "--output-memory",
                        args.output_memory,
                        "--output",
                        str(output),
                    ]
                    if round_id % 2:
                        command.append("--transfer-first")
                    environment = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=os.environ.get(
                            "CUDA_VISIBLE_DEVICES", "0"
                        ).split(",")[0],
                        OMP_NUM_THREADS=str(args.threads),
                        OPENBLAS_NUM_THREADS=str(args.threads),
                        MKL_NUM_THREADS=str(args.threads),
                    )
                    environment["CAMBLAS_ENABLE"] = "1" if engine == "camblas" else "0"
                    environment["CAMBLAS_CUDA_ALGORITHM"] = algorithm
                    environment.pop("CAMBLAS_CUDA_LIBRARY", None)
                    if backend == "control":
                        environment["CAMBLAS_CUDA_LIBRARY"] = manifest["control"][
                            "library"
                        ]
                    elif engine == "camblas":
                        environment["CAMBLAS_CUDA_LIBRARY"] = str(library)
                    pinned = [
                        "taskset",
                        "-c",
                        ",".join(str(cpu) for cpu in cpus),
                        *command,
                    ]
                    (directory / (key + ".command.txt")).write_text(
                        shlex.join(pinned) + "\n"
                    )
                    print(key, flush=True)
                    with (directory / (key + ".log")).open("w") as log:
                        subprocess.run(
                            pinned,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            check=True,
                        )
                    record = json.loads(output.read_text())
                    validate_cuda_record(
                        dict(record, backend="cuda"),
                        cpus,
                        repetitions,
                        modes=("host",)
                        if backend == "coherent"
                        else ("resident", "transfer"),
                    )
                    if record.get("memory_mode") != (
                        "coherent_host" if backend == "coherent" else "explicit_copy"
                    ):
                        raise ValueError("Unexpected memory pipeline")
                    if record.get("engine") != engine or record.get("algorithm") != (
                        algorithm if engine == "camblas" else "pytorch"
                    ):
                        raise ValueError("Unexpected measured implementation")
                    if record.get("pytorch_baseline") != args.pytorch_baseline:
                        raise ValueError("Unexpected PyTorch baseline")
                    if record.get("compile_mode") != (
                        args.compile_mode
                        if args.pytorch_baseline.startswith("compiled")
                        else None
                    ):
                        raise ValueError("Unexpected PyTorch compiler mode")
                    if engine == "camblas":
                        if not any("libcamblas_cuda" in p for p in record["libraries"]):
                            raise ValueError(
                                "The native CAMBLAS library was not loaded"
                            )
                        if not any(record["camblas_cuda_stats"].values()):
                            raise ValueError("No native CAMBLAS launch was observed")
                        build = (
                            manifest["control"]
                            if backend == "control"
                            else manifest["build"]
                        )
                        validate_native_libraries(
                            record, build, control=backend == "control"
                        )
                    identities.add(tuple(record["input_sha256"]))
                    vendor_identities.add(
                        tuple(
                            sorted(
                                (Path(path).name, sha)
                                for path, sha in record["library_sha256"].items()
                                if "libcublas" in path
                            )
                        )
                    )
                    observations[backend].append(record)
            if len(identities) != 1:
                raise ValueError("Inputs changed across backends or rounds")
            if len(vendor_identities) != 1:
                raise ValueError("cuBLAS libraries changed across backends or rounds")
            for backend, records in observations.items():
                for index, record in enumerate(records):
                    reference = observations["pytorch"][index]
                    for mode in (
                        ("host",) if backend == "coherent" else ("resident", "transfer")
                    ):
                        error = max(
                            error,
                            check_outputs(
                                record["measurements"][mode]["outputs"],
                                reference["measurements"][
                                    "transfer" if mode == "host" else mode
                                ]["outputs"],
                                dtype,
                            ),
                        )
            samples = {
                backend + "_" + mode: [
                    statistics.median(r["measurements"][mode]["seconds"]) * 1000
                    for r in records
                ]
                for backend, records in observations.items()
                for mode in (
                    ("host",) if backend == "coherent" else ("resident", "transfer")
                )
            }
            medians = {
                key: statistics.median(values) for key, values in samples.items()
            }
            row = dict(
                workload=workload,
                dtype=dtype,
                medians_ms=medians,
                ranges_ms={
                    key: [min(values), max(values)] for key, values in samples.items()
                },
                round_medians_ms=samples,
                max_scaled_sample_error=error,
                speedup_resident=medians["pytorch_resident"]
                / medians["camblas_resident"],
                speedup_transfer=medians["pytorch_transfer"]
                / medians["camblas_transfer"],
            )
            rows.append(row)
            if manifest["control"]:
                for mode in ("resident", "transfer"):
                    row["speedup_control_" + mode] = (
                        medians["control_" + mode] / medians["camblas_" + mode]
                    )
            (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                f"{workload} {dtype}: resident {row['speedup_resident']:.3f}x, transfers {row['speedup_transfer']:.3f}x",
                flush=True,
            )
            report(directory, rows, manifest)
        for source in sources:
            if (
                digest(source)
                != manifest["source_sha256"][str(source.relative_to(ROOT))]
            ):
                raise ValueError(f"Source changed during measurement: {source}")
        if digest(library) != manifest["build"]["library_sha256"]:
            raise ValueError("Native binary changed during measurement")
        binding_command = manifest["build"].get("tensor_binding_command")
        if (
            binding_command
            and digest(binding_command[-1])
            != manifest["build"]["tensor_binding_sha256"]
        ):
            raise ValueError("Tensor binding changed during measurement")
        if manifest["control"]:
            if control_identity(args.control_library) != manifest["control"]:
                raise ValueError("Saved control changed during measurement")
        manifest["gpu_after"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,utilization.gpu,clocks.sm,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        )
        manifest["processes_after"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv",
            ],
            text=True,
        )
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "complete.json").write_text(
            json.dumps(dict(status="passed", cases=len(rows))) + "\n"
        )


if __name__ == "__main__":
    main()
