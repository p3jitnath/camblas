#!/usr/bin/env python3
"""Compare CAMBLAS and NVPL CPU PyTorch with one CUDA GPU on matching workloads."""

import argparse
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

from compare import WORKLOADS, validate_process_record

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("camblas", "nvpl", "cuda")


def check_outputs(result, reference, dtype):
    """Compare deterministic output and gradient samples and global norms.

    Parameters
    ----------
    result, reference : list of dict
        Numerical signatures from matching eager operations.
    dtype : str
        Precision determining the existing application comparison tolerance.

    Returns
    -------
    float
        Largest sample discrepancy divided by the reference maximum magnitude.

    Raises
    ------
    ValueError
        If outputs, gradients or norms disagree or contain non-finite values.
    """
    tolerance = 3e-4 if dtype == "float32" else 3e-11
    if len(result) != len(reference):
        raise ValueError("Output/gradient count differs from the CPU reference")
    largest = 0.0
    for actual, expected in zip(result, reference):
        if actual["size"] != expected["size"] or len(actual["samples"]) != len(
            expected["samples"]
        ):
            raise ValueError("Output/gradient dimensions differ")
        values = [actual["norm"], actual["max_abs"], *actual["samples"]]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite output/gradient signature")
        error = max(
            abs(x - y) for x, y in zip(actual["samples"], expected["samples"])
        ) / max(expected["max_abs"], 1e-15)
        if error >= tolerance or abs(actual["norm"] - expected["norm"]) > (
            tolerance * max(expected["norm"], 1e-15)
        ):
            raise ValueError(f"Output/gradient mismatch: {error:.3g}")
        largest = max(largest, error)
    return largest


def validate_cuda_record(record, cpus, repetitions, modes=("resident", "transfer")):
    """Reject incorrect device, precision, affinity or incomplete CUDA observations.

    Parameters
    ----------
    record : dict
        Observation produced by ``cuda_workload.py``.
    cpus : list of int
        Required host CPU affinity.
    repetitions : int
        Required number of synchronous timed calls per residency mode.
    modes : tuple of str, optional
        Measurement modes required for this record.

    Raises
    ------
    ValueError
        If CUDA execution or timing evidence does not meet the comparison contract.
    """
    checks = {
        "CUDA backend": record.get("backend") == "cuda",
        "CUDA build": bool(record.get("cuda_version")),
        "CPU affinity": record.get("affinity") == cpus,
        "thread count": record.get("threads") == len(cpus),
        "full FP32": record.get("matmul_precision") == "highest"
        and record.get("tf32_enabled") is False,
        "warm-ups": record.get("warmups", 0) >= 3,
        "repetitions": record.get("repetitions") == repetitions,
        "cuBLAS identity": bool(record.get("library_sha256")),
        "input identities": bool(record.get("input_sha256")),
    }
    for mode in modes:
        values = record.get("measurements", {}).get(mode, {})
        seconds = values.get("seconds", [])
        checks[mode + " timings"] = len(seconds) == repetitions and all(
            math.isfinite(value) and value > 0 for value in seconds
        )
        checks[mode + " outputs"] = bool(values.get("outputs"))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid CUDA observation: " + ", ".join(failed))


def write_report(out, rows, manifest):
    """Write the comparison table with timing definitions and measured conclusions.

    Parameters
    ----------
    out : pathlib.Path
        Results directory.
    rows : list of dict
        Validated per-case median and range summaries.
    manifest : dict
        Hardware, commands and source/library provenance.
    """
    lines = [
        "# CAMBLAS CPU versus PyTorch CUDA",
        "",
        f"Measured on {manifest['hostname']} at {manifest['timestamp']}: "
        f"{len(rows)} workloads/precisions, {manifest['rounds']} fresh-process rounds, "
        f"{manifest['threads']} Grace CPU cores versus one GH200 GPU.",
        "",
        "NVPL is a CPU library. The GPU comparison uses PyTorch CUDA/cuBLAS. "
        "Both CPU builds and the GPU wheel use PyTorch 2.8.0; the GPU wheel includes "
        "CUDA 12.9. CPU builds are project-specific, while the GPU build is the "
        "official wheel, so this compares application configurations, not isolated "
        "BLAS kernels. FP32 uses highest precision with TF32 disabled; FP64 uses "
        "double precision. No autocast, compilation or CUDA graphs are enabled.",
        "",
        "Latencies are wall-clock milliseconds, median of three process medians. "
        "Every GPU call completes before its timer stops. Resident timings exclude "
        "input/output transfers; transfer timings include fresh copies of every "
        "input/weight to the GPU and copies of the output and all gradients back "
        "to pageable CPU memory. CPU timings include the same arithmetic and "
        "allocation. Warm-up, random input generation, import, validation and "
        "CUDA initialisation are excluded. CPU and GPU input values and workload "
        "expressions match the existing application suite.",
        "",
        "GPU speed-up is CAMBLAS latency divided by GPU latency; values above "
        "1 favour the GPU. CAMBLAS/NVPL speed-up is NVPL latency divided by "
        "CAMBLAS latency; values above 1 favour CAMBLAS.",
        "",
        "| Workload | Precision | CAMBLAS CPU ms | NVPL CPU ms | GPU resident ms | "
        "GPU with transfers ms | GPU resident speed-up | GPU transfer speed-up |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = row["medians_ms"]
        lines.append(
            f"| {row['workload']} | {row['dtype']} | {value['camblas']:.4f} | "
            f"{value['nvpl']:.4f} | {value['cuda_resident']:.4f} | "
            f"{value['cuda_transfer']:.4f} | "
            f"{value['camblas'] / value['cuda_resident']:.2f}x | "
            f"{value['camblas'] / value['cuda_transfer']:.2f}x |"
        )
    lines += ["", "Geometric means across the eight equally weighted workloads:", ""]
    for dtype in sorted({row["dtype"] for row in rows}):
        subset = [row["medians_ms"] for row in rows if row["dtype"] == dtype]
        resident = statistics.geometric_mean(
            row["camblas"] / row["cuda_resident"] for row in subset
        )
        transfer = statistics.geometric_mean(
            row["camblas"] / row["cuda_transfer"] for row in subset
        )
        nvpl = statistics.geometric_mean(row["nvpl"] / row["camblas"] for row in subset)
        wins = sum(row["camblas"] < row["cuda_transfer"] for row in subset)
        lines.append(
            f"- {dtype}: GPU resident {resident:.2f}x; GPU with transfers "
            f"{transfer:.2f}x; CAMBLAS versus NVPL {nvpl:.3f}x. CAMBLAS is faster "
            f"than the transfer-inclusive GPU in {wins}/{len(subset)} cases."
        )
    lines += [
        "",
        "All rounds passed sampled output/gradient and global-norm checks against "
        "NVPL. GEMM and Gram workloads also passed 64 independently summed "
        "dot-product checks per process. These are numerical checks, not a "
        "full-output correctness proof or a statistical significance test. "
        "Ranges in summary.json describe the three process medians. This suite "
        "does not establish performance on smaller batches, other hardware or "
        "mixed-precision tensor-core training.",
        "",
        "Sources: [NVPL CPU-only documentation](https://docs.nvidia.com/nvpl/latest/), "
        "[PyTorch CUDA timing and precision](https://docs.pytorch.org/docs/stable/notes/cuda.html).",
        "",
        "Source snapshots and "
        "reproduction commands are saved beside the raw observations.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main():
    """Run sequential, rotated CPU/GPU measurements under the project benchmark lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument(
        "--cuda-python", type=Path, default=ROOT / ".frameworks/envs/cuda/bin/python"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cpu-repetitions", type=int)
    parser.add_argument("--gpu-repetitions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("Use at least three fresh-process rounds")
    available = sorted(os.sched_getaffinity(0))
    if not 1 <= args.threads <= len(available):
        parser.error("Thread count must fit the current CPU affinity/allocation")
    cpus = available[: args.threads]
    sockets = {
        Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id")
        .read_text()
        .strip()
        for cpu in cpus
    }
    if len(sockets) != 1:
        parser.error("Bind the parent to one Grace CPU before running this comparison")
    if socket.gethostname().lower().startswith("login") and not args.dry_run:
        parser.error("Run on allocated compute resources")
    if any(
        os.environ.get(name)
        for name in (
            "LD_PRELOAD",
            "OMP_WAIT_POLICY",
            "GOMP_SPINCOUNT",
            "MALLOC_TRIM_THRESHOLD_",
            "MALLOC_MMAP_THRESHOLD_",
            "MALLOC_ARENA_MAX",
            "GLIBC_TUNABLES",
        )
    ):
        parser.error("Unset preload, allocator and OpenMP wait-policy overrides")
    for repetitions in (args.cpu_repetitions, args.gpu_repetitions):
        if repetitions is not None and repetitions < 5:
            parser.error("Use at least five timed calls")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output or ROOT / "bench/results" / f"cuda_{timestamp}").resolve()
    libraries = {
        backend: ROOT / ".frameworks/prefix" / backend / "lib/libframework_blas.so"
        for backend in ("camblas", "nvpl")
    }
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    manifest = dict(
        timestamp=timestamp,
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        threads=args.threads,
        affinity=cpus,
        rounds=args.rounds,
        cuda_visible_devices=visible_gpu,
        command=os.sys.argv,
        case_count=len(args.workloads) * len(args.dtypes),
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    for executable in [
        args.cuda_python,
        *(ROOT / ".frameworks/envs" / backend / "bin/python" for backend in libraries),
    ]:
        if not executable.is_file():
            parser.error(f"Missing {executable}")
    with (ROOT / ".frameworks/measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        manifest["backends"] = {
            backend: dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for backend, path in libraries.items()
        }
        core = libraries["camblas"].parent / "libcamblas_sve_nr4.so"
        manifest["backends"]["camblas"]["core_sha256"] = hashlib.sha256(
            core.read_bytes()
        ).hexdigest()
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        manifest["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
        manifest["gpu_topology"] = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True
        )
        manifest["source_sha256"] = {}
        source_dir = out / "sources"
        source_dir.mkdir()
        for name in (
            "workload.py",
            "cuda_workload.py",
            "compare.py",
            "compare_cuda.py",
        ):
            path = ROOT / "bench" / name
            if path.is_file():
                contents = path.read_bytes()
                manifest["source_sha256"][name] = hashlib.sha256(contents).hexdigest()
                (source_dir / name).write_bytes(contents)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        rows = []
        commands = []
        print(
            f"{manifest['case_count']} cases, {args.rounds} rounds; output: {out}",
            flush=True,
        )
        for dtype, workload in itertools.product(args.dtypes, args.workloads):
            observations = {backend: [] for backend in BACKENDS}
            cpu_repetitions = args.cpu_repetitions or (
                201 if workload in ("mlp", "backward") else 5
            )
            gpu_repetitions = args.gpu_repetitions or (
                201 if workload in ("mlp", "backward") else 21
            )
            for round_id in range(args.rounds):
                shift = round_id % len(BACKENDS)
                for backend in BACKENDS[shift:] + BACKENDS[:shift]:
                    key = f"{workload}_{dtype}_r{round_id + 1}_{backend}"
                    environment = dict(
                        os.environ,
                        OMP_NUM_THREADS=str(args.threads),
                        OMP_DYNAMIC="FALSE",
                        OPENBLAS_NUM_THREADS=str(args.threads),
                        CUDA_VISIBLE_DEVICES=visible_gpu,
                    )
                    if backend == "cuda":
                        executable, script = (
                            args.cuda_python,
                            ROOT / "bench/cuda_workload.py",
                        )
                        extra = ["--transfer-first"] if round_id % 2 else []
                        repetitions = gpu_repetitions
                    else:
                        executable = ROOT / ".frameworks/envs" / backend / "bin/python"
                        script = ROOT / "bench/workload.py"
                        extra = ["--backend", backend, "--framework", "pytorch"]
                        repetitions = cpu_repetitions
                        environment.update(
                            CAMBLAS_FRAMEWORK_ROOT=str(ROOT),
                            CAMBLAS_FRAMEWORK_TRACE="0",
                            CAMBLAS_FRAMEWORK_THREADS=str(args.threads),
                            CAMBLAS_FRAMEWORK_BRIDGE=str(libraries[backend]),
                        )
                    command = [
                        "taskset",
                        "-c",
                        ",".join(map(str, cpus)),
                        str(executable),
                        str(script),
                        "--dtype",
                        dtype,
                        "--workload",
                        workload,
                        "--threads",
                        str(args.threads),
                        "--repetitions",
                        str(repetitions),
                        "--output",
                        str(out / f"{key}.json"),
                        *extra,
                    ]
                    commands.append(
                        dict(
                            command=command,
                            environment={
                                name: environment[name]
                                for name in (
                                    "OMP_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS",
                                    "CUDA_VISIBLE_DEVICES",
                                )
                            },
                        )
                    )
                    (out / "commands.json").write_text(
                        json.dumps(commands, indent=2) + "\n"
                    )
                    print(f"{key}: {shlex.join(command[:5])}", flush=True)
                    with (out / f"{key}.log").open("w") as log:
                        subprocess.run(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            check=True,
                        )
                    record = json.loads((out / f"{key}.json").read_text())
                    if backend == "cuda":
                        validate_cuda_record(record, cpus, repetitions)
                    else:
                        validate_process_record(
                            record,
                            backend,
                            manifest["backends"][backend],
                            cpus,
                            repetitions,
                        )
                    if record["torch_version"].split("+")[0] != "2.8.0":
                        raise ValueError(
                            "This comparison requires PyTorch 2.8.0 for every backend"
                        )
                    observations[backend].append(record)
            reference = observations["nvpl"][0]["outputs"]
            error = 0.0
            for backend, records in observations.items():
                for record in records:
                    outputs = (
                        [
                            record["measurements"][mode]["outputs"]
                            for mode in ("resident", "transfer")
                        ]
                        if backend == "cuda"
                        else [record["outputs"]]
                    )
                    for result in outputs:
                        error = max(error, check_outputs(result, reference, dtype))
            gpu_inputs = [record["input_sha256"] for record in observations["cuda"]]
            if any(value != gpu_inputs[0] for value in gpu_inputs):
                raise ValueError("GPU input identities changed between rounds")
            round_medians = {
                backend: [
                    statistics.median(record["seconds"]) * 1000
                    for record in observations[backend]
                ]
                for backend in ("camblas", "nvpl")
            }
            for mode in ("resident", "transfer"):
                round_medians["cuda_" + mode] = [
                    statistics.median(record["measurements"][mode]["seconds"]) * 1000
                    for record in observations["cuda"]
                ]
            row = dict(
                workload=workload,
                dtype=dtype,
                threads=args.threads,
                cpu_calls_per_round=cpu_repetitions,
                gpu_calls_per_round=gpu_repetitions,
                round_medians_ms=round_medians,
                medians_ms={
                    name: statistics.median(value)
                    for name, value in round_medians.items()
                },
                ranges_ms={
                    name: [min(value), max(value)]
                    for name, value in round_medians.items()
                },
                max_scaled_sample_error=error,
                gpu_device_round_medians_ms=[
                    statistics.median(
                        record["measurements"]["resident"]["device_seconds"]
                    )
                    * 1000
                    for record in observations["cuda"]
                ],
            )
            rows.append(row)
            (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        write_report(out, rows, manifest)
        (out / "complete.json").write_text(
            json.dumps(dict(status="passed", cases=len(rows)), indent=2) + "\n"
        )
        print(f"Completed: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
