#!/usr/bin/env python3
"""Fresh-process, rotated-order CPU comparisons with at least three rounds."""

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
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("camblas", "openblas", "nvpl")
WORKLOADS = (
    "square1024",
    "square4096",
    "square8192",
    "transpose",
    "gram",
    "mlp",
    "attention",
    "backward",
)


def validate_process_record(record, backend, identity, cpus, repetitions):
    """Reject mistargeted, incomplete or non-finite benchmark observations.

    Parameters
    ----------
    record : dict
        Observation produced by one fresh workload process.
    backend : str
        Requested library name: ``camblas``, ``openblas`` or ``nvpl``.
    identity : dict
        Expected bridge digest and, for CAMBLAS, native-core digest.
    cpus : list of int
        Exact CPU affinity requested for this process.
    repetitions : int
        Required number of timed calls after three warm-ups.

    Raises
    ------
    ValueError
        If the process used a different library or affinity, omitted work, or
        returned a non-finite timing or numerical signature.

    Notes
    -----
    A bridge's identity does not establish which native core the dynamic linker
    resolved. Check the observed core separately before accepting its timings.
    """
    observed_backend = (
        "camblas+openblas-compatibility" if backend == "camblas" else backend
    )
    checks = {
        "bridge identity": record.get("bridge_sha256") == identity["sha256"],
        "backend": record.get("observed_backend") == observed_backend,
        "CPU affinity": record.get("affinity") == cpus,
        "thread count": record.get("threads") == len(cpus),
        "warm-up count": record.get("warmups") == 3,
        "repetition count": record.get("repetitions") == repetitions,
    }
    if backend == "camblas":
        core = record.get("core_identity") or {}
        checks["native-core identity"] = core.get("sha256") == identity["core_sha256"]
    seconds = record.get("seconds", [])
    checks["finite positive timings"] = len(seconds) == repetitions and all(
        math.isfinite(value) and value > 0 for value in seconds
    )
    counters = record.get("counters", [])
    checks["successful BLAS work"] = (
        len(counters) >= 10 and counters[9] == 0 and sum(counters[:4]) > 0
    )
    outputs = record.get("outputs", [])
    checks["finite numerical signatures"] = bool(outputs) and all(
        output["size"] > 0
        and bool(output["samples"])
        and all(
            math.isfinite(value)
            for value in [output["norm"], output["max_abs"], *output["samples"]]
        )
        for output in outputs
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid benchmark record: " + ", ".join(failed))


def _main(resources):
    """Run the selected comparison matrix and validate every process record.

    Parameters
    ----------
    resources : contextlib.ExitStack
        Owns the measurement lock until all comparisons finish, including when
        a subprocess or numerical validation fails.

    Notes
    -----
    Each round rotates the backend order and starts a fresh process. Reported
    latencies are medians of per-round medians, not the fastest observed call.
    Dry runs enumerate commands without acquiring the lock or loading BLAS.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Both frameworks, both precisions, 16/64 cores: 60 cases",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=("numpy", "pytorch"),
        default=["numpy", "pytorch"],
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=["mlp", "backward"]
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=("float32", "float64"), default=["float32"]
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Calls per round; default: NumPy MLP 1001, PyTorch MLP/backward 201, others 5",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--timeout", type=int, default=600, help="Seconds allowed for one fresh process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the matrix and commands without loading frameworks or writing files",
    )
    args = parser.parse_args()
    if args.rounds < 3 or (args.repetitions is not None and args.repetitions < 5):
        parser.error("Use at least three rounds and five timed calls per round")
    if args.full:
        args.frameworks, args.workloads = ["numpy", "pytorch"], list(WORKLOADS)
        args.dtypes, args.threads = ["float32", "float64"], [16, 64]
    cpus = sorted(os.sched_getaffinity(0))
    if min(args.threads) < 1 or max(args.threads) > len(cpus):
        parser.error(
            "Requested thread counts must fit the current CPU affinity/allocation"
        )
    if not args.dry_run and socket.gethostname().lower().startswith("login"):
        parser.error(
            "Run performance comparisons on allocated compute CPUs, not a login node"
        )
    # Defaults only: allocator/wait-policy experiments belong in separate reports.
    overrides = (
        "LD_PRELOAD",
        "OMP_WAIT_POLICY",
        "GOMP_SPINCOUNT",
        "MALLOC_TRIM_THRESHOLD_",
        "MALLOC_MMAP_THRESHOLD_",
        "MALLOC_ARENA_MAX",
        "GLIBC_TUNABLES",
    )
    if any(os.environ.get(name) for name in overrides):
        parser.error(
            "Unset preload, allocator and wait-policy overrides before default-policy comparisons"
        )
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output or ROOT / "results" / timestamp).resolve()
    matrices = [
        (f, t, d, w)
        for f, t, d, w in itertools.product(
            args.frameworks, args.threads, args.dtypes, args.workloads
        )
        if not (f == "numpy" and w == "backward")
    ]
    libraries = {
        b: ROOT / ".frameworks/prefix" / b / "lib/libframework_blas.so"
        for b in BACKENDS
    }
    if not args.dry_run:
        for backend in BACKENDS:
            for file in (
                libraries[backend],
                ROOT / ".frameworks/envs" / backend / "bin/python",
            ):
                if not file.is_file():
                    parser.error(
                        f"Missing {file}; build/install all three variants first"
                    )
        lock = resources.enter_context(
            (ROOT / ".frameworks/measurement.lock").open("a")
        )
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        manifest = {
            "case_count": len(matrices),
            "rounds": args.rounds,
            "affinity": cpus,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "command": os.sys.argv,
            "backends": {
                b: {
                    "path": str(p),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for b, p in libraries.items()
            },
        }
        core = libraries["camblas"].parent / "libcamblas_sve_nr4.so"
        manifest["backends"]["camblas"]["core_sha256"] = hashlib.sha256(
            core.read_bytes()
        ).hexdigest()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{len(matrices)} cases, {args.rounds} fresh-process rounds, all three libraries; output: {out}",
        flush=True,
    )
    rows = []
    for framework, threads, dtype, workload in matrices:
        repetitions = args.repetitions or (
            1001
            if framework == "numpy" and workload == "mlp"
            else 201
            if framework == "pytorch" and workload in ("mlp", "backward")
            else 5
        )
        observations = {b: [] for b in BACKENDS}
        for round_id in range(args.rounds):
            shift = round_id % len(BACKENDS)
            for backend in BACKENDS[shift:] + BACKENDS[:shift]:
                key = f"{framework}_{workload}_{dtype}_{threads}_r{round_id + 1}_{backend}"
                env = dict(
                    os.environ,
                    CAMBLAS_FRAMEWORK_ROOT=str(ROOT),
                    CAMBLAS_FRAMEWORK_BRIDGE=str(libraries[backend]),
                    CAMBLAS_FRAMEWORK_TRACE="0",
                    CAMBLAS_FRAMEWORK_THREADS=str(threads),
                    OPENBLAS_NUM_THREADS=str(threads),
                    OMP_NUM_THREADS=str(threads),
                    OMP_DYNAMIC="FALSE",
                )
                command = [
                    "taskset",
                    "-c",
                    ",".join(map(str, cpus[:threads])),
                    str(ROOT / ".frameworks/envs" / backend / "bin/python"),
                    str(ROOT / "bench/workload.py"),
                    "--backend",
                    backend,
                    "--framework",
                    framework,
                    "--workload",
                    workload,
                    "--dtype",
                    dtype,
                    "--threads",
                    str(threads),
                    "--repetitions",
                    str(repetitions),
                    "--output",
                    str(out / f"{key}.json"),
                ]
                if args.dry_run:
                    print(shlex.join(command))
                    continue
                with (out / f"{key}.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=True,
                    )
                record = json.loads((out / f"{key}.json").read_text())
                validate_process_record(
                    record,
                    backend,
                    manifest["backends"][backend],
                    cpus[:threads],
                    repetitions,
                )
                observations[backend].append(record)
        if args.dry_run:
            continue
        # Check every round against the same output signature. These sampled
        # comparisons supplement, but do not replace, full-output oracle tests.
        reference = observations["nvpl"][0]["outputs"]
        max_error = 0.0
        tolerance = 3e-4 if dtype == "float32" else 3e-11
        for records in observations.values():
            for record in records:
                assert len(record["outputs"]) == len(reference)
                for result, expected in zip(record["outputs"], reference):
                    assert result["size"] == expected["size"] and len(
                        result["samples"]
                    ) == len(expected["samples"])
                    error = max(
                        abs(x - y)
                        for x, y in zip(result["samples"], expected["samples"])
                    ) / max(expected["max_abs"], 1e-15)
                    assert error < tolerance
                    assert abs(result["norm"] - expected["norm"]) <= tolerance * max(
                        expected["norm"], 1e-15
                    )
                    max_error = max(max_error, error)
        rounds = {
            b: [1000 * statistics.median(v["seconds"]) for v in values]
            for b, values in observations.items()
        }
        medians = {b: statistics.median(values) for b, values in rounds.items()}
        rows.append(
            dict(
                framework=framework,
                workload=workload,
                dtype=dtype,
                threads=threads,
                calls_per_round=repetitions,
                round_medians_ms=rounds,
                medians_ms=medians,
                max_scaled_sample_error=max_error,
            )
        )
        (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
    if args.dry_run:
        return
    wins = {
        b: sum(row["medians_ms"]["camblas"] < row["medians_ms"][b] for row in rows)
        for b in ("openblas", "nvpl")
    }
    lines = [
        "# Fresh application comparison",
        "",
        f"{len(rows)} cases; {args.rounds} fresh-process rounds per library. Lower median latency is better. Small differences are not significance tests.",
        "",
        "| Framework | Workload | Precision | Cores | CAMBLAS ms | OpenBLAS ms | NVPL ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row["medians_ms"]
        lines.append(
            f"| {row['framework']} | {row['workload']} | {row['dtype']} | {row['threads']} | {values['camblas']:.6f} | {values['openblas']:.6f} | {values['nvpl']:.6f} |"
        )
    lines += [
        "",
        f"CAMBLAS wins: {wins['openblas']}/{len(rows)} against OpenBLAS; {wins['nvpl']}/{len(rows)} against NVPL.",
        "",
        "The JSON records retain every round, observed library identity, call counter, affinity and numerical check. Application samples and norms do not constitute a full-output proof; run the independent correctness tests as well.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    (out / "complete.json").write_text(
        json.dumps({"status": "passed", "cases": len(rows), "wins": wins}, indent=2)
        + "\n"
    )
    print(f"Completed: {out / 'report.md'}")


def main():
    """Run command-line comparisons and release resources on every exit path."""
    with ExitStack() as resources:
        _main(resources)


if __name__ == "__main__":
    main()
