#!/usr/bin/env python3
"""Compare complete SGLang requests and every raw vocabulary logit."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .verify_llm_weights import check_verified_files, manifest_identity, verify
else:
    from verify_llm_weights import check_verified_files, manifest_identity, verify

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Hash a file used by the measured worker."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(directory):
    """Record the pinned runtime revision and any explicit local source changes."""
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "-C", str(directory), "diff", "--binary", "HEAD", "--", "python"]
    )
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(directory),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "python",
        ],
        text=True,
    ).strip()
    if untracked:
        raise ValueError(f"Untracked SGLang runtime source: {untracked}")
    return revision, patch


def load_verified_logits(directory, result):
    """Reject incomplete, stale or inconsistent vocabulary captures for a worker."""
    import torch

    def identical_bits(a, b):
        """Compare tensor storage bitwise, including signed zeros."""
        return torch.equal(a.view(torch.int32), b.view(torch.int32))

    directory = Path(directory)
    ranks = {}
    for name in result["accuracy_files"]:
        record = torch.load(directory / name, map_location="cpu", weights_only=True)
        logits = record["logits"]
        if logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Invalid raw vocabulary capture")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite raw vocabulary logits")
        ranks.setdefault(record["rank"], []).append(record)
    if set(ranks) != set(range(result["case"]["tensor_parallel"])):
        raise ValueError("Missing model rank")
    count = result["case"]["accuracy_tokens"]
    expected_positions = (
        list(
            range(
                result["case"]["prompt_length"] - 1,
                result["case"]["prompt_length"] - 1 + count,
            )
        )
        * 3
    )
    for rank, records in ranks.items():
        if [r["prediction_position"] for r in records] != expected_positions:
            raise ValueError("Missing or misordered accuracy prediction")
        for i in range(count):
            if not identical_bits(records[i]["logits"], records[i + count]["logits"]):
                raise ValueError("Repeated raw vocabulary logits differ")
        if all(
            identical_bits(records[i]["logits"], records[i + 2 * count]["logits"])
            for i in range(count)
        ):
            raise ValueError("Changed-input raw vocabulary logits did not change")
        for a, b in zip(records, ranks[0]):
            if not identical_bits(a["logits"], b["logits"]):
                raise ValueError(f"Vocabulary logits differ across rank {rank}")
    return ranks


def compare_outputs(reference_directory, actual_directory, *, atol, rtol):
    """Check returned tokens, repeats, ranks and full raw vocabulary vectors."""
    import torch

    def identical_bits(a, b):
        """Compare tensor storage bitwise, including signed zeros."""
        return torch.equal(a.view(torch.int32), b.view(torch.int32))

    directories = [Path(reference_directory), Path(actual_directory)]
    results = [json.loads((p / "result.json").read_text()) for p in directories]
    reference, actual = results
    if any(r.get("state") != "passed" for r in results):
        raise ValueError("A measured worker did not pass")
    for key in ("input_ids", "changed_input_ids", "input_sha256"):
        if reference[key] != actual[key]:
            raise ValueError(f"Worker inputs differ: {key}")
    for key in ("model", "prompt_length", "generated_tokens", "accuracy_tokens"):
        if reference["case"][key] != actual["case"][key]:
            raise ValueError(f"Worker cases differ: {key}")
    for result in results:
        if any(
            s["output_ids"] != reference["samples"][0]["output_ids"]
            for s in result["samples"]
        ):
            raise ValueError("Timed greedy generations differ")
    if [a["output_ids"] for a in reference["accuracy"]] != [
        a["output_ids"] for a in actual["accuracy"]
    ]:
        raise ValueError("Changed-input or accuracy greedy generations differ")

    vectors = [
        load_verified_logits(directory, result)
        for directory, result in zip(directories, results)
    ]
    maximum = 0.0
    bitwise = True
    checked = 0
    for rank in vectors[0]:
        for expected, received in zip(vectors[0][rank], vectors[1][rank]):
            a, b = expected["logits"], received["logits"]
            if a.shape != b.shape or not torch.isfinite(b).all():
                raise ValueError("Invalid vocabulary logit shape or values")
            if not torch.allclose(a, b, atol=atol, rtol=rtol):
                raise ValueError(
                    "Full vocabulary logits exceeded the precision contract"
                )
            maximum = max(maximum, (a - b).abs().max().item())
            identical = identical_bits(a, b)
            if atol == rtol == 0 and not identical:
                raise ValueError("Full vocabulary logits are not bitwise identical")
            bitwise &= identical
            checked += a.numel()
    return dict(
        state="passed",
        atol=atol,
        rtol=rtol,
        bitwise=bitwise,
        maximum_absolute_error=maximum,
        vocabulary_values_checked=checked,
        repeated_raw_logits_bitwise=True,
        all_ranks_identical=True,
        timed_and_changed_input_tokens_identical=True,
    )


def summarise(records, generated_tokens):
    """Retain fresh-process medians and ranges for CPU-to-CPU requests."""
    rows = []
    for prompt in sorted({r["prompt_length"] for r in records}):
        selected = [r for r in records if r["prompt_length"] == prompt]
        for phase, tokens in (
            ("prefill", prompt),
            ("decode", generated_tokens - 1),
            ("request", generated_tokens),
        ):
            samples = {
                label: [
                    r["backends"][label]["measurements"][phase]["median_ms"]
                    for r in selected
                ]
                for label in ("sglang", "camblas")
            }
            medians = {label: statistics.median(v) for label, v in samples.items()}
            rows.append(
                dict(
                    batch=1,
                    prompt_length=prompt,
                    generated_tokens=generated_tokens,
                    phase=phase,
                    processes=len(selected),
                    medians_ms=medians,
                    process_medians_ms=samples,
                    ranges_ms={label: [min(v), max(v)] for label, v in samples.items()},
                    tokens_per_second={
                        label: tokens * 1000 / ms for label, ms in medians.items()
                    },
                    speedup=medians["sglang"] / medians["camblas"],
                )
            )
    return rows


def main():
    """Run serial, rotated fresh processes inside an exclusive GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--weights-verification", type=Path)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--camblas-library", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".frameworks/envs/sglang/bin/python"
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--accuracy-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-cutoff", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside the exclusive Slurm allocation")
    if args.rounds < 3 or min(args.warmups, args.repetitions) < 1:
        parser.error("Use at least three fresh process rounds and positive repetitions")
    if args.accuracy_tokens < 1:
        parser.error("Use a positive accuracy generation length")
    if args.generated_tokens < 2 or min(args.prompt_lengths) < 2:
        parser.error("Both prompt and generation must contain at least two tokens")
    cutoff = datetime.fromisoformat(args.gpu_cutoff)
    if cutoff.tzinfo is None or datetime.now(timezone.utc) >= cutoff:
        parser.error("Provide a future UTC allocation cutoff")
    allocation = subprocess.check_output(
        ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "--oneliner"],
        text=True,
        env={**os.environ, "TZ": "UTC"},
    ).strip()
    fields = dict(word.split("=", 1) for word in allocation.split() if "=" in word)
    if (
        fields["NumNodes"] != "1"
        or fields["OverSubscribe"] != "NO"
        or fields["NodeList"] != os.uname().nodename
    ):
        parser.error("Use a single-node exclusive allocation")
    allocation_end = datetime.fromisoformat(fields["EndTime"]).replace(
        tzinfo=timezone.utc
    )
    if cutoff > allocation_end - timedelta(minutes=5):
        parser.error("GPU cutoff must be at least five minutes before allocation end")
    directory = args.output.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    weights = json.loads(args.weights_manifest.read_text())
    runtime = json.loads(args.runtime.read_text())
    profile = runtime["models"][weights["model"]]
    source = args.sglang_source.absolute()
    revision, source_patch = source_identity(source)
    if revision != runtime["source_revision"]:
        raise ValueError("SGLang source revision differs from the pinned runtime")
    (directory / "sglang.patch").write_bytes(source_patch)
    inputs = [
        Path(__file__),
        ROOT / "bench/llm_workload.py",
        *sorted((ROOT / "camblas").glob("*.py")),
        ROOT / "bench/verify_llm_weights.py",
        args.runtime,
        args.weights_manifest,
        args.camblas_library,
    ]
    fp8_tiles = profile.get("fp8_tiles")
    if fp8_tiles:
        inputs.append(ROOT / fp8_tiles)
    moe_tiles = profile.get("moe_tiles")
    if moe_tiles:
        inputs.append(ROOT / moe_tiles)
    inputs += sorted(args.camblas_library.parent.glob("_camblas_cuda_torch*.so"))
    build_path = args.camblas_library.parent / "build.json"
    build = json.loads(build_path.read_text())
    if digest(args.camblas_library) != build["library_sha256"]:
        raise ValueError("CAMBLAS native library differs from its build record")
    for name, expected in build["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"CAMBLAS build source changed: {name}")
        inputs.append(ROOT / name)
    inputs.append(build_path)
    identities = {str(p.absolute()): digest(p) for p in inputs}
    manifest = dict(
        state="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,
        gpu_inventory=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        allocation=os.environ["SLURM_JOB_ID"],
        allocation_details=allocation,
        node=os.uname().nodename,
        source_revision=revision,
        source_patch_sha256=hashlib.sha256(source_patch).hexdigest(),
        source_sha256=identities,
        model=weights["model"],
        revision=weights["revision"],
        precision=profile["precision"],
        generated_tokens=args.generated_tokens,
        transfer_contract="CPU token inputs to streamed CPU token outputs, including scheduler, GPU computation, inter-GPU communication and model host-table gathers. Model loading, graph capture, warmups, tokenisation and accuracy downloads are excluded; weights and KV cache remain prepared.",
        records=[],
        runtime_libraries=None,
        camblas_build=build,
        runtime=runtime,
        accuracy_tokens=args.accuracy_tokens,
    )

    def save():
        """Write the current measurement and verification manifest."""
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(directory / "manifest.json")

    save()
    try:
        with (ROOT / ".frameworks/measurement.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.weights_verification:
                verified = json.loads(args.weights_verification.read_text())
                if (
                    verified["state"] != "passed"
                    or verified["model"] != weights["model"]
                    or verified["revision"] != weights["revision"]
                    or verified["manifest_sha256"] != manifest_identity(weights)
                    or verified["directory"] != str(args.model_directory.resolve())
                    or verified["configuration"] != weights["configuration"]
                    or verified.get("files", {}) != weights.get("files", {})
                    or set(verified["weights"]) != set(weights["weights"])
                    or any(
                        verified["weights"][name][key] != expected[key]
                        for name, expected in weights["weights"].items()
                        for key in ("sha256", "bytes")
                    )
                ):
                    raise ValueError("Checkpoint verification identity differs")
            else:
                verified = verify(
                    args.model_directory, weights, directory / "weights.json"
                )
            if "parameters" in verified and "storage_elements" not in verified:
                verified["storage_elements"] = verified.pop("parameters")
            check_verified_files(args.model_directory, verified)
            manifest["weights_verification"] = verified
            save()
            for prompt in args.prompt_lengths:
                for round_id in range(args.rounds):
                    record = dict(prompt_length=prompt, round=round_id, backends={})
                    backends = ["sglang", "camblas"]
                    if round_id % 2:
                        backends.reverse()
                    for backend in backends:
                        remaining = (
                            cutoff - datetime.now(timezone.utc)
                        ).total_seconds()
                        worker_timeout = min(args.timeout, int(remaining) - 15)
                        if worker_timeout < 60:
                            raise RuntimeError(
                                "Insufficient allocation time for another worker"
                            )
                        if any(digest(p) != value for p, value in identities.items()):
                            raise ValueError("Benchmark code or native binary changed")
                        if source_identity(source) != (revision, source_patch):
                            raise ValueError("SGLang runtime source changed")
                        active = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid,process_name,used_memory",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ).strip()
                        if active:
                            raise RuntimeError(f"GPU allocation is not quiet: {active}")
                        name = f"p{prompt}_r{round_id}_{backend}"
                        output = directory / name
                        engine_args = {
                            **runtime["engine_args"],
                            **profile["engine_args"],
                            "model_path": str(args.model_directory.resolve()),
                        }
                        case = dict(
                            model=weights["model"],
                            model_directory=str(args.model_directory.resolve()),
                            backend=backend,
                            camblas_operations=profile["camblas_operations"].split(",")
                            if backend == "camblas"
                            else [],
                            prompt_length=prompt,
                            generated_tokens=args.generated_tokens,
                            accuracy_tokens=args.accuracy_tokens,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            tensor_parallel=engine_args["tp_size"],
                            gpu_cutoff=args.gpu_cutoff,
                            engine_args=engine_args,
                            weights_verification=verified,
                            runtime_packages=runtime["packages"],
                        )
                        case_path = directory / (name + ".case.json")
                        case_path.write_text(json.dumps(case, indent=2) + "\n")
                        env = {
                            **os.environ,
                            **runtime.get("environment", {}),
                            **profile.get("environment", {}),
                            "PYTHONPATH": str(source / "python")
                            + os.pathsep
                            + str(ROOT),
                            "SGLANG_PLUGINS": "camblas",
                            "CAMBLAS_ENABLE": "1" if backend == "camblas" else "0",
                            "CAMBLAS_CUDA_LIBRARY": str(
                                args.camblas_library.absolute()
                            ),
                            "CAMBLAS_SGLANG_OPS": profile["camblas_operations"]
                            if backend == "camblas"
                            else "",
                        }
                        env["PATH"] = (
                            str(args.python.absolute().parent)
                            + os.pathsep
                            + env.get("PATH", os.defpath)
                        )
                        if shutil.which("ninja", path=env["PATH"]) is None:
                            raise ValueError(
                                "The SGLang Python environment must provide ninja"
                            )
                        if fp8_tiles:
                            env["CAMBLAS_SGLANG_FP8_TILES"] = str(ROOT / fp8_tiles)
                        else:
                            env.pop("CAMBLAS_SGLANG_FP8_TILES", None)
                        if moe_tiles:
                            env["SGLANG_MOE_CONFIG_DIR"] = str(
                                (ROOT / moe_tiles).parents[2]
                            )
                        command = [
                            str(args.python.absolute()),
                            str(ROOT / "bench/llm_workload.py"),
                            "--case",
                            str(case_path),
                            "--output",
                            str(output),
                        ]
                        with (directory / (name + ".log")).open("w") as log:
                            process = subprocess.Popen(
                                command,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            try:
                                if process.wait(timeout=worker_timeout):
                                    raise RuntimeError(
                                        f"Worker failed; inspect {name}.log"
                                    )
                            except BaseException:
                                if process.poll() is None:
                                    os.killpg(process.pid, signal.SIGTERM)
                                    try:
                                        process.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(process.pid, signal.SIGKILL)
                                        process.wait()
                                raise
                        result = json.loads((output / "result.json").read_text())
                        check_verified_files(args.model_directory, verified)
                        expected_packages = runtime["packages"]
                        if any(
                            result["packages"][name] != value
                            for name, value in expected_packages.items()
                        ):
                            raise ValueError("Worker runtime package versions differ")
                        if (
                            Path(result["sglang_source"]).resolve()
                            != (source / "python/sglang/__init__.py").resolve()
                        ):
                            raise ValueError(
                                "Worker imported a different SGLang source"
                            )
                        for rank in range(case["tensor_parallel"]):
                            loaded = json.loads(
                                (
                                    output / "logits" / f"rank{rank}-libraries.json"
                                ).read_text()
                            )
                            expected = {
                                p: v
                                for p, v in identities.items()
                                if Path(p).name == "libcamblas_cuda.so"
                                or Path(p).name.startswith("_camblas_cuda_torch")
                            }
                            if loaded != (expected if backend == "camblas" else {}):
                                raise ValueError(
                                    "Unexpected native libraries in a measured rank"
                                )
                            libraries = json.loads(
                                (
                                    output / "logits" / f"rank{rank}-runtime.json"
                                ).read_text()
                            )
                            if not libraries:
                                raise ValueError(
                                    "Missing SGLang native library evidence"
                                )
                            if manifest["runtime_libraries"] is None:
                                manifest["runtime_libraries"] = libraries
                            if libraries != manifest["runtime_libraries"] or any(
                                digest(p) != value for p, value in libraries.items()
                            ):
                                raise ValueError(
                                    "SGLang native runtime libraries differ"
                                )
                        record["backends"][backend] = dict(
                            output=str(output),
                            measurements=result["measurements"],
                            setup_seconds=result["setup_seconds"],
                        )
                    record["accuracy"] = compare_outputs(
                        record["backends"]["sglang"]["output"],
                        record["backends"]["camblas"]["output"],
                        atol=profile["atol"],
                        rtol=profile["rtol"],
                    )
                    manifest["records"].append(record)
                    save()
                    print(json.dumps(record), flush=True)
            manifest.update(
                state="passed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                rows=summarise(manifest["records"], args.generated_tokens),
            )
            save()
    except BaseException as error:
        manifest.update(state="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
