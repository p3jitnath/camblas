#!/usr/bin/env python3
"""Run rotated fresh-process Llama 70B comparisons from a local checkpoint."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Return a SHA256 identity for benchmark code and native binaries."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(records, directory, manifest):
    """Retain all process medians, ranges and regressions in JSON and Markdown."""
    groups = {}
    for record in records:
        key = (record["dtype"], record["batch"], record["prompt_length"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (dtype, batch, prompt), selected in sorted(groups.items()):
        for phase, mode in itertools.product(
            ["prefill", "decode", "request"], ["resident", "transfer"]
        ):
            samples = {}
            for label in ["pytorch", "camblas"]:
                samples[label] = [
                    r["measurements"][label][phase][mode]["median_ms"] for r in selected
                ]
            medians = {label: statistics.median(v) for label, v in samples.items()}
            timed_tokens = batch * (
                prompt
                if phase == "prefill"
                else manifest["generated_tokens"] - 1
                if phase == "decode"
                else manifest["generated_tokens"]
            )
            rows.append(
                dict(
                    dtype=dtype,
                    batch=batch,
                    prompt_length=prompt,
                    generated_tokens=manifest["generated_tokens"],
                    phase=phase,
                    mode=mode,
                    processes=len(selected),
                    medians_ms=medians,
                    ranges_ms={label: [min(v), max(v)] for label, v in samples.items()},
                    process_medians_ms=samples,
                    timed_tokens=timed_tokens,
                    tokens_per_second={
                        label: timed_tokens * 1000 / ms for label, ms in medians.items()
                    },
                    throughput_ranges={
                        label: [
                            timed_tokens * 1000 / max(v),
                            timed_tokens * 1000 / min(v),
                        ]
                        for label, v in samples.items()
                    },
                    speedup=medians["pytorch"] / medians["camblas"],
                )
            )
    summary = dict(manifest=manifest, rows=rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Llama 70B: PyTorch CUDA and CAMBLAS CUDA",
        "",
        "Speed-up = PyTorch / CAMBLAS; every process measures both backends with "
        "the same complete pretrained model. Backend and transfer order rotate.",
        "",
        f"Transfer contract: {manifest['transfer_contract']}",
        f"BF16 PyTorch reduction policy: {manifest['bf16_reduction']}; "
        "CAMBLAS uses FP32 accumulation without reduced-precision reductions. "
        "FP32 uses highest precision with TF32 disabled.",
        "Model loading, CPU parameter snapshots and numerical checks are excluded "
        "from timings and recorded separately. No CPU/disk parameter offload or "
        "weight quantisation is used. CUDA devices synchronise before and after calls.",
        "",
        "Throughput counts prompt tokens for prefill, new decode steps for decode "
        "(the first generated token comes from the prepared prefill), and all generated "
        "tokens for the complete request. Request tokens/s includes prefill and the labelled transfers.",
        "",
        "| Precision | Batch | Prompt | Phase | Mode | PyTorch ms | CAMBLAS ms | PyTorch tokens/s | CAMBLAS tokens/s | Speed-up |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row["medians_ms"]
        lines.append(
            f"| {row['dtype']} | {row['batch']} | {row['prompt_length']} | "
            f"{row['phase']} | {row['mode']} | {values['pytorch']:.3f} | "
            f"{values['camblas']:.3f} | {row['tokens_per_second']['pytorch']:.2f} | "
            f"{row['tokens_per_second']['camblas']:.2f} | {row['speedup']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "All process medians and ranges, identities, logits checks and "
            "changed-input checks remain in the JSON reports.",
        ]
    )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    """Run serial GPU workers with unchanged code and verified full-model outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--weights-manifest", type=Path)
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".frameworks/envs/llama/bin/python"
    )
    parser.add_argument(
        "--camblas-library", type=Path, default=ROOT / "build/cuda/libcamblas_cuda.so"
    )
    parser.add_argument(
        "--linear-entry", choices=["native", "matmul"], default="native"
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "bfloat16"], default=["bfloat16"]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument(
        "--bf16-reduction", choices=["full", "reduced"], default="reduced"
    )
    parser.add_argument("--fuse-rms", action="store_true")
    parser.add_argument("--fuse-mlp", action="store_true")
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--fuse-residual", action="store_true")
    parser.add_argument("--copy-weights-every-request", action="store_true")
    parser.add_argument("--toy-cpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or args.repetitions < 1 or args.warmups < 1:
        parser.error(
            "Use at least three fresh processes and positive warm-ups/repetitions"
        )
    if not args.toy_cpu and (
        not args.model_directory or not args.model_directory.is_dir()
    ):
        parser.error("Provide the local Llama 70B checkpoint directory")
    cpus = sorted(os.sched_getaffinity(0))[: args.threads]
    if len(cpus) != args.threads:
        parser.error("Requested host cores are not available")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        state="running",
        started=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,
        rounds=args.rounds,
        affinity=cpus,
        allocation=os.environ.get("SLURM_JOB_ID"),
        generated_tokens=args.generated_tokens,
        bf16_reduction=args.bf16_reduction,
        fusion=dict(
            rms_norm=args.fuse_rms,
            gated_mlp=args.fuse_mlp,
            qkv=args.fuse_qkv,
            residual_rms_norm=args.fuse_residual,
        ),
        toy_cpu=args.toy_cpu,
        records=[],
        source_sha256={
            str(path): digest(path)
            for path in [Path(__file__).resolve(), ROOT / "bench/llama_workload.py"]
        },
        transfer_contract="Every transfer prefill/request uploads all weights and token inputs, then returns final logits/tokens. Decode starts from prepared resident weights/KV cache."
        if args.copy_weights_every_request
        else "Weights and KV cache remain resident; transfer prefill/request copies token inputs and final logits/tokens. Decode starts from a prepared resident cache and copies its outputs.",
    )
    records = []

    def save():
        path = directory / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)

    save()
    try:
        with (ROOT / ".frameworks/measurement.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for dtype, batch, prompt in itertools.product(
                args.dtypes, args.batches, args.prompt_lengths
            ):
                for round_id in range(args.rounds):
                    name = f"{dtype}_b{batch}_p{prompt}_r{round_id}"
                    output = directory / (name + ".json")
                    command = [
                        "taskset",
                        "-c",
                        ",".join(map(str, cpus)),
                        str(args.python),
                        str(ROOT / "bench/llama_workload.py"),
                        "--dtype",
                        dtype,
                        "--batch",
                        str(batch),
                        "--prompt-length",
                        str(prompt),
                        "--generated-tokens",
                        str(args.generated_tokens),
                        "--round",
                        str(round_id),
                        "--warmups",
                        str(args.warmups),
                        "--repetitions",
                        str(args.repetitions),
                        "--linear-entry",
                        args.linear_entry,
                        "--bf16-reduction",
                        args.bf16_reduction,
                        "--output",
                        str(output),
                    ]
                    if args.toy_cpu:
                        command.append("--toy-cpu")
                    else:
                        command.extend(
                            [
                                "--model-directory",
                                str(args.model_directory.resolve()),
                                "--camblas-library",
                                str(args.camblas_library.resolve()),
                            ]
                        )
                    if args.weights_manifest:
                        command.extend(
                            ["--weights-manifest", str(args.weights_manifest.resolve())]
                        )
                    for flag in [
                        "fuse_rms",
                        "fuse_mlp",
                        "fuse_qkv",
                        "fuse_residual",
                        "copy_weights_every_request",
                    ]:
                        if getattr(args, flag):
                            command.append("--" + flag.replace("_", "-"))
                    manifest["current"] = name
                    save()
                    print(name, flush=True)
                    with (directory / (name + ".log")).open("w") as log:
                        subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.timeout,
                        )
                    record = json.loads(output.read_text())
                    assert (
                        record["state"] == "passed"
                        and record["changed_token_inputs_checked"]
                    )
                    assert record["toy_cpu"] == args.toy_cpu
                    if not args.toy_cpu:
                        assert record["parameters"] == 70553706496
                        assert record["matmul_precision"]["float32"] == "highest"
                        assert not record["matmul_precision"]["allow_tf32"]
                    records.append(record)
                    manifest["records"].append(str(output))
                    save()
            for record in records:
                for name, sha in record["source_sha256"].items():
                    assert digest(name) == sha, name
                for name, sha in record.get("library_sha256", {}).items():
                    assert digest(name) == sha, name
            assert manifest["source_sha256"] == {
                name: digest(name) for name in manifest["source_sha256"]
            }
        manifest["state"] = "passed"
        aggregate(records, directory, manifest)
    except BaseException as error:
        manifest.update(state="failed", error=repr(error))
        raise
    finally:
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
