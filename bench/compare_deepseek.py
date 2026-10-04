"""Compare full DeepSeek inference in fresh, serial, rotated process groups."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_weights(checkpoint, manifest, output):
    """Hash every converted shard once, or check a previously verified file identity.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Complete TP4 checkpoint directory.
    manifest : dict
        Public pinned identities or a passed local verification record.
    output : pathlib.Path
        New verification record for the workers.

    Returns
    -------
    pathlib.Path
        Passed verification record including each shard's size and modification time.
    """
    if manifest.get("tensor_parallel") != 4 or set(manifest["converted_weights"]) != {
        f"model{rank}-mp4.safetensors" for rank in range(4)
    }:
        raise ValueError("Provide every shard of the pinned TP4 checkpoint")
    cached = manifest.get("state") == "passed"
    for name, expected in manifest["converted_weights"].items():
        if Path(name).name != name:
            raise ValueError("Checkpoint shard names must be plain filenames")
        path = checkpoint / name
        before = path.stat()
        if before.st_size != expected["bytes"]:
            raise ValueError(f"Checkpoint shard size mismatch: {name}")
        if cached:
            if before.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(
                    f"Weight verification is stale; rehash from the public manifest: {name}"
                )
        else:
            digest = hashlib.sha256()
            offset = 0
            with path.open("rb") as source:
                for block in iter(lambda: source.read(32 * 2**20), b""):
                    digest.update(block)
                    os.posix_fadvise(
                        source.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED
                    )
                    offset += len(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Pinned checkpoint SHA256 mismatch: {name}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Checkpoint changed during verification: {name}")
        expected["mtime_ns"] = after.st_mtime_ns
    manifest.update(state="passed", directory=str(checkpoint.resolve()))
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def check_outputs(actual, expected):
    """Require all vocabulary logits and greedy tokens to match bitwise.

    Parameters
    ----------
    actual, expected : dict
        Matching rank-zero worker records for the same complete checkpoint.

    Raises
    ------
    ValueError
        If model identities, inputs, full logits or generated tokens differ.
    """
    for field in (
        "revision",
        "input_sha256",
        "prompt_length",
        "generated_tokens",
        "reference_fix",
    ):
        if actual[field] != expected[field]:
            raise ValueError(f"Benchmark identity differs: {field}")
    outputs = [(actual["changed_tokens"], expected["changed_tokens"])]
    for phase in ("prefill", "decode", "request"):
        for mode in ("resident", "transfer"):
            outputs.append(
                (
                    actual["measurements"][phase][mode],
                    expected["measurements"][phase][mode],
                )
            )
    if any(
        a["tokens"] != b["tokens"]
        or a["logits_sha256"] != b["logits_sha256"]
        or a["logits"] != b["logits"]
        for a, b in outputs
    ):
        raise ValueError("Full vocabulary logits or generated tokens differ")


def summarise(records):
    """Aggregate process medians and ranges without hiding slower cases.

    Parameters
    ----------
    records : list of dict
        Passed records from matching fresh process groups.

    Returns
    -------
    list of dict
        Phase timings, tokens/second and speed-ups against PyTorch and control.
    """
    rows = []
    for prompt in sorted({r["prompt_length"] for r in records}):
        selected = [r for r in records if r["prompt_length"] == prompt]
        for phase in ("prefill", "decode", "request"):
            for mode in ("resident", "transfer"):
                samples = {
                    label: [
                        r["measurements"][phase][mode]["median_ms"]
                        for r in selected
                        if r["label"] == label
                    ]
                    for label in ("pytorch", "control", "camblas")
                }
                medians = {
                    label: statistics.median(values)
                    for label, values in samples.items()
                }
                generated = selected[0]["generated_tokens"]
                numerator = (
                    prompt if phase == "prefill" else generated - (phase == "decode")
                )
                rows.append(
                    dict(
                        prompt_length=prompt,
                        generated_tokens=generated,
                        phase=phase,
                        mode=mode,
                        process_medians_ms=samples,
                        medians_ms=medians,
                        ranges_ms={
                            label: [min(values), max(values)]
                            for label, values in samples.items()
                        },
                        tokens_per_second={
                            label: numerator * 1000 / value
                            for label, value in medians.items()
                        },
                        speedup=medians["pytorch"] / medians["camblas"],
                        control_speedup=medians["control"] / medians["camblas"],
                    )
                )
    return rows


def main():
    """Run three or more rotated rounds with unchanged checkpoint and control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".frameworks/envs/deepseek/bin/python"
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--control-library", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or min(args.warmups, args.repetitions) < 1:
        parser.error(
            "Use at least three fresh rounds and positive warm-ups/repetitions"
        )
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    verified = verify_weights(
        args.checkpoint,
        json.loads(args.weights_manifest.read_text()),
        directory / "verified_weights.json",
    )
    if args.verify_only:
        return
    if not args.library or not args.control_library:
        parser.error(
            "Comparison requires both candidate and unchanged control libraries"
        )
    manifest = dict(
        state="running",
        started=datetime.now(timezone.utc).isoformat(),
        allocation=os.environ.get("SLURM_JOB_ID"),
        node=os.uname().nodename,
        command=sys.argv,
        checkpoint=str(args.checkpoint.resolve()),
        reference_directory=str(args.reference_directory.resolve()),
        weights_manifest=str(verified),
        rounds=args.rounds,
        transfer_contract="Weights and prepared decode caches remain resident. CPU Engram index download, full-table row gathers, row uploads and dequantisation are timed on every path. Transfer requests additionally upload the prompt and download final vocabulary logits and tokens. Decode excludes prepared prefill; request tokens/s includes prefill.",
        precision="Checkpoint's original packed FP4 experts and FP8 dense weights, E8M0 scales, BF16 activations and FP32 accumulation; TF32 disabled.",
        records=[],
    )
    status = directory / "manifest.json"
    records = []
    try:
        for prompt in args.prompt_lengths:
            for round_id in range(args.rounds):
                labels = ["pytorch", "control", "camblas"]
                labels = labels[round_id % 3 :] + labels[: round_id % 3]
                round_records = []
                for label in labels:
                    name = f"p{prompt}_{label}_r{round_id}"
                    output = directory / (name + ".json")
                    command = [
                        str(args.python.absolute()),
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nnodes=1",
                        "--nproc_per_node=4",
                        str(ROOT / "bench/deepseek_workload.py"),
                        "--checkpoint",
                        str(args.checkpoint.resolve()),
                        "--reference-directory",
                        str(args.reference_directory.resolve()),
                        "--weights-manifest",
                        str(verified),
                        "--backend",
                        "pytorch" if label == "pytorch" else "camblas",
                        "--prompt-length",
                        str(prompt),
                        "--generated-tokens",
                        str(args.generated_tokens),
                        "--warmups",
                        str(args.warmups),
                        "--repetitions",
                        str(args.repetitions),
                        "--round",
                        str(round_id),
                        "--output",
                        str(output),
                    ]
                    if label != "pytorch":
                        command += [
                            "--library",
                            str(
                                (
                                    args.control_library
                                    if label == "control"
                                    else args.library
                                ).resolve()
                            ),
                        ]
                    print("START", name, flush=True)
                    environment = os.environ.copy()
                    environment.update(
                        OMP_NUM_THREADS="16",
                        OPENBLAS_NUM_THREADS="16",
                        PYTORCH_ALLOC_CONF="expandable_segments:True",
                    )
                    with (directory / (name + ".log")).open("w") as log:
                        subprocess.run(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.timeout,
                        )
                    ranks = [
                        json.loads(
                            output.with_name(
                                output.stem + f".rank{rank}.json"
                            ).read_text()
                        )
                        for rank in range(4)
                    ]
                    if any(r["state"] != "passed" for r in ranks):
                        raise ValueError("A model rank did not pass verification")
                    for other in ranks[1:]:
                        check_outputs(other, ranks[0])
                    observation = ranks[0]
                    observation["label"] = label
                    records.append(observation)
                    round_records.append(observation)
                    manifest["records"].append(
                        output.with_name(output.stem + ".rank0.json").name
                    )
                    status.write_text(json.dumps(manifest, indent=2) + "\n")
                    print("DONE", name, flush=True)
                reference = next(r for r in round_records if r["label"] == "pytorch")
                for other in round_records:
                    check_outputs(other, reference)
                previous = next(
                    r
                    for r in records
                    if r["label"] == "pytorch" and r["prompt_length"] == prompt
                )
                check_outputs(reference, previous)
        rows = summarise(records)
        (directory / "summary.json").write_text(
            json.dumps(
                dict(
                    rows=rows,
                    bitwise_full_logits=True,
                    exact_greedy_tokens=True,
                    changed_inputs=True,
                ),
                indent=2,
            )
            + "\n"
        )
        lines = [
            "| Prompt | Phase | Mode | PyTorch ms | CAMBLAS ms | PyTorch tokens/s | CAMBLAS tokens/s | Speed-up |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            ms, throughput = row["medians_ms"], row["tokens_per_second"]
            lines.append(
                f"| {row['prompt_length']} | {row['phase']} | {row['mode']} | {ms['pytorch']:.3f} | {ms['camblas']:.3f} | {throughput['pytorch']:.2f} | {throughput['camblas']:.2f} | {row['speedup']:.3f}× |"
            )
        (directory / "comparison.md").write_text("\n".join(lines) + "\n")
        manifest.update(
            state="passed",
            bitwise_full_logits=True,
            exact_greedy_tokens=True,
            changed_inputs=True,
        )
    except BaseException as error:
        manifest.update(state="failed", error=repr(error))
        raise
    finally:
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        status.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
