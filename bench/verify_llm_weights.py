#!/usr/bin/env python3
"""Verify a complete local checkpoint against its pinned weight manifest."""

import argparse
import hashlib
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest_identity(manifest):
    """Identify parsed checkpoint metadata independently of JSON formatting."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check_verified_files(directory, record):
    """Reject a stale checkpoint verification before or after a measured worker."""
    for name, expected in record["weights"].items():
        stat = (directory / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            expected["bytes"],
            expected["mtime_ns"],
        ):
            raise ValueError(f"Checkpoint verification is stale; rehash {name}")
    for name, expected in record.get("files", {}).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Verified checkpoint metadata changed: {name}")


def verify(directory, manifest, output):
    """Hash every shard and check tensor storage, configuration and file identities.

    Parameters
    ----------
    directory : pathlib.Path
        Existing local checkpoint directory; this function downloads nothing.
    manifest : dict
        Public model identity, expected configuration and shard SHA256 hashes.
    output : pathlib.Path
        New JSON verification record for ``compare_llm.py``.

    Returns
    -------
    dict
        Verified model identity, tensors, stored elements and shard file identities.
    """
    if output.exists():
        raise FileExistsError(output)
    directory = directory.resolve()
    record = dict(
        state="running",
        model=manifest["model"],
        revision=manifest["revision"],
        manifest_sha256=manifest_identity(manifest),
        directory=str(directory),
        verification_started=datetime.now(timezone.utc).isoformat(),
        weights={},
        configuration=manifest["configuration"],
        files=manifest.get("files", {}),
    )
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if set(index["weight_map"].values()) != set(manifest["weights"]):
        raise ValueError("Checkpoint index does not match the pinned shard list")
    for name, expected in manifest["configuration"].items():
        if config.get(name) != expected:
            raise ValueError(f"Unexpected model configuration: {name}")
    for name, expected in manifest.get("files", {}).items():
        if Path(name).name != name:
            raise ValueError("Checkpoint metadata names must be plain filenames")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint metadata SHA256 mismatch: {name}")
    dtype_bytes = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
    allowed_dtypes = manifest.get("dtypes", ["BF16"])
    names = set()
    parameters = 0
    for name, expected in sorted(manifest["weights"].items()):
        if Path(name).name != name:
            raise ValueError("Shard names must be plain filenames")
        path = directory / name
        before = path.stat()
        if before.st_size != expected["bytes"]:
            raise ValueError(f"Shard size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(32 * 2**20), b""):
                digest.update(block)
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Shard SHA256 mismatch: {name}")
        with path.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            if not 2 <= header_size <= min(before.st_size - 8, 64 * 2**20):
                raise ValueError(f"Invalid safetensors header: {name}")
            header = json.loads(source.read(header_size))
        intervals = []
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            if key in names or index["weight_map"].get(key) != name:
                raise ValueError(f"Duplicate or misplaced tensor: {key}")
            if tensor["dtype"] not in allowed_dtypes or any(
                not isinstance(size, int) or size < 0 for size in tensor["shape"]
            ):
                raise ValueError(f"Invalid tensor metadata: {key}")
            elements = math.prod(tensor["shape"])
            start, end = tensor["data_offsets"]
            if not 0 <= start <= end <= before.st_size - 8 - header_size:
                raise ValueError(f"Invalid tensor byte offsets: {key}")
            if end - start != dtype_bytes[tensor["dtype"]] * elements:
                raise ValueError(f"Tensor byte count mismatch: {key}")
            intervals.append((start, end))
            parameters += elements
            names.add(key)
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f"Overlapping tensor storage: {name}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Shard changed during verification: {name}")
        record["weights"][name] = dict(
            sha256=digest.hexdigest(), bytes=after.st_size, mtime_ns=after.st_mtime_ns
        )
        print(f"Verified {name}", flush=True)
    if names != set(index["weight_map"]) or (
        "parameters" in manifest and parameters != manifest["parameters"]
    ):
        raise ValueError("Checkpoint tensor set or parameter count mismatch")
    record.update(
        state="passed",
        storage_elements=parameters,
        tensors=len(names),
        verification_finished=datetime.now(timezone.utc).isoformat(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output)
    return record


def main():
    """Validate a downloaded checkpoint without importing a GPU framework."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/llama31-70b-weights.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = verify(
        args.model_directory, json.loads(args.manifest.read_text()), args.output
    )
    print(json.dumps({key: value for key, value in record.items() if key != "weights"}))


if __name__ == "__main__":
    main()
