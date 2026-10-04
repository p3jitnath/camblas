"""Exercise checkpoint identity checks without loading weights or a GPU."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from bench.verify_llama_weights import verify


class LlamaWeightsTests(unittest.TestCase):
    """Reject changed tensor bytes and inconsistent checkpoint metadata."""

    def fixture(self, directory, *, dtype="BF16", offset=(0, 8)):
        """Create a tiny safetensors-format checkpoint for identity checks."""
        name = "model-00001-of-00001.safetensors"
        key = "model.layers.0.weight"
        header = json.dumps(
            {key: dict(dtype=dtype, shape=[2, 2], data_offsets=list(offset))}
        ).encode()
        payload = struct.pack("<Q", len(header)) + header + bytes(8)
        (directory / name).write_bytes(payload)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map={key: name}))
        )
        (directory / "config.json").write_text(json.dumps(dict(model_type="llama")))
        return dict(
            model="synthetic-checkpoint",
            revision="synthetic",
            configuration=dict(model_type="llama"),
            parameters=4,
            weights={
                name: dict(
                    bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
                )
            },
        )

    def test_verified_identity_and_output_ownership(self):
        """Retain hashes and stat identities and preserve an existing report."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.fixture(directory)
            output = directory / "verified.json"
            record = verify(directory, manifest, output)
            self.assertEqual(record["state"], "passed")
            self.assertEqual(record["parameters"], 4)
            self.assertEqual(record["tensors"], 1)
            self.assertEqual(json.loads(output.read_text()), record)
            with self.assertRaises(FileExistsError):
                verify(directory, manifest, output)

    def test_changed_bytes_size_and_configuration_are_rejected(self):
        """A same-size data change cannot reuse the pinned shard identity."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.fixture(directory)
            name = next(iter(manifest["weights"]))
            path = directory / name
            original = path.read_bytes()
            path.write_bytes(original[:-1] + b"\x01")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(directory, manifest, directory / "changed.json")
            path.write_bytes(original + b"\x00")
            with self.assertRaisesRegex(ValueError, "size"):
                verify(directory, manifest, directory / "size.json")
            path.write_bytes(original)
            (directory / "config.json").write_text('{"model_type": "other"}')
            with self.assertRaisesRegex(ValueError, "configuration"):
                verify(directory, manifest, directory / "config.json.out")

    def test_invalid_tensor_storage_and_parameter_counts_are_rejected(self):
        """Validate BF16 byte ownership after verifying each complete shard."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for dtype, offset in [("F32", (0, 8)), ("BF16", (0, 9)), ("BF16", (0, 6))]:
                manifest = self.fixture(directory, dtype=dtype, offset=offset)
                with self.assertRaises(ValueError):
                    verify(directory, manifest, directory / "invalid.json")
            manifest = self.fixture(directory)
            manifest["parameters"] = 5
            with self.assertRaisesRegex(ValueError, "parameter count"):
                verify(directory, manifest, directory / "count.json")


if __name__ == "__main__":
    unittest.main()
