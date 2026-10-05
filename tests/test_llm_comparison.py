"""Reject invalid full-model comparisons before publishing throughput."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from bench.compare_llm import compare_outputs


class LlmComparisonTests(unittest.TestCase):
    """Check the logit contract and evidence covering every model rank."""

    def fixtures(self, parent):
        """Create paired, repeated two-rank vocabulary captures."""
        directories = [parent / name for name in ("stock", "camblas")]
        for directory in directories:
            directory.mkdir()
            files = []
            for rank in range(2):
                for step in range(3):
                    name = f"rank{rank}-step{step}.pt"
                    torch.save(
                        dict(
                            rank=rank,
                            prediction_position=1,
                            logits=torch.tensor([[1.0, -0.0, 3.0 + (step == 2)]]),
                        ),
                        directory / name,
                    )
                    files.append(name)
            result = dict(
                state="passed",
                input_ids=[1, 2],
                changed_input_ids=[1, 3],
                input_sha256="synthetic",
                case=dict(
                    model="synthetic",
                    prompt_length=2,
                    generated_tokens=2,
                    accuracy_tokens=1,
                    tensor_parallel=2,
                ),
                samples=[dict(output_ids=[7, 9])],
                accuracy=[
                    dict(output_ids=[7]),
                    dict(output_ids=[7]),
                    dict(output_ids=[8]),
                ],
                accuracy_files=files,
            )
            (directory / "result.json").write_text(json.dumps(result))
        return directories

    def test_full_vocabulary_and_bitwise_contract(self):
        """A nonwinning logit change or signed-zero change must fail strict checks."""
        for column, value, message in (
            (0, 1.5, "Full vocabulary logits exceeded"),
            (1, 0.0, "not bitwise identical"),
        ):
            with (
                self.subTest(column=column),
                tempfile.TemporaryDirectory() as temporary,
            ):
                reference, actual = self.fixtures(Path(temporary))
                result = compare_outputs(reference, actual, atol=0, rtol=0)
                self.assertTrue(result["bitwise"])
                self.assertEqual(result["vocabulary_values_checked"], 18)
                for path in actual.glob("*.pt"):
                    record = torch.load(path, weights_only=True)
                    record["logits"][0, column] = value
                    torch.save(record, path)
                with self.assertRaisesRegex(ValueError, message):
                    compare_outputs(reference, actual, atol=0, rtol=0)

    def test_unchanged_prompt_evidence_is_rejected(self):
        """A cached vocabulary vector cannot pass the changed-input check."""
        with tempfile.TemporaryDirectory() as temporary:
            reference, actual = self.fixtures(Path(temporary))
            for path in actual.glob("*.pt"):
                record = torch.load(path, weights_only=True)
                record["logits"][0, 2] = 3.0
                torch.save(record, path)
            with self.assertRaisesRegex(ValueError, "did not change"):
                compare_outputs(reference, actual, atol=0, rtol=0)

    def test_missing_rank_and_prediction_are_rejected(self):
        """Throughput cannot pass with missing rank evidence or shifted tokens."""
        with tempfile.TemporaryDirectory() as temporary:
            reference, actual = self.fixtures(Path(temporary))
            path = actual / "result.json"
            result = json.loads(path.read_text())
            incomplete = {**result, "accuracy_files": result["accuracy_files"][:3]}
            path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, "Missing model rank"):
                compare_outputs(reference, actual, atol=0, rtol=0)
            path.write_text(json.dumps(result))
            capture = actual / result["accuracy_files"][0]
            record = torch.load(capture, weights_only=True)
            record["prediction_position"] += 1
            torch.save(record, capture)
            with self.assertRaisesRegex(ValueError, "misordered accuracy prediction"):
                compare_outputs(reference, actual, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
