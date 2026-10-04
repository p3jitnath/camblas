"""Test orchestration with synthetic records; these are not performance results."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "bench/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
with patch.dict("sys.modules", {"compare": compare}):
    cuda_spec = importlib.util.spec_from_file_location(
        "compare_cuda", ROOT / "bench/compare_cuda.py"
    )
    compare_cuda = importlib.util.module_from_spec(cuda_spec)
    cuda_spec.loader.exec_module(compare_cuda)
    with patch.dict("sys.modules", {"compare_cuda": compare_cuda}):
        gpu_spec = importlib.util.spec_from_file_location(
            "compare_gpu", ROOT / "bench/compare_gpu.py"
        )
        compare_gpu = importlib.util.module_from_spec(gpu_spec)
        gpu_spec.loader.exec_module(compare_gpu)


class ProcessRecordTests(unittest.TestCase):
    """Reject provenance mismatches and non-finite synthetic observations."""

    def test_process_record_validation(self):
        """Accept a complete record and reject each independently corrupted field."""
        identity = {"sha256": "expected-bridge", "core_sha256": "expected-core"}
        record = {
            "bridge_sha256": identity["sha256"],
            "core_identity": {"sha256": identity["core_sha256"]},
            "observed_backend": "camblas+openblas-compatibility",
            "affinity": [4, 5],
            "threads": 2,
            "warmups": 3,
            "repetitions": 5,
            "seconds": [0.001] * 5,
            "counters": [5] + [0] * 11,
            "outputs": [
                {"size": 2, "samples": [1.0, 2.0], "norm": 2.236, "max_abs": 2.0}
            ],
        }
        compare.validate_process_record(record, "camblas", identity, [4, 5], 5)
        corruptions = {
            "bridge_sha256": "different-bridge",
            "core_identity": {"sha256": "different-core"},
            "observed_backend": "openblas",
            "affinity": [5, 6],
            "threads": 1,
            "warmups": 0,
            "repetitions": 4,
            "seconds": [0.001, float("nan"), 0.001, 0.001, 0.001],
            "counters": [0] * 12,
            "outputs": [
                {
                    "size": 2,
                    "samples": [1.0, float("nan")],
                    "norm": 2.236,
                    "max_abs": 2.0,
                }
            ],
        }
        for field, value in corruptions.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare.validate_process_record(
                    record | {field: value}, "camblas", identity, [4, 5], 5
                )
        for field in ("norm", "max_abs"):
            invalid = copy.deepcopy(record)
            invalid["outputs"][0][field] = float("inf")
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare.validate_process_record(invalid, "camblas", identity, [4, 5], 5)
        for duration in (0.0, -0.001, float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                compare.validate_process_record(
                    record | {"seconds": [duration] * 5}, "camblas", identity, [4, 5], 5
                )
        for field in ("core_identity", "affinity", "seconds", "outputs", "counters"):
            incomplete = {
                name: value for name, value in record.items() if name != field
            }
            with self.subTest(missing=field), self.assertRaises(ValueError):
                compare.validate_process_record(
                    incomplete, "camblas", identity, [4, 5], 5
                )
        for backend in ("openblas", "nvpl"):
            # Vendor adapters do not load a CAMBLAS native core.
            vendor = record | {"observed_backend": backend, "core_identity": None}
            compare.validate_process_record(vendor, backend, identity, [4, 5], 5)


class CudaControlTests(unittest.TestCase):
    """Ensure saved comparisons use the intended files and actual content."""

    def test_loaded_control_binary_validation(self):
        """Reject wrong hashes, copied paths, missing bindings and extra cores."""
        identity = dict(
            library="/saved/libcamblas_cuda.so",
            library_sha256="saved-core",
            tensor_binding="/saved/_camblas_cuda_torch.so",
            tensor_binding_sha256="saved-binding",
        )
        correct = {
            identity["library"]: identity["library_sha256"],
            identity["tensor_binding"]: identity["tensor_binding_sha256"],
            "/vendor/libcublas.so": "vendor",
        }
        compare_gpu.validate_native_libraries(
            dict(library_sha256=correct), identity, control=True
        )
        corrupted = [
            correct | {identity["library"]: "candidate-core"},
            correct | {identity["tensor_binding"]: "candidate-binding"},
            {
                path.replace("/saved/", "/candidate/"): value
                for path, value in correct.items()
            },
            {
                path: value
                for path, value in correct.items()
                if path != identity["tensor_binding"]
            },
            correct | {"/extra/libcamblas_cuda.so": "saved-core"},
        ]
        for hashes in corrupted:
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                compare_gpu.validate_native_libraries(
                    dict(library_sha256=hashes), identity, control=True
                )

    def test_saved_control_content_changes(self):
        """Detect binary or source edits independently of a stale build record."""
        with tempfile.TemporaryDirectory(prefix="camblas-control-test-") as directory:
            root = Path(directory)
            core = root / "libcamblas_cuda.so"
            core.write_bytes(b"saved core")
            (root / "build.json").write_text('{"library_sha256": "stale"}')
            source = root / "source/backend.cu"
            source.parent.mkdir()
            source.write_text("saved source")
            identity = compare_gpu.control_identity(core)
            self.assertEqual(identity["library_sha256"], compare_gpu.digest(core))
            self.assertIsNone(identity["tensor_binding"])
            core.write_bytes(b"changed core")
            self.assertNotEqual(compare_gpu.control_identity(core), identity)
            core.write_bytes(b"saved core")
            self.assertEqual(compare_gpu.control_identity(core), identity)
            source.write_text("changed source")
            self.assertNotEqual(compare_gpu.control_identity(core), identity)


class ComparisonTests(unittest.TestCase):
    """Validate comparison policy and reporting without running numerical work."""

    def invoke(self, arguments):
        """Capture the comparison driver output for synthetic records."""
        output = io.StringIO()
        with (
            patch("sys.argv", ["compare.py"] + arguments),
            contextlib.redirect_stdout(output),
        ):
            compare.main()
        return output.getvalue()

    def test_default_policy_rejects_allocator_override(self):
        """Reject inherited allocator overrides in the default-policy comparison."""
        with (
            patch.dict(os.environ, {"MALLOC_TRIM_THRESHOLD_": "1"}, clear=True),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.invoke(["--threads", "1", "--dry-run"])

    def test_rotated_execution_and_counts(self):
        """Verify backend rotation and win counts using synthetic process records."""
        order = []
        with tempfile.TemporaryDirectory(prefix="camblas-tools-test-") as directory:
            root = Path(directory)
            for backend in compare.BACKENDS:
                for file in (
                    root / ".frameworks/prefix" / backend / "lib/libframework_blas.so",
                    root / ".frameworks/envs" / backend / "bin/python",
                ):
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()
            (root / ".frameworks/prefix/camblas/lib/libcamblas_sve_nr4.so").touch()

            def fake_run(command, **kwargs):
                """Write a synthetic record in place of running a benchmark.

                Parameters
                ----------
                command : list of str
                    Workload command containing backend, repetition and output
                    arguments used to construct the synthetic result.
                **kwargs : dict
                    Ignored subprocess options accepted by the test double.
                """
                backend = command[command.index("--backend") + 1]
                order.append(backend)
                repetitions = int(command[command.index("--repetitions") + 1])
                counters = [0] * 12
                counters[0] = repetitions
                record = {
                    "bridge_sha256": compare.hashlib.sha256(b"").hexdigest(),
                    "core_identity": {
                        "sha256": compare.hashlib.sha256(b"").hexdigest()
                    },
                    "observed_backend": "camblas+openblas-compatibility"
                    if backend == "camblas"
                    else backend,
                    "seconds": [0.001 * (compare.BACKENDS.index(backend) + 1)]
                    * repetitions,
                    "affinity": [7],
                    "threads": 1,
                    "warmups": 3,
                    "repetitions": repetitions,
                    "counters": counters,
                    "outputs": [
                        {"size": 1, "samples": [1.0], "norm": 1.0, "max_abs": 1.0}
                    ],
                }
                Path(command[command.index("--output") + 1]).write_text(
                    json.dumps(record)
                )

            with (
                patch.object(compare, "ROOT", root),
                patch.object(
                    compare.socket, "gethostname", return_value="compute-test"
                ),
                patch.object(compare.subprocess, "run", side_effect=fake_run),
                patch.object(compare.os, "sched_getaffinity", return_value={7}),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.invoke(
                    [
                        "--frameworks",
                        "numpy",
                        "--workloads",
                        "mlp",
                        "--threads",
                        "1",
                        "--repetitions",
                        "5",
                        "--output",
                        str(root / "results/test"),
                    ]
                )
            result = json.loads((root / "results/test/complete.json").read_text())
            self.assertEqual(result["cases"], 1)
            self.assertEqual(result["wins"], {"openblas": 1, "nvpl": 1})
            self.assertEqual(
                order,
                [
                    "camblas",
                    "openblas",
                    "nvpl",
                    "openblas",
                    "nvpl",
                    "camblas",
                    "nvpl",
                    "camblas",
                    "openblas",
                ],
            )


if __name__ == "__main__":
    unittest.main()
