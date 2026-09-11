"""Test orchestration with synthetic records; these are not performance results."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "bench/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


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


class ImportSafetyTests(unittest.TestCase):
    """Keep script imports free of argument parsing and native-library loading."""

    def test_entry_points_do_not_execute_on_import(self):
        """Import every standalone tool without launching builds or numerical work."""
        paths = (
            "bench/compare.py",
            "bench/workload.py",
            "scripts/build.py",
            "scripts/frameworks.py",
            "scripts/style.py",
            "tests/test_framework_bridge.py",
            "tests/test_framework_smoke.py",
            "tests/test_mlp_packing.py",
        )
        # Import safety must be testable without NumPy/PyTorch installations.
        # Any attempt to parse a command or load BLAS here is a regression.
        with (
            patch.dict("sys.modules", {"numpy": types.ModuleType("numpy")}),
            patch("argparse.ArgumentParser.parse_args") as parse_args,
            patch("ctypes.CDLL") as load_library,
            patch("subprocess.run") as run_command,
        ):
            for index, path in enumerate(paths):
                with self.subTest(path=path):
                    module_spec = importlib.util.spec_from_file_location(
                        f"camblas_import_test_{index}", ROOT / path
                    )
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    self.assertTrue(callable(module.main))
            parse_args.assert_not_called()
            load_library.assert_not_called()
            run_command.assert_not_called()


class ComparisonTests(unittest.TestCase):
    """Validate comparison policy and reporting without running numerical work."""

    def invoke(self, arguments):
        """Invoke the comparison entry point and capture its standard output.

        Parameters
        ----------
        arguments : list of str
            Command-line arguments after the executable name.

        Returns
        -------
        str
            Captured output from the comparison driver.
        """
        output = io.StringIO()
        with (
            patch("sys.argv", ["compare.py"] + arguments),
            contextlib.redirect_stdout(output),
        ):
            compare.main()
        return output.getvalue()

    def test_full_matrix_dry_run(self):
        """Check that the full matrix schedules exactly 540 fresh processes."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(compare.os, "sched_getaffinity", return_value=set(range(64))),
        ):
            output = self.invoke(["--full", "--dry-run"])
        commands = [line for line in output.splitlines() if line.startswith("taskset ")]
        self.assertEqual(len(commands), 60 * 3 * 3)
        self.assertIn("60 cases, 3 fresh-process rounds", output)
        self.assertFalse(
            any("--framework numpy --workload backward" in line for line in commands)
        )

    def test_requires_three_rounds(self):
        """Reject comparisons that request fewer than three independent rounds."""
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            self.invoke(["--rounds", "2", "--dry-run"])
        self.assertEqual(error.exception.code, 2)

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
