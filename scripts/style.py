#!/usr/bin/env python3
"""Check source formatting and NumPy-style docstrings, or apply formatting fixes."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Run the project's Python and C style tools from the active environment.

    Notes
    -----
    Install ``configs/style-requirements.txt`` into a project-local environment
    first. Checks are read-only unless ``--fix`` is supplied. Source discovery
    excludes generated builds and downloaded framework dependencies.

    Raises
    ------
    subprocess.CalledProcessError
        If a style tool fails or reports an unresolved violation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Apply formatting fixes")
    args = parser.parse_args()
    clang_format = Path(sys.executable).parent / "clang-format"
    if importlib.util.find_spec("ruff") is None or not clang_format.is_file():
        parser.error(
            "Install configs/style-requirements.txt with this Python interpreter first"
        )

    python_dirs = ["bench", "scripts", "tests"]
    c_files = sorted(
        str(path.relative_to(ROOT))
        for directory in ("framework", "include", "src", "tests")
        for path in (ROOT / directory).rglob("*")
        if path.suffix in (".c", ".h")
    )
    ruff = [sys.executable, "-m", "ruff"]
    if args.fix:
        commands = [
            [*ruff, "check", "--select", "I", "--fix", *python_dirs],
            [*ruff, "format", *python_dirs],
            [str(clang_format), "-i", *c_files],
            [*ruff, "check", *python_dirs],
        ]
    else:
        commands = [
            [*ruff, "check", *python_dirs],
            [*ruff, "format", "--check", *python_dirs],
            [str(clang_format), "--dry-run", "--Werror", *c_files],
        ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
