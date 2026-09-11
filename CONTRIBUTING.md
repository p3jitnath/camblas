# Contributing to CAMBLAS

Start with the [README](README.md), create a branch from `main`, and keep each pull request focused. Discuss API, numerical-contract or dispatch-policy changes before implementing them.

```bash
git switch -c your-change
make test CC=gcc-14 PYTHON=python3.11
python3.11 -m compileall -q scripts bench tests
git diff --check
```

Use four-space indentation, descriptive names and comments explaining non-obvious choices. Document Python helpers in NumPy style and C interfaces with their layout, ownership and failure contracts. Check formatting with the pinned project-local tools; use `make format` with the same `PYTHON` to apply fixes.

```bash
python3.11 -m venv .frameworks/style-env
.frameworks/style-env/bin/python -m pip install -r configs/style-requirements.txt
make format-check PYTHON=.frameworks/style-env/bin/python
```

Include the following evidence and checks with each relevant change:

- **Correctness:** add an independent regression test covering affected precisions, transposes, padding, alpha/beta, changed inputs and dispatch boundaries. Preserve caller-owned workspace and synchronous executor contracts; use sanitiser checks where relevant.
- **Performance:** compare against an unchanged CAMBLAS control, OpenBLAS and NVPL with identical inputs, affinity and thread settings. Use at least three fresh-process rounds with rotated order; report medians, ranges and regressions. Record node-sharing conditions and confirm gains on idle allocated CPUs. Test affected cases first, identify rows not remeasured with the candidate, and report any non-default allocator or OpenMP settings separately.
- **Reproduction:** include commands, compiler/dependency versions and library identities; verify actual backend calls. Explain the mechanism, limitations and any ABI or rounding changes.
- **Hygiene:** keep binaries, wheels, dependencies, results, credentials and internal documents out of Git. Use ignored `build/`, `.frameworks/` and `results/` directories, and keep prose concise in British English.

Contributions are accepted under the [MIT licence](LICENSE); preserve copyright notices and any required third-party attribution.
