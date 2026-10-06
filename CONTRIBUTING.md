# Contributing to CAMBLAS

Start with the [README](README.md), create a branch from `main`, and keep each pull request focused. Describe proposed changes to the API, numerical contracts or dispatch policy before implementing them.

```bash
git switch -c your-change
make test CC=gcc-14 PYTHON=python3.11
python3.11 -m compileall -q camblas scripts bench tests
git diff --check
```

Use [ASD-STE100-CONV](https://github.com/p3jitnath/esm.md/blob/main/standards/ASD-STE100-CONV.md) for all documentation, comments, docstrings and work reports. Write connected technical prose in British English, give each paragraph a clear purpose, and preserve equations, units, identifiers and numerical contracts. Keep each prose paragraph or list item on one physical Markdown line, while preserving the layout of code, tables and equations.

Use four-space indentation and descriptive names, with comments that explain choices the code cannot make clear. Preserve NumPy-style Python docstrings, and document the layout, ownership and failure contracts of C interfaces. The pinned project tools check formatting; use `make format` with the same `PYTHON` to apply fixes.

```bash
python3.11 -m venv .frameworks/style-env
.frameworks/style-env/bin/python -m pip install -r configs/style-requirements.txt
make format-check PYTHON=.frameworks/style-env/bin/python
```

Include the evidence needed to assess each change:

- **Correctness:** keep tests that protect required behaviour, and remove duplicate or obsolete cases. Extend an existing independent test where possible; cover the precisions, layouts, alpha/beta values, changed inputs and dispatch boundaries affected by the change. Alternative multiplication algorithms also require cancellation and exceptional-input checks, because overflow checks alone do not establish accuracy. Preserve caller-owned workspace and synchronous executor contracts, and use sanitiser checks when the change affects memory or synchronisation.
- **Performance:** measure the candidate and an unchanged CAMBLAS control under the same inputs, precision, hardware, affinity, thread settings and transfer policy. Use OpenBLAS and NVPL for CPU comparisons, PyTorch for CUDA comparisons, and the same SGLang engine for LLM comparisons. Run at least three fresh processes with rotated backend order on quiet allocated nodes, then report medians, ranges and regressions. Identify results that were not remeasured, and record any allocator or OpenMP settings that differ from the defaults.
- **Reproduction:** record commands, compiler and dependency versions, library identities and evidence of actual backend calls. Explain how the change works, where it applies, and whether it changes the ABI or rounding behaviour.
- **Hygiene:** keep binaries, wheels, dependencies, credentials and generated timing records out of Git. Use ignored `build/`, `.frameworks/` or `bench/results/` directories; keep concise verified reports under `results/`.

Contributions use the [MIT licence](LICENSE); preserve copyright notices and required third-party attribution.
