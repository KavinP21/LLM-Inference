# Development

Python 3.11 or later is required; local Metal regression uses Python 3.12 and MLX
0.32.2. Build the C++ host tools and install the platform extras as described in
the [README](README.md).

```bash
python -m pip install -e '.[dev]'
ruff check python benchmarks tests tools/forge_chat.py
ruff format --check python benchmarks tests tools/forge_chat.py
FORGE_INSPECT_MODEL=build/forge-inspect-model PYTHONPATH=python python -m pytest -q
```

Linux CI runs portable tests without MLX or model downloads. Metal and CUDA tests
need their respective devices; a skipped backend suite does not validate that
backend. Match numerical changes against independent references and test cache
ownership across completion, cancellation, allocation failure, and page boundaries.

Keep commits focused on an actual code or documentation change, describe its
behavior and validation, and commit at the time of the change. Save experiment
configuration, raw observations, source identity, and failed gates alongside the
analysis. A speedup needs a controlled baseline on the same hardware and workload.

Historical checkpoints are immutable. Use their source archives or recorded
revision for old-result audits. New source, model, corpus, or method changes use
a fresh result directory; do not rewrite hashes or weaken a failed gate. Weights,
environments, build output, and temporary benchmark output stay outside Git.
