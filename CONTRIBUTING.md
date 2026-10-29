# Development

Python 3.11+ is required. The Metal tests were last run on an M3 Max with Python
3.12 and MLX 0.32.2. Install the extras for your platform as described in the
[README](README.md), then:

```bash
python -m pip install -e '.[dev]'
ruff check python benchmarks tests tools
ruff format --check python benchmarks tests tools
FORGE_INSPECT_MODEL=build/forge-inspect-model PYTHONPATH=python python -m pytest -q
```

CI runs everything that doesn't need a GPU. Metal tests skip without mlx and
CUDA code is only compiled, so a green CI run says nothing about those backends.
Run the Metal tests locally before merging anything that touches `backends/`,
`paged_kv.py` or the engine.

When changing numerics, compare against an independent reference (the NumPy
reference in the tests, or Transformers via `forge-correctness`), not just the
previous output. Cache changes need tests for completion, cancellation,
allocation failure and page boundaries, since those are where pages leak.

Speedups need a baseline from the same machine and workload. `benchmarks/run_matrix.py`
does five warmups and three measured runs by default. Put new results in
`benchmark-results/` (ignored by git), and only commit raw data that a doc
actually cites. Model weights and build output stay out of git.
