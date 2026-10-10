# Qwen3.5-4B evidence

All measurements use one Apple M3 Max with 128 GB unified memory, MLX 0.32.2
and MLX-VLM 0.7.6. The full checkpoint is `mlx-community/Qwen3.5-4B-4bit`,
revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`. Weights are excluded.
The [usage and numerical contract](../../../docs/qwen35.md) distinguish the
upstream model graph from native Forge execution.

## Preserved development runs

| Report | Source | Interpretation |
| --- | --- | --- |
| [Core initial](core-initial.json) | `04be7ab` | 17 ordinary-reference continuation tokens across three inputs (longest 6,860 tokens); synthetic OCR, follow-up, video order, mixed modalities, thinking and cleanup pass. |
| [Agents initial](agents-initial.json) | `04be7ab` | Both coding modes and single documents pass; delegated documents fail. Retained diagnostic, including child/tool failures. |
| [Agents revised](agents-revised-7239c91.json) | `7239c91` | All four development cases pass; two completed children per delegated case. Delegation is slower. |
| [Speculation initial](speculative-initial.json) | `7239c91` | One forced block at each accepted prefix 0–4 passes. The cancellation case did **not** enter a verification block. Timings lack explicit per-workload warmups; they are diagnostic evidence only. |

Reports retain their original bytes and exact source identity. They are not
silently relabeled as results from subsequent fixes. The small tasks were used
during development, and do not constitute a held-out general-agent benchmark.

## Reproduction

Each driver refuses to replace an existing output. Run from the repository root:

```bash
.venv/bin/python benchmarks/qualify_qwen35.py \
  --model models/qwen3.5-4b-mlx --output /tmp/q35-core.json \
  --full-context-check --sampling-smoke

.venv/bin/python benchmarks/qualify_qwen35_speculative.py \
  --model models/qwen3.5-4b-mlx --output /tmp/q35-spec.json

PYTHONPATH=python .venv/bin/python benchmarks/qualify_qwen35_workers.py \
  --image /tmp/q35-core-fixtures/text.png --output /tmp/q35-workers.json

PYTHONPATH=python .venv/bin/python benchmarks/qualify_qwen35_ui.py \
  --image /tmp/q35-core-fixtures/text.png --output /tmp/q35-ui.json
```

`evaluate_agents.py --config examples/agents-qwen35.json` runs the same coding
and document fixtures used above. The GPU lease serializes cooperating evidence
jobs. A fresh report records its own checkout and source hashes; later results
do not retroactively qualify older artifacts. Sampling is a finite-output smoke
test, visual fixtures are functional checks, and a synthetic context fill checks
execution/reclamation rather than semantic long-context ability.
