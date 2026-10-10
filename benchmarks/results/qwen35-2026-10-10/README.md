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

.venv/bin/python benchmarks/qualify_qwen35_multimodal_speculative.py \
  --image /tmp/q35-core-fixtures/text.png --video /tmp/q35-core-fixtures/sequence.mp4 \
  --output /tmp/q35-visual-spec.json
```

`evaluate_agents.py --config examples/agents-qwen35.json` runs the same coding
and document fixtures used above. The GPU lease serializes cooperating evidence
jobs. A fresh report records its own checkout and source hashes; later results
do not retroactively qualify older artifacts. Sampling is a finite-output smoke
test, visual fixtures are functional checks, and a synthetic context fill checks
execution/reclamation rather than semantic long-context ability.

## Final executed results

The official-model runs below used clean commit `35c262b` and runtime SHA
`04c78c2c896c5e8dac4ec360ea46ac224bb2d66e43f0581703d167a995ab6bb1`.
The runtime remained unchanged throughout the runs. Core/spec drivers at that
commit captured start provenance; the operator checked the same clean source
after execution. Subsequent drivers also enforce an after-run source check.

| Report | Result and scope |
| --- | --- |
| [Core](core-final.json) | 12 checks pass: 17 direct-reference tokens, visual fixtures, thinking, cleanup, 32,766+2 synthetic context execution and seeded sampling smoke. Peak persistent state at 32K is 1,125,253,120 bytes; weights/transients excluded. |
| [Speculation](speculative-final.json) | 12 checks pass: three rounds at prefixes 0–4, live 24-layer transaction cancellation/restart, two uneven allocation boundaries, sequential fallback and warmed parity/timings. Copy/code/arithmetic median ratios 1.920×/1.006×/0.919×; default off. |
| [Multimodal speculation](multimodal-speculative-final.json) | Image/video token comparisons pass with real n-gram verification and 44/27 accepted tokens; no visual speed or semantic-quality claim. The exact [operator procedure](operator-sources/visual_speculative.py) matches the recorded SHA; use the generalized public driver for a new run. |
| [Workers](workers-final.json) | Five checks pass: image HTTP, expanded-context rejection, actual completed visual child, reopened journal (one deduplicated blob) and cancellation/restart. |
| [Desktop](ui-final.json) | Nine checks pass: default model, OCR/follow-up, incomplete thinking, Stop, media reclamation, output clamp, preserved draft and native 0.5B switch; owner/model/temp-file shutdown also passes. |
| [Tasks](agents-final.json) | All four cases pass; 25.93/56.08 seconds coding, 8.35/51.44 documents; two completed children per delegated case. Delegation slower; small development fixtures. |
| [Final regression](regression-final-summary.json), [JUnit](regression-final-tests.xml) | 1,082 tests, zero skips/failures/errors; Metal, loopback, source-mutation guards and fresh C++ host validation. The earlier 1,080-test checkpoint report is retained separately. |

Later source changes to evidence drivers add provenance checks only; the qualified
model/worker runtime hash remains the same. No original-FP16, general task quality,
262K semantic qualification, NVIDIA or multi-host claim is made.
