# Safe stopping point: FP16 prefix caching

Stopped on 2026-10-02 at the user's request, within the ten-minute deadline.
No model/measurement process remains running. No automation or scheduled resume
was created. Source/model/corpus/gate identities remain frozen.

## Current checkpoint

`benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed`

- Prefix-cache/COW implementation is complete; all eight official Qwen/Gemma
  correctness, concurrency and 32K resource gates pass.
- Fresh raw readback passes; 160 prior complete files and 104 live/captured source
  members match. The interrupted first run is preserved separately.
- Final saved regression: 365 tests, zero failures/errors/skips. Host and UBSan
  existing binaries pass. Lint and whitespace checks pass.
- **Performance is not measured yet.** No `matrix` directory or final milestone
  verification exists. The parent's exit 130 was the requested stop after it saved
  validation, not a new model failure. Keep this distinction from the first run's
  recorded unsupported-dispatch exception.
- FP16/app defaults unchanged; prefix cache opt-in and FP16-only. INT8 remains
  experimental, and no CUDA qualification is inferred.

## Resume exactly here

Use the existing root; do not run `--phase all`/`prepare`/`validate` against it.
Those phases deliberately refuse overwrite. Do not alter the frozen runtime,
checkpoint tools, tests, method document, models or workload before measurement.
If sources must change, preserve this root and register a new complete run.

```bash
cd /Users/kavinprabhakar/customLLMInferenceEngine

# One serialized GPU workflow; 24 reports, 5 warmups / 3 measured trials each.
PYTHONPATH=python .venv/bin/python benchmarks/run_prefix_checkpoint.py \
  --output-dir benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed \
  --phase benchmark

# Recompute metrics/provenance/work/cleanup from all raw reports.
PYTHONPATH=python .venv/bin/python benchmarks/run_prefix_checkpoint.py \
  --output-dir benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed \
  --phase audit

# Run only after the measurement driver has actually exited/released its lease.
# This regenerates a test report, not an immutable model or experiment artifact.
PYTHONPATH=python FORGE_INSPECT_MODEL=build/forge-inspect-model-int8 \
  .venv/bin/python -m pytest -q \
  --junitxml=benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/test-results.xml

PYTHONPATH=python .venv/bin/python benchmarks/run_prefix_checkpoint.py \
  --output-dir benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed \
  --phase final-audit --expected-tests 365
```

The handoff index binds today's saved test report. Preserve its original bytes
before regenerating JUnit if retaining that exact snapshot is needed; the final
index should bind the final test report. Never overwrite benchmark JSON or any
registration/source archive. If measurement stops partway through, retain partial
results and explicitly reconcile existing versus absent matrix cells before
continuing; `--phase benchmark` intentionally requires a fresh matrix directory.

Then update the [results report](mlx-prefix-cache-results.md), STATUS/README/roadmap
with cold versus warm costs and measured memory/timing, and close the milestone
only if independent final verification passes. Until then the count remains 12,
including this pending final qualification and the still-open INT8/W4 gates.
The next independent feature after closure is FP16 speculative decoding.
