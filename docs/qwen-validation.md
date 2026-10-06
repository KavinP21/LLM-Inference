# Frozen Qwen INT8 downstream contract

This checkpoint is separate from the stopped two-family coordinate-refined
calibration experiment. It exports only Qwen's already sealed, calibration-eligible
policy. Gemma remains FP16. There is no fitting, expanded search, new precision
partition, threshold change, or inference-kernel/default change here.

## Acceptance and stop rules

Registration happens before official export or held-out generation. It binds the
whole source/statistics/calibration/policy files, all earlier complete evidence,
the fresh 25-prompt refined regression corpus, live validation scripts and a ZIP
of the actual dirty runtime and tests. The source FP16 model is the reference;
this is not a new Transformers accuracy certification.

Two quality reports cover 32 raw greedy outputs and every cached teacher-forced
position, without EOS truncation, for 800 rows each:

- `reconstruct`: native GEMM after FP16-rounded INT8 reconstruction, with rowwise
  decode projections. This is the frozen policy's acceptance contract.
- `metal`: direct small-batch Metal INT8 reductions. This is diagnostic only;
  it cannot inherit acceptance from native GEMM or qualify the default backend.

The strict native gate retains cosine 0.999, exact full continuations, exact
native-versus-composed continuations, complete projection comparisons, exact
packed-byte derivation and cleanup. A separate native/composed comparison hashes
dtype, shape and the signed-zero-sensitive bytes of all 800 cached logit rows.
All accepted flags are recomputed from raw observations, not trusted summaries.

Complete negative quality reports are valid evidence and exit nonzero. Native
quality or byte failure stops **before** batching, 32K or timing. Partial stages
cannot be overwritten or silently rerun. Existing exports are experimental, even
when calibration passed; neither the desktop app nor FP16 defaults are changed.

If quality passes, resource checks cover independent/continuous batching at
concurrency 2/8/16, staggered arrivals, cancellation, page/chunk boundaries and a
32,766-token prompt plus two outputs. Those are same-artifact/resource checks,
not long-context semantic quality. Resource failure stops before timing.

Only after both stages pass, measure FP16, native reconstruction and composed
dequantization in isolated processes: prompt lengths 128/1,024, concurrency 1/8,
32 outputs, five warmups and three trials. Rotate mode order, keep raw request
observations and preserve cache/memory/environment provenance. Direct Metal
timing is deliberately excluded from this native-contract matrix. A completed
matrix does not promise a speedup; storage savings do not imply resident-memory
or latency savings. Hardware-counter profiling remains a separate requirement
for causal performance claims.

## Reproduce

Use fresh directories and the already preserved refined calibration/statistics:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run_qwen_validation_checkpoint.py \
  --output-dir benchmarks/results/qwen-validation-NEW-RUN \
  --artifact-dir models/qwen-validation-NEW-RUN
```

The driver supports `prepare`, `export`, `quality`, `resources`, `benchmark` and
`audit` phases. Later phases refuse failed prerequisite gates. All cooperating
GPU evidence drivers share a fail-fast inherited lease; execute them serially.

After completion, run the full suite with actual Metal access, then perform the
host-only, complete-file final readback:

```bash
PYTHONPATH=python FORGE_INSPECT_MODEL=build/forge-inspect-model-int8 \
  .venv/bin/python -m pytest -q \
  --junitxml=benchmarks/results/qwen-validation-NEW-RUN/test-results.xml
PYTHONPATH=python .venv/bin/python benchmarks/run_qwen_validation_checkpoint.py \
  --output-dir benchmarks/results/qwen-validation-NEW-RUN \
  --phase final-audit --expected-tests 287
```

Use the actual current test count, not a stale number. The final auditor verifies
complete, clean JUnit and an immutable file index. Exit 1 with a complete saved
rejection is expected for a failed acceptance gate; missing evidence, exceptions,
inconsistent flags or source drift are execution errors, never accepted rejection.

## Next after a rejection

Preserve this result, treat its corpus as observed, and leave INT8 experimental.
Another quantization method must use a new calibration-only registration and a
new output-blind regression corpus. Do not spend unlimited checkpoints on the
same exact-token objective: prefix caching/copy-on-write can proceed on validated
FP16 without waiving quantization gates. W4A16 remains gated by per-model quality.
