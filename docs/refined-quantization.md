# Coordinate-refined INT8: a calibration-only checkpoint

This is an offline weight-method experiment for Qwen2 and text-only Gemma 3.
Signed W8 weights, FP32 per-output-row scales, FP16 reconstruction/activations,
version-3 files, inference kernels, tie handling, and FP16/app defaults are unchanged.
The prior [scale-fitting experiment](mlx-scale-aware-results.md) remains frozen
and failed. This new run does not extend or retune that experiment.

## Fixed recipe

Begin with the unchanged [scale-aware recipe](scale-aware-quantization.md) on
fresh FP16 source activations: covariance blocks of 64, damping 0.01, stable
activation ordering, row chunks 256, and the seven scale factors
`1.0, 0.995, 0.99, 0.98, 0.96, 0.92, 0.85`. After fitting scales and compensating
rounding errors, **keep those row scales fixed** and refine the integer values:

1. Visit each block's columns in stable descending covariance-diagonal order.
2. For each output row, consider its current integer value's decrement and
   increment, clipped to `[-127, 127]` using widened integers to prevent wrapping.
3. Evaluate the actual FP16-rounded reconstruction changes, not hypothetical
   exact multiples of a scale. With `e = W - FP16(q*s)` and reconstruction change
   `d` in column `j`, the quadratic-loss change is
   `-2*d*(e@H)[j] + d*d*H[j,j]`.
4. Apply only strict negative changes. Decrement wins exact neighbor-score ties;
   the original value wins zero-improvement ties. Update `e@H` incrementally.
5. At the end of every sweep, recompute the complete block loss from actual
   reconstructed values. Restore any row/block without a strict improvement.
6. Perform exactly **two sweeps**; no stopping threshold or extra rounds are
   introduced after observing official outputs. Finally recompute each complete
   row's objective and restore its original scale-aware coefficients unless the
   new row is strictly better. Scales remain byte-identical throughout.

The guaranteed bound is no worse per-row **block-diagonal calibration loss**
than scale fitting alone on the same moments. Incremental FP64 scores only rank
proposals; fresh complete objectives authorize acceptance. This does not bound
full-covariance error, exact greedy outputs, language quality, or speed. Greedy
coordinate descent can stop at a local optimum. It is not full
[GPTQ](https://arxiv.org/abs/2210.17323), a claim of novel research, or an inference
acceleration. Dense offline weights and covariance matrices still occupy memory;
temporary coordinate work is row-chunked, not constant-total-memory.
Here `H_b = X_b.T @ X_b / N` is an uncentered activation second moment, not a
mean-subtracted statistical covariance; cross-block terms are omitted.

## Identity, tests, and scope

Statistics identify `coordinate_refined_scale_aware_v1`; sealed policies identify
`coordinate_refined_cached_repair_v1`. The bounded no-pickle loader validates the
source checksum, covariance payloads, and exact configuration. Offline conversion
and derivation dispatch through this recorded recipe. Legacy RTN, fixed-scale,
and scale-aware archives keep their old derivation and defaults.

The shared calibration CLI accepts `--weight-method coordinate-refined`. Runtime
execution still consumes the existing packed descriptors; no new kernel or
weight representation is introduced. Tiny Qwen/Gemma tests cover the complete
calibration/export/derivation path and native-versus-composed cached logit bytes.
Portable tests independently compute source-activation projection error, exercise
singular/zero/saturated fixtures, verify unchanged scale bytes, and reject malformed
configurations, method mismatches, incomplete evidence and silent overwrites.

## Pre-fit registration and stop rule

`benchmarks/run_refined_checkpoint.py` registers the complete recipe before either
official fit. The precision search keeps four probe cases/candidate matrices,
two repair rounds/removal candidates/full trials, and a 25% eligible-byte floor.
Only strictly improved full-calibration scores authorize policy repairs.

Calibration uses 32 new prompts and 32 raw greedy tokens, all 1,024 cached positions
per family, rowwise projections, and no early EOS. Normalized/tokenized overlap
guards cover 237 prior/fresh corpus strings. The fresh 25-prompt regression corpus
is used only for overlap rejection; its outputs/logits are not consulted.
These guards detect string/token duplication, not semantic equivalence or
distributional independence; the corpus is not a broad model-quality benchmark.

The contract records complete input/tool/source hashes, previous complete evidence
bindings, and a recoverable archive of dirty runtime, workflow, corpus and test
sources. Git HEAD alone cannot recover the current uncommitted implementation.
No existing experiment/artifact is overwritten, and a partial run cannot be
silently restarted in place. Preserve it and choose a new directory instead.

Both official families must pass all free continuations and cached decisions
exactly, cosine at least 0.999, private cleanup, the projection floor, and the local
bound. Any failure saves both full calibration results and **stops before official
export, held-out inference, resource checks, or performance measurement**. Even
passing calibration is only eligibility for a separate downstream checkpoint;
`strict_checkpoint_passed` remains false here. No post-failure retuning is allowed.

## Reproduction and independent readback

Use existing official FP16 models and offline tokenizers in the project environment.
Run GPU workflows serially with Apple Metal access; the inherited exclusive lease
coordinates these tools, not unrelated desktop applications.

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run_refined_checkpoint.py \
  --output-dir benchmarks/results/refined-NEW-RUN \
  --artifact-dir models/refined-NEW-RUN
```

Or use `--phase prepare`, `--phase calibrate`, and `--phase audit` separately. Do
not edit runtime, tests, registered workflow or corpus files once prepared. A
complete rejected calibration/audit exits 1; inconsistent evidence raises an error.

After calibration exits, independently rederive every selected matrix and compare
it with scale fitting alone on the same moments using FP16 reconstruction and an
independent FP64 quadratic implementation:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/verify_refined_objectives.py \
  --root benchmarks/results/refined-NEW-RUN

PYTHONPATH=python FORGE_INSPECT_MODEL=build/forge-inspect-model-int8 \
  .venv/bin/python -m pytest -q \
  --junitxml=benchmarks/results/refined-NEW-RUN/test-results.xml

PYTHONPATH=python .venv/bin/python benchmarks/audit_refined_checkpoint.py \
  --root benchmarks/results/refined-NEW-RUN --expected-tests 255
```

The independent diagnostic uses a `1e-9` reduction-order verification tolerance,
not a weakened inference gate, and verifies byte-identical scales and coefficient
change counts. Its code and the final auditor are themselves registered before
fitting. Final readback requires the stage `verification.json`, complete selected
matrix/row coverage and a clean, unskipped JUnit suite. Result files are exclusive
writes; inspect saved runs instead of overwriting them.

This checkpoint adds no new held-out, 32K, direct-kernel, performance, hardware-counter,
CUDA, serving, prefix-cache, speculation, W4, cluster, or coding-harness certification.

See [the completed run and acceptance limits](mlx-refined-results.md). Qwen passes
this run's calibration, but Gemma's numerical gate fails; the combined stopped run
creates no official export or downstream evidence.
