# Scale-aware INT8 fitting: a gated calibration checkpoint

This checkpoint changes **offline weight fitting**, not the execution kernels,
packing format, precision, greedy tie policy, defaults, or app. The saved INT8
runtime still consumes signed W8 weights with one FP32 scale per output row,
FP16 activations/reconstruction, and the existing version-3 descriptors.
Qwen2 and text-only Gemma 3 use the same recipe.

The previous block-compensated quantizer fixes its scales to each source row's
maximum absolute weight divided by 127. This method asks whether **activation-
weighted scale selection** improves the actual FP16-rounded reconstruction
objective on fresh source calibration activations. It does not assume that a
lower local objective means better greedy tokens. See the prior
[cached-decode failures](mlx-cached-calibration-results.md).

## Registered method

Source prompt chunks and every cached decode position contribute to the same
block-diagonal covariance collector and paged replay contract described in
[cached calibration](cached-calibration.md). Calibration has 32 new prompts,
32 raw greedy tokens, no early EOS, all 1,024 positions, and rowwise projections.
Normalized and tokenized guards check 180 prior/fresh corpus strings. Neither
held-out output tokens nor logits are evaluated or used for fitting.

For each FP16 projection matrix:

1. Compute the unchanged fixed-scale block-compensated baseline on these **same**
   moments: blocks of 64, damping 0.01, stable activation ordering, and RTN fallback.
2. Compute a per-output-row weighted RTN loss using actual FP16-rounded values.
   Test seven fixed scale factors in order:
   `1.0, 0.995, 0.99, 0.98, 0.96, 0.92, 0.85`.
3. Accept a candidate scale only for strictly lower whole-row weighted RTN loss.
   Zero rows retain scale 1. Equal scores retain the first scale exactly.
4. Run block error compensation once with the selected row scales. Its RTN
   fallback is relative to the **new** scales, not mixed with incompatible old ones.
5. Recompute the complete row's objective. Keep its candidate q/scale pair only
   if it strictly improves on the original compensated baseline. Otherwise retain
   that row's complete original q/scale pair **byte-for-byte**.

The objective is `sum_blocks error_block @ H_block @ error_block.T` for each
output row, evaluated in FP64 on the FP16-rounded reconstruction. Cross-block
correlations are discarded. Temporary objective arrays use at most 256 output
rows per block; whole weight/covariance matrices and compensated working blocks
are still present offline, so this is not a constant-memory algorithm.
No full reconstructed model is retained during runtime inference.

The guaranteed bound is **no worse per-row block-diagonal reconstruction loss
than the old recipe on the same collected moments**, not the full covariance
loss, task accuracy, latency, or model quality. This is a bounded approximation,
not full [GPTQ](https://arxiv.org/abs/2210.17323), AWQ, or a novel research claim.
FP32 scale storage and FP16 rounding are part of the tested objective rather
than replaced by hypothetical exact reconstruction.

The existing joint selector and full-corpus repair are reused with a fixed
smaller evaluation budget: four probe cases, four candidate matrices, two repair
rounds, two removal candidates, and two full trials per round. The 25% eligible
projection-byte floor remains unchanged. Only strictly better full-calibration
scores authorize a repair. These bounds are fixed before running either family;
they are not increased in response to a failed output.

## Data, identity, and backward compatibility

Statistics identify `scale_aware_block_second_order_v1`; policies identify
`scale_aware_cached_repair_v1`. Source/data checksum, quantizer configuration,
statistics checksum, corpus/token IDs, search bounds, and complete precision
partition are sealed. The strict no-pickle NPZ loader supports both old and new
methods, rejects mismatched configurations/methods, and remains bounded before
loading array payloads. Conversion/rederivation dispatch through the recorded
method, not a label that silently substitutes ordinary RTN.

Legacy RTN and fixed-scale block recipes keep their defaults, values, format,
and metadata labels. Explicit scale overrides are offline-only and must be
finite positive FP32 vectors no larger than the source max-based scales. Old
archives/policies still rederive through the old method. Tiny exports for both
families verify deterministic artifact bytes, retained FP16 values, packed
values/scales, and native/composed cached logit bytes.

The new driver writes a **pre-fit** contract with complete source/manifest,
corpus, and tool hashes plus all prior cached-checkpoint result/artifact bindings.
It also captures the actual dirty runtime and workflow sources in `sources.zip`;
a Git commit/hash alone cannot recover uncommitted source. This archive excludes
large model files and the environment, which remain external dependencies with
recorded identities. The archive and each member are checked on readback.

Both family runs see the same registered method. Input or runtime drift aborts
the run. No artifact/report is overwritten. Interrupted statistics or completed
calibration cannot be silently rerun in place; preserve the partial run and use
a fresh directory for a new experiment.

## Stage gate and honest stopping rule

The checkpoint ends at **calibration**, not a complete inference-performance
release. Both families must pass:

- All 32 free continuations exactly equal the source FP16 continuations.
- All 1,024 cached teacher-forced decisions agree, with cosine at least 0.999.
- The 25% eligible-byte floor, private cache cleanup, and local reconstruction bound.

Gate readback recomputes decisions, cosine, inventory bytes, and summaries from
recorded cases and actual source tensors. Flags alone cannot certify acceptance.
If either family fails, the driver preserves both complete calibration results
and **stops before official export, held-out inference, or timing**. It exits 1
for a complete failed experiment; inconsistent/missing evidence raises an error.
There is no post-failure retuning of this run.

Even passing calibration only makes the policy **eligible** for a separate
held-out/export/resource/performance checkpoint. `strict_checkpoint_passed`
remains false here because those downstream stages have not run. The fresh
25-prompt regression corpus stays output-blind on a stopped calibration run.
Neither an improved local objective nor passing tiny tests waives the gate.

## Reproduction

Use the existing official FP16 artifacts and offline tokenizers in the project
environment, with actual Apple Metal access and `PYTHONPATH=python`:

```bash
.venv/bin/python benchmarks/run_scale_checkpoint.py \
  --output-dir benchmarks/results/scale-NEW-RUN \
  --artifact-dir models/scale-NEW-RUN
```

Alternatively run `--phase prepare`, `--phase calibrate`, and `--phase audit`
separately on the same directories. Finish each process before another GPU job.
The inherited exclusive lease protects cooperating workflows, not unrelated apps.
Once prepared, do not edit runtime or registered workflow/corpus files. Test/docs
updates outside the registered sources do not authorize method changes.

Standalone experimental conversion remains available through `forge-quantize`
with its policy and statistics arguments, but this driver never publishes an
official packed candidate from a rejected calibration. Such a manual conversion
does not certify quality or bypass the declared downstream gate.

Linux CI, ASan, full Xcode counters, and RTX execution remain separately
unverified. This checkpoint adds no serving, prefix cache, speculation, W4,
multi-GPU, harness, or new inference-speed claim.

## Completed run and post-registration readback

The [2026-10-01 report](mlx-scale-aware-results.md) records failed exact-token
calibration for both official families, successful local reconstruction checks,
and the honored stopping rule. A stage audit exits 1 for this valid rejection;
that is different from an evidence-integrity exception.

After the calibration process has fully exited, these host-only diagnostics
check saved evidence without fitting, GPU inference, or official export:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/verify_scale_objectives.py \
  --root benchmarks/results/scale-NEW-RUN

# Run the complete regression suite with Apple Metal access, then bind its report.
PYTHONPATH=python FORGE_INSPECT_MODEL=build/forge-inspect-model-int8 \
  .venv/bin/python -m pytest -q \
  --junitxml=benchmarks/results/scale-NEW-RUN/test-results.xml

PYTHONPATH=python .venv/bin/python benchmarks/audit_scale_checkpoint.py \
  --root benchmarks/results/scale-NEW-RUN --expected-tests 222
```

The first command independently reconstructs every selected row and evaluates
its FP64 quadratic against the old fixed-scale recipe. The final command requires
the saved stage `verification.json` from `--phase audit`, repeats its raw readback,
checks diagnostic matrix/row coverage and source bindings, and hashes the clean
JUnit report. These diagnostics were introduced after registration and record
their own source hashes; they are not fitting choices. Diagnostic/final JSON
files are exclusive writes. For the saved run, inspect them instead of rerunning
in place. Adjust the expected test count only to the actual complete suite in a
new checkout/run, never to hide skipped or missing tests.
