# Calibration-only second-order INT8 checkpoint

This is an opt-in offline experiment. It does not change the default FP16 model,
desktop app, decode policy, runtime tensor format, or CUDA implementation. It is
not a completed quantization quality gate merely because export or calibration
succeeds. See [the measured results](mlx-second-order-results.md).

## Hypothesis and frozen experiment

The preceding single-projection ablation ignores interactions, and round-to-nearest
(RTN) minimizes individual weight error without considering activation correlations.
The hypothesis is that a block-diagonal, activation-weighted reconstruction objective,
followed by joint precision selection, can improve greedy parity at the same minimum
25% eligible-projection-byte fraction. The fraction is not 25% of the whole model.

The method, fixed parameters, 32 calibration prompts, original 25 regression prompts,
and fresh 25 regression prompts are defined before either regression is evaluated.
Neither regression corpus supplies logits, tokens, ranking data, or fitted scales.
Their text/token IDs are used during calibration only to reject exact overlap.
The fresh corpus is unseen output data at the freeze, not a semantic independence
claim; some tasks are intentionally familiar categories with different questions.

The two model-family policies and covariance archives are frozen **before any**
regression outputs are requested. No held-out failure is used to choose a new
precision partition or tune damping/block size within this experiment. The old
mixed-precision artifacts and reports remain intact.

## Weight algorithm

Second-order rounding-error compensation is inspired by the
[GPTQ paper](https://arxiv.org/abs/2210.17323). Forge implements a deliberately
different, bounded approximation rather than claiming full GPTQ or AWQ:

1. Collect inputs from the original FP16 model. Each layer has four shared input
   groups: Q/K/V, attention output, gate/up, and down projection. Sample up to eight
   evenly spaced rows per calibration prompt, including its FP16-generated prefix.
2. Accumulate FP64 `XᵀX / n` for contiguous 64-feature blocks. Zero-pad the final
   block; discard cross-block correlations. No dense full-width Hessian is stored.
3. Keep the existing per-output-channel FP32 scales (`max(abs(row)) / 127`, zero
   rows use 1). No clipping search, activation quantization, or scale optimization.
4. Order columns by descending covariance diagonal within each block, stably.
   Add `0.01 * mean(diagonal)` damping, with an absolute floor of `1e-12`.
5. Compute an upper factor of the inverse damped block covariance. Quantize each
   column and compensate its error in the remaining columns of that block.
   Reconstruction is **FP32 multiply rounded to FP16**, matching the runtime.
6. For each row/block, retain RTN values if compensation increases the undamped
   block calibration objective. This is still INT8, not additional FP16 retention.

The fallback guarantees no greater block-diagonal calibration reconstruction loss
than RTN. It guarantees neither full activation reconstruction error (which includes
cross-block terms) nor logit quality, greedy outputs, or downstream task accuracy.
FP16 source activations are not recomputed under progressively quantized layers.

## Interaction-aware mixed precision

The block reconstruction loss per source byte ranks a bounded candidate shortlist.
Starting from the all-FP16 source, each iteration evaluates up to 12 new projections
**together with every already selected INT8 projection** on eight deterministically
spaced calibration teacher-forced prefill-logit probes. Selection minimizes combined
top-1 changes first, then mean squared relative logit error per quantized source byte.
Names break exact score ties deterministically. Stop once at least 25% of eligible
FP16 projection bytes are selected. Whole-matrix granularity may exceed that floor.

This is a bounded greedy search, not an exhaustive optimum. The shortlist and eight
probes can miss interactions elsewhere. Teacher-forced prefill logits are a proxy;
all 32 calibration prompts are separately tested for cached 32-token greedy parity
and numerical similarity. A failed calibration policy remains explicitly failed;
it may still be exported for diagnostic regression without being certified.

## Reproducibility and safety

- A source-bound, checksummed NPZ archive stores block covariance matrices and
  configuration. Q/K/V and gate/up share statistics. NumPy pickle loading is disabled;
  ZIP member sizes and NPY shape/dtype/payload sizes are checked before allocation.
  Matrices must be finite, symmetric, positive semidefinite, correctly padded, and
  cover every supported projection with compatible input width.
- A sealed precision policy binds the FP16 source checksum, statistics checksum,
  algorithm/configuration, corpora identities, and complete FP16/INT8 partition.
- Export refuses to overwrite sources, artifacts, manifests, or statistics. Version-3
  tensors remain signed INT8 with positive finite FP32 scales. Tied embeddings/head,
  norms, biases, and retained FP16 projections are preserved.
- Quality validation independently recomputes **every packed value and scale** from
  the source and frozen statistics and checks every retained source tensor. Supplying
  only a compatible architecture or untrusted manifest label is insufficient.
- Statistics and offline dense copies are never loaded by inference. The inference
  graph is identical to the earlier W8A16 graph for an equivalent partition.
- The evidence driver owns the inherited exclusive GPU lease. No model test or
  other GPU benchmark should overlap its performance matrix.
- Runtime-source, driver, corpus, artifact, statistics, and policy drift after the
  freeze fail closed. Saved gate failures return nonzero; execution errors cannot
  be treated as expected quality failures.

## Runbook

Requires the two existing official FP16 artifacts, cached tokenizers, MLX, NumPy,
and the export/reference optional dependencies. Use new output paths on each run:

```bash
.venv/bin/python benchmarks/run_second_order_checkpoint.py \
  --output-dir benchmarks/results/second-order-NEW-RUN \
  --artifact-dir models/second-order-NEW-RUN
```

The driver calibrates both models, exports calibrated and **same-partition RTN**
controls, writes `frozen.json`, then runs both 25-prompt regressions, direct Metal
and reconstructed modes, continuous batching, boundaries, 32K resource checks,
and a serialized 32-result performance matrix. Each measured configuration uses
five full warmups and three trials. Both modes explicitly use `rowwise` decode;
the normal `batched` default is untouched.

For inspectable stage boundaries use `--phase calibrate`, then `--phase validate`,
`--phase benchmark`, and `--phase audit` with the same evidence directory. Validation
can exit 1 with complete saved failure evidence; read the checkpoint before proceeding.
Do not change source code or the driver between stages. Audit requires all evidence,
including benchmarks; omitted work cannot be marked complete.

Individual export/validation commands:

```bash
.venv/bin/python -m forge_llm.quantization SOURCE.engine CANDIDATE.engine \
  --policy CALIBRATION.json --calibration-stats STATS.npz
.venv/bin/python -m forge_llm.validate_int8 \
  --source SOURCE.engine --model CANDIDATE.engine --tokenizer TOKENIZER \
  --calibration-stats STATS.npz --decode-mode rowwise --int8-mode metal \
  --prompts benchmarks/second-order-regression-prompts.json --output QUALITY.json
```

Strict acceptance requires full calibration and both regression-corpus greedy
parity, the unchanged numerical threshold (`0.999`), kernel-versus-composed parity,
batch consistency, and cache/resource gates. RTN control results are comparative
observations, not acceptance requirements. Performance numbers require controlled
raw trials; absent Xcode counters cannot support occupancy/bandwidth/power claims.
Do not start W4A16 based on a failed strict checkpoint.
