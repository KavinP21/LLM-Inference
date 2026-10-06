# INT8 accuracy hardening — improved numerical gates, greedy gate still failed

Measured on 2026-10-01 on Apple M3 Max, macOS 26.5, Python 3.12, MLX 0.32.2.
Default FP16 artifacts and the desktop app remain unchanged. This delivers calibration,
mixed-precision export/execution, and auditable regression tooling, **not a passed release gate**.
Neither model is authorized to advance to W4A16 on this evidence. CUDA remains unmeasured.
The final [evidence index](../benchmarks/results/int8-hardening-m3-max-2026-10-01/evidence-index.json)
records acceptance status and SHA-256 values for 57 calibration/quality/resource/batch/matrix files.

## What changed

- A separate, frozen 16-prompt calibration corpus, Unicode/case/whitespace and tokenized overlap
  guards, full-model single-projection perturbation measurements, deterministic sensitivity ranking,
  combined-policy calibration trials, and source-bound/checksummed policies.
- Explicit FP16 retention in the existing version-3 artifact. The converter validates the entire
  projection partition and source identity, rejects a calibrated below-25% projection-byte policy,
  and records every retained projection. Runtime loading requires no sidecar or calibration data.
- An explicit `int8_mode="reconstruct"`: one-pass Metal unpacking plus native MLX GEMM at **all**
  row counts, avoiding the direct reduction kernel's different accumulation order. It retains no
  permanent dense copy, but incurs a temporary FP16 matrix and extra launches on decode.
- Engine precision statistics, immutable regression output paths, a four-mode matrix, and a
  reproducible evidence runner that saves failures and exits nonzero rather than calling them passes.
- An exclusive, inherited process lease for cooperating evidence drivers. It rejects simultaneous
  drivers and releases ownership on process exit. This is measurement hygiene, not distributed
  serving or a claim that unrelated applications are prevented from using the GPU.

This heuristic is not [AWQ](https://arxiv.org/abs/2306.00978) activation-aware channel scaling or
[GPTQ](https://arxiv.org/abs/2210.17323) approximate second-order quantization. It changes only which
whole matrices keep their original FP16 values; remaining INT8 matrices use the unchanged deterministic
max/127 scale and nearest-even rounding.

## Calibration-only selection

Eight full-model probes cover deterministic positions across the calibration corpus. Each ablation
changes exactly one projection to its FP16-rounded INT8 reconstruction. Top-1 changes take priority;
otherwise ranking uses mean squared relative logit error per eligible FP16 byte. All selected
perturbations are then tested **together**, generating 32 tokens per prompt and comparing logits at
positions 0/7/31. Single-matrix rankings do not capture interactions, and calibration results are
not held-out or general task-quality certification.

| Family | Requested INT8 projection fraction | Actual fraction | Exact calibration cases / 16 | Minimum logit cosine | Strict calibration |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen | 100% | 100% | 8 | 0.996875 | Failed |
| Qwen | 75% | 75.10% | 10 | 0.999358 | Failed |
| Qwen | 50% | 50.16% | 12 | 0.999754 | Failed |
| Qwen | 25% | 25.58% | 12 | 0.999908 | Failed |
| Gemma | 100% | 100% | 8 | 0.998401 | Failed |
| Gemma | 75% | 75.02% | 10 | 0.998981 | Failed |
| Gemma | 50% | 50.21% | 11 | 0.999488 | Failed |
| Gemma | 25% | 25.11% | 14 | 0.999723 | Failed |

No candidate passes every calibration gate. The final floor candidates are frozen as **failed
experimental policies**, not tuned further after looking at held-out results. Raw rankings, probe
metrics, prompt/token observations, policy partitions, and hashes:
[Qwen calibration](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen-calibration.json),
[Gemma calibration](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma-calibration.json).
All trials release reserved and physical K/V blocks.

The existing 25 prompts are held out **from precision selection**. They were already used in the
previous all-INT8 regression; this is not a previously unseen blind test. Direct-overlap guards do
not establish semantic independence. Calibration code fingerprints are saved separately from the
final runtime fingerprint; the subsequent tiny-model integration test also fixed calibration to
honor native context limits below 2K, without changing either official model's 2K calibration limit.

Calibration corpus SHA-256: `a9a7f58c13e7ef28cdcc158795da645e041ef36c4aebe21cc8d29aae953cf0e8`.
Held-out corpus SHA-256: `3419f1890f6b0a06420328d81e6c29b896c811000321c88621c287dfa63afa6b`.

## Frozen artifact storage

| Family | Original FP16 bytes | Mixed artifact bytes | File reduction | INT8 / total projections | Retained FP16 projections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | 988,084,480 | 896,959,488 | 9.22% | 21 / 168 | 147 |
| Gemma 3 1B | 1,999,794,688 | 1,825,228,544 | 8.73% | 22 / 182 | 160 |

The ~25% figures refer to **eligible projection bytes**, not matrix counts or whole-model savings.
Embedding/head, norms, biases, activations, and K/V remain FP16, and tied weights are still shared.
Mixed data checksums:

- Qwen: `11a0c1e2a23ef9e9d572447be81a6a1f9fbcd9fcd24ac0c6f24b6c14c902fb5b`.
- Gemma: `c0304dfe852ea62f614c26abc2a0f121baef1231d2a48ee418e34b21532ac386`.

Compared with the earlier fully quantized projections, retaining FP16 buys accuracy at a large
storage cost. Do not attach the earlier 35–36% file savings to these policies.

## Validation and remaining gate

The held-out runner verifies that every INT8 matrix/scale derives exactly from the supplied FP16
source and every retained tensor is unchanged. It checks all remaining INT8 projections, 25 prompts
with 32 raw greedy tokens, teacher-forced positions 0/1/7/15/31, and kernel-versus-composed behavior.
The same `cosine >= 0.999`, exact-greedy, and cache-cleanup gates remain in force.

Reconstruction-mode Qwen minimum logit cosine is **0.999849**, with exact 32-token sequences on
**19/25**, versus 15/25 for the earlier all-INT8 direct-kernel candidate. Gemma minimum is
**0.999643**, with **22/25**, versus 17/25 previously. Both reconstruction comparisons match the
composed baseline on all 25 greedy sequences. These are internal Forge FP16 regressions, not new
Transformers certification, perplexity, or general reasoning/coding quality measurements.

| Family | Mode / raw result | Minimum projection cosine | Minimum logit cosine | Teacher-forced top-1 | Exact 32-token cases / 25 | Candidate/composed greedy cases / 25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | [Direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/quality-metal.json) | 0.999930 | 0.999849 | 99.2% | 19 | 25 |
| Qwen | [Reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/quality-reconstruct.json) | 0.999930 | 0.999849 | 99.2% | 19 | 25 |
| Gemma | [Direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/quality-metal.json) | 0.999958 | 0.999643 | 99.2% | 22 | 25 |
| Gemma | [Reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/quality-reconstruct.json) | 0.999958 | 0.999643 | 99.2% | 22 | 25 |

All numerical, candidate/composed, and reclamation sub-gates pass for these frozen artifacts.
Exact greedy output remains failed; improved cosine is not a substitute. This subset's direct
kernel agreement does not fix or certify the previous all-INT8 Qwen kernel's two near-tie cases.
FP16 remains the ordinary/default path. Do not label this checkpoint complete or lower its
threshold because lossy quantization makes near ties difficult.

## Dynamic-arrival batching: a second failed Gemma gate

An identity/code-fingerprint checked replay inserts one request every two iterations, caps active
sequences at eight, mixes output budgets 1/4/8/16/32, and cancels a separately admitted request after
materializing its pages. It compares each output to that **same artifact/mode's independent**
saved output, not to FP16 across different weights. All events/counts and cache reclamation checks
pass. Qwen matches 25/25 for both [reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/batch-reconstruct.json)
and [direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/batch-metal.json).
Gemma matches 24/25 for both [reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/batch-reconstruct.json)
and [direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/batch-metal.json); its
mutex explanation diverges at generated position 3. This acceptance gate remains **failed**.

The unmodified [Gemma FP16 baseline](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/batch-fp16-baseline.json)
also matches only 23/25: mutex at position 3 and the bug-finding prompt at position 17.
Independent replays reproduce the original outputs. The
[FP16 first-divergence diagnostic](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/batch-fp16-diagnostic.json)
records cosines 0.999999623/0.999999558, top-two margins 0/0.015625, and maximum logit
differences 0.0625/0.046875. The
[mixed diagnostic](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/batch-mixed-diagnostic.json)
records cosine 0.999999695, margin 0.015625, and maximum difference 0.03125 at its one flip.
Tracing copies logits to host and is **not performance evidence**.

These are small batch-dependent numerical changes at tied/near-tied decisions, including an FP16
baseline issue rather than solely weight quantization. The exact responsible operator has not been
isolated; clean cache accounting alone is not a correctness proof. No tie-breaking hacks,
prompt-specific exceptions, or silent fallback/default changes were added. Layer-by-layer FP16
batch-shape diagnostics now take priority before declaring multi-request Gemma parity complete.

## 32K execution/resource gates

Each frozen model executes a 32,766-token synthetic prompt plus two output tokens, 64 prefill
chunks and one custom paged-attention decode iteration. Both modes peak at exactly 2,048 physical
pages and reclaim all reserved/materialized pages. This is execution/resource evidence, not 32K
semantic quality or a throughput study.

| Family | Raw mode | Peak K/V bytes | Peak active allocator bytes | Reclaimed |
| --- | --- | ---: | ---: | --- |
| Qwen | [Reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/32k-reconstruct.json) | 402,653,184 | 1,822,918,028 | Yes |
| Qwen | [Direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/32k-metal.json) | 402,653,184 | 1,822,918,028 | Yes |
| Gemma | [Reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/32k-reconstruct.json) | 872,415,232 | 3,806,245,074 | Yes |
| Gemma | [Direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/32k-metal.json) | 872,415,232 | 3,807,899,858 | Yes |

## Regression suite and boundaries

Final suite: **98 tests pass**, including source-bound mixed exports, malformed/overlapping policies,
floor enforcement, ownership restoration on failure, tiny calibration-to-export-to-runtime checks
for both families, byte-exact reconstruction, paged lengths 15/16/17/31/32/33, batched output parity,
cancellation, and process-lease inheritance/contention/release. Portable policy/lease checks are
included in Linux/macOS host CI. CI itself was not observed running here.

C++ host/UBSan tests and both mixed-artifact C++ inspections pass. This milestone does not change
the C++ loader, quantization scheme, CUDA kernels, FP16 default, or app behavior. Previous local
ASan startup and unavailable full-Xcode counter analysis limitations remain unchanged. No GPU
utilization, bandwidth, occupancy, power, or GPU-only kernel-time claim is inferred from wall time.

## Reproduce the frozen-policy evidence

After calibration/conversion in the quantization runbook, use a fresh output directory:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run_int8_hardening.py \
  --source models/qwen2.5-0.5b.engine --model models/qwen2.5-0.5b-int8-mixed.engine \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct --output-dir benchmark-results/qwen-hardened
```

Replace all three model/tokenizer arguments for Gemma. The driver serializes reconstruct/direct
held-out checks, both 32K resource gates, and the FP16/composed/direct/reconstruction matrix. It
continues after an explicitly saved failed quality gate to collect tradeoff evidence, but aborts
on execution errors and finally exits nonzero if the gate failed. Existing output paths are never
overwritten. The process lease covers cooperating drivers, not direct ad hoc CLI/test invocations;
keep other project GPU work idle during measurements.


## Final controlled performance matrix

Every row uses an isolated process, five full 32-token warmups and three measured trials, token-ID
42 prompts, no EOS stopping, and identical scheduler/cache/other-kernel settings. Percentiles cover
three requests at concurrency 1 or 24 at concurrency 8, not reliable production tail estimates.
The four-mode order rotates by configuration. Memory is MLX peak active allocator memory, not
RSS or total system memory; GPU utilization and power remain unavailable/null.

### Qwen2.5 0.5B

| Prompt | Concurrency | Runtime / raw result | Output tokens/s | TTFT p50 ms | TPOT p50 ms | Peak active GiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/fp16-p128-c1.json) | 97.6 | 24.5 | 10.29 | 1.105 |
| 128 | 1 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-dequantize-p128-c1.json) | 60.1 | 37.2 | 15.97 | 1.303 |
| 128 | 1 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-metal-p128-c1.json) | 102.3 | 27.9 | 8.61 | 1.076 |
| 128 | 1 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-reconstruct-p128-c1.json) | 78.0 | 25.9 | 11.52 | 1.076 |
| 128 | 8 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/fp16-p128-c8.json) | 355.5 | 139.3 | 17.66 | 1.120 |
| 128 | 8 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-dequantize-p128-c8.json) | 259.5 | 238.3 | 21.74 | 1.303 |
| 128 | 8 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-metal-p128-c8.json) | 335.4 | 153.2 | 18.33 | 1.091 |
| 128 | 8 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-reconstruct-p128-c8.json) | 323.2 | 166.1 | 18.69 | 1.091 |
| 1024 | 1 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/fp16-p1024-c1.json) | 30.1 | 142.9 | 29.83 | 1.506 |
| 1024 | 1 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-dequantize-p1024-c1.json) | 28.1 | 143.6 | 31.73 | 1.470 |
| 1024 | 1 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-metal-p1024-c1.json) | 30.2 | 150.7 | 29.08 | 1.437 |
| 1024 | 1 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-reconstruct-p1024-c1.json) | 28.9 | 143.5 | 30.99 | 1.437 |
| 1024 | 8 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/fp16-p1024-c8.json) | 53.2 | 1250.2 | 98.99 | 1.593 |
| 1024 | 8 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-dequantize-p1024-c8.json) | 64.0 | 951.3 | 79.91 | 1.557 |
| 1024 | 8 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-metal-p1024-c8.json) | 70.7 | 900.7 | 73.52 | 1.525 |
| 1024 | 8 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/qwen/matrix/int8-reconstruct-p1024-c8.json) | 81.2 | 927.4 | 63.98 | 1.525 |

### Gemma 3 1B

| Prompt | Concurrency | Runtime / raw result | Output tokens/s | TTFT p50 ms | TPOT p50 ms | Peak active GiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/fp16-p128-c1.json) | 52.7 | 47.5 | 18.19 | 2.167 |
| 128 | 1 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-dequantize-p128-c1.json) | 36.6 | 55.3 | 26.12 | 2.286 |
| 128 | 1 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-metal-p128-c1.json) | 54.4 | 52.8 | 17.24 | 2.125 |
| 128 | 1 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-reconstruct-p128-c1.json) | 39.1 | 53.3 | 24.71 | 2.125 |
| 128 | 8 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/fp16-p128-c8.json) | 194.5 | 287.9 | 30.54 | 2.199 |
| 128 | 8 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-dequantize-p128-c8.json) | 153.0 | 349.6 | 39.22 | 2.318 |
| 128 | 8 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-metal-p128-c8.json) | 190.9 | 313.5 | 30.56 | 2.157 |
| 128 | 8 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-reconstruct-p128-c8.json) | 157.4 | 342.5 | 38.05 | 2.157 |
| 1024 | 1 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/fp16-p1024-c1.json) | 17.6 | 290.8 | 49.12 | 2.953 |
| 1024 | 1 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-dequantize-p1024-c1.json) | 15.0 | 314.3 | 59.09 | 2.879 |
| 1024 | 1 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-metal-p1024-c1.json) | 17.3 | 304.3 | 50.18 | 2.760 |
| 1024 | 1 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-reconstruct-p1024-c1.json) | 15.8 | 300.7 | 55.57 | 2.760 |
| 1024 | 8 | [FP16](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/fp16-p1024-c8.json) | 52.0 | 1688.6 | 92.41 | 3.140 |
| 1024 | 8 | [Mixed/composed](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-dequantize-p1024-c8.json) | 47.1 | 1826.8 | 102.66 | 3.066 |
| 1024 | 8 | [Mixed/direct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-metal-p1024-c8.json) | 50.8 | 1766.1 | 93.43 | 2.947 |
| 1024 | 8 | [Mixed/reconstruct](../benchmarks/results/int8-hardening-m3-max-2026-10-01/gemma/matrix/int8-reconstruct-p1024-c8.json) | 47.8 | 1796.6 | 100.59 | 2.947 |

The frozen policies improve fidelity but give up much of the earlier all-INT8 memory saving.
Custom/reconstruction modes reduce peaks versus FP16 by roughly 2.6–4.5% for Qwen and 1.9–6.5%
for Gemma in this matrix. Composed reconstruction can still exceed FP16's peak at short contexts.
The accuracy-oriented reconstruction mode is slower for Gemma in every measured row; it should
not be marketed as a speed optimization. The direct mode is workload-dependent, not a universal
win. Qwen's 1K/concurrency-8 row has a large apparent relative difference and a slower FP16
baseline than the previous session; small trials and host/allocator variation prevent attributing
that difference to a particular kernel without replication/profiler evidence. Do not pool these
latencies with the previous all-INT8 matrix or promote them into a maximum-performance claim.

Final quality, 32K, and both published matrices share runtime-source SHA-256
`1a9a5537e76ee22844a519e0afad8500dbc5793e686c56e0bda66d1c89f5dd3a`, Git HEAD
`a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`, and explicit dirty-source status. All 32 matrix
results have the correct warmup/trial/token counts and zero remaining reserved/materialized K/V
blocks. Capture/synthetic quality work was not run alongside the published matrices.

An overlapping Qwen matrix/repeat was detected and stopped. Its timings remain recoverably
preserved in `qwen/diagnostic-matrix` and `qwen/diagnostic-matrix-repeat`, with
`checkpoint-diagnostic.json`; none is used in the tables above. Qwen's final `matrix/` is a fresh,
fully serialized rerun. The exclusive inherited driver lease was added and tested to prevent the
same cooperating-driver overlap. Failed sandbox-only MLX test discovery is not a failed kernel
test or a valid GPU measurement.
