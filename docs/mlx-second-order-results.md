# MLX second-order INT8 checkpoint — 2026-10-01

This is an experimental implementation checkpoint, not a performance release.
The calibrated weight method and interaction-aware policy selector are implemented,
but the strict greedy-quality gate remains failed. FP16 artifacts, the desktop app,
and the `batched` default are unchanged. W4A16 is not approved by this evidence.

The [method and runbook](second-order-quantization.md) describe the approximation,
frozen parameters, safety checks, and exact acceptance contract. It is not full GPTQ.

## Provenance and experimental scope

Apple M3 Max with 128 GiB unified memory, Python 3.12.10, NumPy 2.5.3, and MLX 0.32.2.
These are locally observed versions, not inferred product specifications. RTX/WSL
execution, AddressSanitizer, Linux CI, and hardware-counter profiling are not certified.
The existing C++ host and UBSan binaries pass; no fresh CMake build is claimed.

All official-model results use runtime source SHA-256
`21fa0be0da47cb8b14bef2bd7f799ce3a6ad6e3381a5026fc46f4d016d1bc898`.
Git HEAD is `a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`; the checkout is dirty. Source hashes, not HEAD alone,
identify this implementation. Policies, artifacts, statistics, corpora, and driver
were frozen at `2026-10-01T18:44:21.266496+00:00`, before regression output evaluation.
See [the freeze manifest](../benchmarks/results/second-order-m3-max-2026-10-01/frozen.json).

The policies are fitted to [32 separate calibration prompts](../benchmarks/second-order-calibration-prompts.json).
Evaluation uses the [original 25 prompts](../benchmarks/quality-prompts.json) and a
[fresh 25-prompt regression](../benchmarks/second-order-regression-prompts.json), 32 raw
greedy tokens each, no chat template and no early EOS. Fresh means unseen **outputs**
at the freeze; familiar task categories are not proof of semantic independence.
No regression output was used to adjust the quantizer or select a new partition.

Quality compares with the identical Forge FP16 source, not a new Transformers or
downstream task-quality certification. All modes explicitly use opt-in rowwise decode.
Numerical teacher-forced probes use prefill logits at positions 0, 1, 7, 15, and 31;
cached free generation is a separate check and can disagree despite high cosine.

## Calibration and artifact size

| Family | INT8 / FP16 projections | Eligible bytes quantized | File size | Source-file saving | Calibration greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | 22 / 146 | 25.8013% | 896,176,128 B | 9.3017% | 27/32 |
| Gemma 3 1B | 22 / 160 | 25.1057% | 1,825,228,544 B | 8.7292% | 26/32 |

The minimum calibration logit cosines are 0.9997953 and 0.9998605. Both **fail**
the exact calibration gate. For the selected projections, the summed block-diagonal
calibration objective is 0.55822 / 0.52253 of the same-partition RTN objective: about
44.2% / 47.7% lower. This is a local reconstruction result, not an end-to-end quality
or speed claim. The hypothetical full covariance objective is not measured.

The archives contain 92,836,404 / 136,355,380 bytes of FP64 statistics. They and the
offline reconstructed calibration copies are never loaded during inference. The
two artifacts remain version 3; embeddings/head, activations, K/V, norms, and biases
stay FP16. Each export also has an equal-size **same-partition RTN control**, which
holds retention and scales fixed while changing rounding/error compensation.

Authoritative calibration evidence:

- [Qwen calibration and sealed policy](../benchmarks/results/second-order-m3-max-2026-10-01/calibration/qwen-calibration.json)
- [Gemma calibration and sealed policy](../benchmarks/results/second-order-m3-max-2026-10-01/calibration/gemma-calibration.json)

Candidate data SHA-256:

| Family | Calibrated | Same-partition RTN |
| --- | --- | --- |
| Qwen | `ca9a362d8c9289865f999bcc962135c310796a1142d94c881d835352c5f46714` | `6d09803933345599c3add6114ceded381a7e42f50e76d1e361c764ea9425a4bb` |
| Gemma | `ad7397c3750be4c55867428e5dd1f0f8cb0e2238bbf8f9ca35e2e8e643e2f313` | `05c6c2e2c2ae721119fc15079e333018a6d9f8244d1bc236b1786adebc5d78bd` |

The source identities, policy hashes, and archive hashes are in the freeze manifest.
Official source projection matrices are not mutually aliased. Arbitrarily shared
projection layouts are outside this calibrated-export checkpoint; always run the
independent packed-value derivation check before using an experimental export.

## Separate-corpus quality and same-partition controls

Every packed value and scale is independently rederived from the FP16 source and
frozen statistics. Retained FP16 source tensors are checked for exact equality.
The two calibrated artifacts also pass the existing C++ version-3 reader/contract.
All random four-row projection numerical checks and sampled logit numerical gates
pass `0.999`; that is not full-layer-output or free-generation token parity.

| Family / method | Original 25 greedy | Fresh 25 greedy | Original minimum logit cosine | Fresh minimum logit cosine |
| --- | ---: | ---: | ---: | ---: |
| Qwen calibrated / direct Metal | 21/25 | 22/25 | 0.9997690 | 0.9998918 |
| Qwen calibrated / reconstruction | 19/25 | 22/25 | 0.9997870 | 0.9998918 |
| Qwen same-partition RTN / reconstruction | 20/25 | 24/25 | 0.9997698 | 0.9998624 |
| Gemma calibrated / direct Metal | 22/25 | 19/25 | 0.9993046 | 0.9998450 |
| Gemma calibrated / reconstruction | 23/25 | 20/25 | 0.9993046 | 0.9998450 |
| Gemma same-partition RTN / reconstruction | 21/25 | 21/25 | 0.9995999 | 0.9998700 |

All twelve reports are in [the regression directory](../benchmarks/results/second-order-m3-max-2026-10-01/regression).
The calibrated reconstruction is worse than same-partition RTN on both Qwen
corpora, better on the original Gemma corpus, and worse on the fresh Gemma corpus.
Thus the local-objective hypothesis does **not** establish general greedy-quality
improvement. Do not advertise this as an accuracy win or choose the best mode per
failed example. Historical mixed policies have different partitions and calibration
data; they are not a controlled A/B for the rounding algorithm.

Direct Metal versus composed reconstruction matches 23/25 cases on each Qwen
corpus and 24/25 on each Gemma corpus. All four native-reconstruction reports match
their composed baseline 25/25. Kernel numerical similarity still passes. This is a
separate failed **cross-kernel greedy** gate, not a cache or batch-consistency pass.

## Dynamic batching, boundaries, and 32K

For each family and candidate mode, all 25 original prompts replay against fresh
same-artifact independent references at saturated concurrency 2, 8, and 16, and a
staggered concurrency-8 trace with output budgets 1/4/8/16/32. Rowwise decode has
2,705/2,705 bitwise-equal logit rows per contract, **10,820 total**. Token decisions,
event counts, historical independent outputs, materialized cancellation, and cache
reclamation pass all four contracts. Unchanged batched-mode negative controls retain
their documented shape dependence; no default numerical-policy switch is implied.

All four candidate-mode contracts also pass lengths 15, 16, 17, 31, 32, 33, 128,
512, and 513 with 128-token prefill chunks and 32 generated tokens. There are
288/288 exact logit rows per contract, **1,152 total**, including Gemma's 512-token
sliding boundary. This is same-artifact consistency, not FP16-source quality.

Every candidate/mode executes a 32,766-token prompt plus two outputs, 64 prefill
iterations and one custom paged-decode iteration. Peak logical and physical pages
are exactly 2,048; all reservations/pages/device K/V bytes are released. Peak K/V
is 402,653,184 B Qwen and 872,415,232 B Gemma. K/V remains FP16. These four runs
certify execution/resources, **not** semantic quality or statistically rigorous
32K latency. Their one-shot timings are not compared as performance claims.

See [the aggregate failed checkpoint](../benchmarks/results/second-order-m3-max-2026-10-01/regression/checkpoint.json)
for separate quality, batching, resource, and calibration flags.

## Cached cross-kernel diagnostic replay

Manual common-prefix paged-cache replay reproduces **all six** saved direct-kernel
versus composed decisions. Metal, reconstruction, and composed modes use the same
artifact and teacher-forced prefix. Final reconstructed/composed argmax agrees in
every replay; direct Metal differs exactly where the free-generation report differs.

- Gemma: original corpus case 24 at generated position 17, fresh case 11 at position 24.
- Qwen: original cases 0/23 at positions 31/3, fresh cases 1/14 at positions 17/9.

The first different captured operation is `mlp.gate_proj` in layer 3 for Gemma and
layer 0 for Qwen, with **identical projection inputs**. Its maximum output difference
ranges from 0.0000305176 to 0.000244141. Cached final-logit cosine is at least
0.999998339, while native top-two margins are zero or 0.015625. This isolates a
cross-implementation floating-point linear-evaluation difference and near-tie
sensitivity; it does not establish a particular compiler instruction/reduction
order as the sole mechanism, nor excuse the failed exact-greedy contract.

See the [Qwen original](../benchmarks/results/second-order-m3-max-2026-10-01/diagnostics/qwen-corpus0-kernel.json),
[Qwen fresh](../benchmarks/results/second-order-m3-max-2026-10-01/diagnostics/qwen-corpus1-kernel.json),
[Gemma original](../benchmarks/results/second-order-m3-max-2026-10-01/diagnostics/gemma-corpus0-kernel.json),
and [Gemma fresh](../benchmarks/results/second-order-m3-max-2026-10-01/diagnostics/gemma-corpus1-kernel.json)
stage reports. Retained intermediates and host copies are diagnostic overhead,
never timing evidence or post-regression fitting data.

## Controlled performance and memory

The 32-result matrix has two families, four execution contracts (FP16, composed
dequantization, direct Metal, one-pass reconstruction), prompt lengths 128/1,024,
concurrency 1/8, and 32 generated tokens. Every configuration has five **full**
warmups and three measured trials in a fresh process. Implementation order rotates.
All use rowwise decode. No other agent GPU tests/profiles overlap measurement;
the inherited lease serializes cooperating workflows, not unrelated desktop apps.

Workloads are repeated synthetic token ID 42, not the textual quality corpora or
interactive chats. The inherited tokenizer label is unused, including Qwen's label
in Gemma benchmark metadata. Do not reinterpret these results as chat-task speed.
TPOT is `(end-to-end - TTFT)/(output tokens - 1)`. TTFT includes admission queueing.

Generated tokens/s over measured trial wall time:

| Family | Prompt / concurrency | FP16 | Composed INT8 | Direct INT8 | Reconstruction |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 1 | 96.62 | 62.35 | 99.77 | 74.07 |
| Qwen | 128 / 8 | 197.76 | 165.32 | 185.27 | 186.45 |
| Qwen | 1,024 / 1 | 30.57 | 27.74 | 30.48 | 27.40 |
| Qwen | 1,024 / 8 | 59.64 | 55.50 | 60.92 | 56.75 |
| Gemma | 128 / 1 | 52.07 | 36.31 | 53.11 | 38.66 |
| Gemma | 128 / 8 | 130.03 | 113.83 | 90.17 | 115.38 |
| Gemma | 1,024 / 1 | 16.95 | 14.78 | 17.12 | 15.32 |
| Gemma | 1,024 / 8 | 42.85 | 26.72 | 32.46 | 42.33 |

Selected p95 latency comparisons in milliseconds:

| Family | Prompt / concurrency | FP16 TTFT | Direct TTFT | FP16 TPOT | Direct TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 1 | 26.69 | 31.09 | 11.21 | 10.22 |
| Qwen | 128 / 8 | 313.99 | 360.38 | 37.10 | 39.94 |
| Qwen | 1,024 / 1 | 138.68 | 149.69 | 30.15 | 29.83 |
| Qwen | 1,024 / 8 | 2,184.83 | 2,111.11 | 111.04 | 107.95 |
| Gemma | 128 / 1 | 50.90 | 54.44 | 18.35 | 17.81 |
| Gemma | 128 / 8 | 611.88 | 977.56 | 54.69 | 80.56 |
| Gemma | 1,024 / 1 | 294.70 | 310.47 | 51.54 | 51.00 |
| Gemma | 1,024 / 8 | 3,642.99 | 5,133.28 | 153.65 | 207.62 |

These are sample quantiles, not robust population-tail estimates: three requests
at concurrency 1 and 24 correlated requests at concurrency 8. Full p50/p95/p99,
raw per-request timings, workload seeds, memory/cache stats, and provenance are in
the [Qwen matrix](../benchmarks/results/second-order-m3-max-2026-10-01/matrix/qwen)
and [Gemma matrix](../benchmarks/results/second-order-m3-max-2026-10-01/matrix/gemma).
All requests finish and all cache accounting returns to zero.

Logical resident weights are 942.29 → 854.64 MiB Qwen and 1,907.13 → 1,740.65 MiB
Gemma. This partial policy saves only about 9.3% / 8.7% of model weight storage,
not the earlier all-INT8 35%+ savings. Measured peak **active MLX allocator bytes**:

| Family | Prompt / concurrency | FP16 MiB | Composed MiB | Direct MiB | Reconstruction MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 8 | 1,147.10 | 1,400.97 | 1,100.27 | 1,100.23 |
| Qwen | 1,024 / 8 | 1,631.38 | 1,544.04 | 1,543.95 | 1,544.00 |
| Gemma | 128 / 8 | 2,251.30 | 2,373.48 | 2,208.67 | 2,208.67 |
| Gemma | 1,024 / 8 | 3,215.21 | 3,139.44 | 3,017.94 | 3,017.94 |

Allocator peaks are not process RSS, hardware-reserved memory, or total system
memory. The composed graph can create full FP32 reconstruction intermediates;
one-pass reconstruction removes that graph operation, but these peaks alone do
not quantify HBM traffic or intermediate lifetimes.

Direct INT8 is close to FP16 in several single-request configurations and slower
in concurrent Gemma configurations. Small positive differences are not advertised
as statistically established speedups. Reconstruction trades reduction consistency
for overhead in some shapes. The new weight-fitting algorithm changes coefficient
values, not execution kernels: this is an execution-mode A/B using the frozen new
partition, **not** evidence that second-order fitting improves inference speed.
No new occupancy, bandwidth, warp-stall, utilization, power, roofline, or hardware
counter attribution is available. Full Xcode counter tooling remains unavailable.

Complete-file snapshots of 17 source/artifact/manifest/statistics/policy files are
identical [before](../benchmarks/results/second-order-m3-max-2026-10-01/integrity-before.json)
and [after](../benchmarks/results/second-order-m3-max-2026-10-01/integrity-after.json)
measurement, binding architecture/descriptors as well as tensor data. The
[evidence index](../benchmarks/results/second-order-m3-max-2026-10-01/evidence-index.json)
checks completeness, source-hash uniformity, workloads, cache reclamation, and raw
file checksums. Its `strict_checkpoint_passed` is correctly **false**.

## Acceptance boundary

The final [JUnit report](../benchmarks/results/second-order-m3-max-2026-10-01/test-results.xml)
records **141 tests passed**, zero failures/errors/skips. Portable coverage includes
an independent correlated-input objective check, dead/diagonal covariance, malformed
configuration/archive/covariance, source/checksum/partition binding, no-clobber export,
deterministic recipe rederivation, legacy RTN preservation, capture restoration, corpus
guards, evidence failure/exit semantics, and whole-file metadata drift. Metal coverage
includes tiny Qwen/Gemma calibration → export → derivation → inference alongside the
unchanged family/kernel/cache/scheduler suite. CI/runbook lists include the new tests;
CI execution itself is not claimed.

The [final verification record](../benchmarks/results/second-order-m3-max-2026-10-01/verification.json)
binds that JUnit file and the evidence index, verifies all 74 indexed JSON hashes,
both statistics archives and 17 whole-file bindings, and recomputes quality flags
from all twelve raw summaries. Unit-test success and same-artifact numerical
consistency are explicitly separate from the **failed** lossy-quantization contract.

Local reconstruction improvement is insufficient to mark this milestone's strict
quality contract passed. The current experiment remains frozen even when a gate
fails. Further method changes require a new calibration-only experiment and an
untouched regression corpus, not repeated fitting to these failed examples.

The next justified work is calibration-only **cached-decode** probe coverage and
bounded policy repair/selection, plus cross-kernel numerical hardening. Keep the
current policies/results immutable and do not advance to W4A16 or publish a broad
quality/speed claim on this failed experiment. Prefix caching, speculation, serving,
distributed routing, and RTX validation remain future work.
