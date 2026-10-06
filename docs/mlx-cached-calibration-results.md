# Cached-decode INT8 checkpoint — 2026-10-01

This checkpoint implements true cached teacher forcing, cached activation
sampling, bounded joint policy repair, all-position regression, and independent
evidence readback. Calibration does **not** pass the strict exact-output gate.
Both new policies remain experimental. FP16/app/default decode behavior is
unchanged; no release tag or W4 advancement is authorized by these results.

See [the method and reproduction contract](cached-calibration.md). The prior
[second-order experiment](mlx-second-order-results.md) remains unchanged.

## Freeze and calibration

Both official-family artifacts and policies were frozen before any held-out
inference at `2026-10-01T21:32:36.678559+00:00`. Runtime source SHA-256:
`057470340b804ce78266f34e44c6ba347ddb0be2fb1d84298df42b4e7bab9a27`.
HEAD is `a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`; the worktree is dirty.
The recorded environment is Apple M3 Max with 128 GiB unified memory,
Python 3.12.10, NumPy 2.5.3, and MLX 0.32.2. These are observed local versions.
The [freeze manifest](../benchmarks/results/cached-calibration-m3-max-2026-10-01/frozen.json)
binds 16 complete source/artifact/manifest/statistics/calibration files, four
corpus files, and eight workflow tools. Runtime sources are independently hashed.
The before/after integrity snapshots also include the freeze itself (17 files).

The new calibration has 32 prompts, 32 raw continuation tokens, no chat template
or early EOS, and all 1,024 cached positions. Probe cases are 0/10/20/31, all 32
positions each. Source replay must match scheduler generation before fitting.
Held-out text supplies only overlap guards; no held-out logits/tokens, failed
previous examples, or output-conditioned adjustments are used for fitting.

| Family | Seed cached top-1 changes / 1,024 | Final changes | Accepted full-corpus repairs | Exact free continuations / 32 | Minimum cached cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | 6 | 3 | 3 | 29 | 0.999681691 |
| Gemma 3 1B | 4 | 3 | 1 | 29 | 0.999467682 |

Qwen repairs improve the full-calibration mismatch count 6 → 5 → 4 → 3. Gemma
improves 4 → 3, then stops when checked swaps cannot strictly improve its full
score. These are within-calibration observations, not general quality gains.
The numerical gate passes but both exact-greedy calibration gates fail.
Raw trials, complete partitions, and failure cases:
[Qwen](../benchmarks/results/cached-calibration-m3-max-2026-10-01/calibration/qwen-calibration.json),
[Gemma](../benchmarks/results/cached-calibration-m3-max-2026-10-01/calibration/gemma-calibration.json).

## Frozen storage

| Family | FP16 file bytes | Candidate file bytes | Eligible projection bytes in INT8 | INT8 / retained FP16 matrices |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 988,084,480 | 896,975,360 | 25.5769% | 21 / 147 |
| Gemma | 1,999,794,688 | 1,824,053,760 | 25.2747% | 23 / 159 |

The fractions are source **projection** bytes, not whole-model storage saving.
Whole-file savings are only 9.2208% Qwen and 8.7879% Gemma, not 25% or the
earlier all-INT8 storage savings.
Activations, K/V, embedding/head, norms, biases, and retained projections remain
FP16. Same-partition RTN controls use the same file sizes and precision partition.
Statistics are offline-only; inference retains no permanent reconstructed model.
Existing C++ version-3 readers accept the artifacts; no new CMake/CUDA build is
claimed. Complete packed-value/scale rederivation is required in every regression.

Candidate data hashes:

- Qwen: `3534c26f05b966f5142177cb62bc580a984091df19e83dc99430f08302dc4395`.
- Gemma: `d1c24ccdb1729cf2f35bdb7a1f5e541a0dea9cbde8756c867648d3e90220f117`.

Policy hashes:

- Qwen: `cca72323bcd18a22ce317af446845d603b546e1f9380bae2f11db58383d58904`.
- Gemma: `8040a7975e5d7d3e870e0ba67dd05efa4c79a16057ca944dab70737252febf0a`.

Statistics SHA-256:

- Qwen: `beed0adc2973e610210aa6bb81dc02b32b490be1ed14d2e37f8dd992f45adf52`.
- Gemma: `6b493353e9fe60d6c4a928b8aa6ca044596532b48c48f958dfb244168beae6f0`.

## All-position cached regression

Each row below tests 25 prompts and 32 raw output tokens. All 800 logit rows
are evaluated through the actual paged prompt/decode path on the **same source
FP16 prefix**, not a growing whole-prompt re-prefill. Corpus 0 is the established
quality corpus; corpus 1 is the fresh cached-regression corpus. Both were fixed
before fitting. They are text/token-disjoint from calibration, not a claim of
semantic independence or of general language-model quality.

| Family | Execution | Corpus | Exact source continuations / 25 | Minimum cached cosine | Candidate/composed exact continuations / 25 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | Direct Metal | 0 | 19 | 0.999615483 | 22 |
| Qwen | Direct Metal | 1 | 22 | 0.999692142 | 23 |
| Qwen | Native reconstruction | 0 | 20 | 0.999681575 | 25 |
| Qwen | Native reconstruction | 1 | 20 | 0.999694075 | 25 |
| Qwen | Same-partition RTN/native | 0 | 19 | 0.999679027 | 25 |
| Qwen | Same-partition RTN/native | 1 | 18 | 0.999683203 | 25 |
| Gemma | Direct Metal | 0 | 19 | **0.998048252** | 25 |
| Gemma | Direct Metal | 1 | 20 | 0.999572692 | 22 |
| Gemma | Native reconstruction | 0 | 19 | **0.998220864** | 25 |
| Gemma | Native reconstruction | 1 | 21 | 0.999556920 | 25 |
| Gemma | Same-partition RTN/native | 0 | 20 | **0.998349318** | 25 |
| Gemma | Same-partition RTN/native | 1 | 20 | 0.999334302 | 25 |

All candidate exact-source-output gates fail. Gemma's original-corpus numerical
gate also fails the unchanged 0.999 threshold, including in native and RTN
execution. This stronger cached coverage reveals a failure that the previous
sparser teacher-forced experiment did not establish; the two protocols are not
interchangeable. Passing calibration cosine is not held-out certification.

Direct Metal disagrees with composed greedy outputs on five Qwen cases and
three Gemma cases across the two corpora. Native reconstruction preserves its
composed greedy outputs; it does **not** restore FP16 source accuracy. Neither
this new method nor the controls establish a consistent held-out quality win.
The raw reports retain every failed case, prefix length, per-position numerical
metric, derivation check, and gate. They are in
[the regression directory](../benchmarks/results/cached-calibration-m3-max-2026-10-01/regression).

`minimum_layer_cosine` in the legacy schema is an isolated per-projection
operator probe, **not** a full-layer hidden-state comparison. The source here is
the engine's internal FP16 contract, not a new Transformers certification.
All-position teacher forcing tests conditional logits; it is separate from
free-generation exactness. No failed output was used to amend these artifacts.

## Batching and resource correctness

All four family/mode contracts pass the unchanged same-artifact rowwise
consistency gates. Each arrival matrix includes concurrency 2/8/16 and a
staggered concurrency-8 run with mixed output budgets. Independent execution
and continuous batching agree on tokens, events, and all **2,705 byte-sensitive
logit fingerprints per contract**, or 10,820 in total. Cancellation occurs after
pages have materialized; every reservation and device page is reclaimed.
Historical single-request default outputs remain unchanged. Default batched
negative controls are retained: they match only 68/26/26/27 rows in the four
arrival workloads (800/800/800/305 total), despite some equal output sequences.
They are not included in the passed consistency claim.

The nine boundary lengths are 15/16/17/31/32/33/128/512/513 with 128-token
prefill chunks. All four contracts match 288/288 byte-sensitive logit rows,
or **1,152 total**, including paging, chunking, and Gemma's sliding window.
Both modes/families also complete the 32,766-token prompt plus two outputs:
64 chunked prefills, one decode iteration, exactly 2,048 physical/logical pages,
and full reclamation. Peak K/V storage is 402,653,184 bytes for Qwen and
872,415,232 bytes for Gemma. These are resource/execution gates, **not** long-
context semantic quality, sustained 32K decoding, or controlled latency results.

## Numerical diagnosis

Every selected INT8 matrix is probed at rows 1/2/8 with fixed random inputs:
63 Qwen and 69 Gemma projection cases. Reconstructed FP16 values match independent
NumPy dequantization, and native rowwise projections match the composed path
in every numeric comparison. Direct projection results differ in all 132 cases.
This isolates a projection-evaluation difference on identical inputs with
equal reconstructed values; it does not identify a particular compiler
instruction, establish a reduction-order cause, or explain every quality failure.
The inputs are synthetic operator probes, not cached task-quality measurements.

The supplemental `bitwise_kernel_logit_rows` quality field uses `array_equal`,
which treats signed zeros as equal. It is **not** literal byte proof. A separate
post-freeze diagnostic records dtype, shape, and raw-byte SHA-256 for all native/
composed cached rows on both corpora. This supplemental script does not alter
the frozen method, acceptance thresholds, artifacts, or direct-kernel contract.
All four byte-sensitive reports pass **800/800 rows each, 3,200 total**, including
dtype, shape, and signed-zero-sensitive bytes. They are in
[the numerical-byte directory](../benchmarks/results/cached-calibration-m3-max-2026-10-01/numerical-bytes).
This establishes the recorded native/composed numerical contract, not source
quality, universal backend equivalence, or direct-kernel exactness.

## Controlled performance and memory

The fresh matrix contains 32 results: two families, four execution modes,
prompt lengths 128/1,024, concurrency 1/8, and 32 outputs. Each result has five
**complete** warmups and three measured trials in an isolated fresh process;
implementation order rotates. All modes use rowwise decode and the same runtime
hash. Tests, captures, calibration, and diagnostic replays do not overlap timed
runs. The inherited lease serializes cooperating drivers, not unrelated apps.

There are 432 measured request observations. Prompts repeat token ID 42; this is
a synthetic execution workload, not the text quality corpus or interactive chat.
The inherited tokenizer label is unused, including the Qwen label in Gemma
metadata. TTFT includes admission queueing. TPOT is
`(end-to-end latency - TTFT) / (output tokens - 1)`.

Generated tokens/s over measured wall time:

| Family | Prompt / concurrency | FP16 | Composed INT8 | Direct INT8 | Native reconstruction |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 1 | 120.97 | 79.96 | 121.15 | 82.37 |
| Qwen | 128 / 8 | 232.10 | 201.75 | 222.90 | 177.69 |
| Qwen | 1,024 / 1 | 31.10 | 28.95 | 30.80 | 28.37 |
| Qwen | 1,024 / 8 | 64.08 | 63.49 | 63.18 | 57.35 |
| Gemma | 128 / 1 | 53.34 | 35.05 | 54.33 | 38.71 |
| Gemma | 128 / 8 | 132.53 | 114.27 | 128.19 | 113.97 |
| Gemma | 1,024 / 1 | 17.67 | 15.46 | 17.65 | 15.82 |
| Gemma | 1,024 / 8 | 47.68 | 43.81 | 46.76 | 44.35 |

Selected p95 latency comparisons, milliseconds:

| Family | Prompt / concurrency | FP16 TTFT | Direct TTFT | FP16 TPOT | Direct TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 1 | 22.42 | 25.69 | 7.90 | 7.80 |
| Qwen | 128 / 8 | 276.70 | 308.03 | 31.08 | 32.35 |
| Qwen | 1,024 / 1 | 137.32 | 144.92 | 28.86 | 29.04 |
| Qwen | 1,024 / 8 | 1,976.23 | 1,996.40 | 103.95 | 107.83 |
| Gemma | 128 / 1 | 49.59 | 55.90 | 17.83 | 17.27 |
| Gemma | 128 / 8 | 592.94 | 649.94 | 53.91 | 55.62 |
| Gemma | 1,024 / 1 | 284.44 | 295.24 | 49.32 | 49.09 |
| Gemma | 1,024 / 8 | 3,144.78 | 3,281.40 | 134.78 | 138.67 |

Three single-request observations and 24 correlated concurrency-8 observations
are sample quantiles, not robust population-tail estimates. All p50/p95/p99,
per-request timing, model/workload identities, allocator peaks, and K/V stats are
preserved in [Qwen's matrix](../benchmarks/results/cached-calibration-m3-max-2026-10-01/matrix/qwen)
and [Gemma's matrix](../benchmarks/results/cached-calibration-m3-max-2026-10-01/matrix/gemma).
Every configuration completes and leaves zero allocated/reserved/device K/V.

Logical resident weight storage is 942.29 → 855.40 MiB Qwen and
1,907.13 → 1,739.53 MiB Gemma. Measured peak **active MLX allocator** memory:

| Family | Prompt / concurrency | FP16 MiB | Composed MiB | Direct MiB | Native MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 128 / 8 | 1,146.99 | 1,397.09 | 1,125.18 | 1,125.25 |
| Qwen | 1,024 / 8 | 1,631.29 | 1,644.51 | 1,577.90 | 1,577.93 |
| Gemma | 128 / 8 | 2,251.30 | 2,235.68 | 2,144.55 | 2,144.55 |
| Gemma | 1,024 / 8 | 3,215.21 | 3,047.20 | 2,986.45 | 2,986.45 |

Allocator peaks are not process RSS, total system memory, or hardware-reserved
memory. Smaller stored weights do not guarantee lower peak memory: the composed
Qwen graph exceeds FP16's peak in both displayed workloads. One-pass reconstruction
avoids full FP32 dequantization intermediates by implementation, but these peaks
do not measure memory traffic or intermediate lifetimes.

Direct throughput is close to FP16 at concurrency 1 and lower in all four
concurrency-8 comparisons. Small positive single-request differences are not
statistically established speedups. Native reconstruction preserves reduction
consistency at an observed throughput cost. This fitting method changes weights
and retention, **not** kernels; the matrix compares execution modes on a frozen
partition, not the inference-speed benefit of cached calibration. Previous
experiments have different policies and cannot serve as controlled method A/Bs.
No new utilization, power, occupancy, bandwidth, roofline, warp-stall, or counter-
based attribution is available; unavailable measurements remain null, not zero.

## Tests and scope

The complete regression records **184 passing tests**, zero failures,
errors, or skips in [the JUnit report](../benchmarks/results/cached-calibration-m3-max-2026-10-01/test-results.xml).
Coverage includes deterministic finite search, full-corpus repair acceptance,
probe-only/equal-score rejection, corpus guards, checksum/partition binding,
projection aliases, private replay ownership/capacity/exception cleanup,
position/chunk/page correctness, source scheduler/replay equality, activation
capture restoration, vectorized/scalar metric agreement, and evidence tampering.
Metal tests exercise both tiny families through FP16/direct/native cached replay
and calibration → export → independent derivation → all-position validation.
The complete existing model/cache/scheduler/kernel tests also run unchanged.

Existing C++ host and UBSan binaries pass; both new candidate artifacts pass the
existing version-3 inspector. The old `host-debug` inspector predates version 3;
the verified test sweep uses `build/forge-inspect-model-int8`. CMake/Ninja are
unavailable locally for a fresh build. No fresh CMake build, ASan, Linux CI run,
RTX correctness/performance, or full Xcode counter inspection is claimed. The
portable CI and local verification lists include the new tests, not a claim
that remote CI has executed.

## Final readback and acceptance boundary

The [evidence index](../benchmarks/results/cached-calibration-m3-max-2026-10-01/evidence-index.json)
contains 76 JSON checksums and verifies the exact 32-result identities/shapes,
raw quality/resource gate consistency, and uniform runtime provenance. The
[final verification](../benchmarks/results/cached-calibration-m3-max-2026-10-01/verification.json)
binds that index, the 184-test JUnit report, the supplemental byte checker,
3,200 actual cached byte identities, and all 17 frozen whole-file bindings.
Before/after complete-file snapshots are identical, including metadata,
descriptors, policies, and statistics rather than just tensor-data headers.
The test suite runs before measurement with JUnit capture and passes again
after measurement. All old 74 indexed JSON checksums and 17 prior complete-file
bindings are independently rechecked unchanged; old statistics are included.

Raw fingerprints/tokens independently determine batch comparisons. Some event
and reference-cleanup observations remain flags from runtime instrumentation,
not independently regenerated from a full stored event timeline. None of this
readback supplies missing profiler counters or general model-quality evidence.
The two post-freeze diagnostic/audit scripts record their own hashes and never
change the eight pre-fit tool bindings or the fitting method.

`readback_passed` is true and `strict_checkpoint_passed` is **false**. Calibration
and validation save complete failed outcomes; their exit 1 is intentional, not
a crashed or incomplete experiment. No threshold, token/tie policy, precision,
default, or artifact is changed to conceal a failure. There is no release tag
or W4 advancement.

The next justified work is a new **calibration-only weight-method experiment**,
not additional swaps selected after these failed outputs. It needs a fresh
untouched regression corpus and the same all-position cached contract. Prefix
caching, speculative decoding, serving/distributed routing, the coding harness,
and on-device RTX validation remain future work. This checkpoint makes the
accuracy investigation reproducible and more faithful to actual decode;
it does not yet make INT8 a quality-approved replacement for FP16.
