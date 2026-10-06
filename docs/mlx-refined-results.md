# MLX coordinate-refined INT8 checkpoint

Date: 2026-10-02. This checkpoint changes offline weight fitting, not inference
kernels, packed format, greedy tie handling, or FP16/app defaults. Both official
calibrations are complete: Qwen passes, while Gemma fails the numerical gate.
The combined checkpoint is rejected at calibration. No official packed candidate,
held-out inference, resource matrix, or timing measurement follows this rejection.

## Implementation and registration

The [method contract](refined-quantization.md) describes two bounded integer-
coordinate sweeps after the unchanged scale-aware recipe, with byte-identical
row scales. Actual FP16-rounded neighbor values determine score changes; freshly
recomputed block/row losses authorize changes, with complete baseline fallback.
The guarantee is calibration-local block-diagonal reconstruction loss, not model
quality, exact tokens, or latency.

The [pre-fit registration](../benchmarks/results/refined-m3-max-2026-10-02/contract.json)
was saved at `2026-10-02T08:11:48.108659+00:00` before either official fit:

- Blocks 64, damping 0.01, stable activation ordering, row chunks 256.
- The existing seven scale factors, then exactly two integer-refinement sweeps.
- Four probe cases/candidate matrices, two repair rounds/removal candidates/full trials.
- A 25% eligible projection-byte floor and an unchanged cosine threshold of 0.999.
- 32 fresh calibration prompts, 32 raw greedy tokens, all 1,024 cached positions
  per family, rowwise projections, and no early EOS.
- String/token overlap guards against 237 corpus strings. The new 25-prompt
  regression corpus is output-blind; guards are not semantic independence checks.

The source artifacts remain the official FP16 Qwen2.5 0.5B Instruct and Gemma 3
1B IT exports. References are this runtime's existing source FP16 engine, not a
new Transformers quality certification. Source data hashes are unchanged:

- Qwen: `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45`.
- Gemma: `92d3081b2facfa8a5eb48dcedd89cb8e230b21a00cfeefd4c32f77ca2a9482ee`.

Local environment: Apple M3 Max, MLX 0.32.2, Python 3.12.10 and NumPy 2.5.3.
Git HEAD is `a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`, with a dirty checkout;
registered runtime SHA-256 is
`5387bb7931410a40862e9ca6d6d9d635af37f0d5b3fd8d63e28808f74b09561b`.
Registration binds 21 complete input/tool files, captures 110 source members
(including 36 test files), and preserves 115 complete prior file bindings.
The source archive, not Git HEAD alone, recovers the dirty implementation.

## Validation scope

All 255 tests pass before and after fitting, including 31 new portable checks and two new
Metal cases. New coverage includes correlated and singular covariance, zero and
saturated weights, unchanged scales, deterministic q/scale and artifact bytes,
method-bound archives/conversion, no-clobber behavior, corpus/registration drift,
complete failed-gate handling, independent quadratics, and final-evidence tampering.
Tiny Qwen/Gemma calibration/export/derivation paths also match native/composed
cached logit bytes. These are integration checks, not official-model accuracy.

Existing C++ host and UBSan binaries pass again. No fresh CMake/Ninja build,
ASan success, Linux CI execution, RTX result, or hardware-counter evidence is
claimed. The older scale checkpoint was fully read back before runtime changes;
its own artifacts, results, source archive, and acceptance failure are preserved.

Both official families must pass exact free continuations, exact cached decisions,
cosine, cleanup, projection floor and local bounds. Failure stops before official
export, held-out inference, resource gates or timing. Calibration success alone
would only authorize a separate downstream checkpoint; strict INT8 certification
is not implied. There is no post-result search expansion or gate relaxation.

## Official-model calibration

| Observation | Qwen2.5 0.5B | Gemma 3 1B |
| --- | ---: | ---: |
| Exact free continuations | 32/32 | 32/32 |
| Changed cached decisions | 0/1,024 | 0/1,024 |
| Minimum cached logit cosine | 0.999688892 | 0.998753977 |
| INT8 / retained FP16 projections | 21 / 147 | 23 / 159 |
| Eligible projection-byte fraction | 25.5769% | 25.2747% |
| Distinct policy evaluations / bound | 85 / 693 | 100 / 749 |
| Attempted / accepted repair rounds | 0 / 0 | 1 / 0 |
| Free / cached exact-token gates | Pass | Pass |
| Cosine / complete calibration gate | Pass | **Fail** |

Qwen's initial selected policy passes full calibration, so no repair runs. Its
[raw report](../benchmarks/results/refined-m3-max-2026-10-02/calibration/qwen.json)
contains every decision and policy trial. Qwen is eligible only for a separately
registered downstream checkpoint, not certified held-out inference. Passing on
these new prompts is not a controlled accuracy improvement over older policies
tested on different corpora; only the local same-moments reconstruction comparison
below isolates this method's effect.

Gemma matches 32/32 free continuations and all 1,024 cached decisions, but its
minimum cosine is `0.9987539766641692`, below the unchanged `0.999` threshold.
Its 23 INT8 / 159 retained FP16 projections cover 25.2747% of eligible bytes.
The search uses 100 distinct evaluations within a bound of 749; one repair round
tries two full alternatives, neither improves the complete score, and no repair
is accepted. The second round is not forced after a rejected round.

Exactly one cached position fails cosine: case 24, generated position 31, for
the map-scale/compass-bearing prompt. Relative L2 error is about 5.013%; the
source top-two margin is 0.890625, while the candidate's is 0.015625. The selected
argmax is the same, but the distributions are not sufficiently similar under
the registered numerical gate. Matching tokens therefore does not certify this
policy. This is observed calibration evidence, not an inferred hardware cause.

The [raw report](../benchmarks/results/refined-m3-max-2026-10-02/calibration/gemma.json)
preserves every case and trial. No threshold, weights, or search budget were
changed after inspecting it; no held-out or timing work follows the rejection.

## Independent reconstruction readback

The registered host-only diagnostic rederives all selected q/scales and the
unchanged scale-aware baseline on the **same moments and precision partition**.
It uses independent FP16 reconstruction and an FP64 `einsum` quadratic rather
than trusting the fitter's objective helper or summary flags.

| Selected projections only | Qwen | Gemma |
| --- | ---: | ---: |
| Matrices / output rows verified | 21 / 102,144 | 23 / 153,216 |
| Scale-aware baseline objective | 2.979372463 | 2.713570034 |
| Coordinate-refined objective | 2.891522497 | 2.625237830 |
| Relative local-loss reduction | 2.9486% | 3.2552% |
| Changed integer coefficients | 2,816,074 | 5,192,931 |
| Changed row scales | 0 | 0 |
| Worsened rows | 0 | 0 |

All 44 matrices and 255,360 rows pass. No positive deltas occur, even within the
independent readback's `1e-9` reduction-order tolerance. Fitting acceptance itself
uses strict comparisons; that diagnostic tolerance does not relax cosine or
token gates. Losses are calibration-local and not comparable between models.
These percentages are **not speedups, general accuracy improvements, or storage
measurements**. No language-quality A/B against the baseline on this corpus was
registered or added after failure.

The [independent report](../benchmarks/results/refined-m3-max-2026-10-02/objective-verification.json)
contains matrix shapes, per-matrix objectives/max row deltas, unchanged-scale and
coefficient-change counts, q/scale hashes, statistics/report bindings and the
diagnostic source hash. It exports no official packed model.

## Preserved stopping point

The [calibration checkpoint](../benchmarks/results/refined-m3-max-2026-10-02/calibration-checkpoint.json)
is `rejected_at_calibration`, with official export, held-out inference, performance
and strict-certification flags all false. Offline fitting wall times in that file
are not inference latency or controlled cross-method performance measurements.
Qwen's individual calibration acceptance does not override the registered
both-family stopping rule.

Keep this experiment frozen. Qwen's sealed policy is eligible for a **separate,
pre-registered Qwen-only downstream checkpoint** with untouched held-out quality,
export derivation, batching/resource gates and controlled performance. That work
has not run here and cannot certify Gemma or silently reuse this result directory.
Gemma stays experimental/FP16 by default; its numerical miss must not be excused
by matching tokens or hidden by a weaker/sparser protocol.

## Final evidence audit and limitations

The [stage verification](../benchmarks/results/refined-m3-max-2026-10-02/verification.json)
recomputes gates from all 2,048 final cached observations and actual source
projection inventories, validates sealed statistics/precision partitions, and
checks four complete calibration/statistics files. It also checks 21 pre-fit
inputs/tools, all 110 archived source members and their archive hash, and 115
unchanged prior complete files. This includes the prior scale results/statistics
and their bindings to earlier cached-checkpoint artifacts and results. Current
runtime source intentionally changes; prior result/artifact bytes do not.

The [final verification](../benchmarks/results/refined-m3-max-2026-10-02/final-verification.json)
repeats the raw stage readback, binds the independent reconstruction records,
checks complete selected matrix/row/coefficient-change coverage, and hashes the
[clean 255-test JUnit report](../benchmarks/results/refined-m3-max-2026-10-02/test-results.xml).
Zero tests are skipped or failed. `readback_passed=true` certifies consistent
recorded evidence, not passing both-family quality: `calibration_stage_passed=false`
and `strict_checkpoint_passed=false` remain. Audit/diagnostic source hashes were
registered before fitting, and result files are never overwritten.

Prose clarification: the frozen final auditor's generic limitation sentence
mentions "failed exact-token gates." In this run **both exact-token gates pass**;
Gemma fails the cosine gate. The structured gates and raw observations are
authoritative. A [hash-bound scope note](../benchmarks/results/refined-m3-max-2026-10-02/scope-clarification.json)
records this wording correction without changing registered source, results,
acceptance criteria, or any failed gate.

The archived implementation and code/test changes remain in the working checkout;
no release tag is created. Lint, whitespace and shell syntax checks pass. Existing
host/UBSan binary results do not establish a new compiler build, ASan, Linux CI,
RTX correctness/performance, or hardware-counter profile. For this recipe there
is no new official packed-model kernel, dynamic-arrival/boundary, 32K, held-out,
latency, throughput, memory, or power certification. Earlier FP16/resource results
retain their own original scope. Prefix caching, speculation, service routing,
clusters, and the coding harness are not added by this checkpoint.
