# MLX scale-aware INT8 calibration results

Date: 2026-10-01. This is a completed **calibration-only experiment**, not a
passed strict INT8 milestone or performance release. The implementation and
reconstruction bound are validated; both official policies fail exact-token
calibration. The registered stop rule was followed: no official packed candidate
was exported, no held-out inference ran, and no new performance/resource matrix
was produced. FP16 execution and desktop defaults remain unchanged.

## Method and frozen contract

The [method contract](scale-aware-quantization.md) describes a fixed seven-factor
per-output-row scale grid, activation-weighted FP16-rounded reconstruction loss,
block error compensation, and a whole-row q/scale fallback to the unchanged
fixed-scale recipe. The method changes offline fitting, not W8A16 execution,
format, accumulation, or greedy tie handling. Cross-block covariance is discarded;
the loss guarantee is local to the collected block-diagonal moments.

The [pre-fit registration](../benchmarks/results/scale-aware-m3-max-2026-10-01/contract.json)
was saved at `2026-10-01T23:01:27.534820+00:00`, before either official fit:

- Block size 64, damping 0.01, stable activation ordering, objective row chunks 256.
- Scale factors `1.0, 0.995, 0.99, 0.98, 0.96, 0.92, 0.85`.
- Four probe cases/candidate matrices, two repair rounds/removal candidates/full trials.
- At least 25% of eligible FP16 projection bytes selected for INT8.
- 32 fresh calibration prompts, 32 raw greedy tokens, all 1,024 cached positions
  per family, and opt-in rowwise projections. EOS does not truncate these checks.
- Normalized/tokenized overlap guards against 180 prior/fresh corpus strings.
  The fresh 25-prompt regression corpus was used only for overlap rejection;
  its output tokens and logits remain untouched.

The official FP16 artifacts are `Qwen/Qwen2.5-0.5B-Instruct` and
`google/gemma-3-1b-it`. Their data checksums remain respectively
`87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` and
`92d3081b2facfa8a5eb48dcedd89cb8e230b21a00cfeefd4c32f77ca2a9482ee`.
References here are the existing source FP16 engine, not a new Transformers
accuracy certification.

Execution used Apple M3 Max, MLX 0.32.2, macOS 26.5 arm64, Python 3.12.10,
and NumPy 2.5.3. Git HEAD was
`a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a` with a dirty checkout.
Registered runtime SHA-256 is
`94c10909233a85c5b6d03930962e868ac3a3c3b43751393cd5d0fc34d58f8074`.
The pre-fit archive captures the actual dirty runtime/workflow/corpus files;
the Git commit alone does not reproduce them.

## Official-model calibration

Every final policy is checked on the complete corpus, separately for free
generation and source-token cached teacher forcing. Numerical similarity does
not substitute for exact decisions.

| Observation | Qwen2.5 0.5B | Gemma 3 1B |
| --- | ---: | ---: |
| Exact free continuations | 25/32 | 29/32 |
| Changed cached decisions | 8/1,024 | 3/1,024 |
| Minimum cached logit cosine | 0.999799623 | 0.999720379 |
| INT8 / retained FP16 projections | 21 / 147 | 23 / 159 |
| Eligible projection-byte fraction | 25.5769% | 25.2747% |
| Distinct policy evaluations / bound | 95 / 693 | 109 / 749 |
| Accepted repairs / attempted rounds | 1 / 2 | 1 / 2 |
| Numerical / floor / private cleanup | Pass | Pass |
| Exact-token calibration gate | **Fail** | **Fail** |

Qwen's accepted repair improves the full-calibration score's secondary relative
error measure but leaves eight cached mismatches. Gemma's accepted repair reduces
cached mismatches from five to three. The second round is rejected for both.
No extra rounds, scale factors, precision exceptions, or tie rules were added
after the results were inspected. Search results on this fresh corpus are not
a controlled language-quality A/B against older policies on different corpora.

Complete raw cases, trial histories, local objectives and sealed policy partitions
are in [Qwen calibration](../benchmarks/results/scale-aware-m3-max-2026-10-01/calibration/qwen.json)
and [Gemma calibration](../benchmarks/results/scale-aware-m3-max-2026-10-01/calibration/gemma.json).
The [stage checkpoint](../benchmarks/results/scale-aware-m3-max-2026-10-01/calibration-checkpoint.json)
records `rejected_at_calibration`, with every downstream execution flag false.
Offline fitting durations in that file are workflow wall times, not inference
latency measurements or a calibrated cross-method speed comparison.

## Independent reconstruction readback

After fitting stopped, a separate host-only diagnostic rederived the selected
packed coefficients/scales and the unchanged fixed-scale baseline on the **same
moments and precision partition**. An independent FP16 reconstruction plus FP64
`einsum` evaluates each row's block-diagonal quadratic, rather than using the
fitter's objective helper or trusting its success flags.

| Selected projections only | Qwen | Gemma |
| --- | ---: | ---: |
| Matrices / output rows verified | 21 / 102,144 | 23 / 153,216 |
| Fixed-scale aggregate objective | 3.331734579 | 3.020819536 |
| Scale-aware aggregate objective | 2.766992569 | 2.737583225 |
| Relative local-loss reduction | 16.95% | 9.38% |
| Rows with changed scales | 58,194 | 89,925 |
| Maximum per-row loss increase | 0 | 0 |

All 44 matrices and 255,360 rows were checked. No positive row deltas occurred,
even within the diagnostic's `1e-9` floating-reduction verification tolerance.
The fitter's acceptance comparison itself remains strict; this readback tolerance
does not relax inference or token gates. Values are calibration-local and are
not comparable across the two models. Lower reconstruction loss is **not** a
demonstrated general accuracy improvement, storage result, or inference speedup.

The [independent diagnostic](../benchmarks/results/scale-aware-m3-max-2026-10-01/objective-verification.json)
contains every matrix's objectives, shape, changed-row counts, coefficient/scale
hashes, statistics/report bindings and the diagnostic source hash. It creates
no packed official `.engine` file. Its code and the final auditor were added
after registration, strictly as read-only diagnostics—not new fitting choices.

## Integrity and regression

- All **222 tests pass**, including 36 new portable tests and two new Metal tests.
  The new coverage includes objective bounds, deterministic old/new bytes,
  malformed configurations/statistics, no-clobber exports, source archives,
  complete failed-gate handling, raw-evidence tampering, and final readback.
- Tiny Qwen and Gemma artifacts pass the new calibration → version-3 export →
  derivation path, with byte-identical native/composed cached logits. This is
  integration evidence, not official-model exact-source accuracy.
- The [stage audit](../benchmarks/results/scale-aware-m3-max-2026-10-01/verification.json)
  reconstructs the gates from all 2,048 final cached observations, validates
  actual source projection inventories and sealed statistics, and verifies four
  complete calibration/statistics file bindings.
- It also verifies 15 pre-fit complete input/tool hashes, 67 captured source
  members and the source archive checksum, and **93 unchanged prior complete
  file bindings** covering the previous cached checkpoint's indexed results,
  artifacts/statistics/manifests, index, verification and JUnit report.
- The [final audit](../benchmarks/results/scale-aware-m3-max-2026-10-01/final-verification.json)
  repeats the stage readback, checks selected-row diagnostic coverage/provenance,
  and binds the [clean 222-test JUnit report](../benchmarks/results/scale-aware-m3-max-2026-10-01/test-results.xml).
  `readback_passed=true` means the evidence is consistent;
  `calibration_stage_passed=false` and `strict_checkpoint_passed=false` remain.
- Existing C++ host and UBSan binaries pass again. No fresh CMake/Ninja build,
  ASan success, Linux CI execution, RTX run, or hardware-counter profiling is
  claimed. Lint, shell syntax and whitespace checks pass.

The archived runtime intentionally differs from the older checkpoint's source
hash because it implements this new recipe. Older artifact/result bytes remain
unchanged; their own source-bound drivers should not be rerun against the new
runtime as if nothing changed. Use their recorded source identities.

## Stopping point and next gate

Preserve this run unchanged. Do not export/rebenchmark these rejected official
policies, broaden its search, or promote it to W4A16 or a release tag. Any further
weight-method experiment needs a separately frozen calibration contract and
genuine held-out discipline; more local fitting alone has not closed exact greedy
parity. The fresh regression corpus remains output-blind for an eligible future
checkpoint.

This method has no new official-model direct-kernel, arrival/boundary, 32K,
held-out, latency, throughput, memory, or profiler certification. Existing FP16
32K/paging/batching results remain scoped to their original reports. Prefix caching,
speculation, service routing, clusters, and the coding harness remain later work.
