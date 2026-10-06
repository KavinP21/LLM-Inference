# Qwen-only frozen INT8 downstream validation

Date: 2026-10-02. The validation implementation and official-model run are
complete. **Strict INT8 acceptance fails on held-out greedy parity.** FP16 and
the desktop app remain unchanged; Gemma is not certified by this Qwen-only run.
There is no new resource, throughput, latency, profiler-counter or CUDA claim.

## Registration and implementation

The [downstream contract](qwen-validation.md) and
[driver](../benchmarks/run_qwen_validation_checkpoint.py) implement a separate
checkpoint, not an extension of the stopped two-family calibration experiment.
The sealed Qwen policy is reused without fitting, search expansion, weight or
scale changes, corpus replacement, threshold relaxation or new tie handling.

The [registration](../benchmarks/results/qwen-validation-m3-max-2026-10-02/contract.json)
was saved at `2026-10-02T18:42:14.623823+00:00`, before official export or any
held-out generation. It binds 10 complete input/tool files, 134 preserved prior
complete files, and 104 captured source members including 37 test files. The
actual dirty source ZIP, not the Git commit alone, recovers this implementation.

- Runtime SHA-256: `5387bb7931410a40862e9ca6d6d9d635af37f0d5b3fd8d63e28808f74b09561b`.
- Policy SHA-256: `22013ece3716df887ae0af74dd3206a0f8d43a2d74b2452f8dcd6321e7a51de3`.
- Statistics SHA-256: `0fcb53197412076af2d3244c1b5087f9f88d2988ac52b0694e74f4cccd4fba99`.
- Contract SHA-256: `e8ddbf7a58102b768a2630c1caebc4fb4c46251e27f8f5095fbd7085d55ee09a`.
- Source archive SHA-256: `5ab2f501f936fc61c8632a8ede72aa9ba8700ad7401ca7203e7648b68728e4f5`.

The reference is the identical Forge FP16 Qwen2.5 0.5B Instruct artifact with
source data hash `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45`.
This is not a new Transformers or statistically broad language-quality evaluation.
The fresh refined 25-prompt corpus was output-blind at registration and is now
observed; it cannot serve as untouched regression data for a future fitted method.

Local execution: Apple M3 Max, MLX 0.32.2, Python 3.12.10, NumPy 2.5.3. Git HEAD
`a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a` is dirty. GPU evidence runs are serial
under the inherited exclusive lease. No inference runtime source changes here.

## Export and integrity

The candidate has 21 INT8 / 147 retained FP16 projection matrices, representing
25.5769% of eligible FP16 projection bytes. Embeddings, LM head, norms, biases,
activations and K/V remain FP16. Both quality runs independently recompute every
selected packed coefficient and scale from the sealed statistics/recipe and
verify all retained tensors against the source.

| Artifact storage | Bytes |
| --- | ---: |
| Source FP16 file | 988,084,480 |
| Mixed FP16/INT8 candidate | 896,975,360 |
| Difference | 91,109,120 |

The file is 9.2208% smaller, not 25.5769% smaller overall. This is disk storage,
not resident GPU memory or a speedup. The candidate data SHA-256 is
`35cd24b96996b877e25c01b182c87b6ea5a2399d63b997c9b5761ab3d0204619`;
its complete file SHA-256 is
`9679e5bc24a59f23fc9d6d5a177ac56c6ac5a9d4d675c32c552e52544b319806`.
The [frozen export record](../benchmarks/results/qwen-validation-m3-max-2026-10-02/frozen.json)
also binds the manifest. The export remains experimental, not installed as a default.

## Held-out quality

Each execution mode covers 25 prompts, 32 raw greedy outputs without EOS
truncation, and all 800 cached teacher-forced positions. Rowwise projections
retain the same-artifact numerical contract established earlier. The cosine
threshold remains 0.999; exact full continuation parity is a separate gate.

| Observation | Native reconstruction | Direct Metal diagnostic |
| --- | ---: | ---: |
| Exact source continuations | 21/25 | 21/25 |
| Changed cached source decisions | 6/800 | 6/800 |
| Minimum projection cosine | 0.999926447 | 0.999926447 |
| Minimum cached logit cosine | 0.999225319 | 0.999230313 |
| Exact composed-baseline continuations | 25/25 | 25/25 |
| Minimum candidate/composed logit cosine | approximately 1 | 0.999990497 |
| Candidate/composed byte-equal cached rows | 800/800 | 5/800 |
| Numerical, kernel, cleanup gates | Pass | Pass |
| Source exact-token gate | **Fail** | **Fail** |

The [native report](../benchmarks/results/qwen-validation-m3-max-2026-10-02/quality/reconstruct.json)
is acceptance-bearing; the
[direct report](../benchmarks/results/qwen-validation-m3-max-2026-10-02/quality/metal.json)
is diagnostic. Its token agreement on this corpus does not imply bitwise logits
or interchangeable execution policies. The separate
[byte control](../benchmarks/results/qwen-validation-m3-max-2026-10-02/quality/native-composed-bytes.json)
checks dtype, shape and signed-zero-sensitive hashes on all 800 native/composed
cached rows. All private caches and engine reservations are reclaimed.

Native failures, with zero-based case/position indices:

| Case | Prompt topic | First divergence | FP16 top-two margin |
| --- | --- | ---: | ---: |
| 3 | Glacier versus iceberg | 27 | 0 |
| 5 | Circuit-breaker labels | 6 | 0.03125 |
| 9 | Thank-you note to neighbor | 10 | 0.015625 |
| 11 | Pop from an empty stack | 0 | 0 |

Cached changes also occur at positions 8 and 22 of case 5. All six are near-tie
or tied source decisions; matching global cosine does not guarantee preserved
argmax. These are recorded margins, not permission to change tie handling or
waive the exact-token gate. Byte identity of the two reconstruction paths narrows
this diagnostic comparison; it is not a proof of general quantization quality.
No controlled accuracy comparison with earlier policies on different corpora is
claimed.

## Stop rule, tests and final readback

The saved [stage verification](../benchmarks/results/qwen-validation-m3-max-2026-10-02/verification.json)
records `rejected_at_held_out_quality` and `strict_checkpoint_passed=false`.
The run exits 1 with complete saved rejection evidence. It creates **no** batching,
boundary, 32K or performance-matrix results. Their driver paths are implemented
and gated, but official-model execution is deliberately untested for this candidate.
No conditional GPU performance result should be inferred from their existence.

All 287 portable/Metal tests pass before and after the official run, including
32 new portable contract tests. Coverage includes real CPU artifact export and
derivation, live policy/corpus/file drift, all-position quality consistency,
non-finite metrics, tie-aware readback without runtime tie changes, malformed byte
identities, complete negative evidence, no-clobber behavior and forbidden
downstream calls. A separate portable CI workflow is added without editing prior
registered CI/tool files. Local passes are not a claim that remote CI ran.

Existing C++ host and UBSan test binaries pass again. No fresh CMake/Ninja,
ASan, Linux execution, RTX integration or hardware-counter result is claimed.
The [JUnit evidence](../benchmarks/results/qwen-validation-m3-max-2026-10-02/test-results.xml)
and complete-file index are checked by the final host-only audit. Evidence
readback can pass while model acceptance fails; these are distinct outcomes.

The [final verification](../benchmarks/results/qwen-validation-m3-max-2026-10-02/final-verification.json)
passes complete-file/raw-gate readback and checks nine indexed evidence files,
287 tests with zero failures/errors/skips, all 1,600 quality rows, all 800 native
byte-control rows, and every prior/input/source binding. It still explicitly
records `strict_checkpoint_passed=false`. Final JUnit SHA-256 is
`98b45ca0be13994d3ae2f78456df7a9671151056889ee5e69b7476219f9bd482`;
the evidence-index SHA-256 is
`c5a00950a3455da196a2fec9ca4ff9e1020d6c7a362b8c9dd371bcad30a2f393`.

## Handoff

Preserve the candidate, policy, statistics, corpora and every prior experiment.
No release tag or W4A16 advancement follows this rejection. Gemma remains FP16
and still needs separate calibration/quality work. A future INT8 method must
register fresh calibration and output-blind regression, not fit to these failures.

The next independent implementation checkpoint is **FP16 prefix caching and
copy-on-write shared pages**, with model/execution-identity isolation, reference
counts, bounded eviction, cancellation/failure cleanup and token/logit parity
against uncached execution. This prevents lossy calibration experiments from
indefinitely blocking the serving/runtime feature roadmap while keeping INT8
acceptance open. The [12-milestone delivery roadmap](roadmap.md) includes both
that open quality gate and the future CUDA, cluster and harness work.
