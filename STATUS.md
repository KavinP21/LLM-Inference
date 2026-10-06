# Implementation status

Last updated: 2026-10-02

This is a development snapshot, not a performance release.

## Validated on Apple M3 Max

- CMake/Ninja `host-debug` configuration and build.
- CTest portable host suite.
- Current full Python regression: 365 tests for portable and Metal-backed model artifacts, export,
  scheduling, physical paging, numerical parity, chunking, batching, cancellation, and cleanup.
- Python syntax for exporter, correctness runner, Forge benchmark, Transformers baseline, matrix
  runner, and result summarizer.
- Cross-language Python-writer/C++-reader model artifacts, including SHA-256 validation.
- A complete synthetic Qwen2 tensor contract through `forge-inspect-model`.
- Shell syntax for WSL environment and Compute Sanitizer scripts.
- MLX 0.32.2 access to the Apple M3 Max Metal device.
- Safe Python parsing and checksum validation of the shared `.engine` model format.
- Backward-compatible artifact version 2 with explicit family, activation, head-width,
  sliding-window, RoPE, scaling, and soft-cap metadata; existing version-1 Qwen artifacts load
  unchanged.
- A backend-independent model contract plus MLX adapter factory for Qwen2 and Gemma 3 text models.
- Complete Qwen2 FP16 execution in MLX: embedding, RMSNorm, biased Q/K/V, RoPE, GQA causal
  attention, output projection, SwiGLU MLP, final normalization, LM head, and greedy selection.
- Physical, lazily materialized 16-token MLX K/V pages with stable block tables and reclamation.
- Backend-neutral request scheduling, cancellation, worst-case capacity reservation, and metrics.
- Scheduler-visible 512-token prefill chunks with decode priority.
- Batched one-token Qwen decode across all runnable requests.
- MLX fused GQA attention for production shapes and an exact online-softmax tiled reference path.
- Synthetic full-model and cached-decode parity against an independent NumPy reference.
- Official `Qwen/Qwen2.5-0.5B-Instruct` export: 291 tensors and data SHA-256
  `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45`.
- Official-model validation over four fixed prompts: identical first 32 raw-greedy tokens,
  identical first-token argmax, and custom-Metal-path logit cosine similarity from `0.9999808` to
  `0.9999977`.
- A four-request MLX benchmark smoke run with zero leaked KV reservations after completion.
- Official-model execution at 2K, 4K, 8K, 16K, and 32K context, with exact expected page counts
  and zero leaked physical pages after every case.
- A controlled materialized/tiled/fused attention A/B plus a captured Metal GPU trace.
- Shape-specialized custom Metal residual/RMSNorm, RoPE/page-write, and SwiGLU kernels.
- A custom online-softmax paged GQA decode kernel driven by per-request block tables.
- Controlled baseline/fusion/full-kernel end-to-end matrices for 128/1K prompts at concurrency 1/8.
- Four end-to-end Xcode GPU captures covering the same 128/1K and concurrency 1/8 shapes.
- A 32,766-token prompt plus two-token generation, including one custom paged-decode iteration,
  with exact 2,048-page occupancy and complete reclamation.
- Complete Gemma 3 text execution: scaled embeddings, offset RMSNorm, explicit attention width,
  Q/K normalization, alternating sliding/global attention, per-layer RoPE bases, tanh-GELU, four
  layer norms, optional attention/final soft caps, tied LM head, and greedy decode.
- Official `google/gemma-3-1b-it` export: 341 tensors, 1,999,794,688-byte artifact, and data SHA-256
  `92d3081b2facfa8a5eb48dcedd89cb8e230b21a00cfeefd4c32f77ca2a9482ee`.
- Official Gemma first-token logit cosine similarity from `0.9999963` to `0.9999982` over four
  prompts, exact first-token argmax for every case, and exact 32-token output for three cases. The
  fourth matches 25 tokens before one documented FP16 near-tie causes a seven-token cascade; this is
  intentionally not claimed as full greedy parity.
- Official Gemma 32K execution with 2,048 physical pages, 872,415,232 peak K/V bytes, one custom
  paged-decode iteration, and complete reclamation.
- An official-model four-request continuous-batching smoke run with no reserved or materialized K/V
  blocks remaining. Its one-trial timing is diagnostic, not a publishable benchmark.

The foundation evidence is in [the first MLX validation report](docs/mlx-foundation-results.md);
the paged, batched, and initial 32K evidence is in
[the MLX 32K validation report](docs/mlx-32k-results.md). Custom-kernel correctness, benchmark, 32K
decode, and trace evidence is in [the MLX Metal optimization report](docs/mlx-metal-results.md).
Gemma-specific evidence is in [the Gemma 3 validation report](docs/mlx-gemma3-results.md).

## MLX custom-kernel checkpoint limitations

- The direct Metal attention kernel consumes block tables over a packed live-page tensor. MLX's
  immutable arrays still require a per-layer `stack` before the launch; this is not zero-copy paging.
- Full custom mode is effectively flat versus baseline for a 1K single-request workload because
  page packing cancels the attention win. The kernel is most beneficial for concurrent decode.
- One prompt is prefilling at a time. Prefill is chunked around decode iterations, but there is no
  multi-prompt chunk packing or preemption policy yet.
- Causal prefill attention remains MLX fused attention over logical page views; the custom paged
  kernel is decode-only.
- The 32K gate is execution/resource evidence, not a statistically rigorous latency benchmark.
- The official Gemma 32-token gate has one numerically unstable greedy decision; the saved report
  distinguishes this from an architecture or cache mismatch and does not mark the gate fully exact.
- Weight-only INT8 is implemented as an opt-in experiment, not a passed quality gate. FP16
  prefix caching now passes validation; its controlled timing/final qualification is pending.
  Speculative decoding, serving, and distributed routing remain future milestones.

## INT8 implementation checkpoint (experimental; quality gate failed)

- Version-3 quantization descriptors, signed INT8 matrices, FP32 per-output-channel scales, and
  Python/C++ validation, while preserving versions 1/2 and unquantized version-2 exports.
- NumPy-only deterministic conversion, tied-weight preservation, source provenance, temporary-file
  validation, and refusal to overwrite existing artifacts/manifests.
- Qwen and Gemma MLX W8A16 execution, with an independently selectable dequantized-matmul baseline.
- Custom Metal small-batch linear and one-pass INT8-to-FP16 reconstruction kernels. Decode reuses
  weight loads across up to 16 rows; prefill retains MLX GEMM without full-sized FP32 intermediates.
- Official INT8 artifacts: 168 Qwen and 182 Gemma projection matrices; embeddings, head, biases,
  norms, activations, and K/V remain FP16. Weight files are roughly 36%/35% smaller respectively.
- All-layer numerical checks and 25-prompt, 32-token greedy regression, with teacher-forced logit
  comparisons and separate kernel-versus-dequantized checks. Failed gates are saved and exit nonzero.
- Controlled FP16/naive-INT8/custom-INT8 matrices and memory accounting; no universal speedup claim.
- Both official INT8 artifacts pass the 32,766-prompt-token/two-output-token resource gate, including
  a custom paged-decode step, exact 2,048-page occupancy, and complete reclamation.
- Final Python regression sweep: 77 tests pass (50 portable/reference/export tests and 27 Metal
  tests), including cross-language quantized artifacts and unchanged FP16 family tests.
- C++ host and undefined-behavior sanitizer checks pass. Local AddressSanitizer stalled in its own
  startup before `main`; it is not claimed to pass. Linux sanitizer CI has been added but awaits CI.

The strict INT8 checkpoint has **not passed**. Default FP16 artifacts and the desktop app remain
unchanged. See [the INT8 contract](docs/quantization.md) and
[the results and limitations](docs/mlx-int8-results.md). Do not advance to W4A16 yet.

## Implemented but awaiting RTX validation

- CUDA compilation for SM 8.6.
- cuBLASLt row-major FP16 linear layers, bias epilogues, and cached plans.
- FP16 Qwen2 full-prompt prefill and batched decode execution.
- Fused RMSNorm/residual, RoPE/paged-cache write, SwiGLU, causal softmax, paged GQA attention,
  embedding, add, and greedy argmax kernels.
- pybind11 `Engine` extension.
- CUDA numerical tests and Compute Sanitizer execution.
- Qwen2.5-0.5B logit similarity and greedy-token parity.
- Nsight Systems/Compute profiles and all performance results.

No CUDA performance number or correctness claim should be published until the second section has
been run successfully on the RTX 3070 Ti.

## INT8 accuracy hardening (implemented; acceptance gates still failed)

- Separate 16-prompt calibration corpus and normalized/tokenized guards against overlap with the
  existing 25-prompt regression; full-model single-projection ablation and combined-policy trials.
- Source-bound/checksummed mixed-precision policies, explicit FP16 retention in version-3 exports,
  unchanged deterministic INT8 rounding, and a mandatory 25% eligible-byte floor for policies.
- Native-GEMM `int8_mode="reconstruct"` at every row count, without permanent dense reconstruction.
- Frozen mixed artifacts: Qwen 21 INT8/147 retained FP16 projections, Gemma 22/160. Files are only
  9.22%/8.73% smaller; do not claim the previous all-INT8 storage saving for these models.
- Both numerical gates now pass: minimum held-out logit cosine 0.999849 Qwen and 0.999643 Gemma.
  Exact 32-token parity remains 19/25 and 22/25, respectively. Both candidate/composed comparisons
  match 25/25 on these frozen policies. Default FP16/app behavior is untouched.
- Dynamic arrivals, mixed output budgets, materialized cancellation, and cache cleanup: Qwen 25/25
  independent-output parity in both modes; Gemma 24/25 in both modes (failed acceptance gate).
- Broader Gemma FP16 baseline replay also fails 2/25 at tied/near-tied decisions. Independently
  replayed divergence logits have cosine above 0.9999995 and top-two margins 0/0.015625. The
  responsible batch-shape operation has not been isolated; do not claim complete batched Gemma parity.
- Both modes/families pass 32,766-prompt-token plus two-output-token execution with 2,048 pages
  and complete reclamation. This is a resource gate, not long-context semantic quality.
- Fresh 32-result FP16/composed/direct/reconstruction matrix, 5 full warmups and 3 trials per row;
  all raw results/checksums/provenance preserved. Overlapping diagnostic Qwen runs are quarantined,
  not used for published numbers. Cooperating drivers now have a tested inherited exclusive lease.
- Final regression suite: 98 tests pass; host/UBSan and mixed-artifact C++ inspection pass. Linux
  sanitizer CI and full Xcode counter inspection remain unverified/unavailable as previously noted.

See [the hardening report](docs/mlx-int8-hardening-results.md). No v0.1.0 tag or W4A16 advancement.

## Decode numerical-consistency checkpoint (passed, opt-in MLX policy)

The batch-shape issue is now isolated to native FP16 projection reductions on identical inputs.
The opt-in `decode_mode="rowwise"` policy preserves single-request reductions while retaining
batched attention/scheduling. All six FP16/frozen-mixed execution configurations match independent
tokens and bitwise logits on the 25-prompt concurrency 2/8/16 and staggered-arrival matrix; original
default single-request tokens are unchanged. The default throughput path remains unchanged and its
negative controls remain saved. See [the numerical contract](docs/decode-numerics.md).

- 16,230 exact arrival-trace logit rows and another 1,728 exact official-artifact boundary rows,
  including page/chunk boundaries and Gemma's 512-token sliding window.
- Paired Gemma teacher-forced stage replay through token 17 reproduces the default near-tie flip
  and eliminates all captured stage differences in rowwise mode.
- Six 32K prompt-plus-decode resource gates with exact 2,048-page occupancy and full reclamation.
- A fresh 32-result controlled batched/rowwise throughput/memory matrix, five complete warmups and
  three trials per row. The consistency mode is slower at short-context concurrency 8 and stays
  opt-in; no universal speedup or counter-based attribution is claimed.
- Final regression: 119 tests pass; host/UBSan pass. Artifact/workload/source/result hashes and
  matrix completeness are audited; failed default controls and failed INT8 quality remain visible.

See [the results and tradeoffs](docs/mlx-batch-numerics-results.md). The desktop defaults and
single-request outputs are unchanged. No tag, CUDA validation, or W4A16 advancement.

## Second-order INT8 checkpoint (implemented; strict acceptance still failed)

- NumPy block-diagonal activation covariance, FP16-rounded error compensation, stable within-block
  activation ordering, damping, and row/block RTN fallback. This is a bounded approximation, not full GPTQ.
- Joint precision selection tests candidates with all already selected projections, rather than
  ranking isolated perturbations alone. Fixed 32-prompt calibration, old/fresh 25-prompt regressions,
  both model policies frozen before evaluation, and no post-regression retuning.
- Source-bound, checksum-validated, bounded no-pickle archives; sealed policies; exact packed-value/
  scale rederivation; same-partition RTN controls; unchanged version-3 runtime weights/API/defaults.
- Qwen 22 INT8/146 FP16 projections and Gemma 22/160. Local calibration reconstruction loss is
  about 44%/48% lower, but greedy results against RTN are mixed, not a general accuracy win.
- All twelve numerical quality gates pass; exact source-output quality still fails. Direct kernels
  also fail six cross-implementation greedy decisions, reproduced by cached stage replay on identical
  projection inputs at ties/one-ULP final margins. Native reconstruction matches its composed baseline.
- Four candidate contracts pass 10,820 arrival-trace and 1,152 boundary bitwise logit rows, event/
  cancellation/cleanup checks, and four 32K prompt-plus-decode resource gates.
- Fresh serialized 32-result FP16/composed/direct/reconstruction matrix with five warmups and three
  trials. Storage savings are only about 9.3%/8.7%; runtime/peak-memory tradeoffs depend on workload.
  No universal speedup or hardware-counter attribution is claimed.
- Final full regression: 141 tests pass. Existing C++ host/UBSan and both new official-artifact
  reader checks pass; no fresh CMake build, ASan, Linux CI, or RTX result is claimed.
- Readback verifies 74 indexed JSON checksums, both statistics archives, and 17 complete-file
  before/after bindings. All old experimental policies, artifacts, and results remain preserved.

See [the method](docs/second-order-quantization.md), [the results and limitations](docs/mlx-second-order-results.md),
and [the final verification](benchmarks/results/second-order-m3-max-2026-10-01/verification.json).
This is not a passed strict INT8 checkpoint. FP16/app defaults remain unchanged; no tag or W4 advancement.

## Cached-decode calibration checkpoint (implemented; strict acceptance failed)

True paged cached teacher forcing, source prompt/decode activation sampling, and bounded joint
policy repair are implemented. Replay honors context/KV limits, owns a private reservation/device
store, requires an idle engine, and leaves scheduler state unchanged. All 184 tests pass, including
bitwise scheduler-versus-replay tests for both families and FP16/direct/reconstructed execution.
Projection aliases are safely split by export; this calibration rejects them for byte accounting.

Both new policies and whole files are frozen before regression. Qwen accepts three repairs
(6 → 5 → 4 → 3 cached mismatches); Gemma accepts one (4 → 3). Both match 29/32 free calibration
continuations and pass cosine 0.999, but strict calibration remains failed. No held-out fitting,
precision/tie/default changes, release tag, or W4 advancement.

- Complete 12-report regression checks all 9,600 cached logit positions. Every candidate still
  fails exact source output; Gemma also fails cosine 0.999 on the original corpus, including native
  reconstruction and RTN controls. Stronger cached coverage is not interchangeable with old sparse checks.
- Direct kernels fail eight cross-implementation greedy cases. All 132 synthetic projection probes
  match independent reconstructed values/native projections numerically, while direct outputs differ.
  Separate byte-sensitive replays verify all 3,200 native/composed cached rows; this is not FP16 accuracy.
- All four candidate contracts pass 10,820 arrival-trace and 1,152 boundary byte-sensitive logit rows,
  materialized cancellation, event checks, and four 32K prompt-plus-decode resource gates with
  exact 2,048-page occupancy and full reclamation. These are not long-context semantic-quality results.
- Fresh isolated 32-result, five-warmup/three-trial performance matrix with 432 measured requests;
  weights are only 9.22%/8.79% smaller, and latency/throughput/peak-memory tradeoffs remain shape-dependent.
- Independent final readback verifies 76 indexed JSON files, the JUnit report, 17 complete frozen files,
  all-position quality/resource gates, and supplemental byte identities. Strict quality stays false.
- The full 184-test suite passes before and after measurement. Existing C++ host/UBSan and candidate
  reader checks pass; no fresh CMake build, ASan, Linux CI run, RTX result, or counter profile is claimed.

See [the contract](docs/cached-calibration.md), [the complete results](docs/mlx-cached-calibration-results.md),
and [final verification](benchmarks/results/cached-calibration-m3-max-2026-10-01/verification.json).
All 74 prior indexed JSON checksums, two statistics archives, and 17 complete prior files remain unchanged.

## Scale-aware INT8 fitting (implemented; stopped at failed calibration)

The new offline method uses a fixed activation-weighted scale grid, FP16-rounded block error
compensation, and whole-row fallback to the old recipe. Old policies/archives keep their original
derivation, format, kernels, and defaults. The method/search/corpora were registered before fitting,
with a recoverable archive of dirty runtime sources and a mandatory stop-before-held-out rule.

- Both complete 32-prompt calibration runs pass cosine 0.999, private cleanup, and the 25% eligible
  projection-byte floor, but exact output fails: Qwen 25/32 free continuations and 8/1,024 changed
  cached decisions; Gemma 29/32 and 3/1,024. Each accepts one of two bounded repair rounds.
- Independent host reconstruction verifies all 44 selected matrices / 255,360 output rows. Local
  block-diagonal loss is 16.95%/9.38% lower on the same moments/partition, with no worsened row.
  This is not a language-quality or speedup claim.
- Both rejected policies/statistics and raw trials are preserved. No official packed export,
  held-out inference, performance matrix, or new 32K/resource certification was run for this method.
  The fresh regression corpus remains output-blind, and the search was not extended after failure.
- All 222 tests pass; existing host/UBSan binaries pass. Final readback verifies raw gates,
  independent diagnostic coverage, JUnit, 15 pre-fit bindings, 67 captured source files, and
  93 unchanged complete files from the previous cached checkpoint. Strict acceptance stays false.

See [the method](docs/scale-aware-quantization.md),
[the results and stopping point](docs/mlx-scale-aware-results.md), and
[final verification](benchmarks/results/scale-aware-m3-max-2026-10-01/final-verification.json).
FP16/app defaults remain unchanged. No W4 advancement, release tag, or new CUDA claim.

## Coordinate-refined INT8 (implemented; Qwen calibration passes, combined gate fails)

A separate, pre-registered two-sweep integer-coordinate pass follows scale fitting, keeps its
row scales byte-identical, and accepts changes only after complete block/row loss checks. All
old quantizers, formats, inference kernels, precision/tie handling and FP16/app defaults remain.

- Both families match all 32 free continuations and all 1,024 cached decisions on fresh calibration.
  Qwen passes cosine 0.999 with minimum 0.999688892. Gemma's minimum is 0.998753977, so its
  numerical gate fails; matching argmax does not excuse this. No repair is accepted for either.
- Independent reconstruction verifies all 44 selected matrices / 255,360 rows, unchanged scales,
  exact coefficient change counts and no worsened row. Local loss is 2.9486%/3.2552% lower than
  scale fitting alone on the same moments/partition; no speedup or general accuracy claim.
- The both-family stop rule is honored. No official packed export, held-out inference, new 32K
  resource gate or performance matrix is executed. Qwen's individual pass does not waive it.
- All 255 tests pass before and after fitting; existing host/UBSan binaries pass. Readback checks
  2,048 raw cached observations, four calibration/statistics bindings, 21 pre-fit inputs/tools,
  110 captured source members (including 36 tests), and 115 unchanged complete prior files.

See [the method](docs/refined-quantization.md), [the complete results](docs/mlx-refined-results.md),
and [final verification](benchmarks/results/refined-m3-max-2026-10-02/final-verification.json).
The strict INT8 checkpoint remains unpassed; no tag, W4 advancement or new CUDA claim.

## Next executable checkpoint

1. Finish **FP16 prefix-cache performance/final qualification** from the saved
   [safe handoff](docs/prefix-cache-handoff.md). Implementation and all eight official
   parity/resource gates pass, including 32K. Run the frozen 24-report cold/disabled/primed
   matrix and final audit before closing the milestone. Keep quantization acceptance independent.
2. Preserve the stopped calibration and the separate Qwen held-out rejection below, including
   their source archives. Do not enlarge either search, retune on observed regression prompts,
   weaken numerical/token gates, or install the candidate as certified. Future fitting requires
   new calibration-only registration and output-blind regression. Gemma remains FP16 and needs
   its own numerical/quality work before any promotion.
3. Full Xcode counter/capture inspection still awaits `xctrace` on this host;
   direct kernels remain a separate numerical contract.
4. Evaluate faster shape-invariant projections with controlled A/B tradeoffs before speed claims.
5. Only after strict INT8 quality also passes, add groupwise W4A16 with equivalent checks.

The CUDA implementation remains preserved for later RTX validation. No CUDA performance or
correctness claim should be published until the WSL2 runbook succeeds on the RTX 3070 Ti.

## Qwen-only downstream checkpoint (complete; strict quality rejected)

The separate frozen-policy validation driver exports the already calibration-eligible Qwen
policy without changing fitting, inference sources, gates or defaults. Registration precedes
official export and every held-out output, preserving the stopped two-family calibration run.

- Native reconstruction and direct Metal each match 21/25 held-out continuations and change
  6/800 cached decisions. Numerical gates pass (minimum cosine 0.999225319/0.999230313), but
  exact-token gates fail. Direct Metal remains a diagnostic contract, not native certification.
- Every packed coefficient/scale is rederived, retained tensors match the FP16 source, and all
  800 native/composed cached logit rows match byte-for-byte with complete cache reclamation.
- The file is 896,975,360 bytes, 9.2208% smaller on disk; no latency or resident-memory claim.
- The registered stop rule prevents all batching/boundary/32K/performance execution. All 287
  tests pass before and after; 32 new portable checks plus separate CI enforce evidence integrity.
- Readback preserves 134 complete prior bindings, 10 registered input/tools and 104 archived
  source members including 37 tests. INT8 remains experimental; FP16/app defaults are unchanged.

See [the contract](docs/qwen-validation.md), [the results](docs/mlx-qwen-validation-results.md)
and [final verification](benchmarks/results/qwen-validation-m3-max-2026-10-02/final-verification.json).
There are [12 major delivery milestones remaining](docs/roadmap.md), not 12 guaranteed turns;
the failed INT8 acceptance remains open, while FP16 prefix-cache qualification proceeds independently.

## FP16 prefix caching (implemented; validation passed; timing deferred)

- Shared immutable physical pages, reference counts, pre-reserved partial-tail COW,
  namespace/model/execution identity, bounded LRU eviction and safe cache-aware admission.
- Both official families match uncached tokens/logit bytes over 40 cases × two chunk sizes,
  eight shared/branched requests, namespaces, eviction and 32K cached execution. Warm exact
  prompts execute no prefill. All eight gates and fresh raw-evidence readback pass.
- The broader protocol exposed an existing forced-fused Qwen short-tail dispatch failure;
  that partial run is preserved. A capability guard plus eight Metal regressions fixes it;
  a fresh frozen run reruns all validation with unchanged workloads/gates.
- 365 full-suite tests pass with saved no-skips JUnit; existing C++ host/UBSan binaries pass.
  Readback verifies 160 complete prior files and 104 live/captured source members.
- User requested a safe stopping point within ten minutes. The GPU driver stopped after
  durable validation and before any timing; no GPU process remains. No speedup or complete
  milestone acceptance is claimed. FP16/app defaults and INT8/CUDA boundaries are unchanged.

See [the API/contract](docs/prefix-cache.md), [validation results](docs/mlx-prefix-cache-results.md)
and [resume commands](docs/prefix-cache-handoff.md). Finish the registered benchmark/final
audit next; only then close this milestone and move to FP16 speculative decoding.
