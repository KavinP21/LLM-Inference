# Ordered implementation roadmap

## Remaining delivery milestones

There are **12 major milestones remaining**, including strict INT8 acceptance below.
This is a delivery roadmap, not a count of turns or experimental attempts. A failed
acceptance gate leaves its milestone open; calibration, validation and optimization
may each need multiple checkpoints. The completed MLX/Metal foundation, physical
paging, continuous batching and 32K execution are not counted again.
FP16 prefix caching is now implemented and all registered validation gates pass,
but its controlled benchmark/final qualification was deferred at the user's stopping
request. It stays in this count until final acceptance; see [the handoff](prefix-cache-handoff.md).

| # | Milestone | Acceptance boundary |
| --- | --- | --- |
| 1 | Strict per-model INT8 acceptance | Frozen Qwen policy passes fresh held-out quality, derivation, batching, resource and controlled measurement gates. Gemma stays FP16 until separately qualified; never pool family results. |
| 2 | Groupwise W4A16 | Explicit formats and packing, MLX/Metal kernels, strict quality and memory/latency A/B on qualified models. |
| 3 | Prefix caching and copy-on-write | Implementation, shared-page/COW/isolation/reclamation and 32K parity gates pass. Frozen cold/disabled/primed A/B and final complete audit remain pending. |
| 4 | Speculative decoding | Draft/target verification, exact greedy acceptance, rollback-safe paged caches and measured end-to-end benefit including draft overhead. |
| 5 | Multi-user streaming service | Concurrent sessions, cancellation, bounded queues, backpressure, authentication, per-user limits and streaming/API tests. |
| 6 | Model pools and resource-aware routing | Admission and load balancing across independent workers, memory budgets, fairness, overload/recovery behavior and mixed-load SLOs. |
| 7 | CUDA on-device validation and feature parity | RTX correctness, Compute Sanitizer, Nsight evidence and measured parity for supported MLX features; no NVIDIA claims from Apple measurements. |
| 8 | Single-node multi-GPU execution | NCCL tensor parallelism, collective correctness, topology-aware placement and measured scaling/communication costs. |
| 9 | Multi-node cluster orchestration | Worker discovery, leases, health checks, failure recovery, placement and controlled distributed load tests; distinguish routing from tensor parallelism. |
| 10 | Qualified model registry and switching | Capability-aware loading and safe swaps across additional families, model-specific context/quantization contracts, tokenizer identity and quality fixtures. No automatic claim for an unimplemented model. |
| 11 | Coding harness v1 | Stable engine/service interface, multi-turn context, file/search/shell/Git tools, permissions, checkpoints and reproducible task evaluations. |
| 12 | Harness hardening and release | Sandboxing, recovery, regression/task evaluations, observability, packaging and measured latency/resource/SLO reports. This does not promise proprietary-harness parity. |

Recommended sequence is the numbered order, with two independent tracks: FP16
prefix caching/serving can proceed while a quantizer is rejected, and RTX validation
can proceed when NVIDIA hardware is available. New model adapters can be qualified
earlier when needed by speculation or serving. Lossy quality failure is never a
reason to block unrelated FP16 work forever or weaken its own acceptance gate.
32K resource execution is already validated on Apple; long-context semantic quality
and the memory/performance of larger models remain model-specific, separate gates.

## Completed checkpoints

- Shared checksummed artifact reader.
- Full Qwen2 MLX model path and cached decode.
- Backend-neutral Python request scheduler and KV reservations.
- Public backend selection and benchmark/correctness integration.
- Official Qwen2.5-0.5B parity: four prompts, 32 greedy tokens each.
- Physical 16-token K/V pages and dynamic block tables.
- Batched one-token decode for all runnable requests.
- Online-softmax tiled reference and MLX fused GQA production attention.
- Chunked prefill with decode-priority scheduling.
- Execution/resource validation at 2K, 4K, 8K, 16K, and 32K.
- Controlled attention A/B and Metal GPU capture tooling.
- Custom Metal residual/RMSNorm, RoPE/page-write, SwiGLU, and paged GQA decode kernels.
- Baseline/fused/full kernel ablations and end-to-end 128/1K concurrency traces.
- A 32K gate that includes a real custom paged-decode iteration.
- Version-2 model-family metadata with backward-compatible version-1 loading.
- A shared model contract and MLX adapter factory for Qwen2 and text-only Gemma 3.
- Official Gemma 3 1B export, Transformers logit parity, continuous-batch smoke execution, and a
  32K prompt-plus-decode resource gate.
- Opt-in MLX rowwise decode projections with retained batched attention: exact same-artifact
  token/logit consistency, boundary/32K gates, preserved defaults, and measured throughput costs.
  See [the numerical-consistency report](mlx-batch-numerics-results.md).

## Quantization acceptance (open); current checkpoint: prefix-cache final qualification

INT8 conversion, version-3 loading, custom Metal execution, and controlled measurement are
implemented. The real-model strict quality gate failed, so INT8 remains experimental; see
[the report](mlx-int8-results.md). This is not permission to skip ahead to 4-bit quantization.
The latest Qwen-only downstream validation is complete but rejected at 21/25 exact
held-out continuations; see [its results](mlx-qwen-validation-results.md). The next
implementation checkpoint is milestone 3 on validated FP16, not another automatic
retuning of the observed regression corpus. It is implemented with all eight validation
gates passed; finish its [saved benchmark/final-audit handoff](prefix-cache-handoff.md)
before advancing to speculative decoding. Milestones 1 and 2 remain quality-gated.

1. Continue INT8 accuracy work. Separate-corpus calibration, explicit mixed-precision policies,
   native-GEMM reconstruction, and regenerated quality/resource/measurement tooling are implemented.
   Both policies meet the 0.999 numerical gate but still fail exact greedy parity (Qwen 19/25,
   Gemma 22/25). See [the hardening report](mlx-int8-hardening-results.md). Freeze new calibration
   experiments before regression; preserve current failed policies and all original FP16 defaults.
   The batch-shape issue is isolated to native FP16 projection reductions. The opt-in rowwise
   projection policy passes same-artifact exact token/logit checks while retaining batched attention
   and leaving the original default unchanged; see [its contract](decode-numerics.md). This resolves
   consistency for that policy, not the separate lossy-quantization greedy-quality failure.
   Block-diagonal second-order INT8 fitting and interaction-aware forward selection are now
   implemented and frozen before two regression corpora. Local reconstruction loss improves,
   but the controlled same-partition quality A/B is mixed and strict greedy gates still fail;
   see [the second-order report](mlx-second-order-results.md). All rowwise batching/boundary/32K
   resource gates pass, independently of quality. Calibration-only cached-decode coverage and
   bounded full-corpus policy repair are now implemented with a new untouched regression corpus,
   whole-file freeze bindings, all-position quality checks, and reconstruction/projection diagnostics;
   see [the cached checkpoint](mlx-cached-calibration-results.md). Both frozen policies still fail
   exact calibration despite fewer cached mismatches. All-position held-out coverage also exposes
   a Gemma cosine failure, including in native and RTN execution; the stronger protocol must not
   be replaced by the older sparse checks. Same-artifact batching/resources remain separate passed
   contracts. Preserve all policies/results and FP16 defaults; use a new calibration-only experiment
   and untouched regression corpus for further method changes.
   The pre-registered scale-aware fitting stage is now implemented and complete: whole-row fallback
   and independent reconstruction checks pass, but Qwen/Gemma exact calibration still fails. Its
   mandatory stop rule leaves the new held-out corpus output-blind and creates no official packed
   export or performance matrix; see [the scale-aware report](mlx-scale-aware-results.md).
   Do not expand this completed search or treat lower local loss as passed model quality.
   A separate two-sweep integer-refinement recipe is now implemented and frozen. Both families
   match all fresh calibration tokens/decisions; Qwen passes the numerical gate, while Gemma fails
   it at one position. The combined run stops before export/held-out/timing; see
   [the refined report](mlx-refined-results.md). A separate Qwen-only downstream
   checkpoint for its eligible sealed policy is now executed and rejected on held-out greedy parity.
   Keep Gemma experimental/FP16, and do not infer its
   quality from Qwen acceptance or relax any gate. Local reconstruction improvements remain distinct
   from held-out model quality and runtime performance.
   The downstream driver registered the sealed Qwen policy and then-untouched refined
   regression corpus before export, checks all 800 cached positions per execution mode,
   and stopped before resources/timing on native-contract quality failure. Direct Metal
   is diagnostic; no family or execution-mode acceptance is inferred from another.
2. Add groupwise W4A16 only after INT8 correctness and memory/performance gates pass.
3. Finish the controlled prefix-cache matrix and independent final qualification.
4. Add generic speculative decoding, then a model-specific MTP path when supported.
5. Build the multi-user streaming service and resource-aware routing layer.
6. Port the validated MLX features to CUDA and add multi-GPU only after single-device parity.

## CUDA validation and parity

The repository is organized so these tasks can be completed and measured independently on the RTX
host.

1. Run and extend the CUDA numerical tests on-device, including block boundaries and non-multiple widths.
2. Validate cuBLASLt row-major layouts and Qwen greedy parity on the exported 0.5B checkpoint.
3. Validate the full-prompt GEMM and causal paged-prefill path, then profile whether tiling is warranted.
4. Verify NVTX ranges appear correctly in an Nsight Systems capture.
5. Run the Transformers FP16 comparison runner on the identical prompt/output corpus.
6. Execute the benchmark matrix and commit raw JSON plus a written analysis.
7. Run Compute Sanitizer and repeat create/destroy/cache-exhaustion stress tests.

After the Apple contracts stabilize, port paged 32K attention, quantization, prefix caching, and
speculative verification to CUDA. Add NCCL tensor parallelism only after single-GPU parity.
