# Implementation status

Last updated: 2026-09-23

This is a development snapshot, not a performance release.

## Validated on Apple M3 Max

- CMake/Ninja `host-debug` configuration and build.
- CTest portable host suite.
- Twenty-five portable pytest tests plus nineteen Metal-backed tests for model artifacts, export,
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
- Quantization, prefix caching, speculative decoding, serving, and distributed routing remain future
  milestones.

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

## Next executable checkpoint

Implement measured weight-only quantization without weakening either family gate:

1. Add versioned quantized tensor metadata and preserve the existing FP16 artifact reader.
2. Implement per-output-channel INT8 weight packing and an MLX dequantized-matmul baseline.
3. Add a fused Metal weight-only INT8 linear path for the shapes that profiler evidence supports.
4. Compare model bytes, peak memory, TTFT, TPOT, and tokens/s against FP16 for both model families.
5. Require per-layer/logit comparison plus greedy regression; record any near-tie separately.
6. Only after INT8 passes, add groupwise W4A16 with the same controlled A/B and correctness gates.

The CUDA implementation remains preserved for later RTX validation. No CUDA performance or
correctness claim should be published until the WSL2 runbook succeeds on the RTX 3070 Ti.
