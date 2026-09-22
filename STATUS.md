# Implementation status

Last updated: 2026-09-22

This is a development snapshot, not a performance release.

## Validated on Apple M3 Max

- CMake/Ninja `host-debug` configuration and build.
- CTest portable host suite.
- Twelve portable pytest tests plus four Metal-backed tests for model artifacts, scheduling,
  cache accounting, numerical parity, and the public MLX engine lifecycle.
- Python syntax for exporter, correctness runner, Forge benchmark, Transformers baseline, matrix
  runner, and result summarizer.
- Cross-language Python-writer/C++-reader model artifacts, including SHA-256 validation.
- A complete synthetic Qwen2 tensor contract through `forge-inspect-model`.
- Shell syntax for WSL environment and Compute Sanitizer scripts.
- MLX 0.32.2 access to the Apple M3 Max Metal device.
- Safe Python parsing and checksum validation of the shared `.engine` model format.
- Complete Qwen2 FP16 execution in MLX: embedding, RMSNorm, biased Q/K/V, RoPE, GQA causal
  attention, output projection, SwiGLU MLP, final normalization, LM head, and greedy selection.
- Contiguous per-sequence K/V state for prompt prefill and autoregressive decode.
- Backend-neutral Python request scheduling, cancellation, capacity reservation, and metrics.
- Synthetic full-model and cached-decode parity against an independent NumPy reference.
- Official `Qwen/Qwen2.5-0.5B-Instruct` export: 291 tensors and data SHA-256
  `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45`.
- Official-model validation over four fixed prompts: identical first 32 raw-greedy tokens,
  identical first-token argmax, and logit cosine similarity from `0.9999914` to `0.9999950`.
- A four-request MLX benchmark smoke run with zero leaked KV reservations after completion.

The exact environment, artifact identity, acceptance results, and smoke-run caveats are recorded in
[the MLX foundation validation report](docs/mlx-foundation-results.md).

## MLX foundation limitations

- Attention materializes its score matrix and is not safe or efficient for 32K prompts.
- K/V tensors are contiguous MLX arrays; the 16-token block pool currently accounts for capacity
  but does not yet provide physical paged storage.
- Active decode requests follow continuous-batching lifecycle semantics but execute one by one;
  they are not yet combined into batched MLX tensors.
- No custom Metal kernels or competitive-performance claims exist yet.
- The tracked benchmark runner works on MLX, but the one-trial smoke result is validation evidence,
  not a publishable benchmark.

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

Implement the 32K-safe Apple execution milestone in this order:

1. Replace contiguous cache storage with physical 16-token MLX/Metal pages.
2. Add batched one-token decode across all runnable requests.
3. Implement online-softmax tiled prefill attention without an `O(context^2)` score allocation.
4. Add chunked prefill and dynamic block tables up to the model's 32,768-token limit.
5. Validate at block boundaries and 2K/4K/8K/16K/32K lengths.
6. Profile MLX operations before choosing the first custom Metal fusion.

The CUDA implementation remains preserved for later RTX validation. No CUDA performance or
correctness claim should be published until the WSL2 runbook succeeds on the RTX 3070 Ti.
