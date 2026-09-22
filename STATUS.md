# Implementation status

Last updated: 2026-09-11

This is a development snapshot, not a benchmarked `v0.1.0` release.

## Validated on Apple M3 Max

- CMake/Ninja `host-debug` configuration and build.
- CTest portable host suite.
- Five pytest tests for deterministic model artifacts, tied-weight aliasing, benchmark statistics,
  cache peak sampling, and model provenance.
- Python syntax for exporter, correctness runner, Forge benchmark, Transformers baseline, matrix
  runner, and result summarizer.
- Cross-language Python-writer/C++-reader model artifacts, including SHA-256 validation.
- A complete synthetic Qwen2 tensor contract through `forge-inspect-model`.
- Shell syntax for WSL environment and Compute Sanitizer scripts.

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

On the WSL2 machine, follow [docs/setup-wsl.md](docs/setup-wsl.md). Stop at the first failure and
preserve its full compiler, test, or sanitizer output. The order is deliberate:

1. Configure and compile the CUDA preset.
2. Run CUDA unit tests.
3. Run Compute Sanitizer.
4. Export and inspect the real checkpoint.
5. Run the correctness corpus.
6. Only then capture profiles and benchmarks.

