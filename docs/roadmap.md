# Ordered implementation roadmap

## Current checkpoint: MLX foundation complete

- Shared checksummed artifact reader.
- Full Qwen2 MLX model path and cached decode.
- Backend-neutral Python request scheduler and KV reservations.
- Public backend selection and benchmark/correctness integration.
- Official Qwen2.5-0.5B parity: four prompts, 32 greedy tokens each.

## Next checkpoint: MLX 32K execution

1. Physical paged K/V arrays and device block tables.
2. Batched decode tensors for all runnable sequences.
3. Online-softmax tiled attention and bounded workspace.
4. Chunked prefill with decode-priority scheduling.
5. Correctness at 2K, 4K, 8K, 16K, and 32K.
6. MLX profiler baseline and first evidence-backed Metal fusion.

## Subsequent Apple checkpoints

1. Custom Metal RMSNorm/residual, RoPE/cache-write, SwiGLU, and attention kernels.
2. Shape-specialized Qwen execution and command/pipeline caching.
3. Gemma 3 1B model adapter.
4. Weight-only INT8 followed by groupwise W4A16.
5. Prefix caching and copy-on-write shared blocks.
6. Generic speculative decoding, then Gemma 4 E2B/MTP.
7. Multi-user streaming service and resource-aware routing.

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
