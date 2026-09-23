# Ordered implementation roadmap

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

## Current checkpoint: weight-only quantization

1. Add weight-only INT8 with per-output-channel scales and controlled FP16 A/B evidence.
2. Add groupwise W4A16 only after INT8 correctness and memory/performance gates pass.
3. Add prefix caching and copy-on-write shared blocks.
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
