# Remaining v0.1 gates

The repository is organized so these tasks can be completed and measured independently on the RTX
host.

1. Run and extend the CUDA numerical tests on-device, including block boundaries and non-multiple widths.
2. Validate cuBLASLt row-major layouts and Qwen greedy parity on the exported 0.5B checkpoint.
3. Validate the full-prompt GEMM and causal paged-prefill path, then profile whether tiling is warranted.
4. Verify NVTX ranges appear correctly in an Nsight Systems capture.
5. Run the Transformers FP16 comparison runner on the identical prompt/output corpus.
6. Execute the benchmark matrix and commit raw JSON plus a written analysis.
7. Run Compute Sanitizer and repeat create/destroy/cache-exhaustion stress tests.

After v0.1, implement weight-only INT8 before adding a server or agent harness.
