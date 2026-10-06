# MLX/Metal INT8 checkpoint — implementation delivered, strict quality gate failed

Measured on 2026-10-01 on Apple M3 Max, MLX 0.32.2. This is an experimental development checkpoint,
not a lossless quantization or general speedup claim. Default FP16 behavior is unchanged. CUDA INT8
is not implemented or measured.

## What was built

- Portable deterministic per-output-channel signed INT8 conversion with FP32 scales.
- Checksummed version-3 artifacts accepted by both Python and C++, with version-1/2 compatibility.
- Qwen2 and Gemma 3 W8A16 execution, preserving FP16 embeddings, tied head, biases, norms, and K/V.
- A custom small-batch Metal linear kernel with shared weight loads, FP32 accumulation, vectorized
  aligned loads, and scalar odd-width tails.
- A one-pass Metal reconstruction kernel for prefill; it removes full-sized FP32 intermediates
  while retaining MLX's matrix multiplication. The simple composed dequantization baseline remains
  selectable.
- Numerical, generation, cache-ownership, and measured-memory checks plus raw result tooling.

## Artifact storage

| Model | FP16 file bytes | INT8 file bytes | Reduction | Quantized projections |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B Instruct | 988,084,480 | 631,501,824 | 36.1% | 168 |
| Gemma 3 1B IT | 1,999,794,688 | 1,303,899,648 | 34.8% | 182 |

INT8 data checksums are `b229ceaab153c193a449e01cbbbf000419f34fc30575fee1ce01f28c07620740`
(Qwen) and `31d5cea203e94a668483a34668c37666ce07b7aadb64695953fed1fdb01d242f`
(Gemma). The source checksums and unique resident-weight byte counts are recorded in every raw
benchmark. File-size reduction is not a claim about peak inference memory.

## Controlled end-to-end comparison

Each configuration used five full 32-token warmups followed by three measured 32-token trials,
synthetic prompts of token ID 42, disabled EOS stopping, identical scheduler/cache/kernel settings,
and a separate process for each runtime. Implementation order rotates by configuration. No other
project GPU test was run concurrently with these final matrices. Warmups now use the complete
output workload; these numbers should not be pooled with older four-token-warmup reports.

Latency percentiles are over three requests for concurrency 1 and 24 requests for concurrency 8;
they are small-sample descriptions, not production tail-latency estimates. Memory is MLX peak
**active allocator** memory during measured repetitions, not RSS or all unified memory. Apple power
and utilization were not collected and remain null. Matrix workloads are throughput tests, not
quality tests. Raw JSON includes observations, checksums, Git revision/dirty state, source and
workload fingerprints, platform, and native build information.

### Qwen2.5 0.5B

| Prompt | Concurrency | Runtime/raw result | Output tokens/s | TTFT p50 ms | TPOT p50 ms | Peak active GiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/qwen/fp16-p128-c1.json) | 122.9 | 19.7 | 7.71 | 1.105 |
| 128 | 1 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-dequantize-p128-c1.json) | 40.1 | 36.8 | 24.49 | 1.559 |
| 128 | 1 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-metal-p128-c1.json) | 118.2 | 33.2 | 7.59 | 0.958 |
| 128 | 8 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/qwen/fp16-p128-c8.json) | 410.0 | 119.2 | 15.23 | 1.120 |
| 128 | 8 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-dequantize-p128-c8.json) | 184.0 | 273.5 | 33.04 | 1.559 |
| 128 | 8 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-metal-p128-c8.json) | 342.0 | 190.5 | 17.04 | 0.974 |
| 1024 | 1 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/qwen/fp16-p1024-c1.json) | 31.0 | 136.8 | 28.89 | 1.506 |
| 1024 | 1 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-dequantize-p1024-c1.json) | 21.0 | 169.9 | 43.60 | 1.527 |
| 1024 | 1 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-metal-p1024-c1.json) | 30.5 | 162.5 | 28.43 | 1.283 |
| 1024 | 8 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/qwen/fp16-p1024-c8.json) | 86.0 | 839.8 | 62.02 | 1.593 |
| 1024 | 8 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-dequantize-p1024-c8.json) | 63.9 | 1113.6 | 82.69 | 1.602 |
| 1024 | 8 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-metal-p1024-c8.json) | 78.6 | 952.5 | 67.50 | 1.371 |

### Gemma 3 1B

| Prompt | Concurrency | Runtime/raw result | Output tokens/s | TTFT p50 ms | TPOT p50 ms | Peak active GiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/gemma/fp16-p128-c1.json) | 53.1 | 46.9 | 17.90 | 2.167 |
| 128 | 1 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-dequantize-p128-c1.json) | 19.0 | 77.5 | 51.61 | 1.935 |
| 128 | 1 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-metal-p128-c1.json) | 55.4 | 67.8 | 16.37 | 1.692 |
| 128 | 8 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/gemma/fp16-p128-c8.json) | 199.6 | 285.1 | 29.74 | 2.199 |
| 128 | 8 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-dequantize-p128-c8.json) | 89.6 | 561.7 | 67.73 | 1.966 |
| 128 | 8 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-metal-p128-c8.json) | 180.2 | 383.0 | 31.16 | 1.724 |
| 1024 | 1 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/gemma/fp16-p1024-c1.json) | 17.9 | 281.4 | 48.36 | 2.953 |
| 1024 | 1 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-dequantize-p1024-c1.json) | 11.0 | 362.0 | 81.94 | 2.495 |
| 1024 | 1 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-metal-p1024-c1.json) | 17.6 | 334.0 | 47.87 | 2.312 |
| 1024 | 8 | [FP16](../benchmarks/results/int8-m3-max-2026-10-01/gemma/fp16-p1024-c8.json) | 53.8 | 1624.8 | 89.76 | 3.140 |
| 1024 | 8 | [Naive INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-dequantize-p1024-c8.json) | 37.3 | 2199.1 | 131.51 | 2.682 |
| 1024 | 8 | [Custom INT8](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-metal-p1024-c8.json) | 46.2 | 1970.3 | 102.85 | 2.500 |

Custom INT8 cuts allocator peaks by roughly 13–15% for Qwen and 20–22% for Gemma in this matrix.
It is much faster than naive reconstruction for short-context decode, but FP16 remains faster in
most end-to-end cases, particularly concurrent ones. Larger batches favor MLX's native FP16 GEMM;
custom GEMV-style arithmetic and prefill reconstruction still cost time. The small single-request
speed differences should not be generalized. Lower weight storage is the demonstrated benefit,
not universally higher throughput.

## Optimization evidence

The initial decode kernel issued separate weight reads for each activation row. The revised kernel
loads a channel's four-element vector once, reuses it across up to 16 rows, and accumulates each row
in FP32. The initial and revised linear microbenchmarks remain saved as diagnostic development
evidence; wall times include Python submission and synchronization, not just GPU execution.

The first custom path still used composed MLX casts/multiplication during prefill. At Qwen
128/concurrency 1, its measured peak active memory was
[1.339 GiB](../benchmarks/results/int8-m3-max-2026-10-01/before-reconstruction/qwen/int8-metal-p128-c1.json),
despite the smaller stored model. Hypothesis: full-sized FP32 reconstruction intermediates inflate
the transient footprint. Change: one-pass INT8-to-FP16 Metal reconstruction, followed by native
MLX matmul. Result:
[0.958 GiB](../benchmarks/results/int8-m3-max-2026-10-01/qwen/int8-metal-p128-c1.json).
The corresponding Gemma peak fell from
[1.935 GiB](../benchmarks/results/int8-m3-max-2026-10-01/before-reconstruction/gemma/int8-metal-p128-c1.json)
to [1.692 GiB](../benchmarks/results/int8-m3-max-2026-10-01/gemma/int8-metal-p128-c1.json).
This is an allocator observation consistent with the removed intermediates, not a hardware-counter
claim about HBM saturation. FP16 reconstruction remains temporary, not a second permanently loaded
model, and still contributes to peak memory.

Full Xcode counter/occupancy analysis is not available on this host: it has Command Line Tools,
but `xctrace` is absent. Captures are retained for inspection on a full Xcode installation; do not
invent bandwidth, occupancy, utilization, or GPU-only timings from synchronized wall times.

Final layer-0 linear measurements cover all seven projections at 1/8/16/128 rows, five warmups and
30 samples per operation:
[Qwen raw profile](../benchmarks/results/int8-m3-max-2026-10-01/qwen-linear-final.json) and
[Gemma raw profile](../benchmarks/results/int8-m3-max-2026-10-01/gemma-linear-final.json).
For eight-row down-projection, Qwen's incremental allocator peaks are 17,432,576 bytes for composed
dequantization, 8,716,288 for fused reconstruction, and 28,672 for direct INT8 linear. Gemma's are
31,850,496, 15,958,016, and 32,768 bytes. The direct kernel removes the dense reconstruction buffer;
the reconstruction kernel is selected for memory, not guaranteed lower microbenchmark latency.

Ignored local captures `profiles/int8/qwen-linear-decode.gputrace` and
`profiles/int8/gemma-linear-decode.gputrace` cover eight-row down-projection with resident FP16,
composed reconstruction, fused reconstruction, and direct INT8. Their bundles contain both custom
kernel names. Earlier `*-linear.gputrace` captures were preliminary 128-row native/reconstruction
captures and are not used as custom-decode evidence. All captures are excluded from timing tables.

## Correctness and acceptance boundary

The strict quality report compares the identical original Forge FP16 weights, not a newly claimed
Transformers certification. Every projection is checked on seeded activations. The 25 fixed
prompts generate 32 raw greedy tokens; teacher-forced logits are compared at positions 0/1/7/15/31,
and first-divergence margins are saved separately. Source tensor identity is verified. No near-tie
exception is treated as a passed exact-greedy gate.

The final regression shows all projection cosines above 0.9998, but full-model minima of 0.99167
for Qwen and 0.99636 for Gemma; exact output matches are 15/25 and 17/25. These fail the unchanged
0.999 logit and all-prompts exact-greedy gates. The final regenerated quality reports are
[Qwen](../benchmarks/results/int8-m3-max-2026-10-01/qwen-quality.json) and
[Gemma](../benchmarks/results/int8-m3-max-2026-10-01/gemma-quality.json).

| Metric | Qwen | Gemma |
| --- | ---: | ---: |
| Minimum projection cosine | 0.999894 | 0.999881 |
| Minimum teacher-forced logit cosine | 0.991668 | 0.996365 |
| Teacher-forced top-1 agreement (125 positions) | 97.6% | 96.8% |
| Exact 32-token output cases | 15/25 | 17/25 |
| Metal/dequantized exact output cases | 23/25 | 25/25 |
| Minimum Metal/dequantized logit cosine | 0.999994 | 0.999996 |

Kernel rounding/reduction order also flips two Qwen near-tie decisions versus the dequantized
baseline. This is reported as a failed kernel exact-greedy gate, not hidden in quantization error.
All three runtimes reclaim their cache pages and reservations in every quality case.

### 32K execution and resource gate

Both INT8 artifacts execute a 32,766-token prompt plus two output tokens, with 64 prefill chunks
and one actual custom paged-decode iteration. Each reaches exactly 2,048 physical 16-token pages
and releases every page and reservation afterward:

| Model/raw result | Peak physical K/V bytes | Pages | Reclaimed |
| --- | ---: | ---: | --- |
| [Qwen](../benchmarks/results/int8-m3-max-2026-10-01/qwen-32k.json) | 402,653,184 | 2,048 | yes |
| [Gemma](../benchmarks/results/int8-m3-max-2026-10-01/gemma-32k.json) | 872,415,232 | 2,048 | yes |

These single-run checks certify execution/resource integrity, not 32K quality or a statistically
rigorous throughput/latency advantage. The cache remains FP16, so quantization does not reduce the
K/V byte counts.

Per-operator reconstruction is byte-exact versus the independent NumPy contract in tested shapes;
linear outputs meet FP16 tolerances. Quantized full models match explicitly FP16-reconstructed
tiny-model artifacts, including batched decode, 15/16/17-token boundaries, cancellation, and cache
reclamation. Closed engines release their owned resident weights. These implementation checks do
not certify the quantized models' task quality.

Host C++ tests and local UndefinedBehaviorSanitizer pass. AddressSanitizer could not run: its
runtime recursively stalls during shadow-memory initialization before `main` on this macOS/toolchain.
A one-second native stack sample confirmed the startup stall; both stalled processes were stopped.
Linux sanitizer CI has been added, not yet observed to pass. CMake/Ninja's previous temporary tool
installation is gone, so current C++ sources were rebuilt directly with AppleClang; no fresh local
CMake/CTest run is claimed.

The final complete pytest run passes **77 tests**: 50 portable/reference/export tests and 27
Metal-backed tests. Cross-language tests use the freshly rebuilt C++ inspector and include valid
Qwen/Gemma INT8 files plus malformed descriptors/scales/ranges. Both official INT8 artifacts also
pass standalone C++ inspection with the same data checksums as Python.

## Next gate

Keep INT8 opt-in/experimental and FP16 as the quality reference. Investigate sensitive layers and
calibration or explicit mixed precision using a **separate** corpus, retain the current corpus as
held-out regression, and publish the retained-FP16 memory/speed tradeoff. Re-run the same strict
gates; only then consider W4A16. Prefix caching and speculative decoding are not silently included
in this checkpoint.
