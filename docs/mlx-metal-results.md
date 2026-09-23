# MLX custom Metal optimization — 2026-09-23

This report is the acceptance record for Forge's custom-Metal checkpoint. The measurements were
taken from an uncommitted working tree based on Git commit
`0e6872272dac1f62cf48c2472514ac2d42689066`. Raw benchmark JSON and `.gputrace` bundles are ignored
local artifacts. They are not silently promoted into checked-in claims.

## Scope and implementation

The optimized Qwen2 path adds four shape-specialized kernels through MLX's custom Metal interface:

- residual add plus FP32-accumulated RMSNorm;
- RoPE plus physical K/V layer-page writes for both prefill and batched decode;
- SiLU-times-up SwiGLU activation;
- online-softmax GQA decode attention driven by per-request `int32` block tables.

GEMMs and causal prefill attention remain MLX operations. The attention kernel uses one simdgroup
per `(request, query_head)`, maps query heads to grouped K/V heads, follows logical-to-packed page
indices, and accumulates softmax state and values in FP32. MLX specializes the kernel template for
the dtype, head counts, head width, and 16-token page size; benchmark warmups occur before timing.

The page layout is now independently replaceable per layer. Before custom attention, live physical
layer-pages are stacked into one MLX tensor and block tables are remapped to that packed tensor.
This is not zero-copy paging: MLX arrays are immutable and the Python custom-kernel API does not
provide a safe pointer table over independent allocations. It does avoid concatenating each
request's entire logical K/V history and avoids right-padding every request to the longest member of
the batch.

## Environment and correctness

| Item | Value |
| --- | --- |
| Host | Apple M3 Max (`applegpu_g15s`) |
| Operating system | macOS 26.5 arm64 |
| Python | 3.12.10 |
| MLX | 0.32.2 |
| Model file | `Qwen/Qwen2.5-0.5B-Instruct`, FP16 |
| Model data SHA-256 | `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` |

The official four-prompt Transformers gate produced 128/128 identical greedy tokens. First-token
logit cosine similarity ranged from `0.9999808` to `0.9999977`; all top-1 logits agreed. The largest
observed absolute logit error was `0.103515625`.

Twenty portable tests and sixteen Metal-backed tests pass. The Metal suite includes numerical
comparison with the fallback path, mixed 300/400-token batched decode, 15/16/17/31/32/33 page
boundaries, partial-prefill cancellation, page reclamation, and cache reuse.

The maximum-context gate used a 32,766-token prompt plus two generated tokens so it executed one
actual custom paged-decode iteration at a 32,767-token K/V length. It used exactly 2,048 physical
pages and 402,653,184 K/V bytes. TTFT was one non-statistical observation of 20,973 ms; end-to-end
time was 21,858 ms. All pages and reservations returned to zero. This is execution and leak evidence,
not a latency benchmark.

## Controlled end-to-end A/B

Each cell below uses five warmups and three measured repetitions, 32 forced output tokens, and the
same deterministic token-ID workload. `baseline` uses MLX compositions and gathered/padded decode
attention. `fused` enables the three transformer fusions but retains fallback attention. `full` also
enables custom paged decode attention.

| Prompt | Concurrency | Mode | Generated tok/s | TTFT p50 | TPOT p50 | E2E p50 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | baseline | 73.88 | 21.94 ms | 13.11 ms | 428.39 ms |
| 128 | 1 | fused | 118.15 | 19.61 ms | 8.05 ms | 269.10 ms |
| 128 | 1 | full | 123.46 | 19.88 ms | 7.67 ms | 258.03 ms |
| 128 | 8 | baseline | 159.44 | 173.08 ms | 43.80 ms | 1,537.45 ms |
| 128 | 8 | fused | 202.49 | 142.39 ms | 34.45 ms | 1,211.81 ms |
| 128 | 8 | full | 386.05 | 118.21 ms | 15.19 ms | 579.18 ms |
| 1,024 | 1 | baseline | 31.31 | 144.97 ms | 28.28 ms | 1,021.65 ms |
| 1,024 | 1 | fused | 36.21 | 146.74 ms | 23.93 ms | 883.39 ms |
| 1,024 | 1 | full | 31.26 | 133.84 ms | 28.66 ms | 1,022.40 ms |
| 1,024 | 8 | baseline | 39.41 | 1,016.65 ms | 162.56 ms | 6,085.35 ms |
| 1,024 | 8 | fused | 42.72 | 976.89 ms | 150.63 ms | 5,644.54 ms |
| 1,024 | 8 | full | 86.53 | 869.33 ms | 60.34 ms | 2,738.83 ms |

Relative to baseline, full mode improved generated throughput by 67.1% at 128/c1, 142.1% at
128/c8, and 119.5% at 1,024/c8. Median TPOT fell by 41.5%, 65.3%, and 62.9%, respectively. At
1,024/c1, full-mode throughput was effectively flat (-0.17%) and TPOT was 1.36% worse. The
intermediate `fused` mode was 15.7% faster there, demonstrating that per-layer page packing and a
single-request, one-simdgroup-per-head attention launch erase the direct-kernel gain at that shape.

The data supports two narrow conclusions. First, fusing bandwidth-bound elementwise chains removes
dispatches and intermediate reads/writes, which dominates short single-request decode. Second,
block-table attention removes per-request gather and longest-sequence padding costs that grow under
batching. It does not establish maximum M-series performance; a persistent or truly contiguous
page pool below the Python MLX layer is still needed to remove page packing.

## Trace evidence and interpretation boundary

Four Xcode-readable end-to-end captures were generated for 128- and 1,024-token prompts at
concurrency 1 and 8. Each trace covers one measured request set with eight generated tokens. Metal
capture instrumentation increased latency by orders of magnitude, so captured timings are excluded
from the table. The traces are for launch ordering, synchronization, kernel occupancy, and memory
inspection only.

The optimization record is therefore:

1. **Hypothesis:** composed residual/RMSNorm, RoPE/write, and SwiGLU create avoidable dispatches and
   intermediate unified-memory traffic. **Change:** one shape-specialized kernel per chain.
   **Result:** fused mode improves the 128/c1 TPOT from 13.11 to 8.05 ms.
2. **Hypothesis:** gathered, longest-context-padded decode wastes work as concurrency grows.
   **Change:** direct block-table traversal with online FP32 softmax. **Result:** moving from fused to
   full mode improves 1,024/c8 throughput from 42.72 to 86.53 tok/s.
3. **Tradeoff:** MLX-level page packing is visible at low concurrency. **Result:** full mode does not
   beat baseline at 1,024/c1 even though fused mode does.

No CUDA result is inferred from this data, and no comparison is made against llama.cpp, MLX-LM, or
vendor engines without an identical controlled workload.
