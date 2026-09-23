# MLX paged 32K validation — 2026-09-22

> Historical checkpoint: the custom-kernel milestone described in
> [the MLX Metal optimization report](mlx-metal-results.md) supersedes the performance limitations
> recorded here. The measurements below remain the acceptance evidence for the preceding paged-MLX
> checkpoint.

This report records acceptance evidence for the second Apple checkpoint. The working tree was based
on Git commit `0e6872272dac1f62cf48c2472514ac2d42689066`; the checkpoint changes were uncommitted when
these measurements were captured. Raw JSON and the GPU trace are ignored local artifacts so model
data and machine-specific traces are not accidentally committed.

## Environment and artifact

| Item | Value |
| --- | --- |
| Host | Apple M3 Max (`applegpu_g15s`) |
| Operating system | macOS 26.5 arm64 |
| Python | 3.12.10 |
| MLX | 0.32.2 |
| Model tensor-data bytes | 988,065,536 |
| Model tensor-data SHA-256 | `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` |

## Correctness

The paged/chunked runtime repeated the official four-prompt, 32-token Transformers gate. All 128
greedy tokens matched. First-token logit cosine similarity ranged from `0.9999880` to `0.9999951`,
and all top-1 logits agreed. The test suite reports 20 portable tests and 15 Metal-backed tests,
including 15/16/17/31/32/33 boundaries, two-sequence batched decode, partial-prefill cancellation,
page reuse, and post-request reclamation.

An additional 1,024-token official-model differential compared 256-token paged/fused chunks with
the original materialized attention path. Cosine similarity was `0.9999916`, maximum absolute logit
error was `0.025390625`, top-1 agreed, and exactly 64 physical pages were present.

## Long-context execution gate

Each row is one deterministic request with one generated token. Prompt work used 512-token chunks.
TTFT is included only to make regressions visible; these single observations are not benchmark
percentiles.

| Context tokens | Chunks | Physical pages | K/V bytes | TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4 | 128 | 25,165,824 | 434.0 ms |
| 4,096 | 8 | 256 | 50,331,648 | 1,080.5 ms |
| 8,192 | 16 | 512 | 100,663,296 | 2,726.3 ms |
| 16,384 | 32 | 1,024 | 201,326,592 | 8,053.3 ms |
| 32,768 | 64 | 2,048 | 402,653,184 | 29,071.6 ms |

Every case allocated exactly `ceil(context / 16)` pages and returned allocated blocks,
reservations, and materialized MLX pages to zero. At 32K, MLX reported 1,912,412,052 peak active
bytes versus a 988,065,664-byte weight baseline. This total includes K/V, activations, allocator
cache, and fused-kernel workspace; it must not be presented as workspace alone.

## Attention optimization evidence

The controlled A/B used 512 query tokens, two warmups, five measured repetitions, and median wall
time after materialization. It compared the original FP32 score-matrix path, exact online-softmax
tiling, and MLX fused grouped-query attention.

| Key tokens | Materialized | Online tiled | Fused GQA | Fused speedup vs materialized |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.957 ms | 1.388 ms | 0.437 ms | 2.19x |
| 2,048 | 2.747 ms | 4.578 ms | 1.956 ms | 1.40x |
| 8,192 | 23.592 ms | 20.155 ms | 6.557 ms | 3.60x |

Fused output cosine similarity remained at least `0.99999989`; maximum absolute error was at most
`0.0002442`. These measurements justify selecting fused MLX GQA for production shapes while
retaining online tiling as the transparent reference/fallback. A 40 KiB Xcode-readable Metal trace
was also captured for the fused kernel. Per-operation allocator peak deltas were inconsistent under
MLX buffer reuse, so this report does not make a per-kernel memory claim from that counter.

## End-to-end directional smoke result

On the same four-request, eight-output-token smoke workload, the old contiguous/sequential snapshot
measured 81.06 generated tokens/s and 43.29 ms median TPOT. The paged/batched snapshot measured
110.93 generated tokens/s and 26.17 ms median TPOT: +36.8% throughput and -39.5% median TPOT.
Queue-inclusive median TTFT increased from 60.67 to 81.02 ms. This was one warmup and one measured
trial, so it is directional validation—not a publishable performance result or an isolated batching
ablation.

## Remaining performance boundary

Physical pages eliminate whole-history K/V copies, but attention still gathers page slices before
the fused MLX call. Decode also pads each batch to its longest context. The next milestone must use
end-to-end traces to fuse residual/RMSNorm and RoPE/page-write, then implement direct block-table
Metal attention. No claim about maximum M-series performance is justified before those changes and
the full warmup/repetition matrix.
