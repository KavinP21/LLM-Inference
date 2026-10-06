# Resume benchmark results — 2026-09-23

This report records a fresh Qwen 2.5-0.5B FP16 benchmark on the current engine checkout. The
inference code was at commit `a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`. The benchmark runner had a
local, uncommitted instrumentation change to record MLX active-memory high-water marks; it did not
change inference behavior.

## Setup and method

| Item | Value |
| --- | --- |
| Host | Apple M3 Max, 128 GB unified memory |
| OS / Python / MLX | macOS 26.5 / Python 3.12.10 / MLX 0.32.2 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, FP16 |
| Model tensor-data SHA-256 | `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` |
| Prompt / output | Repeated token ID 42; fixed 32 generated tokens |
| Matrix | 128- and 1,024-token prompts; concurrency 1 and 8 |
| Repetitions | 5 warmups and 3 measured repetitions per configuration |

The prompts are deterministic synthetic token arrays. They make the kernel-mode comparison
repeatable but do not represent a natural-language quality test. `baseline` uses composed MLX
operations and gathered/padded decode attention. `full` adds custom Metal residual/RMSNorm,
RoPE/page-write, SwiGLU, and block-table paged GQA decode kernels. Both modes use the same model,
scheduler, prompt arrays, output length, and physical paged KV cache.

Generated tokens/s is aggregate throughput across all requests in the configuration. Latency values
are request-level medians across recorded repetitions. Active-memory figures are MLX allocator
high-water marks during those repetitions and include model weights; they are not process RSS or
machine-wide memory use. Peak KV bytes are shown separately.

## Controlled baseline versus full Metal path

| Prompt × concurrency | Generated tok/s, baseline → full | Throughput change | TTFT p50, ms | TPOT p50, ms | E2E p50, ms | Peak active MLX memory, MiB baseline → full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 × 1 | 67.4 → 115.4 | +71.3% | 23.5 → 20.5 | 14.4 → 8.2 | 470.0 → 273.8 | 1,101 → 1,131 |
| 128 × 8 | 151.9 → 376.2 | +147.7% | 185.3 → 124.3 | 45.9 → 16.8 | 1,605.1 → 638.6 | 1,117 → 1,147 |
| 1,024 × 1 | 28.5 → 29.5 | +3.7% | 155.2 → 148.4 | 31.3 → 30.2 | 1,125.7 → 1,083.8 | 1,521 → 1,542 |
| 1,024 × 8 | 36.6 → 81.7 | +123.1% | 1,106.3 → 905.1 | 175.2 → 64.6 | 6,574.7 → 2,903.0 | 1,611 → 1,631 |

The strongest result is concurrent decode: full mode delivered 2.48× throughput at 128-token
prompts and 2.23× at 1,024-token prompts, with about 63% lower median TPOT in both cases. At
1,024 tokens and concurrency 1, throughput improved only 3.7%; the paged-attention launch and page
packing provide little benefit for that shape. The active-memory high-water mark rose by about
20 MiB at 1,024-token concurrency 8, while the peak KV allocation stayed at 99 MiB in both modes.

## 32K context and cache reclamation

A separate single-run resource check used a 32,766-token prompt and generated two tokens, which
executes one decode step after prefill. It completed 64 prefill chunks, reached 2,048 physical pages
and 402,653,184 peak KV bytes (384 MiB), and returned every page and reservation after completion.
Peak MLX active memory was 1,905,063,820 bytes, including the 988,065,664-byte weight baseline.
This is execution/resource evidence only; its one-observation TTFT of 21.93 seconds is not a
statistical latency result.

## Resume-ready wording

- Built a multi-backend FP16 inference runtime for Qwen2.5-0.5B with checksummed memory-mapped
  weights, a 16-token paged KV cache, chunked prefill, and continuous batching; validated a 32K-token
  prompt plus decode with 384 MiB of KV storage and complete reclamation of 2,048 pages.
- Added fused Metal transformer kernels and block-table paged GQA decoding; on an Apple M3 Max,
  reached 376 generated tok/s for eight concurrent requests (128-token prompt, 32-token output),
  2.48× the same-engine MLX fallback with 63% lower median TPOT.
- Sustained 81.7 generated tok/s on eight concurrent 1,024-token prompts, 2.23× baseline throughput
  and 63% lower median TPOT; measured the full run at a 1.59 GiB MLX active-memory peak.

The performance numbers characterize the MLX/Metal path on this M3 Max. They do not establish CUDA
performance or a comparison with external inference engines. CUDA still requires validation on the
target NVIDIA host.

## Reproduction and raw data

The preserved raw JSON and synthetic workloads are under
[`benchmarks/results/resume-m3-max-2026-09-23`](../benchmarks/results/resume-m3-max-2026-09-23/).
Run `benchmarks/run_matrix.py` once with `--mlx-kernel-mode full` and once with
`--mlx-kernel-mode baseline`; for either run, use:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run_matrix.py --backend mlx --mlx-kernel-mode full --model models/qwen2.5-0.5b.engine --output-dir benchmark-results/resume-m3-max-2026-09-23-full --prompt-lengths 128 1024 --output-lengths 32 --concurrencies 1 8 --warmups 5 --repetitions 3
```

The benchmark runner records the active-memory baseline, peak, and increment for MLX runs in
`mlx_memory`, and also records peak KV bytes separately.
