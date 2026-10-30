# Benchmarks and correctness

Everything here was measured on one Apple M3 Max (128 GB) with macOS 26.5,
Python 3.12 and MLX 0.32.2. CUDA hasn't been run on a GPU yet, so there are no
NVIDIA numbers.

## FP16 throughput

Qwen2.5-0.5B-Instruct, FP16. Every request generates 32 tokens. Prompts are fixed
synthetic token arrays, five warmups, three measured repetitions. "Before" is the
previous engine with its custom kernels on (per-page arrays, stacked into one
array before each attention launch); "after" is the current engine (one K/V arena
per layer and the split-K paged decode kernel).

| Prompt × requests | Generated tok/s, before → after | TTFT p50 (ms) | TPOT p50 (ms) | Peak MLX memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 128 × 1 | 115.4 → 152.3 (1.32×) | 20.5 → 15.3 | 8.2 → 6.3 | 1,131 → 1,146 |
| 128 × 8 | 376.2 → 681.2 (1.81×) | 124.3 → 92.5 | 16.8 → 8.4 | 1,147 → 1,146 |
| 1,024 × 1 | 29.5 → 104.0 (3.53×) | 148.4 → 79.1 | 30.2 → 7.4 | 1,542 → 1,560 |
| 1,024 × 8 | 81.7 → 238.7 (2.92×) | 905 → 434 | 64.6 → 18.9 | 1,631 → 1,695 |

Long requests gain the most: decode used to copy a request's pages into one array
before every attention launch, and now reads them in place. Peak memory includes
the ~942 MiB of weights; the arena grows to the pages actually used.

Raw JSON is in
[benchmarks/results/paged-arena-m3-max-2026-10-30](../benchmarks/results/paged-arena-m3-max-2026-10-30/)
(the earlier runs are in
[fp16-m3-max-2026-09-23](../benchmarks/results/fp16-m3-max-2026-09-23/)). To reproduce:

```bash
PYTHONPATH=python python benchmarks/run_matrix.py \
  --backend mlx --model models/qwen2.5-0.5b.engine \
  --output-dir benchmark-results/paged --prompt-lengths 128 1024 \
  --output-lengths 32 --concurrencies 1 8 --warmups 5 --repetitions 3
PYTHONPATH=python python benchmarks/summarize.py benchmark-results/paged
```

## Long context

A 32,766-token prompt plus two generated tokens (so one real decode step at
~32K) prefilled in 7.0 s on Qwen2.5-0.5B (2,048-token chunks; the previous engine
took about 22 s) and 6.6 s on Gemma 3 1B, and every page was released afterwards.
Gemma's arena peaked at 216 MiB because its sliding-window layers return pages
as the window moves. That's one run each, so it's a check that long context works
and doesn't leak, not a latency benchmark.

## Correctness

Both native model families are compared against Hugging Face Transformers running
the same FP16 weights, using greedy decoding with the repetition penalty off:

- **Qwen2.5-0.5B:** all 128 tokens match across four prompts × 32 tokens.
  First-token logit cosine is ≥ 0.99998, and top-1 agrees everywhere. This holds
  with and without the custom kernels.
- **Gemma 3 1B:** all four first-token logit checks pass, and three of the four
  32-token continuations match exactly. The fourth diverges after 25 tokens at
  an FP16 near-tie between the top two logits.

```bash
instinct-correctness --engine-model models/qwen2.5-0.5b.engine \
  --model Qwen/Qwen2.5-0.5B-Instruct --output benchmark-results/qwen-correctness.json
```

The unit tests build tiny random models and check them against an independent
reference (a NumPy implementation for Qwen2, Transformers for Gemma 3) at page
boundaries (15/16/17/31/32/33 tokens), during chunked prefill, with mixed-length
batches, and with the custom kernels on and off. Other
tests cover cancellation mid-prefill and check that pages are always reclaimed.

## Tips for new measurements

- Don't run GPU tests while benchmarking. `benchmarks/gpu_guard.py` stops two of
  the project's own drivers from overlapping.
- Keep Mac and NVIDIA results in separate tables.
- Metal captures (`MTL_CAPTURE_ENABLED=1 instinct-bench ... --capture out.gputrace`)
  slow everything down, so don't mix captured timings with normal runs.
- On CUDA, `nsys profile --trace=cuda,nvtx,cublas` picks up the NVTX ranges
  around prefill and decode.
