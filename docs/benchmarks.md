# Benchmarks and correctness

Everything here was measured on one Apple M3 Max (128 GB) with macOS 26.5,
Python 3.12 and MLX 0.32.2. CUDA hasn't been run on a GPU yet, so there are no
NVIDIA numbers.

## FP16 throughput: custom Metal kernels vs. plain MLX

Qwen2.5-0.5B-Instruct, FP16. Every request generates 32 tokens. Prompts are fixed
synthetic token arrays, five warmups, three measured repetitions. Both modes use
the same model, scheduler and paged KV cache. `baseline` uses composed MLX ops and
gathered/padded decode attention, and `full` turns on the custom kernels listed
below.

| Prompt × requests | Generated tok/s, baseline → full | TTFT p50 (ms) | TPOT p50 (ms) | Peak MLX memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 128 × 1 | 67.4 → 115.4 (1.71×) | 23.5 → 20.5 | 14.4 → 8.2 | 1,101 → 1,131 |
| 128 × 8 | 151.9 → 376.2 (2.48×) | 185.3 → 124.3 | 45.9 → 16.8 | 1,117 → 1,147 |
| 1,024 × 1 | 28.5 → 29.5 (1.04×) | 155.2 → 148.4 | 31.3 → 30.2 | 1,521 → 1,542 |
| 1,024 × 8 | 36.6 → 81.7 (2.23×) | 1,106 → 905 | 175.2 → 64.6 | 1,611 → 1,631 |

The kernels help most with concurrent decode, where they cut TPOT by about 63%.
A single 1,024-token request barely moves. Before each launch the attention kernel
has to stack that request's KV pages into one array (MLX arrays are immutable),
and at that length the copy eats most of the gain. Peak memory includes the
~942 MiB of weights. At 1,024 × 8 the KV cache peaked at 99 MiB in both modes.

Raw JSON for these runs is in
[benchmarks/results/fp16-m3-max-2026-09-23](../benchmarks/results/fp16-m3-max-2026-09-23/).
To reproduce, run the matrix once per mode:

```bash
PYTHONPATH=python python benchmarks/run_matrix.py \
  --backend mlx --mlx-kernel-mode full --model models/qwen2.5-0.5b.engine \
  --output-dir benchmark-results/full --prompt-lengths 128 1024 \
  --output-lengths 32 --concurrencies 1 8 --warmups 5 --repetitions 3
PYTHONPATH=python python benchmarks/summarize.py benchmark-results/full
```

There's also a `fused` mode: the norm, RoPE and SwiGLU fusions without the custom
attention kernel.

### The kernels

All four go through MLX's custom Metal kernel API:

- residual add + RMSNorm, accumulating in FP32
- RoPE fused with the write of K/V into the current cache page (prefill and decode)
- SwiGLU (`silu(gate) * up`)
- paged GQA decode attention: one simdgroup per (request, query head), walking the
  request's block table with an online softmax in FP32

GEMMs and causal prefill attention stay as MLX ops. Gemma uses the RoPE/page-write
and attention kernels but not the Qwen-specific norm and SwiGLU ones.

## Long context

A 32,766-token prompt plus two generated tokens (so one real decode step at
~32K) ran in 64 prefill chunks on 2,048 KV pages (384 MiB), and every page and
reservation was released afterwards. Prefill took about 22 seconds. That's one
run, so it's a check that long context works and doesn't leak, not a latency
number. Gemma 3 1B passes the same check.

```bash
forge-validate-long-context --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/long-context.json
```

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
forge-correctness --engine-model models/qwen2.5-0.5b.engine \
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
- Metal captures (`MTL_CAPTURE_ENABLED=1 forge-bench ... --capture out.gputrace`)
  slow everything down, so don't mix captured timings with normal runs.
- On CUDA, `nsys profile --trace=cuda,nvtx,cublas` picks up the NVTX ranges
  around prefill and decode.
