# Benchmark protocol

Do not publish Mac and RTX measurements in the same comparison. CUDA's canonical result environment
is the RTX 3070 Ti under Ubuntu 24.04 on WSL2; MLX results form a separate M-series matrix.

For every configuration, run five warmups and three recorded repetitions. Preserve the generated
JSON, Git revision, exported model checksum, build type, driver, CUDA version, and compiler flags.

The initial matrix is:

| Dimension | Values |
| --- | --- |
| Prompt tokens | 32, 128, 512, 1024 |
| Output tokens | 32, 128 |
| Concurrency | 1, 2, 4, 8, 16 when memory permits |
| Runtime | Transformers FP16 and Forge FP16 paged |

Run the complete deterministic grid with:

```bash
PYTHONPATH=python python benchmarks/run_matrix.py \
  --model models/qwen2.5-0.5b.engine --with-reference
PYTHONPATH=python python benchmarks/summarize.py benchmark-results
```

For the MLX kernel ablation, use separate output directories so no mode overwrites another:

```bash
for mode in baseline fused full; do
  PYTHONPATH=python python benchmarks/run_matrix.py \
    --backend mlx --mlx-kernel-mode "$mode" \
    --model models/qwen2.5-0.5b.engine \
    --output-dir "benchmark-results/mlx-metal-$mode" \
    --prompt-lengths 128 1024 --output-lengths 32 \
    --concurrencies 1 8 --warmups 5 --repetitions 3
done
```

`baseline` uses composed MLX operations and gathered/padded decode attention. `fused` adds custom
residual/RMSNorm, RoPE/page-write, and SwiGLU. `full` also adds block-table-driven paged decode
attention. Keep these labels and workload files identical for controlled comparisons.

For MLX, first run `forge-validate-long-context --output-tokens 2` at 2K/4K/8K/16K/32K so the gate
includes a true decode step. Then use
`forge-profile-attention` for controlled attention A/B measurements and an optional Xcode Metal
capture. Treat the long-context ladder as execution and resource-integrity evidence, not a
throughput benchmark: it deliberately performs one run per length with no warmup statistics.

Measure queue-inclusive TTFT, end-to-end latency, TPOT, and their p50/p95/p99 distributions.
Report generated tokens/s separately from prompt tokens/s. Sample GPU utilization, power, and
memory with `nvidia-smi`; record KV occupancy and fragmentation from `Engine.stats()`.

Profile selected 128- and 1024-token runs at concurrency 1 and 8:

```bash
nsys profile --trace=cuda,nvtx,cublas --sample=none -o profiles/forge-c1 \
  python -m forge_llm.benchmark ...
ncu --set full --kernel-name regex:paged_attention --export profiles/paged-attention \
  python -m forge_llm.benchmark ...
```

On MLX, use the benchmark's capture option. Capture instrumentation is intrusive, so trace timings
must never be mixed with the uncaptured benchmark table:

```bash
MTL_CAPTURE_ENABLED=1 forge-bench --backend mlx --mlx-kernel-mode full \
  --model models/qwen2.5-0.5b.engine --prompts path/to/workload.json \
  --output benchmark-results/trace-metadata.json --output-length 8 \
  --warmups 1 --repetitions 1 --capture profiles/forge.gputrace
```

For each code optimization, document the hypothesis, relevant timeline or counter, change,
before/after measurement, and tradeoff. The custom MLX attention kernel removes per-request gathers
and batch padding but still stacks immutable layer-pages before launch; measure that copy separately
before claiming zero-copy paging.
