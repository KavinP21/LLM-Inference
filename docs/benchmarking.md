# Benchmark protocol

Do not publish Mac and RTX measurements in the same comparison. The canonical result environment
is the RTX 3070 Ti under Ubuntu 24.04 on WSL2.

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

When a kernel or cache-layout baseline is added, run it through the same schema and workload files;
do not label the current paged implementation as an optimized result until its CUDA profile exists.

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

For each code optimization, document the hypothesis, relevant timeline or counter, change,
before/after measurement, and tradeoff. The initial prefill attention kernel materializes no score
matrix but rereads paged K/V for each query; profile that traffic before attempting a tiled rewrite.
