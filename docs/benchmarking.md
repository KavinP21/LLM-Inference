# Benchmark protocol

## FP16 prefix-cache A/B

The [prefix-cache protocol](prefix-cache.md) first checks bitwise same-artifact logits,
greedy tokens, COW/reclamation and 32K execution for both official families. Timing is
forbidden if any registered gate fails. Its driver saves isolated-process disabled,
cold and primed runs, with five warmups and three measured trials per configuration.
Priming is outside the measured interval; cold insertion cost remains inside. These
are repeated full-prompt workloads at 128/1,024 tokens and concurrency 1/8, not a claim
about hit rates in an unknown application or universal decode speedup. Raw trials
and requests independently reconstruct throughput, TTFT, TPOT and percentiles; MLX
active/peak allocation counters are distinct from KV reservation accounting.

## General matrix and provenance

Do not publish Mac and RTX measurements in the same comparison. CUDA's canonical result environment
is the RTX 3070 Ti under Ubuntu 24.04 on WSL2; MLX results form a separate M-series matrix.

For every configuration, run five warmups and three recorded repetitions. Preserve the generated
JSON, Git revision, exported model checksum, build type, driver, CUDA version, and compiler flags.
Forge warmups now use the full requested output length (older reports used four output tokens).
New results also fingerprint runtime sources and workload contents, including a dirty-worktree flag.
Compare only runs using the same warmup protocol; do not relabel historical results.

The initial matrix is:

| Dimension | Values |
| --- | --- |
| Prompt tokens | 32, 128, 512, 1024 |
| Output tokens | 32, 128 |
| Concurrency | 1, 2, 4, 8, 16 when memory permits |
| Runtime | Transformers FP16 and Forge FP16 paged |

The opt-in INT8 checkpoint has its own isolated FP16/dequantized/fused comparison runner:
`benchmarks/run_int8_matrix.py`. See [the INT8 contract](quantization.md) and
[the measured results and failed quality gate](mlx-int8-results.md). Smaller weight files do not
guarantee lower peak inference memory or a speedup: both are measured separately.

For a frozen mixed-precision policy, `benchmarks/run_int8_hardening.py` regenerates held-out,
arrival/cancellation, 32K, and four-mode results. `--include-reconstruct` adds the accuracy-oriented
native-GEMM mode to the standalone matrix. These cooperating drivers hold an exclusive inherited
GPU-evidence lease and reject concurrent drivers; this does not block unrelated applications or
direct test/CLI invocations. Never run project GPU tests alongside measurements. See
[the hardening results](mlx-int8-hardening-results.md); numerical improvements do not replace the
still-failing exact-greedy gate.

`benchmarks/run_batch_numerics_checkpoint.py` isolates the separate numerical-batching issue:
fresh independent logits/token references, saturated and staggered arrivals, page/chunk/sliding
boundaries, 32K resources, and controlled `batched`/`rowwise` projection A/B runs. It uses the same
exclusive lease. Correctness logit copies and stage instrumentation are **never** enabled during
the performance matrix. Read [the numerical policy](decode-numerics.md); a consistency pass is not
an INT8 quality pass or a speedup claim.

`benchmarks/run_cached_checkpoint.py` adds a fresh calibration-only experiment with actual paged
decode probes, bounded full-calibration repair, whole-file freeze bindings, and all 32 cached logit
positions per regression prompt. Direct projection diagnostics separate reconstruction identity
from floating-point evaluation. Quality/cache gates are recomputed from raw observations at audit;
complete failed experiments remain failed. See [the cached checkpoint runbook](cached-calibration.md).

`benchmarks/run_scale_checkpoint.py` tests a new scale-aware weight method on fresh calibration
data. It registers parameters, source/tool/corpus hashes and a recoverable source archive **before**
fitting, and requires both families to pass calibration before any follow-up export/regression or
performance work. Its stopped calibration runs contain no timing matrix and cannot support
inference-speed claims; see [the scale-aware stage contract](scale-aware-quantization.md).

`benchmarks/run_refined_checkpoint.py` adds a separate pre-registered integer-refinement recipe,
with fixed fitted scales and complete block/row fallback. It uses fresh calibration, unchanged
strict gates, and the same stop-before-held-out rule. Independent reconstruction verifies the
same-moments/same-partition local objective; it is not a throughput or general-quality benchmark.
See [the refined stage contract](refined-quantization.md).

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
