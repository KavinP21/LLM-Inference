# Decode numerical policy

MLX has two explicit decode projection policies. Neither changes the artifact, dtypes, greedy
argmax, prefill chunk size, context limit, admission policy, or cache layout.

```python
from forge_llm import create_engine

engine = create_engine(
    "models/gemma-3-1b-it.engine",
    backend="mlx",
    max_num_sequences=16,
    max_model_length=32768,
    kv_cache_bytes=1024 << 20,
    decode_mode="rowwise",  # default is "batched"
)
```

`batched` shares native matrix multiplications across all runnable requests, preserving the original
throughput-oriented path. Native FP16 reductions can differ slightly with matrix shape. The saved
stage replay isolates differences with **identical** projection inputs and weights at batch widths
1 and 8; it also replays each operation on the same materialized input to distinguish a local
shape effect from propagated upstream error. Tied or nearly tied logits can therefore select
different greedy tokens, even when cosine similarity is extremely high. This is not resolved by
loosening a cosine threshold or adding a prompt-specific tie rule.

`rowwise` sends each decode projection, including the LM head, through the same single-row operation
used by an independent request, concatenating the resulting rows in scheduler order. Embedding,
normalization, activation, RoPE/page writes, and paged attention still execute as a batch. It does
**not** serialize complete decoders, disable continuous batching, or silently change defaults.
Prefill continues to use native multi-token GEMM. A single-request decode graph is unchanged.

For INT8, this policy is independent of `int8_mode`. Direct Metal projections use single-row kernel
calls. Reconstruction modes build one temporary dense matrix per projection and share it among
the row GEMVs; there is no persistent second model or repeated reconstruction per row. Biases,
norms, embeddings, activations, and K/V retain their original precision.

The policy is an accuracy/throughput tradeoff, **not a speed optimization**. Splitting native GEMMs
loses cross-request weight reuse and adds dispatches/intermediate row outputs. Memory and latency
must be measured on the intended workload. It also does not undo lossy weight quantization, make
FP16 match FP32, or certify parity with Transformers.

## Reproducible gates

```bash
PYTHONPATH=python .venv/bin/pytest -q tests/test_decode_policy.py tests/test_batch_numerics_driver.py
# Requires real Metal access:
PYTHONPATH=python .venv/bin/pytest -q tests/test_batch_numerics_metal.py

.venv/bin/python benchmarks/run_batch_numerics_checkpoint.py \
  --output-dir benchmarks/results/batch-numerics-new-run

# Audit the saved, split development checkpoint without using the GPU:
.venv/bin/python benchmarks/audit_batch_checkpoint.py \
  --result-dir benchmarks/results/batch-invariant-m3-max-2026-10-01 \
  --output /tmp/forge-batch-evidence-audit-new.json
```

The checkpoint driver uses the inherited exclusive GPU workflow lease and isolated child
processes. It refuses existing output directories, rotates A/B order, runs five full warmups and
three measured trials per configuration, and records source/artifact/workload checksums. Do not
run unrelated GPU jobs or the desktop chat app during measurement; the lease only excludes
cooperating Forge evidence drivers. Benchmark and context CLIs accept `--decode-mode` explicitly.

Correctness uses freshly generated independent outputs and per-token FP32 logit-byte hashes from
the **same current source**. Historical reports are separately checked for default-token regression,
never treated as fresh references. The older INT8 checker still refuses stale source hashes.
Saturated arrivals cover concurrency 2/8/16 with 32 outputs per request; staggered arrivals use
1/4/8/16/32 output budgets. Capacity-aware insertion never discards excess requests. All runs
require exact event counts, completion, materialized cancellation, and complete cache reclamation.
Tiny two-family tests additionally compare exact logits at page boundaries, verify that attention
still receives the full batch, and inject a projection failure to test cleanup.

Bitwise consistency is verified for the saved hardware/software/artifacts/workloads, not promised
for arbitrary MLX versions, other chips, fallback operators, or unsupported models. A changed
backend or numerical policy must rerun these gates. The 32K checks prove resource integrity with
one decode iteration; they are not long-context semantic-quality or sustained-throughput evidence.

See [the measured results](mlx-batch-numerics-results.md). Strict cross-artifact INT8 greedy quality
remains a separate, failed gate; W4A16 is still deferred.
