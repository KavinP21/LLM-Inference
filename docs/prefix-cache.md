# Prefix caching

Opt-in reuse of KV pages for repeated prompt prefixes in the MLX engine. It's
FP16-only and per engine: no disk cache, nothing shared across workers, and no
CUDA support. With the default `prefix_cache_bytes=0` the engine uses the
plain private-page allocator, exactly as before.

```python
from instinct import MlxEngine

with MlxEngine(
    "models/qwen2.5-0.5b.engine",
    max_model_length=32768,
    kv_cache_bytes=2 << 30,
    prefix_cache_bytes=1 << 30,       # part of kv_cache_bytes, not on top of it
    prefix_cache_max_entries=64,
    decode_mode="rowwise",            # see decode-numerics.md
    prefix_cache_namespace="deployment-v1",
) as engine:
    first = engine.generate([17, 23, 42], 32, cache_namespace="session-A")
    again = engine.generate([17, 23, 42], 32, cache_namespace="session-A")
    assert again == first
    print(engine.stats()["prefix_cache"])
    engine.clear_prefix_cache("session-A")
```

`submit` takes the same `cache_namespace` keyword. Entries only match within one
namespace, but all namespaces share one LRU budget, so this keeps reuse separate.
It isn't an access-control mechanism. The engine isn't thread-safe; callers have
to serialize access.

## How matching works

After each prefill chunk is written, the engine saves a snapshot: the token
prefix, its page table, and the final logit row (copied to host memory so it
doesn't pin a whole chunk of logits on the GPU).

- An exact full-prompt hit produces the first token from the saved logit row and
  skips prefill entirely.
- Otherwise the engine reuses the longest snapshot that ends on one of the
  original chunk boundaries and prefills the rest. It never reuses a partial
  chunk, because a different chunk shape changes the FP16 GEMM and can flip a
  near-tied token.

The cache key also covers the weights' checksum, model config, page size,
chunk/tile sizes, kernel switches and decode mode. Changing any of those on a
live engine raises an error instead of serving stale entries.

## Ownership and copy-on-write

```text
cache snapshot --pin--+--> physical page <--ref-- request A
                      +---------------------ref-- request B

appending to a shared partial last page:
  reserve a page -> clone it -> switch the writer to the clone -> append
```

`SharedKVBlockPool` reference-counts physical pages. Admission reserves the
worst case up front: the request's private pages plus a possible copy of a
shared partial tail. An admitted request can't run out of pages halfway through.
Full shared pages are never written; the KV store rejects writes to pinned pages
as a second line of defence.

LRU eviction drops cache pins, never pages an active request is using.
`clear_prefix_cache` releases pins without disturbing waiting or running
requests. Since completed requests can leave pages pinned in the cache, clear
the cache or close the engine to get back to zero pages.

`stats()["prefix_cache"]` reports hits/misses, matched vs. actually reused tokens
(a hit can be cancelled before it's used), shared and cache-only pages, COW
copies, evictions and admission rejections.

## Status

Correctness is verified on Qwen2.5-0.5B and Gemma 3 1B (FP16, rowwise decode).
Outputs and logit rows are bit-identical between uncached, cold, exact-hit and
suffix-hit runs:

- 40 prompts (25 text + 15 sized around page/chunk boundaries), 32 tokens each,
  at chunk sizes 127 and 512
- eight concurrent requests sharing and branching off one prefix, plus
  cancellation, cache clears with live owners, namespace misses and LRU pressure
- a 32,766-token prompt, where the warm run does no prefill work and every page
  is released afterwards

The first validation run failed. With prefix reuse, a Qwen prefill can end in a
very short tail, and forcing that onto MLX's fused attention kernel raised an
error for unsupported shapes. Those shapes now fall back to MLX's default path.

**Timing hasn't been measured yet**, so there's no speedup claim.
`benchmarks/benchmark_prefix.py` runs one configuration per process (prefix
cache disabled, cold, or primed):

```bash
PYTHONPATH=python python benchmarks/benchmark_prefix.py \
  --model models/qwen2.5-0.5b.engine --mode primed \
  --prompt-length 1024 --concurrency 8 --output benchmark-results/prefix-primed.json
```
