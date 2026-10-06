# FP16 MLX prefix-cache contract

This is opt-in, engine-local reuse of immutable paged K/V state. It is not a model
API wrapper, disk cache, cross-worker cache or service authorization layer. CUDA
prefix caching and quantized prefix caching are not implemented by this checkpoint.
The original `prefix_cache_bytes=0` default leaves the old private-page allocator
and inference policy in use. Desktop app defaults are unchanged.

## Public API

```python
from forge_llm import MlxEngine

with MlxEngine(
    "models/qwen2.5-0.5b.engine",
    max_model_length=32768,
    kv_cache_bytes=2 << 30,
    prefix_cache_bytes=1 << 30,  # A retention limit within, not beyond, KV capacity.
    prefix_cache_max_entries=64,
    prefill_chunk_size=512,
    decode_mode="rowwise",     # Qualified exact same-artifact numerical policy.
    prefix_cache_namespace="deployment-v1",
) as engine:
    first = engine.generate([17, 23, 42], 32, cache_namespace="session-A")
    reused = engine.generate([17, 23, 42], 32, cache_namespace="session-A")
    assert reused == first
    print(engine.stats()["prefix_cache"])
    engine.clear_prefix_cache("session-A")
```

`submit` also accepts keyword-only `cache_namespace`. Exact token tuples and a
request namespace identify entries. Engine identity additionally binds the weight
data checksum, full decoder configuration, page size, chunk/tile sizes, kernel
switches and decode policy. Entries never leave their engine. Changing execution
settings on an enabled engine fails closed; construct another engine instead.
Namespaces must be strings of at most 1,024 UTF-8 bytes. Different namespaces do
not match; they still compete for the same bounded LRU budget. This is isolation
of reuse, not authentication, quota enforcement or timing-side-channel protection.
The synchronous engine API is not thread-safe; future serving must serialize it.

## Exact reuse and numerical boundary

Each completely written prompt chunk publishes a snapshot containing its token
prefix, page table and final FP32 logit row. An exact full-prompt hit produces its
first token from the saved row without a prefill forward pass. For a changed suffix,
lookup selects the longest matching snapshot at an original prefill chunk boundary.
It does **not** reuse an arbitrary partial final chunk of a different prompt: doing
so would change FP16 GEMM shapes and can change reductions or greedy near-ties.

The final logit row is copied to read-only NumPy host storage. This prevents a
small view from retaining an entire chunk's vocabulary logits on the GPU. Cache
rows are bounded by entry count; pending admitted matches can retain additional
rows until consumed/cancelled, separately reported. A hit transfers one row back
to MLX. Cold insertions incur synchronization/copy overhead, which must be measured.
Token-key metadata is also bounded by entry count and maximum context length.

The qualified protocol uses `decode_mode="rowwise"`: single-row projections and
batched attention. Prefix caching can be requested with the unchanged `batched`
default, but this checkpoint does not certify bitwise parity across different
native batched projection shapes or schedules. Neither policy provides new
Transformers-reference semantic certification or long-context task accuracy.

## Ownership, admission and copy-on-write

```text
cache snapshot --pin--+--> immutable physical page <--reference-- request A
                     +------------------------------reference-- request B

write a shared partial tail:
  reserve COW capacity -> clone ownership/masks -> replace private layer arrays
  -> drop writer's old reference -> append only to its private page

last request/reference + last cache pin released -> return physical ID -> drop store
```

`SharedKVBlockPool` counts unique physical pages, not the sum of logical tables.
Its committed budget is unique allocated pages plus unmaterialized private page
reservations plus potential partial-tail COW pages. Admission reserves the entire
worst-case continuation before accepting a request. Publishing a partial snapshot
reserves each affected active writer's COW capacity; if that cannot fit, insertion
is skipped instead of overcommitting admitted work. Full shared prefix pages cannot
be rewritten. The device store independently rejects writes to pinned/shared pages.

COW copies page ownership and written masks. MLX arrays are immutable, so existing
layer arrays can initially be aliased; append operations replace private arrays.
Accounting conservatively counts the new logical physical page immediately. There
is no full-history copy for branching. Attention still packs live layer-pages with
MLX `stack`; this is **not** pointer-addressable zero-copy attention or a new Flash
Attention implementation.

LRU limits both unique pinned pages and entry count. Admission may evict cache pins
but never active references. A matched entry stays protected during admission.
`clear_prefix_cache` releases pins without invalidating waiting matches or running
requests. Completion/cancellation releases request references, so cached pages may
remain intentionally. Clear an idle cache or close the engine for zero page usage.
Exceptions reclaim affected requests; valid previously published snapshots may
remain. Closing is idempotent and releases waiting, prefilling and running owners.

Stats separate logical live tokens, physical covered slots, unique allocations,
active page references, cache-only/shared pages, pending/COW reservations, COW copies,
fragmentation, hits/misses, matched/reused tokens, skips, evictions and rejected
admissions. A submitted hit can be cancelled before reuse: `matched_tokens` is not
`reused_tokens`. KV byte counts are reservation/materialization accounting, not
Apple GPU memory counters; MLX active/peak memory is collected separately.

## Frozen validation and measurement

`benchmarks/run_prefix_checkpoint.py` creates a **fresh** result directory and freezes
source/configuration/workloads/models/gates before official validation. Its source
archive captures the runtime, host source, tests and checkpoint tools. Historical
INT8 results and archives are whole-file bound, not rewritten. Those live auditors
were freshly checked before this milestone changed runtime source; their old runtime
guards are intentionally not bypassed to pretend the new runtime is historical.

Registered coverage for each official FP16 family (Qwen2.5 0.5B and Gemma 3 1B):

- 25 fixed text prompts plus 15 page/chunk-boundary synthetic prompts, each with
  32 outputs, at chunk sizes 127 and 512. Compare uncached, cold, exact-hit and suffix
  branches using greedy tokens and SHA-256/dtype/shape identities of every logit row.
- Eight concurrent shared/branched requests versus independent uncached references;
  waiting-hit cancellation, cache clearing with pending/active owners, namespace misses,
  COW, bounded-entry LRU pressure and final complete reclamation.
- 32,766 prompt tokens plus two generated tokens: uncached/cold/warm row/token parity,
  no warm prefill work, one real paged-decode iteration, complete idle-cache cleanup.
- Portable randomized allocator invariants and tiny-model Metal tests independently
  cover injected failures, immutable-write guards and partial-page branching.

Only after **all** registered parity/resource checks pass: isolated-process disabled,
cold and primed A/B, prompt lengths 128/1,024 and concurrency 1/8, both families,
five warmups and three measured trials per configuration (24 reports). Mode order
rotates across shapes. Prompts are deterministic full-prefix repeats; priming is
explicitly outside timing, while cold insertion is inside. Queue-inclusive latency
starts after each submit returns; total wall time includes submission and scheduling.
TPOT is `(E2E - TTFT) / 31` for 32 outputs. Throughput uses outputs divided by trial
wall **seconds**; `run_once` returns milliseconds, explicitly converted by the driver.

Raw requests/trials, p50/p95/p99, unique KV peaks and MLX active/peak allocation bytes
are saved. Independent readback recomputes workload coverage, token/logit alignment,
cleanup, work counters, throughput and percentiles. The final evidence index binds
all result files and a no-skips full-suite JUnit report. Any gate failure forbids timing;
partial/negative evidence is preserved, not overwritten or tuned into acceptance.

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run_prefix_checkpoint.py \
  --output-dir benchmarks/results/prefix-cache-NEW-RUN --phase all
PYTHONPATH=python FORGE_INSPECT_MODEL=build/forge-inspect-model-int8 \
  .venv/bin/python -m pytest -q \
  --junitxml=benchmarks/results/prefix-cache-NEW-RUN/test-results.xml
PYTHONPATH=python .venv/bin/python benchmarks/run_prefix_checkpoint.py \
  --output-dir benchmarks/results/prefix-cache-NEW-RUN --phase final-audit \
  --expected-tests 356
```

Run GPU workflows serially under the evidence lease. This lease coordinates our
drivers, not unrelated apps or production request routing. No CUDA, GPU-utilization,
power, Xcode hardware-counter, general decode speedup or production-service claim
follows from this checkpoint. Results must distinguish warm savings from cold cost.
