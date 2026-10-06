# Design notes

## Reserve capacity before materializing pages

An admitted request can grow well beyond its prompt. Allocating only current tokens
can exhaust the pool halfway through generation. The allocator reserves maximum
request length at admission, then materializes 16-token pages as needed.

Prefix reuse counts shared pages once and reserves a private copy for partial-tail
append. Cache pins and requests own separate references; clearing pins must not
invalidate active readers. See [runtime.py](../python/forge_llm/runtime.py),
[prefix_cache.py](../python/forge_llm/prefix_cache.py), and their
[ownership tests](../tests/test_prefix_cache.py).

## Make long prompts visible to the scheduler

Chunking prefill lets each iteration service runnable decodes and at most one
prompt chunk. Requests join or leave between iterations. This bounds a unit of
prompt work; it does not implement packed multi-prompt prefill or general fairness.
The loop is in [mlx_engine.py](../python/forge_llm/mlx_engine.py).

## Operation shapes affect FP16 argmax

Batched and single-row native projections can reduce values differently. Small
logit differences can change greedy decisions near a tie, then change the rest of
a continuation. Cosine similarity alone does not establish token parity.

The optional `rowwise` policy evaluates projection rows independently while keeping
attention batched, with a measured dispatch/weight-reuse cost. Prefix snapshots
preserve original prefill chunk boundaries to avoid changing reduction shapes.
These rules address execution consistency, not lossy quantization error. See the
[policy](decode-numerics.md) and [measurements](mlx-batch-numerics-results.md).

## Check fused-kernel eligibility

Prefix validation exposed a Qwen call with five query tokens and GQA factor seven.
Forcing MLX 0.32.2's fused short-query kernel on that shape failed. The guard now
checks head width and the query-length/GQA limit before forcing fusion; other
shapes use normal MLX dispatch. Execution errors are not swallowed.

Read `_fast_attention` in [the Qwen adapter](../python/forge_llm/backends/mlx.py).
[Metal regressions](../tests/test_prefix_cache_metal.py) cover query lengths
1, 4, 5, 6, 7, 8, 9, and 17. The [report](mlx-prefix-cache-results.md) retains
both the failed run and fresh validation.

## Measure page packing with attention

The Metal kernel uses block tables over packed live pages, avoiding per-request
history gathering and longest-request padding. MLX's immutable arrays still
require a per-layer stack. That cost explains the weaker single-request benchmark.
A different page owner needs fresh end-to-end measurements before a speed claim.
