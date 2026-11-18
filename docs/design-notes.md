# Design notes

## Reserve capacity before materializing pages

An admitted request can grow well beyond its prompt. Allocating only current tokens
can exhaust the pool halfway through generation. The allocator reserves maximum
request length at admission, then takes pages as the sequence grows.

Prefix reuse counts shared pages once and reserves a private copy for partial-tail
append. Cache pins and requests own separate references; clearing pins must not
invalidate active readers. See [mlx_engine.py](../python/instinct/mlx_engine.py),
[prefix_cache.py](../python/instinct/prefix_cache.py), and their
[ownership tests](../tests/test_mlx_engine.py).

## Make long prompts visible to the scheduler

Chunking prefill lets each iteration service runnable decodes and at most one
prompt chunk. Requests join or leave between iterations. This bounds a unit of
prompt work; it does not implement packed multi-prompt prefill or general fairness.
The loop is in [mlx_engine.py](../python/instinct/mlx_engine.py).

## Operation shapes affect FP16 argmax

Batched and single-row native projections can reduce values differently. Small
logit differences can change greedy decisions near a tie, then change the rest of
a continuation. Cosine similarity alone does not establish token parity.

So a request decoded alone and the same request decoded next to others can, rarely,
pick different tokens at an FP16 near-tie. The engine tests compare whole batched
runs with running each prompt alone on the tiny models, where no ties occur.

## Check fused-kernel eligibility

MLX chooses fused operations by backend and input shape. Off Metal,
[portable.py](../python/instinct/backends/portable.py) checks representative
shapes before opting into fused operations. Its fallback implementations have
[reference tests](../tests/test_portable_kernels.py).

## Measure page packing with attention

Metal decode follows page tables directly into the K/V arena. Portable decode
gathers the visible rows before attention. Measure time in the full decode step:
a faster attention kernel can lose the gain if it needs more copies or padding.
