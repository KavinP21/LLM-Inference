# Runtime architecture

Forge LLM separates model artifacts and request orchestration from device execution. CUDA and MLX
consume the same validated container and expose the same `submit`/`step`/`cancel`/`generate` API.

```text
                          Python token IDs
                                  │
                         create_engine(backend)
                         ┌────────┴────────┐
                         │                 │
                    MLX Engine        CUDA Engine
                         │                 │
              backend-neutral        C++ scheduler
              Python scheduler            │
                         │                 │
                 model factory        CUDA Qwen model
                  ┌─────┴─────┐             │
             Qwen adapter  Gemma adapter     │
                  └─────┬─────┘             │
                  MLX + custom Metal    cuBLASLt + kernels
                         └────────┬────────┘
                                  │
                  shared model format and test vectors
```

## MLX execution

The Apple backend implements complete Qwen2 and Gemma 3 text graphs with MLX operations. A
validated artifact selects a family adapter; the engine, scheduler, and physical K/V owner depend
only on the adapter protocol. Qwen uses biased projections, standard RMSNorm, SiLU gating, and a
uniform RoPE base. Gemma uses an explicit attention width, bias-free projections, head-wise Q/K
normalization, offset RMSNorm, scaled embeddings, tanh-GELU, four norms per block, two RoPE bases,
and a regular five-local/one-global attention pattern.

Model
weights are copied from read-only memory-mapped NumPy views and materialized before inference.
Aliased tied embeddings share one MLX array. FP32 is used for RMSNorm statistics, attention scores,
softmax, and returned logits; persistent weights and activations remain FP16.

The cache allocator assigns stable physical IDs in 16-token units. Each physical block owns an
independently replaceable MLX layer-page with shape
`[K_or_V][token_in_block][kv_head][head_dimension]` for every materialized layer. A per-request
logical table routes positions to those arrays. Pages materialize lazily, reject overwrites, and
return to a shared free list on completion, cancellation, or failure. Admission reserves worst-case
block capacity so an accepted request cannot be evicted halfway through generation.

Prompt work is limited to one configurable chunk per scheduler iteration. For supported production
head dimensions, prefill uses MLX fused causal GQA attention; the exact online-softmax tiled path is
kept as a bounded-workspace reference and small-shape/soft-cap fallback. Gemma sliding layers gather
only the page range visible to the current chunk, while global layers retain full-prefix attention.
Decode combines embeddings, projections, attention, and MLPs for all runnable sequences in one
batched model call.

Three Qwen custom-Metal chains reduce elementwise launches and intermediate traffic: residual add
plus RMSNorm, RoPE plus physical layer-page write, and SiLU-times-up. Gemma safely reuses the
RoPE/page-write chain but keeps its different norm and activation topology in MLX. Decode attention
packs the live
layer-pages once for the launch, remaps logical block tables into that tensor, and runs one
simdgroup per request/query-head with online FP32 softmax. This avoids per-request logical-history
concatenation and longest-sequence padding. It is not a zero-copy page pool: MLX's immutable array
contract requires the layer-page stack, which remains a measured single-request cost.

## CUDA KV layout

The device allocation is logically
`[physical_block][layer][K_or_V][kv_head][token_in_block][head_dimension]`.
Each sequence owns a logical block table whose entries identify physical blocks. Admission reserves
enough uncommitted capacity for the prompt plus requested output, while physical blocks are taken
from the free list only as sequence length crosses a 16-token boundary. This prevents an admitted
request from failing halfway through generation without eagerly occupying all of its blocks.

## Scheduler semantics

Both state-machine implementations return all currently decoding sequences plus at most one active
prefill. The MLX prefill remains active over multiple chunks, while every iteration still services
all runnable decodes first. Finished and cancelled requests immediately release reservations and
physical pages.

## Execution memory

CUDA model weights, KV blocks, activation buffers, and cuBLASLt workspace are allocated during
engine construction. A CUDA decoding iteration performs no device allocation. Linear weights retain
Hugging Face's `[out_features, in_features]` layout; both backends evaluate `X * Wᵀ`. The MLX
baseline remains selectable for numerical and performance differential tests. A future lower-level
page owner could remove the remaining layer-page stack before custom attention.
