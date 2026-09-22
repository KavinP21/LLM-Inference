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
                MLX Qwen model        CUDA Qwen model
                         │                 │
             MLX Metal operations     cuBLASLt + kernels
                         └────────┬────────┘
                                  │
                  shared model format and test vectors
```

## MLX foundation

The first Apple checkpoint implements the complete model graph with standard MLX operations. Model
weights are copied from read-only memory-mapped NumPy views and materialized before inference.
Aliased tied embeddings share one MLX array. FP32 is used for RMSNorm statistics, attention scores,
softmax, and returned logits; persistent weights and activations remain FP16.

Each live MLX request owns contiguous per-layer K/V arrays. The scheduler accounts for those arrays
in 16-token blocks and reserves worst-case capacity at admission. This establishes the correctness
and lifecycle baseline before physical paging and custom Metal kernels are introduced.

## CUDA KV layout

The device allocation is logically
`[physical_block][layer][K_or_V][kv_head][token_in_block][head_dimension]`.
Each sequence owns a logical block table whose entries identify physical blocks. Admission reserves
enough uncommitted capacity for the prompt plus requested output, while physical blocks are taken
from the free list only as sequence length crosses a 16-token boundary. This prevents an admitted
request from failing halfway through generation without eagerly occupying all of its blocks.

## Scheduler semantics

Both state-machine implementations return all currently decoding sequences plus at most one waiting
prefill. The CUDA engine performs a batched decode. The foundation MLX engine currently visits those
requests sequentially; tensor batching is part of the next checkpoint. Finished and cancelled
requests immediately release reservations.

## Execution memory

CUDA model weights, KV blocks, activation buffers, and cuBLASLt workspace are allocated during
engine construction. A CUDA decoding iteration performs no device allocation. Linear weights retain
Hugging Face's `[out_features, in_features]` layout; both backends evaluate `X * Wᵀ`. MLX K/V
concatenation currently allocates new arrays during decode and is explicitly not the final memory
design.
