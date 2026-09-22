# Runtime architecture

Forge LLM separates portable request management from NVIDIA execution. This keeps scheduler and
KV-allocation invariants testable on macOS while all numerical and performance claims remain tied
to the RTX CUDA target.

```text
Python token IDs
      │
      ▼
  pybind11 Engine ──► FCFS scheduler ──► request lifecycle
      │                     │
      │                     └──► reservation-aware KV block pool
      ▼
QwenModel::forward(batch of one-token queries)
      │
      ├──► cuBLASLt Q/K/V, O, MLP, and LM-head GEMMs
      ├──► fused RMSNorm/residual and SwiGLU kernels
      ├──► RoPE + direct paged K/V scatter
      └──► paged GQA attention + GPU argmax
```

## KV layout

The device allocation is logically
`[physical_block][layer][K_or_V][kv_head][token_in_block][head_dimension]`.
Each sequence owns a logical block table whose entries identify physical blocks. Admission reserves
enough uncommitted capacity for the prompt plus requested output, while physical blocks are taken
from the free list only as sequence length crosses a 16-token boundary. This prevents an admitted
request from failing halfway through generation without eagerly occupying all of its blocks.

## Scheduler semantics

`Scheduler::next` returns all currently decoding sequences plus at most one waiting prefill. The
engine first performs a batched decode and then the selected full-prompt prefill. Finished and
cancelled requests immediately return their blocks. Prefill uses batched linear layers and a causal
paged-attention kernel; it is intentionally not yet tiled like FlashAttention.

## Execution memory

Model weights, KV blocks, activation buffers, and cuBLASLt workspace are allocated during engine
construction. A decoding iteration performs no device allocation. Linear weights retain Hugging
Face's `[out_features, in_features]` layout; the row-major cuBLASLt wrapper evaluates `X * Wᵀ` and
uses the bias epilogue for Q/K/V projections.
