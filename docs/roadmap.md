# Known limits and next steps

## Limits today

- **CUDA hasn't run on a GPU.** The Qwen graph, cuBLASLt plans, kernels, memory
  arenas and tests exist, and CI compiles them, but nothing has been checked for
  correctness or speed on NVIDIA hardware. Its attention kernels also cap context
  at 2,048 tokens.
- **Paged attention on Metal copies pages.** MLX arrays are immutable, so the
  attention kernel stacks each layer's live pages into one array before launch.
  That copy is why a single long request barely speeds up.
- **One prefill chunk per step.** The scheduler doesn't pack several prompts into
  one prefill batch.
- **Gemma's sliding-window layers** limit how much attention work they do, but they
  keep their pages until the request finishes.
- **Qwen3.5 runs through MLX-VLM**, one request per replica. It doesn't get the
  native engine's paged KV cache, continuous batching or prefix reuse.
- **No sampling in the native engine** (greedy only), no native 4-bit weights,
  no streaming or OpenAI-compatible server, and no tensor/pipeline parallelism.
- **Long context is only smoke-tested.** The 32K runs show it executes and frees
  memory, not that output quality holds up or how fast sustained decoding is.

## Next

| Work | Done when |
| --- | --- |
| NVIDIA bring-up | Qwen runs on an RTX card and matches reference tokens and logits, Compute Sanitizer is clean, and there's a benchmark matrix with Nsight traces |
| Prefix-cache timing | Disabled/cold/primed runs at 128 and 1,024 tokens, concurrency 1 and 8, for both models, with insertion and reuse cost reported separately |
| Cheaper page packing | Profile the per-layer page stack on its own, then compare alternatives at the same workload and memory accounting |
| Qwen3.5 | Longer contexts, more varied visual and agent tasks, and load testing. Speculation stays off until it helps on more than copying |
