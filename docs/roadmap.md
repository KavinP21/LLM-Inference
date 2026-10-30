# Known limits and next steps

## Limits today

- **CUDA hasn't run on a GPU.** The Qwen graph, cuBLASLt plans, kernels, memory
  arenas and tests exist, and CI compiles them, but nothing has been checked for
  correctness or speed on NVIDIA hardware. Its attention kernels also cap context
  at 2,048 tokens.
- **One prefill chunk per step.** The scheduler doesn't pack several prompts into
  one prefill batch.
- **One user decodes slower than mlx-vlm did** (65 vs 76 tok/s on Gemma): the
  engine builds each step's graph while the GPU waits, and Gemma's MTP drafter
  isn't wired in. Several users at once are much faster than before.
- **Small batches of 4-bit matmuls** cost about twice a single row in MLX at
  eight rows, which limits how well eight users scale.
- **No tensor or pipeline parallelism.**
- **Long prompts are slow to prefill.** 262K tokens take 29 minutes on Gemma, and
  a long prompt arriving stalls other users' decoding by about half a second a
  token while it prefills.

## Next

| Work | Done when |
| --- | --- |
| NVIDIA bring-up | Qwen runs on an RTX card and matches reference tokens and logits, Compute Sanitizer is clean, and there's a benchmark matrix with Nsight traces |
| Prefix-cache timing | Disabled/cold/primed runs at 128 and 1,024 tokens, concurrency 1 and 8, for both models, with insertion and reuse cost reported separately |
| Pipelined decode | Build step N+1's graph while the GPU runs step N; one user back above 76 tok/s on Gemma |
| MTP drafting | Gemma 4's assistant drafts inside the batched engine and helps at agent-length contexts |
| Qwen3.5 | Longer contexts, more varied visual and agent tasks, and load testing. Speculation stays off until it helps on more than copying |
