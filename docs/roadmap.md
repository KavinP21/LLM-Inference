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
- **Few tokens per expert is slow.** Gemma's MoE cost follows how many distinct
  experts a step touches, and MLX's `gather_qmm` reaches only about a third of
  memory bandwidth with a few rows per expert: one decoding user costs 15 ms a
  step, eight cost 49 ms, and a 16-token prefill chunk costs about 64 ms. So
  a long prompt arriving stalls other users' decoding by about half a second a
  token (512-token chunks). Smaller chunks only trade that stall for a longer
  wait for the new prompt: keeping steps under 100 ms means 16-token chunks
  and about 50 s before an 8K-token prompt's first token. Running the chunk in
  the same forward pass as the decode tokens measured no faster.
- **No tensor or pipeline parallelism.**
- **Long prompts are slow to prefill.** 262K tokens take 29 minutes on Gemma.
  The five global layers attend with 512-wide heads, which MLX runs unfused at
  about 10 TFLOP/s; at 258K tokens a 2,048-token chunk takes 11 s. Larger
  attention tiles didn't help.

## Next

| Work | Done when |
| --- | --- |
| NVIDIA bring-up | Qwen runs on an RTX card and matches reference tokens and logits, Compute Sanitizer is clean, and there's a benchmark matrix with Nsight traces |
| Prefix-cache timing | Disabled/cold/primed runs at 128 and 1,024 tokens, concurrency 1 and 8, for both models, with insertion and reuse cost reported separately |
| Small-batch MoE kernel | A 4-bit expert kernel that reads each touched expert once at near full bandwidth for 2–64 tokens; three users keep decoding in under 100 ms a token while an 8K-token prompt prefills |
| Fused long-context attention | A flash-attention prefill kernel for 512-wide heads; 262K tokens prefill in under 10 minutes |
| Pipelined decode | Build step N+1's graph while the GPU runs step N; one user back above 76 tok/s on Gemma |
| MTP drafting | Gemma 4's assistant drafts inside the batched engine and helps at agent-length contexts |
| Qwen3.5 | Longer contexts, more varied visual and agent tasks, and load testing. Speculation stays off until it helps on more than copying |
