# MLX Gemma 3 model-family checkpoint

Date: 2026-09-23  
Host: Apple M3 Max, 128 GB unified memory  
Runtime: MLX 0.32.2, Transformers 5.17.0 for reference-only checks

This report closes the second-model-family milestone for the MLX runtime. It is a correctness and
resource-validation report, not a claim that the Gemma path has completed the full statistical
benchmark matrix.

## Architecture contract implemented

The adapter follows the text decoder in the
[official Transformers Gemma 3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py)
and the 1B context limit in the
[Google Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3). It does not route
Gemma through the Qwen graph. The implemented differences include:

- explicit 256-element attention heads even though `hidden_size / query_heads` is 288;
- bias-free Q/K/V projections and per-head Q/K RMS normalization;
- Gemma RMSNorm's `1 + weight` rule with FP32 statistics and multiplication before downcast;
- embedding scaling by `sqrt(hidden_size)`;
- five 512-token sliding-attention layers followed by one global layer;
- 10,000 local and 1,000,000 global RoPE bases;
- `query_pre_attn_scalar` attention scaling;
- pre/post attention and pre/post feed-forward normalization;
- approximate tanh-GELU gating and optional attention/final logit soft caps.

The scheduler, admission policy, physical page owner, batched decode loop, RoPE/page-write kernel,
and paged GQA decode kernel are shared. Qwen-only residual/RMSNorm and SwiGLU kernels are not applied
to Gemma. Sliding prefill gathers only the logical K/V range visible to the current chunk; global
layers retain the full prefix.

## Artifact and cross-language validation

`google/gemma-3-1b-it` was exported through the normal Hugging Face command into artifact version 2.

| Field | Value |
| --- | ---: |
| File bytes | 1,999,794,688 |
| Tensors | 341 |
| Data SHA-256 | `92d3081b2facfa8a5eb48dcedd89cb8e230b21a00cfeefd4c32f77ca2a9482ee` |
| Model family | `gemma3_text` |
| Maximum positions | 32,768 |

The exporter reopens the completed file and applies the backend-independent Gemma tensor contract.
The Python memory-mapped reader and C++ inspector both accepted a separately generated version-2
Gemma fixture. The updated readers continue to accept the existing version-1 Qwen artifact.

## Correctness evidence

The synthetic gate builds a two-layer Gemma model with an explicit attention width that differs from
`hidden_size / heads`, a four-token sliding window, and alternating local/global layers. It checks:

- chunked prefill logits against `transformers.Gemma3ForCausalLM` with FP16-exported weights;
- six greedy cached-decode tokens against independent full-prefix Transformers evaluation;
- mixed-length continuous batching and custom-Metal versus MLX-fallback output;
- complete physical-page reclamation.

All three Gemma tests pass. Together with the Qwen suite, 19 Metal-backed tests pass.

The official-model gate uses the same four fixed prompts as the Qwen checkpoint and 32 raw-greedy
tokens per prompt. Initial-prompt logits pass the `>= 0.999` requirement in every case:

| Case | Logit cosine | Max absolute error | Matching greedy prefix |
| ---: | ---: | ---: | ---: |
| 1 | 0.9999980 | 0.062500 | 32 / 32 |
| 2 | 0.9999976 | 0.046875 | 32 / 32 |
| 3 | 0.9999963 | 0.062500 | 25 / 32 |
| 4 | 0.9999982 | 0.041016 | 32 / 32 |

The strict 32-token gate is therefore **not marked fully exact**. Case 3 has one root decision at
generated-token index 25 where the reference top-two scores are separated by only a few FP16 units;
the remaining seven positional differences are the expected autoregressive cascade. Both the MLX
fallback and custom paged-decode paths reach the same root decision, while a full-prefix Forge
recomputation selects the reference token. This localizes the issue to accumulated FP16 kernel-order
differences in cached decoding rather than the Gemma graph, sliding mask, or custom attention alone.
The correctness runner stays strict and exits nonzero; it also records the reference margin and
top-two candidates instead of hiding the exception.

The Qwen regression gate remains exact for all 128 generated tokens, with cosine similarity from
0.9999808 to 0.9999977.

## 32K resource gate

The long-context command processed a deterministic 32,766-token prompt, produced two tokens, and
therefore exercised one real custom paged-decode iteration after prefill.

| Observation | Value |
| --- | ---: |
| Prefill iterations at 512 tokens/chunk | 64 |
| Decode iterations | 1 |
| Peak physical K/V blocks | 2,048 |
| Peak materialized blocks | 2,048 |
| Peak K/V bytes | 872,415,232 |
| Blocks/reservations/pages after completion | 0 / 0 / 0 |
| Reclamation check | passed |

The run used a 1,024 MiB logical K/V pool because Gemma's 256-wide K/V head makes its 32K cache
larger than the default 512 MiB pool. Observed timing is intentionally treated only as diagnostic:
this was one execution/resource trial, not the warmup/repetition matrix required for a performance
claim.

## Continuous-batching smoke gate

An official-model run admitted four requests, completed a warmup plus one measured repetition, and
returned to zero allocated, reserved, and materialized K/V blocks. It exercised the public model
factory and shared request API with `model_type = gemma3_text`. The one-trial throughput and latency
are retained in the ignored raw JSON but are not published as benchmark results.

## Remaining limitations

- Persistent sliding-layer K/V pages are still retained until request completion. Attention work is
  window-bounded, but a future ring/page reclamation policy could reduce the 32K cache footprint.
- Prefill processes one request chunk per scheduler iteration; there is no multi-prompt packed
  prefill yet.
- The custom paged kernel still receives an MLX-packed live-page tensor rather than a single mutable,
  pointer-addressable page pool.
- The next checkpoint is measured weight-only INT8, followed by groupwise W4A16. Quantization must
  preserve separate Qwen and Gemma correctness gates.
