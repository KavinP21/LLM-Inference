# Batch size and greedy output

With the default `decode_mode="batched"`, all running requests share one GEMM per
projection. FP16 reductions in MLX can come out slightly differently depending on
matrix shape, so the same prompt can produce different logits when it decodes
alone vs. next to seven other requests. The differences are tiny (the first ones
show up around 1e-5 in layer 0), but when two logits are nearly tied, greedy
decoding can pick a different token.

`decode_mode="rowwise"` sends each decode projection, including the LM head,
through the same single-row operation a lone request would use, then concatenates
the rows. Embeddings, norms, activations, RoPE/page writes and paged attention
are still batched, and prefill still uses the normal multi-token GEMM. A request's
decode output then doesn't depend on what else is running.

```python
from forge_llm import create_engine

engine = create_engine("models/gemma-3-1b-it.engine", backend="mlx",
                       decode_mode="rowwise")
```

It works the same way for INT8 artifacts. The direct Metal kernel is called once
per row, and the reconstruct modes build the dense FP16 matrix once and reuse it
across rows.

## Results

Qwen2.5-0.5B and Gemma 3 1B, 25 prompts × 32 tokens, decoded at concurrency 2, 8
and 16 and with staggered arrivals, compared with each prompt run alone:

| Model | Exact matches, `batched` (c2 / c8 / c16 / staggered) | `rowwise` |
| --- | --- | --- |
| Qwen FP16 | 23 / 21 / 23 / 25 | 25/25 everywhere, all logits bit-identical |
| Gemma FP16 | 24 / 25 / 23 / 23 | 25/25 everywhere, all logits bit-identical |

The mixed-precision INT8 artifacts behave the same way. Note that this is about
consistency across batch shapes for one artifact. It doesn't make INT8 match FP16
([quantization.md](quantization.md)), and it doesn't make FP16 match FP32.

The cost is throughput once several requests are decoding:

| Model, prompt × requests | `batched` tok/s | `rowwise` tok/s |
| --- | ---: | ---: |
| Qwen FP16, 128 × 1 | 122.8 | 122.4 |
| Qwen FP16, 128 × 8 | 426.3 | 231.5 |
| Qwen FP16, 1024 × 8 | 61.0 | 71.5 |
| Gemma FP16, 128 × 8 | 196.3 | 130.0 |
| Gemma FP16, 1024 × 8 | 53.8 | 48.3 |

Single requests are unaffected, and short concurrent prompts lose 34–46%.
`batched` stays the default.

## Tests

```bash
PYTHONPATH=python python -m pytest -q tests/test_decode_policy.py \
  tests/test_batch_numerics_driver.py tests/test_batch_numerics_metal.py
```

The Metal tests check exact logits at page boundaries on tiny Qwen and Gemma
models, check that attention still gets the whole batch in rowwise mode, and
inject a projection failure to check cleanup. `benchmarks/validate_batch_numerics.py`
runs the same comparison on a real model.
