# INT8 weights (experimental)

Forge can convert a Qwen2 or Gemma 3 FP16 artifact into a W8A16 artifact (INT8
weights, FP16 activations) and run it on MLX/Metal. It works, and it saves memory,
but it doesn't reproduce the FP16 model's greedy output closely enough to be the
default. FP16 stays the default everywhere. CUDA rejects INT8 artifacts.

## Converting and running

```bash
PYTHONPATH=python python -m forge_llm.quantization \
  models/qwen2.5-0.5b.engine models/qwen2.5-0.5b-int8.engine
```

```python
from forge_llm import create_engine

with create_engine("models/qwen2.5-0.5b-int8.engine", backend="mlx",
                   int8_mode="metal") as engine:
    print(engine.generate([1, 2, 3], max_new_tokens=32))
```

The converter refuses to overwrite its output or its source, and validates the
result before writing it. Only the attention Q/K/V/O and MLP gate/up/down
projections are quantized. Embeddings, the tied output head, biases, norms and the
KV cache stay FP16.

## Numerics

Each projection row gets one FP32 scale:

```
scale = max(abs(weight)) / 127          # all-zero rows use scale = 1
q     = clip(round_half_even(weight / scale), -127, 127)
w'    = fp16(fp32(q) * scale)
```

There's no activation quantization, group-wise scaling or clipping search in the
basic converter.

`int8_mode` picks how the quantized projections run:

| Mode | Decode (1–16 rows) | Prefill |
| --- | --- | --- |
| `dequantize` | Rebuild an FP16 matrix with MLX ops, then matmul | Same |
| `metal` / `auto` | Custom W8A16 Metal kernel | One-pass Metal unpack to FP16, then MLX matmul |
| `reconstruct` | One-pass Metal unpack, then MLX matmul | Same |

The direct kernel assigns four output channels to a 128-thread group, accumulates
in FP32 and reuses each weight load across the batch rows. It avoids building a
dense FP16 matrix during decode, but its reduction order differs from MLX's GEMM,
which is enough to flip near-tied greedy decisions. `reconstruct` keeps MLX's
reduction order at the cost of a temporary FP16 matrix per projection. No mode
keeps a second permanent copy of the weights.

## Mixed precision

Most of the quality loss comes from a handful of projections. `--retain-fp16`
(repeatable) keeps named projections in FP16:

```bash
PYTHONPATH=python python -m forge_llm.quantization \
  models/qwen2.5-0.5b.engine models/qwen2.5-0.5b-int8-mixed.engine \
  --retain-fp16 model.layers.0.mlp.down_proj.weight

PYTHONPATH=python python -m forge_llm.validate_int8 \
  --source models/qwen2.5-0.5b.engine --model models/qwen2.5-0.5b-int8-mixed.engine \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct --int8-mode reconstruct \
  --output benchmark-results/qwen-mixed-quality.json
```

`validate_int8` compares against the same FP16 artifact: 25 prompts, 32 greedy
tokens each, plus teacher-forced logit cosine at fixed positions. It exits nonzero
when the comparison fails.

## What was tried

All on an M3 Max with MLX 0.32.2, comparing against the FP16 artifact on 25 held-out
prompts (32 greedy tokens each). The bar was an exact match on all 25 and logit
cosine ≥ 0.999. Nothing cleared it.

| Approach | Qwen2.5-0.5B exact | Gemma 3 1B exact | Notes |
| --- | ---: | ---: | --- |
| Per-channel INT8, every projection | 15/25 | 17/25 | Min logit cosine 0.9917 / 0.9964 |
| Mixed precision, ~25% of projection bytes INT8 | 19/25 | 22/25 | Cosine 0.99985 / 0.99964 |
| Block second-order fitting (GPTQ-style, block-diagonal) + joint policy search | 19–22/25 | 19–23/25 | 44–48% lower local reconstruction loss than RTN, same token quality |
| Calibrating through the real paged decode path, with policy repair | 19–22/25 | 19–21/25 | Gemma cosine fell to 0.998 on one corpus |
| Activation-weighted scale search | — | — | 25/32 and 29/32 on calibration prompts, so held-out wasn't run |
| Integer coordinate refinement on top of scale search | — | — | Qwen matched 32/32 on calibration; Gemma's cosine was 0.99875, so stopped |
| Qwen-only held-out run of the refined policy | 21/25 | — | 6 of 800 teacher-forced decisions changed |

Ranges cover two prompt sets and the different execution modes. Lower
reconstruction loss didn't turn into better greedy agreement in any of these.
Several of the inspected flips were near-ties, with the top two FP16 logits
within 0.016 of each other, so a different reduction order alone can change the
pick.

Storage and speed for the all-INT8 artifacts:

| | Qwen2.5-0.5B | Gemma 3 1B |
| --- | ---: | ---: |
| File size, FP16 → INT8 | 988 MB → 632 MB (−36%) | 2.00 GB → 1.30 GB (−35%) |
| Peak MLX memory, 128-token prompt | 1.11 → 0.96 GiB | 2.17 → 1.69 GiB |

| Qwen2.5-0.5B, generated tok/s | FP16 | INT8 `metal` | INT8 `dequantize` |
| --- | ---: | ---: | ---: |
| 128 prompt, 1 request | 122.9 | 118.2 | 40.1 |
| 128 prompt, 8 requests | 410.0 | 342.0 | 184.0 |
| 1024 prompt, 1 request | 31.0 | 30.5 | 21.0 |
| 1024 prompt, 8 requests | 86.0 | 78.6 | 63.9 |

The custom kernel is up to 3× faster than naive dequantization for short single
requests, but FP16 is still faster, especially with concurrent requests where
MLX's FP16 GEMM wins.
On this model size, INT8 only buys memory.

The calibration code behind these experiments (sensitivity ranking, second-order
fitting, cached-decode calibration, scale search, refinement), the scripts that
ran them and their raw output have been removed since none of it made it past
the experiment stage. It's all in git history: `git checkout 2bd1005 -- python
benchmarks tests` brings it back.

## Artifact format v3

Versions 1 and 2 are still readable, and FP16 exports still write version 2.
Version 3 adds dtype 4 (signed INT8) and, after the tensor table, a `uint32`
descriptor count followed by one descriptor per quantized weight:

| Field | Encoding |
| --- | --- |
| Weight name length | uint16 |
| Scheme | uint8, 1 = symmetric per-output-channel INT8 |
| Axis | uint8, must be 0 |
| Scale name length | uint16 |
| Weight name, scale name | UTF-8, no terminator |

Every INT8 tensor needs a descriptor, and every scale must be a positive, finite
FP32 vector with one entry per output channel. `-128` is rejected. Checksums cover
descriptors and scales, and the Python and C++ readers share the same checks.
