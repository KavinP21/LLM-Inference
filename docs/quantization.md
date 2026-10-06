# Weight-only INT8 (experimental)

Forge can convert either supported FP16 checkpoint to a portable W8A16 artifact and run it on
MLX/Metal. INT8 is opt-in by selecting that artifact; ordinary FP16 loading/generation is unchanged.
It is **not a passed quality checkpoint**: the initial real-model regression misses the existing
0.999 logit-cosine and exact-greedy gates. Do not advance to W4A16 or present it as lossless.

## Conversion and use

The converter uses NumPy only, works on macOS/Linux, and refuses to overwrite an existing artifact
or its FP16 source. It validates the result before publishing it and records the source checksum.

```bash
PYTHONPATH=python .venv/bin/python -m forge_llm.quantization \
  models/qwen2.5-0.5b.engine models/qwen2.5-0.5b-int8.engine
PYTHONPATH=python .venv/bin/python -m forge_llm.quantization \
  models/gemma-3-1b-it.engine models/gemma-3-1b-it-int8.engine
```

```python
from forge_llm import create_engine

with create_engine("models/qwen2.5-0.5b-int8.engine", backend="mlx",
                   max_model_length=32768, kv_cache_bytes=512 << 20,
                   int8_mode="metal") as engine:
    output_ids = engine.generate([1, 2, 3], max_new_tokens=32)
    print(engine.stats())
```

Use Gemma's own tokenizer and at least 1,024 MiB K/V capacity for its full 32K resource gate.
The context limit and reservation-aware scheduler are unchanged; INT8 does not quantize the cache.
CUDA explicitly rejects INT8 artifacts until a separately validated NVIDIA implementation exists.

## Numerical contract

For each `[output_channel, input_channel]` projection row:

```
scale = max(abs(weight)) / 127       # FP32; all-zero rows use scale=1
q = clip(round_to_nearest_even(weight / scale), -127, 127)
reconstructed_weight = FP16(FP32(q) * scale)
```

Only attention Q/K/V/O and MLP gate/up/down weights are quantized. Embeddings, the tied output
head, biases, normalization weights, activations, and K/V remain FP16. Scales remain FP32. There is
no clipping calibration, activation quantization, GPTQ/AWQ calibration, or groupwise quantization.
Projection storage approaches half the FP16 size, but whole-model savings are smaller because the
large vocabulary embedding/head is retained and tied aliases remain shared.

`int8_mode="dequantize"` reconstructs a temporary FP16 matrix before each MLX matmul. This is the
simple numerical/performance baseline; it never stores a second permanent dense model.
`"metal"` and `"auto"` use the custom W8A16 kernel for 1–16 activation rows and the dequantized
GEMM path for larger prefill batches, using a one-pass custom reconstruction kernel that avoids
full-sized FP32 cast/multiply intermediates. Four output channels share a 128-thread group. One SIMD
group accumulates a channel in FP32, vectorizes four contiguous elements per lane, and reuses each
loaded weight across the small batch's rows. Odd-width tails use scalar loads. This reduces weight
reads and removes full-matrix reconstruction during decode; it is not an INT8 tensor-core GEMM.
`int8_mode` controls these quantized operators independently of `custom_metal`, which controls the
other transformer fusions. Use `int8_mode="dequantize"` explicitly for an all-composed INT8 baseline.

`int8_mode="reconstruct"` uses the one-pass Metal reconstruction kernel and **native MLX GEMM at
every row count**, including decode. This preserves the composed baseline's reduction order without
its full-sized FP32 reconstruction intermediates. It still creates a temporary dense FP16 matrix
for every quantized projection. It is an accuracy-oriented alternative, not a faster direct kernel
or a fix to the experimental direct kernel's near-tie decisions. All modes keep stored weights INT8
and do not retain a second permanent dense model.

## Calibration and explicit mixed precision

The offline calibration workflow ranks **whole projections**, keeping sensitive matrices in FP16.
It does not fit clipping scales, transform activation channels, perform GPTQ/AWQ, or train weights.
The per-output-channel quantizer above remains byte-for-byte unchanged for every INT8 matrix.

```bash
PYTHONPATH=python .venv/bin/python -m forge_llm.calibrate_int8 \
  --source models/qwen2.5-0.5b.engine --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --output benchmark-results/qwen-calibration.json

PYTHONPATH=python .venv/bin/python -m forge_llm.quantization \
  models/qwen2.5-0.5b.engine models/qwen2.5-0.5b-int8-mixed.engine \
  --policy benchmark-results/qwen-calibration.json

PYTHONPATH=python .venv/bin/python -m forge_llm.validate_int8 \
  --source models/qwen2.5-0.5b.engine --model models/qwen2.5-0.5b-int8-mixed.engine \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct --int8-mode reconstruct \
  --output benchmark-results/qwen-mixed-held-out.json
```

Calibration uses the separately frozen 16-prompt `benchmarks/calibration-prompts.json`. It refuses
direct overlap with the 25-prompt regression after Unicode/case/whitespace normalization and after
tokenization. Held-out outputs never influence ranking or policy selection. Overlap guards do not
prove semantic independence or prevent all forms of benchmark overfitting.

The tool measures single-matrix reconstructed INT8 perturbations on eight deterministic,
teacher-forced full-model probes. Ranking prioritizes changed top-1 decisions, then mean squared
relative logit error per eligible FP16 byte. This is a heuristic: interactions are not inferred from
single-matrix scores. It tests the combined policy on all 16 calibration prompts, generating 32
tokens and comparing teacher-forced logits at positions 0/7/31. Candidates retain at least
100%, 75%, 50%, or 25% of eligible projection bytes in INT8, with whole-matrix rounding. The first
passing candidate is frozen; if none passes, the final 25%-floor candidate is saved as **failed**.
Calibration exits nonzero on failure, and the minimum floor cannot be changed through the CLI.

Policies partition every supported projection into FP16 or INT8, bind to the exact source data
checksum, and include a canonical policy checksum and both corpus hashes. The converter validates
the policy, refuses a below-25% policy, and records the exact retained projection list, actual
quantized byte fraction, and policy in its manifest. A checksum is integrity/provenance, **not a
quality certificate**. Converting a saved failed policy is allowed for explicitly measured research
experiments; no such conversion changes the runtime's experimental status or FP16 defaults.

For manual experiments, repeat `--retain-fp16 EXACT_PROJECTION_NAME` instead of `--policy`.
Unknown/duplicate names, retaining every projection, source mismatch, malformed partitions, and
overwrite/requantization attempts are rejected. Mixed artifacts use the existing version-3 format;
no sidecar is needed during inference. Engine stats report retained matrices and the actual INT8
fraction of eligible projection bytes. The fraction is **not whole-model memory savings**.

During calibration only, temporary FP16 reconstructed copies are used to isolate numerical
perturbations efficiently. That tool's memory/time is not inference performance. Fresh held-out,
32K, and isolated-process benchmark runs are required for the exported artifact. Current failed
and passing sub-gates, tradeoffs, and raw evidence are in
[the hardening report](mlx-int8-hardening-results.md).

Batch-shape consistency is a separate contract. `decode_mode="rowwise"` preserves independent
projection reductions while retaining batched attention, in FP16 and INT8. Its
[exact numerical/resource gates pass](mlx-batch-numerics-results.md), but frozen mixed-policy
cross-artifact quality still matches only 19/25 Qwen and 22/25 Gemma cases. Do not treat this as a
lossless quantization fix. The throughput-oriented default remains unchanged.

MLX still allocates immutable result arrays and packs live K/V pages. Do not apply the CUDA
no-token-loop-allocation claim to this backend. `close()` drops resident weight ownership even if
the caller keeps the closed engine object; MLX's freed-buffer cache is distinct from active memory.

## Artifact version 3

Versions 1 and 2 remain readable. Unquantized exports still use version 2. Version 3 keeps the
version-2 configuration and tensor table and adds dtype ID 4 for signed INT8. After the tensor
table comes a little-endian `uint32` descriptor count, followed by descriptors:

| Field | Encoding |
| --- | --- |
| Weight-name byte count | uint16 |
| Scheme | uint8; 1 = symmetric per-output-channel INT8 |
| Axis | uint8; must be 0 |
| Scale-name byte count | uint16 |
| Weight name, then scale name | UTF-8 bytes, no terminators |

Names identify existing tensors. Each weight must be rank-2 signed INT8, every INT8 tensor must
have a descriptor, and each scale tensor must be FP32 `[output_channels]` with positive finite
values. `-128` is invalid. Checksums cover descriptors and scales, offsets remain 256-byte aligned,
and bounds/shape/overlap/alias checks run before execution. Python and C++ consume the same format.
Model contracts allow INT8 only for the seven known projections of existing layers.

## Reproduce correctness and measurements

```bash
PYTHONPATH=python .venv/bin/python -m forge_llm.validate_int8 \
  --source models/qwen2.5-0.5b.engine --model models/qwen2.5-0.5b-int8.engine \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct --output benchmark-results/qwen-int8-quality.json

PYTHONPATH=python .venv/bin/python benchmarks/run_int8_matrix.py \
  --source models/qwen2.5-0.5b.engine --model models/qwen2.5-0.5b-int8.engine \
  --output-dir benchmark-results/qwen-int8 --warmups 5 --repetitions 3

MTL_CAPTURE_ENABLED=1 PYTHONPATH=python .venv/bin/python -m forge_llm.profile_int8 \
  --model models/qwen2.5-0.5b-int8.engine --output benchmark-results/qwen-int8-linear.json \
  --capture profiles/qwen-int8-linear.gputrace
```

The quality tool checks source identity, every projection's numerical error, 25 fixed prompts with
32 raw greedy tokens, and teacher-forced logits at positions 0/1/7/15/31. It separately reports
quantization error and fused-versus-dequantized error, saves failed results, and exits nonzero if
the strict checkpoint fails. It compares to the identical Forge FP16 artifact, not a new claim of
Transformers certification. Token agreement on a small corpus is not a task-quality or perplexity
study.

The matrix uses separate processes, identical synthetic token-ID workloads, five **full-output**
warmups, three recorded trials, and rotated implementation order. It records throughput, TTFT,
TPOT, allocator peaks, cache cleanup, both artifact checksums, and dirty-source provenance. Do not
run other project GPU tests alongside it. Captured runs and diagnostic pilots are not speed claims.
MLX active-memory peaks are not process RSS or total system memory. Apple GPU utilization/power
are unavailable in this runner and remain null, not invented NVIDIA readings.

The kernel API is documented in [MLX's custom Metal kernel guide](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
Current measurements and the failed quality gate are in [the INT8 result report](mlx-int8-results.md).
