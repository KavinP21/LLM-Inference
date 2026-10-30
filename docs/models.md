# Models

Studio and the agent runtime use two models:

| | Gemma 4 26B A4B | Qwen3.5-4B |
| --- | --- | --- |
| Role | Driver: every chat and root agent | Sub-agents |
| Checkpoint | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (QAT 4-bit) | `mlx-community/Qwen3.5-4B-4bit` |
| Architecture | MoE, 128 experts / 8 active (~4B active params), 25 sliding-window + 5 global layers | Hybrid: 24 GatedDeltaNet + 8 attention layers |
| Memory | 14.5 GiB + 0.8 GiB MTP drafter | 2.8 GiB |
| Decode on an M3 Max | ~85 tok/s | ~109 tok/s |
| Inputs | Text, images | Text, images, short video |

Both run through [the VLM runner](vlm-runner.md), which wraps mlx-vlm 0.7.6 and
adds prompt reuse, so neither uses Instinct's native Qwen2/Gemma 3 graph.

```bash
pip install -e '.[mlx,vlm,agents]'
python -m instinct.prepare_models    # ~18 GB into models/, pinned revisions
```

## Using a model directly

```python
from instinct import create_engine

with create_engine() as engine:     # Gemma 4; pass a path for another checkpoint
    result = engine.generate(
        [{"role": "user", "content": "What's in this picture?",
          "media": [{"type": "image", "path": "/absolute/path/photo.png"}]}],
        max_tokens=512,
        effort="low",                   # off | low | medium | high
    )
    print(result.text)                  # the answer
    print(result.reasoning)             # the thinking, if effort isn't "off"
    print(result.metrics)               # ttft_ms, decode_tps, reused_tokens, ...
```

Pass the whole conversation each time. The runner recognizes a conversation it
has seen and only processes the new messages.

**Effort.** `off` disables thinking and decodes greedily. `low` and `medium` let
the model think for up to 1,024 and 4,096 tokens, after which the thinking block
is closed for it. `high` is limited only by `max_tokens`. With thinking on,
sampling follows each model card: Gemma 1.0/0.95/64 and Qwen 0.7/0.8/20 for
temperature/top-p/top-k.

**Tool calls.** Pass OpenAI-style tool specs as `tools=`. Each model answers in
its own trained format: Gemma uses `<|tool_call>call:name{...}<tool_call|>`,
and Qwen3.5 uses `<tool_call><function=name>...`. The agent runtime parses both,
and a reply without a tool call counts as the final answer.

**Images and video.** Media items take a local `path`. Agent workers send image
bytes with the request (8 MiB per item, 16 MiB per conversation) and key them by
SHA-256, so a follow-up about the same image reuses the cached prefix instead of
encoding it again.

## Other checkpoints

Anything mlx-vlm can load works as `"runner": "vlm"`, including text-only
models. Tested: Qwen3-30B-A3B (4-bit, 93 tok/s) and Qwen2.5-7B-Instruct (4-bit,
73 tok/s). Tool calls in the plain JSON (Hermes) format are parsed for those.

The native engine still runs FP16 Qwen2 and Gemma 3 `.engine` files with its own
paged KV cache and Metal kernels. `instinct-import-mlx` can rebuild an FP16 engine
file from an MLX 4-bit Qwen2 checkpoint (not the original weights, since the
quantization error stays).

## Limits

- One request runs at a time per worker process. Concurrent agents use more
  replicas: Studio runs two Qwen sub-agent workers, which start on first use and
  stop after ten idle minutes.
- Gemma 4's audio input and its 262K context haven't been tried here. The Studio
  configuration caps the driver at a 60K context and sub-agents at 32K.
- Nothing here runs on CUDA.
