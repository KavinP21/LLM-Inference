# Models

Instinct uses two models, both run by its own MLX engine:

| | Gemma 4 26B A4B | Qwen3.5-4B |
| --- | --- | --- |
| Role | Driver: every chat and root agent | Sub-agents |
| Checkpoint | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (QAT 4-bit) | `mlx-community/Qwen3.5-4B-4bit` |
| Architecture | MoE, 128 experts / 8 active (~4B active params), 25 sliding-window + 5 global layers | Hybrid: 24 Gated DeltaNet + 8 attention layers |
| Memory | 14.5 GiB | 2.8 GiB |
| Decode on an M3 Max, one user | ~65 tok/s | ~76 tok/s |
| Three users at once, each | ~49 tok/s | ~53 tok/s |
| Inputs | Text, images | Text |

```bash
pip install -e '.[mlx,agents]'
python -m instinct.prepare_models    # ~17 GB into models/, pinned revisions
```

The adapters (`backends/gemma4.py`, `backends/qwen35.py`) load the checkpoints'
safetensors directly and keep the 4-bit weights packed. `tests/test_native_checkpoints.py`
compares them with mlx-vlm when it's installed (`pip install -e '.[reference]'`).

## Using a model directly

```python
from instinct.chat_runner import ChatRunner

with ChatRunner("models/gemma-4-26b-a4b-mlx") as runner:
    result = runner.generate(
        [{"role": "user", "content": "What's in this picture?",
          "media": [{"type": "image", "path": "/absolute/path/photo.png"}]}],
        max_tokens=512,
        effort="low",                   # off | low | medium | high
    )
    print(result.text)                  # the answer
    print(result.reasoning)             # the thinking, if effort isn't "off"
    print(result.metrics)               # ttft_ms, decode_tps, reused_tokens, ...
```

`generate` can be called from many threads at once; one engine thread batches
everything in flight. Pass the whole conversation each time. The runner
recognizes a conversation it has seen and only processes the new messages.

**Effort.** `off` disables thinking and decodes greedily. `low` and `medium` let
the model think for up to 1,024 and 4,096 tokens, after which the runner closes
the thinking block for it. `high` is limited only by `max_tokens`. With thinking
on, sampling follows each model card: Gemma 1.0/0.95/64 and Qwen 0.7/0.8/20 for
temperature/top-p/top-k.

**Tool calls.** Pass OpenAI-style tool specs as `tools=`. Each model answers in
its own trained format: Gemma uses `<|tool_call>call:name{...}<tool_call|>`,
and Qwen3.5 uses `<tool_call><function=name>...`. The agent runtime parses both,
and a reply without a tool call counts as the final answer.

**Images.** Gemma 4 encodes images with its own vision tower
(`backends/gemma4_vision.py`): up to 280 soft tokens per image, attending to
each other in both directions. Agent workers send image bytes with the request
(8 MiB per item, 16 MiB per conversation) keyed by SHA-256, so a follow-up about
the same image reuses the cached prefix instead of encoding it again.

**8-bit K/V.** A worker's `"engine_options": {"kv_bits": 8}` stores keys and
values as int8 with one fp16 scale per token and head, which halves the K/V
memory a context needs. The paged decode kernel applies the scales as it reads.
On an 8,000-token prompt the last-position logits keep a cosine of 0.9990
(Gemma) and 0.9994 (Qwen) to 16-bit, the top token matches, and greedy output
stays identical for the first 39 of 64 tokens, which is about as far as bf16
rounding alone carries it. Qwen decodes at the same speed; Gemma is about 10%
slower at that length, from quantizing on 30 layers each step. The app's
profile uses it for the Qwen sub-agents, so 8 GiB holds two full 256K contexts.

The engine also runs FP16 Qwen2 and Gemma 3 `.engine` files.

## Limits

- The Gemma driver batches up to four requests (several chats at once) and the
  Qwen sub-agent worker up to eight; the Qwen worker starts on first use and
  stops after ten idle minutes.
- Gemma's MTP drafter isn't wired into the engine yet, so Gemma decodes without
  speculation. Qwen3.5's vision tower isn't ported, so sub-agents take text only.
- Both models run with their full 262,144-token context, but prefilling a long
  prompt is slow: 29 minutes for 262K tokens on Gemma (see
  [benchmarks](benchmarks.md#256k-context)). Gemma 4's audio input hasn't been
  tried here.
- Nothing here runs on CUDA.
