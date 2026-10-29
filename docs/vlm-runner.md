# MLX-VLM runner (Gemma 4, Qwen3.5)

`instinct.vlm_runner.VlmRunner` runs any checkpoint mlx-vlm can load. Its forward
pass is mlx-vlm's; what the runner adds is prompt reuse, which matters far more
than raw decode speed for an agent. Every agent step re-sends the whole
conversation, so a naive server re-prefills thousands of tokens per step.

```bash
python -m instinct.prepare_models     # Gemma 4 26B A4B (QAT 4-bit) + drafter, Qwen3.5-4B
```

```python
from instinct.vlm_runner import VlmRunner

runner = VlmRunner("models/gemma-4-26b-a4b-mlx",
                   draft_path="models/gemma-4-26b-a4b-assistant")
result = runner.generate([{"role": "user", "content": "Hi"}], max_tokens=256,
                         effort="off")   # off | low | medium | high
print(result.text, result.metrics)
```

Agent workers use it with `"runner": "vlm"` (see
[examples/agents-gemma4.json](../examples/agents-gemma4.json)).

## How prompts are reused

**Sessions.** After a reply, the runner keeps the KV cache and the exact token
stream the model saw and produced. When the next request is the same
conversation plus new messages, it renders only the new part and appends those
tokens. Re-rendering the whole conversation would not reproduce the cached tokens.
Qwen3.5's template drops the empty `<think>` block from earlier turns, for
example. Appending also works for caches that can't be rolled back: Gemma's
sliding-window layers and Qwen3.5's recurrent state.

**Snapshots.** A prompt that doesn't continue a session is prefilled up to its
last stable boundary (everything before the final message), and the cache is
copied there. A new chat message, or a sibling sub-agent with the same system
prompt and tools, starts from the longest snapshot that prefixes it. Restoring
is bit-exact; a copy takes 8–23 ms.

For this to work the agent runtime keeps prompts append-only. The tool catalog
is the same every turn, and status notes are stored as messages instead of a
rewritten trailing menu.

**Drafting.** Gemma 4 ships a 4-layer MTP "assistant" drafter. On the 26B A4B
MoE a 3-token verify round costs about 2.4 single decode steps, since it touches
more experts. Drafting therefore helps tool calls (+12%), edits (+7%) and code
(+5%) but hurts free prose (−14%). The runner only drafts on greedy agent turns
and backs off for a few requests when the measured tokens per round fall below
break-even.

**Reasoning effort.** `off` disables thinking. `low` and `medium` cap it at 1,024
and 4,096 tokens, forcing the thinking block closed at the budget. `high` is
bounded only by `max_tokens`.

## Measurements

Apple M3 Max, MLX 0.32.2, mlx-vlm 0.7.6, `benchmarks/bench_runner.py`
(1.6K-token system prompt, medians of 3):

| | Qwen3.5-4B (4-bit) | Gemma 4 26B A4B (QAT 4-bit) |
| --- | ---: | ---: |
| Weights in memory | 2.8 GiB | 14.5 GiB (+0.8 GiB drafter) |
| Load time | 1.2 s | 1.7 s |
| First token, cold prompt | 945 ms | 1,346 ms |
| First token, follow-up turn | 57 ms | 84 ms |
| First token, new chat with the same system prompt | 63 ms | 99 ms |
| Decode | 109 tok/s | ~85 tok/s |
| Prefill | ~1,700 tok/s | ~1,200 tok/s |

Gemma's prefill is close to compute-bound (all experts are active over a long
prompt), which is why avoiding prefill matters more than anything else. Before
this runner, Qwen3.5 went through a hand-written decode loop that synchronized
with the GPU twice per token and rebuilt its cache for every request: about 80
tok/s, and a full re-prefill on every agent step.

`tests/test_vlm_runner.py` runs against the real checkpoints when they're
downloaded.
