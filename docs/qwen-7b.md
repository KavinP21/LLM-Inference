# Qwen 7B models

Forge's native Qwen2 engine accepts dense Qwen2/Qwen2.5 7B FP16 artifacts.
The optional MLX-LM worker also accepts local affine 4-bit checkpoints with the
validated 28-layer, 3,584-hidden, 28-query-head and 4-KV-head architecture.
Context limits come from the model configuration, and each checkpoint retains
its own EOS ID. The tokenizer's advertised context can exceed the model's limit.

The Qwen2.5 [7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json),
[7B base](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json),
[Coder Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/config.json)
and [Coder base](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/raw/main/config.json)
configs share this attention geometry. Base-model prompting is separate from
instruction/tool prompting; the supplied agent and desktop profiles use Instruct.
The cached Instruct checkpoint is the local execution target. Compatibility of
the other variants' geometry does not establish their task quality.

## Local 4-bit task agents

Install the optional runner as described in [agent setup](agent-runtime.md).
Use an existing local MLX affine 4-bit/group-64 checkpoint directory:

```bash
ln -s /path/to/local/qwen2.5-7b-instruct-4bit models/qwen2.5-7b-mlx
PYTHONPATH=python .venv/bin/python -m forge_llm.agents.cli \
  --state-dir .forge run --config examples/agents-qwen2.5-7b-mlx-lm.json \
  --workspace . --task 'Read the architecture and identify a resource limit. Cite the source file.'
```

The example starts two independently supervised 7B replicas, each with an
16,384-token context and 1 GiB persistent KV budget. Logical agents share those
replicas. Add `--allow-write --allow-tests` for coding tasks. Completion checks,
child scope, recovery and bounded actions are the same as for other task models.
The coordinator's conservative input reservation is capped at 12,000 tokens,
leaving room for the coding tool catalog and independent-review instructions.

Persistent full KV costs **57,344 bytes/token** for this model: 448 MiB at 8K,
896 MiB at 16K and 1.75 GiB at 32K. Buffers round up to 256-token increments.
Weights and temporary arrays require additional memory. The runner validates
actual cache counts, dimensions and dtypes, and rejects enabled sliding-window
attention. A disabled `sliding_window` metadata value is accepted.

This path uses pinned upstream MLX-LM 0.28.3 execution. Forge kernel, prefix-cache
and speculative settings remain unsupported by this optional runner.

## Native engine and desktop chat

For an already local MLX Qwen2 checkpoint, reconstruct its quantized weights into
Forge's FP16 artifact format:

```bash
forge-import-mlx /path/to/local/qwen2.5-7b-instruct-4bit \
  models/qwen2.5-7b-mlx-reconstructed.engine
```

The resulting artifact represents the quantized model; it does not recover
original FP16 weights. Original FP16 weights can instead be exported with
`forge-export Qwen/Qwen2.5-7B-Instruct models/qwen2.5-7b.engine`.
Model weights and checkpoint links are excluded from Git.

The desktop app lists **Qwen2.5-7B-Instruct · reconstructed FP16** in its model
picker, after the default Qwen3.5-4B and Qwen2.5-0.5B. That profile uses the reconstructed
artifact above, the matching cached tokenizer, a 32K context and 2 GiB KV budget.
Switching models clears the conversation and closes the previous engine before
loading the next. Loading, generation and cleanup belong to one worker thread;
closing the window cancels at the next scheduler boundary and reclaims its model.
Tokenizers are loaded offline; missing artifacts/tokenizers produce a load error.

For other native artifacts, use `create_engine` with their matching tokenizer
and an explicit KV budget. Native speculative decoding remains opt-in under its
[separate numerical contract](speculative-decoding.md).

## What's been checked

- **Adapter vs. MLX-LM:** the MLX-LM worker matches direct MLX-LM greedy
  generation on all 66 output tokens across three prompts, including an
  8,163-token one, and releases its KV cache on completion and on cancel.
- **Desktop chat:** loading the reconstructed 7B profile, switching models,
  a two-turn conversation, a failed load and closing the window all work. Those
  were short prompts, so the 32K setting is a configured capacity, not something
  that's been tested for quality.
- **Agent tasks:** with two 4-bit replicas, single-agent document analysis passes.
  Both coding modes fail (it keeps producing invalid Python until the step budget
  runs out), and the delegated document run wrongly reports the release as ready.
  See [agent-runtime.md](agent-runtime.md#how-real-models-did).

So the 7B model is fine for inference and chat, but use Qwen3.5-4B or Qwen3 for
agent work. The Coder and base variants share the same config geometry but
haven't been run.
