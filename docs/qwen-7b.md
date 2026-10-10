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

The desktop app keeps the 0.5B default and adds **Qwen2.5-7B-Instruct ·
reconstructed FP16** to its model picker. That profile uses the reconstructed
artifact above, the matching cached tokenizer, a 32K context and 2 GiB KV budget.
Switching models clears the conversation and closes the previous engine before
loading the next. Loading, generation and cleanup belong to one worker thread;
closing the window cancels at the next scheduler boundary and reclaims its model.
Tokenizers are loaded offline; missing artifacts/tokenizers produce a load error.

For other native artifacts, use `create_engine` with their matching tokenizer
and an explicit KV budget. Native speculative decoding remains opt-in under its
[separate numerical contract](speculative-decoding.md).

## Executed checks and task limits

The [883-test regression](../benchmarks/results/qwen7b-2026-10-10/regression-summary.json)
passes without skips, including Metal references and authenticated loopback
transport. An isolated validation interpreter used the installed dependency
files without editable import hooks; imports were checked against the tested
checkout. No MLX or MLX-LM upgrade was required.

The [actual desktop check](../benchmarks/results/qwen7b-2026-10-10/native-chat.json)
loads the default 0.5B model, switches to the native reconstructed 7B profile,
remembers a name across two turns, retains the failed-load status after Clear
chat, and closes its model worker. The 32K profile is a configured capacity;
this chat check uses short prompts and does not establish 32K semantic quality.

The [adapter reference](../benchmarks/results/qwen7b-2026-10-10/adapter-reference.json)
matches direct pinned MLX-LM greedy generation on all **66 output tokens across
three prompts**, including an 8,163-token synthetic prompt. Each request uses
a fresh cache, with the same loaded quantized weights shared sequentially.
The long case peaks at 484,442,112 persistent KV bytes. Completion and explicit
cancellation reclaim reservations and cache buffers. This checks the adapter;
it is not an independent original-FP16-weight or performance comparison.

The [task evaluation](../benchmarks/results/qwen7b-2026-10-10/tasks-final.json)
ran the existing coding/document fixtures from a clean checkout at `6efd1d4`,
with unchanged runtime sources and two actual 7B replicas. Startup took 9.57 s,
excluded from task times:

| Task | Single agent | Delegated | Independent gate |
| --- | --- | --- | --- |
| Python repair | 105.97 s, 20 calls | 260.09 s, 41 calls, 2 children | Both fail. Repeated invalid Python is rejected; the step budget ends the run. |
| Document analysis | 8.87 s, 2 calls | 35.81 s, 9 calls, 2 children | Single passes all eight facts. Delegation incorrectly claims release readiness. |

Coding gates use immutable operator-owned tests and four further inputs withheld
until final grading. Document completion checks structure/workflow; independent
grading checks factual values. A completed run or `completion_verified` flag
does not waive a failed factual gate.

The [first 8K-profile trial](../benchmarks/results/qwen7b-2026-10-10/tasks-initial.json)
is retained. Its coding catalog exceeded the conservative context budget before
dispatch; the corrected 16K example and a preflight regression address that
configuration issue. Its delegated document answer also failed factual grading.

**This cached 7B Instruct checkpoint is supported for inference and chat, but it
has not passed the autonomous coding or delegated document gates.** Use the
previously qualified Qwen3 configuration for those agent tasks and require task
acceptance criteria. Coder/base variants have matching config geometry but were
not executed here; no general task-quality or CUDA/multi-host claim is made.
