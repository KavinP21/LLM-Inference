# Runtime architecture

Instinct separates model artifacts and request orchestration from device execution. CUDA and MLX
consume the same validated container and expose the same `submit`/`step`/`cancel`/`generate` API.

```text
                          Python token IDs
                                  │
                         create_engine(backend)
                         ┌────────┴────────┐
                         │                 │
                    MLX Engine        CUDA Engine
                         │                 │
              backend-neutral        C++ scheduler
              Python scheduler            │
                         │                 │
                 model factory        CUDA Qwen model
                  ┌─────┴─────┐             │
             Qwen adapter  Gemma adapter     │
                  └─────┬─────┘             │
                  MLX + custom Metal    cuBLASLt + kernels
                         └────────┬────────┘
                                  │
                  shared model format and test vectors
```

## MLX execution

Gemma 4 and Qwen3.5 checkpoints still run through the [VLM runner](vlm-runner.md) (mlx-vlm with
Instinct's prompt reuse) until their native adapters land. See [models.md](models.md).

`MlxEngine` is a continuous-batching engine. Each `step()` decodes one token for every running
request in a single batched forward pass, then prefills one chunk of a waiting prompt (512 tokens
while others are decoding, 2,048 when nothing else is), so a new long prompt delays the others by
at most a chunk. A request is admitted only when a sequence slot and K/V pages for its whole budget
are free, so a request that starts always finishes.

**K/V storage** (`paged_kv.py`). Every attention layer keeps keys and values in one arena of token
rows, `[rows, kv_heads, head_dim]`, and a page is 64 consecutive rows. Each sequence has a page
table per layer group, so sequences of any length share one buffer with no padding. Writes are
scatter updates that MLX applies in place. Arenas start small and double as pages are first used,
up to the configured budget. Layers with the same shape and window form a group with its own page
pool, and a sliding-window group returns pages as soon as they fall out of the window, so its
memory stays bounded however long a sequence gets. Recurrent layers keep per-sequence state in
slot arrays.

**Attention** (`batches.py`, `backends/metal_kernels.py`). Prefill gathers the sequence's rows and
runs MLX's attention with a causal (or window) mask, in query tiles when the score matrix would be
large. Decode uses a split-K paged kernel that reads the arena through the page tables: a
threadgroup per sequence, KV head and slice of the key range serves every query head of that KV
head, and a second pass merges the slices. Model adapters see one interface for both
(`rope`, `attend`, `state`), so each is written once.

**Prefix cache** (`prefix_cache.py`, opt-in). See [prefix-cache.md](prefix-cache.md).

**Sampling** (`sampling.py`). Each request brings its own temperature, top-k, top-p, min-p and
seed, so a greedy agent step and a sampled chat share a batch. `force()` makes a request emit given
tokens next, which is how thinking budgets end a thought.

The `.engine` adapter (`backends/dense.py`) runs FP16 Qwen2 and Gemma 3 files; tests compare it with
a NumPy reference and with Transformers.

## Task agents and worker replicas

The [agent runtime](agent-runtime.md) coordinates logical tasks over independent model replicas:

```text
                  task + explicitly registered workspace tools
                                     │
                          AgentRuntime ↔ SQLite journal
                                     │
                        bounded generation requests
                                     │
                                WorkerPool
                      ┌──────────────┼────────────────┐
                      │              │                │
                local actor    supervised process   configured HTTP worker
                      │              │                │
                      └──────────────┼────────────────┘
                                     │
                          worker-owned complete model
                       MlxEngine / CUDA Engine / MLX speculation / VLM runner
```

Each logical agent owns its assigned scope, conversation, inbox, children, dependencies and result.
Spawning creates journaled task state and schedules model calls on the available pool; it does not
create another weight copy or acquire another machine. An actor serializes access to its engine.
Separate workers may run calls concurrently, with independent weights and K/V capacity. Supervised
local workers use separate processes and authenticated loopback job endpoints; configured remote
workers use the same bounded submit/poll/cancel protocol. Request affinity and idempotent job IDs
support bounded retries. Placement on a selected NVIDIA device is a worker-launch setting, not a
tensor-parallel collective.

State-derived native function catalogs or the legacy action protocol restrict tool names,
arguments, actual agent IDs and child grants. Batches are limited to independent spawns or granted
replay-safe reads. File edits use per-agent observed versions; writes and test execution require
operator permission. Model output does not become an arbitrary shell command or evaluated program.
Configured tests do execute repository Python, so this is a bounded tool interface rather than an
operating-system sandbox.

The journal retains exact model inputs, token reservations, prepared actions, effects and delivered
outcomes. Coordinator leases, cancellation and conservative recovery protect those contracts;
ambiguous non-replayable effects are not silently repeated. Optional operator-selected completion
checks distinguish an agent's final response from a verified criterion. The runtime tests use scripted
models; how real models did on actual tasks is in
[agent-runtime.md](agent-runtime.md#how-real-models-did).

Worker HTTP generation jobs are implemented. Streaming chat sessions, an OpenAI-compatible API,
automatic cluster discovery/placement, NCCL tensor parallelism and multi-host scaling aren't
implemented. Each current replica must fit its complete
model on its selected device.

## CUDA KV layout

The device allocation is logically
`[physical_block][layer][K_or_V][kv_head][token_in_block][head_dimension]`.
Each sequence owns a logical block table whose entries identify physical blocks. Admission reserves
enough uncommitted capacity for the prompt plus requested output, while physical blocks are taken
from the free list only as sequence length crosses a 16-token boundary. This prevents an admitted
request from failing halfway through generation without eagerly occupying all of its blocks.

## Scheduler semantics

Both state-machine implementations return all currently decoding sequences plus at most one active
prefill. The MLX prefill remains active over multiple chunks, while every iteration still services
all runnable decodes first. Finished and cancelled requests immediately release reservations and
request ownership. In private-page mode that also frees physical pages immediately. With prefix
caching, pins may intentionally retain them until eviction, idle-cache clearing or engine close.

## Execution memory

CUDA model weights, KV blocks, activation buffers, and cuBLASLt workspace are allocated during
engine construction. A CUDA decoding iteration performs no device allocation. Linear weights retain
Hugging Face's `[out_features, in_features]` layout; both backends evaluate `X * Wᵀ`. The MLX
baseline remains selectable for numerical and performance differential tests. A future lower-level
page owner could remove the remaining layer-page stack before custom attention.
