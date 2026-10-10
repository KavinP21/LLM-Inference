# Runtime architecture

Forge LLM separates model artifacts and request orchestration from device execution. CUDA and MLX
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

The Apple backend implements complete Qwen2 and Gemma 3 text graphs with MLX operations. A
validated artifact selects a family adapter; the engine, scheduler, and physical K/V owner depend
only on the adapter protocol. Qwen uses biased projections, standard RMSNorm, SiLU gating, and a
uniform RoPE base. Gemma uses an explicit attention width, bias-free projections, head-wise Q/K
normalization, offset RMSNorm, scaled embeddings, tanh-GELU, four norms per block, two RoPE bases,
and a regular five-local/one-global attention pattern.

Model
weights are copied from read-only memory-mapped NumPy views and materialized before inference.
Aliased tied embeddings share one MLX array. FP32 is used for RMSNorm statistics, attention scores,
softmax, and returned logits; activations remain FP16. FP16 is the validated weight path; optional
version-3 artifacts store internal projections as signed INT8 plus FP32 per-output-channel scales.
Small decode batches use a fused Metal W8A16 linear kernel; prefill reconstructs ephemeral FP16
matrices for MLX GEMM. No dense second model is retained. The INT8 quality checkpoint remains
experimental; see [its contract and limitations](quantization.md).

Mixed-precision version-3 artifacts preserve sensitive projections in their original FP16 form.
An offline, source-bound policy chooses the partition; inference reads tensor descriptors only.
`int8_mode="reconstruct"` uses one-pass Metal unpacking and native GEMM for decode as well as prefill,
preserving the composed numerical path at the cost of temporary dense matrices and extra launches.
The original direct small-batch kernel remains independently selectable and experimental.

Offline quantization now has a [cached-decode calibration contract](cached-calibration.md):
private reservation-aware teacher forcing, source prompt/decode activation sampling, and bounded
joint policy repair. A repair must improve the full calibration score without crossing the 25%
eligible-byte floor. Statistics/search policies are never loaded by inference. Regression observes
every cached continuation position; whole-prompt prefill probes remain available for legacy runs.

The cache allocator assigns stable physical IDs in 16-token units. Each physical block owns an
independently replaceable MLX layer-page with shape
`[K_or_V][token_in_block][kv_head][head_dimension]` for every materialized layer. A per-request
logical table routes positions to those arrays. Pages materialize lazily, reject overwrites, and
return to a shared free list on completion, cancellation, or failure. Admission reserves worst-case
block capacity so an accepted request cannot be evicted halfway through generation.

Opt-in FP16 [prefix caching](prefix-cache.md) replaces private-page ownership with
`SharedKVBlockPool` and a bounded `PrefixCache`. Immutable snapshots pin physical pages
independently of requests. Multiple logical tables may share one page; admission counts
unique pages plus future private and partial-tail COW reservations. A shared tail is cloned
before append, and only the final reference returns its ID to the free list. The device store
also rejects direct shared-page writes. Cache eviction/clear drops pins, not active references.
This is engine-local and namespace-scoped; CUDA and INT8 do not inherit it.

Cache lookup retains original prefill chunk shapes: exact full-prompt snapshots can bypass
prefill, while changed suffixes reuse only original chunk boundaries. Each snapshot copies one
final FP32 logit row to bounded host memory rather than retaining a whole GPU vocabulary matrix.
Execution identity binds artifact/configuration and kernel/numerical settings; changing settings
requires another engine. The default remains disabled, including in the desktop application.

Prompt work is limited to one configurable chunk per scheduler iteration. For supported production
head dimensions, prefill uses MLX fused causal GQA attention; the exact online-softmax tiled path is
kept as a bounded-workspace reference and small-shape/soft-cap fallback. Gemma sliding layers gather
only the page range visible to the current chunk, while global layers retain full-prefix attention.
Decode combines embeddings, projections, attention, and MLPs for all runnable sequences in one
batched model call.

`decode_mode="rowwise"` optionally splits only projection reductions (including the tied head)
into independent single-row operations. The scheduler, norm/activation chains, page writes, and
paged attention remain batched; prefill and single-request numerics are unchanged. This addresses
native FP16 GEMM batch-shape drift at a measured dispatch/weight-reuse cost. Reconstruction modes
share one temporary matrix among the row GEMVs. The default remains `batched`; see
[the numerical contract](decode-numerics.md).

Three Qwen custom-Metal chains reduce elementwise launches and intermediate traffic: residual add
plus RMSNorm, RoPE plus physical layer-page write, and SiLU-times-up. Gemma safely reuses the
RoPE/page-write chain but keeps its different norm and activation topology in MLX. Decode attention
packs the live
layer-pages once for the launch, remaps logical block tables into that tensor, and runs one
simdgroup per request/query-head with online FP32 softmax. This avoids per-request logical-history
concatenation and longest-sequence padding. It is not a zero-copy page pool: MLX's immutable array
contract requires the layer-page stack, which remains a measured single-request cost.

## Speculative execution

The [greedy speculative controller](speculative-decoding.md) owns one request and a private paged
cache. Prompt/history n-grams supply proposals without another model; an optional compatible draft
model has its own cache. The target evaluates the pending output token and proposals in one causal
`forward_paged_chunk` call. Matching proposals are accepted, followed by the target correction or
bonus token. Rejected K/V slots lose both their occupancy masks and underlying values before reuse.
Context, output and KV budgets apply before execution; completion, cancellation and failure release
request caches.

Native FP16 block projections and attention can use different reductions from one-token decoding.
The block policy therefore has an explicit numerical contract rather than a universal canonical
token-parity guarantee. A sequential verification mode and no-draft baseline use ordinary decode
calls. The existing continuous scheduler and prefix-cache snapshots are not shared with this
standalone decoder. CUDA verification, probabilistic speculation and multi-request speculative
batching remain separate implementation work.

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
                       MlxEngine / CUDA Engine / MLX speculation / optional MLX-LM
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
checks distinguish an agent's final response from a verified criterion. Control tests and local
worker execution do not establish general task quality; [model qualification](agent-results.md)
is tracked separately.

Worker HTTP generation jobs are implemented. Streaming chat sessions, an OpenAI-compatible API,
automatic cluster discovery/placement, NCCL tensor parallelism and physical multi-host/GPU scaling
qualification remain outside that transport contract. Each current replica must fit its complete
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
