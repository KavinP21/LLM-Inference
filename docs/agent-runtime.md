# Task agents and model replicas

Forge separates logical agents from physical model replicas. An agent owns its
task, role, conversation, inbox, children, dependencies and result. A worker
owns one loaded model and its K/V memory. Multiple agents share a bounded pool
of workers; spawning a child does not load another copy of the weights.

This supports repository work and document analysis through actual file tools,
optional edits/test execution, and parent synthesis. It is not tensor/model
parallelism: each worker must fit a complete model on its selected device.

## Install and run

```bash
pip install -e '.[mlx,agents,test]'
PYTHONPATH=python .venv/bin/python -m forge_llm.agents.cli \
  --state-dir .forge run --config examples/agents-local.json --workspace . \
  --task 'Read the runtime architecture and identify the most important resource limitation. Cite the file and explain its effect.'
```

`forge-agents` and `forge-worker` are installed entry points for the same modules.
The example supervises two independent processes, each with its own loaded
model and authenticated loopback endpoint. Each example replica enables bounded
256 MiB FP16 prefix reuse for repeated task context. They run on the same Mac GPU;
replicas do not create extra hardware or guarantee lower latency.

The local task configuration selects the reconstructed 7B model and its native
tool template. Import/export its artifact first. `examples/agents-local-smoke.json`
uses the smaller 0.5B model for transport smoke tests.

For a coding task, add `--allow-write --allow-tests`. Writes require a version
obtained from reading the file. Prefer `replace_text` for a narrow edit;
`edit_lines` replaces an inclusive range of complete source lines and is the
preferred multi-line edit. Preserve the actual indentation and omit displayed
line-number labels. Outside lines remain unchanged; a nonempty replacement
without a final terminator retains the selected block's boundary terminator.
`write_file` replaces the entire file. The coordinator serializes its own writes
and rejects stale versions. This is an optimistic check, not a transaction with
unmanaged external editors. New reports/patch proposals can be saved through
`save_artifact` even when repository writes are disabled.

Each agent's successful read records its own complete file hash. An edit checks
that version automatically; an optional short receipt/full SHA selects an explicit
version. Receipts from another agent/file are rejected. Redundant receipt/SHA
arguments must agree. Native models use the automatically checked version without
copying hash/receipt arguments. Python edits must compile before publication; code
is not executed by the preflight check. Fresh test caches
prevent rapid same-size edits from executing stale bytecode.

File tools reject traversal, escaping symlinks, private/Git metadata, binary or
nonregular files, and oversized reads/writes. Searches have bounded file/byte
coverage and return explicit truncation. Test execution accepts existing Python
test files, uses a configured Python interpreter without a shell, captures
bounded output, and kills its owned process group on timeout/cancellation. It
executes repository Python code; it is not an operating-system sandbox.

## Model capability and local artifacts

The runtime does not turn a small model into a strong planner. The initial
Qwen2.5-0.5B task evaluation failed all four single/delegated trials on protocol
errors. Those failed outputs are retained separately from passing control tests.
Use the small model for worker smoke tests; judge task quality using an
appropriately capable model and executable acceptance criteria.

Forge can also import an already local MLX affine-4-bit Qwen2 checkpoint into
the existing FP16 artifact format:

```bash
forge-import-mlx /path/to/local/mlx-qwen2-checkpoint models/qwen-reconstructed.engine
```

The import uses actual `mlx.dequantize`, records source hashes, validates the
tensor contract and refuses overwrite. It reconstructs the **quantized model**;
it does not recover original FP16 weights, run native 4-bit inference, or change
the earlier INT8 acceptance gates. The local cached 7B conversion produced a
15.23 GB artifact; two replicas need substantial unified memory. Point each
worker's `model` and `tokenizer` at the corresponding artifact/source tokenizer.

## Delegation, messages and tool results

Tool-aware Qwen workers use Hugging Face's native function-call template and
function-result messages. Function catalogs contain only this agent's granted
tools and currently legal controls. `native_tool_calls` is opt-in; a tokenizer
that ignores function definitions fails readiness. Ordinary generation retains its
existing behavior. Legacy backends use the validated JSON action interface.

A turn may produce one call or a bounded batch of up to four independent spawns
or read-only calls. The entire batch is validated before its effects; mixed
mutations/finish/control batches are rejected. Per-item effects and pending batch
state survive coordinator interruption. There is no evaluation
of model-produced Python, expressions or shell commands.

| Action | Effect |
| --- | --- |
| `tool` | Call an explicitly registered tool with validated arguments. |
| `spawn` | Create a bounded child with a focused task/role and optional dependencies on existing children. |
| `send` | Deliver a durable message to an allowed parent, child or sibling. |
| `wait` | Suspend until the selected direct children settle; deliver their results or failures. |
| `finish` | Record a final result after child outcomes and pending messages have been reviewed. |

The state-derived action menu contains real agent IDs and child statuses. Invalid
actions receive bounded repair feedback. Agents may solve small tasks themselves;
delegation should serve a specific independent deliverable. The root synthesizes
evidence and must surface unresolved child failures.

Children have read-only editing permissions by default. An explicit `spawn.tools`
is a complete grant limited to the parent's existing permissions. Parent/root
requirements supply context while the assigned child task defines its scope.
Completed child outcomes are delivered once; repeated waits or identical journal
queries receive no-progress feedback.

Child messages quote inherited parent goals as background and place the assigned
scope last. Parent workflow steps belong to the parent; a child may delegate only
when its current function catalog advertises spawning.

Spawning is bounded by total agents, hierarchy depth, actions per agent, total
tokens, concurrency and wall-clock deadline. Capacity comes from the operator's
worker configuration. The model cannot provision machines or bypass those bounds.
Requests remain on their chosen replica across bounded transport retries.

## Completion criteria

An agent ending its dialogue and a task passing acceptance criteria are separate
outcomes. `RunResult.completion_verified` becomes true only when an explicitly
configured checker passes. Configure test paths, required artifacts, a final-answer
JSON schema, or a minimum number of completed reviews through `acceptance`:

```json
{"acceptance":{"tests":["tests/test_feature.py"],"min_children":2}}
```

Tests also require `--allow-tests`. A failing checker returns actual diagnostics
and lets the agent use its remaining budget. Rejections and validator timeouts are
bounded and journaled; replay does not silently repeat a stored decision. The
Python API also accepts an async `completion_validator` returning
`CompletionCheck(passed, feedback)`. Select checks appropriate to the task;
format validation does not certify factual correctness.

Configured acceptance-test files are protected from workspace-tool edits, and
their original hashes are checked before and after testing and across CLI resume.
Changing the assertions cannot qualify a repair. These checks are not a sandbox
against arbitrary repository Python code; select trusted tests that reflect the
requested behavior. The benchmark's oracle also lives outside the agent workspace.

## Journaling, context and recovery

The local `.forge` directory contains a private SQLite journal, manifests,
results and artifacts. It is Git-ignored. The journal records tasks, messages,
exact model inputs, token reservations, prepared actions, effects and outcomes.
A renewable lease prevents two coordinators from driving the same run.

```bash
forge-agents --state-dir .forge inspect --run-id RUN_ID
forge-agents --state-dir .forge cancel --run-id RUN_ID
forge-agents --state-dir .forge resume --run-id RUN_ID \
  --config examples/agents-local.json --workspace .
```

Resume requires the original workspace, tool permissions, configuration,
implementation and worker artifact/tokenizer fingerprints. Pass the original
write/test flags when applicable. A resumed model call uses its exact saved input.
Interrupted or ambiguous calls remain conservatively charged to the token budget.
Idempotent worker requests have bounded retention; after coordinator/worker
restart or expiry, inference can be recomputed, and unknown costs stay charged.

Completed spawn/message effects are idempotent. Read-only tools can be replayed
after interruption; a writing tool with an unknown outcome is not automatically
executed again. Its agent fails with a visible reason. Known preflight rejections
can be repaired without pretending a mutation occurred.

Context projection preserves the system instruction and complete original task,
adds current child/dependency state and journal receipts, and includes recent
messages within a conservative byte-based token reservation. Omitted/truncated
evidence is marked explicitly. The built-in `journal_read` tool retrieves earlier
agent evidence with bounded pagination. Normal evidence pages skip the reader's
own recall copies; full audit metadata is available with `include_control`.
The full source evidence remains in the
journal; an initial task that cannot fit fails explicitly.

## Remote machines and GPU placement

Run one worker for each desired replica on its host. Select `--backend cuda` or
`mlx`; a process worker's `cuda_visible_devices` setting isolates its GPU choice.
The original CUDA engine still needs NVIDIA device qualification.

```bash
export FORGE_WORKER_TOKEN='a-long-random-credential'
forge-worker --model /models/qwen.engine --tokenizer /models/tokenizer \
  --backend cuda --host 0.0.0.0 --port 8443 \
  --tls-cert /certs/worker.crt --tls-key /certs/worker.key
```

Use `examples/agents-remote.json` with real HTTPS URLs and token environment
variable names. Credentials are not stored in config/journal manifests. Servers
support TLS, bounded queues/HTTP handlers/body/result sizes, idempotent submission,
health identity, deadlines and cancellation. Plain HTTP is accepted for loopback;
LAN plaintext requires explicit opt-in on both server/client. Clients reject
redirects so a credential is not forwarded to another endpoint.

## Speculative workers

An MLX worker may opt into the same speculative decoder:

```json
{"type":"process","name":"replica","model":"models/qwen2.5-0.5b.engine",
 "tokenizer":"Qwen/Qwen2.5-0.5B-Instruct","backend":"mlx",
 "speculative":{"draft_tokens":4,"verification_mode":"block","adaptive":true}}
```

Its default draft reuses prompt/history n-grams. Block verification retains the
explicit FP16 numerical boundary in [speculative-decoding.md](speculative-decoding.md).
Use sequential verification for the canonical reduction path; it has no speculative
speed claim. Speculation is disabled by default.

## Validation

Control tests use scripted decisions and injected faults. They exercise real
HTTP endpoints, process ownership, concurrency, cancellation, retries, durability,
budgets, dependency failures, safe edits, context projection and recovery. They
establish runtime behavior, not model intelligence.

```bash
PYTHONPATH=python pytest -q tests/test_agent_runtime.py tests/test_agent_workers.py \
  tests/test_agent_tools.py tests/test_agent_integration.py tests/test_request_history.py
PYTHONPATH=python python benchmarks/evaluate_agents.py \
  --config examples/agents-local.json --output benchmarks/results/new-agent-evaluation.json
```

The task evaluator compares single-agent and delegated modes on a code repair
graded by fresh executable tests and a two-document synthesis graded against exact
facts. It saves model/source identity, all decisions and failures, time, token
usage and child counts. These are two controlled tasks, not broad capability or
multi-computer throughput evidence. See the [qualification report](agent-results.md).
