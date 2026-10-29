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
pip install -e '.[mlx,vlm,agents,test]'
python -m forge_llm.prepare_models
forge-agents --state-dir .forge run --config examples/agents-studio.json \
  --workspace . --task 'Review the architecture and explain its resource limits.'
```

`examples/agents-studio.json` is the setup Studio uses: Gemma 4 26B A4B runs
every root agent, and two Qwen3.5-4B workers handle sub-agents (see
[models.md](models.md)). `examples/agents-gemma4.json` is Gemma alone, and
`examples/agents-qwen35.json` is two Qwen3.5 replicas. Add `--image FILE` or
`--video FILE` for visual tasks; media is stored once in the journal and
inherited by children.

### Worker groups

Each worker can name a `group`. The runtime sends root agents to
`runtime.driver_group` and sub-agents to `runtime.subagent_group`. Within a
group, an agent goes back to the replica that served its previous step when
that replica is free, since that's where its prompt is cached; otherwise it
takes the least busy running replica. Sub-agents get the smaller of the
configured context and their model's own window.

`"lazy": true` process workers start in the background without loading their
model, load it on their first request (~1.2 s for Qwen3.5) and exit after
`idle_seconds` (default 600) without work. Workers stream partial output back to
the coordinator while they generate.

`forge-agents` and `forge-worker` are installed entry points for the same
modules. Replicas on one Mac share its GPU, so extra replicas add concurrency
but not hardware.

The native-engine examples still work: `examples/agents-local.json` runs a
reconstructed FP16 7B `.engine` file, and `examples/agents-local-smoke.json`
uses the 0.5B model for transport smoke tests.

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

## Model capability

The runtime doesn't make a small model a good planner. Qwen2.5-0.5B failed every
recorded task on protocol errors, so use it only for smoke tests, and judge task
quality with a capable model and executable acceptance criteria.

## Delegation, messages and tool results

Tool-aware workers use each model's native function-call template and
function-result messages. The catalog lists this agent's granted tools and the
agent controls, and stays the same every turn so the prompt cache keeps working. `native_tool_calls` is opt-in; a tokenizer
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

Child handoffs separate the model's unverified conclusions from up to three
recorded read-only observations. Source text, observed file/line metadata, exact
journal event IDs, arguments and recovery cursors let the parent check decisive
claims. Listings, empty searches, failed reads, mutations and recursive recall
copies are excluded. Excerpts and total payloads are bounded; full observations
remain in the journal. Missing measurements stay unknown rather than becoming
zeros or approval claims. Citing a source doesn't make a conclusion drawn
from it correct.

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
a format check doesn't verify that the facts are right.

Configured acceptance-test files are protected from workspace-tool edits, and
their original hashes are checked before and after testing and across CLI resume.
Editing the assertions can't make a repair pass. These checks are not a sandbox
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
messages within a conservative token reservation. Bytes alone overcount code
about 3.5x, so once the backend has measured an agent's previous request, the
estimate is that count plus a byte bound for only what was appended since. It
is still an upper bound, and agents keep several times more history before
anything is dropped. Omitted/truncated evidence is marked explicitly. The
built-in `journal_read` tool retrieves earlier agent evidence with bounded
pagination. Normal evidence pages skip the reader's
own recall copies; full audit metadata is available with `include_control`.
The full source evidence remains in the
journal; an initial task that cannot fit fails explicitly.

## Remote machines and GPU placement

Run one worker for each desired replica on its host. Select `--backend cuda` or
`mlx`; a process worker's `cuda_visible_devices` setting isolates its GPU choice.
The CUDA engine hasn't been run on an NVIDIA GPU yet.

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

## Testing

The runtime tests drive scripted "models" through real HTTP endpoints and worker
processes, with injected faults. They cover concurrency, cancellation, retries,
journaling and recovery, budgets, dependency failures, version-checked edits and
context compaction. They test the runtime's behavior, not how capable a model is
at the tasks.

```bash
PYTHONPATH=python pytest -q tests/test_agent_runtime.py tests/test_agent_workers.py \
  tests/test_agent_tools.py tests/test_agent_integration.py tests/test_request_history.py
PYTHONPATH=python python benchmarks/evaluate_agents.py \
  --config examples/agents-local.json --output benchmark-results/agent-evaluation.json
```

## How real models did

`benchmarks/evaluate_agents.py` runs two small tasks, once with a single agent
and once delegating to two children:

- **Code repair:** fix a buggy `mean()` and its empty-input case. It's graded with
  a separate copy of the tests the agent can't edit, plus four extra inputs the
  agent never sees.
- **Document analysis:** read two reports and return eight typed facts, including
  which evidence was excluded and which claims are unverified. It's graded field
  by field.

| Model (runner) | Code, single / delegated | Docs, single / delegated |
| --- | --- | --- |
| Qwen3.5-4B, 4-bit (MLX-VLM) | Pass, 25.9 s / Pass, 56.1 s | Pass, 8.4 s / Pass, 51.4 s |
| Qwen3-30B-A3B, 4-bit (via MLX-LM, before the VLM runner) | Pass, 17.4 s / Pass, 110.2 s | Pass, 8.8 s / Pass, 25.6 s |
| Qwen2.5-7B-Instruct, 4-bit (via MLX-LM) | Fail / Fail | Pass / Fail |
| Qwen2.5-0.5B (native) | Fail / Fail | Fail / Fail |

Each delegated run spawned and completed two children. Delegation was slower
every time and didn't improve correctness on these tasks, so it's only worth
using for genuinely independent pieces of work. Qwen3.5's first delegated docs
run failed because the strict parser rejected valid XML tool calls that had a
sentence of explanation around them. The parser now accepts that. The 7B model
mostly failed by making invalid edits, looping on reads, or writing an incomplete
synthesis.

These are two small tasks used during development, so they don't say much about
general reliability. Everything ran on one M3 Max with two replicas on the same
GPU.
