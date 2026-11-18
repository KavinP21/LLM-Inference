# Instinct internals

The app handles conversations, tool approvals and the browser UI. The agent
runtime schedules work and the inference engine runs the model.

## Experience

The transcript shows replies, tool activity, child agents and approval
requests. The Workspace panel shows agent tasks, file changes and command
output. Accepted messages are saved even while models are loading. Request IDs
prevent a browser retry from submitting the same message twice.

## Components and boundaries

| Component | Responsibility |
| --- | --- |
| `agents.AgentRuntime` | Existing bounded agent loop, child dependencies, messages, native tool calls, compaction, leases and recovery |
| `agents.AgentStore` | Existing durable SQLite journal, input snapshots, idempotent effects and token reservations |
| `harness.service` | Operator sessions, lineage, backend lifecycle, per-run tools and live snapshots |
| `harness.conversation` | Durable chats, ordered turns, prompt acceptance, startup queuing and conversation context |
| `harness.developer` | Confined file views, agent controls, actual change/command receipts and read-only Git inspection |
| `harness.commands` | Reviewed argv commands, bounded output and owned process-group cancellation |
| `harness.policy` | Exact-invocation approval, expiry, cancellation, read-only mode and lifecycle hooks |
| `harness.extensions` | Selected local skills and explicitly configured MCP processes |
| `harness.server` | App HTTP, authenticated requests, static UI and trace export |
| `harness.web` | Persistent composer, transcript, inline activity/review, chat history and secondary settings |

One asyncio loop owns the journal, policies, runtime tasks and model pool.
HTTP handlers dispatch onto that loop; they never share a SQLite connection
across threads. Logical agents share model replicas instead of each loading
weights. The existing worker configuration selects the model/backend. Startup
starts in the background, and model-loading errors appear in the conversation. A deterministic demo
mode exercises real scheduling and tools and is always labeled as simulation.

## Run lifecycle

Creation binds the workspace, policy, runtime budget, selected skill and worker
configuration to a durable run manifest. The manifest prevents recovery with
different permissions or a different backend configuration. Live model identities
are fingerprinted using the existing worker identity API before recovery.
The event journal is the source of truth. Browser reconnection rereads it.

Steering appends an operator message durably; a generation already in flight
finishes against its recorded input, and the next turn sees the message.
Cancellation propagates through runtime and worker cancellation.
Interrupted runs retain remaining budgets and deadline; recovery is explicit.
Terminal agent runs remain immutable. A conversation links ordered runs while
preserving prior user and assistant roles in bounded model context. Current
requests stay pinned when older history is compacted. The app also exposes the
run API separately from chats.

## Tool policy

Read-only mode exposes inspection tools. Review mode also exposes edits,
artifact creation and selected tests, but each effect requires an operator
decision against its exact arguments. Approval does not create a standing
grant. File edits retain the existing per-agent read receipts and concurrent
version checks. A preview shows the intended diff before execution. Testing
executes trusted repository Python and is not an operating-system sandbox.
Children receive read tools by default and cannot escalate beyond their parent's
tool set. Uncertain side effects are not automatically replayed after a crash.

The app listens on loopback by default. A separate listener for other devices
requires an app key and can use TLS. See [device setup](app.md#other-devices).
The local browser exchanges a bootstrap token for a session cookie.

In review mode, command approval covers the exact arguments, working directory
and timeout. App commands run under the macOS sandbox. MCP processes come from
the operator's configuration and use the same review policy for tool calls.

## Extensions and tracing

Skills are discovered only in configured directories. Only names/descriptions
enter the catalog; bodies load on selection or a tool request. Skill prose does
not change tool permissions. MCP clients expose validated namespaced schemas
through the same tool and approval pipeline. Ordered async hooks can reject a
tool before execution and observe results.

Export includes the run, agents, receipts, usage and manifest. Token counts
include measured inputs/outputs, outstanding reservations and uncertain work.
The timeline surfaces context compaction, backend retries, rejected actions and
completion checks rather than claiming a run succeeded from prose alone.

## Tests and limits

Portable tests cover durable lifecycle, branching, steering, restart, approvals,
path confinement, stdio protocol negotiation, cancellation and HTTP authorization.
Browser QA covers empty, live, review, error and narrow layouts. A small local
model smoke run checks integration separately from deterministic tests.

The transcript streams partial model output when the worker supports it.
[Agent evaluations](agent-runtime.md#how-real-models-did) describe the tasks
used to check model behavior.

## Public design references

- [OpenAI running agents](https://developers.openai.com/api/docs/guides/agents/running-agents): persistent sessions, bounded loops and resumable state.
- [Claude SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents): independent contexts, tool scopes, messaging and bounded delegation.
- [Claude SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks): explicit lifecycle interception and tool review.
- [MCP specification](https://modelcontextprotocol.io/specification): protocol lifecycle and tool schemas. Supported revisions are recorded with the implementation.
