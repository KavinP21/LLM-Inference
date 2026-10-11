# Forge Studio harness

Forge Studio presents the existing task-agent runtime as a conversational assistant.
The design is based on public documentation and original implementation; no
leaked proprietary implementation is used. The local inference engine remains
the model execution layer. Harness scheduling does not improve a model's
reasoning ability by itself.

## Experience

The normal path is open, type, Enter, read the reply, and type a follow-up in
the same composer. Backend loading happens automatically. User messages, answers,
tool activity, child-agent progress and approval prompts share one transcript.
Technical controls are secondary settings rather than required task setup.
The custom palette and styling do not introduce another workflow to learn.

Enter sends, Shift+Enter inserts a newline, and Escape/Stop interrupts active
work. Accepted messages are durable before model startup. Request IDs prevent
an uncertain browser retry from creating duplicate work. Follow-ups preserve
conversation roles and context rather than asking the user to branch a mission.
A collapsed Chats rail supplies history and a New chat action.

## Components and boundaries

| Component | Responsibility |
| --- | --- |
| `agents.AgentRuntime` | Existing bounded agent loop, child dependencies, messages, native tool calls, compaction, leases and recovery |
| `agents.AgentStore` | Existing durable SQLite journal, input snapshots, idempotent effects and token reservations |
| `harness.service` | Operator sessions, lineage, backend lifecycle, per-run tools and live snapshots |
| `harness.conversation` | Durable chats, ordered turns, prompt acceptance, startup queuing and conversation context |
| `harness.policy` | Exact-invocation approval, expiry, cancellation, read-only mode and lifecycle hooks |
| `harness.extensions` | Selected local skills and explicitly configured MCP processes |
| `harness.server` | Loopback HTTP, authenticated mutations, static UI and trace export |
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
requests stay pinned if older history is compacted. Internal run IDs and lineage
do not require the user to learn a separate branching workflow. The legacy
mission endpoints remain available for programmatic integrations.

## Tool policy

Read-only mode exposes inspection tools. Review mode also exposes edits,
artifact creation and selected tests, but each effect requires an operator
decision against its exact arguments. Approval does not create a standing
grant. File edits retain the existing per-agent read receipts and concurrent
version checks. A preview shows the intended diff before execution. Testing
executes trusted repository Python and is not an operating-system sandbox.
Children receive read tools by default and cannot escalate beyond their parent's
tool set. Uncertain side effects are not automatically replayed after a crash.

The server binds only to loopback. A randomly generated browser bootstrap token
is exchanged for an HttpOnly SameSite cookie; the UI removes the token fragment.
Mutations require JSON, the expected Host, and a same-origin Origin if supplied.
There are no arbitrary model-generated shell commands or automatic plugin
installation. MCP commands come from an operator-selected local configuration.
MCP tool annotations are descriptive, never an authority to bypass review.

## Extensibility and evidence

Skills are discovered only in configured directories. Only names/descriptions
enter the catalog; bodies load on selection or a tool request. Skill prose does
not change tool permissions. MCP clients expose validated namespaced schemas
through the same tool and approval pipeline. Ordered async hooks can reject a
tool before execution and observe results.

Export includes the run, agents, receipts, usage and manifest. Token counts
include measured inputs/outputs, outstanding reservations and uncertain work.
The timeline surfaces context compaction, backend retries, rejected actions and
completion checks rather than claiming a run succeeded from prose alone.

## Validation and intended limits

Portable tests cover durable lifecycle, branching, steering, restart, approvals,
path confinement, stdio protocol negotiation, cancellation and HTTP authorization.
Browser QA covers empty, live, review, error and narrow layouts. A small local
model smoke run checks integration separately from deterministic tests.

This initial release does not claim frontier model capability, broad task
reliability, full OS/container isolation, remote public serving, collaboration
between human accounts, arbitrary computer use or hosted automation. The design
keeps those separate from working local orchestration. Model execution is shown
at action boundaries; token-by-token streaming is a separate worker capability.

## Public design references

- [OpenAI running agents](https://developers.openai.com/api/docs/guides/agents/running-agents): persistent sessions, bounded loops and resumable state.
- [Claude SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents): independent contexts, tool scopes, messaging and bounded delegation.
- [Claude SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks): explicit lifecycle interception and tool review.
- [MCP specification](https://modelcontextprotocol.io/specification): protocol lifecycle and tool schemas. Supported revisions are recorded with the implementation.

These references describe patterns, not code copied into Forge.
