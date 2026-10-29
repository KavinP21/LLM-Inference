# Conversation redesign validation — October 11, 2026

The previous mission dashboard is replaced with one persistent conversation:
automatic model startup, a bottom composer, ordinary follow-ups, inline tool
activity and approval diffs, and secondary history/settings drawers.

`verification.json` records 603 passing regression tests with zero skips,
failures or errors. After final wording/API refinements, 43 focused tests passed
for conversation lifecycle, review, legacy behavior and actual loopback HTTP.
Twenty new conversation tests cover durable startup acceptance, queued messages,
late follow-ups, cancellation, failed startup/retry, reload, idempotency,
legacy budget migration and keeping the latest goal during context projection.

Browser QA used the actual HTTP server, SQLite journal, runtime and file tools
with an explicitly labeled deterministic action backend. It did not mock API
responses. Enter, multiline input, queued follow-ups, inline approve/decline,
Stop, same-chat continuation, reload/draft restoration, settings persistence,
agent disclosure, mobile focus isolation and a real artifact download passed.
Final desktop/mobile screenshots show that tested conversation flow; they are
not evidence of model inference. Initial observed UI bugs were fixed and their
interactions checked again.

`live-chat-smoke.json` separately records three sequential real Qwen3.5-4B
turns. The first prompt started its worker automatically, the second recalled
the first turn's codename, and the third read the fixture README and returned
its observed heading. All three completed with plain-language output. The
source revision and file/recipe fingerprints identify this execution.
This narrow integration fixture does not establish broad assistant capability
or reliability, and its elapsed time includes startup rather than measuring
throughput.

Reproduce with the existing local checkpoint installed:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/smoke_studio_chat.py \
  --output benchmark-results/studio-chat-smoke.json

PYTHONPATH=python .venv/bin/python -m pytest -q \
  tests/test_agent_*.py tests/test_chat_runtime.py tests/test_harness_*.py
```

The redesigned interaction references public
[Claude Code interactive documentation](https://code.claude.com/docs/en/interactive-mode).
Its normal flow uses Enter to submit, Shift+Enter for a newline, and Escape to
stop. Forge retains its own visual theme and implementation. This release still
updates activity at action boundaries and does not add token streaming, general
shell execution, OS isolation, Studio media upload or remote MCP transports.
