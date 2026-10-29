# Developer workspace and image validation — October 12, 2026

This checkpoint adds a toggleable developer panel, image attachments, reviewed
commands and scheduler-owned per-agent controls to the existing chat workflow.

- Final agent/chat/harness regression: **647 passed**, no skips, failures or errors.
- Later focused cancellation/review hardening: **30 passed**, including immediate
  rejection of an approval racing a stop request or parent failure.
- Actual browser workflow: file viewing and references, a 197,251-byte PNG
  selected/previewed/sent through real HTTP, scoped image preview after reload,
  clipboard paste/removal, exact diff approval, created-file preview, report
  insertion, and messaging/stopping one child while its sibling/main continued.
- Actual Git fixture: one modified Python file appeared separately from agent
  receipts and its real HEAD diff loaded. Git pathspecs are treated literally;
  configured external programs and private-file inspection are guarded.
- Desktop and mobile: 1280×900 and 390×844; mobile page width remained 390 and
  closing/inserting from the drawer restored the composer. Console errors: zero.

Browser evidence uses an explicitly labeled deterministic action backend, with
real APIs, SQLite, tools and filesystem operations. It does not claim model
inference. Native file drag was not exercised because the available automation
API lacks that operation; encoding/format/limit and handler boundary logic was
probed separately. Real picker and clipboard interactions were exercised.

`live-image-smoke.json` records the separate real Qwen3.5-4B check: it identified
a red rectangle, then a blue rectangle in a follow-up using the retained image.
In a separate chat it called `run_command`, received approval only for one exact
Python print command, executed it and reported the actual marker output.
The report identifies source checkpoint `60054a0` and module fingerprints.
This is a small integration fixture, not broad vision/coding reliability or
performance qualification. Subsequent UI/Git/cancellation safeguards are covered
by focused tests and recorded with the next checkpoint.

Reproduce with the existing local checkpoint:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/smoke_studio_image.py \
  --output benchmark-results/studio-image-smoke.json

PYTHONPATH=python .venv/bin/python -m pytest -q \
  tests/test_agent_*.py tests/test_chat_runtime.py tests/test_harness_*.py
```

See `verification.json` for the executed scopes, and the
[Studio guide](../../../docs/forge-studio.md) for controls and limits. Commands
are reviewed argv execution rather than an interactive TTY or OS sandbox.
Token streaming, Studio video upload, remote MCP and computer use remain outside
this release.
