# Forge Studio integration checks — October 11, 2026

These checks validate the new operator harness. They do not qualify broad agent
capability, production reliability, CUDA execution or inference performance.

| Check | Executed result |
| --- | --- |
| Agent/chat/harness regression | 583 passed; zero failures, errors or skips |
| Artifact download and HTTP follow-up | 14 passed after the final artifact route change |
| Real local Qwen3.5-4B mission | Coordinator delegated one README inspection; child read the fixture and finished; coordinator returned the observed project title |
| Browser UI | Desktop and 390px mobile deterministic fixtures; create, scope inspection, event filtering, exact diff review/approval, output, cancellation, follow-up and branching |
| Static checks | Ruff lint/format, JavaScript syntax, zsh launcher syntax and Git whitespace checks passed |
| Package build | Wheel built; all three UI assets and the `forge-studio` console entry point present |

`verification.json` records the regression and UI scope. `live-smoke.json`
records the model configuration, worker fingerprints, harness file hashes,
recipe hash, usage, events and actual delivered result. The source checkpoint
is `2af7242`; the smoke recipe is added with this evidence checkpoint.
No GPU inference runs in the regression fixture suite. The separate smoke does
execute the installed local model through the existing process backend.

The real-model run used one complete Qwen3.5-4B MLX-VLM worker, read-only tools,
two logical agents, a tiny README fixture and no files to edit. This confirms
delegation and native tool integration for that mission; it is not evidence for
general reasoning, coding or factual accuracy. Reported elapsed time includes
loading and is not a throughput benchmark.

Reproduce from the repository root with the existing local checkpoint installed:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/smoke_studio.py \
  --output benchmark-results/studio-live-smoke.json

PYTHONPATH=python .venv/bin/python -m pytest -q \
  tests/test_agent_*.py tests/test_chat_runtime.py tests/test_harness_*.py
```

Browser QA used explicitly labeled deterministic UI fixtures. Actual HTTP
authorization/Origin/Host checks run separately against the real loopback
server. The UI fixture is not described as an authenticated live-model browser
session. Artifact downloads received API and JavaScript syntax checks after
the main browser fixture pass.

Current harness limits: action-boundary polling, application-level confinement,
text mission input, selected stdio MCP calls, and one local operator. Token
streaming, OS/container isolation, Studio media upload, remote MCP, computer
use, scheduled jobs and multi-user collaboration remain outside this release.
