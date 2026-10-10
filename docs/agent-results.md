# Agent qualification and diagnostic results

The [final frozen-source run](../benchmarks/results/agents-published-2026-10-10/qwen3-final-tasks.json)
passed all four registered cases on commit `1477c4b`, with a clean checkout and
unchanged sources throughout execution. Two supervised processes loaded the same
already cached Qwen3-30B-A3B checkpoint through the explicitly selected MLX-LM
runner. Startup took 21.65 seconds and is excluded from the task times below.

| Task | Single agent | Delegated | Actual children | Independent result |
| --- | --- | --- | --- | --- |
| Python repair | 17.40 s, 6 model calls | 110.18 s, 28 calls | 2 | Both pass all six tests, including four additional inputs withheld until final grading. |
| Document analysis | 8.84 s, 4 model calls | 25.61 s, 9 calls | 2 | Both return all eight correct typed facts, including excluded evidence and unverified release/energy claims. |

Delegation was slower in these fixtures. No task-quality or latency improvement
over one capable agent is established. These two development fixtures demonstrate
actual tool use, edits/tests, child execution and source-grounded synthesis; they
do not establish broad coding/research reliability, unseen task generalization,
multi-host scaling, or an independent causal benefit from the handoff change.
The inference work here is upstream MLX-LM execution; Forge-native speculation
has its own separate measured contract.

The subsequent [Qwen7B qualification](qwen-7b.md#executed-checks-and-task-limits)
checks the new dense 4-bit adapter separately. Its single-agent document case
passes, while both coding modes and delegated document analysis fail their
independent gates. Native 7B chat and direct-MLX-LM generation parity pass;
inference support does not establish autonomous task quality for that checkpoint.

Control correctness and model task quality are separate checks. Scripted models
exercise concurrency, durable effects, cancellation, recovery, budget limits,
tool permissions and transport failures. They do not establish reasoning ability.

The available physical machine is an Apple M3 Max with 128 GB unified memory.
Two supervised loopback processes can each load a full model. The earlier
[0.5B worker smoke](../benchmarks/results/agents-2026-10-09/worker-smoke.json)
records actual generation, cancellation, recovery and process cleanup. It does
not establish coding or research task success, multiple physical GPUs or
multi-host deployment.

## Actual task procedure

[evaluate_agents.py](../benchmarks/evaluate_agents.py) runs the same two small
tasks through single-agent and delegated workflows, with the same tool interface
and global budget. The delegated prompt requests two independent reviews; it
therefore has a different instruction and additional coordination work.

The coding task repairs an arithmetic mean and the empty-input behavior, then
runs tests. Independent grading uses an operator-owned test copy outside the
agent workspace, including four additional input cases. Editing the visible
tests cannot manufacture acceptance. The document task reads two supplied
reports, computes six numeric fields and distinguishes excluded evidence and
two unverified claims. Its fact grader checks exact fields and types.

Completion steering checks executable coding tests or the requested document
answer structure and completed children. Document fact values are withheld from
the model. A format check alone is not factual verification.

The earlier Forge-native 7B trials used an FP16 reconstruction of an already
cached MLX affine-4-bit Qwen2.5-7B model. The import does not recover original
FP16 weights. The final qualification instead uses the cached Qwen3 checkpoint
through MLX-LM, as identified above. Artifact and source fingerprints are recorded
in the reports. No model download or multi-computer experiment was performed.

## Retained failed development trials

| Trial | Coding single / delegated | Documents single / delegated | Interpretation |
| --- | --- | --- | --- |
| [0.5B initial](../benchmarks/results/agents-2026-10-09/qwen05-tasks-initial.json) | Fail / fail | Fail / fail | Protocol failures; suitable for transport smoke only. |
| [7B initial](../benchmarks/results/agents-2026-10-09/qwen7b-tasks-initial.json) | Fail / fail | Pass / fail | Initial JSON interface and delegation issues. |
| [7B revised](../benchmarks/results/agents-2026-10-10/qwen7b-tasks-revised.json) | Fail / fail | Pass / fail | Invalid edits, evidence loops and incomplete synthesis. |
| [7B native initial](../benchmarks/results/agents-2026-10-10/qwen7b-native-tasks.json) | Fail / fail | Fail / fact answer passes with **zero children** | The last result does not qualify delegation. |
| [7B completion-check diagnostics](../benchmarks/results/agents-2026-10-10/qwen7b-checker-diagnostics.json) | Fail / fail | Pass / pass with **two children** | Loaded development code; source files changed during the run. |

These reports contain the actual model outputs and tool events from earlier
source versions. They are diagnostic evidence, not measurements of the final
implementation. Their failures motivated native tool templates, validated
bounded batches, per-agent automatic edit version checks, complete Python
compilation preflight, clearer diagnostics, and executable completion criteria.
The last document pair took 20.13 seconds for one agent and 150.17 seconds for
delegation. Both children inspected actual reports, but this is not a latency
advantage or a result from the final frozen sources.

## Frozen-source continuation

The [first published-source continuation](../benchmarks/results/agents-published-2026-10-10/qwen7b-before-line-edits.json)
ran commit `f5a1a6e` with no source changes during execution. Coding failed in
147.58/450.18 seconds for single/delegated modes; document analysis passed in
19.17 seconds for one agent and failed in 327.75 seconds with delegation.
Compilation/acceptance checks prevented invalid edits from becoming a success.
A child following parent spawning instructions exposed an assignment-scope
problem. Line-range edits, inspection-first catalogs and stronger child scope
framing address those concrete failures and were evaluated in the subsequent
continuations below.

The current grader uses immutable
copies of visible tests for completion steering and adds four further inputs only
after the run. Earlier diagnostic checkers also returned those additional tests'
failure output to the model; they are not held-out evaluations. These two small
fixtures were used during development and are not a broad agent benchmark.

The [line/scope 7B continuation](../benchmarks/results/agents-published-2026-10-10/qwen7b-line-scope-tasks.json)
again failed both coding workflows; document single-agent analysis passed, while
delegation completed but made an unsupported release claim. The separate
[short numerical check](../benchmarks/results/agents-published-2026-10-10/7b-numerics.json)
matched an independent CPU FP16 Transformers SDPA reference at all nine token
positions. Final-output cosine was 0.99999846/0.99999881 for fallback/custom MLX.
This checks short prefill against the same reconstructed artifact, not original
FP16 weights, long cached decoding, or general model reasoning. An invalid
eager-attention reference produced NaNs and is retained separately.

The [optional-runner candidate](../benchmarks/results/agents-published-2026-10-10/qwen3-optional-runner-candidate.json)
uses two actual MLX-LM 0.28.3 replicas of an already cached Qwen3-30B-A3B
affine-4-bit checkpoint, with MLX 0.32.2. Both coding modes passed all six
independent tests (17.19/73.49 seconds); document single-agent analysis passed
(8.41 seconds). Delegated document analysis completed in 23.43 seconds but
incorrectly promoted an excluded measurement. These results establish useful
coding execution, not a passed document delegation gate. Upstream inference is
identified explicitly; it is not Forge-native W4A16 qualification.

Two small tasks, even when
successful, would not establish general coding/research reliability or an
advantage over one capable agent. Single-GPU replicas can increase contention;
delegation should be selected for independent deliverables rather than assumed
to improve latency or correctness.
