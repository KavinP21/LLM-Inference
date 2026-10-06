# FP16 prefix-cache checkpoint: validation passed, timing deferred

2026-10-02, Apple M3 Max, MLX 0.32.2, Python 3.12.10, macOS 26.5.
The implementation and all registered correctness/resource gates pass. The user
requested a safe stopping point within ten minutes; execution was stopped **after**
the durable validation checkpoint and **before** the performance matrix. This is
not yet the completed performance-qualified milestone, and no speedup is claimed.

## Implemented

- Opt-in, FP16-only MLX prefix snapshots with exact token keys and request namespaces.
- Unique-page reference counting, immutable cache pins, pre-reserved partial-tail
  copy-on-write capacity, safe admission and bounded LRU page/entry budgets.
- Exact full-prompt reuse without a prefill forward pass; suffix reuse at original
  chunk boundaries to preserve FP16 operation shapes.
- Read-only host final-logit rows with separately reported cache/pending-match bytes.
- Cancellation, clear/eviction with live owners, exception cleanup and idempotent close.
- Public constructor options, `cache_namespace`, `clear_prefix_cache`, and detailed stats.
- A frozen validation/benchmark driver, raw-evidence audit tests and portable CI workflow.

The API, ownership diagram, numerical boundary and reproduction commands are in
[the contract](prefix-cache.md). Prefix caching is disabled by default; FP16 artifacts,
desktop app settings, quantization gates and CUDA validation boundaries are unchanged.

## Verified scope

Both official FP16 artifacts were used without fitting or changing their weights:
Qwen2.5-0.5B-Instruct and Gemma 3 1B text-only. The numerical policy is `rowwise`.

| Gate | Qwen | Gemma |
| --- | --- | --- |
| 40 cases, chunk size 127, 32 outputs per path | Passed | Passed |
| 40 cases, chunk size 512, 32 outputs per path | Passed | Passed |
| Eight concurrent shared/branched requests | Passed | Passed |
| 32,766-token prompt plus two outputs | Passed | Passed |

Each 40-case group contains 25 fixed text prompts and 15 page/chunk boundary lengths.
Uncached, cold, exact-hit and suffix-branch paths have identical token IDs and
FP32 logit-row SHA-256/dtype/shape identities. There are 25,600 observed quality
logit rows across all five paths and both families/chunk sizes; this is same-artifact
cache consistency, not new Transformers semantic certification. Warm hits execute
zero prefill tokens. Branches keep original chunk shapes; for example, a 513-token
prompt with 127-token chunks reuses 508 tokens and prefills its five-token suffix.

Stress checks include waiting-hit cancellation, clearing pins while eight admitted
matches remain valid, namespace misses, COW and LRU pressure. Each family's stress
run records 22 COW copies and four pressure evictions, then zero live reservations,
physical pages and device-store bytes after idle-cache clearing. Tiny-model Metal
tests independently exercise active cancellation, allocator/decode failures and
direct shared-page write rejection.

| 32K resource observation | Qwen | Gemma |
| --- | --- | --- |
| Peak materialized logical physical pages | 2,049 | 2,049 |
| Accounted peak K/V bytes | 402,849,792 | 872,841,216 |
| Retained host final-logit bytes (64 entries) | 38,895,616 | 67,108,864 |
| Warm prefill tokens / actual decode rows | 0 / 1 | 0 / 1 |
| Idle clear: allocated/reserved/device bytes | 0 / 0 / 0 | 0 / 0 / 0 |

The extra page versus the private-cache 2,048-page baseline is the reserved private
copy of the cached partial tail. These are conservative page/materialization counts,
not hardware memory counters: COW initially aliases immutable MLX arrays, then
replaces touched private arrays. Host logit bytes are a separate bounded cost.
32K results establish execution and cache consistency, not long-context task quality
or a statistically rigorous long-context latency result.

## Preserved first run and dispatch fix

The [first frozen run](../benchmarks/results/prefix-cache-m3-max-2026-10-02/execution-failure.json)
completed Gemma validation but stopped on an existing uncached Qwen dispatch error:
forced fused attention was unavailable for a five-query, GQA-factor-seven tail.
No timing ran. Its source archive, registration, completed reports and exception
record are preserved and whole-file bound into the fresh checkpoint.

A capability guard now leaves unsupported short tails to MLX's default fallback,
while supported shapes retain their dispatch. Eight actual-Metal regression cases
cover query lengths 1, 4, 5, 6, 7, 8, 9 and 17 against MLX's default operator.
The pinned upstream eligibility rule is linked in the contract. All official
validation was rerun from scratch under a new source identity with the same corpus,
workloads and gates; the first run was not overwritten or promoted to acceptance.

## Evidence and stopping state

Fresh root: `benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed`.

- [Registration](../benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/contract.json)
  and captured `sources.zip`: runtime SHA-256
  `e46b6d7dfaa230e49d6748925330b0f8b916db305f9c347f0a6cc281da398892`.
- [Eight passed gates](../benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/validation-checkpoint.json)
  and [fresh independent readback](../benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/validation-readback.json).
- 160 unchanged complete prior files and 104 captured source members verified;
  live archived runtime/tests/tools also match. Historical INT8 live auditors were
  checked before runtime edits; their old runtime guards are not bypassed now.
- [Full regression JUnit](../benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/test-results.xml):
  365 tests pass, no skips, failures or errors. There are 78 new tests: 52 portable
  ownership/audit cases and 26 Metal prefix/dispatch cases. The existing C++ host
  and UBSan binaries pass; no fresh C++/CUDA build or remote CI execution is claimed.
- [Environment](../benchmarks/results/prefix-cache-m3-max-2026-10-02-dispatch-fixed/environment.json)
  and [resume handoff](prefix-cache-handoff.md).

The GPU driver exited at the requested validation boundary and no benchmark child
was launched. There is no `matrix`, final milestone verification, release tag or
new CUDA claim. Strict INT8 remains rejected/experimental; W4 remains gated.

## Remaining acceptance work

Run the registered 24-report isolated-process disabled/cold/primed matrix, then
recompute its raw timings/work counters, save stage verification, rerun the full
suite and bind the final complete evidence index. Priming is outside timing; cold
insertion is inside. Report both, including regressions. The exact resume sequence
is in the handoff. Only then close this milestone and proceed to FP16 speculative
decoding; the roadmap still counts 12 open delivery milestones today.

Limitations: no cross-engine/process cache, authentication or tenant quotas, no
pointer-addressable zero-copy attention, no quantized or CUDA prefix path, no new
hardware-counter profiling, and no claim of universal inference acceleration.
The cache metadata is bounded; existing scheduler request history remains retained
and must be addressed by future long-running service lifecycle work.
