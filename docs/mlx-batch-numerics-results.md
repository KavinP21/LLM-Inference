# MLX decode numerical-consistency checkpoint

Date: 2026-10-01. Apple M3 Max, 128 GB unified memory, macOS 26.5, native arm64 Python 3.12,
MLX 0.32.2. This is a development checkpoint, not a production-serving release.

The new opt-in `decode_mode="rowwise"` passes the saved same-artifact batching-consistency gates
for Qwen2.5-0.5B and Gemma 3 1B FP16, plus both frozen mixed-INT8 execution modes. It keeps
attention/scheduling batched and leaves the original throughput default and single-request output
contract unchanged. It does **not** repair cross-artifact lossy INT8 quality, certify 25-prompt
Transformers parity, or improve the original Gemma reference near-tie case.

## Evidence and provenance

The final runtime-source SHA-256 is
`a96763acc8106e313efe3df1e3dde5f2a77ebc01f8c9974ef854325d6461d649`; Git HEAD is
`a19fbd386e5e22a7e7e5c963c46c9ef1796a8e2a`. The checkout is dirty and contains earlier/user-owned
changes; the source fingerprint, not HEAD alone, identifies the measured implementation.

All artifacts retain their previous checksummed data identities:

| Artifact | Model-data SHA-256 |
| --- | --- |
| Qwen FP16 | `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` |
| Gemma FP16 | `92d3081b2facfa8a5eb48dcedd89cb8e230b21a00cfeefd4c32f77ca2a9482ee` |
| Qwen frozen mixed | `11a0c1e2a23ef9e9d572447be81a6a1f9fbcd9fcd24ac0c6f24b6c14c902fb5b` |
| Gemma frozen mixed | `c0304dfe852ea62f614c26abc2a0f121baef1231d2a48ee418e34b21532ac386` |

The [serialized regression checkpoint](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/checkpoint.json)
contains checksums of six official-model reports and the paired stage replays. The
[final evidence inventory](../benchmarks/results/batch-invariant-m3-max-2026-10-01/final-evidence-index.json)
audits 54 raw reports, 16 workload files, three checkpoint manifests, exact A/B coverage, all
resource/token/cache gates, and the unchanged runtime fingerprint. The
[contract and reproduction commands](decode-numerics.md) explain the numerical policy and its limits.
The preliminary standalone Gemma replay and pre-change stage trace in the parent result directory
are development diagnostics; the frozen regression subdirectory is authoritative.

## Root cause and intervention

Hypothesis: a batch-shape-dependent reduction, rather than cache ownership, introduces tiny
differences that propagate into near-tied greedy decisions.

The [default stage replay](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/gemma-stages-batched.json)
compares independent decode with eight-row decode under the same teacher-forced history. It retains
lazy stage arrays without inserting extra evaluation barriers into the original forward path,
then copies them after normal decode materialization. A second, single-row replay on the exact
materialized batched input isolates local shape dependence from upstream error.

| Case index | Generated position | First changed operation | Changed outputs | Maximum absolute difference |
| --- | --- | --- | --- | --- |
| 22 | 1 | Layer 0 attention output projection | 2 | 0.0000152588 |
| 22 | 2 | Layer 0 K projection | 3 | 0.0000305176 |
| 22 | 3 | Layer 0 Q projection | 1 | 0.0000610352 |
| 24 | 1 | Layer 0 attention output projection | 1 | 0.0001220703 |
| 24 | 2 | Layer 0 attention output projection | 1 | 0.0000004768 |
| 24 | 3 | Layer 0 Q projection | 1 | 0.0000019073 |

Each first changed projection receives bitwise-identical inputs and the same FP16 weights. The
same-input single-row replay reproduces the local discrepancy. This directly identifies the native
matrix-multiply path's shape-dependent numerics; it does not identify the underlying GPU reduction
instruction order or hardware-counter cause. Fixed eight-row teacher forcing is not the original
staggered arrival trace, and need not flip the same token at the same position.

Code change: only decode projections/head use single-row GEMV/kernel operations, concatenated in
scheduler order. Normalizations, activations, RoPE/cache writes, and paged attention remain batched.
Prefill and dtypes are unchanged. INT8 reconstruction is shared by all row GEMVs in a projection.
No epsilon, tie nudging, prompt detection, output substitution, or hidden FP32-head policy is used.

The [paired rowwise stage replay](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/gemma-stages-rowwise.json)
has identical inputs/outputs at every captured stage in all six replays. Timing instrumentation is
not enabled in either correctness tool; neither stage replay is performance evidence.

The [extended default replay](../benchmarks/results/batch-invariant-m3-max-2026-10-01/extended-stage-replay/gemma-stages-batched.json)
traces both prompts through generated position 17. For case 24 at position 17, the first changed
operation is layer 0 Q projection: identical inputs, two changed outputs, maximum difference
0.0001220703125. The final argmax changes from 236746 to 1282, reproducing the documented near-tie
flip. The [paired extended rowwise replay](../benchmarks/results/batch-invariant-m3-max-2026-10-01/extended-stage-replay/gemma-stages-rowwise.json)
has no changed stage or argmax in any of its 34 teacher-forced steps. This remains a diagnostic
shape comparison, not a timing run or a 34-token free-generation claim.

## Exact batching gates

Fresh independent references use the same current runtime and artifact as the concurrent runs;
historical default outputs are only a separate regression check. Each execution configuration runs
25 prompts with 32 outputs at saturated concurrency 2, 8, and 16, plus 25 staggered requests with
1/4/8/16/32 output budgets. All cases run with capacity-aware arrivals and no silently dropped work.

| Execution | Default exact-token cases: c2 / c8 / c16 / staggered | Rowwise exact tokens | Rowwise exact logit rows |
| --- | --- | --- | --- |
| [Qwen FP16](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/qwen-fp16-batch.json) | 23 / 21 / 23 / 25 | 25/25 in all four traces | 2,705/2,705 |
| [Qwen mixed/direct](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/qwen-mixed-metal-batch.json) | 25 / 25 / 25 / 25 | 25/25 in all four traces | 2,705/2,705 |
| [Qwen mixed/reconstruct](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/qwen-mixed-reconstruct-batch.json) | 25 / 24 / 24 / 25 | 25/25 in all four traces | 2,705/2,705 |
| [Gemma FP16](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/gemma-fp16-batch.json) | 24 / 25 / 23 / 23 | 25/25 in all four traces | 2,705/2,705 |
| [Gemma mixed/direct](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/gemma-mixed-metal-batch.json) | 24 / 24 / 24 / 24 | 25/25 in all four traces | 2,705/2,705 |
| [Gemma mixed/reconstruct](../benchmarks/results/batch-invariant-m3-max-2026-10-01/regression/gemma-mixed-reconstruct-batch.json) | 25 / 25 / 25 / 24 | 25/25 in all four traces | 2,705/2,705 |

That is 16,230 exact emitted logit rows in rowwise mode, measured with FP32 logit-byte hashes,
not a cosine tolerance. Every request has exact event counts, all workloads drain, materialized
cancellation releases its pages, and all reservations/device K/V return to zero. All six independent
reference sets reproduce their historical default tokens. Default negative controls remain failed
where shown; a numerical-policy pass does not relabel them as correct.

Fresh cross-artifact independent-token comparisons still give Qwen **19/25** and Gemma **22/25**
against FP16, in both INT8 modes. Those strict quality gates remain failed. Direct and reconstruction
mixed outputs match each other on all 25 independent cases. See the preserved
[INT8 accuracy report](mlx-int8-hardening-results.md); no weights or calibration policy were changed.

## Boundary and 32K resource gates

All six execution configurations also pass nine synthetic token-ID prompts of lengths
15, 16, 17, 31, 32, 33, 128, 512, and 513, with 128-token prefill chunks and 32 outputs. Concurrent
rowwise execution matches independent tokens and all 288 logit rows per configuration exactly,
including Gemma's 512-token local-attention boundary. This is numerical/resource validation, not
language-model semantic quality.

Every configuration additionally processes a 32,766-token synthetic prompt and generates two
tokens: 64 prefill chunks, one decode iteration, and exactly 2,048 physical 16-token pages. Qwen peak
K/V is 402,653,184 bytes; Gemma is 872,415,232 bytes. Every run reclaims all blocks/reservations.
These are single-run resource checks, not sustained 32K decode throughput or concurrent 32K capacity.
The [resource/matrix manifest](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/checkpoint.json)
contains links-by-path and checksums for all six boundary and all six context reports.

## Controlled performance tradeoff

The serialized performance matrix uses isolated processes, identical token-ID workloads, rotated
mode order, five complete warmups, three measured trials, and 32 outputs per request. Shapes are
128/1,024 prompt tokens and concurrency 1/8. FP16 and frozen-mixed/direct artifacts are measured;
there is no reconstruction-mode speed claim. No correctness host copies or stage hooks run during
measurement. The exclusive lease prevents competing Forge evidence drivers, not unrelated apps.

All 32 rows completed on the same runtime fingerprint, with all requested tokens executed and
zero remaining cache reservations/device bytes. B/R means batched/rowwise. Linked throughput cells
open the complete paired raw results, including per-request latency distributions and memory.

| Artifact | Prompt | Concurrency | Batched tok/s | Rowwise tok/s | p50 TTFT ms B/R | p50 TPOT ms B/R | Peak active GiB B/R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen FP16 | 128 | 1 | [122.8](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-batched-p128-c1.json) | [122.4](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-rowwise-p128-c1.json) | 20.0 / 20.1 | 7.73 / 7.73 | 1.105 / 1.105 |
| Qwen FP16 | 128 | 8 | [426.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-batched-p128-c8.json) | [231.5](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-rowwise-p128-c8.json) | 119.8 / 131.5 | 14.49 / 30.13 | 1.120 / 1.120 |
| Qwen FP16 | 1024 | 1 | [31.2](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-batched-p1024-c1.json) | [31.2](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-rowwise-p1024-c1.json) | 134.3 / 134.7 | 28.68 / 28.75 | 1.506 / 1.506 |
| Qwen FP16 | 1024 | 8 | [61.0](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-batched-p1024-c8.json) | [71.5](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-fp16-rowwise-p1024-c8.json) | 877.1 / 860.5 | 92.29 / 77.36 | 1.593 / 1.593 |
| Qwen mixed/direct | 128 | 1 | [121.5](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-batched-p128-c1.json) | [121.9](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-rowwise-p128-c1.json) | 23.9 / 23.4 | 7.71 / 7.67 | 1.076 / 1.076 |
| Qwen mixed/direct | 128 | 8 | [406.6](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-batched-p128-c8.json) | [222.9](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-rowwise-p128-c8.json) | 135.2 / 148.0 | 14.95 / 30.87 | 1.091 / 1.091 |
| Qwen mixed/direct | 1024 | 1 | [31.2](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-batched-p1024-c1.json) | [31.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-rowwise-p1024-c1.json) | 140.9 / 141.4 | 28.47 / 28.41 | 1.437 / 1.437 |
| Qwen mixed/direct | 1024 | 8 | [79.1](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-batched-p1024-c8.json) | [64.9](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/qwen-mixed-metal-rowwise-p1024-c8.json) | 876.8 / 911.5 | 67.53 / 83.11 | 1.525 / 1.525 |
| Gemma FP16 | 128 | 1 | [50.9](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-batched-p128-c1.json) | [50.7](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-rowwise-p128-c1.json) | 49.5 / 48.7 | 18.53 / 18.75 | 2.167 / 2.167 |
| Gemma FP16 | 128 | 8 | [196.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-batched-p128-c8.json) | [130.0](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-rowwise-p128-c8.json) | 289.4 / 299.1 | 30.24 / 51.08 | 2.199 / 2.199 |
| Gemma FP16 | 1024 | 1 | [17.7](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-batched-p1024-c1.json) | [17.8](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-rowwise-p1024-c1.json) | 286.0 / 282.4 | 48.94 / 48.89 | 2.953 / 2.953 |
| Gemma FP16 | 1024 | 8 | [53.8](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-batched-p1024-c8.json) | [48.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-fp16-rowwise-p1024-c8.json) | 1622.8 / 1646.1 | 89.22 / 105.41 | 3.140 / 3.140 |
| Gemma mixed/direct | 128 | 1 | [54.2](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-batched-p128-c1.json) | [54.4](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-rowwise-p128-c1.json) | 52.8 / 52.3 | 17.31 / 17.27 | 2.125 / 2.125 |
| Gemma mixed/direct | 128 | 8 | [193.7](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-batched-p128-c8.json) | [125.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-rowwise-p128-c8.json) | 310.6 / 324.8 | 30.13 / 52.49 | 2.157 / 2.157 |
| Gemma mixed/direct | 1024 | 1 | [17.7](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-batched-p1024-c1.json) | [17.8](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-rowwise-p1024-c1.json) | 298.4 / 294.5 | 48.56 / 48.55 | 2.760 / 2.760 |
| Gemma mixed/direct | 1024 | 8 | [52.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-batched-p1024-c8.json) | [47.3](../benchmarks/results/batch-invariant-m3-max-2026-10-01/resources-and-matrix/matrix/gemma-mixed-metal-rowwise-p1024-c8.json) | 1696.3 / 1705.3 | 91.16 / 107.07 | 2.947 / 2.947 |

At 128 tokens/concurrency 8, rowwise FP16 throughput is 45.7% lower for Qwen and 33.8% lower for
Gemma in this run. The single-request graph is identical and single-request timings are close;
small differences are not claimed as a speedup. The Qwen FP16 1K/concurrency-8 pair reverses the
expected direction in this session. Without independent replication and GPU-counter attribution,
that observation is not evidence of an optimization win. Long-context page-pack/allocator and
execution-order effects remain unresolved. This policy is selected for numerical consistency,
not advertised as faster.

Workloads are pre-tokenized synthetic token IDs; the CLI's tokenizer field is an unused requested
label in these files (including the inherited Qwen label on Gemma runs), not tokenizer execution.
The actual model checksum, workload IDs/hash, and prompt lengths determine the experiment.

The expected tradeoff is more projection dispatches and less cross-request weight reuse. The stage
trace and code establish the numerical mechanism; wall-clock results measure total cost. Full
Xcode counter inspection remains unavailable, so bandwidth, occupancy, instruction order, power,
and launch-versus-memory attribution are not claimed. Allocation measurements are MLX active/peak
memory, not process RSS. Three trials and 3/24 measured requests per row do not certify production
p95/p99 behavior; raw observations and within-run percentile summaries are retained.

## Hardening and limitations

- Final full regression sweep: **119 tests passed** on the final runtime sources, including the
  three evidence-integrity rejection tests added after the initial 116-test sweep. The
  [machine-readable test report](../benchmarks/results/batch-invariant-m3-max-2026-10-01/test-results.xml)
  retains individual case outcomes.
- Portable dispatch tests cover invalid modes before device initialization, decode-only selection,
  shared reconstruction, and direct-kernel selection beyond sixteen rows.
- Two-family Metal tests cover exact independent logits at page boundaries in FP16, direct INT8,
  reconstructed INT8, and composed dequantization; attention still receives the full batch.
- Injected projection failure cancels all scheduled requests and reclaims their physical pages.
- Capacity-aware arrival tests prevent the earlier development driver from submitting more live
  requests than the runtime's bounded admission contract permits.
- Existing artifact, exporter, reference, scheduler, cache, precision-policy, and GPU-lease tests pass.
- The portable evidence auditor rejects tampered hashes, directory-escaping paths, stale runtime
  sources, incomplete/duplicate A/B coverage, changed workload bytes, dropped benchmark tokens,
  and non-reclaimed cache state. It preserves the failed INT8 quality flag separately.
- C++ host and UBSan executables pass. Local ASan startup and Linux sanitizer CI remain unverified,
  as previously documented. The CUDA implementation has no new GPU validation or performance claim.

This checkpoint closes numerical batching consistency **in an explicit tested mode**. It does not
add a serving API, prefix cache, speculation, distributed routing, or W4A16. The next quality task
is a calibration-only algorithm that accounts for interacting quantization errors, frozen before
regression. A faster shape-invariant projection path would require independent numerical and
controlled performance gates; it cannot silently replace this single-request reduction contract.
