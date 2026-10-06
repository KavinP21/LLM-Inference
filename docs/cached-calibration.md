# Cached-decode calibration and bounded policy repair

This checkpoint replaces a selection proxy, not the inference engine or its
defaults. FP16, the app, `decode_mode="batched"`, greedy argmax, packed tensor
format, and the direct Metal kernels are unchanged. The former second-order
policies and evidence remain reproducible and are not overwritten.

## What changed

The previous selector compared a few whole-prompt prefill logits. A one-token
decode executes different projections and attention, with a persistent page
history. A selector should see that actual path before its policy is frozen.

`cached_decode.replay_cached` runs the normal chunked prompt operator, followed
by `decode_paged_batch` with source continuation tokens. Generated position 0
is the final prompt row; position p feeds source token p-1 at absolute position
`prompt_length + p - 1`. The final source token is a target, not an input.
Every intermediate operation is materialized. No failed candidate output is
fed back into the teacher-forced prefix.

Replay owns a separate reservation-aware pool and physical device store. Pages
grow lazily across 16-token boundaries. It enforces the engine's context/KV
limits, requires an idle engine, and releases its reservation and device arrays
on success or exceptions. It never submits requests, changes scheduler state,
or evicts production requests. This is offline diagnostic infrastructure, not
a new live-serving feature or a latency measurement.

Tests compare replay logits **bitwise** with scheduler-produced logits for both
families, FP16/direct/reconstructed INT8, chunked prompts, and page boundaries.
The full-model reference collection additionally verifies that replay decisions
equal independent source generation before any fitting proceeds.

## Frozen method

The weight quantizer is still [the block-diagonal second-order recipe](second-order-quantization.md):
64-feature blocks, damping 0.01, stable activation order, fixed per-output RTN
scales, FP16-rounded reconstruction, and no-worse-local-objective RTN fallback.
It is not full [GPTQ](https://arxiv.org/abs/2210.17323). Source activations now
include up to eight evenly spaced rows **per prompt chunk** and the single input
row at **every cached decode position**. Q/K/V and gate/up share group samples.
Covariance archives retain the old recipe identifier for exact rederivation;
the new **selection** algorithm is `block_second_order_cached_repair_v1`.

The [32 calibration prompts](../benchmarks/cached-calibration-prompts.json),
[fresh 25-prompt regression](../benchmarks/cached-regression-prompts.json), all
parameters, and selection bounds are fixed before evaluation. Normalized and
tokenized overlap guards exclude the original and preceding regressions.
Held-out text is used only by that guard, never by the evaluator or repair loop.
Fresh outputs are unseen at freeze; familiar task categories do not establish
semantic independence or broad model-quality certification.

The deterministic search is:

1. Rank projections by block reconstruction loss per eligible FP16 byte.
2. Starting from FP16, test up to eight new projections **together with every
   already selected projection** on four evenly spaced calibration prompts.
   Observe all 32 cached positions of each probe prompt, not only early logits.
3. Minimize top-1 changes, then numerical-floor shortfall, then mean squared
   relative logit error. Names break ties. Stop when at least 25% of eligible
   projection bytes are INT8; this is not 25% of whole-model storage.
4. Evaluate that seed on all 32 calibration prompts and all 32 positions.
5. For at most three repair rounds, propose single-matrix swaps from the four
   highest local-loss selected projections to the eight lowest local-loss
   unselected projections. Reject any swap that breaks the 25% byte floor.
6. Rank proposals on the fixed probes, evaluate the best four on the **full**
   calibration corpus, and accept only a strictly improved full-corpus score.
   A probe-only improvement, equal score, or held-out outcome cannot authorize
   a swap. Stop when strict calibration decisions pass or no checked swap wins.
7. Independently rerun free greedy generation and all cached logits, then seal
   the complete FP16/INT8 partition and its search/statistics identities.

The algorithm caches aggregate scores, not KV histories. It reports every
distinct trial, seed, proposal count, full trial, accepted repair, and finite
evaluation bound. It can miss better combinations, stop at a local minimum,
or fail exact parity. Calibration failure is an explicit nonzero saved outcome,
not an execution error or permission to weaken acceptance.

Projection aliases are rejected by this calibration's eligible-byte accounting.
The exporter itself safely splits source projection aliases after independent
conversion, so distinct recipes or mixed dtypes cannot silently share storage.
Ordinary tied embedding/head aliases remain intact. Official artifacts have
unique projection storage, so this does not change their inference layout.

## Quality and numerical contracts

Both family policies, archives, artifacts, manifests, source artifacts, corpus
files, runtime source hash, and workflow tools are frozen before **any** held-out
inference. Freeze checks use complete-file hashes, not just claimed data hashes
in headers. New files are created exclusively; existing results are preserved.

Regression compares original FP16, candidate, and composed-dequantization
baseline on the original and fresh 25 prompts. It checks 32 free greedy tokens
and every cached teacher-forced logit position. Every packed value/scale is
independently rederived; every retained source tensor must remain identical.
Same-partition RTN artifacts are controls, not alternative acceptance targets.

The strict gates are unchanged: exact source greedy outputs, cosine at least
0.999, candidate-versus-composed greedy parity, same-artifact batch consistency,
and complete KV cleanup/resource checks. Cached numerical coverage is stronger
than the previous five prefill positions. A high cosine does not excuse a token
mismatch. Lossy quantization and cross-kernel evaluation are separate issues.

`probe_int8_numerics.py` checks all quantized projections at one/two/eight rows:
custom and composed reconstruction versus independent NumPy, followed by native
and direct projections on identical inputs. Native rowwise reconstruction is
the existing opt-in consistency contract; direct results remain experimental.
Matching reconstructed values with differing projected outputs isolates linear
evaluation differences, **not** a particular compiler instruction or universal
reduction-order explanation. Nothing changes scales, precision, or tie handling
to make a failed regression disappear. MLX custom-kernel APIs are documented in
the [official guide](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html);
the actual local MLX version is recorded in results rather than inferred from
that rolling documentation.

`array_equal` fields in quality/projection diagnostics establish numeric equality,
not signed-zero-sensitive byte identity, despite the supplemental quality field's
`bitwise_kernel_logit_rows` name. `verify_native_cached_bytes.py` separately hashes
dtype, shape, and raw bytes for every cached native/composed row on both corpora.
It is a post-freeze diagnostic only: it cannot change the policies, source-quality
gates, or the direct-kernel contract. Same-artifact batch tests already compare
byte-sensitive row fingerprints. This distinction is tested with +0/-0 examples.

## Reproduction

Use the existing two official FP16 artifacts and offline tokenizers. From a GPU-
enabled Apple Silicon process with the project environment and `PYTHONPATH=python`:

```bash
.venv/bin/python benchmarks/run_cached_checkpoint.py \
  --output-dir benchmarks/results/cached-NEW-RUN \
  --artifact-dir models/cached-NEW-RUN
```

Or execute `--phase calibrate`, `validate`, `benchmark`, and `audit` separately
against the same directory. Wait for each process to exit before another GPU
job. The inherited exclusive lease serializes cooperating drivers, not unrelated
applications. Do not change runtime/tool files after the freeze; use a new run.

Validation includes 12 quality reports (9,600 cached rows), four arrival matrices,
four boundary matrices, four 32K resource gates, and two projection diagnostics.
Benchmarking uses fresh processes, rotated execution order, 128/1,024-token
synthetic prompts, concurrency 1/8, 32 outputs, five full warmups, and three trials
for FP16/composed/direct/reconstructed paths: 32 results total. The same source
hash binds quality and performance. Tests/profiles never overlap timed runs.

Audit recomputes quality summaries/gates from per-case observations, arrival/
boundary comparisons from per-request fingerprints, and 32K gates from raw
resource records. It requires the exact benchmark identities/shapes and cleanup,
indexes JSON checksums, and refuses absent, stale, or internally inconsistent
evidence. A complete failed experiment exits 1; missing or inconsistent evidence
is an error. There is no release tag or W4 advancement on a failed strict gate.

For the supplemental native byte comparison, after validation and before timed
benchmarking, run:

```bash
.venv/bin/python benchmarks/verify_native_cached_bytes.py \
  --root benchmarks/results/cached-NEW-RUN
```

The separate reports carry their diagnostic script hash. They were not a new
fitting input or part of the pre-fit eight-tool manifest in the recorded run.

After the final complete test suite and evidence index, the additional readback
binds byte-diagnostic coverage, raw gate recomputation, indexed hashes, complete
before/after files, and the final JUnit report:

```bash
.venv/bin/python benchmarks/audit_cached_checkpoint.py \
  --root benchmarks/results/cached-NEW-RUN --expected-tests NUMBER-OF-TESTS
```

Supply the actual complete-suite count, not a subset. This post-freeze auditor
and its tests are validation tooling, not fitting inputs; its hash is recorded
in `verification.json`. A successful readback can certify that a **failed**
experiment is complete and internally consistent; it does not waive quality.

Unavailable full-Xcode counter inspection, Linux sanitizers, and RTX validation
remain outside this checkpoint. Calibrating differently is not proof of a faster
kernel, broad quality gain, GPU occupancy, bandwidth, or long-context semantics.
