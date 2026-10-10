# Greedy speculative decoding

Forge has an independent `SpeculativeEngine` for FP16 Qwen2 and text-only
Gemma 3 on MLX/Metal. It is a one-request decoder; run independent instances in
worker processes to serve concurrent agent tasks. The existing continuous
batching engine keeps its execution policy.

```python
from forge_llm.speculative import SpeculativeEngine

with SpeculativeEngine("models/qwen2.5-0.5b.engine", draft_tokens=4) as engine:
    result = engine.generate_result(prompt_token_ids, max_new_tokens=128,
                                    eos_token_ids=[151645])
    print(result.tokens, result.finish_reason, result.stats.to_dict())
```

Pass token IDs produced by the target model's tokenizer. The default draft is
`NGramDraft`: it finds an earlier continuation of a repeated history suffix,
using the longest match and then the most recent occurrence. This requires no
second model and is useful when copying provided text, writing repeated code
structures, or continuing repeated formatting. It may produce no proposals for
open-ended reasoning or unfamiliar prose. In that case the target uses its
ordinary single-token decoder. Acceptance feedback reduces the proposal budget
when rejections dominate and increases it after full acceptance.

## Actual verification and rollback

Prefill predicts the first output token. After each round, all committed history
except the latest output token is cached. A round performs **one causal target
forward pass** over that pending token plus up to `draft_tokens` proposals. Row
`i` predicts proposal `i`; the last row predicts a bonus token. The controller
accepts the matching prefix, then emits the target correction at the first
mismatch, or emits the bonus if every proposal matches. It rolls K/V back to the
pending token and accepted prefix before the next round.

Rollback clears both written-slot masks and rejected tensor values. The store's
append implementation adds K/V into zero slots, so only clearing masks would
silently corrupt the replacement token. Page identifiers remain reserved;
whole rejected pages can be materialized again. The KV budget is checked for
prompt plus the full output allowance before execution. Output/EOS/context
limits apply to both drafts and accepted output; provider errors, target errors,
cancellation and normal completion release both target and draft K/V.

`SpeculativeStats` counts proposed/accepted/rejected tokens, actual target
forward calls, verification blocks, rolled-back tokens, draft time, full
request elapsed time, prefill calls and peak target KV bytes. A reduced number
of target calls alone is not evidence of a speedup; acceptance, target block
cost, draft cost and actual wall-clock time all matter.

## Numerical contract

`verification_mode="block"` uses native FP16 `forward_paged_chunk` to verify
all candidates together. It is greedy with respect to those target logits and
causal masks. It does not promise bitwise equality or identical greedy output
for every prompt against `MlxEngine` single-token decoding: FP16 projection and
attention reduction shapes differ between block verification and one-token
execution. A small logit margin can change an argmax even without an algorithmic
error. Accepted K/V is produced by the same block execution policy. Prefix
caches from the continuous engine are not shared.

For an explicit canonical fallback, use `verification_mode="sequential"`.
It validates proposals through canonical `decode_paged_batch` calls and retains
the same accept/rollback state machine, but performs one target call per input
token and offers no speculative acceleration. `draft=NoDraft()` runs the normal
one-token decoder directly, with the same chunked prefill as this engine, and
is the matched timing baseline. The benchmark also independently compares
output tokens with `MlxEngine(decode_mode="rowwise")`.

## Optional draft model

```python
with SpeculativeEngine(
    "target.engine", draft_model_path="small-draft.engine",
    target_tokenizer_fingerprint="sha256-of-target-tokenizer-json",
    draft_tokenizer_fingerprint="sha256-of-draft-tokenizer-json",
    draft_kv_cache_bytes=128 << 20,
) as engine:
    tokens = engine.generate(prompt_token_ids, 128, eos_token_ids)
```

A model draft retains its own paged cache and synchronizes accepted history by
truncating the rejected branch and appending the committed correction. It
executes its proposal tokens sequentially, while the target verifies a block.
Both models must be FP16, use the same model/token-vocabulary family, vocabulary
size and EOS ID, and the draft context limit must cover the target limit.
Distinct tensor-data checksums require equal nonempty tokenizer fingerprints.
Compute those from the actual tokenizer files; they are an explicit caller
contract because engine artifacts currently contain no tokenizer. Equal vocab
sizes alone do not establish compatible token ID meanings. A same-artifact
draft is supported for correctness checks but is unlikely to be faster.

`draft=` also accepts a custom provider with `start(prompt, budget)`,
`propose(history, count, cancelled)` and `finish()`. Providers must respect the
token budget, return valid integer IDs, and check cancellation during expensive
work. The controller rejects invalid providers and frees request caches.

## Cancellation and ownership

`generate_result(..., cancelled=event.is_set)` returns a partial result with
`finish_reason="cancelled"`. Cancellation is checked before/after prefill,
drafting and verification; an already launched Metal block is allowed to finish.
`generate()` raises `GenerationCancelled` with the partial `.result`, so callers
cannot confuse cancellation with successful completion. One instance rejects
simultaneous requests and rejects closing during active generation. Cancel and
wait for the request before closing it. Worker processes should own independent
engine instances and set cancellation events for their requests.

## Validation and reproduction

Portable tests exhaust acceptance positions, bonus generation, EOS, output and
context limits, cancellation, injected failures, invalid drafts, reentrancy and
rollback values/masks. Tiny Metal tests cover both families at 15/16/17 and
31/32/33-token page boundaries, full acceptance and early/middle/late rejection,
replacement writes, draft synchronization, resource failure and cleanup.

```bash
.venv/bin/python -m pytest -q tests/test_speculative.py tests/test_speculative_metal.py
.venv/bin/python benchmarks/benchmark_speculative.py \
  --model models/qwen2.5-0.5b.engine --tokenizer /path/to/local/qwen/tokenizer \
  --chat --output benchmarks/results/new-speculative-run/qwen.json \
  --require-token-parity
```

The benchmark uses real prompt/history proposals, fixed output budgets,
independent canonical outputs, warmups and alternating timing order. It saves
raw output tokens, acceptance, timing and model/source/tokenizer provenance.
Text tasks cover copying, repeated code and arithmetic explanation. Its
`--require-token-parity` option saves the evidence and then fails if any target
greedy token output differs. Oracle/replay draft providers are used in unit
tests only, never as performance evidence.

No CUDA speculative kernel path or multi-request speculative batching is
implemented by this decoder. No remote-device speed or general speedup is
implied by local Metal measurements.

### Local evidence, 2026-10-09

On the Apple M3 Max, MLX 0.32.2, the supplied FP16 Qwen2.5-0.5B-Instruct and
Gemma 3-1B-IT artifacts were each tested on the three text tasks above. Each
variant received one full warmup per prompt, followed by three alternating-order
measurements with a fixed 64-token output allowance. The default n-gram draft
and adaptive four-token proposal ceiling were used. Model loading is outside
timing; prompt processing, drafting, verification, rollback and request cleanup
are inside timing.

| Model | Task | No-draft median, s | Speculative median, s | Speedup | Accepted proposals |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 0.5B | Copy a sentence repeatedly | 0.783 | 0.205 | 3.83x | 100.0% |
| Qwen 0.5B | Repeated Python functions | 0.885 | 0.576 | 1.54x | 51.5% |
| Qwen 0.5B | Arithmetic explanation | 0.791 | 0.696 | 1.14x | 45.0% |
| Gemma 1B | Copy a sentence repeatedly | 1.389 | 0.480 | 2.89x | 100.0% |
| Gemma 1B | Repeated Python functions | 1.414 | 1.072 | 1.32x | 57.6% |
| Gemma 1B | Arithmetic explanation | 1.490 | 1.367 | 1.09x | 35.3% |

Every timed output matched the independently generated canonical `MlxEngine`
greedy token sequence exactly on these tasks: 36 timed trials across both
models, three prompts, two variants and three repetitions. This is measured
parity on these workloads, not a universal guarantee for FP16 blocks. Copying
reduced target decode calls from 63 to 15 for the 64 output tokens; code and
arithmetic accepted fewer proposals and benefited less. With only three
repetitions, the small arithmetic gains should be treated cautiously.

The 64-token outputs are truncated fragments. These are inference-throughput
measurements, not evidence that the models completed the full eight-copy or
eight-function tasks, or that orchestration improved task success. No external
engine comparison or distributed/GPU scaling was performed.

Raw outputs, timing samples, token counts, implementation hashes and provenance:

- [Qwen evidence](../benchmarks/results/speculative-published-2026-10-09/qwen.json)
- [Gemma evidence](../benchmarks/results/speculative-published-2026-10-09/gemma.json)

The final speculative suite passed 105 tests (87 portable, 18 tiny Metal).

### Optional local MLX checkpoint reconstruction

`python -m forge_llm.import_mlx SOURCE_DIRECTORY NEW_ENGINE_PATH` can convert a
local MLX affine 4-bit, group-size-64 Qwen2 checkpoint into a Forge FP16 engine.
It uses `mlx.core.dequantize`, defaults to a CPU stream, verifies the exact
Qwen2 tensor contract and publishes only fully validated output. It refuses to
overwrite either the engine or its provenance manifest, performs no download,
and keeps the source cache read-only.

The resulting artifact is **FP16 reconstructed from quantized weights**, not
the original FP16 checkpoint. The reconstruction does not recover quantization
loss, implement native W4 inference, or establish generation equivalence with
MLX quantized matrix multiplication. Its sidecar records this distinction,
source file/config/tokenizer SHA-256 hashes, the affine scheme, MLX version,
conversion device and every reconstructed matrix. Evaluate model behavior
before relying on it for agent tasks.

Cancellation is sticky after any callback reports it. Draft providers are read
only through their token allowance plus one excess token, so an accidentally
unbounded proposal iterator fails the budget check and releases request caches.
