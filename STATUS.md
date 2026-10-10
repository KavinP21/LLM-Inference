# Implementation status

Updated October 10, 2026. The measured device is an Apple M3 Max with MLX 0.32.2.

## FP16 runtime

Qwen2 and Gemma 3 text decoders execute through the shared artifact reader,
model-adapter interface, scheduler, and physical KV-page store. Long prompts use
chunked prefill; runnable requests share batched decode. Custom Metal kernels have
selectable MLX fallbacks. Cancellation and completion release request resources.

| Check | Result | Report |
| --- | --- | --- |
| Qwen2.5-0.5B reference | Four fixed prompts, 32 exact greedy tokens each; first-token cosine ≥ 0.999 | [Foundation](docs/mlx-foundation-results.md) |
| Gemma 3 1B reference | Four first-token numerical gates pass; one of four continuations diverges after 25 tokens | [Gemma](docs/mlx-gemma3-results.md) |
| Paging and 32K | Both models process 32,766 prompt tokens and two outputs; full reclamation | [Qwen](docs/mlx-32k-results.md), [Gemma](docs/mlx-gemma3-results.md) |
| Custom kernels | Independent numerical checks, fallback comparisons, workload-specific measurements | [Metal](docs/mlx-metal-results.md) |
| Controlled performance | September 23 baseline/full matrix; five warmups and three repetitions | [Measurements](docs/resume-benchmark-results.md) |
| Rowwise decode | Opt-in projections for same-artifact consistency across batch shapes | [Policy](docs/decode-numerics.md), [results](docs/mlx-batch-numerics-results.md) |

The October 6 local regression passed all **365 Python tests without skips**.
Portable C++ host tests also passed after a fresh Clang C++20 build. The local suite
uses synthetic GPU-backed references; it does not rerun every historical
experiment on the downloaded official models.
The [QA summary](benchmarks/results/repository-qa-2026-10-06/summary.json) and
[JUnit report](benchmarks/results/repository-qa-2026-10-06/test-results.xml) record this check.

## Prefix caching

Opt-in FP16 MLX prefix reuse pins immutable pages independently of active requests,
uses copy-on-write tails, and reserves private/copy capacity before admission.
Namespaces, execution identity, and bounded LRU retention constrain reuse.

All eight registered Qwen/Gemma parity, concurrency, and 32K resource gates pass.
The initial run exposed an unsupported fused Qwen short-tail dispatch. A capability
check and eight Metal regressions cover the fix; fresh validation passed.
**Timing and final performance qualification remain pending.**

See the [API](docs/prefix-cache.md), [results](docs/mlx-prefix-cache-results.md),
and [frozen measurement procedure](docs/prefix-cache-handoff.md).

## Quantization experiments

Version-3 artifacts support W8A16, per-channel scales, mixed precision, and direct
Metal or reconstructed native projections. Calibration methods include
block-diagonal compensation, cached-decode coverage, scale fitting, and coordinate
refinement. **Strict held-out quality is rejected.** The latest Qwen candidate
matches 21 of 25 continuations and changes 6 of 800 cached decisions. Lower local
loss does not establish exact generation parity. FP16 remains the default.

- [Initial INT8](docs/mlx-int8-results.md) and [hardening](docs/mlx-int8-hardening-results.md)
- [Second-order fitting](docs/mlx-second-order-results.md)
- [Cached calibration](docs/mlx-cached-calibration-results.md)
- [Scale fitting](docs/mlx-scale-aware-results.md) and [refinement](docs/mlx-refined-results.md)
- [Qwen held-out rejection](docs/mlx-qwen-validation-results.md)

The [evidence index](benchmarks/results/README.md) links saved runs and source archives.

## Speculation and task agents

Opt-in greedy MLX speculation verifies proposed blocks and physically clears
rejected K/V. The frozen Qwen/Gemma reports show 1.08–4.06x median speedups across
six fixed 64-token workloads; all 36 measured continuations match independent
canonical greedy references. This is workload-specific evidence with an explicit
FP16 numerical boundary, not a universal losslessness or full-task speed claim.
See [speculative decoding](docs/speculative-decoding.md).

Task orchestration now supports bounded hierarchies, messages, dependencies,
supervised local processes, authenticated remote replicas, journaled recovery,
scoped workspace tools and operator-selected completion criteria. Only one
physical Mac was available. Model-quality diagnostics, including failures and
the slower successful document delegation, are retained in
[agent results](docs/agent-results.md).

The implementation regression passed **827 Python tests without skips** in
29.09 seconds, including synthetic Metal references and real loopback transport.
A fresh Clang C++20 host build passed. No CUDA device tests or multi-host
measurements were run. Final frozen-source task evaluation is separate from
these control tests.

## Remaining limits

- CUDA Qwen execution, cuBLASLt plans, kernels, arenas, and tests are implemented.
  RTX correctness, Compute Sanitizer, and NVIDIA performance are unverified.
- Metal attention stacks live layer-pages before launch because MLX arrays are
  immutable. Page packing limits single-request performance.
- Prefill processes one request chunk per iteration; no packed multi-prompt prefill.
- Gemma sliding layers bound attention work but retain pages until completion.
- A 32K prompt plus two outputs checks execution and reclamation, not long-context
  semantic quality or sustained decoding latency.
- Prefix reuse is engine-local. Workers explicitly forget terminal request history;
  direct engine callers can use `forget` after consuming a request's final state.
- Sampling, native W4A16 and tensor/pipeline parallel execution are unimplemented.
- Speculation is opt-in greedy MLX execution with an explicit numerical contract.
  Replica workers and task orchestration are implemented; multi-host/GPU deployments
  and general task quality require separate qualification.
