# MLX foundation validation — 2026-09-22

This report records the acceptance evidence for the correctness-first Apple backend. The working
tree was based on Git commit `a615b927e1e69b5170b90e1380ca1ec1ac272f1f`; these changes were still
uncommitted when the measurements were captured.

## Environment

| Item | Value |
| --- | --- |
| Host | Apple M3 Max |
| Metal architecture | `applegpu_g15s` |
| Operating system | macOS 26.5 arm64 |
| Python | 3.12.10 |
| MLX | 0.32.2 |
| NumPy | 2.5.3 |
| PyTorch reference | 2.14.0 |
| Transformers reference | 5.17.0 |

## Model artifact

| Item | Value |
| --- | --- |
| Source | `Qwen/Qwen2.5-0.5B-Instruct` |
| Tensor entries | 291 |
| File bytes | 988,084,480 |
| Tensor-data bytes | 988,065,536 |
| Tensor-data SHA-256 | `87dde32c2f28ffbcb70016efea4332c4306e3f72c64f9df131def7654bff6c45` |

## Correctness acceptance

The four prompts in `benchmarks/prompts.json` were evaluated for 32 generated tokens using raw
greedy argmax. Qwen's downloaded generation configuration specifies a repetition penalty, so the
reference command explicitly sets it to `1.0`; otherwise the reference and runtime would implement
different decoding policies.

| Measurement | Result |
| --- | --- |
| Exact greedy continuations | 4/4 |
| Matching generated tokens | 128/128 |
| First-token top-1 agreement | 4/4 |
| Minimum logit cosine similarity | 0.9999914 |
| Maximum logit cosine similarity | 0.9999950 |

The complete Python suite reports 16 passing tests: 12 portable tests and four tests that execute
on Metal. The existing C++ host test binary also passes, and the C++ model inspector accepts the
same exported official checkpoint.

## Benchmark smoke test

One warmup and one measured trial were run with four simultaneous requests and eight output tokens
per request. The final smoke run produced approximately 81.06 generated tokens/second and median
TPOT of 43.29 ms. These numbers verify the runner, metrics schema, multi-request lifecycle, and cleanup;
they are **not performance claims**. The current implementation executes active requests
sequentially and uses contiguous K/V arrays with untiled attention.

The next performance baseline must be collected only after batched decode and physical paged K/V
storage land, with the full warmup/trial matrix described in `docs/benchmarking.md`.
