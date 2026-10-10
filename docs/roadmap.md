# Next work

The immediate work is measurement and hardware qualification, followed by focused
runtime extensions.

| Work | Completion criterion |
| --- | --- |
| Prefix-cache performance | Run the registered 24-report disabled/cold/primed matrix, recompute metrics, repeat regression, and finish the independent audit. Separate insertion and reuse costs. |
| NVIDIA qualification | Run Qwen on RTX, verify reference logits/tokens, run Compute Sanitizer, and collect a controlled matrix and Nsight traces. |
| Bounded request lifecycle | Implemented terminal `forget` API and worker cleanup; continue sustained production-load qualification. |
| Reduce page-pack overhead | Profile layer-page stacking separately and compare alternatives at matched output, workload, and memory accounting. |
| Quantization quality | Register a new calibration-only experiment and untouched held-out corpus. Preserve failed policies and strict gates before INT8 promotion or W4A16. |

The original prefix-cache run is frozen. Its [measurement procedure](prefix-cache-handoff.md)
requires recorded sources and inputs. New code requires a new result directory;
see the [evidence index](../benchmarks/results/README.md).

Greedy speculative decoding and task orchestration are now implemented, with
[bounded local evidence](agent-results.md) and a separate
[speculative numerical/performance contract](speculative-decoding.md). Further
qualification should use diverse tasks, source checks, resource/latency budgets,
and independent task sets rather than promoting these small development fixtures.
Streaming serving and tensor/pipeline multi-GPU execution require separate designs
and validation.
