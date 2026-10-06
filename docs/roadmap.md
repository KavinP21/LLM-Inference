# Next work

The immediate work is measurement and hardware qualification, followed by focused
runtime extensions.

| Work | Completion criterion |
| --- | --- |
| Prefix-cache performance | Run the registered 24-report disabled/cold/primed matrix, recompute metrics, repeat regression, and finish the independent audit. Separate insertion and reuse costs. |
| NVIDIA qualification | Run Qwen on RTX, verify reference logits/tokens, run Compute Sanitizer, and collect a controlled matrix and Nsight traces. |
| Bounded request lifecycle | Release or evict completed records with explicit API semantics; test sustained arrivals and cancellation. |
| Reduce page-pack overhead | Profile layer-page stacking separately and compare alternatives at matched output, workload, and memory accounting. |
| Quantization quality | Register a new calibration-only experiment and untouched held-out corpus. Preserve failed policies and strict gates before INT8 promotion or W4A16. |

The original prefix-cache run is frozen. Its [measurement procedure](prefix-cache-handoff.md)
requires recorded sources and inputs. New code requires a new result directory;
see the [evidence index](../benchmarks/results/README.md).

After prefix-cache qualification, the next substantial feature is greedy speculative
decoding: draft/target verification, rollback-safe page ownership, and measurement
including draft overhead. Serving and multi-GPU execution require separate designs
and validation.
