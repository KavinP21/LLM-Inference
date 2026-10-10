# Orchestration and inference serving

In ML research, orchestration coordinates experiments, data preparation, jobs,
hardware, dependencies, retries and results. In inference engineering, it routes
requests and manages model replicas, resource limits, cancellation and health.
In a task agent system, it additionally coordinates goals, tools, messages,
delegated work and acceptance checks. These are different layers.

Forge's existing scheduler orchestrates token generation and paged K/V memory.
The new [agent runtime](agent-runtime.md) adds task orchestration above it:
logical agents share supervised model replicas, can create bounded children,
communicate through a durable journal, inspect/edit a workspace, and check
completion against operator-selected criteria. A remote replica can generate
for the same coordinator; file tools execute in the coordinator's workspace.

## Relationship to other engines

| System | Relevant emphasis | Relationship to Forge |
| --- | --- | --- |
| vLLM | Broad inference serving, request scheduling, distributed inference and speculative methods | An established serving engine; Forge is a smaller, inspectable runtime with a focused model/device surface. |
| SGLang | Efficient serving and structured generation, including distributed prefill/decode and speculative methods | Serving and structured generation overlap; durable task goals and tool effects are a higher layer. |
| TensorRT-LLM | NVIDIA-oriented optimized inference, parallel execution and speculative methods | Forge's locally measured path is MLX/Metal; its CUDA implementation remains device-unqualified here. |

The official [vLLM speculative documentation](https://docs.vllm.ai/en/v0.22.0/features/speculative_decoding/),
[SGLang disaggregated serving documentation](https://docs.sglang.ai/backend/pd_disaggregation.html),
[SGLang speculative documentation](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
and [TensorRT-LLM feature list](https://nvidia.github.io/TensorRT-LLM/key-features.html)
describe those serving capabilities. Their existence does not establish a
performance comparison: no matched vLLM/SGLang/TensorRT-LLM benchmark was run.

Forge's benefit is explicit, reviewable implementations of model artifacts,
scheduling, numerical execution, K/V lifetime, speculative verification and
task coordination. It is not a claim of new speculative-decoding algorithms,
production-scale serving parity, or better general agent reasoning.

Whole-model replicas and tensor parallelism are also different. Forge can route
independent work to multiple replicas, but each replica must fit its whole
model on a device. Splitting one forward pass across GPUs is not implemented.
Only one physical Mac was available for this work; cross-machine/CUDA task
quality and network deployment remain unmeasured.
