# Qwen3.5-4B daily driver

The default model for `create_engine()` and the desktop chat is the complete
`mlx-community/Qwen3.5-4B-4bit` checkpoint, including its vision tower. Forge
loads it locally through **MLX-VLM 0.7.6**, with MLX 0.32.2. This is an explicit
upstream model adapter with Forge-owned admission, request lifecycle, hybrid
state accounting, media handling and task-worker integration. It does not use
Forge's native Qwen2 graph, physical paged cache or custom attention kernels.

## Setup

Use native Apple Silicon Python 3.12 with Metal access. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF" \
  python -m pip install -e '.[mlx,qwen35,agents,test]'
python -m forge_llm.prepare_qwen35
python tools/forge_chat.py
```

The preparation command downloads revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` into the Git-ignored directory
`models/qwen3.5-4b-mlx`. It validates the nested configuration and full tensor
layout and fingerprints weights, tokenizer and processors. Run
`python -m forge_llm.prepare_qwen35 --verify-only` to verify an existing copy.
The checkpoint contains 3,034,300,695 weight bytes, including 297 vision tensors.
Weights are not committed. Ordinary execution is offline and does not trust
remote model code. The earlier MLX-LM runner remains pinned to 0.28.3.

```python
from forge_llm import create_engine

with create_engine() as engine:
    answer = engine.generate_chat(
        [{"role": "user", "content": "Explain hybrid attention in three sentences."}],
        max_new_tokens=256,
        enable_thinking=False,
    )
    print(answer.text)
```

The return value separates final `text`, `reasoning`, token counts,
`finish_reason`, `truncated` and `thinking_complete`. Thinking is off by default;
enable it explicitly for a larger reasoning budget. An exhausted budget inside
the thinking segment returns no final answer. The desktop exposes Thinking,
image/video attachment, Stop, Clear and the older 0.5B/7B profiles. It retains
private attachment copies for visual follow-ups and removes them on Clear,
model switch or shutdown.

## Images, video and sampling

```python
with create_engine() as engine:
    answer = engine.generate_chat([{
        "role": "user",
        "content": [
            {"type": "image", "image": "/absolute/path/photo.png"},
            {"type": "video", "video": "/absolute/path/clip.mp4"},
            {"type": "text", "text": "Read the image and describe the video in order."},
        ],
    }], max_new_tokens=256)
    print(answer.text)
```

The processor expands visual tokens before context admission. Images and videos
use the actual vision tower, modality-specific token placement and multimodal
rotary positions. Defaults bound a request to four images, one video, 32 sampled
frames and 120 seconds of video; individual image inputs are limited to 16 million
pixels. The processing limits are roughly one million pixels per image and two
million per video frame. Local regular files are required. Remote URLs and audio
input are unsupported.

Greedy decoding is the reproducible default. Ordinary decoding also accepts
`temperature`, `top_p`, `top_k` and `presence_penalty` in `create_engine` options:

```python
engine = create_engine(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
```

These controls are separate from the optional greedy speculative path. Sampling
and a nonzero presence penalty cannot be combined with speculation.

## Hybrid state and limits

The 32 text layers contain 24 GatedDeltaNet blocks and eight full-attention
blocks. Full attention uses 16 query heads, four KV heads and **256-dimensional
heads**; dividing hidden size by query heads would give the wrong cache geometry.
The recurrent state stays FP32. A single sequence requires approximately 48 MiB
of recurrent state, 1.125 MiB of convolution state and 32,768 bytes of attention
KV per retained token. Weights and transient vision/compute arrays are separate
from this persistent state budget.

Defaults are a 32,768-token context, 2 GiB persistent state budget and 256-token
prefill chunks. Prefill computes vocabulary logits only for the last prompt token.
The checkpoint advertises 262,144 positions, but full-length execution and semantic
quality have not been qualified here. Limits can be configured up to checkpoint
capacity and are checked against expanded prompts and actual cache bytes.

One active request runs per engine/physical replica. Multiple logical agents share
supervised replicas; each replica owns a complete checkpoint. Qwen3.5 does not
inherit native Forge continuous batching or prefix reuse. `submit`, `step`,
`cancel`, `forget`, `stats`, `close` and `generate` are available; failure and
cancellation release both recurrent and KV state. Model operations belong to one
owner thread, including loading, preparation and shutdown.

## Tools and agents

```bash
forge-agents --state-dir .forge run \
  --config examples/agents-qwen35.json --workspace . \
  --task 'Inspect the architecture and have two children independently review the resource limits.'

forge-agents --state-dir .forge run \
  --config examples/agents-qwen35.json --workspace . \
  --image /absolute/path/chart.png --task 'Read the chart and check the calculation.'
```

Add `--allow-write --allow-tests` for authorized repository edits and test
execution. `--video` also accepts a local short clip. The example supervises two
32K/2 GiB replicas on this Mac. It uses the model's trained XML function syntax,
schema-checked arguments and bounded call batches. Explanatory text around valid
calls is accepted only for this explicitly advertised dialect. It is never
evaluated as code or interpreted as a tool call.

Worker requests carry bounded, hashed media bytes rather than client filesystem
paths. The journal stores each media blob once and retains references through
children, visual follow-ups and resume. Worker materializations are private and
request-owned. Context and global token admission cover expanded visual inputs.
Resume binds the checkpoint, processor, runner versions and adapter implementation.
Remote replicas can use the existing authenticated worker protocol; only one
physical Mac was available for testing.

## Optional speculative decoding

```python
with create_engine(speculative={
    "draft_tokens": 4, "verification_mode": "block", "adaptive": True,
}) as engine:
    answer = engine.generate_chat(
        [{"role": "user", "content": "Repeat this paragraph exactly: ..."}],
        max_new_tokens=256,
    )
```

History n-grams provide proposals. Verification uses upstream Qwen3.5's recurrent
transaction protocol, not ordinary KV trimming: accepted convolution/recurrent
states and KV cursors are committed together, and rejected state is released.
`verification_mode="sequential"` uses ordinary one-token target execution.
Draft blocks are bounded to one through eight tokens; additional live histories
are reserved before execution. The downloaded checkpoint has no MTP tensors, so
there is no MTP drafter.

Block evaluation and single-token evaluation can have different floating-point
reductions. Passing fixed greedy comparisons does not establish universal
canonical equality. Speculation stays **off by default**: copying can benefit,
while less repetitive workloads can be slower. See the saved qualification reports
for exact workloads, source versions, outputs and limits.

## Executed qualification

The initial full-model checks matched an independent ordinary MLX-VLM reference
on three prompts, including a 6,860-token input. The total continuation coverage
was 17 tokens; this is not sustained long-context decoding or original-FP16 parity.
Synthetic image OCR, a visual follow-up, red-then-blue video order, mixed image/video,
thinking separation, truncated thinking and request cleanup passed.

On frozen source `7239c91`, both single and delegated coding/document fixtures
passed their independent gates. Each delegated case executed two completed
children. Coding took 25.38/53.85 seconds and document analysis 8.36/51.25 seconds;
startup was excluded. Delegation was slower. The earlier delegated document failure
is retained alongside the revised run. These are two development fixtures, not
broad autonomous-task qualification.

The [evidence directory](../benchmarks/results/qwen35-2026-10-10/) records separate
source identities for core inference, task workflows and speculation. Portable
tests cover malformed checkpoints/media, context/state admission, every accepted
prefix length, abort/error cleanup, parser validation, journaling and UI ownership.
CUDA Qwen3.5, audio, tensor/pipeline parallelism, streaming serving and full 262K
qualification remain outside this implementation.

Architecture and trained formats come from the [official Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B)
and [MLX-VLM v0.7.6](https://github.com/Blaizzy/mlx-vlm/tree/v0.7.6).
