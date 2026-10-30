"""Native adapters against mlx-vlm on the real checkpoints.

These need the downloaded models (``python -m instinct.prepare_models``) and
mlx-vlm as a reference, so they skip anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")
transformers = pytest.importorskip("transformers")

from instinct.mlx_engine import MlxEngine

MODELS = Path(__file__).resolve().parents[1] / "models"
CHECKPOINTS = ["gemma-4-26b-a4b-mlx", "qwen3.5-4b-mlx"]


@pytest.fixture(scope="module", params=CHECKPOINTS)
def pair(request):
    path = MODELS / request.param
    if not (path / "config.json").is_file():
        pytest.skip(f"{request.param} isn't downloaded")
    from mlx_vlm.utils import load

    reference, _ = load(str(path))
    engine = MlxEngine(
        path,
        max_num_sequences=4,
        kv_cache_bytes=4 << 30,
        prefix_cache_bytes=1 << 30,
        max_model_length=16384,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    yield engine, reference.language_model, tokenizer
    engine.close()


def _prompt(tokenizer, text):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _reference_logits(model, ids):
    return np.array(model(mx.array([ids])).logits[0, -1].astype(mx.float32))


def test_logits_match_mlx_vlm_inside_and_past_the_window(pair):
    engine, reference, tokenizer = pair
    short = _prompt(tokenizer, "Write a Python function that reverses a linked list.")
    long = (short * 80)[:2500]
    # bf16 through 30 MoE layers: the long prompt drifts a little further.
    for ids, bound in [(short, 0.999), (long, 0.998)]:
        ours, theirs = engine.prefill_logits(ids), _reference_logits(reference, ids)
        cosine = float(
            np.dot(ours, theirs) / np.linalg.norm(ours) / np.linalg.norm(theirs)
        )
        assert cosine > bound
        assert ours.argmax() == theirs.argmax()


def test_greedy_continuation_matches_mlx_vlm(pair):
    engine, reference, tokenizer = pair
    from mlx_vlm.models.cache import make_prompt_cache

    ids = _prompt(tokenizer, "List three prime numbers and say why each is prime.")
    cache = make_prompt_cache(reference)
    logits = reference(mx.array([ids]), cache=cache).logits[0, -1]
    expected = []
    for _ in range(24):
        token = int(mx.argmax(logits).item())
        expected.append(token)
        logits = reference(mx.array([[token]]), cache=cache).logits[0, -1]
    assert engine.generate(ids, 24) == expected


def test_batched_users_start_like_running_alone(pair):
    # Batching changes the matmul kernels' shapes, so a bf16 near-tie can
    # eventually flip a token; the openings must agree.
    engine, _, tokenizer = pair
    prompts = [
        _prompt(tokenizer, "Explain recursion to a ten-year-old."),
        _prompt(tokenizer, "What does the `yield` keyword do in Python?"),
        _prompt(tokenizer, "Name the planets in order."),
    ]
    alone = [engine.generate(p, 16) for p in prompts]
    ids = [engine.submit(p, 16) for p in prompts]
    while engine.has_work():
        engine.step()
    for request_id, solo in zip(ids, alone):
        assert engine.request(request_id).output[:8] == solo[:8]


def test_follow_up_turn_from_the_prefix_cache_matches_a_cold_start(pair, monkeypatch):
    # Covers recurrent-state snapshots on Qwen3.5 and sliding-window pages on
    # Gemma 4 (the entry runs past the 1,024-token window). The reply's K/V was
    # written one token at a time and a cold start computes it as one chunk,
    # so compare logits rather than long greedy runs. Two cold prefills of
    # these tokens with 512- and 2,048-token chunks agree to about 0.996.
    engine, _, tokenizer = pair
    first = _prompt(tokenizer, "Write a long story about a dragon who learns to code.")
    reply = engine.generate(first, 1100, cache_namespace="warm")
    follow = (
        first + reply[:-1] + tokenizer.encode(" The end.", add_special_tokens=False)
    )
    seen = []
    choose = engine._choose
    monkeypatch.setattr(
        engine,
        "_choose",
        lambda requests, logits: (
            seen.append(np.array(logits[0])) or choose(requests, logits)
        ),
    )
    hits = engine.stats()["prefix_cache"]["hits"]
    engine.generate(follow, 1, cache_namespace="warm")
    assert engine.stats()["prefix_cache"]["hits"] == hits + 1
    warm, cold = seen[-1], engine.prefill_logits(follow)
    assert (
        float(np.dot(warm, cold) / np.linalg.norm(warm) / np.linalg.norm(cold)) > 0.99
    )


def test_image_input_matches_mlx_vlm(pair):
    engine, reference, tokenizer = pair
    if getattr(engine.model, "vision", None) is None:
        pytest.skip("this model has no native vision tower")
    from instinct.backends.gemma4_vision import load_image
    from mlx_vlm.utils import load, prepare_inputs

    image = Path(__file__).resolve().parents[1] / "docs" / "images" / "app-chat.jpg"
    ours = engine.model.vision.encode(load_image(image))
    model, processor = load(str(MODELS / "gemma-4-26b-a4b-mlx"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What app is this?"},
            ],
        }
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = prepare_inputs(
        processor,
        images=[str(image)],
        prompts=prompt,
        image_token_index=model.config.image_token_id,
        add_special_tokens=False,
    )
    theirs = model.encode_image(inputs["pixel_values"]).reshape(ours.shape)
    a = np.array(ours.astype(mx.float32))
    b = np.array(theirs.astype(mx.float32))
    cosine = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    assert cosine.mean() > 0.99

    ids = inputs["input_ids"].flatten().tolist()
    start = ids.index(model.config.image_token_id)
    extra = {
        k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")
    }
    expected = int(mx.argmax(model(inputs["input_ids"], **extra).logits[0, -1]).item())
    request = engine.submit(ids, 1, images=[(start, ours)], cache_namespace="image")
    while engine.has_work():
        engine.step()
    assert engine.request(request).output == [expected]
