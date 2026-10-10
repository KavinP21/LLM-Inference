"""Portable contract tests; fake model outputs are not model-quality evidence."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import struct
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from forge_llm.qwen35_checkpoint import (
    Qwen35Checkpoint,
    inspect_checkpoint,
    validate_config,
)
from forge_llm.qwen35_engine import (
    FIXED_STATE_BYTES,
    KV_BYTES_PER_TOKEN,
    Qwen35Engine,
    Qwen35Runtime,
    _load_runtime,
    split_response,
)


def config():
    return {
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "text_config": {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "eos_token_id": 248044,
            "attention_bias": False,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
            "in_channels": 3,
        },
    }


class Array:
    def __init__(self, shape, name="float16", *, backing_bytes=None):
        self.shape = tuple(shape)
        self.dtype = SimpleNamespace(name=name, size=4 if name == "float32" else 2)
        self.nbytes = math.prod(self.shape) * self.dtype.size
        self.backing_bytes = backing_bytes or self.nbytes

    def __getitem__(self, indices):
        shape = list(self.shape)
        for index, selector in enumerate(
            indices if isinstance(indices, tuple) else (indices,)
        ):
            if isinstance(selector, slice):
                shape[index] = len(range(*selector.indices(shape[index])))
        return Array(shape, self.dtype.name, backing_bytes=self.backing_bytes)


class KV:
    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values


class Recurrent:
    def __init__(self):
        self.cache = [None, None]
        self.left_padding = self.lengths = None

    @property
    def state(self):
        return self.cache


class Fake:
    def __init__(self, *, prompt_length=7, outputs=(42, 248046), decoded="answer"):
        self.prompt_length, self.outputs, self.decoded = (
            prompt_length,
            list(outputs),
            decoded,
        )
        self.calls, self.preparations, self.templates, self.caches, self.samples = (
            [],
            [],
            [],
            [],
            [],
        )
        self.resets = self.clears = self.syncs = self.detaches = 0
        self.prediction_count = 0
        self.corrupt = None
        self.failure = None
        self.tokenizer = SimpleNamespace(
            apply_chat_template=self.template,
            decode=lambda ids, **kw: self.decoded,
            chat_template="trained-template",
        )
        self.processor = SimpleNamespace(
            apply_chat_template=self.template, tokenizer=self.tokenizer
        )
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(_rope_deltas=None, _position_ids=None)
        )

    def template(self, messages, **kwargs):
        self.templates.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return "rendered prompt"

    def prepare(self, prompt, images, videos, limits):
        self.preparations.append((prompt, images, videos, dict(limits)))
        return {
            "input_ids": np.array([list(range(1, self.prompt_length + 1))]),
            "attention_mask": np.ones((1, self.prompt_length), dtype=np.int32),
        }

    def embed(self, inputs):
        self.model.language_model._position_ids = "current request"
        self.model.language_model._rope_deltas = "current deltas"
        return {
            "inputs_embeds": Array((1, self.prompt_length, 2560)),
            "position_ids": Array((3, 1, self.prompt_length), "float32"),
            "rope_deltas": "current deltas",
        }

    def make_cache(self):
        caches = [KV() if (i + 1) % 4 == 0 else Recurrent() for i in range(32)]
        self.caches.append(caches)
        return caches

    def forward(self, tokens, caches, options):
        self.calls.append((list(tokens), dict(options)))
        if self.failure is not None:
            raise self.failure
        for cache in caches:
            if isinstance(cache, KV):
                cache.offset += len(tokens)
                capacity = math.ceil(cache.offset / 256) * 256
                cache.keys = Array((1, 4, capacity, 256))
                cache.values = Array((1, 4, capacity, 256))
            else:
                cache.cache = [
                    Array((1, 3, 8192), backing_bytes=(len(tokens) + 3) * 8192 * 2),
                    Array((1, 32, 128, 128), "float32"),
                ]
        if self.corrupt:
            self.corrupt(caches)
        index = self.prediction_count
        if not options.get("skip_logits", False):
            self.prediction_count += 1
        return SimpleNamespace(logits=None, tokens=self.outputs, prediction_index=index)

    def greedy(self, output):
        return [output.tokens[min(output.prediction_index, len(output.tokens) - 1)]]

    def sample(self, output, history, options):
        self.samples.append((list(history), dict(options)))
        return self.greedy(output)

    def detach(self, value):
        self.detaches += 1
        return Array(value.shape, value.dtype.name)

    def reset(self):
        self.resets += 1
        self.prediction_count = 0
        self.model.language_model._position_ids = (
            self.model.language_model._rope_deltas
        ) = None

    def synchronize(self):
        self.syncs += 1

    def clear(self):
        self.clears += 1

    def runtime(self, checkpoint):
        return Qwen35Runtime(
            self.model,
            self.processor,
            self.tokenizer,
            self.make_cache,
            self.prepare,
            self.embed,
            self.forward,
            self.greedy,
            lambda values: None,
            self.synchronize,
            self.clear,
            self.reset,
            kv_type=KV,
            arrays_type=Recurrent,
            sample=self.sample,
            detach_array=self.detach,
        )


def engine(fake=None, **options):
    fake = fake or Fake()
    checkpoint = Qwen35Checkpoint(
        Path("/test"),
        config(),
        {},
        {"model_data_sha256": "model", "processor_signature": "processor"},
        123,
        (),
    )
    return Qwen35Engine(
        "/test",
        _checkpoint_loader=lambda path: checkpoint,
        _runtime_loader=fake.runtime,
        **options,
    ), fake


def test_generation_owns_all_cache_state_and_forgets_bounded_history():
    model, fake = engine()
    prepared = model.prepare_chat([{"role": "user", "content": "test"}])
    request = model.submit(prepared, 4)
    first = model.step()
    assert first[0].token == 42 and not first[0].finished
    assert (
        model.stats()["kv_persistent_bytes"]
        == FIXED_STATE_BYTES + 256 * KV_BYTES_PER_TOKEN
    )
    assert model._cache[0].cache[0].backing_bytes == model._cache[0].cache[0].nbytes
    assert model.step()[0].finished
    assert model.stats()["active_requests"] == 0
    assert model.stats()["kv_persistent_bytes"] == 0
    assert all(
        (c.keys is None and c.values is None)
        if isinstance(c, KV)
        else c.cache == [None, None]
        for c in fake.caches[0]
    )
    assert fake.model.language_model._rope_deltas is None
    with pytest.raises(RuntimeError, match="forget"):
        model.submit([1], 1)
    model.forget(request)
    assert model._history == model._output == []
    model.close()
    model.close()
    assert model.model is model.tokenizer is model.processor is None


def test_prefill_chunked_guard_and_cancellation_release_every_layer():
    model, fake = engine(Fake(prompt_length=600))
    prepared = model.prepare_chat([{"role": "user", "content": "test"}])

    def guard():
        if len(fake.calls) >= 1:
            raise InterruptedError("cancel prefill")

    model.set_request_guard(guard)
    with pytest.raises(InterruptedError, match="prefill"):
        model.submit(prepared, 8)
    assert len(fake.calls) == 1 and len(fake.calls[0][0]) == 256
    assert model.stats()["kv_reserved_bytes"] == 0 and model._terminal is None
    assert fake.model.language_model._position_ids is None
    fake.prompt_length = 7
    assert (
        model.generate_chat([{"role": "user", "content": "retry"}], 2).text == "answer"
    )


def test_cancel_decode_and_failures_are_reusable_without_stale_positions():
    model, fake = engine()
    request = model.submit([1, 2], 5)
    model.step()
    model.cancel(request)
    model.cancel(request)
    model.forget(request)
    fake.failure = RuntimeError("forward failed")
    with pytest.raises(RuntimeError, match="forward failed"):
        model.submit([1], 3)
    assert model._terminal is None and model.stats()["kv_reserved_bytes"] == 0
    fake.failure = None
    assert model.generate([1, 2], 2) == [42, 248046]
    assert fake.resets >= 6


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda c: setattr(c[3], "offset", 1.0),
        lambda c: setattr(c[7], "offset", 999),
        lambda c: setattr(c[3], "keys", Array((1, 3, 256, 256))),
        lambda c: setattr(c[3], "values", None),
        lambda c: c[0].cache.__setitem__(1, Array((1, 32, 128, 128), "float16")),
        lambda c: c[0].cache.__setitem__(0, Array((1, 4, 8192))),
        lambda c: c[0].cache.__setitem__(1, None),
    ],
)
def test_runtime_geometry_drift_fails_and_discards_all_state(corrupt):
    model, fake = engine()
    fake.corrupt = corrupt
    with pytest.raises(RuntimeError):
        model.submit([1, 2], 3)
    assert model._active is model._terminal is None
    assert model.stats()["kv_persistent_bytes"] == 0
    fake.corrupt = None
    assert model.generate([1], 1) == [42]


def test_admission_context_state_history_and_foreign_prepared_prompt():
    model, fake = engine(
        max_model_length=1024,
        kv_cache_bytes=FIXED_STATE_BYTES + 256 * KV_BYTES_PER_TOKEN,
    )
    with pytest.raises(ValueError, match="state"):
        model.submit(list(range(1, 257)), 2)
    with pytest.raises(ValueError, match="context"):
        model.submit([1] * 1024, 1)
    assert not fake.caches
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    other, _ = engine()
    with pytest.raises(ValueError, match="belong"):
        other.submit(prepared, 1)
    request = model.submit(prepared, 4)
    with pytest.raises(ValueError, match="history"):
        model._reserve_history(1)
    model.cancel(request)
    model.forget(request)


def test_actor_ownership_and_read_only_health():
    model, _ = engine()
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="actor"):
            pool.submit(model.generate, [1], 1).result()
        info = pool.submit(model.build_info).result()
    assert (
        info["supports_images"]
        and info["supports_video"]
        and not info["supports_audio"]
    )
    assert info["kv_bytes_per_token"] == 32768


def test_sampling_history_and_greedy_speculation_validation():
    model, fake = engine(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
    model.generate([1, 2], 2)
    assert fake.samples[0][0] == [1, 2]
    assert fake.samples[1][0] == [1, 2, 42]
    assert fake.samples[0][1]["presence_penalty"] == 1.5
    with pytest.raises(ValueError, match="greedy"):
        engine(temperature=0.7, speculative={"draft_tokens": 4})
    with pytest.raises(ValueError, match="speculative"):
        engine(speculative={"draft_tokens": True})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_length": True},
        {"kv_cache_bytes": 1},
        {"prefill_chunk_size": 257},
        {"max_video_frames": 3},
        {"max_video_seconds": float("nan")},
        {"top_k": True},
        {"presence_penalty": 3},
        {"temperature": True},
    ],
)
def test_resource_limit_validation_precedes_model_loading(kwargs):
    with pytest.raises(ValueError):
        engine(**kwargs)


def test_native_tool_and_thinking_template_options_are_exact():
    model, fake = engine()
    specs = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read file",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
    ]
    model.prepare_chat(
        [
            {"role": "assistant", "content": "<tool_call>call</tool_call>"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "continue"},
        ],
        tools=specs,
        enable_thinking=True,
    )
    specs[0]["function"]["name"] = "mutated"
    messages, kwargs = fake.templates[0]
    assert (
        kwargs["enable_thinking"] is True
        and kwargs["tools"][0]["function"]["name"] == "read"
    )
    assert messages[1]["role"] == "tool"


def test_images_and_video_are_all_retained_in_multiturn_preparation(
    tmp_path, monkeypatch
):
    from PIL import Image

    picture = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video-for-processor-contract")
    captures = []

    class Capture:
        def __init__(self, path):
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, field):
            return {1: 320, 2: 240, 3: 20, 4: 10}[field]

        def release(self):
            self.released = True

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=Capture,
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FRAME_COUNT=3,
            CAP_PROP_FPS=4,
        ),
    )
    model, fake = engine()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(picture)},
                {"type": "text", "text": "inspect"},
            ],
        },
        {"role": "assistant", "content": "red"},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video)},
                {"type": "text", "text": "compare"},
            ],
        },
    ]
    prepared = model.prepare_chat(messages)
    assert fake.preparations[0][1:3] == ([str(picture)], [str(video)])
    assert (
        len(prepared.media_fingerprints) == 2
        and prepared.media_fingerprints[0][1]
        == hashlib.sha256(picture.read_bytes()).hexdigest()
    )
    assert captures[0].released
    assert fake.preparations[0][3]["video_frames"] == 32
    assert fake.templates[0][0][0]["content"][0]["type"] == "image"


def test_media_bounds_urls_decode_and_expanded_context(tmp_path):
    model, fake = engine(media_bytes_limit=4, max_model_length=16)
    given = tmp_path / "big.png"
    given.write_bytes(b"too many bytes")
    for path in (str(given), "https://example.invalid/image.png"):
        with pytest.raises(ValueError, match="media|local"):
            model.prepare_chat([{"role": "user", "content": "q"}], images=[path])
    assert not fake.preparations
    fake.prompt_length = 16
    with pytest.raises(ValueError, match="expanded"):
        model.prepare_chat([{"role": "user", "content": "q"}])


def test_chat_context_clamp_uses_expanded_media_length_and_keeps_api_strict(tmp_path):
    from PIL import Image

    image = tmp_path / "context-image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    fake = Fake(prompt_length=15, outputs=(42,))
    model, _ = engine(fake, max_model_length=16)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image", "image": str(image)},
            ],
        }
    ]
    with pytest.raises(ValueError, match="context"):
        model.generate_chat(messages, 32768)
    assert not fake.calls and model._prepared is None
    reply = model.generate_chat(messages, 32768, clamp_to_context=True)
    assert reply.input_tokens == 15 and reply.output_tokens == 1
    assert reply.truncated and reply.finish_reason == "length"
    assert model.stats()["active_requests"] == 0
    assert (
        model.stats()["kv_reserved_bytes"] == model.stats()["kv_persistent_bytes"] == 0
    )
    assert fake.preparations[-1][1] == [str(image)]
    model.close()


@pytest.mark.parametrize("limit", [0, -1, True, "2"])
def test_chat_clamp_does_not_repair_invalid_generation_limits(limit):
    model, fake = engine(max_model_length=16)
    with pytest.raises(ValueError, match="generation limit"):
        model.generate_chat(
            [{"role": "user", "content": "q"}], limit, clamp_to_context=True
        )
    assert not fake.calls and model._prepared is None
    assert model.stats()["active_requests"] == 0
    model.close()


def test_clamped_chat_still_rejects_a_prompt_that_fills_the_context():
    model, fake = engine(Fake(prompt_length=16), max_model_length=16)
    with pytest.raises(ValueError, match="expanded"):
        model.generate_chat(
            [{"role": "user", "content": "q"}], 4, clamp_to_context=True
        )
    assert not fake.calls and model._prepared is None
    model.close()


def test_discard_prepared_releases_context_rejected_payload_and_allows_reuse():
    model, fake = engine(Fake(prompt_length=15), max_model_length=16)
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    array = weakref.ref(prepared.inputs["input_ids"])
    with pytest.raises(ValueError, match="context"):
        model.submit(prepared, 2)
    assert model._prepared is prepared and array() is not None
    before = fake.syncs, fake.resets, fake.clears
    model.discard_prepared()
    assert model._prepared is None and prepared.inputs == {} and array() is None
    assert (fake.syncs, fake.resets, fake.clears) == tuple(
        value + 1 for value in before
    )
    assert fake.model.language_model._position_ids is None
    model._guard()
    with pytest.raises(ValueError, match="belong"):
        model.submit(prepared, 1)
    fake.prompt_length = 7
    assert (
        model.generate_chat([{"role": "user", "content": "retry"}], 2).text == "answer"
    )
    model.close()


def test_discard_and_chat_preserve_an_active_or_unforgotten_request():
    model, fake = engine()
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    request = model.submit(prepared, 4)
    guard = model._guard
    syncs = fake.syncs
    with pytest.raises(RuntimeError, match="finish and forget"):
        model.discard_prepared()
    with pytest.raises(RuntimeError, match="finish and forget"):
        model.generate_chat([{"role": "user", "content": "other"}], 2)
    assert model._active == request and prepared.inputs and model._guard is guard
    assert fake.syncs == syncs
    model.cancel(request)
    assert prepared.inputs == {}
    with pytest.raises(RuntimeError, match="finish and forget"):
        model.discard_prepared()
    model.forget(request)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="actor"):
            pool.submit(model.discard_prepared).result()
    model.discard_prepared()
    model.close()


def test_chat_admission_error_survives_cleanup_failure_with_payload_released():
    model, fake = engine(Fake(prompt_length=15), max_model_length=16)

    def failed_sync():
        raise RuntimeError("injected cleanup failure")

    model._runtime.synchronize = failed_sync
    with pytest.raises(ValueError, match="context") as error:
        model.generate_chat([{"role": "user", "content": "q"}], 2)
    assert any("cleanup failure" in note for note in error.value.__notes__)
    assert model._prepared is None and model._active is None
    assert fake.clears > 0 and fake.model.language_model._position_ids is None
    model._runtime.synchronize = fake.synchronize
    model.close()


def test_custom_eos_is_per_request_and_does_not_replace_default_stops():
    model, fake = engine(Fake(outputs=(0, 248046)))
    assert model.generate([1, 2], 4, eos_token_ids=[0]) == [0]
    assert model._default_eos == frozenset({248044, 248046})
    assert model.generate([1, 2], 4) == [0, 248046]
    assert model._eos == model._default_eos
    assert model.stats()["active_requests"] == model.stats()["kv_persistent_bytes"] == 0
    model.close()


def test_forward_failure_invalidates_prepared_arrays_but_keeps_diagnostic_token_ids():
    model, fake = engine()
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    expected_ids = prepared.input_ids
    fake.failure = RuntimeError("injected forward failure")
    with pytest.raises(RuntimeError, match="injected forward failure"):
        model.generate(prepared, 4)
    assert prepared.inputs == {} and prepared.input_ids == expected_ids
    assert prepared.input_tokens == len(expected_ids)
    assert model._prepared is None and model._active is None
    assert (
        model.stats()["kv_reserved_bytes"] == model.stats()["kv_persistent_bytes"] == 0
    )
    fake.failure = None
    assert (
        model.generate_chat([{"role": "user", "content": "retry"}], 2).text == "answer"
    )
    model.close()


def test_incomplete_thinking_never_becomes_final_or_tool_action():
    result = split_response(
        "reason <tool_call>danger</tool_call>",
        enable_thinking=True,
        finish_reason="length",
        input_tokens=4,
        output_tokens=8,
    )
    assert result.text == "" and not result.thinking_complete and result.truncated
    completed = split_response(
        "reason</think>\n<tool_call>safe</tool_call>",
        enable_thinking=True,
        finish_reason="stop",
        input_tokens=4,
        output_tokens=9,
    )
    assert (
        completed.reasoning == "reason"
        and completed.text == "<tool_call>safe</tool_call>"
    )
    assert completed.thinking_complete and not completed.truncated


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_dim", True),
        ("num_hidden_layers", 31),
        ("mamba_ssm_dtype", "float16"),
        ("tie_word_embeddings", 1),
        ("eos_token_id", 248046),
    ],
)
def test_config_rejects_unvalidated_hybrid_metadata(field, value):
    selected = config()
    selected["text_config"][field] = value
    with pytest.raises(ValueError):
        validate_config(selected)


def checkpoint_fixture(tmp_path):
    selected = config()
    (tmp_path / "config.json").write_text(json.dumps(selected))
    for name in (
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        (tmp_path / name).write_text("{}")
    header = {}
    offset = 0

    def tensor(name, shape, dtype="BF16"):
        nonlocal offset
        size = math.prod(shape) * (4 if dtype == "F32" else 2)
        header[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    for i in range(32):
        if (i + 1) % 4 == 0:
            for name in ("k_proj", "v_proj"):
                tensor(
                    f"language_model.model.layers.{i}.self_attn.{name}.scales",
                    [1024, 40],
                )
        else:
            tensor(
                f"language_model.model.layers.{i}.linear_attn.conv1d.weight",
                [8192, 4, 1],
            )
            tensor(f"language_model.model.layers.{i}.linear_attn.A_log", [32], "F32")
    tensor("vision_tower.patch_embed.proj.weight", [1024, 2, 16, 16, 3])
    raw = json.dumps(header).encode()
    with (tmp_path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)
    return tmp_path


def test_checkpoint_fingerprints_include_vision_processor_and_weight_bytes(tmp_path):
    source = checkpoint_fixture(tmp_path)
    first = inspect_checkpoint(source)
    assert first.unchanged() and first.weight_bytes > 5 << 20
    (source / "video_preprocessor_config.json").write_text('{"fps":1}')
    second = inspect_checkpoint(source)
    assert not first.unchanged()
    assert (
        first.identity["processor_signature"] != second.identity["processor_signature"]
    )
    assert first.identity["model_data_sha256"] == second.identity["model_data_sha256"]
    with (source / "model.safetensors").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    third = inspect_checkpoint(source)
    assert second.identity["model_data_sha256"] != third.identity["model_data_sha256"]


def test_checkpoint_index_cannot_escape_or_silently_replace_a_shard(tmp_path):
    source = checkpoint_fixture(tmp_path)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"anything": "../other.safetensors"}})
    )
    with pytest.raises(ValueError, match="local weight shards"):
        inspect_checkpoint(source)


def test_malformed_safetensors_offsets_fail_before_any_runtime(tmp_path):
    source = checkpoint_fixture(tmp_path)
    raw = json.dumps(
        {"fake": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 3]}}
    ).encode()
    (source / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"000"
    )
    with pytest.raises(ValueError, match="byte range"):
        inspect_checkpoint(source)


def test_prefill_evaluates_only_the_final_prompt_token_logits():
    model, fake = engine(Fake(prompt_length=600))
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    request = model.submit(prepared, 1)
    assert [len(tokens) for tokens, _ in fake.calls] == [256, 256, 87, 1]
    assert [options["skip_logits"] for _, options in fake.calls] == [
        True,
        True,
        True,
        False,
    ]
    assert all(
        options["position_ids"].shape == (3, 1, 600) for _, options in fake.calls
    )
    assert model.step()[0].token == 42
    model.forget(request)


def test_processor_failure_and_mutated_preparation_cannot_leak_or_forward():
    model, fake = engine()
    prepared = model.prepare_chat([{"role": "user", "content": "q"}])
    prepared.inputs["input_ids"][0, 0] = 900
    with pytest.raises(ValueError, match="changed"):
        model.submit(prepared, 3)
    assert not fake.calls
    fake.prompt_length = model.max_model_length
    with pytest.raises(ValueError, match="expanded"):
        model.prepare_chat([{"role": "user", "content": "q"}])
    assert model._prepared is None and fake.clears > 0


def test_legitimate_unaligned_kv_capacity_and_idempotent_history_release():
    model, fake = engine()
    request = model.submit([1] * 263, 4)
    for cache in model._cache:
        if isinstance(cache, KV):
            cache.keys = Array((1, 4, 510, 256))
            cache.values = Array((1, 4, 510, 256))
    model._observe_cache()
    release = model._reserve_history(24 << 20)
    assert model.stats()["speculative_reserved_bytes"] == 24 << 20
    release()
    release()
    assert model.stats()["speculative_reserved_bytes"] == 0
    model.cancel(request)
    model.forget(request)


def test_production_bridge_uses_offline_strict_load_and_separate_modality_masks(
    monkeypatch,
):
    """Exercise the actual runtime bridge with CPU arrays and fake upstream modules."""
    import forge_llm.qwen35_engine as module

    load_calls, prepare_calls, vision_calls, scatter_calls = [], [], [], []
    tokenizer = SimpleNamespace(
        chat_template="trained", apply_chat_template=lambda *args, **kw: "prompt"
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=SimpleNamespace(),
        video_processor=SimpleNamespace(),
    )

    def vision(values, grid):
        vision_calls.append((values.copy(), grid.copy()))
        return values, None

    vision.patch_embed = SimpleNamespace(
        proj=SimpleNamespace(weight=SimpleNamespace(dtype=np.dtype("float16")))
    )
    lm = SimpleNamespace(
        make_cache=lambda: [],
        model=SimpleNamespace(
            embed_tokens=lambda ids: np.zeros((*ids.shape, 1), dtype=np.float16)
        ),
        get_rope_index=lambda ids, images, videos, mask: (
            np.zeros((3, 1, ids.shape[1]), dtype=np.int32),
            np.array([[-2]]),
        ),
        _position_ids="old",
        _rope_deltas="old",
    )

    def scatter(features, embeddings, ids, image_token, video_token):
        scatter_calls.append((image_token, video_token))
        embeddings = embeddings.copy()
        embeddings[(ids == image_token) | (ids == video_token)] = features
        return embeddings, None

    model = SimpleNamespace(
        language_model=lm,
        vision_tower=vision,
        config=SimpleNamespace(image_token_index=248056, video_token_index=248057),
        merge_input_ids_with_image_features=scatter,
    )

    def load_model(path, **kwargs):
        load_calls.append(("model", kwargs))
        return model

    def load_processor(path, **kwargs):
        load_calls.append(("processor", kwargs))
        return processor

    def prepare(processor, **kwargs):
        prepare_calls.append(kwargs)
        return {"input_ids": np.array([[1]])}

    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        gpu="gpu",
        default_device=lambda: "gpu",
        stream=lambda stream: contextlib.nullcontext(),
        eval=lambda *args: None,
        synchronize=lambda *args: None,
        clear_cache=lambda: None,
        array=np.array,
        int32=np.int32,
        argmax=np.argmax,
        all=np.all,
        isfinite=np.isfinite,
    )
    modules = {
        "mlx": SimpleNamespace(core=mx),
        "mlx.core": mx,
        "mlx_vlm.generate.common": SimpleNamespace(generation_stream="actor-stream"),
        "mlx_vlm.models.cache": SimpleNamespace(ArraysCache=Recurrent, KVCache=KV),
        "mlx_vlm.sample_utils": SimpleNamespace(
            make_sampler=lambda **kwargs: None,
            make_logits_processors=lambda **kwargs: [],
        ),
        "mlx_vlm.utils": SimpleNamespace(
            load_model=load_model, load_processor=load_processor, prepare_inputs=prepare
        ),
    }
    for prefix in ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.models"):
        package = ModuleType(prefix)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, prefix, package)
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        module, "version", lambda name: "0.7.6" if name == "mlx-vlm" else "0.32.2"
    )
    runtime = _load_runtime(
        SimpleNamespace(path=Path("/local-only"), unchanged=lambda: True)
    )
    assert all(
        options["local_files_only"] and options["trust_remote_code"] is False
        for _, options in load_calls
    )
    assert load_calls[0][1]["strict"] is True
    runtime.prepare(
        "prompt",
        ["image"],
        ["video"],
        {"image_pixels": 1024, "video_pixels": 2048, "video_frames": 8},
    )
    assert prepare_calls[0]["max_frames"] == 8 and prepare_calls[0]["frame_factor"] == 2
    assert (
        processor.image_processor.max_pixels == 1024
        and processor.video_processor.max_pixels == 2048
    )
    runtime.reset_positions()
    assert lm._position_ids is lm._rope_deltas is None
    inputs = {
        "input_ids": np.array([[248057, 1, 248056, 248056]]),
        "pixel_values": np.array([[7], [8]], dtype=np.float32),
        "pixel_values_videos": np.array([[9]], dtype=np.float32),
        "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2]]),
        "video_grid_thw": np.array([[1, 2, 2]]),
    }
    embedded = runtime.embed(inputs)
    assert embedded["inputs_embeds"].reshape(-1).tolist() == [9, 0, 7, 8]
    assert scatter_calls == [(248056, 248056), (248057, 248057)]
    assert len(vision_calls) == 2 and all(
        values.dtype == np.float16 for values, _ in vision_calls
    )
    assert embedded["rope_deltas"].tolist() == [[-2]]
