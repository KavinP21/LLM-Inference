"""CPU-only adapter tests. No optional library imports or model/GPU execution."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from forge_llm.agents import mlx_lm_runner as runner_module
from forge_llm.agents.backends import (
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    LocalForgeBackend,
    RemoteWorkerBackend,
    WorkerConfig,
)
from forge_llm.agents.mlx_lm_runner import MlxLmRunner, _Runtime, inspect_checkpoint
from forge_llm.agents.protocol import ChatMessage
from forge_llm.agents.worker import WorkerHTTPServer, WorkerService


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "local-checkpoint"
    directory.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "max_position_embeddings": 16384,
        "eos_token_id": 151645,
        "quantization": {"bits": 4, "group_size": 64},
        "sliding_window": None,
        "torch_dtype": "bfloat16",
    }
    (directory / "config.json").write_text(json.dumps(config))
    header = {}
    length = 512 * 32 * 2
    for layer in range(48):
        for projection in ("k_proj", "v_proj"):
            offset = len(header) * length
            header[f"model.layers.{layer}.self_attn.{projection}.scales"] = {
                "dtype": "BF16",
                "shape": [512, 32],
                "data_offsets": [offset, offset + length],
            }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\x00" * (len(header) * length)
    )
    return directory


class FakeRuntime:
    def __init__(self, *, tokens=None, delay=0.0, fail=False, prefill_cancel=None):
        self.tokens = tokens or [17, 18, 151645]
        self.delay, self.fail, self.prefill_cancel = delay, fail, prefill_cancel
        self.events = []
        self.caches = []
        self.owner = None

    def own(self):
        owner = threading.get_ident()
        if self.owner is None:
            self.owner = owner
        assert self.owner == owner

    def make_cache(self):
        self.own()
        self.caches = [
            SimpleNamespace(keys=None, values=None, offset=0) for _ in range(48)
        ]
        return self.caches

    def allocate(self, total):
        rounded = (total + 255) // 256 * 256
        for cache in self.caches:
            cache.keys = SimpleNamespace(
                dtype=SimpleNamespace(size=2),
                shape=(1, 4, rounded, 128),
                nbytes=4 * rounded * 128 * 2,
            )
            cache.values = SimpleNamespace(
                dtype=SimpleNamespace(size=2),
                shape=(1, 4, rounded, 128),
                nbytes=4 * rounded * 128 * 2,
            )
            cache.offset = total

    def generate(
        self,
        prompt,
        *,
        max_tokens,
        prompt_cache,
        prefill_step_size,
        prompt_progress_callback,
    ):
        self.own()
        assert prompt_cache is self.caches and prefill_step_size > 0
        try:
            prompt_progress_callback(0, len(prompt))
            self.allocate(len(prompt))
            if self.prefill_cancel:
                self.prefill_cancel()
            prompt_progress_callback(len(prompt), len(prompt))
            if self.fail:
                self.fail = False
                raise RuntimeError("injected native forward failure")
            for index in range(max_tokens):
                time.sleep(self.delay)
                self.allocate(len(prompt) + index + 1)
                self.events.append("yield")
                yield self.tokens[index % len(self.tokens)], None
        finally:
            self.events.append("generator_closed")

    def synchronize(self):
        self.own()
        self.events.append("sync")

    def clear(self):
        self.own()
        self.events.append("clear")

    def load(self, info):
        return _Runtime(
            SimpleNamespace(),
            self.make_cache,
            self.generate,
            self.synchronize,
            self.clear,
            "0.28.3",
            "0.32.2",
        )


def runner(checkpoint, fake=None, **options):
    fake = fake or FakeRuntime()
    return MlxLmRunner(
        checkpoint,
        max_model_length=options.pop("max_model_length", 8192),
        kv_cache_bytes=options.pop("kv_cache_bytes", 1 << 30),
        _runtime_loader=fake.load,
        **options,
    ), fake


def test_checkpoint_hashes_actual_bytes_and_validates_kv_dimensions(checkpoint):
    first = inspect_checkpoint(checkpoint)
    assert len(first.identity["model_data_sha256"]) == 64 and first.unchanged()
    file = checkpoint / "model.safetensors"
    with file.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    changed = inspect_checkpoint(checkpoint)
    assert changed.identity["model_data_sha256"] != first.identity["model_data_sha256"]
    assert (
        changed.identity["model_config_sha256"] == first.identity["model_config_sha256"]
    )
    config = json.loads((checkpoint / "config.json").read_text())
    config["num_key_value_heads"] = 8
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="validated Qwen3"):
        inspect_checkpoint(checkpoint)


def test_checkpoint_load_set_changes_invalidate_fingerprint(checkpoint):
    snapshot = inspect_checkpoint(checkpoint)
    extra = checkpoint / "model-extra.safetensors"
    extra.write_bytes(b"extra shard not fingerprinted")
    assert not snapshot.unchanged()
    extra.unlink()
    assert snapshot.unchanged()


def test_adapter_greedy_eos_reclaims_cache_and_forgets_history(checkpoint):
    engine, fake = runner(checkpoint)
    try:
        for _ in range(12):
            request = engine.submit([1, 2], 6, [151645])
            events = []
            while not events or not events[-1].finished:
                events += engine.step()
            assert [event.token for event in events] == [17, 18, 151645]
            assert (
                engine.stats()["kv_reserved_bytes"]
                == engine.stats()["kv_persistent_bytes"]
                == 0
            )
            assert engine.stats()["last_kv_peak_bytes"] == 256 * 98304
            assert all(
                cache.keys is None and cache.values is None for cache in fake.caches
            )
            engine.forget(request)
            assert engine._terminal is None
        assert fake.events[-3:] == ["generator_closed", "sync", "clear"]
    finally:
        engine.close()


def test_cache_rounding_and_lookahead_are_in_admission_budget(checkpoint):
    engine, _ = runner(checkpoint, kv_cache_bytes=256 * 98304)
    try:
        with pytest.raises(ContextLengthError, match="persistent KV"):
            engine.submit([1] * 255, 2, [])
        assert engine._active is None and engine._cache == []
        request = engine.submit([1] * 255, 1, [])
        assert engine.step()[0].finished
        engine.forget(request)
        with pytest.raises(ContextLengthError, match="context"):
            engine.submit([1] * 8192, 1, [])
    finally:
        engine.close()


@pytest.mark.parametrize("chunk", [0, True, 257, 511, 700, 2049])
def test_prefill_chunks_cannot_defeat_native_cache_rounding(checkpoint, chunk):
    with pytest.raises(ValueError, match="prefill_step_size"):
        WorkerConfig(
            str(checkpoint),
            str(checkpoint),
            runner="mlx_lm",
            engine_options={"prefill_step_size": chunk},
        )
    with pytest.raises(ValueError):
        runner(checkpoint, prefill_step_size=chunk)


def test_prefill_guard_and_failure_close_generator_before_release(checkpoint):
    cancelled = False

    def guard():
        if cancelled:
            raise GenerationCancelledError("cancelled during prefill")

    def cancel_in_prefill():
        nonlocal cancelled
        cancelled = True

    engine, fake = runner(checkpoint, FakeRuntime(prefill_cancel=cancel_in_prefill))
    try:
        engine.set_request_guard(guard)
        request = engine.submit([1, 2], 4, [])
        with pytest.raises(GenerationCancelledError):
            engine.step()
        assert fake.events == ["generator_closed", "sync", "clear"]
        assert engine.stats()["active_requests"] == 0
        assert engine.stats()["last_kv_peak_bytes"] == 256 * 98304
        engine.forget(request)
    finally:
        engine.close()
    engine, fake = runner(checkpoint, FakeRuntime(fail=True))
    try:
        request = engine.submit([1], 2, [])
        with pytest.raises(RuntimeError, match="native forward"):
            engine.step()
        assert all(cache.keys is None for cache in fake.caches)
        engine.forget(request)
        request = engine.submit([1], 1, [])
        assert engine.step()[0].finished
        engine.forget(request)
    finally:
        engine.close()


def test_adapter_cancel_and_close_do_not_allow_concurrent_requests(checkpoint):
    engine, fake = runner(checkpoint)
    request = engine.submit([1], 10, [])
    with pytest.raises(RuntimeError, match="previous"):
        engine.submit([1], 1, [])
    with pytest.raises(RuntimeError, match="active"):
        engine.forget(request)
    engine.step()
    engine.cancel(request)
    engine.forget(request)
    engine.close()
    engine.close()
    assert engine._runtime.model is None
    assert all(cache.keys is None for cache in fake.caches)


def test_wrong_kv_dtype_is_rejected_and_cache_is_reclaimed(checkpoint):
    class WrongDtype(FakeRuntime):
        def allocate(self, total):
            super().allocate(total)
            self.caches[0].keys.dtype.size = 4

    engine, fake = runner(checkpoint, WrongDtype())
    try:
        request = engine.submit([1], 1, [])
        with pytest.raises(RuntimeError, match="dtype"):
            engine.step()
        assert all(cache.keys is None for cache in fake.caches)
        assert fake.events[-2:] == ["sync", "clear"]
        engine.forget(request)
    finally:
        engine.close()


def test_failed_generator_construction_returns_no_retained_request(checkpoint):
    class FailedConstruction(FakeRuntime):
        def generate(self, *args, **kwargs):
            raise RuntimeError("generator construction failed")

    engine, fake = runner(checkpoint, FailedConstruction())
    try:
        with pytest.raises(RuntimeError, match="construction"):
            engine.submit([1], 1, [])
        assert engine._active is engine._terminal is None
        assert engine._cache == []
        assert fake.events == ["sync", "clear"]
    finally:
        engine.close()


def test_platform_and_version_fail_before_optional_execution_import(
    checkpoint, monkeypatch
):
    info = inspect_checkpoint(checkpoint)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        runner_module._load_runtime(info)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runner_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        runner_module,
        "version",
        lambda name: "0.28.2" if name == "mlx-lm" else "0.32.2",
    )
    with pytest.raises(RuntimeError, match="requires version 0.28.3"):
        runner_module._load_runtime(info)


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "cuda"},
        {"local_files_only": False},
        {"speculative": {}},
        {"engine_options": {"decode_mode": "rowwise"}},
        {"engine_options": {"prefix_cache_bytes": 100}},
        {"engine_options": {"custom_metal": True}},
        {"runner": "invented"},
    ],
)
def test_external_config_rejects_incompatible_or_ignored_features(checkpoint, options):
    config = {
        "model": str(checkpoint),
        "tokenizer": str(checkpoint),
        "runner": "mlx_lm",
        **options,
    }
    with pytest.raises(ValueError):
        WorkerConfig(**config)


class Tokenizer:
    chat_template = "fake-native-template"
    eos_token_id = 151645

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        assert 151645 not in tokens
        return (
            '<tool_call>{"name":"finish","arguments":{"result":"fixture"}}</tool_call>'
        )


def actor(checkpoint, fake=None, **options):
    fake = fake or FakeRuntime()
    config = WorkerConfig(
        str(checkpoint),
        str(checkpoint),
        runner="mlx_lm",
        backend="mlx",
        native_tool_calls=True,
        **options,
    )
    backend = LocalForgeBackend(
        config,
        engine_factory=lambda: runner(checkpoint, fake)[0],
        tokenizer_factory=Tokenizer,
    )
    return backend, fake


def test_existing_actor_external_runner_deadline_cancellation_and_native_decode(
    checkpoint,
):
    async def run():
        backend, fake = actor(checkpoint)
        try:
            await backend.start()
            outputs = await asyncio.gather(
                *[
                    backend.generate([ChatMessage("user", "x")], 4, f"actor-{i}")
                    for i in range(3)
                ]
            )
            assert all(
                output.finish_reason == "stop"
                and output.output_tokens == 3
                and output.text.startswith("<tool_call>")
                for output in outputs
            )
            health = backend.health()
            assert health["runner"] == "mlx_lm" and health["runner_version"] == "0.28.3"
            assert health["last_runner_stats"]["kv_persistent_bytes"] == 0
            assert len(health["model_data_sha256"]) == 64
        finally:
            await backend.close()
        backend, fake = actor(
            checkpoint, FakeRuntime(delay=0.02, tokens=[17]), request_timeout=0.01
        )
        try:
            with pytest.raises(GenerationTimeoutError):
                await backend.generate([ChatMessage("user", "x")], 20, "deadline")
            assert all(cache.keys is None for cache in fake.caches)
        finally:
            await backend.close()

    asyncio.run(run())


@contextmanager
def serving(backend):
    service = WorkerService(backend)
    server = None
    try:
        service.start()
        server = WorkerHTTPServer(("127.0.0.1", 0), service, "cpu-runner-test-token")
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        if server:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        service.close()


def test_external_fake_runner_uses_real_native_http_pipeline(checkpoint):
    backend, fake = actor(checkpoint)
    with serving(backend) as url:

        async def run():
            remote = RemoteWorkerBackend(
                url, "cpu-runner-test-token", poll_interval=0.002
            )
            health = await remote.health()
            assert health["runner"] == "mlx_lm" and health["native_tool_calls"]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "description": "Finish.",
                        "parameters": {
                            "type": "object",
                            "properties": {"result": {"type": "string"}},
                            "required": ["result"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            result = await remote.generate_action(
                [ChatMessage("user", "x")], 4, "native-http", tools
            )
            assert result.text.endswith("</tool_call>") and result.output_tokens == 3
            await remote.close()

        asyncio.run(run())
    assert all(cache.keys is None for cache in fake.caches)
