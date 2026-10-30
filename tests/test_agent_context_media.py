"""Context, durable media and adapter identity fault tests using CPU fakes."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from types import SimpleNamespace

import pytest
from instinct.agents.backends import (
    BackendError,
    LocalModelBackend,
    RemoteWorkerBackend,
    RequestValidationError,
    WorkerConfig,
    WorkerPool,
)
from instinct.agents.media import media_from_file
from instinct.agents.protocol import ChatMessage, Generation
from instinct.agents.runtime import AgentRuntime, RuntimeConfig
from instinct.agents.store import AgentStore, StoreError
from test_agent_media import Runner
from test_agent_workers import TOKEN, serving


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "source.png"
    path.write_bytes(b"bounded control fixture")
    return media_from_file(path, mime="image/png")


def actor(tmp_path, engine):
    (tmp_path / "config.json").write_text(
        '{"model_type": "gemma4", "vision_config": {"model_type": "gemma4_vision"}}'
    )
    return LocalModelBackend(
        WorkerConfig(
            str(tmp_path),
            str(tmp_path),
            runner="chat",
            backend="mlx",
            max_model_length=32768,
            kv_cache_bytes=2 << 30,
            native_tool_calls=True,
        ),
        engine_factory=lambda: engine,
    )


def test_runtime_worker_http_cap_rejects_expanded_prompt_before_submit_and_refunds(
    media, tmp_path
):
    engine = Runner()
    engine.input_tokens = 18000
    with serving(actor(tmp_path, engine)) as (url, backend, service):

        async def run():
            client = RemoteWorkerBackend(url, TOKEN, poll_interval=0.002)
            await client.health()
            assert client.supports_context_limits is True
            pool = WorkerPool({"replica": client})
            runtime = AgentRuntime(
                pool,
                config=RuntimeConfig(
                    max_context_tokens=12000,
                    max_output_tokens=64,
                    max_total_tokens=50000,
                ),
            )
            try:
                result = await runtime.run("Inspect supplied media", media=[media])
                assert result.status == "failed"
                assert result.usage.total_tokens == 0
                assert all(
                    not __import__("pathlib").Path(path).exists()
                    for path in engine.paths
                )
            finally:
                await pool.close()

        asyncio.run(run())


def test_atomic_media_failure_rolls_back_run_agents_messages_and_new_blobs(media):
    with AgentStore() as store:
        original = store._store_media

        def fail_after_blob_insert(items):
            references = original(items)
            if items:
                raise StoreError("simulated disk write failure after inserting blob")
            return references

        store._store_media = fail_after_blob_insert
        runtime = AgentRuntime(SimpleNamespace(), store=store)
        with pytest.raises(StoreError, match="simulated"):
            asyncio.run(runtime.run("Inspect image", run_id="atomic", media=[media]))
        for table in ("runs", "agents", "messages", "media_blobs"):
            assert (
                store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                == 0
            )


def test_legacy_backend_keeps_original_call_signature_when_model_context_fits(media):
    class Legacy:
        supports_native_tools = True
        max_model_length = 16000
        calls = 0

        async def generate_action(self, messages, maximum, request_id, tool_specs):
            self.calls += 1
            return Generation(
                '<tool_call>{"name":"finish","arguments":{"result":"observed"}}</tool_call>',
                2000,
                30,
            )

    backend = Legacy()
    runtime = AgentRuntime(backend)
    result = asyncio.run(runtime.run("Inspect image", media=[media]))
    assert result.status == "completed" and backend.calls == 1
    backend = Legacy()
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_context_tokens=12000))
    result = asyncio.run(runtime.run("Inspect image", media=[media]))
    assert result.status == "failed" and backend.calls == 0
    assert result.usage.total_tokens == 0


def test_context_cap_contract_violation_cannot_be_reported_completed(media):
    class Violating:
        supports_native_tools = True
        supports_context_limits = True
        max_model_length = 32768

        async def generate_action(
            self, messages, maximum, request_id, tool_specs, *, max_context_tokens
        ):
            assert max_context_tokens == 12000
            return Generation(
                '<tool_call>{"name":"finish","arguments":{"result":"false success"}}</tool_call>',
                18000,
                30,
            )

    runtime = AgentRuntime(
        Violating(),
        config=RuntimeConfig(max_context_tokens=12000, max_total_tokens=50000),
    )
    result = asyncio.run(runtime.run("Inspect image", media=[media]))
    assert result.status == "failed" and result.usage.uncertain_tokens > 0


class Bounded:
    supports_context_limits = True
    calls = 0

    async def generate(self, messages, maximum, request_id, *, max_context_tokens=None):
        self.calls += 1
        self.last_limit = max_context_tokens
        return Generation("bounded", 2, 1)


def test_http_context_limit_is_validated_and_part_of_idempotency_fingerprint():
    with serving(Bounded()) as (url, model, service):

        async def run():
            client = RemoteWorkerBackend(url, TOKEN, poll_interval=0.002)
            try:
                first = await client.generate(
                    [ChatMessage("user", "q")], 2, "bounded", max_context_tokens=128
                )
                second = await client.generate(
                    [ChatMessage("user", "q")], 2, "bounded", max_context_tokens=128
                )
                assert first == second and model.calls == 1 and model.last_limit == 128
                with pytest.raises(RequestValidationError, match="different job"):
                    await client.generate(
                        [ChatMessage("user", "q")], 2, "bounded", max_context_tokens=127
                    )
                assert model.calls == 1
                for invalid in (True, 0, 1.5, 1000001):
                    with pytest.raises(RequestValidationError):
                        await client.generate(
                            [ChatMessage("user", "q")],
                            2,
                            "invalid",
                            max_context_tokens=invalid,
                        )
                    with pytest.raises(RequestValidationError):
                        await asyncio.to_thread(
                            client._http,
                            "POST",
                            "/v1/jobs",
                            {
                                "request_id": "unvalidated-client",
                                "messages": [{"role": "user", "content": "q"}],
                                "max_tokens": 2,
                                "max_context_tokens": invalid,
                            },
                        )
                assert model.calls == 1
            finally:
                await client.close()

        asyncio.run(run())


def test_http_legacy_health_does_not_claim_context_capability_or_break_text():
    class Legacy:
        calls = 0

        async def generate(self, messages, maximum, request_id):
            self.calls += 1
            return Generation("legacy", 2, 1)

    with serving(Legacy()) as (url, model, service):

        async def run():
            client = RemoteWorkerBackend(url, TOKEN, poll_interval=0.002)
            try:
                assert (await client.health())["supports_context_limits"] is False
                with pytest.raises(RequestValidationError, match="does not enforce"):
                    await client.generate(
                        [ChatMessage("user", "q")], 2, "cap", max_context_tokens=128
                    )
                assert model.calls == 0
                assert (
                    await client.generate([ChatMessage("user", "q")], 2, "legacy")
                ).text == "legacy"
            finally:
                await client.close()

        asyncio.run(run())


def test_invalid_context_health_capability_is_not_coerced(monkeypatch):
    client = RemoteWorkerBackend("http://127.0.0.1:1", TOKEN)
    monkeypatch.setattr(
        client,
        "_http",
        lambda *args: {
            "schema": "instinct_worker_v1",
            "ready": True,
            "supports_context_limits": "true",
        },
    )
    with pytest.raises(BackendError, match="context-limit capability"):
        asyncio.run(client.health())


@pytest.mark.parametrize("fail", [False, True])
def test_worker_settled_future_and_exception_payload_are_not_retained(fail):
    from instinct.agents.worker import WorkerService

    entered, release = threading.Event(), threading.Event()

    class Payload:
        pass

    class Model:
        async def generate(self, messages, maximum, request_id):
            payload = Payload()
            self.payload_ref = weakref.ref(payload)
            entered.set()
            assert await asyncio.to_thread(release.wait, 2.0)
            if fail:
                raise RuntimeError("model failed after allocating temporary arrays")
            return Generation("completed", 2, 1)

    model = Model()
    service = WorkerService(model)
    service.start()
    try:
        submission = {
            "request_id": "retention",
            "messages": [{"role": "user", "content": "q"}],
            "max_tokens": 2,
        }
        service.submit(submission)
        assert entered.wait(2.0)
        with service._lock:
            future_ref = weakref.ref(service._jobs["retention"].future)
        release.set()
        deadline = time.monotonic() + 2.0
        while (
            service.get("retention")["status"] == "pending"
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        result = service.get("retention")
        assert result["status"] == ("failed" if fail else "completed")
        with service._lock:
            assert service._jobs["retention"].future is None
        # The callback may still be returning on its loop thread. Wait for that
        # bounded handoff, collecting the otherwise unreachable traceback cycle.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            gc.collect()
            if future_ref() is None and model.payload_ref() is None:
                break
            time.sleep(0.002)
        assert future_ref() is None and model.payload_ref() is None
        assert service.submit(submission) == result
    finally:
        release.set()
        service.close()
