"""CPU lifecycle tests, including actual authenticated loopback HTTP transport."""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from itertools import pairwise
from types import SimpleNamespace

import pytest
from forge_llm.agents.backends import (
    AuthenticationError,
    BackendError,
    CapacityError,
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    LocalForgeBackend,
    RemoteWorkerBackend,
    WorkerConfig,
    WorkerPool,
    WorkerUnavailableError,
)
from forge_llm.agents.protocol import ChatMessage, Generation
from forge_llm.agents.worker import WorkerHTTPServer, WorkerService

CHAT = [ChatMessage("user", "hello")]
TOKEN = "a-test-token-that-is-not-a-real-credential"


def native_specs():
    return [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Return the verified result.",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        }
    ]


class FakeTokenizer:
    chat_template = "test"
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True and add_generation_prompt is True
        return [1] * (sum(len(message["content"]) for message in messages) + 1)

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return ",".join(str(t) for t in tokens if t)


class FakeEngine:
    backend = "fake"
    max_model_length = 128

    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.thread_ids = set()
        self.active = None
        self.submitted = 0
        self.cancelled = []
        self.forgotten = []
        self.closed = False
        self.steps = 0

    def _own_thread(self):
        self.thread_ids.add(threading.get_ident())
        assert len(self.thread_ids) == 1, "engine touched by multiple threads"

    def submit(self, prompt, max_tokens, eos):
        self._own_thread()
        assert self.active is None, "non-thread-safe engine used concurrently"
        self.submitted += 1
        self.active = [self.submitted, max_tokens, 0]
        return self.submitted

    def step(self):
        self._own_thread()
        time.sleep(self.delay)
        self.steps += 1
        if self.fail:
            self.fail = False
            raise RuntimeError("injected kernel failure")
        request_id, maximum, count = self.active
        count += 1
        self.active[2] = count
        finished = count == maximum
        if finished:
            self.active = None
        return [SimpleNamespace(request_id=request_id, token=17, finished=finished)]

    def cancel(self, request_id):
        self._own_thread()
        self.cancelled.append(request_id)
        self.active = None

    def forget(self, request_id):
        self._own_thread()
        self.forgotten.append(request_id)

    def close(self):
        self._own_thread()
        assert self.active is None
        self.closed = True


def local(engine=None, **config_options):
    engine = engine or FakeEngine()
    return LocalForgeBackend(
        WorkerConfig("fake.engine", "fake-tokenizer", **config_options),
        engine_factory=lambda: engine,
        tokenizer_factory=FakeTokenizer,
    ), engine


async def eventually(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not complete before timeout")
        await asyncio.sleep(0.002)


def test_local_actor_serializes_engine_and_releases_history():
    async def run():
        backend, engine = local(FakeEngine(delay=0.002), max_pending=4)
        try:
            await backend.start()
            outputs = await asyncio.gather(
                *(backend.generate(CHAT, 3, f"req-{i}") for i in range(4))
            )
            assert [output.text for output in outputs] == ["17,17,17"] * 4
            assert all(
                output.input_tokens == 6 and output.output_tokens == 3
                for output in outputs
            )
            assert all(output.finish_reason == "length" for output in outputs)
            assert len(engine.thread_ids) == 1
            assert engine.submitted == 4
            assert engine.forgotten == [1, 2, 3, 4]
            assert backend.health()["active"] == 0
        finally:
            await backend.close()
        assert engine.closed
        with pytest.raises(WorkerUnavailableError):
            await backend.generate(CHAT, 1, "closed")

    asyncio.run(run())


def test_local_context_bound_rejects_before_engine_submission():
    async def run():
        backend, engine = local()
        try:
            with pytest.raises(ContextLengthError):
                await backend.generate([ChatMessage("user", "x" * 127)], 1, "too-long")
            assert engine.submitted == 0
            output = await backend.generate(CHAT, 1, "fits")
            assert output.output_tokens == 1
        finally:
            await backend.close()

    asyncio.run(run())


def test_local_accepts_chat_template_batch_encoding():
    class DictTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **options):
            return {
                "input_ids": super().apply_chat_template(messages, **options),
                "attention_mask": [1] * 6,
            }

    async def run():
        backend = LocalForgeBackend(
            WorkerConfig("fake.engine", "fake"),
            engine_factory=FakeEngine,
            tokenizer_factory=DictTokenizer,
        )
        try:
            assert (await backend.generate(CHAT, 1, "batch-encoding")).input_tokens == 6
        finally:
            await backend.close()

    asyncio.run(run())


def test_gemma_chat_template_preserves_system_and_merges_adjacent_evidence():
    observed = []

    class GemmaTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **options):
            observed.extend(messages)
            assert all(message["role"] in {"user", "assistant"} for message in messages)
            assert all(a["role"] != b["role"] for a, b in pairwise(messages))
            return [1, 2, 3]

    async def run():
        engine = FakeEngine()
        engine.model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="gemma3_text", vocab_size=100, eos_token_id=0
            )
        )
        backend = LocalForgeBackend(
            WorkerConfig("fake.engine", "fake"),
            engine_factory=lambda: engine,
            tokenizer_factory=GemmaTokenizer,
        )
        try:
            await backend.generate(
                [
                    ChatMessage("system", "instructions"),
                    ChatMessage("user", "task"),
                    ChatMessage("user", "evidence"),
                    ChatMessage("assistant", "action"),
                    ChatMessage("user", "result"),
                ],
                1,
                "gemma",
            )
            assert observed == [
                {"role": "user", "content": "instructions\n\ntask\n\nevidence"},
                {"role": "assistant", "content": "action"},
                {"role": "user", "content": "result"},
            ]
        finally:
            await backend.close()

    asyncio.run(run())


def test_local_bounded_admission_and_duplicate_id():
    async def run():
        backend, engine = local(FakeEngine(delay=0.02), max_pending=1)
        try:
            task = asyncio.create_task(backend.generate(CHAT, 4, "first"))
            await eventually(lambda: engine.submitted == 1)
            with pytest.raises(ValueError, match="already active"):
                await backend.generate(CHAT, 1, "first")
            with pytest.raises(CapacityError):
                await backend.generate(CHAT, 1, "overflow")
            await task
            await backend.generate(CHAT, 1, "next")
        finally:
            await backend.close()

    asyncio.run(run())


def test_local_cancel_reclaims_engine_and_allows_next_request():
    async def run():
        backend, engine = local(FakeEngine(delay=0.01))
        try:
            task = asyncio.create_task(backend.generate(CHAT, 20, "cancel-me"))
            await eventually(lambda: engine.steps >= 1)
            await backend.cancel("cancel-me")
            with pytest.raises(GenerationCancelledError):
                await task
            assert engine.cancelled == [1] and engine.forgotten == [1]
            await backend.generate(CHAT, 1, "after-cancel")
        finally:
            await backend.close()

    asyncio.run(run())


def test_local_async_task_cancel_keeps_capacity_until_actor_stops():
    async def run():
        backend, engine = local(FakeEngine(delay=0.05), max_pending=1)
        try:
            task = asyncio.create_task(backend.generate(CHAT, 20, "cancel-task"))
            await eventually(lambda: engine.submitted == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(CapacityError):
                await backend.generate(CHAT, 1, "too-soon")
            await eventually(lambda: backend.health()["active"] == 0)
            assert engine.cancelled == [1]
            await backend.generate(CHAT, 1, "later")
        finally:
            await backend.close()

    asyncio.run(run())


def test_local_deadline_includes_queue_and_failure_reclaims_request():
    async def run():
        entered, release = threading.Event(), threading.Event()

        class BlockedEngine(FakeEngine):
            def step(self):
                entered.set()
                assert release.wait(timeout=2.0)
                return super().step()

        backend, engine = local(BlockedEngine(), max_pending=2, request_timeout=0.025)
        try:
            first = asyncio.create_task(backend.generate(CHAT, 5, "one"))
            await eventually(entered.is_set)
            queued = asyncio.create_task(backend.generate(CHAT, 1, "queued"))
            await eventually(lambda: backend.health()["active"] == 2)
            # Both deadlines pass while the first engine step is blocked, so
            # the second request provably expires in the queue before submit.
            await asyncio.sleep(0.03)
            release.set()
            results = await asyncio.gather(first, queued, return_exceptions=True)
            assert all(isinstance(result, GenerationTimeoutError) for result in results)
            assert engine.submitted == 1
            assert engine.cancelled == [1] and engine.forgotten == [1]
        finally:
            release.set()
            await backend.close()
        backend, engine = local(FakeEngine(fail=True))
        try:
            with pytest.raises(RuntimeError, match="injected"):
                await backend.generate(CHAT, 1, "fails")
            assert engine.cancelled == [1] and engine.forgotten == [1]
            await backend.generate(CHAT, 1, "recovers")
        finally:
            await backend.close()

    asyncio.run(run())


class AsyncModel:
    def __init__(
        self, *, delay=0.01, fail=False, text="answer", worker_id="fake-worker"
    ):
        self.delay, self.fail, self.text, self.worker_id = delay, fail, text, worker_id
        self.calls = []
        self.cancelled = []
        self.active = 0
        self.peak = 0

    async def generate(self, messages, max_tokens, request_id):
        self.calls.append(request_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("private failure detail")
            return Generation(self.text, 6, min(2, max_tokens), "length", "fake-model")
        except asyncio.CancelledError:
            self.cancelled.append(request_id)
            raise
        finally:
            self.active -= 1

    def health(self):
        return {
            "worker_id": self.worker_id,
            "model": "fake-model",
            "backend": "fake",
            "ready": True,
        }


class NativeAsyncModel(AsyncModel):
    supports_native_tools = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.catalogs = []
        self.message_history = []

    async def generate_action(self, messages, max_tokens, request_id, tool_specs):
        self.catalogs.append(tool_specs)
        self.message_history.append(tuple(messages))
        return await self.generate(messages, max_tokens, request_id)


def test_local_native_catalog_reaches_template_and_is_copied_before_queue():
    entered, release = threading.Event(), threading.Event()
    observed = []

    class NativeTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, tools=None
        ):
            observed.append(tools)
            return [1] * (12 if tools is not None else 6)

    class BlockedEngine(FakeEngine):
        def step(self):
            if self.submitted == 1:
                entered.set()
                assert release.wait(timeout=2)
            return super().step()

    async def run():
        backend = LocalForgeBackend(
            WorkerConfig("fake.engine", "fake", native_tool_calls=True, max_pending=2),
            engine_factory=BlockedEngine,
            tokenizer_factory=NativeTokenizer,
        )
        catalog = native_specs()
        expected = json.loads(json.dumps(catalog))
        try:
            plain = asyncio.create_task(
                backend.generate(CHAT, 1, "plain-native-worker")
            )
            await eventually(entered.is_set)
            queued = asyncio.create_task(
                backend.generate_action(CHAT, 1, "native-queued", catalog)
            )
            await eventually(lambda: backend.health()["active"] == 2)
            catalog[0]["function"]["parameters"]["properties"]["result"]["type"] = (
                "integer"
            )
            release.set()
            outputs = await asyncio.gather(plain, queued)
            assert outputs[0].input_tokens == 6 and outputs[1].input_tokens == 12
            assert observed == [None, expected]
            assert (
                backend.supports_native_tools and backend.health()["native_tool_calls"]
            )
        finally:
            release.set()
            await backend.close()

    asyncio.run(run())


def test_native_opt_in_and_mixed_pool_capability_are_explicit():
    async def run():
        disabled, _ = local()
        native = NativeAsyncModel()
        try:
            assert not disabled.supports_native_tools
            with pytest.raises(ValueError, match="disabled"):
                await disabled.generate_action(CHAT, 1, "disabled", native_specs())
            mixed = WorkerPool([disabled, native])
            assert not mixed.supports_native_tools
            with pytest.raises(ValueError, match="every replica"):
                await mixed.generate_action(CHAT, 1, "mixed", native_specs())
            enabled = WorkerPool([native])
            assert enabled.supports_native_tools
        finally:
            await disabled.close()

    asyncio.run(run())


def test_live_http_native_catalog_capability_and_idempotency():
    with serving(NativeAsyncModel(delay=0.025)) as (url, model, _):

        async def run():
            backend = client(url)
            assert not backend.supports_native_tools
            assert (await backend.health())["native_tool_calls"]
            assert backend.supports_native_tools
            body = payload("native-id")
            body["timeout"] = backend.request_timeout
            body["tools"] = native_specs()
            assert (await asyncio.to_thread(backend._http, "POST", "/v1/jobs", body))[
                "status"
            ] == "pending"
            changed = json.loads(json.dumps(body))
            changed["tools"][0]["function"]["description"] = "Different exact catalog."
            with pytest.raises(BackendError, match="different job"):
                await asyncio.to_thread(backend._http, "POST", "/v1/jobs", changed)
            result = await backend.generate_action(CHAT, 2, "native-id", native_specs())
            assert result.text == "answer"
            assert model.calls == ["native-id"] and model.catalogs == [native_specs()]
            native_history = CHAT + [
                ChatMessage(
                    "assistant",
                    '<tool_call>{"name":"finish","arguments":{"result":"fixture"}}</tool_call>',
                ),
                ChatMessage("tool", "A successful native function result."),
            ]
            await backend.generate_action(
                native_history, 2, "native-history", native_specs()
            )
            assert model.message_history[-1] == tuple(native_history)
            await backend.close()

        asyncio.run(run())


def test_live_http_rejects_native_catalog_on_disabled_worker_without_dispatch():
    with serving() as (url, model, _):

        async def run():
            backend = client(url)
            body = payload()
            body["tools"] = native_specs()
            with pytest.raises(ValueError, match="native tool"):
                await asyncio.to_thread(backend._http, "POST", "/v1/jobs", body)
            assert model.calls == []

        asyncio.run(run())


def test_native_id_conflict_cannot_cancel_an_existing_accepted_job():
    entered, release = threading.Event(), threading.Event()

    class GatedNative(NativeAsyncModel):
        async def generate(self, messages, maximum, request_id):
            self.calls.append(request_id)
            entered.set()
            try:
                assert await asyncio.to_thread(release.wait, 2.0)
                return Generation("original accepted job", 6, 2, "length", "fixture")
            except asyncio.CancelledError:
                self.cancelled.append(request_id)
                raise

    with serving(GatedNative()) as (url, model, _):

        async def run():
            original, conflicting = client(url), client(url)
            await asyncio.gather(original.health(), conflicting.health())
            task = asyncio.create_task(
                original.generate_action(CHAT, 2, "shared-native-id", native_specs())
            )
            try:
                await eventually(entered.is_set)
                changed = native_specs()
                changed[0]["function"]["description"] = "A conflicting catalog."
                with pytest.raises(ValueError, match="different job"):
                    await conflicting.generate_action(
                        CHAT, 2, "shared-native-id", changed
                    )
                release.set()
                assert (await task).text == "original accepted job"
                assert model.calls == ["shared-native-id"] and model.cancelled == []
            finally:
                release.set()
                await asyncio.gather(original.close(), conflicting.close())
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        [{"type": "function", "function": {"name": "x"}}],
        [native_specs()[0]] * 33,
        [{**native_specs()[0], "execute": "unsafe"}],
    ],
)
def test_native_catalog_validation_rejects_before_model_dispatch(bad):
    with serving(NativeAsyncModel()) as (url, model, _):

        async def run():
            backend = client(url)
            await backend.health()
            with pytest.raises(ValueError):
                await backend.generate_action(CHAT, 2, "invalid-tools", bad)
            assert model.calls == []

        asyncio.run(run())


def test_cached_qwen_native_template_and_tags_with_real_tokenizer_cpu_only():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        pytest.skip("local Qwen tokenizer fixture is not cached; no download permitted")
    from forge_llm.agents.native_tools import parse_native_action

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
        tools=native_specs(),
    )
    assert "tool_call" in rendered and "Return the verified result." in rendered
    call = '<tool_call>{"name":"finish","arguments":{"result":"verified"}}</tool_call>'
    encoded = tokenizer.encode(call + tokenizer.eos_token, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == call
    assert parse_native_action(decoded, native_specs()) == {
        "action": "finish",
        "result": "verified",
    }
    captured = []

    class RecordingEngine(FakeEngine):
        max_model_length = 2048

        def submit(self, prompt, maximum, eos):
            captured.append(list(prompt))
            return super().submit(prompt, maximum, eos)

    async def run():
        backend = LocalForgeBackend(
            WorkerConfig("fake.engine", "cached-qwen", native_tool_calls=True),
            engine_factory=RecordingEngine,
            tokenizer_factory=lambda: tokenizer,
        )
        try:
            history = CHAT + [
                ChatMessage("assistant", call),
                ChatMessage("tool", "Function result evidence."),
            ]
            native_rendered = tokenizer.apply_chat_template(
                [
                    {"role": message.role, "content": message.content}
                    for message in history
                ],
                tokenize=False,
                add_generation_prompt=True,
                tools=native_specs(),
            )
            assert "<tool_response>" in native_rendered
            result = await backend.generate_action(
                history, 1, "actual-tokenizer", native_specs()
            )
            expected = tokenizer.apply_chat_template(
                [
                    {"role": message.role, "content": message.content}
                    for message in history
                ],
                tokenize=True,
                add_generation_prompt=True,
                tools=native_specs(),
            )
            ids = (
                expected["input_ids"]
                if isinstance(expected, dict) or hasattr(expected, "keys")
                else expected
            )
            assert captured == [ids] and result.input_tokens == len(ids)
        finally:
            await backend.close()

    asyncio.run(run())


@contextmanager
def serving(backend=None, **service_options):
    backend = backend or AsyncModel()
    service = WorkerService(backend, **service_options)
    service.start()
    server = WorkerHTTPServer(("127.0.0.1", 0), service, TOKEN)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", backend, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
        service.close()


def client(url, **options):
    return RemoteWorkerBackend(url, TOKEN, poll_interval=0.002, **options)


def payload(request_id="job", *, max_tokens=2, timeout=1.0):
    return {
        "request_id": request_id,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "timeout": timeout,
    }


def test_live_http_health_identity_and_concurrent_generation():
    with serving(AsyncModel(delay=0.03)) as (url, model, service):

        async def run():
            backend = client(url, max_concurrency=3)
            identity = await backend.health()
            assert identity["worker_id"] == "fake-worker" and identity["pid"] > 0
            assert identity["schema"] == "forge_worker_v1"
            outputs = await asyncio.gather(
                *(backend.generate(CHAT, 2, f"http-{i}") for i in range(3))
            )
            assert all(output.text == "answer" for output in outputs)
            assert model.peak == 3
            assert service.health()["active"] == 0
            await backend.close()

        asyncio.run(run())


def test_live_http_authentication_and_remote_plaintext_opt_in():
    with serving() as (url, _, _):

        async def run():
            with pytest.raises(AuthenticationError):
                await RemoteWorkerBackend(url, "wrong").health()

        asyncio.run(run())
    with pytest.raises(ValueError, match="allow_insecure"):
        RemoteWorkerBackend("http://192.168.1.100:8090", TOKEN)
    assert RemoteWorkerBackend("https://model.example:443", TOKEN).url.startswith(
        "https"
    )
    assert RemoteWorkerBackend("http://192.168.1.100:8090", TOKEN, allow_insecure=True)
    for url in (
        "http://user:pass@localhost:8090",
        "http://localhost:8090/path",
        "http://localhost:8090?secret=yes",
    ):
        with pytest.raises(ValueError):
            RemoteWorkerBackend(url, TOKEN)


def test_live_http_idempotency_pending_complete_and_conflict():
    with serving(AsyncModel(delay=0.03)) as (url, model, _):
        backend = client(url)
        first = backend._http("POST", "/v1/jobs", payload())
        repeated = backend._http("POST", "/v1/jobs", payload())
        assert first["status"] == repeated["status"] == "pending"
        with pytest.raises(BackendError, match="different job"):
            backend._http("POST", "/v1/jobs", payload(max_tokens=3))
        deadline = time.monotonic() + 1
        while True:
            result = backend._http("GET", "/v1/jobs/job")
            if result["status"] == "completed":
                break
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert backend._http("POST", "/v1/jobs", payload()) == result
        assert model.calls == ["job"]


def test_live_http_ambiguous_submission_retries_same_id_once():
    with serving() as (url, model, _):

        async def run():
            backend = client(url)
            original = backend._http
            submit_count = 0

            def lose_ack(method, path, body=None, timeout=None):
                nonlocal submit_count
                result = original(method, path, body, timeout)
                if method == "POST":
                    submit_count += 1
                    if submit_count == 1:
                        raise WorkerUnavailableError("simulated lost acknowledgement")
                return result

            backend._http = lose_ack
            result = await backend.generate(CHAT, 2, "ambiguous")
            assert result.text == "answer"
            assert submit_count == 2 and model.calls == ["ambiguous"]

        asyncio.run(run())


def test_live_http_saturation_then_recovery():
    with serving(AsyncModel(delay=0.04), max_pending=1) as (url, model, _):

        async def run():
            backend = client(url, max_concurrency=2)
            task = asyncio.create_task(backend.generate(CHAT, 2, "accepted"))
            await eventually(lambda: bool(model.calls))
            with pytest.raises(CapacityError):
                await backend.generate(CHAT, 2, "rejected")
            await task
            assert (await backend.generate(CHAT, 2, "next")).text == "answer"

        asyncio.run(run())


def test_live_http_client_limit_and_task_cancellation():
    with serving(AsyncModel(delay=1)) as (url, model, service):

        async def run():
            backend = client(url)
            task = asyncio.create_task(backend.generate(CHAT, 2, "aborted"))
            await eventually(lambda: bool(model.calls))
            with pytest.raises(CapacityError):
                await backend.generate(CHAT, 2, "extra")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await eventually(lambda: model.cancelled == ["aborted"])
            assert service.health()["active"] == 0
            await backend.close()

        asyncio.run(run())


def test_live_http_server_deadline_cancels_backend():
    with serving(AsyncModel(delay=1), max_timeout=0.025) as (url, model, _):

        async def run():
            backend = client(url, request_timeout=1.0)
            with pytest.raises(GenerationTimeoutError):
                await backend.generate(CHAT, 2, "server-timeout")
            await eventually(lambda: model.cancelled == ["server-timeout"])

        asyncio.run(run())


def test_live_http_client_deadline_sends_cancellation():
    with serving(AsyncModel(delay=1)) as (url, model, _):

        async def run():
            backend = client(url, request_timeout=0.025)
            with pytest.raises(GenerationTimeoutError):
                await backend.generate(CHAT, 2, "client-timeout")
            await eventually(lambda: model.cancelled == ["client-timeout"])

        asyncio.run(run())


def test_live_http_failure_is_not_retried_or_exposed_as_traceback():
    with serving(AsyncModel(fail=True)) as (url, model, _):

        async def run():
            with pytest.raises(BackendError, match="RuntimeError") as error:
                await client(url).generate(CHAT, 2, "failed")
            assert "private failure detail" not in str(error.value)
            assert model.calls == ["failed"]

        asyncio.run(run())


def test_live_http_rejects_oversized_and_duplicate_json_and_nan():
    with serving() as (url, model, _):
        # Oversized declared length is rejected before reading any body. Do not
        # race a megabyte send against the server deliberately closing its socket.
        connection = http.client.HTTPConnection(
            url.removeprefix("http://"), timeout=1.0
        )
        connection.putrequest("POST", "/v1/jobs")
        connection.putheader("Authorization", "Bearer " + TOKEN)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str((1 << 20) + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
        bodies = [b'{"request_id":"a","request_id":"b"}', b'{"timeout":NaN}']
        for body in bodies:
            request = urllib.request.Request(
                url + "/v1/jobs",
                data=body,
                headers={
                    "Authorization": "Bearer " + TOKEN,
                    "Content-Type": "application/json",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=1.0)
            assert error.value.code == 400
            error.value.close()
        assert model.calls == []


def test_live_http_response_size_bound():
    with serving(AsyncModel(text="x" * 4096), max_result_bytes=512) as (url, _, _):

        async def run():
            with pytest.raises(BackendError, match="BackendError"):
                await client(url).generate(CHAT, 2, "too-large")
            tiny = client(url, max_response_bytes=16)
            with pytest.raises(BackendError, match="size limit"):
                await tiny.health()

        asyncio.run(run())


def test_history_retention_is_bounded_and_expiry_is_advertised():
    model = AsyncModel(delay=0)
    service = WorkerService(model, max_history=2, history_ttl=0.02)
    service.start()
    try:
        for number in range(3):
            service.submit(payload(f"h-{number}"))
            deadline = time.monotonic() + 1
            while service.get(f"h-{number}")["status"] == "pending":
                assert time.monotonic() < deadline
                time.sleep(0.001)
        with pytest.raises(ValueError, match="unknown"):
            service.get("h-0")
        assert len(service._jobs) == 2
        assert service.health()["history_ttl"] == 0.02
        time.sleep(0.025)
        with pytest.raises(ValueError, match="unknown"):
            service.get("h-2")
    finally:
        service.close()


def test_history_retention_has_a_total_byte_budget():
    service = WorkerService(
        AsyncModel(delay=0),
        max_history=100,
        max_result_bytes=256,
        max_history_bytes=300,
    )
    service.start()
    try:
        for number in range(3):
            service.submit(payload(f"bytes-{number}"))
            deadline = time.monotonic() + 1
            while service.get(f"bytes-{number}")["status"] == "pending":
                assert time.monotonic() < deadline
                time.sleep(0.001)
        assert len(service._jobs) == 1
        assert sum(job.result_bytes for job in service._jobs.values()) <= 300
        assert service.health()["max_history_bytes"] == 300
    finally:
        service.close()


def test_pool_parallel_replicas_serializes_each_and_routes_explicitly():
    async def run():
        one, two = (
            AsyncModel(delay=0.015, text="one"),
            AsyncModel(delay=0.015, text="two"),
        )
        pool = WorkerPool({"a": one, "b": two}, max_pending=8)
        results = await asyncio.gather(
            *(pool.generate(CHAT, 2, f"p-{i}") for i in range(6))
        )
        assert {result.text for result in results} == {"one", "two"}
        assert one.peak == two.peak == 1
        assert len(one.calls) == len(two.calls) == 3
        assert pool.loads == {"a": 0, "b": 0}
        assert (await pool.generate(CHAT, 2, "pinned", worker="b")).text == "two"
        with pytest.raises(ValueError, match="unknown"):
            await pool.generate(CHAT, 2, "unknown", worker="c")
        await pool.close()

    asyncio.run(run())


def test_pool_queue_bound_and_queued_cancel_never_dispatches():
    async def run():
        model = AsyncModel(delay=0.04)
        pool = WorkerPool([model], max_pending=2)
        first = asyncio.create_task(pool.generate(CHAT, 2, "running"))
        await eventually(lambda: model.calls == ["running"])
        queued = asyncio.create_task(pool.generate(CHAT, 2, "queued"))
        await asyncio.sleep(0)
        with pytest.raises(CapacityError):
            await pool.generate(CHAT, 2, "overflow")
        await pool.cancel("queued")
        with pytest.raises(asyncio.CancelledError):
            await queued
        await first
        assert model.calls == ["running"]
        assert pool.loads == {"worker-0": 0}
        await pool.generate(CHAT, 2, "recovered")

    asyncio.run(run())


def test_pool_failure_releases_slot_and_never_reroutes_ambiguous_request():
    async def run():
        failed, okay = AsyncModel(fail=True), AsyncModel(text="okay")
        pool = WorkerPool({"failed": failed, "okay": okay})
        with pytest.raises(RuntimeError):
            await pool.generate(CHAT, 2, "failure", worker="failed")
        assert failed.calls == ["failure"] and okay.calls == []
        assert pool.loads == {"failed": 0, "okay": 0}
        assert (await pool.generate(CHAT, 2, "success", worker="okay")).text == "okay"

    asyncio.run(run())


def test_pool_retry_request_id_stays_on_same_worker_after_failure():
    class AmbiguousOnce(AsyncModel):
        async def generate(self, messages, maximum, request_id):
            if not self.calls:
                self.calls.append(request_id)
                raise WorkerUnavailableError("acknowledgement lost after execution")
            return await super().generate(messages, maximum, request_id)

    async def run():
        first, second = AmbiguousOnce(text="first"), AsyncModel(text="second")
        pool = WorkerPool({"first": first, "second": second})
        with pytest.raises(WorkerUnavailableError):
            await pool.generate(CHAT, 2, "sticky-retry")
        assert (await pool.generate(CHAT, 2, "sticky-retry")).text == "first"
        assert first.calls == ["sticky-retry", "sticky-retry"] and not second.calls
        with pytest.raises(ValueError, match="another worker"):
            await pool.generate(CHAT, 2, "sticky-retry", worker="second")
        assert (
            await pool.generate(CHAT, 2, "new-job", worker="second")
        ).text == "second"

    asyncio.run(run())


def test_pool_cancel_stops_awaiter_even_if_worker_cancel_transport_fails():
    class BrokenCancel(AsyncModel):
        async def cancel(self, request_id):
            raise WorkerUnavailableError("worker disconnected before cancellation")

    async def run():
        model = BrokenCancel(delay=1)
        pool = WorkerPool([model])
        task = asyncio.create_task(pool.generate(CHAT, 2, "cancel-offline"))
        await eventually(lambda: bool(model.calls))
        with pytest.raises(WorkerUnavailableError):
            await pool.cancel("cancel-offline")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.loads == {"worker-0": 0}

    asyncio.run(run())


def test_pool_close_cancels_active_and_queued_generations():
    async def run():
        model = AsyncModel(delay=1)
        pool = WorkerPool([model])
        active = asyncio.create_task(pool.generate(CHAT, 2, "close-active"))
        await eventually(lambda: bool(model.calls))
        queued = asyncio.create_task(pool.generate(CHAT, 2, "close-queued"))
        await asyncio.sleep(0)
        await pool.close()
        results = await asyncio.gather(active, queued, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert model.calls == ["close-active"] and model.cancelled == ["close-active"]
        assert pool.loads == {"worker-0": 0}

    asyncio.run(run())


def test_pool_live_http_two_distinct_worker_identities():
    with (
        serving(AsyncModel(worker_id="replica-a", text="a")) as (url_a, a, _),
        serving(AsyncModel(worker_id="replica-b", text="b")) as (url_b, b, _),
    ):

        async def run():
            remote_a, remote_b = client(url_a), client(url_b)
            identities = await asyncio.gather(remote_a.health(), remote_b.health())
            assert {identity["worker_id"] for identity in identities} == {
                "replica-a",
                "replica-b",
            }
            pool = WorkerPool({"a": remote_a, "b": remote_b})
            outputs = await asyncio.gather(
                pool.generate(CHAT, 2, "distributed-a"),
                pool.generate(CHAT, 2, "distributed-b"),
            )
            assert {output.text for output in outputs} == {"a", "b"}
            assert len(a.calls) == len(b.calls) == 1
            await pool.close()

        asyncio.run(run())


def test_local_speculative_worker_uses_cancellation_and_measured_usage():
    class SpecEngine:
        max_model_length = 128

        def generate_result(self, tokens, maximum, eos, *, cancelled):
            assert not cancelled()
            return SimpleNamespace(
                tokens=[17, 18],
                finish_reason="length",
                stats=SimpleNamespace(to_dict=lambda: {"accepted_draft_tokens": 1}),
            )

        def close(self):
            pass

    async def run():
        backend = LocalForgeBackend(
            WorkerConfig(
                "fake.engine", "fake", backend="mlx", speculative={"draft_tokens": 2}
            ),
            engine_factory=SpecEngine,
            tokenizer_factory=FakeTokenizer,
        )
        try:
            output = await backend.generate(CHAT, 2, "spec")
            assert output.text == "17,18" and output.output_tokens == 2
            assert backend.health()["last_speculative_stats"] == {
                "accepted_draft_tokens": 1
            }
        finally:
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"max_pending": 0},
        {"request_timeout": float("nan")},
        {"max_model_length": True},
        {"backend": "cuda", "speculative": {}},
        {"engine_options": {"max_num_sequences": 99}},
    ],
)
def test_worker_config_rejects_invalid_resource_overrides(options):
    with pytest.raises(ValueError):
        WorkerConfig("fake.engine", "fake", **options)


@pytest.mark.parametrize(
    "messages,maximum,request_id",
    [
        ([], 2, "x"),
        (CHAT, True, "x"),
        (CHAT, 2, "x/y"),
        ([ChatMessage("tool", "x")], 2, "x"),
        ([ChatMessage("user", 3)], 2, "x"),
    ],
)
def test_requests_reject_invalid_types_roles_and_ids(messages, maximum, request_id):
    async def run():
        backend, _ = local()
        try:
            with pytest.raises((ValueError, TypeError)):
                await backend.generate(messages, maximum, request_id)
        finally:
            await backend.close()

    asyncio.run(run())
