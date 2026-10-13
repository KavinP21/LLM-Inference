"""Conversational sends exercise the actual journal and scheduler without a GPU."""

import asyncio
import json

import pytest
from forge_llm.agents.protocol import Generation
from forge_llm.harness import service as service_module
from forge_llm.harness.conversation import ConversationRuntime
from forge_llm.harness.service import StudioService


class ChatBackend:
    def __init__(self, *, pause_first=False):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.pause_first = pause_first
        self.closed = False

    async def generate(self, messages, max_tokens, request_id):
        self.calls.append(list(messages))
        self.started.set()
        if self.pause_first and len(self.calls) == 1:
            await self.release.wait()
        return Generation(
            json.dumps({"action": "finish", "result": f"Answer {len(self.calls)}"}),
            100,
            20,
            model="fixture",
        )

    async def cancel(self, request_id):
        return None

    async def close(self):
        self.closed = True


def make_service(tmp_path, **kwargs):
    return StudioService(tmp_path, tmp_path / ".forge", demo=True, **kwargs)


async def settle(service, chat_id):
    for _ in range(500):
        detail = await service.dispatch("GET", f"/api/chats/{chat_id}", {})
        if all(
            t["status"] not in {"loading", "running", "queued"} for t in detail["turns"]
        ):
            return detail
        await asyncio.sleep(0.01)
    pytest.fail("conversation did not settle")


def startup_fixture(tmp_path, monkeypatch, *, failure=False, gate=None):
    backend = ChatBackend()
    attempts = []

    async def create_pool(config):
        attempts.append(config)
        if gate:
            await gate.wait()
        if failure and len(attempts) == 1:
            raise RuntimeError("fixture model unavailable")
        return backend

    async def identities(pool):
        return {"fixture": {"model": "test-model"}}

    monkeypatch.setattr(service_module, "create_pool", create_pool)
    monkeypatch.setattr(service_module, "worker_identities", identities)
    service = StudioService(
        tmp_path,
        tmp_path / ".forge",
        config={"workers": [{"model": "test-model"}]},
    )
    return service, backend, attempts


def test_first_send_is_durable_before_automatic_startup(tmp_path, monkeypatch):
    async def check():
        gate = asyncio.Event()
        service, backend, attempts = startup_fixture(tmp_path, monkeypatch, gate=gate)
        try:
            sent = await service.dispatch("POST", "/api/chats", {"message": "Hello"})
            assert sent["status"] == "loading" and sent["run_id"] is None
            assert (
                service.store.connection.execute(
                    "SELECT COUNT(*) FROM runs"
                ).fetchone()[0]
                == 0
            )
            detail = await service.conversations.detail(sent["chat_id"])
            assert detail["turns"][0]["user"] == "Hello"
            assert detail["turns"][0]["active"] is True
            await asyncio.sleep(0.01)
            assert len(attempts) == 1 and service.backend_status == "loading"
            gate.set()
            detail = await settle(service, sent["chat_id"])
            assert detail["turns"][0]["output"] == "Answer 1"
            assert detail["chat"]["status"] == "completed"
            assert isinstance(
                service.runtimes[detail["turns"][0]["run_id"]], ConversationRuntime
            )
            assert len(backend.calls) == 1  # A greeting can finish without tools.
        finally:
            await service.close()

    asyncio.run(check())


def test_followups_use_real_multi_turn_history_and_original_budgets(tmp_path):
    async def check():
        backend = ChatBackend()
        service = make_service(
            tmp_path, backend=backend, config={"runtime": {"max_agents": 2}}
        )
        try:
            chat = (
                await service.conversations.create(
                    {"message": "My color is blue", "policy": "read_only"}
                )
            )["chat_id"]
            await settle(service, chat)
            service.config = {"runtime": {"max_agents": 7}}
            await service.conversations.message(
                chat, {"message": "Remember that color"}
            )
            await settle(service, chat)
            await service.conversations.message(
                chat, {"message": "What did I say first?"}
            )
            detail = await settle(service, chat)
            run_id = detail["turns"][-1]["run_id"]
            actual = service.store.messages(run_id, "root")
            assert [(m.role, m.content) for m in actual[1:6]] == [
                ("user", "My color is blue"),
                ("assistant", "Answer 1"),
                ("user", "Remember that color"),
                ("assistant", "Answer 2"),
                ("user", "What did I say first?"),
            ]
            assert service.store.run(run_id)["config"]["max_agents"] == 2
            assert detail["chat"]["policy"] == "read_only"
            assert (
                not service.runtimes[run_id].mutating_tools
                & service.runtimes[run_id].tools.keys()
            )
            assert len(detail["turns"]) == 3
            assert "plain text or Markdown" in actual[0].content
        finally:
            await service.close()

    asyncio.run(check())


def test_busy_send_queues_and_cannot_disappear_during_final_generation(tmp_path):
    async def check():
        backend = ChatBackend(pause_first=True)
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (await service.conversations.create({"message": "First request"}))[
                "chat_id"
            ]
            await backend.started.wait()
            queued = await service.conversations.message(
                chat, {"message": "Focus on the important detail"}
            )
            assert queued["status"] == "queued" and queued["queued"] is True
            detail = await service.conversations.detail(chat)
            assert [t["status"] for t in detail["turns"]] == ["running", "queued"]
            assert detail["chat"]["status"] == "running"
            assert len(backend.calls) == 1
            backend.release.set()
            detail = await settle(service, chat)
            assert [t["status"] for t in detail["turns"]] == ["completed", "completed"]
            assert any(
                m.content.startswith("Focus on the important detail")
                for m in backend.calls[1]
            )
            assert [t["output"] for t in detail["turns"]] == ["Answer 1", "Answer 2"]
        finally:
            await service.close()

    asyncio.run(check())


def test_multiple_queued_messages_drain_serially_with_only_prior_context(tmp_path):
    async def check():
        backend = ChatBackend(pause_first=True)
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (await service.conversations.create({"message": "First"}))["chat_id"]
            await backend.started.wait()
            await service.conversations.message(chat, {"message": "Second"})
            await service.conversations.message(chat, {"message": "Third"})
            backend.release.set()
            detail = await settle(service, chat)
            assert len(detail["turns"]) == 3 and len(backend.calls) == 3
            assert not any(m.content == "Third" for m in backend.calls[1])
            assert any(m.content == "Second" for m in backend.calls[2])
        finally:
            await service.close()

    asyncio.run(check())


def test_retry_nonce_deduplicates_creates_and_queued_messages_after_reload(tmp_path):
    async def check():
        backend = ChatBackend(pause_first=True)
        service = make_service(tmp_path, backend=backend)
        payload = {"message": "First", "request_id": "create-1"}
        sent = await service.conversations.create(payload)
        duplicate = await service.conversations.create(payload)
        assert duplicate["chat_id"] == sent["chat_id"] and duplicate["duplicate"]
        await backend.started.wait()
        message = {"message": "Second", "request_id": "message-1"}
        queued = await service.conversations.message(sent["chat_id"], message)
        retried = await service.conversations.message(sent["chat_id"], message)
        assert retried["turn_id"] == queued["turn_id"]
        backend.release.set()
        await settle(service, sent["chat_id"])
        await service.close()
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            assert (await service.conversations.create(payload))["chat_id"] == sent[
                "chat_id"
            ]
            assert (await service.conversations.message(sent["chat_id"], message))[
                "turn_id"
            ] == queued["turn_id"]
            detail = await service.conversations.detail(sent["chat_id"])
            assert len(detail["turns"]) == 2
            with pytest.raises(ValueError, match="different message"):
                await service.conversations.message(
                    sent["chat_id"], {"message": "Changed", "request_id": "message-1"}
                )
        finally:
            await service.close()

    asyncio.run(check())


def test_startup_failure_is_visible_and_next_send_recovers(tmp_path, monkeypatch):
    async def check():
        service, backend, attempts = startup_fixture(
            tmp_path, monkeypatch, failure=True
        )
        try:
            chat = (await service.conversations.create({"message": "Start here"}))[
                "chat_id"
            ]
            detail = await settle(service, chat)
            assert detail["turns"][0]["status"] == "failed"
            assert "fixture model unavailable" in detail["turns"][0]["error"]
            assert detail["turns"][0]["run_id"] is None
            await service.conversations.message(chat, {"message": "Try again"})
            detail = await settle(service, chat)
            assert len(attempts) == 2
            assert detail["turns"][1]["output"] == "Answer 1"
            assert len(backend.calls) == 1
        finally:
            await service.close()

    asyncio.run(check())


def test_bounded_startup_timeout_does_not_cancel_shared_loader(tmp_path, monkeypatch):
    async def check():
        gate = asyncio.Event()
        service, backend, attempts = startup_fixture(tmp_path, monkeypatch, gate=gate)
        service.conversations.startup_timeout_seconds = 0.02
        try:
            chat = (await service.conversations.create({"message": "Start"}))["chat_id"]
            detail = await settle(service, chat)
            assert detail["turns"][0]["status"] == "failed"
            assert "startup timed out" in detail["turns"][0]["error"]
            assert not service._loader.done()
            gate.set()
            await service._loader
            await service.conversations.message(chat, {"message": "Now continue"})
            assert (await settle(service, chat))["turns"][-1]["output"] == "Answer 1"
            assert len(attempts) == 1 and len(backend.calls) == 1
        finally:
            await service.close()

    asyncio.run(check())


def test_cancel_loading_preserves_queued_message_and_shared_startup(
    tmp_path, monkeypatch
):
    async def check():
        gate = asyncio.Event()
        service, backend, attempts = startup_fixture(tmp_path, monkeypatch, gate=gate)
        try:
            chat = (
                await service.conversations.create({"message": "Cancelled prompt"})
            )["chat_id"]
            await asyncio.sleep(0.01)
            await service.conversations.message(chat, {"message": "Queued prompt"})
            assert (await service.conversations.cancel(chat, {}))["cancel_requested"]
            assert not service._loader.cancelled()
            gate.set()
            detail = await settle(service, chat)
            assert [t["status"] for t in detail["turns"]] == ["cancelled", "completed"]
            assert len(attempts) == 1 and len(backend.calls) == 1
            assert (
                service.store.agent(detail["turns"][1]["run_id"], "root").task
                == "Queued prompt"
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_cancel_active_reply_drains_queued_followup(tmp_path):
    async def check():
        backend = ChatBackend(pause_first=True)
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (await service.conversations.create({"message": "First"}))["chat_id"]
            await backend.started.wait()
            await service.conversations.message(chat, {"message": "Second"})
            await service.conversations.cancel(chat, {})
            detail = await settle(service, chat)
            assert [t["status"] for t in detail["turns"]] == ["cancelled", "completed"]
            assert detail["turns"][1]["output"] == "Answer 2"
        finally:
            await service.close()

    asyncio.run(check())


def test_immediate_cancel_before_controller_starts_still_drains_queue(tmp_path):
    async def check():
        backend = ChatBackend()
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (
                await service.conversations.create({"message": "Cancel immediately"})
            )["chat_id"]
            await service.conversations.message(chat, {"message": "Next request"})
            await service.conversations.cancel(chat, {})
            detail = await settle(service, chat)
            assert [turn["status"] for turn in detail["turns"]] == [
                "cancelled",
                "completed",
            ]
            assert len(backend.calls) == 1
        finally:
            await service.close()

    asyncio.run(check())


def test_reload_preserves_transcript_and_marks_pending_sends_interrupted(
    tmp_path, monkeypatch
):
    async def check():
        gate = asyncio.Event()
        service, _, _ = startup_fixture(tmp_path, monkeypatch, gate=gate)
        chat = (
            await service.conversations.create({"message": "First durable prompt"})
        )["chat_id"]
        await service.conversations.message(chat, {"message": "Queued durable prompt"})
        await asyncio.sleep(0.01)
        await service.close()
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            detail = await service.conversations.detail(chat)
            assert [t["user"] for t in detail["turns"]] == [
                "First durable prompt",
                "Queued durable prompt",
            ]
            assert all(
                t["status"] == "interrupted" and not t["active"]
                for t in detail["turns"]
            )
            await service.conversations.message(chat, {"message": "Continue"})
            assert (await settle(service, chat))["turns"][-1]["status"] == "completed"
        finally:
            await service.close()

    asyncio.run(check())


def test_old_mission_lineage_is_grouped_and_can_continue_in_chat(tmp_path):
    async def check():
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            first = (
                await service.create_run({"task": "Old first", "policy": "read_only"})
            )["run_id"]
            await service.jobs[first]
            second = (
                await service.create_run(
                    {"task": "Old followup", "policy": "read_only"}, parent_id=first
                )
            )["run_id"]
            await service.jobs[second]
            overview = await service.conversations.overview()
            assert len(overview["chats"]) == 1
            chat = overview["chats"][0]["chat_id"]
            assert [
                t["run_id"] for t in (await service.conversations.detail(chat))["turns"]
            ] == [first, second]
            await service.conversations.message(chat, {"message": "Continue normally"})
            detail = await settle(service, chat)
            assert (
                len(detail["turns"]) == 3
                and detail["turns"][-1]["output"] == "Answer 3"
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_history_bounds_are_explicit_and_latest_request_survives_projection(tmp_path):
    async def check():
        backend = ChatBackend()
        service = make_service(tmp_path, backend=backend)
        service.conversations.history_bytes = 500
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Original long context " + "x" * 3000}
                )
            )["chat_id"]
            await settle(service, chat)
            await service.conversations.message(
                chat, {"message": "Latest exact request"}
            )
            detail = await settle(service, chat)
            turn = detail["turns"][-1]
            assert turn["context_truncated"] is True
            history = service.store.messages(turn["run_id"], "root")
            assert "Conversation history is bounded" in history[1].content
            assert sum(len(m.content.encode()) for m in history[1:-2]) < 1000
            assert any(
                m.content.startswith("Latest exact request") for m in backend.calls[-1]
            )
            assert detail["turns"][0]["user"].endswith("x" * 3000)
        finally:
            await service.close()

    asyncio.run(check())


def test_long_latest_request_is_preserved_when_runtime_projects_inherited_history(
    tmp_path,
):
    async def check():
        backend = ChatBackend()
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Old request " + "x" * 6000}
                )
            )["chat_id"]
            await settle(service, chat)
            latest = "Latest complete request " + "y" * 11900
            await service.conversations.message(chat, {"message": latest})
            detail = await settle(service, chat)
            turn = detail["turns"][-1]
            assert turn["status"] == "completed", turn["error"]
            assert any(
                m.role == "user" and m.content == latest for m in backend.calls[-1]
            )
            assert any(
                event["kind"] == "conversation_context_projected"
                for event in turn["events"]
            )
            assert service.store.messages(turn["run_id"], "root")[-2].content == latest
        finally:
            await service.close()

    asyncio.run(check())


def test_legacy_followup_adopts_current_defaults_without_rewriting_historical_budget(
    tmp_path,
):
    async def check():
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            run_id = (
                await service.create_run(
                    {"task": "Old request", "max_total_tokens": 32000}
                )
            )["run_id"]
            await service.jobs[run_id]
            chat_id = (await service.conversations.overview())["chats"][0]["chat_id"]
            await service.conversations.message(chat_id, {"message": "New request"})
            detail = await settle(service, chat_id)
            assert service.store.run(run_id)["config"]["max_total_tokens"] == 32000
            assert (
                service.store.run(detail["turns"][-1]["run_id"])["config"][
                    "max_total_tokens"
                ]
                == 128000
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_new_private_configuration_raises_tool_timeout_without_rewriting_older_run(
    tmp_path,
):
    async def check():
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            chat = (await service.conversations.create({"message": "First"}))["chat_id"]
            detail = await settle(service, chat)
            original_id = detail["turns"][0]["run_id"]
            row = service.store.connection.execute(
                "SELECT configuration FROM studio_chats WHERE chat_id=?", (chat,)
            ).fetchone()
            config = json.loads(row[0])
            config["tool_timeout_seconds"] = 390
            service.store.connection.execute(
                "UPDATE studio_chats SET configuration=? WHERE chat_id=?",
                (json.dumps(config), chat),
            )
            original_config = service.store.run(original_id)["config"]
            await service.conversations.message(chat, {"message": "Next"})
            detail = await settle(service, chat)
            assert (
                service.store.run(detail["turns"][-1]["run_id"])["config"][
                    "tool_timeout_seconds"
                ]
                == 450
            )
            assert service.store.run(original_id)["config"] == original_config
            assert (
                service.store.run(detail["turns"][-1]["run_id"])["config"][
                    "max_total_tokens"
                ]
                == config["max_total_tokens"]
            )
        finally:
            await service.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 12001},
        {"message": "Okay", "policy": "free"},
        {"message": "Okay", "request_id": []},
        {"message": "Okay", "shell": "anything"},
    ],
)
def test_invalid_sends_create_no_chat(tmp_path, payload):
    async def check():
        service = make_service(tmp_path)
        try:
            with pytest.raises(ValueError):
                await service.conversations.create(payload)
            assert not (await service.conversations.overview())["chats"]
        finally:
            await service.close()

    asyncio.run(check())
