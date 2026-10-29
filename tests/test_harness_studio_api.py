"""Chat settings, rename/delete, the live snapshot and its event stream."""

import asyncio
import http.client
import json
import re
import threading

import pytest
from forge_llm.harness.server import StudioHTTPServer
from forge_llm.harness.service import worker_state
from test_harness_conversation import ChatBackend, make_service, settle

WORKERS = [
    {"name": "gemma-4", "group": "driver", "model": "models/gemma-4-26b-a4b-mlx"},
    {"name": "qwen-1", "group": "subagent", "model": "models/qwen3.5-4b-mlx"},
]


def test_model_and_effort_follow_the_chat_into_its_runs(tmp_path):
    async def check():
        backend = ChatBackend()
        service = make_service(tmp_path, backend=backend, config={"workers": WORKERS})
        try:
            models = await service.dispatch("GET", "/api/models", {})
            assert [(m["id"], m["label"]) for m in models["models"]] == [
                ("driver", "Gemma 4 26B A4B"),
                ("subagent", "Qwen3.5-4B"),
            ]
            assert models["efforts"] == ["off", "low", "medium", "high"]
            chat = (
                await service.conversations.create(
                    {"message": "Hi", "model": "subagent", "effort": "low"}
                )
            )["chat_id"]
            detail = await settle(service, chat)
            config = service.store.run(detail["turns"][0]["run_id"])["config"]
            assert (config["driver_group"], config["effort"]) == ("subagent", "low")
            assert detail["chat"]["model"] == "subagent"

            settings = await service.dispatch(
                "POST", f"/api/chats/{chat}/settings", {"effort": "high"}
            )
            assert (settings["model"], settings["effort"]) == ("subagent", "high")
            await service.conversations.message(
                chat, {"message": "Again", "model": "driver"}
            )
            detail = await settle(service, chat)
            config = service.store.run(detail["turns"][1]["run_id"])["config"]
            assert (config["driver_group"], config["effort"]) == ("driver", "high")

            for bad in ({"model": "gpt"}, {"effort": "max"}, {"title": "x"}, {}):
                with pytest.raises(ValueError):
                    await service.dispatch("POST", f"/api/chats/{chat}/settings", bad)
        finally:
            await service.close()

    asyncio.run(check())


def test_rename_and_delete(tmp_path):
    async def check():
        backend = ChatBackend(pause_first=True)
        service = make_service(tmp_path, backend=backend)
        try:
            chat = (await service.conversations.create({"message": "First"}))["chat_id"]
            renamed = await service.dispatch(
                "POST", f"/api/chats/{chat}/rename", {"title": "  Repo   tour "}
            )
            assert renamed["title"] == "Repo tour"
            with pytest.raises(ValueError):
                await service.dispatch(
                    "POST", f"/api/chats/{chat}/rename", {"title": "   "}
                )
            # Deleting a chat mid-reply stops the reply too.
            await backend.started.wait()
            run_id = (await service.dispatch("GET", f"/api/chats/{chat}", {}))["turns"][
                0
            ]["run_id"]
            await service.dispatch("POST", f"/api/chats/{chat}/delete", {})
            overview = await service.dispatch("GET", "/api/chats", {})
            assert all(item["chat_id"] != chat for item in overview["chats"])
            with pytest.raises(ValueError):
                await service.dispatch("GET", f"/api/chats/{chat}", {})
            await asyncio.gather(service.jobs[run_id], return_exceptions=True)
            assert service.store.run(run_id)["status"] == "cancelled"
        finally:
            backend.release.set()
            await service.close()

    asyncio.run(check())


def test_history_trims_whole_blocks_so_the_prompt_prefix_stays_put(tmp_path):
    async def check():
        backend = ChatBackend()
        service = make_service(tmp_path, backend=backend)
        service.conversations.history_bytes = 1_200
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Message 0 " + "x" * 200}
                )
            )["chat_id"]
            await settle(service, chat)
            first_kept = []
            for n in range(1, 12):
                await service.conversations.message(
                    chat, {"message": f"Message {n} " + "x" * 200}
                )
                detail = await settle(service, chat)
                history = service.store.messages(detail["turns"][-1]["run_id"], "root")
                kept = [
                    int(match.group(1))
                    for message in history
                    if (match := re.match(r"Message (\d+) ", message.content))
                ]
                first_kept.append(kept[0])
                assert kept[-1] == n
            # Old turns go in batches: the start of the history moves rarely,
            # never backwards, and the latest messages are always there.
            assert first_kept == sorted(first_kept) and first_kept[-1] > 0
            assert len(set(first_kept)) <= 4
        finally:
            await service.close()

    asyncio.run(check())


def test_live_snapshot_versions_change_with_the_journal(tmp_path):
    async def check():
        service = make_service(tmp_path, backend=ChatBackend())
        try:
            before = await service.dispatch("GET", "/api/live", {})
            assert before["generations"] == [] and before["backend"]["status"]
            chat = (await service.conversations.create({"message": "Hi"}))["chat_id"]
            await settle(service, chat)
            after = await service.dispatch("GET", "/api/live", {})
            assert after["version"] != before["version"]
        finally:
            await service.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    ("health", "state"),
    [
        ({"error": "down"}, "error"),
        ({"started": False}, "stopped"),
        ({"started": False, "starting": True}, "starting"),
        ({"loaded": False, "active": 1}, "starting"),
        ({"loaded": True, "active": 1}, "busy"),
        ({"loaded": True, "active": 0}, "idle"),
    ],
)
def test_worker_state(health, state):
    assert worker_state(health) == state


def test_event_stream_requires_auth_and_sends_the_live_snapshot(tmp_path):
    async def check():
        service = make_service(tmp_path, backend=ChatBackend())
        server = StudioHTTPServer(("127.0.0.1", 0), service, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def first_event(auth):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            headers = {"Origin": server.origin}
            if auth:
                headers["Cookie"] = "forge_studio=" + server.session_token
            connection.request("GET", "/api/events", headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                connection.close()
                return response.status, None
            assert response.getheader("Content-Type") == "text/event-stream"
            line = response.fp.readline().decode()
            connection.close()
            return response.status, json.loads(line.removeprefix("data: "))

        try:
            assert (await asyncio.to_thread(first_event, False))[0] == 401
            status, event = await asyncio.to_thread(first_event, True)
            assert (
                status == 200 and {"version", "generations", "workers"} <= event.keys()
            )
        finally:
            service.closing = True
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await service.close()

    asyncio.run(check())
