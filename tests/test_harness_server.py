"""Actual loopback HTTP: browser authentication, origin checks and payload bounds."""

import asyncio
import http.client
import json
import threading

import pytest
from instinct.harness.server import AppServer
from instinct.harness.service import AppService


def test_loopback_http_security_and_run_roundtrip(tmp_path):
    async def check():
        (tmp_path / "README.md").write_text("# Fixture\n")
        service = AppService(tmp_path, tmp_path / ".instinct", demo=True)
        service.backend.delay = 0.01
        server = AppServer(("127.0.0.1", 0), service, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None, *, auth=True, headers=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            selected = {"Content-Type": "application/json", "Origin": server.origin}
            if auth:
                selected["Cookie"] = "instinct_session=" + server.session_token
            selected.update(headers or {})
            connection.request(method, path, body=body, headers=selected)
            response = connection.getresponse()
            result = (response.status, response.read(), dict(response.getheaders()))
            connection.close()
            return result

        async def call(*args, **kwargs):
            return await asyncio.to_thread(request, *args, **kwargs)

        try:
            assert (await call("GET", "/api/state", auth=False))[0] == 401
            assert (
                await call(
                    "POST", "/api/auth", json.dumps({"token": "invalid"}), auth=False
                )
            )[0] == 401
            assert (
                await call(
                    "POST", "/api/auth", json.dumps({"token": "invalid-λ"}), auth=False
                )
            )[0] == 401
            status, _, headers = await call(
                "POST",
                "/api/auth",
                json.dumps({"token": server.bootstrap_token}),
                auth=False,
            )
            assert status == 200
            assert "HttpOnly; SameSite=Strict" in headers["Set-Cookie"]
            assert (
                await call("GET", "/api/state", headers={"Host": "attacker.example"})
            )[0] == 403
            assert (
                await call(
                    "POST",
                    "/api/runs",
                    "{}",
                    headers={"Origin": "https://attacker.example"},
                )
            )[0] == 403
            assert (
                await call(
                    "POST", "/api/runs", "{}", headers={"Content-Type": "text/plain"}
                )
            )[0] == 400
            assert (await call("POST", "/api/runs", '{"task":"a","task":"b"}'))[
                0
            ] == 400
            assert (
                await call(
                    "POST", "/api/runs", "{}", headers={"Content-Length": "128001"}
                )
            )[0] == 400
            status, body, _ = await call(
                "POST",
                "/api/runs",
                json.dumps({"task": "Inspect evidence", "max_agents": 1}),
            )
            assert status == 200, body
            run_id = json.loads(body)["run_id"]
            await asyncio.wait_for(service.jobs[run_id], 5)
            status, body, headers = await call("GET", f"/api/runs/{run_id}/export")
            assert status == 200
            assert json.loads(body)["run"]["status"] == "completed"
            assert headers["Content-Disposition"].endswith('"instinct-trace.json"')
            assert (await call("GET", "/../../pyproject.toml"))[0] == 404
            assert headers["X-Content-Type-Options"] == "nosniff"
            first_payload = json.dumps(
                {
                    "message": "Inspect this fixture",
                    "policy": "read_only",
                    "request_id": "http-chat-first",
                }
            )
            status, body, _ = await call("POST", "/api/chats", first_payload)
            assert status == 200, body
            first = json.loads(body)
            status, body, _ = await call("POST", "/api/chats", first_payload)
            assert status == 200
            assert json.loads(body)["turn_id"] == first["turn_id"]
            chat_id = first["chat_id"]
            status, body, _ = await call(
                "POST",
                f"/api/chats/{chat_id}/message",
                json.dumps(
                    {
                        "message": "Explain the inspection",
                        "request_id": "http-chat-followup",
                    }
                ),
            )
            assert status == 200, body
            second = json.loads(body)
            await asyncio.wait_for(service.conversations.jobs[first["turn_id"]], 5)
            await asyncio.wait_for(service.conversations.jobs[second["turn_id"]], 5)
            status, body, _ = await call("GET", f"/api/chats/{chat_id}")
            assert status == 200
            chat = json.loads(body)
            assert len(chat["turns"]) == 2
            assert all(turn["status"] == "completed" for turn in chat["turns"])
            assert chat["turns"][1]["user"] == "Explain the inspection"
            assert len(chat["transcript"]) == 4
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await service.close()

    asyncio.run(check())


def test_public_binding_rejected():
    with pytest.raises(ValueError, match="127.0.0.1"):
        AppServer(("0.0.0.0", 0), None, None)
