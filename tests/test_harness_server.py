"""Actual loopback HTTP: browser authentication, origin checks and payload bounds."""

import asyncio
import http.client
import json
import threading

import pytest
from forge_llm.harness.server import StudioHTTPServer
from forge_llm.harness.service import StudioService


def test_loopback_http_security_and_run_roundtrip(tmp_path):
    async def check():
        (tmp_path / "README.md").write_text("# Fixture\n")
        service = StudioService(tmp_path, tmp_path / ".forge", demo=True)
        service.backend.delay = 0.01
        server = StudioHTTPServer(("127.0.0.1", 0), service, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None, *, auth=True, headers=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            selected = {"Content-Type": "application/json", "Origin": server.origin}
            if auth:
                selected["Cookie"] = "forge_studio=" + server.session_token
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
            assert headers["Content-Disposition"].endswith('"forge-trace.json"')
            assert (await call("GET", "/../../pyproject.toml"))[0] == 404
            assert headers["X-Content-Type-Options"] == "nosniff"
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await service.close()

    asyncio.run(check())


def test_public_binding_rejected():
    with pytest.raises(ValueError, match="127.0.0.1"):
        StudioHTTPServer(("0.0.0.0", 0), None, None)
