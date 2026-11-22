"""The owner's side of sharing: the app stays on loopback, and invites are
made, listed and revoked through the app's own API."""

import asyncio
import http.client
import json
import threading

import pytest
from instinct.harness.server import AppServer
from instinct.harness.service import AppService
from instinct.harness.sharing import Sharing


@pytest.fixture
def owner(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def build():
        return AppService(tmp_path, tmp_path / ".state", demo=True)

    service = asyncio.run_coroutine_threadsafe(build(), loop).result()
    sharing = Sharing(tmp_path / "home", lambda: service, loop, funnel=False)
    service.sharing = sharing

    def call(method, path, body=None):
        return asyncio.run_coroutine_threadsafe(
            service.dispatch(method, path, body or {}), loop
        ).result(10)

    yield service, sharing, call
    sharing.stop()
    asyncio.run_coroutine_threadsafe(service.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)


def test_the_app_server_only_ever_binds_to_loopback(owner):
    service, _, _ = owner
    for host in ("0.0.0.0", "192.168.1.20", "::"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            AppServer((host, 0), service, None)


def test_api_keys_no_longer_sign_browsers_into_the_app(owner, tmp_path):
    service, _, _ = owner
    key = service.api_keys.create("phone")["key"]
    loop = asyncio.new_event_loop()
    server = AppServer(("127.0.0.1", 0), service, loop, token="t" * 40)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(
            "POST",
            "/api/auth",
            json.dumps({"key": key}),
            {
                "Content-Type": "application/json",
                "Host": f"127.0.0.1:{server.server_port}",
            },
        )
        response = connection.getresponse()
        assert response.status == 401 and response.getheader("Set-Cookie") is None
    finally:
        server.shutdown()
        server.server_close()


def test_sharing_is_off_until_turned_on_and_invites_are_managed(owner):
    _, sharing, call = owner
    status = call("GET", "/api/guests")
    assert status["enabled"] is False and status["listening"] is False
    status = call("POST", "/api/guests/sharing", {"enabled": True, "port": 18766})
    assert status["listening"] and status["local_url"].startswith("http://127.0.0.1:")
    # The guest listener is loopback only; Funnel is what publishes it.
    assert sharing.server.server_address[0] == "127.0.0.1"
    made = call(
        "POST",
        "/api/guests/invites",
        {"name": "Interviewer", "hours": 2, "limits": {"concurrent": 4}},
    )
    assert made["token"].startswith("inv_")
    assert made["link"].endswith("/#invite=" + made["token"])
    listed = call("GET", "/api/guests")["invites"]
    assert listed[0]["name"] == "Interviewer" and "token" not in listed[0]
    status = call("POST", f"/api/guests/invites/{made['id']}/revoke")
    assert status["invites"][0]["active"] is False
    assert sharing.invites.check(made["token"]) is None
    status = call("POST", "/api/guests/sharing", {"enabled": False})
    assert status["listening"] is False
