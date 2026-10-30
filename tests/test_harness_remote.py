"""Serving other devices: key sign-in, origins, TLS and network settings."""

import asyncio
import http.client
import json
import shutil
import ssl
import subprocess
import threading
from types import SimpleNamespace

import pytest
from instinct.harness.cli import remote_settings
from instinct.harness.server import AppServer
from instinct.harness.service import AppService, network_status, save_network


@pytest.fixture
def remote(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def build():
        return AppService(
            tmp_path, tmp_path / ".state", demo=True, keys_file=tmp_path / "keys.json"
        )

    service = asyncio.run_coroutine_threadsafe(build(), loop).result()
    servers = []

    def start(tls=None):
        server = AppServer(
            ("127.0.0.1", 0), service, loop, token="t" * 40, tls=tls, remote=True
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield service, start
    for server in servers:
        server.shutdown()
        server.server_close()
    asyncio.run_coroutine_threadsafe(service.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)


def request(server, method, path, body=None, *, host=None, headers=(), context=None):
    host = host or f"phone-name.local:{server.server_port}"
    if context is not None:
        connection = http.client.HTTPSConnection(
            "127.0.0.1", server.server_port, timeout=5, context=context
        )
    else:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
    sent = {"Host": host, "Content-Type": "application/json", **dict(headers)}
    connection.request(
        method, path, json.dumps(body) if body is not None else None, sent
    )
    response = connection.getresponse()
    return response.status, response.getheader("Set-Cookie"), response.read()


def test_other_devices_sign_in_with_an_app_key(remote):
    service, start = remote
    server = start()
    api_key = service.api_keys.create("laptop")["key"]
    app_key = service.api_keys.create("phone", "app")["key"]
    assert request(server, "GET", "/")[0] == 200
    assert request(server, "GET", "/api/settings")[0] == 401
    # The Mac app's one-time token doesn't work here, nor does an API-only key.
    assert request(server, "POST", "/api/auth", {"token": "t" * 40})[0] == 401
    assert request(server, "POST", "/api/auth", {"key": api_key})[0] == 401
    status, cookie, _ = request(server, "POST", "/api/auth", {"key": app_key})
    assert status == 200 and "HttpOnly" in cookie and "Secure" not in cookie
    session = cookie.split(";")[0]
    assert (
        request(server, "GET", "/api/settings", headers={"Cookie": session})[0] == 200
    )
    # Any name the device used works on this port; another site's origin doesn't.
    elsewhere = {"Cookie": session, "Origin": "https://evil.example"}
    assert request(server, "GET", "/api/settings", headers=elsewhere)[0] == 403
    assert (
        request(
            server,
            "GET",
            "/api/settings",
            host="phone-name.local:1",
            headers={"Cookie": session},
        )[0]
        == 403
    )
    status, _, body = request(
        server, "GET", "/v1/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    assert status == 200 and json.loads(body)["object"] == "list"


@pytest.mark.skipif(
    shutil.which("openssl") is None, reason="needs openssl to make a certificate"
)
def test_tls_sessions_get_secure_cookies(remote, tmp_path):
    service, start = remote
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-keyout",
            key,
            "-out",
            cert,
        ],
        check=True,
        capture_output=True,
    )
    settings = remote_settings(
        SimpleNamespace(
            remote_host="127.0.0.1",
            remote_port=0,
            tls_cert=cert,
            tls_key=key,
            allow_http=False,
            network_file=None,
        )
    )
    server = start(settings["tls"])
    client = ssl.create_default_context()
    client.check_hostname, client.verify_mode = False, ssl.CERT_NONE
    app_key = service.api_keys.create("phone", "app")["key"]
    status, cookie, _ = request(
        server, "POST", "/api/auth", {"key": app_key}, context=client
    )
    assert status == 200 and "Secure" in cookie


def test_remote_settings_need_tls_or_an_explicit_http_choice(tmp_path):
    def args(**values):
        defaults = dict(
            remote_host=None,
            remote_port=None,
            tls_cert=None,
            tls_key=None,
            allow_http=False,
            network_file=None,
        )
        return SimpleNamespace(**{**defaults, **values})

    assert remote_settings(args()) is None
    with pytest.raises(ValueError, match="allow-http"):
        remote_settings(args(remote_host="0.0.0.0"))
    assert remote_settings(args(remote_host="0.0.0.0", allow_http=True)) == {
        "host": "0.0.0.0",
        "port": 8765,
        "tls": None,
    }
    saved = tmp_path / "network.json"
    save_network(saved, {"enabled": True, "port": 9000, "allow_http": True})
    assert remote_settings(args(network_file=saved))["port"] == 9000
    for bad in (
        {"enabled": True},
        {"port": 80},
        {"tls_cert": str(tmp_path / "missing.pem")},
        {"surprise": 1},
    ):
        with pytest.raises(ValueError):
            save_network(saved, bad)
    status = network_status(SimpleNamespace(network_file=saved, remote_url=None))
    assert (
        status["settings"]["port"] == 9000
        and status["editable"]
        and status["addresses"]
    )
