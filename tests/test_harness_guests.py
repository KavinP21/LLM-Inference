"""Guests over the gateway: invites, browser runs with relayed tools, quotas,
revocation, and the job protocol for guests' own apps."""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest
from instinct.agents.backends import RemoteWorkerBackend
from instinct.agents.backends.config import AuthenticationError
from instinct.agents.protocol import ChatMessage, Generation
from instinct.harness.gateway import Gateway, GatewayServer
from instinct.harness.guests import Invites, validate_limits


def call(name, **arguments):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


class Pool:
    """Stands in for the owner's model workers."""

    routes = True
    supports_native_tools = True
    supports_context_limits = True
    native_call_dialect = "json"

    def __init__(self, script=(), delay=0.0):
        self.script = list(script)
        self.calls = []
        self.delay = delay
        self.cancelled = []

    def context_limit(self, group):
        return 32768

    def partial(self, request_id):
        return None

    async def _answer(self, messages, request_id, options):
        self.calls.append((request_id, list(messages), options))
        await asyncio.sleep(self.delay)
        text = self.script.pop(0) if self.script else "Done."
        return Generation(text, 100, 10)

    async def generate(self, messages, max_tokens, request_id, **options):
        return await self._answer(messages, request_id, options)

    async def generate_action(self, messages, max_tokens, request_id, specs, **options):
        return await self._answer(messages, request_id, options)

    async def cancel(self, request_id):
        self.cancelled.append(request_id)


CONFIG = {
    "workers": [
        {"name": "gemma-4", "group": "driver", "model": "models/gemma-4-26b-a4b-mlx"},
        {"name": "qwen", "group": "subagent", "model": "models/qwen3.5-4b-mlx"},
        {"name": "secret", "group": "abliterated", "model": "models/other"},
    ],
    "runtime": {
        "driver_group": "driver",
        "subagent_group": "subagent",
        "max_context_tokens": 32768,
        "max_output_tokens": 2048,
    },
}


@pytest.fixture
def host(tmp_path):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    pool = Pool()
    service = SimpleNamespace(
        backend=pool, backend_status="ready", demo=False, config=CONFIG
    )
    invites = Invites(tmp_path / "invites.json")
    gateway = Gateway(invites, lambda: service, loop)
    server = GatewayServer(0, gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(
        gateway=gateway, invites=invites, pool=pool, port=server.server_port, loop=loop
    )
    server.shutdown()
    server.server_close()
    gateway.close()
    loop.call_soon_threadsafe(loop.stop)


class Browser:
    """A browser on the guest's machine: same-origin requests with a cookie."""

    def __init__(self, port):
        self.port, self.cookie = port, None

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        sent = {"Origin": f"http://127.0.0.1:{self.port}"}
        if body is not None:
            sent["Content-Type"] = "application/json"
        if self.cookie:
            sent["Cookie"] = self.cookie
        sent.update(headers or {})
        connection.request(
            method, path, json.dumps(body) if body is not None else None, sent
        )
        response = connection.getresponse()
        data = json.loads(response.read() or b"{}")
        cookie = response.getheader("Set-Cookie")
        if cookie and "Max-Age=0" not in cookie:
            self.cookie = cookie.split(";")[0]
        connection.close()
        return response.status, data

    def sign_in(self, token):
        return self.request("POST", "/guest/session", {"invite": token})


def run_to_end(browser, run_id, files):
    """Poll the run, carrying out tool calls on ``files`` (the guest's folder)."""
    after, deadline = 0, time.time() + 20
    while time.time() < deadline:
        _, page = browser.request(
            "GET", f"/guest/runs/{run_id}/events?after={after}&wait=2"
        )
        after = page["next"]
        for event in page["events"]:
            if event["type"] == "tool":
                name, args = event["name"], event["args"]
                if name == "read_file" and args["path"] in files:
                    answer = {"result": json.dumps({"content": files[args["path"]]})}
                elif name == "write_file":
                    files[args["path"]] = args["content"]
                    answer = {"result": json.dumps({"path": args["path"]})}
                else:
                    answer = {"error": f"no such file: {args.get('path')}"}
                status, _ = browser.request(
                    "POST", f"/guest/runs/{run_id}/tools/{event['id']}", answer
                )
                assert status == 200
            if event["type"] == "done":
                return event
    raise AssertionError("the run didn't finish")


def test_invites_store_only_hashes_and_expire(tmp_path):
    invites = Invites(tmp_path / "invites.json")
    made = invites.create("Interviewer", hours=1, limits={"concurrent": 3})
    token = made["token"]
    saved = (tmp_path / "invites.json").read_text()
    assert token not in saved and made["id"] in saved
    assert (tmp_path / "invites.json").stat().st_mode & 0o777 == 0o600
    assert "token" not in invites.list()[0]
    assert invites.check(token)["limits"]["concurrent"] == 3
    assert invites.check(token + "x") is None and invites.check("sk-other") is None
    invites.revoke(made["id"])
    assert invites.check(token) is None and not invites.list()[0]["active"]
    short = invites.create("Brief", hours=1 / 3600)
    time.sleep(1.1)
    assert invites.check(short["token"]) is None
    with pytest.raises(ValueError):
        validate_limits({"concurrent": 99})
    with pytest.raises(ValueError):
        validate_limits({"output_tokens": 8192, "context_tokens": 8192})


def test_browser_run_relays_tools_and_keeps_nothing(host):
    host.pool.script = [
        call("read_file", path="notes.md"),
        call("write_file", path="summary.md", content="Short version."),
        "I read notes.md and wrote summary.md.",
    ]
    token = host.invites.create("Guest")["token"]
    browser = Browser(host.port)
    status, me = browser.sign_in(token)
    assert status == 200 and me["name"] == "Guest"
    assert [m["id"] for m in me["models"]] == ["driver", "subagent"]
    files = {"notes.md": "The guest's own notes."}
    status, started = browser.request(
        "POST",
        "/guest/runs",
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "task": "Summarize my notes into summary.md",
            "tools": ["read_file", "write_file", "list_files"],
        },
    )
    assert status == 200, started
    done = run_to_end(browser, started["run_id"], files)
    assert done["status"] == "completed"
    assert done["result"] == "I read notes.md and wrote summary.md."
    assert files["summary.md"] == "Short version."
    # The model saw the earlier turns and the browser's tool result.
    _, messages, options = host.pool.calls[-1]
    contents = " ".join(m.content for m in messages)
    assert "Hello!" in contents and "The guest's own notes." in contents
    assert options["group"] == "driver"
    assert options["affinity"].startswith("guest:")
    # Nothing of the run is kept: no journal, no files on the host.
    run = host.gateway.runs.runs[started["run_id"]]
    assert run.store is None and run.runtime is None
    assert host.invites.list()[0]["tokens_today"] == 3 * 110


def test_guests_never_reach_the_owner(host):
    token = host.invites.create("Guest")["token"]
    browser = Browser(host.port)
    browser.sign_in(token)
    for path in ("/api/state", "/api/chats", "/v1/models", "/api/folders"):
        assert browser.request("GET", path)[0] == 404
    assert browser.request("POST", "/v1/chat/completions", {"messages": []})[0] in {
        403,
        404,
    }
    # Another site can't drive a signed-in browser.
    status, _ = browser.request(
        "POST", "/guest/runs", {"task": "x"}, {"Origin": "https://evil.example"}
    )
    assert status == 403
    # The owner's other models stay off limits.
    status, _ = browser.request("POST", "/guest/runs", {"task": "x", "model": "secret"})
    assert status == 400


def test_revoking_signs_out_and_stops_running_work(host):
    host.pool.script = [call("read_file", path="a.md")]
    made = host.invites.create("Guest")
    browser = Browser(host.port)
    browser.sign_in(made["token"])
    _, started = browser.request(
        "POST", "/guest/runs", {"task": "Read a.md", "tools": ["read_file"]}
    )
    # Wait for the tool call, then revoke before answering it.
    page = {"events": []}
    while not any(e["type"] == "tool" for e in page["events"]):
        _, page = browser.request(
            "GET", f"/guest/runs/{started['run_id']}/events?wait=2"
        )
    host.invites.revoke(made["id"])
    host.gateway.revoke(made["id"])
    deadline = time.time() + 10
    while host.gateway.runs.active(made["id"]) and time.time() < deadline:
        time.sleep(0.05)
    assert host.gateway.runs.active(made["id"]) == 0
    assert browser.request("GET", "/guest/me")[0] == 401
    assert browser.sign_in(made["token"])[0] == 401


def test_quotas_limit_concurrency_and_daily_tokens(host):
    host.pool.delay = 0.5
    made = host.invites.create(
        "Guest", limits={"concurrent": 1, "daily_tokens": 10_000}
    )
    browser = Browser(host.port)
    browser.sign_in(made["token"])
    first = browser.request("POST", "/guest/runs", {"task": "one"})
    second = browser.request("POST", "/guest/runs", {"task": "two"})
    assert first[0] == 200 and second[0] == 400
    run_to_end(browser, first[1]["run_id"], {})
    host.invites.charge(made["id"], 10_000)
    status, error = browser.request("POST", "/guest/runs", {"task": "three"})
    assert status == 400 and "tokens for today" in error["error"]["message"]


def test_guest_apps_use_the_job_protocol_with_the_invite(host):
    host.pool.script = ["Hello from the host's model."]
    token = host.invites.create("Laptop")["token"]
    url = f"http://127.0.0.1:{host.port}"

    async def ask(secret):
        backend = RemoteWorkerBackend(url, secret, route="subagent", request_timeout=20)
        try:
            return await backend.generate([ChatMessage("user", "Hi")], 64, "req-1")
        finally:
            await backend.close()

    generation = asyncio.run(ask(token))
    assert generation.text == "Hello from the host's model."
    _, _, options = host.pool.calls[-1]
    assert options["group"] == "subagent"
    assert host.pool.calls[-1][0].startswith("guest.")
    with pytest.raises(AuthenticationError):
        asyncio.run(ask("inv_not-a-real-invite"))


def test_the_guest_app_is_served_with_a_strict_policy(host):
    connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=5)
    for path, kind in (
        ("/", "text/html"),
        ("/guest.js", "text/javascript"),
        ("/guest.css", "text/css"),
    ):
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.read()
        assert response.status == 200 and response.getheader("Content-Type").startswith(
            kind
        )
        policy = response.getheader("Content-Security-Policy")
        assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
