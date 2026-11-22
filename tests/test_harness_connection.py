"""A guest's own app using a host's models through the gateway."""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest
from instinct.agents.cli import create_pool
from instinct.agents.protocol import ChatMessage
from instinct.harness.connection import TOKEN_ENV, HostConnection, parse_link
from instinct.harness.gateway import Gateway, GatewayServer
from instinct.harness.guests import Invites
from instinct.harness.sharing import Sharing
from test_harness_guests import CONFIG, Pool


@pytest.fixture
def host(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    pool = Pool()
    service = SimpleNamespace(
        backend=pool,
        backend_status="ready",
        demo=False,
        config=CONFIG,
        identities={
            "gemma-4": {
                "model": "/Users/owner/models/gemma-4-26b-a4b-mlx",
                "model_data_sha256": "a" * 64,
                "model_config_sha256": "b" * 64,
                "tokenizer_signature": "c" * 64,
            }
        },
    )
    invites = Invites(tmp_path / "host" / "invites.json")
    server = GatewayServer(0, Gateway(invites, lambda: service, loop))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    made = invites.create("Guest laptop", limits={"concurrent": 3})
    yield SimpleNamespace(
        pool=pool,
        invites=invites,
        invite=made,
        link=f"http://127.0.0.1:{server.server_port}/#invite={made['token']}",
    )
    server.shutdown()
    server.server_close()
    loop.call_soon_threadsafe(loop.stop)
    os.environ.pop(TOKEN_ENV, None)


def test_links_are_parsed_strictly():
    url, invite = parse_link("https://mac.tail1234.ts.net/#invite=inv_" + "a" * 43)
    assert url == "https://mac.tail1234.ts.net" and invite.startswith("inv_")
    for bad in (
        "https://mac.ts.net/",
        "http://203.0.113.9/#invite=inv_" + "a" * 43,  # plain HTTP off this Mac
        "ftp://mac/#invite=inv_" + "a" * 43,
        42,
    ):
        with pytest.raises(ValueError):
            parse_link(bad)


def test_guests_see_model_fingerprints_but_not_the_hosts_paths(host):
    from instinct.harness.connection import health, parse_link

    identity = health(*parse_link(host.link))
    assert identity["model_data_sha256"] == "a" * 64
    assert identity["model"] == "gemma-4-26b-a4b-mlx"
    assert "/Users/owner" not in str(identity)


def test_a_guest_app_runs_its_models_on_the_host(host, tmp_path):
    connection = HostConnection(tmp_path / "guest" / "host-connection.json")
    status = connection.connect(host.link)
    assert status["connected"] and status["host"].startswith("http://127.0.0.1:")
    saved = connection.path
    assert saved.stat().st_mode & 0o777 == 0o600
    config = connection.config()
    assert [w["route"] for w in config["workers"]] == ["driver", "subagent"]
    assert all(w["max_concurrency"] == 3 for w in config["workers"])
    assert host.invite["token"] not in str(config)  # The token stays in the env.
    host.pool.script = ["From the host."]

    async def ask():
        pool = await create_pool(config)
        try:
            return await pool.generate(
                [ChatMessage("user", "Hi")], 64, "turn-1", group="subagent"
            )
        finally:
            await pool.close()

    assert asyncio.run(ask()).text == "From the host."
    assert host.pool.calls[-1][2]["group"] == "subagent"
    # Revoked on the host: the guest app finds out on its next connect.
    host.invites.revoke(host.invite["id"])
    with pytest.raises(ValueError, match="expired or been revoked"):
        connection.connect(host.link)
    assert connection.disconnect()["connected"] is False and not saved.exists()


def test_a_guest_mac_cannot_reshare_borrowed_models(tmp_path):
    sharing = Sharing(tmp_path, lambda: None, asyncio.new_event_loop(), funnel=False)
    sharing.blocked = "This Mac uses another Mac's models, so it can't share them."
    with pytest.raises(ValueError, match="can't share"):
        sharing.update({"enabled": True})
    with pytest.raises(ValueError, match="can't share"):
        sharing.dispatch("POST", ["api", "guests", "invites"], {"name": "x"})
    assert sharing.status()["listening"] is False
