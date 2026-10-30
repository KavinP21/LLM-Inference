"""Images persist once, remain private, and travel through real chat histories."""

import asyncio
import base64
import hashlib
import http.client
import json
import os
import struct
import threading
import zlib
from dataclasses import asdict

import pytest
from instinct.agents import media as media_module
from instinct.agents.backends import WorkerPool
from instinct.agents.media import MAX_MEDIA_REQUEST_BYTES
from instinct.agents.protocol import Generation
from instinct.agents.runtime import RuntimeConfig
from instinct.agents.store import AgentStore
from instinct.harness.server import MAX_BODY, AppHandler, AppServer
from instinct.harness.service import AppService


def png_bytes(width=2, height=2, *, noisy=False):
    def chunk(kind, content):
        return (
            struct.pack("!I", len(content))
            + kind
            + content
            + struct.pack("!I", zlib.crc32(kind + content))
        )

    rows = b"".join(
        b"\x00" + (os.urandom(width * 4) if noisy else b"\xff\x00\x00\xff" * width)
        for _ in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def image(data=None, *, mime="image/png", kind="image"):
    data = png_bytes() if data is None else data
    return {
        "type": kind,
        "mime": mime,
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": base64.b64encode(data).decode(),
    }


class ImageBackend:
    supports_media = True
    supports_context_limits = True
    max_model_length = 32768

    def __init__(self, *, pause_first=False):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.pause_first = pause_first

    async def generate(self, messages, max_tokens, request_id, **options):
        self.calls.append(list(messages))
        if any(message.media for message in messages):
            assert options.get("max_context_tokens") == 24000
        self.started.set()
        if self.pause_first and len(self.calls) == 1:
            await self.release.wait()
        return Generation(
            json.dumps({"action": "finish", "result": "Image fixture answer"}), 500, 30
        )

    async def cancel(self, request_id):
        return None

    async def close(self):
        return None


def make_service(tmp_path, *, backend=None):
    return AppService(
        tmp_path, tmp_path / ".instinct", demo=True, backend=backend or ImageBackend()
    )


async def settle(service, chat_id):
    for _ in range(500):
        detail = await service.conversations.detail(chat_id)
        if all(
            turn["status"] not in {"loading", "queued", "running"}
            for turn in detail["turns"]
        ):
            return detail
        await asyncio.sleep(0.01)
    pytest.fail("image conversation did not settle")


def test_image_only_send_persists_metadata_and_private_preview_before_reply(tmp_path):
    async def check():
        service = make_service(tmp_path)
        attached = image()
        try:
            sent = await service.dispatch(
                "POST",
                "/api/chats",
                {"message": "", "media": [attached], "media_names": ["red-square.png"]},
            )
            detail = await service.conversations.detail(sent["chat_id"])
            turn = detail["turns"][0]
            assert turn["user"] == "Describe the attached image."
            assert turn["media"][0]["name"] == "red-square.png"
            assert turn["media"][0]["size"] == len(png_bytes())
            assert "data" not in turn["media"][0]
            assert attached["data"] not in json.dumps(detail)
            assert (
                await service.dispatch("GET", turn["media"][0]["url"], {}) == attached
            )
            assert (
                tmp_path / ".instinct" / "instinct.sqlite"
            ).stat().st_mode & 0o777 == 0o600
            detail = await settle(service, sent["chat_id"])
            assert detail["turns"][0]["status"] == "completed", detail["turns"][0][
                "error"
            ]
            assert any(
                message.media == (attached,) for message in service.backend.calls[0]
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_followup_keeps_original_image_on_its_original_user_role_and_deduplicates_storage(
    tmp_path,
):
    async def check():
        service = make_service(tmp_path)
        attached = image()
        try:
            chat = (
                await service.conversations.create(
                    {"message": "What is in this image?", "media": [attached]}
                )
            )["chat_id"]
            await settle(service, chat)
            await service.conversations.message(chat, {"message": "What color is it?"})
            detail = await settle(service, chat)
            second = service.backend.calls[1]
            assert any(
                message.role == "user"
                and message.content == "What is in this image?"
                and message.media == (attached,)
                for message in second
            )
            assert any(
                message.content.startswith("What color is it?") and not message.media
                for message in second
            )
            await service.conversations.message(
                chat, {"message": "Same image again", "media": [attached]}
            )
            await settle(service, chat)
            assert (
                service.store.connection.execute(
                    "SELECT COUNT(*) FROM media_blobs"
                ).fetchone()[0]
                == 1
            )
            assert sum(len(message.media) for message in service.backend.calls[2]) == 1
            assert len(detail["turns"]) == 2
        finally:
            await service.close()

    asyncio.run(check())


def test_image_nonce_dedup_persists_through_reload_and_preview_scope_is_chat_local(
    tmp_path,
):
    async def check():
        attached = image()
        payload = {
            "message": "Image request",
            "media": [attached],
            "media_names": ["one.png"],
            "request_id": "image-nonce",
        }
        service = make_service(tmp_path)
        sent = await service.conversations.create(payload)
        await settle(service, sent["chat_id"])
        await service.close()
        service = make_service(tmp_path)
        try:
            duplicate = await service.conversations.create(payload)
            assert duplicate["duplicate"] and duplicate["turn_id"] == sent["turn_id"]
            other = (await service.conversations.create({"message": "Unrelated chat"}))[
                "chat_id"
            ]
            with pytest.raises(ValueError, match="not part"):
                await service.conversations.media(other, attached["sha256"])
            with pytest.raises(ValueError, match="different message"):
                await service.conversations.create(
                    dict(payload, media_names=["different.png"])
                )
            assert (
                await service.conversations.media(sent["chat_id"], attached["sha256"])
                == attached
            )
            await service.conversations.message(
                sent["chat_id"], {"message": "Use the original image"}
            )
            detail = await settle(service, sent["chat_id"])
            assert detail["turns"][-1]["status"] == "completed"
            # The unrelated chat runs concurrently; find this follow-up's call.
            followup = next(
                call
                for call in reversed(service.backend.calls)
                if any("Use the original image" in m.content for m in call)
            )
            assert any(message.media for message in followup)
        finally:
            await service.close()

    asyncio.run(check())


def test_queued_image_is_available_immediately_and_survives_cancel_and_reload(tmp_path):
    async def check():
        service = make_service(tmp_path, backend=ImageBackend(pause_first=True))
        attached = image()
        chat = (await service.conversations.create({"message": "First text request"}))[
            "chat_id"
        ]
        await service.backend.started.wait()
        queued = await service.conversations.message(
            chat, {"message": "Queued image", "media": [attached]}
        )
        assert queued["status"] == "queued"
        assert await service.conversations.media(chat, attached["sha256"]) == attached
        await service.close()
        service = make_service(tmp_path)
        try:
            detail = await service.conversations.detail(chat)
            assert detail["turns"][1]["status"] == "interrupted"
            assert detail["turns"][1]["media"][0]["sha256"] == attached["sha256"]
            await service.conversations.message(
                chat, {"message": "Continue with the queued image"}
            )
            detail = await settle(service, chat)
            assert detail["turns"][-1]["status"] == "completed"
            assert any(
                message.content == "Queued image" and message.media
                for message in service.backend.calls[-1]
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_unsupported_model_fails_visibly_before_run_launch_but_preserves_image(
    tmp_path,
):
    async def check():
        service = AppService(tmp_path, tmp_path / ".instinct", demo=True)
        try:
            attached = image()
            chat = (
                await service.conversations.create({"message": "", "media": [attached]})
            )["chat_id"]
            detail = await settle(service, chat)
            assert detail["turns"][0]["run_id"] is None
            assert detail["turns"][0]["status"] == "failed"
            assert "demo does not analyze images" in detail["turns"][0]["error"]
            assert (
                await service.conversations.media(chat, attached["sha256"]) == attached
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_real_worker_pool_checks_each_physical_replica_cached_media_advertisement(
    tmp_path,
):
    class CachedRemote(ImageBackend):
        supports_media = None

        def __init__(self, supported=True):
            super().__init__()
            self.identity = {"supports_media": supported}

    async def check():
        remote = CachedRemote()
        pool = WorkerPool({"qwen": remote})
        service = make_service(tmp_path, backend=pool)
        try:
            assert not hasattr(pool, "supports_media")
            service.identities = {"qwen": {"model": "fixture-qwen"}}
            chat = (
                await service.conversations.create(
                    {"message": "Inspect", "media": [image()]}
                )
            )["chat_id"]
            detail = await settle(service, chat)
            assert detail["turns"][0]["status"] == "completed", detail["turns"][0][
                "error"
            ]
            assert any(message.media for message in remote.calls[0])
            pool.backends["text"] = CachedRemote(False)
            with pytest.raises(ValueError, match="does not analyze images"):
                service.conversations._require_image_backend()
        finally:
            await service.close()

    asyncio.run(check())


def test_current_image_survives_projection_of_large_inherited_text(tmp_path):
    async def check():
        service = make_service(tmp_path)
        first, second = image(png_bytes(2, 2)), image(png_bytes(3, 2))
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Old request " + "x" * 6000, "media": [first]}
                )
            )["chat_id"]
            await settle(service, chat)
            latest = "Latest complete request " + "y" * 11000
            await service.conversations.message(
                chat, {"message": latest, "media": [second]}
            )
            detail = await settle(service, chat)
            assert detail["turns"][-1]["status"] == "completed", detail["turns"][-1][
                "error"
            ]
            assert detail["turns"][-1]["context_truncated"]
            assert any(
                message.content == latest and message.media == (second,)
                for message in service.backend.calls[-1]
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_inherited_image_count_is_bounded_and_omissions_are_visible(tmp_path):
    async def check():
        service = make_service(tmp_path)
        images = [image(png_bytes(width + 1, 2)) for width in range(9)]
        try:
            chat = (
                await service.conversations.create(
                    {"message": "First image", "media": images[:1]}
                )
            )["chat_id"]
            await settle(service, chat)
            for index, attached in enumerate(images[1:], start=2):
                await service.conversations.message(
                    chat, {"message": f"Image {index}", "media": [attached]}
                )
                detail = await settle(service, chat)
            assert detail["turns"][-1]["status"] == "completed"
            assert detail["turns"][-1]["context_truncated"]
            assert sum(len(message.media) for message in service.backend.calls[-1]) == 8
            assert (
                await service.conversations.media(chat, images[0]["sha256"])
                == images[0]
            )
        finally:
            await service.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "payload",
    [
        {"media": [image(b"not an image")]},
        {"media": [image(mime="image/jpeg")]},
        {"media": [image(b"video", mime="video/mp4", kind="video")]},
        {"media": [dict(image(), data="invalid base64")]},
        {"media": [dict(image(), sha256="0" * 64)]},
        {"media": [dict(image(), mime="image/svg+xml")]},
        {"media": [image()] * 9},
        {"media": [image()], "media_names": []},
        {"media": [image()], "media_names": [12]},
        {"media": [image()], "media_names": ["x" * 241]},
        {"media": [image()], "media_names": ["line\nbreak.png"]},
        {"media": [dict(image(), unknown="field")]},
    ],
)
def test_invalid_image_requests_create_no_turn_or_blob(tmp_path, payload):
    async def check():
        service = make_service(tmp_path)
        try:
            with pytest.raises((ValueError, TypeError)):
                await service.conversations.create(dict(payload, message="Inspect"))
            assert not (await service.conversations.overview())["chats"]
            assert (
                service.store.connection.execute(
                    "SELECT COUNT(*) FROM media_blobs"
                ).fetchone()[0]
                == 0
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_individual_and_combined_media_byte_limits_are_checked_before_persistence(
    tmp_path, monkeypatch
):
    async def check():
        service = make_service(tmp_path)
        try:
            monkeypatch.setattr(media_module, "MAX_MEDIA_BYTES", 100)
            with pytest.raises(ValueError, match="limit"):
                await service.conversations.create(
                    {"message": "Inspect", "media": [image(png_bytes() + b"x" * 101)]}
                )
            monkeypatch.setattr(media_module, "MAX_TOTAL_MEDIA_BYTES", 100)
            with pytest.raises(ValueError, match="combined"):
                await service.conversations.create(
                    {"message": "Inspect", "media": [image(), image()]}
                )
            assert (
                service.store.connection.execute(
                    "SELECT COUNT(*) FROM media_blobs"
                ).fetchone()[0]
                == 0
            )
        finally:
            await service.close()

    asyncio.run(check())


def test_http_authorizes_before_reading_upload_body_and_only_image_send_routes_get_large_limit():
    handler = object.__new__(AppHandler)
    calls = []
    handler.path = "/api/chats"
    handler._origin_ok = lambda: True
    handler._authenticated = lambda: False
    handler._json = lambda status, body: calls.append((status, body))
    handler._body = lambda **kwargs: pytest.fail("unauthorized body was read")
    handler.do_POST()
    assert calls[0][0] == 401
    assert handler.close_connection is True
    handler._authenticated = lambda: True
    limits = []
    handler._body = lambda *, limit: limits.append(limit) or {"message": "Inspect"}
    handler._dispatch = lambda *args: None
    for path in (
        "/api/chats",
        "/api/chats/abc/message",
        "/api/chats/abc/cancel",
        "/api/runs",
        "/api/workspace/files",
    ):
        handler.path = path
        handler.do_POST()
    assert limits == [
        MAX_MEDIA_REQUEST_BYTES,
        MAX_MEDIA_REQUEST_BYTES,
        MAX_BODY,
        MAX_BODY,
        MAX_BODY,
    ]


def test_children_inherit_newest_distinct_parent_images_with_count_and_byte_bounds(
    monkeypatch,
):
    store = AgentStore()
    try:
        images = [image(png_bytes(width + 1, 2)) for width in range(9)]
        store.create_run(
            "images",
            "First request",
            asdict(RuntimeConfig()),
            "system",
            media=images[:1],
        )
        for attached in images[1:]:
            store.add_message("images", "root", "user", "Next image", media=[attached])
        store.add_message(
            "images", "root", "user", "Same latest image again", media=images[-1:]
        )
        child = store.spawn(
            "images",
            "root",
            "spawn-child",
            "Inspect latest images",
            "Image inspector",
            [],
            "system",
        )
        inherited = store.messages("images", child)[1].media
        assert len(inherited) == 8
        assert inherited[-1] == images[-1]
        assert images[0] not in inherited
        assert len({item["sha256"] for item in inherited}) == 8
        monkeypatch.setattr(media_module, "MAX_TOTAL_MEDIA_BYTES", 160)
        bounded = store.spawn(
            "images",
            "root",
            "spawn-bounded",
            "Inspect latest images",
            "Image inspector",
            [],
            "system",
        )
        inherited = store.messages("images", bounded)[1].media
        assert inherited[-1] == images[-1]
        assert sum(len(base64.b64decode(item["data"])) for item in inherited) <= 160
        assert any(
            event["kind"] == "media_inherited"
            and event["payload"]["older_items_omitted"]
            for event in store.events("images", agent_id=bounded)
        )
    finally:
        store.close()


def test_http_large_image_send_authenticated_preview_and_body_bounds(tmp_path):
    async def check():
        service = make_service(tmp_path)
        server = AppServer(("127.0.0.1", 0), service, asyncio.get_running_loop())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None, *, auth=True, declared_size=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            headers = {"Origin": server.origin, "Content-Type": "application/json"}
            if auth:
                headers["Cookie"] = "instinct_session=" + server.session_token
            if declared_size is not None:
                headers["Content-Length"] = str(declared_size)
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            result = response.status, response.read(), dict(response.getheaders())
            connection.close()
            return result

        async def call(*args, **kwargs):
            return await asyncio.to_thread(request, *args, **kwargs)

        try:
            assert (
                await call(
                    "POST",
                    "/api/chats",
                    auth=False,
                    declared_size=MAX_MEDIA_REQUEST_BYTES,
                )
            )[0] == 401
            assert (
                await call(
                    "POST", "/api/chats", declared_size=MAX_MEDIA_REQUEST_BYTES + 1
                )
            )[0] == 400
            assert (
                await call("POST", "/api/chats/any/cancel", declared_size=MAX_BODY + 1)
            )[0] == 400
            attached = image(png_bytes(192, 192, noisy=True))
            body = json.dumps({"message": "Read this image", "media": [attached]})
            assert len(body) > MAX_BODY
            status, returned, headers = await call("POST", "/api/chats", body)
            assert status == 200, returned
            chat_id = json.loads(returned)["chat_id"]
            assert "blob:" in headers["Content-Security-Policy"]
            detail = await settle(service, chat_id)
            assert detail["turns"][0]["status"] == "completed", detail["turns"][0][
                "error"
            ]
            url = detail["turns"][0]["media"][0]["url"]
            assert (await call("GET", url, auth=False))[0] == 401
            status, returned, _ = await call("GET", url)
            assert status == 200 and json.loads(returned) == attached
            assert (await call("POST", "/api/runs", body))[0] == 400
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await service.close()

    asyncio.run(check())
