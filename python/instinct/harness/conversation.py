"""Durable conversations over the existing scoped agent scheduler.

The HTTP send boundary persists the user's message before model startup. Each
reply remains a separately auditable run, while ordinary follow-ups retain a
bounded sequence of real user/assistant messages from the same conversation.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import subprocess
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING

from ..agents.media import MAX_MEDIA_ITEMS, MAX_TOTAL_MEDIA_BYTES, validate_media
from ..agents.protocol import ChatMessage, conservative_input_tokens
from ..agents.runtime import AgentRuntime, RuntimeConfig
from ..agents.store import TERMINAL
from ..chat_formats import EFFORT_BUDGETS
from .checkpoints import CheckpointError
from .policy import POLICIES

if TYPE_CHECKING:
    from .service import StudioService

# Per-chat choices the composer can change between messages.
SETTINGS = {"model", "effort"}

SETTLED = TERMINAL | {"completed_with_failures", "interrupted"}

# A chat view gets every event of its latest turns, but only what the transcript
# shows for older ones, and long text is clipped. A long chat then stays cheap
# to refresh while a reply streams.
RECENT_TURNS = 8
OLD_TURN_EVENTS = {"tool_result", "generation", "operator_message"}


def _clip(value, limit):
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"\n… [{len(value) - limit:,} more characters]"
    if isinstance(value, dict):
        return {key: _clip(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item, limit) for item in value]
    return value


def _slim(events, limit):
    return [dict(event, payload=_clip(event["payload"], limit)) for event in events]


def _old_turn_events(events):
    responses = [
        event
        for event in events
        if event["kind"] == "model_response" and event["agent_id"] == "root"
    ]
    kept = [event for event in events if event["kind"] in OLD_TURN_EVENTS]
    # The last root response carries the reasoning shown as "Thought".
    return _slim(sorted(kept + responses[-1:], key=lambda event: event["id"]), 600)


IMAGE_MIMES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def _image_input(payload):
    """Validate the established bounded transport plus image MIME signatures."""
    media = validate_media(payload.get("media", ()))
    for item in media:
        if item["type"] != "image" or item["mime"] not in IMAGE_MIMES:
            raise ValueError(
                "Studio attachments support PNG, JPEG, WebP, and BMP images"
            )
        data = base64.b64decode(item["data"])
        signatures = {
            "image/png": len(data) >= 24
            and data.startswith(b"\x89PNG\r\n\x1a\n")
            and data[12:16] == b"IHDR",
            "image/jpeg": len(data) >= 4 and data.startswith(b"\xff\xd8\xff"),
            "image/webp": len(data) >= 12
            and data[:4] == b"RIFF"
            and data[8:12] == b"WEBP",
            "image/bmp": len(data) >= 26 and data.startswith(b"BM"),
        }
        if not signatures[item["mime"]]:
            raise ValueError("image bytes do not match the declared MIME type")
    names = payload.get("media_names")
    if names is None:
        names = [
            f"image-{index + 1}{IMAGE_MIMES[item['mime']]}"
            for index, item in enumerate(media)
        ]
    if (
        not isinstance(names, list)
        or len(names) != len(media)
        or any(
            not isinstance(name, str)
            or not 1 <= len(name) <= 240
            or any(ord(character) < 32 for character in name)
            for name in names
        )
    ):
        raise ValueError("media_names must contain one short filename for each image")
    message = payload.get("message", "")
    if media and isinstance(message, str) and not message.strip():
        message = "Describe the attached image."
    return media, names, ConversationController._text(message)


def _media_size(item):
    return len(item["data"]) // 4 * 3 - (
        len(item["data"]) - len(item["data"].rstrip("="))
    )


class ConversationRuntime(AgentRuntime):
    """Keep chat behavior and the current request across context projection."""

    def _system_prompt(self, config, *, task="", **kwargs):
        return super()._system_prompt(config, task=task, **kwargs) + (
            "\nConversation behavior: Answer the latest user request naturally and "
            "concisely, replying in plain text or Markdown. Greetings, explanations "
            "and simple questions need no tools. Use tools and sub-agents only when "
            "they help this request. "
            "Earlier user and assistant messages are conversation context; "
            "earlier assistant claims are not verified file or execution evidence. "
            "A history omission marker means some prior context is unavailable; "
            "do not invent what was omitted. The latest assigned user request "
            "is the goal of this run; prior prompts are context rather than new "
            "assignments. Continue this request even when older chat messages "
            "are later projected out of the model context."
        )

    def _anchor_indices(self, run_id, agent_id, history):
        agent = self.store.agent(run_id, agent_id)
        if agent.parent_id is None:
            # A chat's root agent starts after earlier turns; compaction should
            # keep this run's request, not the oldest message of the chat.
            for index in range(len(history) - 1, 0, -1):
                message = history[index]
                if message.role == "user" and message.content == agent.task:
                    return [0, index]
        return super()._anchor_indices(run_id, agent_id, history)

    def _model_context(self, run_id, agent_id, config, *, schema_input_bound=0):
        projected = super()._model_context(
            run_id, agent_id, config, schema_input_bound=schema_input_bound
        )
        agent = self.store.agent(run_id, agent_id)
        if agent.parent_id is not None:
            # Sub-agents have no chat history; the runtime already pins their
            # assigned scope.
            return projected
        task = agent.task
        current_media = next(
            (
                message.media
                for message in reversed(self.store.messages(run_id, agent_id))
                if message.role == "user" and message.content == task
            ),
            (),
        )
        has_task = any(
            message.role == "user"
            and (message.content == task or message.content.startswith(task + "\n\n"))
            for message in projected
        )
        limit = (
            config.max_context_tokens - config.max_output_tokens - schema_input_bound
        )
        expected_media = {item["sha256"] for item in current_media}
        projected_media = {
            item["sha256"] for message in projected for item in message.media
        }
        if (
            has_task
            and expected_media <= projected_media
            and self._input_bound(run_id, agent_id, projected) <= limit
        ):
            return projected
        # The general runtime pins the first user message. A conversation has
        # genuine inherited roles before the new request, so pin this run's
        # current task if projection dropped it. Keep the final action menu.
        marker = ChatMessage(
            "user",
            "[Earlier conversation context was omitted to retain the complete current request. The saved Studio transcript still contains those messages.]",
        )
        tail = []
        # DSL agents end with the legal menu, which is re-added below.
        for message in projected[1:] if self.native_mode else projected[1:-1]:
            if message.role == "user" and message.content == task:
                continue
            if message.role == "user" and message.content.startswith(task + "\n\n"):
                # The general runtime appends its current legal menu to the
                # last user message; retain that menu without duplicating task.
                message = ChatMessage(
                    "user", message.content[len(task) + 2 :], message.media
                )
            tail.append(message)
        projected = [
            projected[0],
            ChatMessage("user", task, current_media),
            marker,
            *tail,
        ]
        if not self.native_mode:
            projected.append(
                ChatMessage("user", self._legal_menu(run_id, agent_id, config))
            )
        while len(projected) > 4 and conservative_input_tokens(projected) > limit:
            del projected[3]
        self.store.event(
            run_id,
            agent_id,
            "conversation_context_projected",
            {"current_request_preserved": True, "messages": len(projected)},
        )
        return projected


class ConversationController:
    def __init__(
        self,
        service: StudioService,
        *,
        startup_timeout_seconds: float = 180,
        history_bytes: int = 48_000,
    ):
        self.service = service
        self.startup_timeout_seconds = startup_timeout_seconds
        self.history_bytes = history_bytes
        self.jobs: dict[str, asyncio.Task] = {}
        self.watchers: dict[str, asyncio.Task] = {}
        # Run details for finished turns never change; keep them.
        self._details: dict[str, dict] = {}
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS studio_chats (
                chat_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                policy TEXT NOT NULL, configuration TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL,
                legacy INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS studio_chat_turns (
                turn_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES studio_chats(chat_id),
                run_id TEXT UNIQUE REFERENCES runs(run_id),
                user TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                created REAL NOT NULL, context_truncated INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS studio_chat_turn_order
                ON studio_chat_turns(chat_id, created);
            CREATE TABLE IF NOT EXISTS studio_chat_requests (
                request_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES studio_chats(chat_id),
                turn_id TEXT NOT NULL REFERENCES studio_chat_turns(turn_id),
                kind TEXT NOT NULL, fingerprint TEXT NOT NULL
            );
        """)
        columns = {
            row["name"]
            for row in self.db.execute("PRAGMA table_info(studio_chat_turns)")
        }
        for name in ("media", "media_names"):
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE studio_chat_turns ADD COLUMN {name} TEXT NOT NULL DEFAULT '[]'"
                )
        if "checkpoint" not in columns:
            # The workspace commit taken before the turn ran; see checkpoints.py.
            self.db.execute("ALTER TABLE studio_chat_turns ADD COLUMN checkpoint TEXT")
        # There is no live controller after a process restart. Preserve pending
        # user messages and make the interruption visible instead of re-sending.
        self.db.execute(
            "UPDATE studio_chat_turns SET status='interrupted',error=? "
            "WHERE run_id IS NULL AND status IN ('loading','queued')",
            ("Interrupted while starting the model. Send a message to continue.",),
        )
        self._manifests_seen = -1
        self._import_legacy()

    @property
    def db(self):
        return self.service.store.connection

    @staticmethod
    def _text(value, label="message", limit=12_000):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{label} must contain 1 to {limit} characters")
        return value.strip()

    def _chat(self, chat_id):
        row = self.db.execute(
            "SELECT * FROM studio_chats WHERE chat_id=?", (chat_id,)
        ).fetchone()
        if row is None:
            raise ValueError("conversation does not exist")
        return row

    def _latest(self, chat_id):
        return self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? "
            "ORDER BY created DESC,rowid DESC LIMIT 1",
            (chat_id,),
        ).fetchone()

    def _status(self, turn):
        if turn is None:
            return "completed"
        if turn["run_id"]:
            run_id = turn["run_id"]
            run = self.service.store.run(run_id)
            job = self.service.jobs.get(run_id)
            if run["status"] not in TERMINAL | {"completed_with_failures"} and (
                job is None or job.done()
            ):
                return "interrupted"
            return run["status"]
        return turn["status"]

    def _active_turn(self, chat_id):
        for turn in self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? ORDER BY created,rowid",
            (chat_id,),
        ):
            if self._status(turn) in {"loading", "running"}:
                return turn
        return None

    def _chat_status(self, chat_id):
        active = self._active_turn(chat_id)
        return self._status(active or self._latest(chat_id))

    def _response(self, chat_id, turn_id, **extra):
        turn = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        return {
            "chat_id": chat_id,
            "turn_id": turn_id,
            "run_id": turn["run_id"],
            "status": self._status(turn),
            **extra,
        }

    def _request(self, payload, *, kind, chat_id=None):
        request_id = payload.get("request_id")
        if request_id is not None and (
            not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
        ):
            raise ValueError("request_id must contain 1 to 128 characters")
        canonical = {k: v for k, v in payload.items() if k != "request_id"}
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        row = (
            self.db.execute(
                "SELECT * FROM studio_chat_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if request_id
            else None
        )
        if row and (
            row["fingerprint"] != fingerprint
            or row["kind"] != kind
            or (chat_id is not None and row["chat_id"] != chat_id)
        ):
            raise ValueError("request_id was already used for a different message")
        return request_id, fingerprint, row

    def _remember(self, request_id, fingerprint, kind, chat_id, turn_id):
        if request_id:
            self.db.execute(
                "INSERT INTO studio_chat_requests VALUES(?,?,?,?,?)",
                (request_id, chat_id, turn_id, kind, fingerprint),
            )

    def _import_legacy(self):
        """Group existing run lineages without changing their original journals."""
        # Every run has a manifest, so this only rescans when a run was added.
        count = self.db.execute("SELECT COUNT(*) FROM studio_manifests").fetchone()[0]
        if count == self._manifests_seen:
            return
        self._manifests_seen = count
        rows = self.db.execute(
            "SELECT m.run_id,m.manifest,r.created,r.config FROM studio_manifests m "
            "JOIN runs r ON m.run_id=r.run_id ORDER BY r.created"
        ).fetchall()
        known = {
            row["run_id"]: row["chat_id"]
            for row in self.db.execute(
                "SELECT run_id,chat_id FROM studio_chat_turns WHERE run_id IS NOT NULL"
            )
        }
        for row in rows:
            run_id = row["run_id"]
            if run_id in known:
                continue
            manifest = json.loads(row["manifest"])
            chat_id = known.get(manifest.get("parent_id"), "legacy-" + run_id)
            with self.service.store.transaction():
                self.db.execute(
                    "INSERT OR IGNORE INTO studio_chats VALUES(?,?,?,?,?,?,1)",
                    (
                        chat_id,
                        manifest["brief"][:80],
                        manifest["policy"],
                        row["config"],
                        row["created"],
                        row["created"],
                    ),
                )
                self.db.execute(
                    "INSERT INTO studio_chat_turns "
                    "(turn_id,chat_id,run_id,user,status,created) VALUES(?,?,?,?,?,?)",
                    (
                        run_id,
                        chat_id,
                        run_id,
                        manifest["brief"],
                        "running",
                        row["created"],
                    ),
                )
                self.db.execute(
                    "UPDATE studio_chats SET updated=MAX(updated,?) WHERE chat_id=?",
                    (row["created"], chat_id),
                )
            known[run_id] = chat_id

    async def create(self, payload):
        if not isinstance(payload, dict) or payload.keys() - {
            "message",
            "policy",
            "request_id",
            "media",
            "media_names",
            "model",
            "effort",
        }:
            raise ValueError("unsupported conversation fields")
        media, names, message = _image_input(payload)
        policy = payload.get("policy", "review")
        if policy not in POLICIES:
            raise ValueError("policy must be review or read_only")
        request_id, fingerprint, prior = self._request(payload, kind="create")
        if prior:
            return self._response(prior["chat_id"], prior["turn_id"], duplicate=True)
        self._can_start()
        configuration = self._settings(asdict(self.service._configuration({})), payload)
        chat_id, turn_id, now = uuid.uuid4().hex, uuid.uuid4().hex, time.time()
        with self.service.store.transaction():
            self.db.execute(
                "INSERT INTO studio_chats VALUES(?,?,?,?,?,?,0)",
                (
                    chat_id,
                    " ".join(message.split())[:80],
                    policy,
                    json.dumps(configuration),
                    now,
                    now,
                ),
            )
            self._insert_turn(
                chat_id, turn_id, message, now, media=media, media_names=names
            )
            self._remember(request_id, fingerprint, "create", chat_id, turn_id)
        self._launch(chat_id, turn_id)
        return self._response(chat_id, turn_id)

    def _can_start(self):
        if self.service.closing:
            raise ValueError("Studio is shutting down")
        active = sum(not job.done() for job in self.jobs.values())
        legacy_active = 0
        for run_id, job in self.service.jobs.items():
            if job.done():
                continue
            turn = self.db.execute(
                "SELECT turn_id FROM studio_chat_turns WHERE run_id=?", (run_id,)
            ).fetchone()
            controller = self.jobs.get(turn["turn_id"]) if turn else None
            if controller is None or controller.done():
                legacy_active += 1
        if active + legacy_active >= 4:
            raise ValueError("Four conversations are active; wait or stop one")

    def _insert_turn(
        self,
        chat_id,
        turn_id,
        message,
        now,
        status="loading",
        *,
        media=(),
        media_names=(),
    ):
        self.db.execute(
            "INSERT INTO studio_chat_turns "
            "(turn_id,chat_id,user,status,created,media,media_names) VALUES(?,?,?,?,?,?,?)",
            (
                turn_id,
                chat_id,
                message,
                status,
                now,
                json.dumps(self.service.store._store_media(media)),
                json.dumps(media_names),
            ),
        )
        self.db.execute(
            "UPDATE studio_chats SET updated=? WHERE chat_id=?", (now, chat_id)
        )

    async def message(self, chat_id, payload):
        if not isinstance(payload, dict) or payload.keys() - {
            "message",
            "request_id",
            "media",
            "media_names",
            "model",
            "effort",
        }:
            raise ValueError("unsupported message fields")
        media, names, message = _image_input(payload)
        chat = self._chat(chat_id)
        request_id, fingerprint, prior = self._request(
            payload, kind="message", chat_id=chat_id
        )
        if prior:
            return self._response(chat_id, prior["turn_id"], duplicate=True)
        active = self._active_turn(chat_id)
        if active is None:
            self._can_start()
        elif (
            self.db.execute(
                "SELECT COUNT(*) FROM studio_chat_turns WHERE chat_id=? AND status='queued'",
                (chat_id,),
            ).fetchone()[0]
            >= 16
        ):
            raise ValueError(
                "Sixteen messages are queued; wait for a reply or stop the conversation"
            )
        turn_id = uuid.uuid4().hex
        with self.service.store.transaction():
            if chat["legacy"]:
                # Adopt current defaults for the first new conversational turn;
                # old failed budgets and journals remain unchanged.
                self.db.execute(
                    "UPDATE studio_chats SET legacy=0,configuration=? WHERE chat_id=?",
                    (json.dumps(asdict(self.service._configuration({}))), chat_id),
                )
            if SETTINGS & payload.keys():
                current = json.loads(self._chat(chat_id)["configuration"])
                self.db.execute(
                    "UPDATE studio_chats SET configuration=? WHERE chat_id=?",
                    (json.dumps(self._settings(current, payload)), chat_id),
                )
            self._insert_turn(
                chat_id,
                turn_id,
                message,
                time.time(),
                "queued" if active else "loading",
                media=media,
                media_names=names,
            )
            self._remember(request_id, fingerprint, "message", chat_id, turn_id)
        if active:
            # A queued turn cannot disappear when an in-flight generation
            # finishes. It has its own durable prompt and always runs next.
            if active["run_id"] and active["turn_id"] not in self.jobs:
                self._watch_legacy(chat_id, active["run_id"])
        else:
            self._launch(chat_id, turn_id)
        return self._response(chat_id, turn_id, queued=bool(active))

    def _settings(self, configuration: dict, payload: dict) -> dict:
        """Apply a chat's model and effort choice to its runtime configuration."""
        configuration = dict(configuration)
        if "model" in payload:
            choices = {model["id"] for model in self.service.models()}
            if payload["model"] not in choices:
                raise ValueError("unknown model")
            configuration["driver_group"] = payload["model"]
        if "effort" in payload:
            if payload["effort"] not in EFFORT_BUDGETS:
                raise ValueError(f"effort must be one of {sorted(EFFORT_BUDGETS)}")
            configuration["effort"] = payload["effort"]
        RuntimeConfig(**configuration)
        return configuration

    def _configure(self, chat_id, payload):
        if not isinstance(payload, dict) or not payload or set(payload) - SETTINGS:
            raise ValueError("settings accept model and effort")
        chat = self._chat(chat_id)
        configuration = self._settings(json.loads(chat["configuration"]), payload)
        self.db.execute(
            "UPDATE studio_chats SET configuration=? WHERE chat_id=?",
            (json.dumps(configuration), chat_id),
        )
        return {"chat_id": chat_id, **self._chat_settings(configuration)}

    def _chat_settings(self, configuration: dict) -> dict:
        models = self.service.models()
        default = models[0]["id"] if models else None
        return {
            "model": configuration.get("driver_group") or default,
            "effort": configuration.get("effort", "off"),
            "context_limit": configuration.get("max_context_tokens"),
        }

    async def rename(self, chat_id, payload):
        if not isinstance(payload, dict) or set(payload) != {"title"}:
            raise ValueError("rename accepts only title")
        title = " ".join(self._text(payload["title"], "title", 200).split())
        if not title:
            raise ValueError("title must not be empty")
        self._chat(chat_id)
        self.db.execute(
            "UPDATE studio_chats SET title=? WHERE chat_id=?", (title, chat_id)
        )
        return {"chat_id": chat_id, "title": title}

    async def rewind(self, chat_id, payload):
        """Drop a message and everything after it, optionally putting the
        workspace files back as they were before it ran. With ``message`` the
        edited text is sent in its place, keeping the original images."""
        if not isinstance(payload, dict) or set(payload) - {
            "turn_id",
            "restore_files",
            "message",
        }:
            raise ValueError("rewind accepts turn_id, restore_files and message")
        restore = payload.get("restore_files", False)
        if not isinstance(restore, bool):
            raise ValueError("restore_files must be true or false")
        self._chat(chat_id)
        rows = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? ORDER BY created,rowid",
            (chat_id,),
        ).fetchall()
        index = next(
            (
                i
                for i, row in enumerate(rows)
                if row["turn_id"] == payload.get("turn_id")
            ),
            None,
        )
        if index is None:
            raise ValueError("that message is not in this chat")
        target, dropped = rows[index], rows[index:]
        if restore and not target["checkpoint"]:
            raise ValueError("there is no file checkpoint from before that message")
        await self._stop_turns(dropped)
        restored = None
        if restore:
            restored = await asyncio.to_thread(
                self.service.checkpoints.restore, target["checkpoint"]
            )
        with self.service.store.transaction():
            for row in dropped:
                self.db.execute(
                    "DELETE FROM studio_chat_requests WHERE turn_id=?",
                    (row["turn_id"],),
                )
                self.db.execute(
                    "DELETE FROM studio_chat_turns WHERE turn_id=?", (row["turn_id"],)
                )
            self.db.execute(
                "UPDATE studio_chats SET updated=? WHERE chat_id=?",
                (time.time(), chat_id),
            )
        result = {
            "chat_id": chat_id,
            "message": target["user"],
            "removed": len(dropped),
            "restored": restored,
        }
        if "message" in payload:
            media = self._turn_media(target)
            sent = await self.message(
                chat_id,
                {
                    "message": payload["message"],
                    "media": [
                        {key: item[key] for key in ("type", "mime", "data")}
                        for item in media
                    ],
                    "media_names": json.loads(target["media_names"]),
                },
            )
            result["sent"] = sent
        return result

    async def _stop_turns(self, rows):
        # The run first, so it settles as cancelled rather than interrupted.
        for turn in rows:
            run = turn["run_id"] and self.service.jobs.get(turn["run_id"])
            if run and not run.done():
                await self.service.cancel(turn["run_id"])
                await asyncio.gather(run, return_exceptions=True)
            job = self.jobs.pop(turn["turn_id"], None)
            if job is not None and not job.done():
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
            self._details.pop(turn["run_id"], None)

    async def delete(self, chat_id, payload):
        if payload:
            raise ValueError("delete takes no fields")
        self._chat(chat_id)
        turns = self.db.execute(
            "SELECT turn_id, run_id FROM studio_chat_turns WHERE chat_id=?", (chat_id,)
        ).fetchall()
        await self._stop_turns(turns)
        watcher = self.watchers.pop(chat_id, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
        with self.service.store.transaction():
            self.db.execute(
                "DELETE FROM studio_chat_requests WHERE chat_id=?", (chat_id,)
            )
            self.db.execute("DELETE FROM studio_chat_turns WHERE chat_id=?", (chat_id,))
            self.db.execute("DELETE FROM studio_chats WHERE chat_id=?", (chat_id,))
        return {"chat_id": chat_id, "deleted": True}

    def _user_messages(self, turn):
        messages = [
            {
                "role": "user",
                "content": turn["user"],
                "created": turn["created"],
                "media": self._media_metadata(turn),
            }
        ]
        if turn["run_id"]:
            messages.extend(
                {
                    "role": "user",
                    "content": event["payload"]["message"],
                    "created": event["created"],
                }
                for event in self.service.store.events(
                    turn["run_id"], agent_id="root", kinds=("operator_message",)
                )
            )
        return messages

    def _turn_media(self, turn):
        return self.service.store._load_media(json.loads(turn["media"]))

    def _media_metadata(self, turn):
        refs, names = json.loads(turn["media"]), json.loads(turn["media_names"])
        result = []
        for index, item in enumerate(refs):
            row = self.db.execute(
                "SELECT length(data) AS encoded_length,substr(data,-2) AS tail FROM media_blobs WHERE sha256=?",
                (item["sha256"],),
            ).fetchone()
            if row is None:
                raise RuntimeError("saved image is unavailable")
            result.append(
                dict(
                    item,
                    name=names[index]
                    if index < len(names)
                    else f"image-{index + 1}{IMAGE_MIMES[item['mime']]}",
                    size=row["encoded_length"] // 4 * 3 - row["tail"].count("="),
                    url=f"/api/chats/{turn['chat_id']}/media/{item['sha256']}",
                )
            )
        return result

    async def media(self, chat_id, sha256):
        self._chat(chat_id)
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError("invalid image fingerprint")
        associated = any(
            item["sha256"] == sha256
            for row in self.db.execute(
                "SELECT media FROM studio_chat_turns WHERE chat_id=?", (chat_id,)
            )
            for item in json.loads(row["media"])
        )
        if not associated:
            raise ValueError("image is not part of this conversation")
        row = self.db.execute(
            "SELECT * FROM media_blobs WHERE sha256=?", (sha256,)
        ).fetchone()
        if row is None:
            raise RuntimeError("saved image is unavailable")
        return {
            "type": row["kind"],
            "mime": row["mime"],
            "sha256": sha256,
            "data": row["data"],
        }

    def _require_image_backend(self):
        def advertised(backend):
            replicas = getattr(backend, "backends", None)
            if isinstance(replicas, dict):
                return bool(replicas) and all(
                    advertised(replica) for replica in replicas.values()
                )
            # Resume manifests intentionally contain only stable identity
            # fields. RemoteWorkerBackend caches the complete health response
            # separately; LocalModelBackend exposes it synchronously.
            identity = getattr(backend, "identity", None)
            if isinstance(identity, dict) and "supports_media" in identity:
                return identity["supports_media"] is True
            if getattr(backend, "supports_media", False) is True:
                return True
            health = getattr(backend, "health", None)
            if callable(health) and not inspect.iscoroutinefunction(health):
                result = health()
                return isinstance(result, dict) and result.get("supports_media") is True
            return False

        supported = advertised(self.service.backend)
        if not supported:
            if self.service.demo:
                raise ValueError(
                    "This demo does not analyze images. Choose an image-capable model to describe attachments."
                )
            raise ValueError(
                "The selected model does not support images. Choose an image-capable model and send a message to retry."
            )
        limit = getattr(self.service.backend, "max_model_length", None)
        if type(limit) is not int or limit <= 0:
            raise ValueError(
                "Image inference requires a model with an advertised context limit"
            )

    async def _ready(self):
        if self.service.backend_status == "ready":
            return
        await self.service.start_backend()
        loader = getattr(self.service, "_loader", None)
        if loader is None:
            raise RuntimeError("The model could not start. Send a message to retry.")
        try:
            await asyncio.wait_for(
                asyncio.shield(loader), timeout=self.startup_timeout_seconds
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "Model startup timed out. Send a message to retry once the model is ready."
            ) from exc
        if self.service.backend_status != "ready":
            raise RuntimeError(
                self.service.backend_error
                or "The model failed to start. Send a message to retry."
            )

    def _turn_group(self, row):
        group = [ChatMessage("user", row["user"])]
        group.extend(
            ChatMessage(item["role"], item["content"])
            for item in self._user_messages(row)[1:]
        )
        if row["run_id"]:
            root = self.service.store.agent(row["run_id"], "root")
            if root.result:
                group.append(ChatMessage("assistant", root.result))
            elif root.error:
                group.append(
                    ChatMessage(
                        "assistant",
                        "[Previous reply did not complete: " + root.error + "]",
                    )
                )
        elif row["error"]:
            group.append(
                ChatMessage(
                    "assistant",
                    "[Previous reply did not complete: " + row["error"] + "]",
                )
            )
        return group

    def _history_start(self, sizes: list[int]) -> int:
        """Index of the oldest turn that still fits in the history budget.

        Dropping only as much as needed would move the start of the prompt on
        every message and void the model's prompt cache each time. Instead,
        replay the chat from its first turn: whenever the history overflows,
        drop old turns until it's under half the budget. The start then stays
        put for the next several messages.
        """
        start = total = 0
        for newest, size in enumerate(sizes):
            total += size
            if total > self.history_bytes:
                while start < newest and total > self.history_bytes // 2:
                    total -= sizes[start]
                    start += 1
        return start

    async def _history(self, chat_id, turn_id, *, current_media=()):
        rows = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? AND rowid < "
            "(SELECT rowid FROM studio_chat_turns WHERE turn_id=?) "
            "ORDER BY created,rowid",
            (chat_id, turn_id),
        ).fetchall()
        groups = [self._turn_group(row) for row in rows]
        sizes = [
            sum(len(item.content.encode("utf-8")) + 32 for item in group)
            for group in groups
        ]
        start = self._history_start(sizes)
        truncated = start > 0
        selected = groups[start:]
        if sum(sizes[start:]) > self.history_bytes:
            # The latest turn alone is over budget: keep a marked excerpt.
            truncated = True
            group, room = groups[-1], self.history_bytes
            selected = [
                [
                    ChatMessage(
                        item.role,
                        item.content.encode("utf-8")[: room // len(group) - 80].decode(
                            "utf-8", "ignore"
                        )
                        + "\n[Excerpt; full message remains in the conversation transcript.]",
                    )
                    for item in group
                ]
            ]
        # Re-attach images to their turns, newest first, within the media limits.
        media_seen = {item["sha256"] for item in current_media}
        media_count = len(current_media)
        media_bytes = sum(_media_size(item) for item in current_media)
        kept_rows = rows[len(rows) - len(selected) :] if selected else []
        for index in range(len(selected) - 1, -1, -1):
            attached = []
            for item in self._turn_media(kept_rows[index]):
                if item["sha256"] in media_seen:
                    continue
                size = _media_size(item)
                if (
                    media_count >= MAX_MEDIA_ITEMS
                    or media_bytes + size > MAX_TOTAL_MEDIA_BYTES
                ):
                    truncated = True
                    continue
                attached.append(item)
                media_seen.add(item["sha256"])
                media_count += 1
                media_bytes += size
            if attached:
                first = selected[index][0]
                selected[index] = [
                    ChatMessage(first.role, first.content, tuple(attached)),
                    *selected[index][1:],
                ]
        history = [item for group in selected for item in group]
        if truncated:
            history.insert(
                0,
                ChatMessage(
                    "user",
                    "[Conversation history is bounded: older turns, images, or long messages were omitted or excerpted. The full transcript remains saved in Studio; do not assume omitted facts.]",
                ),
            )
        return history, truncated

    def _launch(self, chat_id, turn_id):
        async def drive():
            try:
                await self._ready()
                turn = self.db.execute(
                    "SELECT * FROM studio_chat_turns WHERE turn_id=?", (turn_id,)
                ).fetchone()
                if turn["status"] != "loading":
                    return
                chat = self._chat(chat_id)
                if chat["policy"] != "read_only":
                    try:
                        checkpoint = await asyncio.to_thread(
                            self.service.checkpoints.snapshot, f"before turn {turn_id}"
                        )
                    except (CheckpointError, OSError, subprocess.SubprocessError):
                        checkpoint = None  # Rewind still works; files stay as they are.
                    self.db.execute(
                        "UPDATE studio_chat_turns SET checkpoint=? WHERE turn_id=?",
                        (checkpoint, turn_id),
                    )
                current_media = self._turn_media(turn)
                history, truncated = await self._history(
                    chat_id, turn_id, current_media=current_media
                )
                if current_media or any(message.media for message in history):
                    self._require_image_backend()
                result = await self.service.create_run(
                    {"task": turn["user"], "policy": chat["policy"]},
                    _history=history,
                    _configuration=RuntimeConfig(**json.loads(chat["configuration"])),
                    _conversation=True,
                    _media=current_media,
                )
                run_id = result["run_id"]
                self.db.execute(
                    "UPDATE studio_chat_turns SET run_id=?,status='running',context_truncated=? WHERE turn_id=?",
                    (run_id, int(truncated), turn_id),
                )
                self.service.store.event(
                    run_id,
                    "root",
                    "conversation_context",
                    {
                        "chat_id": chat_id,
                        "inherited_messages": len(history),
                        "history_truncated": truncated,
                        "history_bytes_limit": self.history_bytes,
                    },
                )
                await self.service.jobs[run_id]
                run = self.service.store.run(run_id)
                self.db.execute(
                    "UPDATE studio_chat_turns SET status=?,error=? WHERE turn_id=?",
                    (run["status"], run["error"], turn_id),
                )
            except asyncio.CancelledError:
                self.db.execute(
                    "UPDATE studio_chat_turns SET status='interrupted',error=? WHERE turn_id=? AND status<>'cancelled'",
                    ("The reply was interrupted. Send a message to continue.", turn_id),
                )
                raise
            except Exception as exc:
                self.db.execute(
                    "UPDATE studio_chat_turns SET status='failed',error=? WHERE turn_id=?",
                    (f"{type(exc).__name__}: {exc}", turn_id),
                )
            finally:
                self.db.execute(
                    "UPDATE studio_chats SET updated=? WHERE chat_id=?",
                    (time.time(), chat_id),
                )
                self._drain(chat_id)

        self.jobs[turn_id] = asyncio.create_task(drive())

    def _drain(self, chat_id):
        if self.service.closing or self._active_turn(chat_id):
            return
        turn = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? AND status='queued' "
            "ORDER BY created,rowid LIMIT 1",
            (chat_id,),
        ).fetchone()
        if turn:
            self.db.execute(
                "UPDATE studio_chat_turns SET status='loading' WHERE turn_id=?",
                (turn["turn_id"],),
            )
            self._launch(chat_id, turn["turn_id"])

    def _watch_legacy(self, chat_id, run_id):
        if chat_id in self.watchers and not self.watchers[chat_id].done():
            return

        async def watch():
            job = self.service.jobs.get(run_id)
            if job:
                await asyncio.shield(job)
            self._drain(chat_id)

        self.watchers[chat_id] = asyncio.create_task(watch())

    async def cancel(self, chat_id, payload):
        if payload:
            raise ValueError("cancel does not accept fields")
        self._chat(chat_id)
        turn = self._active_turn(chat_id)
        if turn is None:
            return {"chat_id": chat_id, "cancel_requested": False}
        if turn["run_id"]:
            await self.service.cancel(turn["run_id"])
        else:
            self.db.execute(
                "UPDATE studio_chat_turns SET status='cancelled',error=NULL WHERE turn_id=?",
                (turn["turn_id"],),
            )
            job = self.jobs.get(turn["turn_id"])
            if job and not job.done():
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
            # A task cancelled before its first event-loop turn never enters
            # drive()'s finally block, so explicitly release its queued turn.
            self._drain(chat_id)
        return {"chat_id": chat_id, "cancel_requested": True}

    async def overview(self):
        self._import_legacy()
        state = self.service.summary()
        chats = []
        for row in self.db.execute(
            "SELECT * FROM studio_chats ORDER BY updated DESC LIMIT 100"
        ):
            chats.append(
                {
                    "chat_id": row["chat_id"],
                    "title": row["title"],
                    "updated": row["updated"],
                    "status": self._chat_status(row["chat_id"]),
                }
            )
        return {key: state[key] for key in ("workspace", "mode", "backend")} | {
            "chats": chats
        }

    async def detail(self, chat_id):
        self._import_legacy()
        chat = self._chat(chat_id)
        state = self.service.summary()
        turns, transcript = [], []
        rows = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? ORDER BY created,rowid",
            (chat_id,),
        ).fetchall()
        for index, row in enumerate(rows):
            messages = self._user_messages(row)
            turn = {
                "turn_id": row["turn_id"],
                "run_id": row["run_id"],
                "user": row["user"],
                "output": None,
                "status": self._status(row),
                "error": row["error"],
                "created": row["created"],
                "active": self._status(row) in {"loading", "running"},
                "events": [],
                "agents": [],
                "approvals": [],
                "artifacts": [],
                "messages": messages,
                "media": self._media_metadata(row),
                "context_truncated": bool(row["context_truncated"]),
                "checkpoint": bool(row["checkpoint"]),
            }
            if row["run_id"]:
                detail = self._details.get(row["run_id"])
                if detail is None:
                    detail = await self.service.detail(row["run_id"])
                    if not detail["active"] and detail["run"]["status"] in SETTLED:
                        self._details[row["run_id"]] = detail
                turn.update(
                    {
                        key: detail[key]
                        for key in (
                            "output",
                            "error",
                            "active",
                            "events",
                            "agents",
                            "approvals",
                            "artifacts",
                            "usage",
                        )
                    }
                )
                turn["context_truncated"] |= any(
                    event["kind"]
                    in {"context_projected", "conversation_context_projected"}
                    for event in detail["events"]
                )
                if turn["active"] or index >= len(rows) - RECENT_TURNS:
                    turn["events"] = _slim(detail["events"], 4000)
                else:
                    if "old_events" not in detail:
                        detail["old_events"] = _old_turn_events(detail["events"])
                    turn["events"] = detail["old_events"]
            turns.append(turn)
            transcript.extend(
                dict(message, turn_id=row["turn_id"]) for message in messages
            )
            if turn["output"]:
                transcript.append(
                    {
                        "role": "assistant",
                        "content": turn["output"],
                        "turn_id": row["turn_id"],
                        "created": row["created"],
                    }
                )
        return {
            "chat": {
                "chat_id": chat_id,
                "title": chat["title"],
                "status": self._chat_status(chat_id),
                "policy": chat["policy"],
                **self._chat_settings(json.loads(chat["configuration"])),
            },
            "turns": turns,
            "transcript": transcript,
            **{key: state[key] for key in ("backend", "mode", "workspace")},
        }

    async def dispatch(self, method, parts, payload):
        if len(parts) == 2:
            if method == "GET":
                return await self.overview()
            if method == "POST":
                return await self.create(payload)
        if len(parts) >= 3:
            chat_id = parts[2]
            if method == "GET" and len(parts) == 3:
                return await self.detail(chat_id)
            if method == "POST" and len(parts) == 4:
                if parts[3] == "message":
                    return await self.message(chat_id, payload)
                if parts[3] == "cancel":
                    return await self.cancel(chat_id, payload)
                if parts[3] == "settings":
                    return self._configure(chat_id, payload)
                if parts[3] == "rename":
                    return await self.rename(chat_id, payload)
                if parts[3] == "delete":
                    return await self.delete(chat_id, payload)
                if parts[3] == "rewind":
                    return await self.rewind(chat_id, payload)
            if method == "GET" and len(parts) == 5 and parts[3] == "media":
                return await self.media(chat_id, parts[4])
        raise ValueError("unknown conversation API route")

    async def close(self):
        tasks = [*self.jobs.values(), *self.watchers.values()]
        for job in tasks:
            if not job.done():
                job.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
