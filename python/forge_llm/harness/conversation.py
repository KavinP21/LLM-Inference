"""Durable conversations over the existing scoped agent scheduler.

The HTTP send boundary persists the user's message before model startup. Each
reply remains a separately auditable run, while ordinary follow-ups retain a
bounded sequence of real user/assistant messages from the same conversation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING

from ..agents.protocol import ChatMessage, conservative_input_tokens
from ..agents.runtime import AgentRuntime, RuntimeConfig
from ..agents.store import TERMINAL

if TYPE_CHECKING:
    from .service import StudioService

SETTLED = TERMINAL | {"completed_with_failures", "interrupted"}


class ConversationRuntime(AgentRuntime):
    """Keep chat behavior and the current request across context projection."""

    def _system_prompt(self, config, *, task="", **kwargs):
        return super()._system_prompt(config, task=task, **kwargs) + (
            "\nConversation behavior: Answer the latest user request naturally and "
            "concisely. Put the user-facing answer in the finish result as plain "
            "text or Markdown; use JSON in that result only when the user asks "
            "for JSON. Greetings, explanations, and simple questions can finish "
            "directly. Use tools and delegation only when they help this request. "
            "Earlier user and assistant messages are conversation context; "
            "earlier assistant claims are not verified file or execution evidence. "
            "A history omission marker means some prior context is unavailable; "
            "do not invent what was omitted. The latest assigned user request "
            "is the goal of this run; prior prompts are context rather than new "
            "assignments. Continue this request even when older chat messages "
            "are later projected out of the model context."
        )

    def _model_context(self, run_id, agent_id, config, *, schema_input_bound=0):
        projected = super()._model_context(
            run_id, agent_id, config, schema_input_bound=schema_input_bound
        )
        task = self.store.agent(run_id, agent_id).task
        has_task = any(
            message.role == "user"
            and (message.content == task or message.content.startswith(task + "\n\n"))
            for message in projected
        )
        limit = (
            config.max_context_tokens - config.max_output_tokens - schema_input_bound
        )
        if has_task and conservative_input_tokens(projected) <= limit:
            return projected
        # The general runtime pins the first user message. A conversation has
        # genuine inherited roles before the new request, so pin this run's
        # current task if projection dropped it. Keep the final action menu.
        marker = ChatMessage(
            "user",
            "[Earlier conversation context was omitted to retain the complete current request. The saved Studio transcript still contains those messages.]",
        )
        tail = []
        for message in projected[1:-1]:
            if message.role == "user" and message.content == task:
                continue
            if message.role == "user" and message.content.startswith(task + "\n\n"):
                # The general runtime appends its current legal menu to the
                # last user message; retain that menu without duplicating task.
                message = ChatMessage("user", message.content[len(task) + 2 :])
            tail.append(message)
        menu = (
            self._native_state(run_id, agent_id, config)
            if self.native_mode
            else self._legal_menu(run_id, agent_id, config)
        )
        projected = [
            projected[0],
            ChatMessage("user", task),
            marker,
            *tail,
            ChatMessage("user", menu),
        ]
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
        history_bytes: int = 8_000,
    ):
        self.service = service
        self.startup_timeout_seconds = startup_timeout_seconds
        self.history_bytes = history_bytes
        self.jobs: dict[str, asyncio.Task] = {}
        self.watchers: dict[str, asyncio.Task] = {}
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
        # There is no live controller after a process restart. Preserve pending
        # user messages and make the interruption visible instead of re-sending.
        self.db.execute(
            "UPDATE studio_chat_turns SET status='interrupted',error=? "
            "WHERE run_id IS NULL AND status IN ('loading','queued')",
            ("Interrupted while starting the model. Send a message to continue.",),
        )
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
        }:
            raise ValueError("unsupported conversation fields")
        message = self._text(payload.get("message"))
        policy = payload.get("policy", "review")
        if policy not in {"review", "read_only"}:
            raise ValueError("policy must be review or read_only")
        request_id, fingerprint, prior = self._request(payload, kind="create")
        if prior:
            return self._response(prior["chat_id"], prior["turn_id"], duplicate=True)
        self._can_start()
        configuration = asdict(self.service._configuration({}))
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
            self._insert_turn(chat_id, turn_id, message, now)
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

    def _insert_turn(self, chat_id, turn_id, message, now, status="loading"):
        self.db.execute(
            "INSERT INTO studio_chat_turns "
            "(turn_id,chat_id,user,status,created) VALUES(?,?,?,?,?)",
            (turn_id, chat_id, message, status, now),
        )
        self.db.execute(
            "UPDATE studio_chats SET updated=? WHERE chat_id=?", (now, chat_id)
        )

    async def message(self, chat_id, payload):
        if not isinstance(payload, dict) or payload.keys() - {"message", "request_id"}:
            raise ValueError("unsupported message fields")
        message = self._text(payload.get("message"))
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
            self._insert_turn(
                chat_id,
                turn_id,
                message,
                time.time(),
                "queued" if active else "loading",
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

    def _user_messages(self, turn):
        messages = [
            {"role": "user", "content": turn["user"], "created": turn["created"]}
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

    async def _history(self, chat_id, turn_id):
        rows = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? AND rowid < "
            "(SELECT rowid FROM studio_chat_turns WHERE turn_id=?) "
            "ORDER BY created DESC,rowid DESC LIMIT 100",
            (chat_id, turn_id),
        ).fetchall()
        total = self.db.execute(
            "SELECT COUNT(*) FROM studio_chat_turns WHERE chat_id=? AND rowid < "
            "(SELECT rowid FROM studio_chat_turns WHERE turn_id=?)",
            (chat_id, turn_id),
        ).fetchone()[0]
        budget = self.history_bytes
        selected = []
        truncated = len(rows) < total
        for row in rows:
            group = [
                (item["role"], item["content"]) for item in self._user_messages(row)
            ]
            if row["run_id"]:
                root = self.service.store.agent(row["run_id"], "root")
                if root.result:
                    group.append(("assistant", root.result))
                elif root.error:
                    group.append(
                        (
                            "assistant",
                            "[Previous reply did not complete: " + root.error + "]",
                        )
                    )
            elif row["error"]:
                group.append(
                    (
                        "assistant",
                        "[Previous reply did not complete: " + row["error"] + "]",
                    )
                )
            size = sum(len(content.encode("utf-8")) + 32 for _, content in group)
            if size > budget:
                truncated = True
                # Retain an explicitly marked excerpt of the most recent turn
                # rather than silently losing it when one answer is very long.
                if not selected and budget >= 256:
                    excerpt = []
                    remaining = budget
                    for role, content in group:
                        allocation = max(
                            0, remaining // (len(group) - len(excerpt)) - 80
                        )
                        text = content.encode("utf-8")[:allocation].decode(
                            "utf-8", "ignore"
                        )
                        excerpt.append(
                            (
                                role,
                                text
                                + "\n[Excerpt; full message remains in the conversation transcript.]",
                            )
                        )
                        remaining -= len(excerpt[-1][1].encode("utf-8")) + 32
                    selected.append(excerpt)
                break
            selected.append(group)
            budget -= size
        history = [item for group in reversed(selected) for item in group]
        if truncated:
            history.insert(
                0,
                (
                    "user",
                    "[Conversation history is bounded: older turns or long messages were omitted or excerpted. The full transcript remains saved in Studio; do not assume omitted facts.]",
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
                history, truncated = await self._history(chat_id, turn_id)
                result = await self.service.create_run(
                    {"task": turn["user"], "policy": chat["policy"]},
                    _history=history,
                    _configuration=RuntimeConfig(**json.loads(chat["configuration"])),
                    _conversation=True,
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
        state = await self.service.state()
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
        state = await self.service.state()
        turns, transcript = [], []
        rows = self.db.execute(
            "SELECT * FROM studio_chat_turns WHERE chat_id=? ORDER BY created,rowid",
            (chat_id,),
        ).fetchall()
        for row in rows:
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
                "context_truncated": bool(row["context_truncated"]),
            }
            if row["run_id"]:
                detail = await self.service.detail(row["run_id"])
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
        raise ValueError("unknown conversation API route")

    async def close(self):
        tasks = [*self.jobs.values(), *self.watchers.values()]
        for job in tasks:
            if not job.done():
                job.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
