"""Guests: people who use this Mac's models from their own machines.

A guest never reaches the owner's app, folders, chats, terminal or ``/v1``.
They get a separate listener (see ``gateway.py``) that offers two things:

* **Agent runs for the browser app.** The agent loop runs here, against the
  shared model workers, with a journal kept only in memory. Every file tool
  the agents call is sent back to the guest's browser, which carries it out
  on a folder the guest picked on their own machine. When a run ends, nothing
  of it stays here.
* **Model calls for the guest's own Instinct app.** The worker job protocol,
  so a guest's app runs its whole harness (projects, terminal, sandboxed
  commands) on their machine and only borrows the models.

Access is by invite: a random token shown once, stored here only as a hash,
with an expiry, a revocation switch and quotas (requests at once, tokens a
day, context and reply length). Guests can use the main and sub-agent
models; any other worker (such as the abliterated Gemma) stays the owner's.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..agents.backends.config import CapacityError, RequestValidationError
from ..agents.protocol import ChatMessage
from ..agents.runtime import RuntimeConfig
from ..agents.store import AgentStore
from ..agents.tools import WorkspaceTools
from .conversation import ConversationRuntime

PREFIX = "inv_"
DAY = 86_400

# What an invite allows unless the owner sets otherwise.
DEFAULT_LIMITS = {
    "concurrent": 2,  # model calls at once
    "daily_tokens": 2_000_000,  # prompt plus reply tokens, rolling per day
    "context_tokens": 65_536,
    "output_tokens": 8_192,
}
LIMIT_RANGES = {
    "concurrent": (1, 16),
    "daily_tokens": (10_000, 1_000_000_000),
    "context_tokens": (4_096, 262_144),
    "output_tokens": (256, 32_768),
}

# Tools the browser app can carry out on the guest's folder. They're the
# owner's workspace tools, by the same names and schemas, so the models
# behave the same; the browser implements them (see web/guest.js).
RELAYED_TOOLS = (
    "list_files",
    "read_file",
    "search",
    "write_file",
    "replace_text",
    "edit_lines",
    "make_directory",
    "move_path",
    "delete_path",
)
READ_ONLY_TOOLS = frozenset({"list_files", "read_file", "search"})
TOOL_TIMEOUT = 300.0
MAX_TOOL_RESULT = 256_000


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def validate_limits(raw: Any) -> dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    if raw is None:
        return limits
    if not isinstance(raw, dict) or raw.keys() - LIMIT_RANGES.keys():
        raise ValueError(f"limits are {sorted(LIMIT_RANGES)}")
    for name, value in raw.items():
        low, high = LIMIT_RANGES[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
        limits[name] = value
    if limits["output_tokens"] >= limits["context_tokens"]:
        raise ValueError("output_tokens must be below context_tokens")
    return limits


class Invites:
    """Invite records in one 0600 file. Only token hashes are stored."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._invites: list[dict[str, Any]] = []
        if self.path.is_file():
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, list):
                self._invites = [
                    item
                    for item in loaded
                    if isinstance(item, dict)
                    and {"id", "name", "sha256"} <= item.keys()
                ]
        self._dirty_since: float | None = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(self._invites, stream, indent=2)
        os.replace(temporary, self.path)
        self._dirty_since = None

    @staticmethod
    def _public(item: dict[str, Any]) -> dict[str, Any]:
        usage = item.get("usage", {})
        return {
            "id": item["id"],
            "name": item["name"],
            "created": item["created"],
            "expires": item.get("expires"),
            "revoked": bool(item.get("revoked")),
            "active": Invites._valid(item),
            "limits": item["limits"],
            "tokens_today": usage.get("tokens", 0) if usage.get("day") == _day() else 0,
            "last_used": item.get("last_used"),
        }

    @staticmethod
    def _valid(item: dict[str, Any]) -> bool:
        expires = item.get("expires")
        return not item.get("revoked") and (expires is None or time.time() < expires)

    def create(
        self, name: Any, *, hours: Any = 24, limits: Any = None
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
            raise ValueError("name the invite in 1 to 60 characters")
        if hours is not None and (
            isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not 0 < hours <= 24 * 90
        ):
            raise ValueError("an invite lasts from a few minutes to 90 days")
        token = PREFIX + secrets.token_urlsafe(32)
        now = time.time()
        item = {
            "id": uuid.uuid4().hex[:12],
            "name": name.strip(),
            "sha256": _hash(token),
            "created": now,
            "expires": None if hours is None else now + float(hours) * 3600,
            "revoked": False,
            "limits": validate_limits(limits),
            "usage": {"day": _day(), "tokens": 0},
        }
        with self._lock:
            self._invites.append(item)
            self._save()
        # The token exists only in this response; it can't be shown again.
        return {**self._public(item), "token": token}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(item) for item in reversed(self._invites)]

    def revoke(self, invite_id: str) -> None:
        with self._lock:
            for item in self._invites:
                if item["id"] == invite_id:
                    item["revoked"] = True
                    self._save()
                    return
        raise ValueError("no such invite")

    def delete(self, invite_id: str) -> None:
        with self._lock:
            kept = [item for item in self._invites if item["id"] != invite_id]
            if len(kept) == len(self._invites):
                raise ValueError("no such invite")
            self._invites = kept
            self._save()

    def check(self, token: Any) -> dict[str, Any] | None:
        """The invite a token opens, if it is still valid."""
        if not isinstance(token, str) or not token.startswith(PREFIX):
            return None
        digest = _hash(token)
        with self._lock:
            for item in self._invites:
                if hmac.compare_digest(item["sha256"], digest) and self._valid(item):
                    item["last_used"] = time.time()
                    return self._public(item)
        return None

    def get(self, invite_id: str) -> dict[str, Any] | None:
        """The invite, if it is still valid (re-checked on every request)."""
        with self._lock:
            for item in self._invites:
                if item["id"] == invite_id and self._valid(item):
                    return self._public(item)
        return None

    def remaining(self, invite_id: str) -> int:
        with self._lock:
            for item in self._invites:
                if item["id"] == invite_id:
                    usage = item.setdefault("usage", {})
                    if usage.get("day") != _day():
                        usage.update(day=_day(), tokens=0)
                    return item["limits"]["daily_tokens"] - usage["tokens"]
        return 0

    def charge(self, invite_id: str, tokens: int) -> None:
        with self._lock:
            for item in self._invites:
                if item["id"] == invite_id:
                    usage = item.setdefault("usage", {})
                    if usage.get("day") != _day():
                        usage.update(day=_day(), tokens=0)
                    usage["tokens"] += max(0, int(tokens))
                    # Counts are written at most every few seconds.
                    if self._dirty_since is None:
                        self._dirty_since = time.monotonic()
                    if time.monotonic() - self._dirty_since > 5:
                        self._save()
                    return

    def flush(self) -> None:
        with self._lock:
            if self._dirty_since is not None:
                self._save()


class GuestModels:
    """The shared model pool, as one invite may use it.

    Enforces the invite's quotas on every call and keeps guests to the main
    and sub-agent groups. Calls may come from the app's event loop (browser
    runs) or from a job thread's loop (guests' own apps); the pool always runs
    on the app's loop.
    """

    routes = True
    supports_context_limits = True

    def __init__(self, gateway: Any, invite_id: str) -> None:
        self.gateway = gateway
        self.invite_id = invite_id
        self._active = 0
        self._lock = threading.Lock()

    @property
    def pool(self) -> Any:
        pool = self.gateway.pool()
        if pool is None:
            raise CapacityError("the models aren't loaded yet")
        return pool

    @property
    def limits(self) -> dict[str, int]:
        invite = self.gateway.invites.get(self.invite_id)
        if invite is None:
            raise PermissionError("this invite was revoked or has expired")
        return invite["limits"]

    @property
    def supports_native_tools(self) -> bool:
        return bool(getattr(self.gateway.pool(), "supports_native_tools", False))

    @property
    def native_call_dialect(self) -> str:
        return getattr(self.gateway.pool(), "native_call_dialect", "json")

    @property
    def max_model_length(self) -> int:
        return self.limits["context_tokens"]

    def context_limit(self, group: str | None) -> int:
        model = self.pool.context_limit(group) or self.limits["context_tokens"]
        return min(model, self.limits["context_tokens"])

    def partial(self, request_id: str) -> str | None:
        pool = self.gateway.pool()
        return pool.partial(request_id) if pool is not None else None

    async def generate(self, messages, max_tokens, request_id, **options):
        return await self._call(None, messages, max_tokens, request_id, options)

    async def generate_action(
        self, messages, max_tokens, request_id, tool_specs, **options
    ):
        return await self._call(tool_specs, messages, max_tokens, request_id, options)

    async def _call(self, tool_specs, messages, max_tokens, request_id, options):
        limits = self.limits
        group = options.pop("group", None) or self.gateway.groups()["driver"]
        if group not in self.gateway.groups().values():
            raise RequestValidationError("guests can use the main and sub-agent models")
        options.pop("worker", None)
        if self.gateway.invites.remaining(self.invite_id) <= 0:
            raise CapacityError("this invite has used its tokens for today")
        if type(max_tokens) is not int or max_tokens > limits["output_tokens"]:
            max_tokens = min(int(max_tokens), limits["output_tokens"])
        context = options.get("max_context_tokens") or limits["context_tokens"]
        options["max_context_tokens"] = min(context, self.context_limit(group))
        affinity = options.get("affinity")
        options["affinity"] = f"guest:{self.invite_id}:{affinity or request_id}"
        with self._lock:
            if self._active >= limits["concurrent"]:
                raise CapacityError("this invite already has its requests running")
            self._active += 1
        # Keep guests' request ids apart from the owner's and each other's.
        scoped = f"guest.{self.invite_id}.{request_id}"[:200]
        try:
            call = (
                self.pool.generate(messages, max_tokens, scoped, group=group, **options)
                if tool_specs is None
                else self.pool.generate_action(
                    messages, max_tokens, scoped, tool_specs, group=group, **options
                )
            )
            loop = self.gateway.loop
            if asyncio.get_running_loop() is loop:
                generation = await call
            else:
                future = asyncio.run_coroutine_threadsafe(call, loop)
                try:
                    generation = await asyncio.wrap_future(future)
                except asyncio.CancelledError:
                    future.cancel()
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(self.pool.cancel(scoped))
                    )
                    raise
        finally:
            with self._lock:
                self._active -= 1
        self.gateway.invites.charge(
            self.invite_id, generation.input_tokens + generation.output_tokens
        )
        return generation

    async def cancel(self, request_id: str) -> None:
        pool = self.gateway.pool()
        if pool is not None:
            await pool.cancel(f"guest.{self.invite_id}.{request_id}"[:200])


class GroupModels:
    """One model group of an invite, for the worker job protocol."""

    def __init__(self, models: GuestModels, group: str) -> None:
        self.models, self.group = models, group
        self.supports_context_limits = True

    @property
    def supports_native_tools(self) -> bool:
        return self.models.supports_native_tools

    def health(self) -> dict[str, Any]:
        pool = self.models.gateway.pool()
        limit = self.models.context_limit(self.group) if pool is not None else 0
        return {
            **self.models.gateway.identity(self.group),
            "worker_id": f"guest-{self.group}",
            "native_call_dialect": self.models.native_call_dialect
            if pool is not None
            else "json",
            "max_model_length": limit,
            "capabilities": ["greedy_chat", "cancellation"],
        }

    async def generate(self, messages, max_tokens, request_id, **options):
        return await self.models.generate(
            messages, max_tokens, request_id, group=self.group, **options
        )

    async def generate_action(
        self, messages, max_tokens, request_id, tool_specs, **options
    ):
        return await self.models.generate_action(
            messages, max_tokens, request_id, tool_specs, group=self.group, **options
        )

    async def cancel(self, request_id: str) -> None:
        await self.models.cancel(request_id)


class ToolError(Exception):
    """What the guest's browser reported when a tool failed."""


@dataclass
class GuestRun:
    """One agent run for a browser guest. Lives only in memory."""

    run_id: str
    invite_id: str
    tools: tuple[str, ...]
    created: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    status: str = "running"
    task: asyncio.Task | None = None
    runtime: Any = None
    store: AgentStore | None = None
    changed: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, event: dict[str, Any]) -> None:
        with self.changed:
            self.events.append({"seq": len(self.events), **event})
            self.changed.notify_all()

    def partial(self, models: GuestModels) -> dict[str, str]:
        if self.runtime is None:
            return {}
        live = {}
        for (run_id, agent_id), request_id in list(self.runtime._requests.items()):
            if run_id != self.run_id:
                continue
            text = models.partial(f"guest.{self.invite_id}.{request_id}"[:200])
            if text:
                live[agent_id] = text[-4000:]
        return live


class RelayTools(WorkspaceTools):
    """The workspace tools, carried out by the guest's browser."""

    def __init__(self, run: GuestRun, loop: asyncio.AbstractEventLoop) -> None:
        self.run, self.loop = run, loop
        names = set(run.tools)
        self.specs = tuple(spec for spec in WorkspaceTools.specs if spec.name in names)

    def available(self):
        # The browser offers only the tools its guest allows.
        return self.specs

    def mapping(self) -> dict[str, Any]:
        return {spec.name: self._relay(spec.name) for spec in self.specs}

    def _relay(self, name: str):
        async def call(args: dict) -> str:
            call_id = uuid.uuid4().hex[:16]
            future = self.loop.create_future()
            self.run.pending[call_id] = future
            self.run.emit({"type": "tool", "id": call_id, "name": name, "args": args})
            try:
                outcome = await asyncio.wait_for(future, TOOL_TIMEOUT)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    "the guest's browser didn't answer; is the tab still open?"
                ) from None
            finally:
                self.run.pending.pop(call_id, None)
            self.run.emit(
                {"type": "tool_done", "id": call_id, "ok": "error" not in outcome}
            )
            if "error" in outcome:
                raise ToolError(outcome["error"])
            return outcome["result"]

        return call


class BrowserRuns:
    """Agent runs for browser guests, in memory, forgotten when they end."""

    KEEP_FINISHED = 600.0

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.runs: dict[str, GuestRun] = {}
        self._lock = threading.Lock()

    def _get(self, invite_id: str, run_id: str) -> GuestRun:
        with self._lock:
            run = self.runs.get(run_id)
        if run is None or run.invite_id != invite_id:
            raise KeyError("no such run")
        return run

    def _prune(self) -> None:
        now = time.time()
        with self._lock:
            for run_id, run in list(self.runs.items()):
                if run.status != "running" and now - run.created > self.KEEP_FINISHED:
                    del self.runs[run_id]

    def active(self, invite_id: str | None = None) -> int:
        with self._lock:
            return sum(
                run.status == "running"
                and (invite_id is None or run.invite_id == invite_id)
                for run in self.runs.values()
            )

    def _config(self, invite: dict, payload: dict) -> RuntimeConfig:
        owner = dict(self.gateway.config().get("runtime", {}))
        limits = invite["limits"]
        groups = self.gateway.groups()
        model = payload.get("model", "driver")
        if model not in groups:
            raise ValueError(f"model must be one of {sorted(groups)}")
        effort = payload.get("effort", "off")
        context = min(limits["context_tokens"], owner.get("max_context_tokens", 65536))
        output = min(limits["output_tokens"], owner.get("max_output_tokens", 4096))
        if output >= context // 2:
            output = context // 4
        config = RuntimeConfig(**owner)
        return replace(
            config,
            driver_group=groups[model],
            subagent_group=groups.get("subagent", groups[model]),
            max_context_tokens=context,
            max_output_tokens=output,
            max_total_tokens=max(
                4000,
                min(
                    self.gateway.invites.remaining(invite["id"]),
                    config.max_total_tokens,
                ),
            ),
            max_concurrent_generations=min(
                limits["concurrent"], config.max_concurrent_generations
            ),
            tool_timeout_seconds=TOOL_TIMEOUT + 30,
            deadline_seconds=min(config.deadline_seconds, 4 * 3600),
            effort=effort,
        )

    async def start(self, invite: dict, payload: dict) -> dict:
        """Start a turn: ``messages`` are the conversation so far (kept by
        the browser), ``task`` the new request, ``tools`` the ones the
        browser can carry out, ``model`` driver or subagent, ``effort``."""
        allowed = {"messages", "task", "tools", "model", "effort", "media"}
        if not isinstance(payload, dict) or payload.keys() - allowed:
            raise ValueError(f"a run takes {sorted(allowed)}")
        if self.gateway.pool() is None:
            raise ValueError("the models are still loading; try again in a minute")
        self._prune()
        if self.active(invite["id"]) >= invite["limits"]["concurrent"]:
            raise ValueError("finish or stop a running reply first")
        if self.active() >= self.gateway.max_guest_runs:
            raise ValueError("this host is busy; try again shortly")
        if self.gateway.invites.remaining(invite["id"]) <= 0:
            raise ValueError("this invite has used its tokens for today")
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip() or len(task) > 12_000:
            raise ValueError("the message must have 1 to 12000 characters")
        tools = payload.get("tools", [])
        if not isinstance(tools, list) or set(tools) - set(RELAYED_TOOLS):
            raise ValueError(f"tools must come from {list(RELAYED_TOOLS)}")
        history = []
        raw = payload.get("messages", [])
        if not isinstance(raw, list) or len(raw) > 400:
            raise ValueError("messages must be a list of up to 400 turns")
        total = 0
        for item in raw:
            if (
                not isinstance(item, dict)
                or item.keys() - {"role", "content"}
                or item.get("role") not in {"user", "assistant"}
                or not isinstance(item.get("content"), str)
            ):
                raise ValueError(
                    "earlier messages need a user or assistant role and text"
                )
            total += len(item["content"])
            history.append(ChatMessage(item["role"], item["content"]))
        if total > 4 * invite["limits"]["context_tokens"]:
            raise ValueError(
                "the conversation is too long for this invite; start a new chat"
            )
        media = ()
        if payload.get("media"):
            from ..agents.media import validate_media

            media = validate_media(payload["media"])
        config = self._config(invite, payload)
        run = GuestRun(uuid.uuid4().hex, invite["id"], tuple(tools))
        models = self.gateway.models(invite["id"])
        relay = RelayTools(run, self.gateway.loop)
        mapping = relay.mapping()
        descriptions = relay.tool_descriptions()
        store = AgentStore(":memory:")
        runtime = ConversationRuntime(
            models,
            store,
            mapping,
            config,
            tool_descriptions=descriptions,
            tool_schemas=relay.schemas,
            replay_safe_tools=set(READ_ONLY_TOOLS) & set(mapping),
            mutating_tools=set(mapping) - READ_ONLY_TOOLS,
        )
        store.create_run(
            run.run_id,
            task,
            asdict(config),
            runtime._system_prompt(config, task=task),
            tools=sorted({*runtime.tools, "journal_read"}),
            completion_required=False,
            media=media,
        )
        if history:
            with store.transaction():
                store.connection.execute(
                    "DELETE FROM messages WHERE run_id=? AND agent_id='root' AND role='user'",
                    (run.run_id,),
                )
                for message in history:
                    store.add_message(run.run_id, "root", message.role, message.content)
                store.add_message(run.run_id, "root", "user", task, media=media)
        run.runtime, run.store = runtime, store
        with self._lock:
            self.runs[run.run_id] = run

        async def drive() -> None:
            final: dict[str, Any] = {"type": "done", "status": "failed", "result": None}
            try:
                result = await runtime.resume(run.run_id)
                final.update(
                    status=result.status,
                    result=result.output,
                    error=result.error,
                    usage=asdict(result.usage),
                    agents=len(result.agents),
                )
            except asyncio.CancelledError:
                final.update(status="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the guest.
                final.update(error=f"{type(exc).__name__}: {exc}")
            finally:
                for future in run.pending.values():
                    if not future.done():
                        future.cancel()
                # The journal goes with the run: nothing of it stays here.
                run.runtime = None
                if run.store is not None:
                    run.store.connection.close()
                    run.store = None
                run.status = final["status"]
                run.emit(final)

        run.task = asyncio.create_task(drive())
        return {"run_id": run.run_id}

    def events(
        self, invite_id: str, run_id: str, after: int, wait: float
    ) -> dict[str, Any]:
        """New events since ``after``, waiting up to ``wait`` seconds."""
        run = self._get(invite_id, run_id)
        deadline = time.monotonic() + max(0.0, min(wait, 25.0))
        models = self.gateway.models(invite_id)
        with run.changed:
            while len(run.events) <= after and run.status == "running":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                run.changed.wait(min(remaining, 0.25))
                if run.partial(models):
                    # Live text counts as news, at most four times a second.
                    break
            events = run.events[after:]
        return {
            "events": events,
            "next": after + len(events),
            "partial": run.partial(models),
            "status": run.status,
        }

    def answer(self, invite_id: str, run_id: str, call_id: str, payload: Any) -> None:
        run = self._get(invite_id, run_id)
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError('answer with {"result": text} or {"error": text}')
        ((kind, value),) = payload.items()
        if kind not in {"result", "error"} or not isinstance(value, str):
            raise ValueError('answer with {"result": text} or {"error": text}')
        if len(value) > MAX_TOOL_RESULT:
            value = value[:MAX_TOOL_RESULT] + "\n[result cut at 256 KB]"
        future = run.pending.get(call_id)
        if future is None:
            raise KeyError("that tool call isn't waiting")
        self.gateway.loop.call_soon_threadsafe(
            lambda: future.done() or future.set_result({kind: value})
        )

    def cancel(self, invite_id: str, run_id: str) -> None:
        run = self._get(invite_id, run_id)
        if run.task is not None:
            self.gateway.loop.call_soon_threadsafe(run.task.cancel)

    def cancel_invite(self, invite_id: str) -> None:
        with self._lock:
            runs = [run for run in self.runs.values() if run.invite_id == invite_id]
        for run in runs:
            if run.task is not None:
                self.gateway.loop.call_soon_threadsafe(run.task.cancel)
