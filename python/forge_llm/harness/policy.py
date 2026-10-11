"""Operator policy for exact, short-lived tool invocations.

Approvals are audit records, not replayable capabilities. Only the gate and
event loop that created a live request can release its waiting invocation.
Restarting a gate invalidates outstanding records in its configured run.
"""

from __future__ import annotations

import asyncio
import copy
import difflib
import hashlib
import inspect
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..agents.context import tool_actor
from ..agents.store import TERMINAL, AgentStore, StoreError
from ..agents.tools import ToolValidationError, WorkspaceTools

Tool = Callable[[dict[str, Any]], Awaitable[str]]
Hook = Callable[[dict[str, Any]], Awaitable[Any]]
Preview = Callable[..., Any]


def _canonical(value: Any) -> str:
    """Reject non-JSON values rather than silently changing approval arguments."""

    def validate(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise ToolValidationError("approval arguments must contain finite JSON values")

    try:
        validate(value)
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (RecursionError, ValueError) as exc:
        raise ToolValidationError("approval arguments must be finite JSON") from exc


class HookRegistry:
    """Ordered, operator-provided async callbacks with one event time budget.

    ``await register('before_tool', callback)`` installs a callback receiving an
    isolated payload. Raising or returning False rejects the event. Callbacks
    must be cooperative async functions; this is not a sandbox for untrusted
    code. Cancellation propagates without releasing the guarded tool.
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("hook timeout must be finite and positive")
        self.timeout_seconds = timeout_seconds
        self._callbacks: dict[str, list[Hook]] = {}

    async def register(self, event: str, callback: Hook) -> None:
        if not isinstance(event, str) or not event or len(event) > 80:
            raise ValueError("hook event must contain 1 to 80 characters")
        if not inspect.iscoroutinefunction(
            callback
        ) and not inspect.iscoroutinefunction(getattr(callback, "__call__", None)):
            raise TypeError("hooks must be async callbacks")
        self._callbacks.setdefault(event, []).append(callback)

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        async def dispatch() -> None:
            for callback in tuple(self._callbacks.get(event, ())):
                if await callback(copy.deepcopy(payload)) is False:
                    raise ToolValidationError(f"hook rejected {event}")

        try:
            await asyncio.wait_for(dispatch(), self.timeout_seconds)
        except asyncio.CancelledError:
            raise
        except ToolValidationError:
            raise
        except Exception as exc:
            raise ToolValidationError(
                f"hook failed for {event}: {type(exc).__name__}"
            ) from exc


@dataclass
class _Waiting:
    event: asyncio.Event
    loop: asyncio.AbstractEventLoop
    args_json: str
    tool: str
    actor: tuple[str, str]
    approved: bool | None = None


class ApprovalGate:
    """Wrap tools with read-only or exact-invocation review policy.

    ``pending(run_id)`` returns durable previews for the operator. The service
    calls ``decide(id, approved=True)`` on the waiting tool's event loop; this
    records a decision and wakes that invocation, never executes a tool itself.
    ``preview`` may be a mapping of tool names to callbacks taking args, or one
    callback taking (name, args). Preview callbacks must only inspect state.
    Specify ``run_id`` for concurrent per-run gates so startup invalidation is
    confined to the run being recovered.
    """

    def __init__(
        self,
        store: AgentStore,
        mode: str = "review",
        timeout_seconds: float = 300,
        *,
        run_id: str | None = None,
        preview: Preview | Mapping[str, Preview] | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        if mode not in {"review", "read_only"}:
            raise ValueError("policy mode must be review or read_only")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("approval timeout must be finite and positive")
        self.store, self.mode, self.timeout_seconds = store, mode, timeout_seconds
        self.run_id, self.preview, self.hooks = run_id, preview, hooks
        self._owner = uuid.uuid4().hex
        self._waiting: dict[str, _Waiting] = {}
        self.store.connection.executescript("""
            CREATE TABLE IF NOT EXISTS harness_approvals (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                agent_id TEXT NOT NULL, tool TEXT NOT NULL, args TEXT NOT NULL,
                args_hash TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                expires REAL NOT NULL, resolved REAL, owner TEXT NOT NULL,
                preview TEXT,
                FOREIGN KEY(run_id,agent_id) REFERENCES agents(run_id,agent_id)
            );
            CREATE INDEX IF NOT EXISTS harness_approvals_run
                ON harness_approvals(run_id,status,created);
        """)
        where = "status IN ('pending','approved','executing')"
        values: tuple[Any, ...] = ()
        if run_id is not None:
            where += " AND run_id=?"
            values = (run_id,)
        with self.store.transaction():
            rows = list(
                self.store.connection.execute(
                    f"SELECT * FROM harness_approvals WHERE {where}",
                    values,
                )
            )
            for row in rows:
                self._resolve(
                    row, "interrupted" if row["status"] == "executing" else "stale"
                )

    def _resolve(self, row: Any, status: str) -> None:
        self.store.connection.execute(
            "UPDATE harness_approvals SET status=?,resolved=? WHERE id=?",
            (status, time.time(), row["id"]),
        )
        self.store.event(
            row["run_id"],
            row["agent_id"],
            "approval_resolved",
            {
                "id": row["id"],
                "tool": row["tool"],
                "status": status,
            },
        )

    def _row(self, approval_id: str) -> Any:
        return self.store.connection.execute(
            "SELECT * FROM harness_approvals WHERE id=?",
            (approval_id,),
        ).fetchone()

    def _active(self, actor: tuple[str, str]) -> None:
        if self.run_id is not None and actor[0] != self.run_id:
            raise ToolValidationError("tool actor belongs to another policy run")
        try:
            run, agent = self.store.run(actor[0]), self.store.agent(*actor)
        except StoreError as exc:
            raise ToolValidationError(
                "review requires a recorded run and agent"
            ) from exc
        if run["status"] in TERMINAL or run["cancel_requested"]:
            raise ToolValidationError("approval run is no longer active")
        if run["deadline"] <= time.time() or agent.status in TERMINAL:
            raise ToolValidationError("approval agent is no longer active")

    def _check(self, row: Any, waiting: _Waiting) -> None:
        if (
            row is None
            or row["owner"] != self._owner
            or row["args"] != waiting.args_json
            or row["tool"] != waiting.tool
            or (row["run_id"], row["agent_id"]) != waiting.actor
            or row["args_hash"]
            != hashlib.sha256(waiting.args_json.encode()).hexdigest()
        ):
            raise ToolValidationError("approval does not match this exact invocation")
        if row["expires"] <= time.time():
            raise ToolValidationError("approval expired; request review again")
        self._active(waiting.actor)

    def pending(self, run_id: str) -> list[dict[str, Any]]:
        """List live requests, expiring stale/cancelled requests before display."""
        if self.run_id is not None and run_id != self.run_id:
            return []
        result = []
        with self.store.transaction():
            rows = list(
                self.store.connection.execute(
                    "SELECT * FROM harness_approvals WHERE run_id=? AND status='pending' ORDER BY created,id",
                    (run_id,),
                )
            )
            for row in rows:
                try:
                    self._active((row["run_id"], row["agent_id"]))
                except ToolValidationError:
                    self._resolve(row, "cancelled")
                    continue
                if row["expires"] <= time.time():
                    self._resolve(row, "expired")
                    continue
                result.append(
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "run_id",
                            "agent_id",
                            "tool",
                            "status",
                            "created",
                            "expires",
                            "args_hash",
                        )
                    }
                    | {
                        "args": json.loads(row["args"]),
                        "preview": json.loads(row["preview"])
                        if row["preview"]
                        else None,
                    }
                )
        return result

    def decide(self, approval_id: str, approved: bool) -> None:
        """Resolve once, on the originating event loop, for this exact request."""
        if type(approved) is not bool:
            raise ToolValidationError("approval decision must be a boolean")
        waiting = self._waiting.get(approval_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ToolValidationError(
                "approval requires its active event loop"
            ) from exc
        if waiting is None or waiting.loop is not loop:
            raise ToolValidationError("approval is stale or belongs to another gate")
        row = self._row(approval_id)
        try:
            self._check(row, waiting)
        except ToolValidationError:
            if row is not None and row["status"] in {"pending", "approved"}:
                self._resolve(
                    row, "expired" if row["expires"] <= time.time() else "stale"
                )
                waiting.event.set()
            raise
        with self.store.transaction():
            row = self._row(approval_id)
            self._check(row, waiting)
            if row["status"] != "pending" or waiting.approved is not None:
                raise ToolValidationError("approval has already been resolved")
            self._resolve(row, "approved" if approved else "denied")
        waiting.approved = approved
        waiting.event.set()

    async def _preview(self, name: str, args: dict[str, Any]) -> str | None:
        callback = self.preview
        if isinstance(callback, Mapping):
            callback = callback.get(name)
            arguments = (copy.deepcopy(args),)
        else:
            arguments = (name, copy.deepcopy(args))
        if callback is None:
            return None

        async def invoke() -> Any:
            if inspect.iscoroutinefunction(callback):
                return await callback(*arguments)
            value = await asyncio.to_thread(callback, *arguments)
            return await value if inspect.isawaitable(value) else value

        try:
            value = await asyncio.wait_for(invoke(), min(self.timeout_seconds, 10))
            return _canonical(value)
        except asyncio.CancelledError:
            raise
        except ToolValidationError:
            raise
        except Exception as exc:
            raise ToolValidationError(
                f"approval preview failed: {type(exc).__name__}"
            ) from exc

    async def _review(self, name: str, args_json: str, preview: str | None) -> str:
        actor = tool_actor.get()
        self._active(actor)
        approval_id, now = uuid.uuid4().hex, time.time()
        waiting = _Waiting(
            asyncio.Event(), asyncio.get_running_loop(), args_json, name, actor
        )
        self._waiting[approval_id] = waiting
        try:
            with self.store.transaction():
                self.store.connection.execute(
                    "INSERT INTO harness_approvals(id,run_id,agent_id,tool,args,args_hash,status,created,expires,owner,preview) VALUES(?,?,?,?,?,?,'pending',?,?,?,?)",
                    (
                        approval_id,
                        *actor,
                        name,
                        args_json,
                        hashlib.sha256(args_json.encode()).hexdigest(),
                        now,
                        now + self.timeout_seconds,
                        self._owner,
                        preview,
                    ),
                )
                self.store.event(
                    *actor,
                    "approval_requested",
                    {
                        "id": approval_id,
                        "tool": name,
                        "args": json.loads(args_json),
                        "args_hash": hashlib.sha256(args_json.encode()).hexdigest(),
                        "expires": now + self.timeout_seconds,
                        "preview": json.loads(preview) if preview is not None else None,
                    },
                )
            deadline = asyncio.get_running_loop().time() + self.timeout_seconds
            while not waiting.event.is_set():
                row = self._row(approval_id)
                self._check(row, waiting)
                if row["status"] != "pending":
                    raise ToolValidationError(f"approval is {row['status']}")
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._resolve(row, "expired")
                    raise ToolValidationError("approval expired; request review again")
                try:
                    await asyncio.wait_for(waiting.event.wait(), min(remaining, 0.1))
                except TimeoutError:
                    pass
            row = self._row(approval_id)
            self._check(row, waiting)
            if waiting.approved is not True or row["status"] != "approved":
                raise ToolValidationError("operator denied this tool invocation")
            return approval_id
        except BaseException as exc:
            row = self._row(approval_id)
            if row is not None and row["status"] in {"pending", "approved"}:
                self._resolve(
                    row,
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "expired"
                    if row["expires"] <= time.time()
                    else "stale",
                )
            self._waiting.pop(approval_id, None)
            raise

    def wrap(self, name: str, tool: Tool, *, mutating: bool = False) -> Tool:
        """Return an async tool with one immutable argument snapshot per call."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be nonempty")

        async def guarded(args: dict[str, Any]) -> str:
            if type(args) is not dict:
                raise ToolValidationError("tool arguments must be a JSON object")
            args_json = _canonical(args)
            snapshot = json.loads(args_json)
            actor = tool_actor.get()
            if self.run_id is not None and actor[0] != self.run_id:
                raise ToolValidationError("tool actor belongs to another policy run")
            if mutating and self.mode == "read_only":
                raise ToolValidationError("read_only policy rejects mutating tools")
            payload = {
                "run_id": actor[0],
                "agent_id": actor[1],
                "tool": name,
                "args": snapshot,
                "mutating": mutating,
            }
            if self.hooks is not None:
                await self.hooks.emit("before_tool", payload)
            approval_id = None
            try:
                if mutating:
                    preview = await self._preview(name, snapshot)
                    approval_id = await self._review(name, args_json, preview)
                    if await self._preview(name, snapshot) != preview:
                        raise ToolValidationError(
                            "tool preview changed; request review again"
                        )
                    waiting = self._waiting[approval_id]
                    with self.store.transaction():
                        row = self._row(approval_id)
                        self._check(row, waiting)
                        if row["status"] != "approved" or waiting.approved is not True:
                            raise ToolValidationError("approval is no longer available")
                        self._resolve(row, "executing")
                result = await tool(snapshot)
            except BaseException as exc:
                if approval_id is not None:
                    row = self._row(approval_id)
                    if row["status"] == "executing":
                        status = (
                            "interrupted"
                            if isinstance(exc, asyncio.CancelledError)
                            else "failed"
                        )
                    else:
                        status = (
                            "cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "stale"
                        )
                    self._resolve(row, status)
                raise
            finally:
                if approval_id is not None:
                    self._waiting.pop(approval_id, None)
            if approval_id is not None:
                self._resolve(self._row(approval_id), "completed")
            if self.hooks is not None:
                try:
                    await self.hooks.emit("after_tool", payload | {"result": result})
                except ToolValidationError as exc:
                    # The tool already ran. Preserve its receipt and audit the
                    # callback failure rather than report a pre-execution denial.
                    try:
                        self.store.run(actor[0])
                        self.store.event(
                            *actor,
                            "hook_failed",
                            {"event": "after_tool", "tool": name, "error": str(exc)},
                        )
                    except StoreError:
                        pass
            return result

        return guarded


def workspace_preview(
    tools: WorkspaceTools, name: str, args: dict[str, Any]
) -> dict[str, Any] | None:
    """Inspect a workspace edit using the same confinement and version guards.

    This helper never grants a read receipt or writes files. The gate rechecks
    its output after review so a changed before-image requires fresh approval.
    """
    if name not in {"write_file", "replace_text", "edit_lines", "save_artifact"}:
        return None
    artifact = name == "save_artifact"
    path = tools._path(args.get("path"), artifact=artifact)
    lexical = (tools.artifacts if artifact else tools.workspace) / args["path"]
    tools._assert_mutable(path, lexical)
    data = tools._safe_read(path) if path.exists() else b""
    digest = hashlib.sha256(data).hexdigest() if path.exists() else "missing"
    expected = "missing" if artifact else tools._expected_version(path, args)
    if digest != expected:
        raise ToolValidationError("file version changed; read it again before editing")
    try:
        before = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolValidationError("edit preview requires UTF-8 text") from exc
    if "\x00" in before:
        raise ToolValidationError("binary files are unavailable")
    if name == "replace_text":
        old, replacement = args.get("old_text"), args.get("new_text")
        if (
            not isinstance(old, str)
            or not old
            or not isinstance(replacement, str)
            or before.count(old) != 1
        ):
            raise ToolValidationError(
                "old_text must match exactly once and new_text must be text"
            )
        after = before.replace(old, replacement, 1)
    elif name == "edit_lines":
        start, end, replacement = (
            args.get("start_line"),
            args.get("end_line"),
            args.get("content"),
        )
        lines = before.splitlines(keepends=True)
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(lines)
            or not isinstance(replacement, str)
        ):
            raise ToolValidationError(
                "edit preview requires a valid inclusive line range and text"
            )
        endings = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
        if replacement and not replacement.endswith(tuple(endings)):
            last = lines[end - 1]
            replacement += (
                "\r\n"
                if last.endswith("\r\n")
                else last[-1:]
                if last.endswith(tuple(endings))
                else ""
            )
        after = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
    else:
        after = args.get("content")
        if not isinstance(after, str):
            raise ToolValidationError("edit preview requires content text")
    if "\x00" in after or len(after.encode()) > tools.max_file_bytes:
        raise ToolValidationError("edit preview exceeds text size limit")
    if not artifact and path.suffix == ".py":
        try:
            compile(after, args["path"], "exec", dont_inherit=True)
        except SyntaxError as exc:
            raise ToolValidationError(
                f"Python syntax error in edit preview: {exc.msg}"
            ) from exc
    relative = args["path"]
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )
    return {
        "path": relative,
        "diff": diff,
        "before_sha256": digest,
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
    }
