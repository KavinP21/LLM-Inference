"""Read-only developer inspection of current files and recorded agent work.

Workspace reads show the current disk. Change provenance comes from successful
tool receipts and exact approval records; a proposed diff is never presented as
an executed edit. Viewing a file does not grant an agent a write receipt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from ..agents.store import TERMINAL
from ..agents.tools import _IGNORED, _PRIVATE, ToolValidationError

if TYPE_CHECKING:
    from .service import AppService

MAX_FILE_BYTES = 128_000
MAX_RECORDS = 500
MAX_RUNS = 50
EDIT_TOOLS = frozenset({"write_file", "replace_text", "edit_lines", "save_artifact"})
COMMAND_TOOLS = frozenset({"run_tests", "run_command", "command_run"})
SETTLED = TERMINAL | {"completed_with_failures"}
LANGUAGES = {
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cu": "cuda",
    ".c": "c",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".svg": "xml",
    ".rs": "rust",
    ".go": "go",
}


def _decode(value: str) -> str:
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError as exc:
        raise ToolValidationError("file path must use UTF-8") from exc


def _object(value: Any) -> dict | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _identity(agent: str, tool: str, args: dict) -> tuple[str, str, str]:
    return agent, tool, json.dumps(args, sort_keys=True, ensure_ascii=False)


class DeveloperWorkspace:
    """Compose with AppService; dispatch on the service's SQLite owner loop.

    ``handles(method, path)`` lets the service route these endpoints before its
    conversation catch-all. File enumeration and reading run off the loop, but
    journal access and operator messages stay on that loop.
    """

    def __init__(self, service: AppService):
        # Local import keeps composition safe when service imports this helper.
        from .service import AppWorkspaceTools

        self.service = service
        self._tools_type = AppWorkspaceTools
        self.tools = self._tools_type(
            service.workspace,
            service.state_dir / "artifacts",
            state_dir=service.state_dir,
            max_file_bytes=MAX_FILE_BYTES,
        )

    @staticmethod
    def handles(method: str, path: str) -> bool:
        parts = path.strip("/").split("/")
        return (
            parts[:2] == ["api", "workspace"]
            or (
                method == "GET"
                and len(parts) == 4
                and parts[:2] == ["api", "chats"]
                and parts[3] == "developer"
            )
            or (
                method == "POST"
                and len(parts) == 6
                and parts[:2] == ["api", "runs"]
                and parts[3] == "agents"
                and parts[5] in {"message", "cancel"}
            )
        )

    def _path(self, value: str, *, run_id: str | None = None) -> tuple[Any, Path]:
        tools = (
            self.tools
            if run_id is None
            else self._tools_type(
                self.service.workspace,
                self.service.state_dir / "artifacts" / run_id,
                state_dir=self.service.state_dir,
                max_file_bytes=MAX_FILE_BYTES,
            )
        )
        path = tools._path(value, artifact=run_id is not None)
        base = tools.workspace if run_id is None else tools.artifacts
        blocked = {part.casefold() for part in _IGNORED | _PRIVATE | {".instinct"}}
        parts = (*Path(value).parts, *path.relative_to(base).parts)
        state = str(self.service.state_dir).casefold()
        if any(
            part.casefold() in blocked or part.casefold().startswith(".env.")
            for part in parts
        ) or (
            run_id is None
            and (
                str(path).casefold() == state
                or str(path).casefold().startswith(state + os.sep)
            )
        ):
            raise ToolValidationError("ignored or private files are unavailable")
        return tools, path

    def _file(self, value: str) -> dict:
        tools, path = self._path(value)
        data = tools._safe_read(path)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolValidationError("file preview requires UTF-8 text") from exc
        if "\x00" in content:
            raise ToolValidationError("binary files are unavailable")
        return {
            "path": str(path.relative_to(self.service.workspace)),
            "content": content,
            "sha256": hashlib.sha256(data).hexdigest(),
            "total_lines": len(content.splitlines()),
            "language": LANGUAGES.get(path.suffix.lower(), "text"),
            "size": len(data),
            "source": "current_workspace",
        }

    def _list(self, payload: dict) -> dict:
        if type(payload) is not dict or payload.keys() - {"query", "path", "limit"}:
            raise ToolValidationError("file search supports query, path and limit")
        query, base, limit = (
            payload.get("query", ""),
            payload.get("path", "."),
            payload.get("limit", 200),
        )
        if not isinstance(query, str) or len(query) > 200:
            raise ToolValidationError("file query must contain at most 200 characters")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ToolValidationError(
                "file limit must be an integer between 1 and 1000"
            )
        _, directory = self._path(base)
        if not directory.exists():
            raise ToolValidationError("file search directory does not exist")
        result, truncated = [], False
        for path in self.tools._files(directory):
            relative = str(path.relative_to(self.service.workspace))
            if query.casefold() not in relative.casefold():
                continue
            try:
                _, safe = self._path(relative)
                size = safe.stat().st_size
            except (OSError, ValueError):
                continue
            if len(result) == limit:
                truncated = True
                break
            result.append(
                {
                    "path": relative,
                    "size": size,
                    "language": LANGUAGES.get(path.suffix.lower(), "text"),
                }
            )
        return {"files": result, "truncated": truncated, "source": "current_workspace"}

    def _approval_records(self, run_id: str) -> list[dict]:
        gate = self.service.gates.get(run_id)
        if gate is not None:
            gate.pending(run_id)  # Resolve expired/cancelled requests before display.
        exists = self.service.store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='harness_approvals'"
        ).fetchone()
        if not exists:
            return []
        rows = self.service.store.connection.execute(
            "SELECT * FROM harness_approvals WHERE run_id=? ORDER BY created,id",
            (run_id,),
        )
        return [
            dict(
                row,
                args=json.loads(row["args"]),
                preview=json.loads(row["preview"]) if row["preview"] else None,
            )
            for row in rows
        ]

    def _events(self, run_id: str) -> tuple[list[dict], bool]:
        rows = self.service.store.connection.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?",
            (run_id, MAX_RECORDS + 1),
        ).fetchall()
        return [
            dict(row, payload=json.loads(row["payload"]))
            for row in reversed(rows[:MAX_RECORDS])
        ], len(rows) > MAX_RECORDS

    def _recorded_file(
        self,
        run_id: str,
        agent_id: str,
        name: str,
        args: dict,
        report: dict,
        approval: dict | None,
        created: float,
    ) -> dict | None:
        digest, size = report.get("sha256"), report.get("bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(size) is not int
            or size < 0
        ):
            return None
        artifact = name == "save_artifact"
        path = args.get("path")
        try:
            _, safe_path = self._path(path, run_id=run_id if artifact else None)
        except (OSError, ValueError, TypeError):
            return None
        if report.get("path") != str(safe_path):
            return None
        preview = approval.get("preview") if approval else None
        before = preview.get("before_sha256") if isinstance(preview, dict) else None
        change = (
            "created"
            if artifact
            or before == "missing"
            or args.get("expected_sha256") == "missing"
            else "modified"
            if name in {"replace_text", "edit_lines"} or before is not None
            else None
        )
        return {
            "path": path,
            "origin": "artifact" if artifact else "workspace",
            "run_id": run_id,
            "agent_id": agent_id,
            "change": change,
            "size": size,
            "sha256": digest,
            "created": created,
            "tool": name,
            "verified": True,
        }

    def _current(self, receipt: dict) -> dict:
        try:
            tools, path = self._path(
                receipt["path"],
                run_id=receipt["run_id"] if receipt["origin"] == "artifact" else None,
            )
            data = tools._safe_read(path)
        except FileNotFoundError:
            return {"exists": False, "current_matches_receipt": False}
        except (OSError, ValueError):
            return {"exists": None, "current_matches_receipt": None}
        digest = hashlib.sha256(data).hexdigest()
        return {
            "exists": True,
            "current_sha256": digest,
            "current_size": len(data),
            "current_matches_receipt": digest == receipt["sha256"],
        }

    async def chat(self, chat_id: str) -> dict:
        controller, store = self.service.conversations, self.service.store
        controller._chat(chat_id)
        turns = store.connection.execute(
            "SELECT run_id FROM studio_chat_turns WHERE chat_id=? AND run_id IS NOT NULL ORDER BY created DESC,rowid DESC LIMIT ?",
            (chat_id, MAX_RUNS + 1),
        ).fetchall()
        runs = [row["run_id"] for row in reversed(turns[:MAX_RUNS])]
        agents, files, changes, commands, reads = [], {}, [], [], []
        truncated = len(turns) > MAX_RUNS
        for run_id in runs:
            run = store.run(run_id)
            job = self.service.jobs.get(run_id)
            live = (
                job is not None
                and not job.done()
                and run["status"] not in SETTLED
                and not run["cancel_requested"]
            )
            approvals = self._approval_records(run_id)
            grouped = {}
            for row in approvals:
                grouped.setdefault(
                    _identity(row["agent_id"], row["tool"], row["args"]), []
                ).append(row)
            records, active_commands, started = [], {}, {}
            for agent in store.agents(run_id):
                data = asdict(agent)
                data["current_action"] = data.pop("pending_action")
                data["can_message"] = live and agent.status not in TERMINAL
                runtime = self.service.runtimes.get(run_id)
                data["can_cancel"] = (
                    live
                    and runtime is not None
                    and run_id in runtime._active
                    and agent.status not in TERMINAL
                )
                agents.append(data)
            events, omitted = self._events(run_id)
            truncated |= omitted
            for event in events:
                payload, agent_id = event["payload"], event["agent_id"]
                if not isinstance(payload, dict):
                    continue
                name, args = payload.get("name"), payload.get("args")
                if event["kind"] not in {
                    "tool_started",
                    "tool_result",
                } or not isinstance(args, dict):
                    continue
                key = _identity(agent_id, name, args)
                if event["kind"] == "tool_started":
                    started[key] = event["created"]
                eligible = [
                    row
                    for row in grouped.get(key, ())
                    if started.get(key, 0) <= row["created"] <= event["created"]
                ]
                approval = eligible[-1] if eligible else None
                if name in COMMAND_TOOLS:
                    if event["kind"] == "tool_started":
                        record = {
                            "tool": name,
                            "args": args,
                            "status": "running" if live else "interrupted",
                            "output": None,
                            "agent_id": agent_id,
                            "run_id": run_id,
                            "created": event["created"],
                            "verified": False,
                        }
                        records.append(record)
                        active_commands[key] = record
                    else:
                        report = _object(payload.get("result"))
                        record = active_commands.pop(key, None)
                        if record is None:
                            record = {
                                "tool": name,
                                "args": args,
                                "agent_id": agent_id,
                                "run_id": run_id,
                                "created": event["created"],
                            }
                            records.append(record)
                        success = (
                            report is not None and type(report.get("exit_code")) is int
                        )
                        record.update(
                            status="completed"
                            if success and report["exit_code"] == 0
                            else "failed",
                            output=report.get("output")
                            if report
                            else payload.get("result"),
                            exit_code=report.get("exit_code") if report else None,
                            verified=success,
                        )
                        if approval:
                            record["approval_id"] = approval["id"]
                if event["kind"] != "tool_result":
                    continue
                report = _object(payload.get("result"))
                if (
                    name == "read_file"
                    and report
                    and isinstance(report.get("sha256"), str)
                ):
                    try:
                        self._path(args.get("path"))
                    except (OSError, ValueError, TypeError):
                        continue
                    reads.append(
                        {
                            "path": args["path"],
                            "run_id": run_id,
                            "agent_id": agent_id,
                            "sha256": report["sha256"],
                            "created": event["created"],
                        }
                    )
                if name in EDIT_TOOLS and report:
                    receipt = self._recorded_file(
                        run_id, agent_id, name, args, report, approval, event["created"]
                    )
                    if receipt:
                        files[(run_id, receipt["origin"], receipt["path"])] = receipt
                        changes.append(
                            receipt
                            | {
                                "diff": approval["preview"].get("diff")
                                if approval
                                and isinstance(approval["preview"], dict)
                                and approval["preview"].get("after_sha256")
                                == receipt["sha256"]
                                else None,
                                "status": "executed",
                                "approval_id": approval["id"] if approval else None,
                            }
                        )
            verified_approvals = {
                change["approval_id"]
                for change in changes
                if change["run_id"] == run_id
            }
            for approval in approvals:
                key = _identity(
                    approval["agent_id"], approval["tool"], approval["args"]
                )
                status = approval["status"]
                state = (
                    "proposed"
                    if status in {"pending", "approved"}
                    else "running"
                    if status == "executing"
                    else "unconfirmed"
                    if status == "completed"
                    else status
                )
                if approval["tool"] in COMMAND_TOOLS:
                    record = next(
                        (
                            row
                            for row in records
                            if row.get("approval_id") == approval["id"]
                        ),
                        None,
                    )
                    if record is None:
                        record = active_commands.get(key)
                    if record is None:
                        record = {
                            "tool": approval["tool"],
                            "args": approval["args"],
                            "status": state,
                            "output": None,
                            "agent_id": approval["agent_id"],
                            "run_id": run_id,
                            "created": approval["created"],
                            "verified": False,
                        }
                        records.append(record)
                    record["approval_id"] = approval["id"]
                    if not record.get("verified") and status in {
                        "pending",
                        "approved",
                        "denied",
                        "expired",
                        "cancelled",
                        "stale",
                        "failed",
                        "interrupted",
                    }:
                        record["status"] = state
                if (
                    approval["tool"] in EDIT_TOOLS
                    and approval["id"] not in verified_approvals
                    and isinstance(approval["preview"], dict)
                ):
                    preview = approval["preview"]
                    try:
                        self._path(
                            preview.get("path"),
                            run_id=run_id
                            if approval["tool"] == "save_artifact"
                            else None,
                        )
                    except (OSError, ValueError, TypeError):
                        continue
                    changes.append(
                        {
                            "path": preview["path"],
                            "origin": "artifact"
                            if approval["tool"] == "save_artifact"
                            else "workspace",
                            "run_id": run_id,
                            "agent_id": approval["agent_id"],
                            "diff": preview.get("diff"),
                            "status": state,
                            "approval_id": approval["id"],
                            "created": approval["created"],
                            "verified": False,
                        }
                    )
            commands.extend(records)
        selected = sorted(files.values(), key=lambda row: row["created"])[-MAX_RECORDS:]
        for receipt, current in zip(
            selected,
            await asyncio.gather(
                *(asyncio.to_thread(self._current, receipt) for receipt in selected)
            ),
            strict=True,
        ):
            receipt.update(current)
        usage = dict(
            store.connection.execute(
                "SELECT COALESCE(SUM(r.input_tokens),0) AS input_tokens,COALESCE(SUM(r.output_tokens),0) AS output_tokens,COALESCE(SUM(r.uncertain_tokens),0) AS uncertain_tokens,COALESCE(SUM(r.reserved_tokens),0) AS reserved_tokens FROM runs r JOIN studio_chat_turns t ON t.run_id=r.run_id WHERE t.chat_id=?",
                (chat_id,),
            ).fetchone()
        )
        usage["total_tokens"] = sum(usage.values())
        changes.sort(key=lambda row: row["created"])
        commands.sort(key=lambda row: row["created"])
        for rows, field, limit in (
            (changes, "diff", 24_000),
            (commands, "output", 8_000),
        ):
            for row in rows:
                value = row.get(field)
                if isinstance(value, str) and len(value) > limit:
                    row[field] = (
                        value[: limit // 2]
                        + "\n[Inspection excerpt truncated]\n"
                        + value[-limit // 2 :]
                    )
                    row[field + "_truncated"] = True
        return {
            "agents": agents[-MAX_RECORDS:],
            "files": selected,
            "changes": changes[-MAX_RECORDS:],
            "commands": commands[-MAX_RECORDS:],
            "reads": reads[-MAX_RECORDS:],
            "usage": usage,
            "truncated": truncated
            or any(
                len(items) > MAX_RECORDS
                for items in (agents, files, changes, commands, reads)
            ),
            "file_source": "journal_receipts",
            "current_file_source": "current_disk",
        }

    async def _git(
        self, args: list[str], *, neutralize_filters: bool = True
    ) -> tuple[int, bytes, bool]:
        options = []
        if neutralize_filters:
            # Even status/diff can invoke configured clean filters. Inspect
            # option names only, then neutralize every filter command for this
            # read. Refuse inspection if that bounded catalog was truncated.
            _, configured, truncated = await self._git(
                ["config", "--null", "--name-only", "--get-regexp", "^filter\\."],
                neutralize_filters=False,
            )
            if truncated:
                return 1, b"", True
            for key in configured.split(b"\0"):
                name = key.decode("utf-8", "strict")
                if name and name.rsplit(".", 1)[-1].casefold() in {
                    "clean",
                    "smudge",
                    "process",
                    "required",
                }:
                    options.extend(
                        [
                            "-c",
                            name
                            + (
                                "=false"
                                if name.casefold().endswith(".required")
                                else "="
                            ),
                        ]
                    )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        environment.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "--literal-pathspecs",
                "--no-pager",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "diff.external=",
                "-c",
                "diff.renames=false",
                *options,
                "-C",
                str(self.service.workspace),
                *args,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError:
            return 127, b"", False
        data, truncated = bytearray(), False

        async def collect():
            nonlocal truncated
            while chunk := await process.stdout.read(8192):
                room = max(0, MAX_FILE_BYTES - len(data))
                data.extend(chunk[:room])
                truncated |= len(chunk) > room
            await process.wait()

        try:
            await asyncio.wait_for(collect(), 5)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()
            raise
        return process.returncode, bytes(data), truncated

    async def git_status(self) -> dict:
        code, data, truncated = await self._git(
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=normal",
                "--no-renames",
            ]
        )
        entries = []
        for record in data.split(b"\0")[:-1]:
            if len(record) < 4:
                continue
            path = record[3:].decode("utf-8", "replace")
            try:
                self._path(path)
            except (ValueError, OSError):
                continue
            entries.append(
                {"path": path, "index": chr(record[0]), "workspace": chr(record[1])}
            )
        return {
            "available": code == 0,
            "files": entries,
            "truncated": truncated,
            "source": "git_worktree",
        }

    async def dispatch(self, method: str, path: str, payload: dict) -> dict:
        parts = path.strip("/").split("/")
        if path == "/api/workspace/files" and method in {"GET", "POST"}:
            return await asyncio.to_thread(
                self._list, payload if method == "POST" else {}
            )
        if (
            method == "GET"
            and len(parts) >= 4
            and parts[:3] == ["api", "workspace", "file"]
        ):
            return await asyncio.to_thread(self._file, _decode("/".join(parts[3:])))
        if method == "GET" and path == "/api/workspace/git":
            return await self.git_status()
        if (
            method == "GET"
            and len(parts) >= 4
            and parts[:3] == ["api", "workspace", "diff"]
        ):
            name = _decode("/".join(parts[3:]))
            self._path(name)
            code, data, truncated = await self._git(
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=3",
                    "HEAD",
                    "--",
                    name,
                ]
            )
            return {
                "path": name,
                "diff": data.decode("utf-8", "replace"),
                "available": code == 0,
                "truncated": truncated,
                "source": "git_worktree",
            }
        if (
            method == "GET"
            and len(parts) == 4
            and parts[:2] == ["api", "chats"]
            and parts[3] == "developer"
        ):
            return await self.chat(parts[2])
        if (
            method == "POST"
            and len(parts) == 6
            and parts[:2] == ["api", "runs"]
            and parts[3] == "agents"
            and parts[5] in {"message", "cancel"}
        ):
            if parts[5] == "cancel":
                if payload:
                    raise ToolValidationError(
                        "agent cancellation does not accept fields"
                    )
                runtime = self.service.runtimes.get(parts[2])
                job = self.service.jobs.get(parts[2])
                run = self.service.store.run(parts[2])
                if (
                    runtime is None
                    or job is None
                    or job.done()
                    or run["status"] in SETTLED
                    or run["cancel_requested"]
                ):
                    raise ToolValidationError(
                        "agent cancellation requires an active mission"
                    )
                await runtime.cancel_agent(parts[2], parts[4])
                return {
                    "run_id": parts[2],
                    "agent_id": parts[4],
                    "cancel_requested": True,
                }
            if type(payload) is not dict or set(payload) != {"message"}:
                raise ToolValidationError("agent steering requires a message field")
            return await self.service.message(
                parts[2], {"message": payload["message"], "agent_id": parts[4]}
            )
        raise ValueError("unknown developer workspace route")
