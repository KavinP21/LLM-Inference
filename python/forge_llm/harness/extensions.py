"""Operator-selected local skills and bounded MCP stdio tools.

Skills are read as data; neither discovery nor loading executes their scripts.
MCP subprocesses run operator-provided argv, never commands supplied by a model.
Supported tool dialects: 2026-07-28 (discover/per-request metadata), and legacy
2025-11-25, 2025-06-18, 2025-03-26 (initialize). This adapter deliberately omits
remote transports, sampling, elicitation, subscriptions, and automatic retries.
See https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio
and https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import signal
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..agents.native_tools import _check_schema, _validate
from ..agents.protocol import _unique_object
from ..agents.tools import ToolValidationError

_SKILL_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_PRIVATE = {".git", ".env", ".ssh", ".aws", "credentials", "secrets"}
_MODERN = "2026-07-28"
_LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26")


def _bounded_integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ToolValidationError(f"{name} must be an integer in [{low},{high}]")
    return value


def _safe_read(root: Path, relative: str, limit: int, *, prefix: bool = False) -> bytes:
    """Descriptor-relative no-follow traversal also rejects FIFO/device reads."""
    path = PurePosixPath(relative)
    if (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or path.is_absolute()
        or any(p in {"", ".", ".."} for p in relative.split("/"))
        or any(p in _PRIVATE or p.startswith(".env.") for p in path.parts)
    ):
        raise ToolValidationError("skill resources require a public relative path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory = os.open(root, flags | os.O_DIRECTORY)
    except OSError as exc:
        raise ToolValidationError(
            "skill directory is unavailable or uses a symlink"
        ) from exc
    try:
        for part in path.parts[:-1]:
            next_directory = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(path.parts[-1], flags | os.O_NONBLOCK, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ToolValidationError("skill resources must be regular files")
            if not prefix and info.st_size > limit:
                raise ToolValidationError("skill resource exceeds the byte limit")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(limit if prefix else limit + 1)
            if not prefix and len(data) > limit:
                raise ToolValidationError("skill resource exceeds the byte limit")
            return data
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ToolValidationError(
            "skill resource is unavailable or uses a symlink"
        ) from exc
    finally:
        os.close(directory)


def _skill_metadata(data: bytes) -> tuple[str, str]:
    """Read scalar name/description YAML fields without a YAML evaluator.

    Other fields are ignored. Quoted scalars and literal/folded descriptions
    cover the portable SKILL.md catalog fields; aliases/tags are not evaluated.
    """
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ToolValidationError("SKILL.md requires YAML front matter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ToolValidationError(
            "skill front matter exceeds the metadata limit"
        ) from exc
    fields: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        index += 1
        match = re.match(r"^(name|description):\s*(.*)$", line)
        if not match:
            continue
        key, value = match.groups()
        if key in fields:
            raise ToolValidationError("duplicate skill metadata field")
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            block = []
            while index < end and (not lines[index] or lines[index][0].isspace()):
                block.append(lines[index].strip())
                index += 1
            value = ("\n" if value[0] == "|" else " ").join(block).strip()
        elif value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ToolValidationError("invalid quoted skill metadata") from exc
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        else:
            value = value.split(" #", 1)[0].strip()
        if not isinstance(value, str) or not value.strip():
            raise ToolValidationError("skill name and description must be text")
        fields[key] = value
    name, description = fields.get("name", ""), fields.get("description", "")
    if not _SKILL_NAME.fullmatch(name) or not 1 <= len(description) <= 2048:
        raise ToolValidationError("skill requires a bounded name and description")
    return name, description


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    directory: Path
    source: Path

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "source": str(self.source),
        }


class SkillRegistry:
    """Discover only configured roots or their direct SKILL.md child folders."""

    replay_safe_tools = frozenset({"skill_list", "skill_load", "skill_read"})
    schemas = {
        "skill_list": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "skill_load": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "skill_read": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "path": {"type": "string"}},
            "required": ["name", "path"],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        max_skills: int = 128,
        max_file_bytes: int = 128_000,
        max_metadata_bytes: int = 16_384,
    ) -> None:
        if isinstance(roots, (str, bytes)) or len(roots) > 16:
            raise ToolValidationError("skills require at most 16 explicit roots")
        self.roots = tuple(Path(root).resolve(strict=True) for root in roots)
        if any(not root.is_dir() for root in self.roots):
            raise ToolValidationError("skill roots must be directories")
        self.max_skills = _bounded_integer(max_skills, 1, 512, "max_skills")
        self.max_file_bytes = _bounded_integer(
            max_file_bytes, 1, 1_000_000, "max_file_bytes"
        )
        self.max_metadata_bytes = _bounded_integer(
            max_metadata_bytes, 128, 64_000, "max_metadata_bytes"
        )
        self.records: dict[str, SkillRecord] = {}
        self.warnings: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        records: dict[str, SkillRecord] = {}
        warnings: list[str] = []
        for root in self.roots:
            if (root / "SKILL.md").exists():
                candidates = [root]
            else:
                with os.scandir(root) as entries:
                    selected = []
                    for index, entry in enumerate(entries):
                        if index >= 2048:
                            raise ToolValidationError(
                                "skill root exceeds 2048 directory entries"
                            )
                        if entry.is_dir(follow_symlinks=False):
                            selected.append(Path(entry.path))
                candidates = sorted(selected)
            for directory in candidates:
                if not (directory / "SKILL.md").exists():
                    continue
                if len(records) >= self.max_skills:
                    raise ToolValidationError("skill catalog exceeds the skill limit")
                try:
                    data = _safe_read(
                        root,
                        (directory / "SKILL.md").relative_to(root).as_posix(),
                        self.max_metadata_bytes,
                        prefix=True,
                    )
                    name, description = _skill_metadata(data)
                except ValueError as exc:
                    warnings.append(f"{directory.name}: {exc}")
                    continue
                if name in records:
                    raise ToolValidationError(f"duplicate skill name: {name}")
                records[name] = SkillRecord(
                    name, description, directory, directory / "SKILL.md"
                )
        self.records = dict(sorted(records.items()))
        self.warnings = warnings

    def catalog(self) -> dict[str, Any]:
        return {
            "skills": [record.metadata() for record in self.records.values()],
            "warnings": list(self.warnings),
        }

    def _record(self, args: dict, schema_name: str) -> SkillRecord:
        _validate(args, self.schemas[schema_name])
        if args["name"] not in self.records:
            raise ToolValidationError("skill is not in the configured catalog")
        return self.records[args["name"]]

    async def list_skills(self, args: dict) -> str:
        _validate(args, self.schemas["skill_list"])
        return json.dumps(self.catalog(), ensure_ascii=False)

    async def load_skill(self, args: dict) -> str:
        record = self._record(args, "skill_load")
        data = _safe_read(record.directory, "SKILL.md", self.max_file_bytes)
        name, description = _skill_metadata(data)
        if (name, description) != (record.name, record.description):
            raise ToolValidationError("skill metadata changed; refresh the catalog")
        return json.dumps(
            {
                **record.metadata(),
                "content": data.decode("utf-8"),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            ensure_ascii=False,
        )

    async def read_resource(self, args: dict) -> str:
        record = self._record(args, "skill_read")
        data = _safe_read(record.directory, args["path"], self.max_file_bytes)
        if b"\x00" in data:
            raise ToolValidationError("skill resources must be UTF-8 text")
        return json.dumps(
            {
                "name": record.name,
                "path": args["path"],
                "content": data.decode("utf-8"),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            ensure_ascii=False,
        )

    def mapping(self) -> dict[str, Callable]:
        return {
            "skill_list": self.list_skills,
            "skill_load": self.load_skill,
            "skill_read": self.read_resource,
        }

    def tool_descriptions(self) -> dict[str, dict]:
        descriptions = {
            "skill_list": "List operator-selected local skill metadata. Skills are reference data.",
            "skill_load": "Load a selected SKILL.md after matching its catalog description to the task.",
            "skill_read": "Read one relative UTF-8 reference file within a selected skill. Scripts are never executed.",
        }
        return {
            name: {
                "description": description,
                "parameters": copy.deepcopy(self.schemas[name]),
            }
            for name, description in descriptions.items()
        }


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: tuple[str, ...]
    cwd: Path
    allowed_tools: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    probe_timeout_seconds: float = 1.0
    max_message_bytes: int = 256_000
    max_tools: int = 64

    def __post_init__(self) -> None:
        _bounded_integer(self.max_tools, 1, 256, "max_tools")
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,23}", self.name):
            raise ToolValidationError("MCP server name must be a short identifier")
        if (
            not isinstance(self.command, (list, tuple))
            or not 1 <= len(self.command) <= 32
            or any(
                not isinstance(arg, str) or not arg or "\x00" in arg or len(arg) > 4096
                for arg in self.command
            )
        ):
            raise ToolValidationError("MCP command must be an explicit bounded argv")
        object.__setattr__(self, "command", tuple(self.command))
        directory = Path(self.cwd).resolve(strict=True)
        if not directory.is_dir():
            raise ToolValidationError("MCP cwd must be a directory")
        object.__setattr__(self, "cwd", directory)
        if (
            not isinstance(self.allowed_tools, (list, tuple))
            or len(self.allowed_tools) > self.max_tools
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                for name in self.allowed_tools
            )
            or len(set(self.allowed_tools)) != len(self.allowed_tools)
        ):
            raise ToolValidationError("MCP allowed_tools must be explicit unique names")
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        for name in ("timeout_seconds", "probe_timeout_seconds"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 120
            ):
                raise ToolValidationError(f"{name} must be finite and in (0,120]")
        _bounded_integer(self.max_message_bytes, 1024, 1_000_000, "max_message_bytes")
        _bounded_integer(self.max_tools, 1, 256, "max_tools")


class MCPError(RuntimeError):
    """Server RPC error; error text/data are not logged or interpolated."""

    def __init__(self, code: int, data: Any = None):
        super().__init__(f"MCP request rejected (code {code})")
        self.code, self.data = code, data


class MCPStdioClient:
    """Serialized bounded JSON-RPC calls to a single owned local process.

    Server annotations never grant replay safety or permissions. Operators must
    approve/grant the selected tool names through the harness's normal policy.
    Any timeout, cancellation, or malformed response closes the owned process.
    """

    replay_safe_tools = frozenset()

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.protocol_version: str | None = None
        self.schemas: dict[str, dict] = {}
        self._descriptions: dict[str, dict] = {}
        self._remote_names: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._closed = False

    async def __aenter__(self) -> MCPStdioClient:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _metadata(self) -> dict:
        return {
            "io.modelcontextprotocol/protocolVersion": _MODERN,
            "io.modelcontextprotocol/clientInfo": {
                "name": "forge-studio",
                "version": "0.1.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }

    async def start(self) -> None:
        if self.process is not None or self._closed:
            raise RuntimeError("MCP client has already been started or closed")
        # No shell and no ambient credentials. stderr is deliberately discarded.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "TMPDIR",
                "TEMP",
                "TMP",
                "SYSTEMROOT",
                "WINDIR",
            }
        }
        environment.setdefault("PATH", os.defpath)
        self.process = await asyncio.create_subprocess_exec(
            *self.config.command,
            cwd=self.config.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=self.config.max_message_bytes + 1,
            start_new_session=os.name == "posix",
        )
        try:
            try:
                result = await self._request(
                    "server/discover",
                    {"_meta": self._metadata()},
                    timeout=self.config.probe_timeout_seconds,
                    probe=True,
                )
            except MCPError as exc:
                if exc.code in {-32020, -32021, -32022}:
                    raise RuntimeError(
                        "modern MCP server has no supported protocol version"
                    ) from exc
                result = None
            except TimeoutError:
                result = None
            if result is None:
                result = await self._request(
                    "initialize",
                    {
                        "protocolVersion": _LEGACY[0],
                        "capabilities": {},
                        "clientInfo": {"name": "forge-studio", "version": "0.1.0"},
                    },
                )
                if result.get("protocolVersion") not in _LEGACY:
                    raise RuntimeError("unsupported legacy MCP protocol version")
                self.protocol_version = result["protocolVersion"]
                await self._send(
                    {"jsonrpc": "2.0", "method": "notifications/initialized"}
                )
            else:
                if _MODERN not in result.get("supportedVersions", []):
                    raise RuntimeError(
                        "modern MCP server has no supported protocol version"
                    )
                self.protocol_version = _MODERN
            if (
                not isinstance(result.get("capabilities"), dict)
                or "tools" not in result["capabilities"]
            ):
                raise RuntimeError("MCP server did not advertise tools")
            await self._discover_tools()
        except BaseException:
            await self.close()
            raise

    async def _send(self, message: dict) -> None:
        if self.process is None or self.process.stdin is None or self._closed:
            raise RuntimeError("MCP client is unavailable")
        data = (
            json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        if len(data) > self.config.max_message_bytes:
            raise ToolValidationError("MCP request exceeds the message limit")
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def _request(
        self,
        method: str,
        params: dict,
        *,
        timeout: float | None = None,
        probe: bool = False,
    ) -> dict:
        async with self._lock:
            identifier = self._next_id
            self._next_id += 1
            parameters = copy.deepcopy(params)
            if self.protocol_version == _MODERN:
                parameters["_meta"] = self._metadata()

            async def exchange() -> dict:
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": identifier,
                        "method": method,
                        "params": parameters,
                    }
                )
                assert self.process is not None and self.process.stdout is not None
                for _ in range(128):
                    try:
                        line = await self.process.stdout.readline()
                    except ValueError as exc:
                        raise RuntimeError(
                            "MCP server exceeded its message limit"
                        ) from exc
                    if not line or len(line) > self.config.max_message_bytes:
                        raise RuntimeError(
                            "MCP server closed or exceeded its message limit"
                        )
                    try:
                        message = json.loads(
                            line,
                            object_pairs_hook=_unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ValueError("nonfinite JSON")
                            ),
                        )
                    except (ValueError, UnicodeError, RecursionError) as exc:
                        raise RuntimeError("MCP server sent malformed JSON") from exc
                    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                        raise RuntimeError(
                            "MCP server sent an invalid JSON-RPC message"
                        )
                    if "method" in message:
                        if (
                            not isinstance(message["method"], str)
                            or "result" in message
                            or "error" in message
                        ):
                            raise RuntimeError(
                                "MCP server sent an invalid notification or request"
                            )
                        if "id" in message:
                            if type(message["id"]) not in (str, int):
                                raise RuntimeError(
                                    "MCP server sent an invalid request id"
                                )
                            # No optional client capabilities were advertised.
                            if self.protocol_version == _MODERN:
                                raise RuntimeError("modern MCP server sent a request")
                            await self._send(
                                {
                                    "jsonrpc": "2.0",
                                    "id": message["id"],
                                    "error": {
                                        "code": -32601,
                                        "message": "Client capability unavailable",
                                    },
                                }
                            )
                        continue
                    if (
                        type(message.get("id")) is not int
                        or message["id"] != identifier
                    ):
                        # A timed-out discover probe may respond during fallback.
                        if (
                            type(message.get("id")) is int
                            and message["id"] < identifier
                        ):
                            continue
                        raise RuntimeError("MCP response has an unexpected request id")
                    if ("result" in message) == ("error" in message):
                        raise RuntimeError(
                            "MCP response requires exactly one result or error"
                        )
                    if "error" in message:
                        error = message["error"]
                        if (
                            not isinstance(error, dict)
                            or type(error.get("code")) is not int
                        ):
                            raise RuntimeError("MCP server sent an invalid error")
                        raise MCPError(error["code"], error.get("data"))
                    if not isinstance(message["result"], dict):
                        raise RuntimeError("MCP result must be an object")
                    return message["result"]
                raise RuntimeError("MCP server exceeded the notification limit")

            try:
                return await asyncio.wait_for(
                    exchange(), timeout or self.config.timeout_seconds
                )
            except MCPError:
                raise
            except TimeoutError:
                try:
                    await asyncio.wait_for(
                        self._send(
                            {
                                "jsonrpc": "2.0",
                                "method": "notifications/cancelled",
                                "params": {
                                    "requestId": identifier,
                                    "reason": "Request timed out",
                                },
                            }
                        ),
                        0.2,
                    )
                except (OSError, RuntimeError, TimeoutError):
                    pass
                if not probe:
                    await self.close()
                raise
            except BaseException:
                await self.close()
                raise

    async def _discover_tools(self) -> None:
        cursor: str | None = None
        seen: set[str] = set()
        remote: dict[str, dict] = {}
        for _ in range(16):
            result = await self._request(
                "tools/list", {"cursor": cursor} if cursor else {}
            )
            if result.get("resultType", "complete") != "complete" or not isinstance(
                result.get("tools"), list
            ):
                raise RuntimeError("MCP tools/list requires a complete tool list")
            for tool in result["tools"]:
                if (
                    not isinstance(tool, dict)
                    or not isinstance(tool.get("name"), str)
                    or tool["name"] in remote
                ):
                    raise RuntimeError("MCP tool names must be unique strings")
                remote[tool["name"]] = tool
                if len(remote) > self.config.max_tools:
                    raise RuntimeError("MCP tool catalog exceeds the tool limit")
            cursor = result.get("nextCursor")
            if cursor is None:
                break
            if (
                not isinstance(cursor, str)
                or not cursor
                or len(cursor) > 4096
                or cursor in seen
            ):
                raise RuntimeError("MCP tool pagination cursor is invalid")
            seen.add(cursor)
        else:
            raise RuntimeError("MCP tool pagination exceeds the page limit")
        for name in self.config.allowed_tools:
            if name not in remote:
                raise ToolValidationError(
                    "selected MCP tool is missing from the server catalog"
                )
            tool = remote[name]
            parameters = copy.deepcopy(tool.get("inputSchema"))
            _check_schema(parameters)
            if parameters.get("type") != "object":
                raise ToolValidationError("MCP tool inputs must be object schemas")
            parameters["additionalProperties"] = False
            digest = hashlib.sha256(name.encode()).hexdigest()[:8]
            alias = f"mcp_{self.config.name}_{re.sub('[^A-Za-z0-9_-]', '_', name)[:24]}_{digest}"
            self.schemas[alias] = parameters
            self._remote_names[alias] = name
            description = tool.get("description", name)
            if not isinstance(description, str) or len(description) > 4096:
                raise ToolValidationError("MCP description exceeds the text limit")
            self._descriptions[alias] = {
                "description": f"Operator-selected {self.config.name} MCP tool: {description}",
                "parameters": copy.deepcopy(parameters),
            }

    async def call_tool(self, alias: str, args: dict) -> str:
        if alias not in self._remote_names:
            raise ToolValidationError("MCP tool is not in the selected allowlist")
        try:
            _validate(args, self.schemas[alias])
        except ValueError as exc:
            raise ToolValidationError(str(exc)) from exc
        result = await self._request(
            "tools/call", {"name": self._remote_names[alias], "arguments": args}
        )
        if result.get("resultType", "complete") != "complete":
            raise RuntimeError(
                "MCP input-required results need operator interaction; unsupported by this adapter"
            )
        if (
            not isinstance(result.get("content"), list)
            or type(result.get("isError", False)) is not bool
        ):
            raise RuntimeError("MCP tool result requires content and a boolean isError")
        return json.dumps(result, ensure_ascii=False, allow_nan=False)

    def mapping(self) -> dict[str, Callable]:
        def bind(alias: str) -> Callable:
            async def invoke(args: dict) -> str:
                return await self.call_tool(alias, args)

            return invoke

        return {alias: bind(alias) for alias in self._remote_names}

    def tool_descriptions(self) -> dict[str, dict]:
        return copy.deepcopy(self._descriptions)

    @property
    def mutating_tools(self) -> frozenset[str]:
        return frozenset(self._remote_names)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 0.2)
        except TimeoutError:
            pass
        # Signal the owned group even if its leader exited, so descendants cannot
        # keep a pipe open or survive a timeout on POSIX.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                elif process.returncode is None:
                    process.terminate() if sig == signal.SIGTERM else process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 0.2)
            except TimeoutError:
                continue
            if os.name != "posix":
                break
