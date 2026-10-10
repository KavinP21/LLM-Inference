"""Bounded workspace tools, with explicit opt-in writes and test execution.

The model supplies arguments, never a shell command. File reads expose hashes so
an optional edit can require the version the agent actually inspected.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import signal
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, ClassVar

from .context import tool_actor

_IGNORED = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "build",
    "models",
    "benchmark-results",
    "results",
}
_PRIVATE = {".env", ".aws", ".ssh", ".codex", ".agents", "credentials", "secrets"}


class ToolValidationError(ValueError):
    """Rejected before any mutation or process execution."""

    work_started = False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _arguments(
    args: dict, required: set[str], optional: set[str] = frozenset()
) -> None:
    if (
        not isinstance(args, dict)
        or not required <= args.keys()
        or args.keys() - required - optional
    ):
        raise ToolValidationError(
            f"expected arguments {sorted(required)}; optional {sorted(optional)}"
        )


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ToolValidationError(f"{name} must be an integer between {low} and {high}")
    return value


class WorkspaceTools:
    """Tools confined to one operator-selected directory and an artifact directory."""

    descriptions: ClassVar[dict[str, str]] = {
        "list_files": 'List workspace files. Arguments: {"path":".","limit":100}.',
        "read_file": 'Read UTF-8 lines and SHA256. Arguments: {"path":"file","start_line":1,"max_lines":120}.',
        "search": 'Search literal text in workspace files. Arguments: {"query":"text","path":".","limit":30}.',
        "save_artifact": 'Save a new reviewable artifact. Arguments: {"path":"report.md","content":"..."}. Existing files cannot be overwritten.',
        "write_file": 'Replace a COMPLETE file after read_file. Last read version is checked automatically; omit hashes and receipts. Never write partial excerpts. Arguments: {"path":"file","content":"..."}. Explicit read_receipt or expected_sha256 remains available through the tool API.',
        "replace_text": 'Replace one unique substring after read_file. Include leading spaces in code matches and indent EVERY replacement line; whitespace is never added. Last read version is checked automatically; omit hashes and receipts. Arguments: {"path":"file","old_text":"exact unique text","new_text":"replacement"}. Explicit read_receipt or expected_sha256 remains available through the tool API.',
        "edit_lines": 'Preferred for multiline source edits: replace an inclusive whole-line range from read_file. Provide complete replacement lines with exact indentation and no displayed line-number labels. Unselected lines stay unchanged; a missing trailing newline preserves the original range boundary. Last read version is checked automatically; omit hashes and receipts. Arguments: {"path":"file","start_line":1,"end_line":3,"content":"complete replacement lines"}.',
        "run_tests": 'Run configured Python pytest on explicit workspace test files. Arguments: {"paths":["tests/test_example.py"]}.',
    }
    replay_safe_tools = frozenset({"list_files", "read_file", "search"})
    schemas: ClassVar[dict[str, dict]] = {
        "list_files": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "additionalProperties": False,
        },
        "read_file": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "save_artifact": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "write_file": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "read_receipt": {"type": "string"},
                "expected_sha256": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "replace_text": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {
                    "type": "string",
                    "description": "Exact unique file text, including leading spaces; exclude displayed line-number labels.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Exact replacement with correct indentation on every line. No whitespace is added automatically.",
                },
                "read_receipt": {"type": "string"},
                "expected_sha256": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        "edit_lines": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1_000_000,
                    "description": "First existing line to replace, inclusive; use read_file line numbers.",
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1_000_000,
                    "description": "Last existing line to replace, inclusive; must be at least start_line.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete replacement lines, exactly indented; no displayed line-number labels. Empty text deletes the selected lines.",
                },
                "read_receipt": {"type": "string"},
                "expected_sha256": {"type": "string"},
            },
            "required": ["path", "start_line", "end_line", "content"],
            "additionalProperties": False,
        },
        "run_tests": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 20,
                }
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
    }

    def tool_descriptions(self) -> dict:
        result = {}
        for name in self.mapping():
            parameters = copy.deepcopy(self.schemas[name])
            if name in {"write_file", "replace_text", "edit_lines"}:
                # Native models use the automatically recorded per-agent read
                # version. Keep explicit version selection in the direct API,
                # without asking the model to copy opaque hashes or receipts.
                parameters["properties"].pop("read_receipt", None)
                parameters["properties"].pop("expected_sha256", None)
            result[name] = {
                "description": self.descriptions[name].split("Arguments:")[0].strip(),
                "parameters": parameters,
            }
        return result

    def __init__(
        self,
        workspace: str | Path,
        artifacts: str | Path,
        *,
        allow_write: bool = False,
        allow_tests: bool = False,
        python: str | Path = sys.executable,
        max_file_bytes: int = 256_000,
        max_scan_files: int = 10_000,
        test_timeout: float = 60.0,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ToolValidationError("workspace must be a directory")
        self.artifacts = Path(artifacts).resolve()
        self.allow_write, self.allow_tests = allow_write, allow_tests
        # Resolving a venv Python symlink selects the base interpreter and loses
        # its packages. Validate existence while retaining the launcher path.
        executable = Path(python).absolute()
        if not executable.is_file():
            raise ToolValidationError("configured Python executable does not exist")
        self.python = str(executable)
        self.max_file_bytes = _integer(max_file_bytes, 1, 4_000_000, "max_file_bytes")
        self.max_scan_files = _integer(max_scan_files, 1, 100_000, "max_scan_files")
        if not 0 < test_timeout <= 600:
            raise ToolValidationError("test_timeout must be in (0,600]")
        self.test_timeout = test_timeout
        self._mutation_lock = asyncio.Lock()
        self._test_lock = asyncio.Lock()
        self._receipt_lock = threading.Lock()
        self._receipts: dict[str, tuple[tuple[str, str], str, str]] = {}
        self._observed: dict[tuple[tuple[str, str], str], str] = {}
        self._next_receipt = 1
        self._protection_lock = threading.Lock()
        self._protected_canonical: set[Path] = set()
        self._protected_lexical: set[Path] = set()

    def _protection_selection(self, paths: list[str]) -> list[tuple[str, Path]]:
        if not isinstance(paths, list):
            raise TypeError("protected paths must be an explicit list")
        if len(paths) > 1000:
            raise ToolValidationError("select at most 1000 protected files")
        selected = [(value, self._path(value)) for value in paths]
        if len({value for value, _ in selected}) != len(selected):
            raise ToolValidationError(
                "protected file selection contains duplicate paths"
            )
        return selected

    def protected_file_hashes(self, paths: list[str]) -> dict[str, str]:
        """Hash explicitly selected bounded regular files without executing them."""
        return {
            value: hashlib.sha256(self._safe_read(path)).hexdigest()
            for value, path in self._protection_selection(paths)
        }

    def protect_files(self, paths: list[str]) -> dict[str, str]:
        """Prevent workspace tools from mutating operator-selected files.

        Registration is atomic after all selected reads succeed. This confines
        writes through this tool object; it is not an operating-system sandbox.
        """
        selected = self._protection_selection(paths)
        hashes = {
            value: hashlib.sha256(self._safe_read(path)).hexdigest()
            for value, path in selected
        }
        with self._protection_lock:
            self._protected_canonical.update(path for _, path in selected)
            self._protected_lexical.update(
                self.workspace / value for value, _ in selected
            )
        return hashes

    def _assert_mutable(self, path: Path, lexical: Path) -> None:
        with self._protection_lock:
            protected = (
                path in self._protected_canonical or lexical in self._protected_lexical
            )
        if protected:
            raise ToolValidationError(
                "operator-protected file cannot be edited with workspace tools"
            )

    def _record_read(self, path: Path, digest: str) -> str:
        actor = tool_actor.get()
        with self._receipt_lock:
            receipt = f"read-{self._next_receipt}"
            self._next_receipt += 1
            if len(self._receipts) == 4096:
                self._receipts.clear()
                self._observed.clear()
            self._receipts[receipt] = (actor, str(path), digest)
            self._observed[(actor, str(path))] = digest
        return receipt

    def _expected_version(self, path: Path, args: dict) -> str:
        if "expected_sha256" in args and "read_receipt" not in args:
            if not isinstance(args["expected_sha256"], str):
                raise ToolValidationError("expected_sha256 must be text")
            return args["expected_sha256"]
        actor = tool_actor.get()
        with self._receipt_lock:
            if "read_receipt" in args:
                receipt = (
                    self._receipts.get(args["read_receipt"])
                    if isinstance(args["read_receipt"], str)
                    else None
                )
                if receipt is None or receipt[:2] != (actor, str(path)):
                    raise ToolValidationError(
                        "receipt is unavailable for this agent/file; read_file again"
                    )
                if "expected_sha256" in args and args["expected_sha256"] != receipt[2]:
                    raise ToolValidationError(
                        "receipt and expected_sha256 refer to different file versions"
                    )
                return receipt[2]
            observed = self._observed.get((actor, str(path)))
        if observed is not None:
            return observed
        if not path.exists():
            return "missing"
        raise ToolValidationError(
            "read_file before editing this file; no version was observed by this agent"
        )

    def mapping(self) -> dict:
        names = ["list_files", "read_file", "search", "save_artifact"]
        if self.allow_write:
            names.extend(["write_file", "replace_text", "edit_lines"])
        if self.allow_tests:
            names.append("run_tests")
        return {name: getattr(self, name) for name in names}

    def _path(self, value: Any, *, artifact: bool = False) -> Path:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ToolValidationError("path must be a nonempty relative string")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ToolValidationError("path must stay inside the selected directory")
        if any(
            p in _PRIVATE or p.startswith(".env.") or p == ".git"
            for p in relative.parts
        ):
            raise ToolValidationError("private or Git metadata path is unavailable")
        base = self.artifacts if artifact else self.workspace
        result = (base / relative).resolve()
        if not result.is_relative_to(base):
            raise ToolValidationError("symlink escapes the selected directory")
        if any(
            p in _PRIVATE or p.startswith(".env.") or p == ".git"
            for p in result.relative_to(base).parts
        ):
            raise ToolValidationError("symlink resolves to private or Git metadata")
        return result

    def _files(self, path: Path):
        if path.is_file():
            yield path
            return
        count = 0
        for directory, dirs, names in os.walk(path, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in _IGNORED | _PRIVATE and not d.startswith(".")
            )
            for name in sorted(names):
                if name.startswith("."):
                    continue
                candidate = Path(directory) / name
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                count += 1
                if count > self.max_scan_files:
                    raise ToolValidationError(
                        "scan file limit reached; narrow the search path"
                    )
                yield candidate

    def _safe_read(self, path: Path) -> bytes:
        flags = (
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ToolValidationError("only regular files are available")
            data = stream.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes:
            raise ToolValidationError("file exceeds read limit; choose a smaller file")
        return data

    async def list_files(self, args: dict) -> str:
        return await asyncio.to_thread(self._list_files, args)

    def _list_files(self, args: dict) -> str:
        _arguments(args, set(), {"path", "limit"})
        path = self._path(args.get("path", "."))
        if not path.exists():
            raise FileNotFoundError(args.get("path", "."))
        limit = _integer(args.get("limit", 100), 1, 1000, "limit")
        files = []
        for candidate in self._files(path):
            if len(files) == limit:
                return _json({"files": files, "truncated": True})
            files.append(str(candidate.relative_to(self.workspace)))
        return _json({"files": files, "truncated": False})

    async def read_file(self, args: dict) -> str:
        result = json.loads(await asyncio.to_thread(self._read_file, args))
        # Only a successfully returned read grants a version. A timed-out
        # background read must not silently authorize a later write.
        result["read_receipt"] = self._record_read(
            self._path(args["path"]), result["sha256"]
        )
        return _json(result)

    def _read_file(self, args: dict) -> str:
        _arguments(args, {"path"}, {"start_line", "max_lines"})
        path = self._path(args["path"])
        data = self._safe_read(path)
        content = data.decode("utf-8")
        if "\x00" in content:
            raise ToolValidationError("binary files are unavailable")
        start = _integer(args.get("start_line", 1), 1, 1_000_000, "start_line")
        count = _integer(args.get("max_lines", 120), 1, 1000, "max_lines")
        lines = content.splitlines()
        selected = lines[start - 1 : start - 1 + count]
        digest = hashlib.sha256(data).hexdigest()
        return _json(
            {
                "path": args["path"],
                "sha256": digest,
                "total_lines": len(lines),
                "start_line": start,
                "content": "\n".join(
                    f"{start + i}: {line}" for i, line in enumerate(selected)
                ),
                "truncated": start - 1 + count < len(lines),
            }
        )

    async def search(self, args: dict) -> str:
        return await asyncio.to_thread(self._search, args)

    def _search(self, args: dict) -> str:
        _arguments(args, {"query"}, {"path", "limit"})
        query = args["query"]
        if not isinstance(query, str) or not query or len(query) > 1000:
            raise ToolValidationError("query must contain 1 to 1000 characters")
        limit = _integer(args.get("limit", 30), 1, 200, "limit")
        matches = []
        scanned_bytes = 0
        for path in self._files(self._path(args.get("path", "."))):
            if path.stat().st_size > self.max_file_bytes:
                continue
            scanned_bytes += path.stat().st_size
            if scanned_bytes > 16_000_000:
                raise ToolValidationError(
                    "scan byte limit reached; narrow the search path"
                )
            try:
                data = self._safe_read(path).decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "\x00" in data:
                continue
            for line, content in enumerate(data.splitlines(), 1):
                if query in content:
                    if len(matches) == limit:
                        return _json({"matches": matches, "truncated": True})
                    matches.append(
                        {
                            "path": str(path.relative_to(self.workspace)),
                            "line": line,
                            "content": content[:2000],
                        }
                    )
        return _json({"matches": matches, "truncated": False})

    async def _write(self, args: dict, *, artifact: bool) -> str:
        _arguments(
            args,
            {"path", "content"},
            set() if artifact else {"expected_sha256", "read_receipt"},
        )
        if not artifact and not self.allow_write:
            raise PermissionError("workspace writes are disabled")
        content = args["content"]
        if (
            not isinstance(content, str)
            or len(content.encode("utf-8")) > self.max_file_bytes
        ):
            raise ToolValidationError("content exceeds write limit or is not text")
        path = self._path(args["path"], artifact=artifact)
        lexical = (self.artifacts if artifact else self.workspace) / args["path"]
        self._assert_mutable(path, lexical)
        if not artifact and path.suffix == ".py":
            try:
                # Parsing an AST accepts return/yield/await outside a function.
                # Compilation validates their enclosing scopes too, without
                # importing the file or executing any user code.
                compile(content, args["path"], "exec", dont_inherit=True)
            except SyntaxError as exc:
                lines = content.splitlines()
                source_line = exc.text or (
                    lines[exc.lineno - 1]
                    if exc.lineno is not None and 0 < exc.lineno <= len(lines)
                    else ""
                )
                raise ToolValidationError(
                    f"Python syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}; "
                    f"source: {source_line.rstrip()!r}"
                ) from exc
        if path == (self.artifacts if artifact else self.workspace):
            raise ToolValidationError("cannot replace directory")
        async with self._mutation_lock:
            self._assert_mutable(path, lexical)
            expected = "missing" if artifact else self._expected_version(path, args)
            if not isinstance(expected, str):
                raise ToolValidationError("expected_sha256 must be a string")
            if path.exists():
                if path.stat().st_size > self.max_file_bytes:
                    raise ToolValidationError("existing file exceeds write limit")
                actual = hashlib.sha256(self._safe_read(path)).hexdigest()
            else:
                actual = "missing"
            if expected != actual:
                raise ToolValidationError(
                    "file version changed; read it again before editing"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".forge-write-", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content.encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return _json(
            {
                "path": str(path),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "bytes": len(content.encode()),
            }
        )

    async def write_file(self, args: dict) -> str:
        return await self._write(args, artifact=False)

    async def replace_text(self, args: dict) -> str:
        _arguments(
            args, {"path", "old_text", "new_text"}, {"expected_sha256", "read_receipt"}
        )
        if not self.allow_write:
            raise PermissionError("workspace writes are disabled")
        old, new = args["old_text"], args["new_text"]
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ToolValidationError(
                "old_text must be nonempty text and new_text must be text"
            )
        path = self._path(args["path"])
        self._assert_mutable(path, self.workspace / args["path"])
        if not path.is_file() or path.stat().st_size > self.max_file_bytes:
            raise ToolValidationError(
                "edit requires an existing file within the size limit"
            )
        data = await asyncio.to_thread(self._safe_read, path)
        expected = self._expected_version(path, args)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ToolValidationError(
                "file version changed; read it again before editing"
            )
        content = data.decode("utf-8")
        if content.count(old) != 1:
            raise ToolValidationError(
                "old_text must match exactly once; read a narrower unique section"
            )
        return await self._write(
            {
                "path": args["path"],
                "content": content.replace(old, new, 1),
                "expected_sha256": expected,
            },
            artifact=False,
        )

    async def save_artifact(self, args: dict) -> str:
        return await self._write(args, artifact=True)

    async def edit_lines(self, args: dict) -> str:
        """Replace an existing inclusive line range without changing its neighbors.

        Content and indentation are never inferred. Caller-provided internal
        line endings remain exact. If nonempty replacement content omits its
        final line ending, the selected range's original final ending is kept.
        Empty content deletes the selected whole lines, including their endings.
        """
        _arguments(
            args,
            {"path", "start_line", "end_line", "content"},
            {"expected_sha256", "read_receipt"},
        )
        if not self.allow_write:
            raise PermissionError("workspace writes are disabled")
        start = _integer(args["start_line"], 1, 1_000_000, "start_line")
        end = _integer(args["end_line"], 1, 1_000_000, "end_line")
        if end < start:
            raise ToolValidationError("end_line must be at least start_line")
        replacement = args["content"]
        if (
            not isinstance(replacement, str)
            or len(replacement.encode("utf-8")) > self.max_file_bytes
        ):
            raise ToolValidationError("content exceeds write limit or is not text")
        path = self._path(args["path"])
        self._assert_mutable(path, self.workspace / args["path"])
        if not path.is_file() or path.stat().st_size > self.max_file_bytes:
            raise ToolValidationError(
                "edit requires an existing file within the size limit"
            )
        data = await asyncio.to_thread(self._safe_read, path)
        expected = self._expected_version(path, args)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ToolValidationError(
                "file version changed; read it again before editing"
            )
        try:
            original = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolValidationError("line editing requires UTF-8 text") from exc
        if "\x00" in original or "\x00" in replacement:
            raise ToolValidationError("binary files are unavailable")
        lines = original.splitlines(keepends=True)
        if end > len(lines):
            raise ToolValidationError(
                f"range exceeds the file's {len(lines)} existing lines; read_file again"
            )

        def ending(text: str) -> str:
            if text.endswith("\r\n"):
                return "\r\n"
            # Match splitlines(), which also recognizes Unicode line endings.
            return (
                text[-1]
                if text and text[-1] in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
                else ""
            )

        if replacement and not ending(replacement):
            replacement += ending(lines[end - 1])
        changed = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        report = json.loads(
            await self._write(
                {"path": args["path"], "content": changed, "expected_sha256": expected},
                artifact=False,
            )
        )
        report.update(start_line=start, end_line=end)
        return _json(report)

    async def run_tests(self, args: dict) -> str:
        _arguments(args, {"paths"})
        if not self.allow_tests:
            raise PermissionError("test execution is disabled")
        paths = args["paths"]
        if not isinstance(paths, list) or not 1 <= len(paths) <= 20:
            raise ToolValidationError("paths must contain 1 to 20 test files")
        selected = []
        for name in paths:
            path = self._path(name)
            if (
                not path.is_file()
                or path.suffix != ".py"
                or not path.name.startswith("test_")
            ):
                raise ToolValidationError(
                    "only explicit existing Python test files are accepted"
                )
            selected.append(str(path))
        async with self._test_lock:
            with tempfile.TemporaryDirectory(prefix="forge-test-cache-") as cache:
                return await self._run_test_process(selected, cache)

    async def _run_test_process(self, selected: list[str], cache: str) -> str:
        environment = dict(os.environ)
        # Fresh cache avoids same-second/same-size edits running stale pyc files.
        environment["PYTHONPYCACHEPREFIX"] = cache
        process = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            "pytest",
            "-q",
            "--",
            *selected,
            cwd=self.workspace,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        output = bytearray()
        truncated = False

        async def collect():
            nonlocal truncated
            while chunk := await process.stdout.read(8192):
                room = max(0, 32_000 - len(output))
                output.extend(chunk[:room])
                truncated |= len(chunk) > room
            await process.wait()

        try:
            await asyncio.wait_for(collect(), self.test_timeout)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()
            raise
        return _json(
            {
                "exit_code": process.returncode,
                "output": output.decode("utf-8", errors="replace"),
                "truncated": truncated,
            }
        )
