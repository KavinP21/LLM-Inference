"""Interactive shells for the app's terminal panel.

Each terminal is the user's login shell in a pseudo-terminal, started in the
workspace and run in the same Seatbelt sandbox as the agent's commands, so it
can only write inside the workspace and temp folders. Unlike agent commands it
always has network access: the user types these commands themselves.
"""

from __future__ import annotations

import fcntl
import os
import signal
import struct
import termios
import threading
import uuid
import warnings
from pathlib import Path

from . import sandbox

HISTORY_BYTES = 1 << 20
MAX_TERMINALS = 8


class Terminal:
    def __init__(self, workspace: Path, cols: int = 100, rows: int = 30):
        self.id = uuid.uuid4().hex
        self.output = bytearray()
        self.start = 0  # Offset of output[0] in everything the shell printed.
        self.closed = False
        self.changed = threading.Condition()
        environment = {
            **{
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "SHELL"}
            },
            **sandbox.environment(),
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "INSTINCT_TERMINAL": "1",
        }
        argv = sandbox.command(
            [sandbox.shell(), "-l"], workspace, network=True, shell_files=True
        )
        with warnings.catch_warnings():
            # The server is multi-threaded; the child only chdirs and execs,
            # which is safe after fork.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid, fd = os.forkpty()
        if pid == 0:  # Child: become the shell. Nothing but exec happens here.
            try:
                os.chdir(workspace)
                os.execve(argv[0], argv, environment)
            finally:
                os._exit(127)
        self.pid, self.fd = pid, fd
        self.resize(cols, rows)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                data = b""
            with self.changed:
                if not data:
                    self.closed = True
                    self.changed.notify_all()
                    break
                self.output += data
                if len(self.output) > HISTORY_BYTES:
                    drop = len(self.output) - HISTORY_BYTES
                    del self.output[:drop]
                    self.start += drop
                self.changed.notify_all()
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        os.close(self.fd)

    def read(self, offset: int, timeout: float) -> tuple[int, bytes, bool]:
        """Output after ``offset``, waiting up to ``timeout`` for some."""
        with self.changed:
            end = self.start + len(self.output)
            if offset >= end and not self.closed:
                self.changed.wait(timeout)
                end = self.start + len(self.output)
            begin = max(offset, self.start)
            return end, bytes(self.output[begin - self.start :]), self.closed

    def write(self, data: str) -> None:
        if not self.closed:
            os.write(self.fd, data.encode("utf-8"))

    def resize(self, cols: int, rows: int) -> None:
        if not self.closed:
            size = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, size)

    def close(self) -> None:
        if not self.closed:
            try:
                os.killpg(os.getpgid(self.pid), signal.SIGHUP)
            except (ProcessLookupError, PermissionError):
                pass


class Terminals:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions: dict[str, Terminal] = {}

    def create(self, payload: dict) -> dict:
        self._prune()
        if len(self.sessions) >= MAX_TERMINALS:
            raise ValueError(f"at most {MAX_TERMINALS} terminals can be open")
        cols, rows = self._size(payload)
        terminal = Terminal(self.workspace, cols, rows)
        self.sessions[terminal.id] = terminal
        return {"id": terminal.id, "sandboxed": sandbox.available()}

    def get(self, terminal_id: str) -> Terminal:
        terminal = self.sessions.get(terminal_id)
        if terminal is None:
            raise ValueError("terminal does not exist")
        return terminal

    def dispatch(self, method: str, parts: list[str], payload: dict) -> dict:
        if method == "POST" and len(parts) == 2:
            return self.create(payload)
        if method != "POST" or len(parts) != 4:
            raise ValueError("unknown terminal route")
        terminal, action = self.get(parts[2]), parts[3]
        if action == "input":
            data = payload.get("data")
            if not isinstance(data, str) or len(data) > 65536:
                raise ValueError("input must be text of at most 64 KiB")
            terminal.write(data)
        elif action == "resize":
            terminal.resize(*self._size(payload))
        elif action == "close":
            terminal.close()
            self.sessions.pop(terminal.id, None)
        else:
            raise ValueError("unknown terminal action")
        return {"id": terminal.id}

    @staticmethod
    def _size(payload: dict) -> tuple[int, int]:
        cols, rows = payload.get("cols", 100), payload.get("rows", 30)
        if type(cols) is not int or type(rows) is not int:
            raise ValueError("cols and rows must be integers")
        return min(max(cols, 10), 500), min(max(rows, 4), 200)

    def _prune(self) -> None:
        for terminal_id, terminal in list(self.sessions.items()):
            if terminal.closed:
                del self.sessions[terminal_id]

    def close(self) -> None:
        for terminal in self.sessions.values():
            terminal.close()
        self.sessions.clear()
