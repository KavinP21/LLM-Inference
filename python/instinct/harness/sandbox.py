"""Run commands so they can only change the workspace.

On macOS, commands go through Seatbelt (``sandbox-exec``): they can read
anything the user can, but can write only inside the workspace and the
temporary directories, and can reach the network only on localhost unless
network access is turned on. Elsewhere commands run unconfined, and the
``sandboxed`` flag in results says so.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def available() -> bool:
    return sys.platform == "darwin" and os.access(SANDBOX_EXEC, os.X_OK)


def _quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


# A shell keeps its history and completion cache in the home folder; the
# terminal (not agent commands) may write those.
SHELL_FILES = (
    ".zsh_history",
    ".zsh_history.LOCK",
    ".zsh_history.new",
    ".bash_history",
    ".zsh_sessions",
)


def profile(workspace: Path, *, network: bool, shell_files: bool = False) -> str:
    writable = {
        workspace.resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/private/tmp"),
        Path("/private/var/folders"),
    }
    roots = " ".join(f"(subpath {_quote(path)})" for path in sorted(writable))
    if shell_files:
        home = Path.home().resolve()
        roots += "".join(f" (subpath {_quote(home / name)})" for name in SHELL_FILES)
        # zsh names its completion cache .zcompdump-<host>-<version>.
        escaped = str(home).replace(".", "\\.").replace('"', "")
        roots += f' (regex #"^{escaped}/\\.zcompdump")'
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny file-write* (require-not (require-any "
        + roots
        + ' (literal "/dev/null") (literal "/dev/zero") (literal "/dev/ptmx")'
        + ' (literal "/dev/dtracehelper") (regex #"^/dev/tty") (regex #"^/dev/fd/"))))',
    ]
    if not network:
        rules += [
            "(deny network-outbound (remote ip))",
            '(allow network-outbound (remote ip "localhost:*"))',
        ]
    return "\n".join(rules)


def command(
    argv: list[str], workspace: Path, *, network: bool, shell_files: bool = False
) -> list[str]:
    """Return ``argv`` wrapped in the sandbox, when the platform has one."""
    if not available():
        return argv
    rules = profile(workspace, network=network, shell_files=shell_files)
    return [SANDBOX_EXEC, "-p", rules, *argv]


def environment() -> dict[str, str]:
    """Caches that tools would write into the home directory go to temp instead."""
    cache = Path(tempfile.gettempdir()) / "instinct-cache"
    cache.mkdir(exist_ok=True)
    return {
        "XDG_CACHE_HOME": str(cache),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "npm_config_cache": str(cache / "npm"),
        "GOCACHE": str(cache / "go"),
    }


def shell() -> str:
    return os.environ.get("SHELL") or shutil.which("zsh") or "/bin/sh"
