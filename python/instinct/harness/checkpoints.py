"""Workspace checkpoints for rewinding a chat.

Before each turn that may change files, the whole workspace is committed to a
private git repository kept in the app's state folder (never the workspace's
own .git). This catches changes made by commands as well as by edit tools.
Restoring a checkpoint puts tracked files back and removes files created since;
it is itself checkpointed first, so a restore can be undone.

Files the workspace's .gitignore excludes are not checkpointed, and nor are
the folders in EXCLUDED, which tend to be large and rebuildable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

EXCLUDED = (
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    ".instinct/",
    "build/",
    "dist/",
    ".DS_Store",
)


class CheckpointError(RuntimeError):
    pass


class Checkpoints:
    def __init__(self, workspace: Path, state_dir: Path):
        self.workspace = Path(workspace).resolve()
        self.git_dir = Path(state_dir) / "checkpoints.git"

    @staticmethod
    def available() -> bool:
        return shutil.which("git") is not None

    def _git(self, *args: str, check: bool = True) -> str:
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": os.environ.get("HOME", "/"),
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.workspace),
            "GIT_AUTHOR_NAME": "Instinct",
            "GIT_AUTHOR_EMAIL": "checkpoints@instinct.local",
            "GIT_COMMITTER_NAME": "Instinct",
            "GIT_COMMITTER_EMAIL": "checkpoints@instinct.local",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        result = subprocess.run(
            ["git", *args],
            cwd=self.workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if check and result.returncode != 0:
            raise CheckpointError(result.stderr.strip() or f"git {args[0]} failed")
        return result.stdout.strip()

    def _init(self) -> None:
        if (self.git_dir / "HEAD").exists():
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git("init", "--quiet")
        self._git("config", "core.autocrlf", "false")
        self._git("config", "gc.auto", "0")
        (self.git_dir / "info").mkdir(exist_ok=True)
        (self.git_dir / "info" / "exclude").write_text("\n".join(EXCLUDED) + "\n")

    def snapshot(self, label: str) -> str:
        """Commit the workspace as it is now and return the commit id."""
        if not self.available():
            raise CheckpointError("git is not installed")
        self._init()
        self._git("add", "--all")
        self._git("commit", "--quiet", "--allow-empty", "--no-verify", "-m", label)
        return self._git("rev-parse", "HEAD")

    def restore(self, commit: str) -> dict:
        """Put the workspace back to ``commit``."""
        self._git("cat-file", "-e", f"{commit}^{{commit}}")
        current = self.snapshot(f"before restoring {commit[:12]}")
        created = self._git(
            "diff", "--name-only", "--diff-filter=A", "-z", commit, current
        ).split("\0")
        for name in filter(None, created):
            path = self.workspace / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        changed = self._git("diff", "--name-only", commit, current).split("\n")
        if self._git("ls-tree", "--name-only", commit):
            self._git("checkout", "--force", commit, "--", ".")
        self._prune_empty_folders(created)
        return {
            "restored": commit,
            "undo": current,
            "files": [name for name in changed if name],
        }

    def _prune_empty_folders(self, removed: list[str]) -> None:
        folders = {
            parent
            for name in removed
            if name
            for parent in (self.workspace / name).parents
            if parent != self.workspace and parent.is_relative_to(self.workspace)
        }
        for folder in sorted(folders, key=lambda path: len(path.parts), reverse=True):
            try:
                folder.rmdir()
            except OSError:
                pass
