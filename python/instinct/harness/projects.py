"""The folders Instinct has worked in, with titles the operator can change.

Kept in one JSON file for the app (``projects.json`` next to the keys). Each
project is a folder; its chats and checkpoints live in its own state folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

MAX_PROJECTS = 200


class Projects:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
            except ValueError:
                loaded = []
            self._items = [
                p
                for p in loaded
                if isinstance(p, dict) and {"id", "path", "title"} <= p.keys()
            ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with os.fdopen(
            os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
        ) as stream:
            json.dump(self._items, stream, indent=2)
        temporary.replace(self.path)

    def touch(self, folder: Path) -> dict[str, Any]:
        """Record that ``folder`` was opened, adding it if it's new."""
        # The same spelling the Mac app hashes for the state folder.
        folder = Path(os.path.abspath(folder))
        if not folder.is_dir():
            raise ValueError(f"{folder} isn't a folder")
        with self._lock:
            project = next((p for p in self._items if p["path"] == str(folder)), None)
            if project is None:
                project = {
                    "id": secrets.token_hex(6),
                    "path": str(folder),
                    "title": folder.name or str(folder),
                    "pinned": False,
                }
                self._items.append(project)
            project["last_opened"] = int(time.time())
            # Forget the oldest unpinned ones past the limit.
            unpinned = sorted(
                (p for p in self._items if not p.get("pinned")),
                key=lambda p: p.get("last_opened", 0),
            )
            for old in unpinned[: max(0, len(self._items) - MAX_PROJECTS)]:
                self._items.remove(old)
            self._save()
            return dict(project)

    def get(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            for project in self._items:
                if project["id"] == project_id:
                    return dict(project)
        raise ValueError("unknown project")

    def update(
        self, project_id: str, *, title: str | None = None, pinned: bool | None = None
    ) -> dict[str, Any]:
        with self._lock:
            project = next((p for p in self._items if p["id"] == project_id), None)
            if project is None:
                raise ValueError("unknown project")
            if title is not None:
                title = " ".join(title.split())
                if not title or len(title) > 80:
                    raise ValueError("a project title has 1 to 80 characters")
                project["title"] = title
            if pinned is not None:
                project["pinned"] = bool(pinned)
            self._save()
            return dict(project)

    def remove(self, project_id: str) -> None:
        """Take a project off the list (its folder and chats stay)."""
        with self._lock:
            kept = [p for p in self._items if p["id"] != project_id]
            if len(kept) == len(self._items):
                raise ValueError("unknown project")
            self._items = kept
            self._save()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = [dict(p) for p in self._items]
        for project in items:
            project["missing"] = not Path(project["path"]).is_dir()
        return sorted(
            items, key=lambda p: (not p.get("pinned"), -p.get("last_opened", 0))
        )


def state_folder(home: Path, folder: Path) -> Path:
    """Where a project's chats live in the app: one folder per workspace path."""
    folder = Path(os.path.abspath(folder))
    digest = hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    return home / "workspaces" / f"{folder.name}-{digest}"


def browse(path: str | None, *, root: Path) -> dict[str, Any]:
    """Subfolders of ``path`` under ``root`` (the home folder), for picking a project."""
    root = root.resolve()
    folder = (Path(path).expanduser() if path else root).resolve()
    if folder != root and root not in folder.parents:
        raise ValueError("folders outside your home folder can't be browsed")
    if not folder.is_dir():
        raise ValueError("that folder doesn't exist")
    entries = []
    try:
        for child in sorted(folder.iterdir(), key=lambda c: c.name.lower()):
            if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
                continue
            entries.append({"name": child.name, "path": str(child)})
    except PermissionError:
        raise ValueError("Instinct can't read that folder") from None
    return {
        "path": str(folder),
        "parent": str(folder.parent) if folder != root else None,
        "folders": entries[:500],
    }
