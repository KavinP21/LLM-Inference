"""Bearer keys for the OpenAI-compatible API.

Only SHA-256 hashes are stored (mode 0600); a key's text is shown once, when
it is made. Keys open ``/v1`` and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

PREFIX = "sk-instinct-"


class ApiKeys:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._keys: list[dict[str, Any]] = []
        if self.path.is_file():
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, list):
                self._keys = [
                    k
                    for k in loaded
                    if isinstance(k, dict) and {"id", "name", "sha256"} <= k.keys()
                ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with os.fdopen(
            os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
        ) as stream:
            json.dump(self._keys, stream, indent=2)
        temporary.replace(self.path)

    def create(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 64:
            raise ValueError("a key needs a name of up to 64 characters")
        key = PREFIX + secrets.token_urlsafe(32)
        record = {
            "id": secrets.token_hex(6),
            "name": name,
            "sha256": hashlib.sha256(key.encode()).hexdigest(),
            "created": int(time.time()),
            "last_used": None,
        }
        with self._lock:
            self._keys.append(record)
            self._save()
        return {**self._public(record), "key": key}

    def verify(self, authorization: str | None) -> dict[str, Any] | None:
        """The key record for an ``Authorization: Bearer ...`` header, if valid."""
        if not authorization or not authorization.startswith("Bearer "):
            return None
        digest = hashlib.sha256(authorization[7:].strip().encode()).hexdigest()
        with self._lock:
            for record in self._keys:
                if hmac.compare_digest(record["sha256"], digest):
                    now = int(time.time())
                    if not record.get("last_used") or now - record["last_used"] > 60:
                        record["last_used"] = now
                        self._save()
                    return self._public(record)
        return None

    def revoke(self, key_id: str) -> None:
        with self._lock:
            kept = [k for k in self._keys if k["id"] != key_id]
            if len(kept) == len(self._keys):
                raise ValueError("unknown key")
            self._keys = kept
            self._save()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(k) for k in self._keys]

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {k: record.get(k) for k in ("id", "name", "created", "last_used")}
