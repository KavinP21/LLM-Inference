"""Bounded, content-addressed media for authenticated model-worker requests."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path

MAX_MEDIA_BYTES = 8 << 20
MAX_TOTAL_MEDIA_BYTES = 16 << 20
MAX_MEDIA_ITEMS = 8
MAX_MEDIA_REQUEST_BYTES = 24 << 20
_TYPES = {
    "image/png": ("image", ".png"),
    "image/jpeg": ("image", ".jpg"),
    "image/webp": ("image", ".webp"),
    "image/bmp": ("image", ".bmp"),
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "video/x-msvideo": ("video", ".avi"),
    "video/webm": ("video", ".webm"),
    "video/x-m4v": ("video", ".m4v"),
}
_HASH = re.compile(r"^[a-f0-9]{64}$")


def validate_media(items) -> tuple[dict[str, str], ...]:
    if not isinstance(items, (list, tuple)) or len(items) > MAX_MEDIA_ITEMS:
        raise ValueError(f"media must contain at most {MAX_MEDIA_ITEMS} items")
    result, total = [], 0
    for item in items:
        if not isinstance(item, dict) or item.keys() != {
            "type",
            "mime",
            "data",
            "sha256",
        }:
            raise ValueError("media requires exactly type, mime, data and sha256")
        if any(not isinstance(value, str) for value in item.values()):
            raise TypeError("media fields must be text")
        expected = _TYPES.get(item["mime"])
        if expected is None or item["type"] != expected[0]:
            raise ValueError("unsupported image/video media type")
        if not _HASH.fullmatch(item["sha256"]):
            raise ValueError("invalid media fingerprint")
        if len(item["data"]) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
            raise ValueError("encoded media exceeds the byte limit")
        try:
            data = base64.b64decode(item["data"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("media must use canonical base64") from exc
        if not data or len(data) > MAX_MEDIA_BYTES:
            raise ValueError("media exceeds the byte limit or is empty")
        if base64.b64encode(data).decode("ascii") != item["data"]:
            raise ValueError("media must use canonical base64")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("media fingerprint does not match its bytes")
        total += len(data)
        if total > MAX_TOTAL_MEDIA_BYTES:
            raise ValueError("combined media exceeds the byte limit")
        result.append(dict(item))
    return tuple(result)


def media_from_file(path: str | Path, *, mime: str) -> dict[str, str]:
    if mime not in _TYPES:
        raise ValueError("unsupported media MIME type")
    selected = Path(path)
    if not selected.is_file():
        raise ValueError("media must be an existing regular file")
    with selected.open("rb") as stream:
        data = stream.read(MAX_MEDIA_BYTES + 1)
    if not data or len(data) > MAX_MEDIA_BYTES:
        raise ValueError("media exceeds the byte limit or is empty")
    return {
        "type": _TYPES[mime][0],
        "mime": mime,
        "data": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def materialize_media(items, directory: Path) -> list[dict[str, str]]:
    """Decode validated bytes into a request-owned private directory."""
    parts = []
    for index, item in enumerate(validate_media(items)):
        path = directory / f"{index}-{item['sha256']}{_TYPES[item['mime']][1]}"
        with path.open("xb") as stream:
            stream.write(base64.b64decode(item["data"]))
        path.chmod(0o600)
        parts.append({"type": item["type"], item["type"]: str(path)})
    return parts
