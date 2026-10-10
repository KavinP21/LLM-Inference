"""Typed backend contract and deliberately small, strict agent action language.

The runtime treats model output as untrusted data.  There is no Python, shell,
or expression evaluator behind this protocol.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    model: str | None = None


class ModelBackend(Protocol):
    async def generate(
        self, messages: Sequence[ChatMessage], max_tokens: int, request_id: str
    ) -> Generation: ...


class ProtocolError(ValueError):
    """A response was not a single valid bounded action."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _text(value: Any, field: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ProtocolError(
            f"{field} must be a {'possibly empty ' if empty else 'nonempty '}string"
        )
    if len(value) > limit:
        raise ProtocolError(f"{field} exceeds {limit} characters")
    return value


def _ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 64:
        raise ProtocolError(f"{field} must be a list of at most 64 agent IDs")
    result = [_text(item, field, 128) for item in value]
    if len(result) != len(set(result)):
        raise ProtocolError(f"{field} contains duplicate IDs")
    return result


def parse_action(
    text: str, *, max_chars: int = 32_768, allowed_tools: Collection[str] = ()
) -> dict[str, Any]:
    """Accept one strict object; registered direct tools canonicalize to `tool`.

    Direct tools require exactly action and args.  No unknown, fuzzy, path, or
    shell aliases are recovered, and the caller supplies the closed allowlist.
    """
    if not isinstance(text, str) or len(text) > max_chars:
        raise ProtocolError(f"response must be text of at most {max_chars} characters")
    try:
        action = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"invalid JSON constant: {value}")
            ),
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ProtocolError('response must be an object with string field "action"')
    stack = [action]
    while stack:
        value = stack.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolError("JSON numbers must be finite")
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    kind = action["action"]
    schemas = {
        "tool": ({"action", "name", "args"}, set()),
        "spawn": ({"action", "task"}, {"role", "dependencies", "tools"}),
        "send": ({"action", "to", "message"}, set()),
        "wait": ({"action", "agents"}, set()),
        "finish": ({"action", "result"}, set()),
    }
    if kind not in schemas and kind in allowed_tools:
        if set(action) != {"action", "args"}:
            raise ProtocolError(
                f"direct tool {kind} requires exactly action and args; put file path inside args, not a top-level name"
            )
        if not isinstance(action["args"], dict):
            raise ProtocolError("args must be an object")
        return {"action": "tool", "name": kind, "args": action["args"]}
    if kind not in schemas:
        raise ProtocolError(f"unknown action: {kind}")
    required, optional = schemas[kind]
    missing, extra = required - action.keys(), action.keys() - required - optional
    if missing or extra:
        raise ProtocolError(
            f"invalid {kind} fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if kind == "tool":
        _text(action["name"], "name", 128)
        if not isinstance(action["args"], dict):
            raise ProtocolError("args must be an object")
    elif kind == "spawn":
        _text(action["task"], "task", 12_000)
        if "role" in action:
            if action["role"] is None:
                action.pop(
                    "role"
                )  # Explicit null selects the same documented default as omission.
            else:
                _text(action["role"], "role", 256)
        if "dependencies" in action:
            _ids(action["dependencies"], "dependencies")
        if "tools" in action:
            _ids(action["tools"], "tools")
    elif kind == "send":
        _text(action["to"], "to", 128)
        _text(action["message"], "message", 8_000)
    elif kind == "wait":
        if not _ids(action["agents"], "agents"):
            raise ProtocolError("wait must include at least one child ID")
    else:
        _text(action["result"], "result", 24_000)
    return action


def conservative_input_tokens(messages: Sequence[ChatMessage]) -> int:
    """Budget reservation, not tokenizer measurement (UTF-8 bytes + framing).

    Byte-level tokenizers cannot have more non-special tokens than input bytes.
    Backends must report their measured usage; exotic tokenizers that exceed
    this bound are rejected instead of silently bypassing the run budget.
    """
    return sum(len(message.content.encode("utf-8")) + 32 for message in messages) + 32
