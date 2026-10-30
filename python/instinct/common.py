"""Small pieces shared by the agent runtime, the workers and the app."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


def strict_decoder(error: type[Exception] = ValueError) -> json.JSONDecoder:
    """A JSON decoder that rejects duplicate object keys and non-finite numbers."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise error(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(name: str) -> Any:
        raise error(f"JSON numbers must be finite, not {name}")

    return json.JSONDecoder(object_pairs_hook=unique, parse_constant=constant)


def strict_json(data: str | bytes, error: type[Exception] = ValueError) -> Any:
    """Parse JSON that has no duplicate object keys and only finite numbers."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return strict_decoder(error).decode(data)


def text_field(value: Any, label: str, limit: int = 12_000) -> str:
    """A non-blank string of at most ``limit`` characters, stripped."""
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must contain 1 to {limit} characters")
    return value.strip()


def clip(value: Any, limit: int) -> Any:
    """Shorten every string inside ``value`` to ``limit`` characters, saying how much went."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"\n… [{len(value) - limit:,} more characters]"
    if isinstance(value, dict):
        return {key: clip(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [clip(item, limit) for item in value]
    return value


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model is told and which method runs it."""

    name: str
    description: str | Callable[[Any], str]
    parameters: dict[str, Any]
    method: str


class Toolset:
    """Tools declared once as ``ToolSpec``s.

    ``mapping`` (name to coroutine) and ``tool_descriptions`` (what the
    runtime advertises) both come from ``specs``; a toolset whose tools depend
    on its settings overrides ``available``.
    """

    specs: tuple[ToolSpec, ...] = ()

    @property
    def schemas(self) -> dict[str, dict[str, Any]]:
        return {spec.name: spec.parameters for spec in self.specs}

    def available(self) -> Iterable[ToolSpec]:
        return self.specs

    def mapping(self) -> dict[str, Callable]:
        return {spec.name: getattr(self, spec.method) for spec in self.available()}

    def tool_descriptions(self) -> dict[str, dict[str, Any]]:
        return {
            spec.name: {
                "description": spec.description(self)
                if callable(spec.description)
                else spec.description,
                "parameters": copy.deepcopy(spec.parameters),
            }
            for spec in self.available()
        }
