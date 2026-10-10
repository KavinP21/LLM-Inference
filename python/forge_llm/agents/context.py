"""Trusted tool caller identity, isolated across concurrent asyncio agents."""

from contextvars import ContextVar

tool_actor: ContextVar[tuple[str, str]] = ContextVar(
    "forge_tool_actor", default=("direct", "direct")
)
