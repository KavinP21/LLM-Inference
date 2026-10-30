"""Run limits, results and the context-compaction settings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...chat_formats import EFFORT_BUDGETS
from ..store import (
    AgentRecord,
)

Tool = Callable[[dict[str, Any]], str | Awaitable[str]]

# Context compaction (native agents): once the context passes COMPACT_AT of the
# input budget, the model summarizes its work and messages older than about
# COMPACT_TAIL of the budget (in bytes) are replaced by that summary.
COMPACT_AT = 0.9
COMPACT_TAIL = 0.4
COMPACT_SUMMARY_TOKENS = 2048
COMPACT_SUMMARY_CHARS = 16_000
COMPACT_REQUEST = (
    "[runtime] Your context is nearly full. Earlier messages will be replaced by "
    "a summary you write now, so reply with plain text only and no tool calls. "
    "Cover the goal, what you have done and found, every file you read or "
    "changed (with paths), decisions and why, open problems, and your next "
    "steps. Keep exact names, paths, commands and error messages you will need."
)
SUMMARY_PREFIX = (
    "[runtime] Summary of your earlier work on this task; the full messages are "
    "in the journal.\n\n"
)


@dataclass(frozen=True)
class RuntimeConfig:
    max_agents: int = 8
    max_depth: int = 2
    max_steps_per_agent: int = 24
    max_concurrent_generations: int = 2
    max_output_tokens: int = 768
    max_total_tokens: int = 128_000
    max_context_tokens: int = 24_000
    max_protocol_errors: int = 3
    max_backend_retries: int = 2
    generation_timeout_seconds: float = 120.0
    tool_timeout_seconds: float = 30.0
    deadline_seconds: float = 900.0
    max_tool_result_chars: int = 8_000
    max_stored_tool_chars: int = 262_144
    lease_seconds: float = 15.0
    compact_history: bool = True
    max_identical_tool_results: int = 3
    max_completion_rejections: int = 3
    completion_timeout_seconds: float = 30.0
    # Reasoning effort passed to the model: off, low, medium or high.
    effort: str = "off"
    # Worker groups for root agents and for sub-agents (see WorkerPool.groups).
    driver_group: str | None = None
    subagent_group: str | None = None

    def __post_init__(self) -> None:
        integers = (
            "max_agents",
            "max_steps_per_agent",
            "max_concurrent_generations",
            "max_output_tokens",
            "max_total_tokens",
            "max_context_tokens",
            "max_protocol_errors",
            "max_tool_result_chars",
            "max_stored_tool_chars",
            "max_identical_tool_results",
            "max_completion_rejections",
        )
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_depth", "max_backend_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "generation_timeout_seconds",
            "tool_timeout_seconds",
            "deadline_seconds",
            "lease_seconds",
            "completion_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 < value < float("inf")
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if self.lease_seconds < 3:
            raise ValueError("lease_seconds must be at least 3 seconds")
        if self.effort not in EFFORT_BUDGETS:
            raise ValueError(f"effort must be one of {sorted(EFFORT_BUDGETS)}")
        for name in ("driver_group", "subagent_group"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or null")
        if self.max_tool_result_chars > self.max_stored_tool_chars:
            raise ValueError("context tool limit exceeds storage tool limit")
        if not isinstance(self.compact_history, bool):
            raise TypeError("compact_history must be a boolean")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    uncertain_tokens: int
    reserved_tokens: int

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.uncertain_tokens
            + self.reserved_tokens
        )


@dataclass(frozen=True)
class CompletionCheck:
    passed: bool
    feedback: str = ""


CompletionValidator = Callable[[str, str, str], Awaitable[CompletionCheck]]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    output: str | None
    agents: tuple[AgentRecord, ...]
    usage: TokenUsage
    error: str | None = None
    acceptance_passed: bool | None = None
    completion_verified: bool = False
