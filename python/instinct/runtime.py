"""Request states and events shared by the engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SequenceState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


TERMINAL = frozenset(
    {SequenceState.COMPLETED, SequenceState.CANCELLED, SequenceState.REJECTED}
)


@dataclass(frozen=True)
class TokenEvent:
    request_id: int
    token: int
    finished: bool
    # "stop" (an end token), "length" (the request's token budget) or
    # "context" (the model's context limit); empty while running.
    reason: str = ""
