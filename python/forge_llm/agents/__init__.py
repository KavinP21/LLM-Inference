"""Supervised local and remote model replicas for task orchestration."""

from .backends import (
    LocalForgeBackend,
    LocalProcessBackend,
    RemoteWorkerBackend,
    WorkerConfig,
    WorkerPool,
)
from .protocol import ChatMessage, Generation

__all__ = [
    "ChatMessage",
    "Generation",
    "LocalForgeBackend",
    "LocalProcessBackend",
    "RemoteWorkerBackend",
    "WorkerConfig",
    "WorkerPool",
]
