"""Model replicas and bounded routing for the agent runtime.

Each local backend owns one engine on a dedicated thread. Separate worker
processes own separate complete model replicas, including their KV memory.
This is request distribution, not model or tensor parallelism.
"""

from .config import (
    CHAT_OPTIONS,
    DIALECTS,
    SAMPLING_OPTIONS,
    AuthenticationError,
    BackendError,
    CapacityError,
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    RequestValidationError,
    WorkerConfig,
    WorkerUnavailableError,
    validate_context_limit,
    validate_options,
    validate_request,
)
from .lazy import LazyBackend
from .local import LocalModelBackend
from .pool import WorkerPool
from .remote import LocalProcessBackend, RemoteWorkerBackend, is_loopback

__all__ = [
    "AuthenticationError",
    "BackendError",
    "CHAT_OPTIONS",
    "CapacityError",
    "ContextLengthError",
    "DIALECTS",
    "GenerationCancelledError",
    "GenerationTimeoutError",
    "LazyBackend",
    "LocalModelBackend",
    "LocalProcessBackend",
    "RemoteWorkerBackend",
    "RequestValidationError",
    "SAMPLING_OPTIONS",
    "WorkerConfig",
    "WorkerPool",
    "WorkerUnavailableError",
    "is_loopback",
    "validate_context_limit",
    "validate_options",
    "validate_request",
]
