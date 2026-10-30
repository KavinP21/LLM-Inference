"""Routing requests across workers."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from ..protocol import ChatMessage, Generation, ModelBackend
from .config import (
    CapacityError,
    RequestValidationError,
    WorkerUnavailableError,
    _positive,
    validate_context_limit,
    validate_options,
    validate_request,
)


class WorkerPool:
    """Bounded least-loaded replica routing, optionally pinned by worker name.

    A generation stays on its selected replica. Retries retain that route for
    route_ttl seconds and max_route_history IDs within this pool's lifetime.
    Route history is in memory; a coordinator restart loses this affinity.
    """

    def __init__(
        self,
        backends: Mapping[str, ModelBackend] | Sequence[ModelBackend],
        *,
        max_pending: int = 32,
        per_worker_concurrency: int = 1,
        max_route_history: int = 1024,
        route_ttl: float = 600.0,
        groups: Mapping[str, str] | None = None,
    ) -> None:
        if (
            type(max_pending) is not int
            or max_pending <= 0
            or type(per_worker_concurrency) is not int
            or per_worker_concurrency <= 0
        ):
            raise ValueError("worker pool limits must be positive integers")
        self.backends = (
            dict(backends)
            if isinstance(backends, Mapping)
            else {f"worker-{i}": backend for i, backend in enumerate(backends)}
        )
        if not self.backends or any(
            not isinstance(name, str) or not name for name in self.backends
        ):
            raise ValueError("worker pool requires named backends")
        if len({id(b) for b in self.backends.values()}) != len(self.backends):
            raise ValueError("each replica must own a distinct backend")
        self.max_pending = max_pending
        if type(max_route_history) is not int or max_route_history < max_pending:
            raise ValueError("max_route_history must be at least max_pending")
        self.max_route_history = max_route_history
        self.route_ttl = _positive(route_ttl, "route_ttl")
        self._routes: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._limits = {
            name: min(
                per_worker_concurrency,
                getattr(backend, "max_concurrency", per_worker_concurrency),
            )
            for name, backend in self.backends.items()
        }
        self._loads = dict.fromkeys(self.backends, 0)
        self._condition = asyncio.Condition()
        self._requests: dict[str, str | None] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False
        self._cursor = 0
        # Worker name -> group (e.g. "driver", "subagent"). Requests can ask for
        # a group; ungrouped requests may use any replica.
        self.groups = dict(groups or {})
        if set(self.groups) - set(self.backends):
            raise ValueError("groups name unknown workers")
        # Affinity key (an agent) -> the replica that holds its prompt cache.
        self._affinity: OrderedDict[str, str] = OrderedDict()

    # The runtime passes group/affinity only to backends that route.
    routes = True

    def members(self, group: str | None) -> list[str]:
        if group is None:
            return list(self.backends)
        chosen = [name for name in self.backends if self.groups.get(name) == group]
        return chosen or list(self.backends)

    def context_limit(self, group: str | None) -> int | None:
        """Smallest context window among the replicas a group can use."""
        limits = [
            getattr(self.backends[name], "max_model_length", None)
            for name in self.members(group)
        ]
        limits = [limit for limit in limits if type(limit) is int and limit > 0]
        return min(limits) if limits else None

    def partial(self, request_id: str) -> str | None:
        for backend in self.backends.values():
            text = getattr(backend, "partial", {}).get(request_id)
            if text is not None:
                return text
        return None

    @property
    def loads(self) -> dict[str, int]:
        return dict(self._loads)

    @property
    def max_model_length(self):
        limits = [
            getattr(backend, "max_model_length", None)
            for backend in self.backends.values()
        ]
        return (
            max(limits)
            if all(type(limit) is int and limit > 0 for limit in limits)
            else None
        )

    @property
    def native_call_dialect(self):
        # Generations report their own dialect; this is the fallback when every
        # replica runs the same model family.
        dialects = {
            getattr(backend, "native_call_dialect", "json")
            for backend in self.backends.values()
        }
        return dialects.pop() if len(dialects) == 1 else "json"

    @property
    def supports_context_limits(self) -> bool:
        return all(
            getattr(backend, "supports_context_limits", False) is True
            for backend in self.backends.values()
        )

    @property
    def supports_native_tools(self) -> bool:
        return all(
            getattr(backend, "supports_native_tools", False) is True
            and callable(getattr(backend, "generate_action", None))
            for backend in self.backends.values()
        )

    def _prune_routes(self) -> None:
        now = time.monotonic()
        for request_id, (_, created) in tuple(self._routes.items()):
            if request_id not in self._requests and now - created > self.route_ttl:
                del self._routes[request_id]
        for request_id in tuple(self._routes):
            if len(self._routes) <= self.max_route_history:
                break
            if request_id not in self._requests:
                del self._routes[request_id]

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        *,
        worker: str | None = None,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        group: str | None = None,
        affinity: str | None = None,
    ) -> Generation:
        return await self._dispatch(
            messages,
            max_tokens,
            request_id,
            worker=worker,
            max_context_tokens=max_context_tokens,
            options=options,
            group=group,
            affinity=affinity,
        )

    async def generate_action(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: Sequence[dict[str, Any]],
        *,
        worker: str | None = None,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        group: str | None = None,
        affinity: str | None = None,
    ) -> Generation:
        if not self.supports_native_tools:
            raise RequestValidationError(
                "every replica in a native-tool pool must support native tool calls"
            )
        from ..native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
        return await self._dispatch(
            messages,
            max_tokens,
            request_id,
            worker=worker,
            tool_specs=specs,
            max_context_tokens=max_context_tokens,
            options=options,
            group=group,
            affinity=affinity,
        )

    async def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        *,
        worker: str | None = None,
        tool_specs: list[dict[str, Any]] | None = None,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        group: str | None = None,
        affinity: str | None = None,
    ) -> Generation:
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages=tool_specs is not None
        )
        if worker is not None and worker not in self.backends:
            raise ValueError("unknown worker route")
        max_context_tokens = validate_context_limit(max_context_tokens)
        if max_context_tokens is not None and not self.supports_context_limits:
            raise RequestValidationError(
                "every pool replica must support request context limits"
            )
        backend_options = (
            {}
            if max_context_tokens is None
            else {"max_context_tokens": max_context_tokens}
        )
        options = validate_options(options)
        if options:
            backend_options["options"] = options
        selected: str | None = None
        async with self._condition:
            if self._closed:
                raise WorkerUnavailableError("worker pool is closed")
            if request_id in self._requests:
                raise ValueError("request_id is already active in this pool")
            if len(self._requests) >= self.max_pending:
                raise CapacityError("worker pool queue is full")
            self._prune_routes()
            previous = self._routes.get(request_id)
            if previous:
                if worker is not None and worker != previous[0]:
                    raise ValueError("request_id is already pinned to another worker")
                worker = previous[0]
            self._requests[request_id] = None
            task = asyncio.current_task()
            if task is not None:
                self._tasks[request_id] = task
        try:
            async with self._condition:
                while selected is None:
                    if self._closed:
                        raise WorkerUnavailableError("worker pool is closed")
                    names = [worker] if worker else self.members(group)
                    available = [
                        name for name in names if self._loads[name] < self._limits[name]
                    ]
                    if available:
                        preferred = self._affinity.get(affinity) if affinity else None
                        if preferred in available:
                            # Same replica as last time: its cache holds this
                            # agent's prompt.
                            selected = preferred
                        else:

                            def rank(name: str) -> tuple[bool, int]:
                                # Running replicas first, then the least loaded.
                                started = getattr(self.backends[name], "started", True)
                                return (not started, self._loads[name])

                            best = min(rank(name) for name in available)
                            ties = [name for name in available if rank(name) == best]
                            selected = ties[self._cursor % len(ties)]
                            self._cursor += 1
                        if affinity:
                            self._affinity[affinity] = selected
                            self._affinity.move_to_end(affinity)
                            while len(self._affinity) > self.max_route_history:
                                self._affinity.popitem(last=False)
                        self._loads[selected] += 1
                        self._requests[request_id] = selected
                        self._routes[request_id] = (selected, time.monotonic())
                        self._routes.move_to_end(request_id)
                        self._prune_routes()
                    else:
                        await self._condition.wait()
            if tool_specs is not None:
                return await self.backends[selected].generate_action(
                    messages,
                    max_tokens,
                    request_id,
                    tool_specs=tool_specs,
                    **backend_options,
                )
            return await self.backends[selected].generate(
                messages, max_tokens, request_id, **backend_options
            )
        finally:
            async with self._condition:
                if selected is not None:
                    self._loads[selected] -= 1
                self._requests.pop(request_id, None)
                self._tasks.pop(request_id, None)
                self._condition.notify_all()

    async def cancel(self, request_id: str) -> None:
        async with self._condition:
            selected = self._requests.get(request_id)
            if selected is None:
                retained = self._routes.get(request_id)
                if retained is not None:
                    selected = retained[0]
            task = self._tasks.get(request_id)
        try:
            if selected:
                cancel = getattr(self.backends[selected], "cancel", None)
                if cancel:
                    await cancel(request_id)
        finally:
            if task is not None and task is not asyncio.current_task():
                task.cancel()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            active_tasks = tuple(self._tasks.values())
            self._condition.notify_all()
        for task in active_tasks:
            if task is not asyncio.current_task():
                task.cancel()
        await asyncio.gather(
            *(task for task in active_tasks if task is not asyncio.current_task()),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(
                backend.close()
                for backend in self.backends.values()
                if hasattr(backend, "close")
            ),
            return_exceptions=False,
        )


WorkerPoolBackend = WorkerPool
