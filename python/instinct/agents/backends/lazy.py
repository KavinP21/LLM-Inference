"""Workers that start on first use and stop when idle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..protocol import Generation
from .config import WorkerConfig
from .local import LocalModelBackend


class LazyBackend:
    """Starts a worker on its first request and stops it after it has been
    idle for ``idle_seconds``, so rarely used replicas don't hold memory.

    Its identity comes from the configuration, so it's the same whether or not
    the process is running.
    """

    def __init__(
        self,
        config: WorkerConfig,
        start: Callable[[], Any],
        *,
        idle_seconds: float | None = 600.0,
        prestart: bool = False,
    ) -> None:
        self.config = config
        self._start = start
        self.idle_seconds = idle_seconds
        self.backend: Any = None
        self._lock = asyncio.Lock()
        # With prestart the process is launched right away (imports take most
        # of its startup) and loads the model on its first request.
        self._starting = asyncio.ensure_future(start()) if prestart else None
        self._active = 0
        self._idle_timer: asyncio.TimerHandle | None = None
        probe = LocalModelBackend(config)
        self._identity = probe.health()
        probe._executor.shutdown(wait=False)
        self.native_call_dialect = probe.native_call_dialect
        self.supports_native_tools = config.native_tool_calls
        self.supports_context_limits = True
        self.max_concurrency = config.max_pending if config.chat_runner else 1
        self.max_model_length = config.max_model_length

    @property
    def started(self) -> bool:
        return self.backend is not None

    @property
    def partial(self) -> dict[str, str]:
        return getattr(self.backend, "partial", {}) if self.backend else {}

    async def _ensure(self) -> Any:
        async with self._lock:
            if self.backend is None:
                starting, self._starting = self._starting, None
                try:
                    self.backend = await (starting or self._start())
                except Exception:
                    if starting is None:
                        raise
                    # A failed prestart shouldn't poison later requests.
                    self.backend = await self._start()
            return self.backend

    def _arm_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self.idle_seconds and self._active == 0 and self.backend is not None:
            loop = asyncio.get_running_loop()
            self._idle_timer = loop.call_later(
                self.idle_seconds, lambda: asyncio.ensure_future(self._stop_if_idle())
            )

    async def _stop_if_idle(self) -> None:
        async with self._lock:
            if self._active == 0 and self.backend is not None:
                backend, self.backend = self.backend, None
                await backend.close()

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Generation:
        self._active += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        try:
            backend = await self._ensure()
            return await getattr(backend, method)(*args, **kwargs)
        finally:
            self._active -= 1
            self._arm_idle_timer()

    async def generate(self, *args: Any, **kwargs: Any) -> Generation:
        return await self._call("generate", *args, **kwargs)

    async def generate_action(self, *args: Any, **kwargs: Any) -> Generation:
        return await self._call("generate_action", *args, **kwargs)

    async def cancel(self, request_id: str) -> None:
        if self.backend is not None:
            await self.backend.cancel(request_id)

    async def health(self) -> dict[str, Any]:
        if self.backend is None:
            return {
                **self._identity,
                "ready": True,
                "started": False,
                "starting": self._active > 0,
            }
        health = self.backend.health()
        if asyncio.iscoroutine(health):
            health = await health
        return {**health, "started": True}

    async def close(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        if self._starting is not None:
            starting, self._starting = self._starting, None
            backend = await asyncio.gather(starting, return_exceptions=True)
            if not isinstance(backend[0], BaseException):
                await backend[0].close()
        async with self._lock:
            if self.backend is not None:
                backend, self.backend = self.backend, None
                await backend.close()
