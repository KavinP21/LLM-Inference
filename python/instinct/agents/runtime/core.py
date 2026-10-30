"""Bounded hierarchical agents with durable action delivery and useful failures.

Scheduling correctness does not confer planning ability on a language model.
This runtime makes model decisions auditable and bounded, and supplies concrete
tools, focused delegation, dependency results, and parent synthesis.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..native_tools import NoNativeCall, parse_native_actions
from ..protocol import (
    Generation,
    ModelBackend,
    ProtocolError,
    parse_action,
)
from ..store import (
    TERMINAL,
    AgentRecord,
    AgentStore,
    BudgetError,
    LeaseError,
    StoreError,
)
from .actions import ActionMixin
from .config import CompletionValidator, RunResult, RuntimeConfig, TokenUsage, Tool
from .history import ContextMixin
from .prompts import PromptMixin


class AgentRuntime(PromptMixin, ContextMixin, ActionMixin):
    def __init__(
        self,
        backend: ModelBackend,
        store: AgentStore | str | Path = ":memory:",
        tools: Mapping[str, Tool] | None = None,
        config: RuntimeConfig | None = None,
        *,
        tool_descriptions: Mapping[str, Any] | None = None,
        replay_safe_tools: set[str] | frozenset[str] = frozenset(),
        mutating_tools: set[str] | frozenset[str] = frozenset(
            {"write_file", "replace_text", "edit_lines"}
        ),
        tool_schemas: Mapping[str, dict[str, Any]] | None = None,
        completion_validator: CompletionValidator | None = None,
        completion_root_only: bool = True,
    ) -> None:
        self.backend = backend
        if completion_validator is not None and not callable(completion_validator):
            raise TypeError("completion_validator must be callable")
        if not isinstance(completion_root_only, bool):
            raise TypeError("completion_root_only must be a boolean")
        self.completion_validator = completion_validator
        self.completion_root_only = completion_root_only
        self.store = store if isinstance(store, AgentStore) else AgentStore(store)
        self.tools = dict(tools or {})
        if any(
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 128
            or not callable(function)
            for name, function in self.tools.items()
        ):
            raise ValueError(
                "tools must map nonempty names of at most 128 characters to callables"
            )
        self.tool_descriptions = dict(tool_descriptions or {})
        self.native_mode = bool(getattr(backend, "supports_native_tools", False))
        self.tool_schemas = dict(tool_schemas or {})
        for name, description in self.tool_descriptions.items():
            if isinstance(description, dict) and isinstance(
                description.get("parameters"), dict
            ):
                self.tool_schemas.setdefault(name, description["parameters"])
        self.tool_schemas["journal_read"] = {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "after_event_id": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                "max_chars": {"type": "integer", "minimum": 256, "maximum": 4000},
                "offset": {"type": "integer", "minimum": 0},
                "include_control": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        }
        if self.native_mode and not callable(getattr(backend, "generate_action", None)):
            raise TypeError(
                "native backend requires async generate_action with tool_specs"
            )
        if self.native_mode and self.tools.keys() - self.tool_schemas.keys():
            raise ValueError(
                "native mode requires explicit parameter schemas for every registered tool"
            )
        if "journal_read" in self.tools:
            raise ValueError(
                "journal_read is reserved for the runtime's durable evidence reader"
            )
        if self.tools.keys() & {"tool", "spawn", "send", "wait", "finish"}:
            raise ValueError("tool names cannot collide with reserved agent actions")
        self.replay_safe_tools = frozenset(replay_safe_tools)
        self.mutating_tools = frozenset(mutating_tools)
        if any(not isinstance(name, str) for name in self.mutating_tools):
            raise TypeError("mutating_tools must contain tool names")
        if self.replay_safe_tools - self.tools.keys():
            raise ValueError("replay_safe_tools contains unknown tool names")
        self.config = config or RuntimeConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        self._active: set[str] = set()
        self._owners: dict[str, str] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._run_semaphores: dict[str, asyncio.Semaphore] = {}
        # Set when a local caller cancels, so the coordinator doesn't wait for
        # its next poll while an in-flight generation finishes first.
        self._wakeups: dict[str, asyncio.Event] = {}

    async def run(self, task: str, *, run_id: str | None = None, media=()) -> RunResult:
        from ..media import validate_media

        attachments = validate_media(media)
        if not isinstance(task, str) or not task.strip() or len(task) > 12_000:
            raise ValueError(
                "task must be a nonempty string of at most 12000 characters"
            )
        run_id = uuid.uuid4().hex if run_id is None else run_id
        if (
            not isinstance(run_id, str)
            or not 1 <= len(run_id) <= 96
            or any(
                not (char.isascii() and (char.isalnum() or char in "_-"))
                for char in run_id
            )
        ):
            raise ValueError(
                "run_id must contain 1 to 96 ASCII letters, numbers, underscores or hyphens"
            )
        self.store.create_run(
            run_id,
            task,
            asdict(self.config),
            self._system_prompt(self.config, task=task),
            tools=sorted({*self.tools, "journal_read"}),
            completion_required=self.completion_validator is not None,
            completion_root_only=self.completion_root_only,
            media=attachments,
        )
        return await self._drive(run_id, self.config, recover=False)

    async def resume(self, run_id: str) -> RunResult:
        run = self.store.run(run_id)
        if run["status"] not in {"running", "interrupted"}:
            return self.result(run_id)
        if run["completion_required"] and self.completion_validator is None:
            raise ValueError(
                "resume requires the original configured completion validator"
            )
        if (
            run["completion_required"]
            and bool(run["completion_root_only"]) != self.completion_root_only
        ):
            raise ValueError("resume must preserve completion validator scope")
        config = RuntimeConfig(**run["config"])
        return await self._drive(run_id, config, recover=True)

    def _wake(self, run_id: str) -> None:
        wakeup = self._wakeups.get(run_id)
        if wakeup is not None:
            wakeup.set()

    async def cancel(self, run_id: str) -> None:
        self.store.request_cancel(run_id)
        self._wake(run_id)
        await self._cancel_requests(run_id)

    async def cancel_agent(self, run_id: str, agent_id: str) -> None:
        """Request operator cancellation; the owning scheduler stops its subtree."""
        if run_id not in self._active:
            raise ValueError("agent cancellation requires an active coordinator")
        self._assert_lease(run_id)
        if self.store.agent(run_id, agent_id).status in TERMINAL:
            raise ValueError("agent has already settled")
        if agent_id == "root":
            await self.cancel(run_id)
            return
        self.store.event(
            run_id, agent_id, "agent_cancel_requested", {"agent_id": agent_id}
        )
        self._wake(run_id)

    async def _agent_cancellations(self, run_id: str, running: dict) -> None:
        for event in self.store.events(run_id, kinds=("agent_cancel_requested",)):
            key = f"{run_id}/operator-agent-cancel/{event['id']}"
            if self.store.effect(key) is not None:
                continue
            agents = {a.agent_id: a for a in self.store.agents(run_id)}
            scope = {event["payload"]["agent_id"]}
            while True:
                expanded = scope | {
                    a.agent_id for a in agents.values() if a.parent_id in scope
                }
                if expanded == scope:
                    break
                scope = expanded
            jobs, requests = [], []
            for target in scope:
                job = running.pop(target, None)
                if job is not None:
                    job.cancel()
                    jobs.append(job)
                request = self._requests.get((run_id, target))
                if request is not None:
                    requests.append(request)
            await asyncio.gather(
                *(self._cancel_request(run_id, request) for request in requests)
            )
            await asyncio.gather(*jobs, return_exceptions=True)
            self._assert_lease(run_id)
            for target in scope:
                if (
                    target in agents
                    and self.store.agent(run_id, target).status not in TERMINAL
                ):
                    self._fail(
                        run_id,
                        target,
                        "operator stopped this agent",
                        status="cancelled",
                    )
            self.store.record_effect(key, {"agents": sorted(scope)})

    def result(self, run_id: str) -> RunResult:
        run = self.store.run(run_id)
        return RunResult(
            run_id,
            run["status"],
            self.store.agent(run_id, "root").result,
            self.store.agents(run_id),
            TokenUsage(
                run["input_tokens"],
                run["output_tokens"],
                run["uncertain_tokens"],
                run["reserved_tokens"],
            ),
            run["error"],
            self.store.agent(run_id, "root").completion_passed,
            self.store.agent(run_id, "root").completion_passed is True,
        )

    def _assert_lease(self, run_id: str) -> None:
        run = self.store.run(run_id)
        if (
            run["lease_owner"] != self._owners.get(run_id)
            or (run["lease_until"] or 0) <= time.time()
        ):
            raise LeaseError(f"coordinator lease lost for {run_id}")

    async def _cancel_requests(self, run_id: str) -> None:
        cancel = getattr(self.backend, "cancel", None)
        if cancel is None:
            return
        ids = [request for (run, _), request in self._requests.items() if run == run_id]

        await asyncio.gather(
            *(self._cancel_request(run_id, request) for request in ids)
        )

    async def _cancel_request(self, run_id: str, request_id: str) -> None:
        cancel = getattr(self.backend, "cancel", None)
        if cancel is None:
            return
        try:
            await asyncio.wait_for(cancel(request_id), timeout=5)
        except Exception as exc:  # noqa: BLE001 - interchangeable cancellation adapters may fail arbitrarily.
            try:
                self._assert_lease(run_id)
            except LeaseError:
                return
            self.store.event(
                run_id,
                None,
                "worker_cancel_failed",
                {"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"},
            )

    def _inherited_goals(self, run_id: str, parent_id: str) -> str:
        root = self.store.agent(run_id, "root")
        parent = self.store.agent(run_id, parent_id)
        goals = [("Root goal", root.task)]
        if parent_id != "root":
            goals.append(("Parent goal", parent.task))
        limit = 2000 // len(goals)
        marker = " [TRUNCATED; full goal retained in journal. Ask parent for any missing relevant constraints.]"
        rows = []
        for label, text in goals:
            if len(json.dumps(text, ensure_ascii=False)) > limit:
                low, high = 0, min(len(text), limit)
                while low < high:
                    middle = (low + high + 1) // 2
                    if (
                        len(json.dumps(text[:middle] + marker, ensure_ascii=False))
                        <= limit
                    ):
                        low = middle
                    else:
                        high = middle - 1
                text = text[:low] + marker
            rows.append(
                label + " (quoted background): " + json.dumps(text, ensure_ascii=False)
            )
        return (
            "BACKGROUND GOALS: relevant constraints only. Parent orchestration and workflow instructions belong to the parent. Complete only the assigned role and scope stated LAST below. Review/proposal assignments return findings without editing files.\n"
            + "\n".join(rows)
        )

    def _fail(
        self, run_id: str, agent_id: str, error: str, *, status: str = "failed"
    ) -> None:
        with self.store.transaction():
            self.store.update_agent(
                run_id,
                agent_id,
                status=status,
                error=error,
                pending_action=None,
                action_phase=None,
            )
            self.store.event(run_id, agent_id, status, {"error": error})

    def _summary(self, agent: AgentRecord, limit: int) -> str:
        """Keep reported conclusions separate from actual scoped tool receipts."""
        receipts = self._handoff_receipts(agent, min(1500, max(256, limit // 3)))
        omitted: list[int] = []
        summary = {
            "agent_id": agent.agent_id,
            "status": agent.status,
            "task": self._context_excerpt(agent.task, min(400, max(32, limit // 12))),
            "result": (
                self._context_excerpt(agent.result, min(1200, max(32, limit // 8)))
                if agent.result is not None
                else None
            ),
            "error": (
                self._context_excerpt(agent.error, min(400, max(32, limit // 8)))
                if agent.error is not None
                else None
            ),
            "report_kind": "unverified_child_claim",
            "observed_evidence": receipts,
            "omitted_evidence_event_ids": omitted,
        }
        text = json.dumps(summary, ensure_ascii=False)
        # Prefer the latest complete receipt over a nested, opaque JSON fragment.
        # Removed receipts remain recoverable by their exact journal event IDs.
        while len(text) > limit and len(receipts) > 2:
            omitted.append(receipts.pop(0)["event_id"])
            text = json.dumps(summary, ensure_ascii=False)
        if len(text) > limit and receipts:
            excerpts = []
            for receipt in receipts:
                source = receipt.get("observed_source")
                field = (
                    "content_excerpt"
                    if source is not None
                    else "observed_result_excerpt"
                )
                target = source if source is not None else receipt
                excerpts.append((target, field, target[field]))
            low, high = 0, max(len(original) for _, _, original in excerpts)
            while low < high:
                middle = (low + high + 1) // 2
                for target, field, original in excerpts:
                    target[field] = self._context_excerpt(original, middle)
                if len(json.dumps(summary, ensure_ascii=False)) <= limit:
                    low = middle
                else:
                    high = middle - 1
            for target, field, original in excerpts:
                target[field] = self._context_excerpt(original, low)
            text = json.dumps(summary, ensure_ascii=False)
        # Exceptionally tiny limits or large argument receipts may not fit even
        # their metadata. Keep the exact omitted IDs rather than overrun context.
        while len(text) > limit and receipts:
            omitted.append(receipts.pop(0)["event_id"])
            text = json.dumps(summary, ensure_ascii=False)
        return self._context_excerpt(text, limit)

    def _handoff_receipts(
        self, agent: AgentRecord, excerpt_chars: int
    ) -> list[dict[str, Any]]:
        """At most three recent source observations, never another agent's claims."""
        readonly = (
            self.replay_safe_tools
            - self.mutating_tools
            - {
                "journal_read",
                "list_files",
            }
        )
        receipts = []
        for event in self.store.recent_tool_results(agent.run_id, agent.agent_id):
            payload = event["payload"]
            name, result = payload.get("name"), payload.get("result")
            if (
                name not in readonly
                or not isinstance(result, str)
                or not result.strip()
                or result.startswith("Tool failed:")
            ):
                continue
            try:
                observed = json.loads(result)
            except (ValueError, RecursionError):
                observed = None
            if observed in ({}, [], ""):
                continue
            if isinstance(observed, dict) and observed.get("matches") == []:
                continue
            receipt = {
                "event_id": event["id"],
                "tool": name,
                "journal_cursor": {
                    "agent_id": agent.agent_id,
                    "after_event_id": event["id"] - 1,
                    "limit": 1,
                },
            }
            args = payload.get("args", {})
            args_text = json.dumps(args, ensure_ascii=False)
            if len(args_text) <= 512:
                receipt["args"] = args
            else:
                receipt["args_excerpt"] = self._context_excerpt(args_text, 512)
            if (
                name == "read_file"
                and isinstance(observed, dict)
                and isinstance(observed.get("content"), str)
            ):
                source = {
                    "content_excerpt": self._context_excerpt(
                        observed["content"], excerpt_chars
                    ),
                    "content_chars": len(observed["content"]),
                }
                # Preserve observed metadata only; never infer absent measurements.
                for key in ("path", "start_line", "total_lines", "truncated"):
                    value = observed.get(key)
                    if isinstance(value, (str, int, bool)):
                        source[key] = (
                            self._context_excerpt(value, 256)
                            if isinstance(value, str)
                            else value
                        )
                receipt["observed_source"] = source
            else:
                receipt["observed_result_excerpt"] = self._context_excerpt(
                    result, excerpt_chars
                )
            receipts.append(receipt)
            if len(receipts) == 3:
                break
        return list(reversed(receipts))

    def _refresh_waiters(self, run_id: str, config: RuntimeConfig) -> bool:
        changed = False
        agents = {agent.agent_id: agent for agent in self.store.agents(run_id)}
        for agent in agents.values():
            if agent.status == "queued":
                dependencies = [agents[dep] for dep in agent.dependencies]
                failed = [
                    dep for dep in dependencies if dep.status in {"failed", "cancelled"}
                ]
                if failed:
                    self._fail(
                        run_id,
                        agent.agent_id,
                        "dependency failed: "
                        + ", ".join(dep.agent_id for dep in failed),
                    )
                    changed = True
                elif all(dep.status == "completed" for dep in dependencies):
                    with self.store.transaction():
                        self.store.add_message(
                            run_id,
                            agent.agent_id,
                            "user",
                            "Dependency evidence:\n"
                            + "\n".join(
                                self._summary(dep, config.max_tool_result_chars)
                                for dep in dependencies
                            ),
                        )
                        self.store.update_agent(
                            run_id, agent.agent_id, status="runnable"
                        )
                        self.store.event(
                            run_id,
                            agent.agent_id,
                            "dependencies_ready",
                            {"agents": list(agent.dependencies)},
                        )
                    changed = True
            elif agent.status == "waiting":
                children = [agents[child] for child in agent.wait_for]
                if all(child.status in TERMINAL for child in children):
                    with self.store.transaction():
                        self.store.add_message(
                            run_id,
                            agent.agent_id,
                            "tool" if self.native_mode else "user",
                            "Child outcomes (reports are unverified claims; verify decisive claims against observed source evidence; report failures):\n"
                            + "\n".join(
                                self._summary(child, config.max_tool_result_chars)
                                for child in children
                            ),
                        )
                        self.store.update_agent(
                            run_id, agent.agent_id, status="runnable", wait_for=[]
                        )
                        for child in children:
                            delivery_key = (
                                f"{run_id}/{agent.agent_id}/outcome/{child.agent_id}"
                            )
                            if self.store.effect(delivery_key) is None:
                                self.store.record_effect(
                                    delivery_key, {"status": child.status}
                                )
                        self.store.event(
                            run_id,
                            agent.agent_id,
                            "children_settled",
                            {"agents": list(agent.wait_for)},
                        )
                    changed = True
        return changed

    async def _drive(
        self, run_id: str, config: RuntimeConfig, *, recover: bool
    ) -> RunResult:
        if run_id in self._active:
            raise LeaseError(f"run {run_id} already active in this runtime")
        # One runtime shares a global semaphore across all its runs.  A resumed
        # run cannot increase the configured process concurrency.
        owner = uuid.uuid4().hex
        self.store.acquire_lease(run_id, owner, config.lease_seconds)
        self._active.add(run_id)
        self._owners[run_id] = owner
        self._run_semaphores[run_id] = asyncio.Semaphore(
            config.max_concurrent_generations
        )
        wakeup = self._wakeups[run_id] = asyncio.Event()
        running: dict[str, asyncio.Task[None]] = {}
        try:
            for agent in self.store.agents(run_id):
                if agent.tools is None:
                    self.store.update_agent(
                        run_id,
                        agent.agent_id,
                        tools=sorted(self._allowed_tools(run_id, agent.agent_id)),
                    )
            if recover:
                self.store.recover(run_id)
                self.store.connection.execute(
                    "UPDATE runs SET status='running' WHERE run_id=?", (run_id,)
                )
            while True:
                self.store.renew_lease(run_id, owner, config.lease_seconds)
                run = self.store.run(run_id)
                reason = (
                    "cancel requested"
                    if run["cancel_requested"]
                    else (
                        "run deadline exceeded"
                        if time.time() >= run["deadline"]
                        else None
                    )
                )
                if reason:
                    for job in running.values():
                        job.cancel()
                    await self._cancel_requests(run_id)
                    await asyncio.gather(*running.values(), return_exceptions=True)
                    for agent in self.store.agents(run_id):
                        if agent.status not in TERMINAL:
                            self._fail(
                                run_id,
                                agent.agent_id,
                                reason,
                                status="cancelled"
                                if run["cancel_requested"]
                                else "failed",
                            )
                    self.store.finish_run(
                        run_id,
                        "cancelled" if run["cancel_requested"] else "failed",
                        reason,
                    )
                    break
                await self._agent_cancellations(run_id, running)
                # Failure propagation and parent wake-ups must reach a fixed
                # point before declaring deadlock from a stale snapshot.
                while self._refresh_waiters(run_id, config):
                    pass
                # A failed subtree cannot leave orphan work running after its
                # parent settles.  Cancellation reaches descendants before a
                # parent can synthesize their terminal outcomes.
                while True:
                    current = {
                        item.agent_id: item for item in self.store.agents(run_id)
                    }
                    orphans = [
                        item
                        for item in current.values()
                        if item.status not in TERMINAL
                        and item.parent_id
                        and current[item.parent_id].status in {"failed", "cancelled"}
                    ]
                    if not orphans:
                        break
                    for orphan in orphans:
                        job = running.pop(orphan.agent_id, None)
                        if job is not None:
                            job.cancel()
                            await asyncio.gather(job, return_exceptions=True)
                        self._fail(
                            run_id,
                            orphan.agent_id,
                            "parent agent failed or was cancelled",
                            status="cancelled",
                        )
                    while self._refresh_waiters(run_id, config):
                        pass
                agents = self.store.agents(run_id)
                root = next(agent for agent in agents if agent.agent_id == "root")
                if root.status in {"failed", "cancelled"}:
                    for job in running.values():
                        job.cancel()
                    await self._cancel_requests(run_id)
                    await asyncio.gather(*running.values(), return_exceptions=True)
                    for agent in self.store.agents(run_id):
                        if agent.status not in TERMINAL:
                            self._fail(
                                run_id,
                                agent.agent_id,
                                "parent run failed",
                                status="cancelled",
                            )
                    self.store.finish_run(run_id, root.status, root.error)
                    break
                if root.status == "completed":
                    if any(agent.status not in TERMINAL for agent in agents):
                        raise StoreError("root completed with unsettled descendants")
                    status = (
                        "completed_with_failures"
                        if any(agent.status != "completed" for agent in agents)
                        else "completed"
                    )
                    self.store.finish_run(run_id, status)
                    break
                for agent in agents:
                    if agent.status == "runnable" and agent.agent_id not in running:
                        running[agent.agent_id] = asyncio.create_task(
                            self._turn(run_id, agent.agent_id, config)
                        )
                if not running:
                    self._fail(
                        run_id,
                        "root",
                        "no runnable work: unresolved dependency or wait state",
                    )
                    continue
                woken = asyncio.ensure_future(wakeup.wait())
                try:
                    done, _ = await asyncio.wait(
                        [*running.values(), woken],
                        timeout=min(1.0, config.lease_seconds / 3),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    woken.cancel()
                wakeup.clear()
                for agent_id, job in list(running.items()):
                    if job in done:
                        del running[agent_id]
                        if job.cancelled():
                            # A worker's cancellation cancels its generation
                            # coroutine, not the coordinator.  Explicit run
                            # cancellation settles all agents in the next loop.
                            if self.store.run(run_id)["cancel_requested"]:
                                continue
                            self._fail(
                                run_id,
                                agent_id,
                                "model worker cancelled the active generation",
                            )
                            continue
                        # Unexpected internal failures stop the run instead of
                        # disguising a programming error as a model decision.
                        job.result()
        except LeaseError:
            for job in running.values():
                job.cancel()
            await asyncio.gather(*running.values(), return_exceptions=True)
            raise
        except asyncio.CancelledError:
            for job in running.values():
                job.cancel()
            await self._cancel_requests(run_id)
            await asyncio.gather(*running.values(), return_exceptions=True)
            try:
                self._assert_lease(run_id)
            except LeaseError:
                pass
            else:
                self.store.connection.execute(
                    "UPDATE runs SET status='interrupted' WHERE run_id=?", (run_id,)
                )
                self.store.event(
                    run_id,
                    None,
                    "interrupted",
                    {"reason": "coordinator task cancelled; resume explicitly"},
                )
            raise
        except Exception as exc:
            for job in running.values():
                job.cancel()
            await self._cancel_requests(run_id)
            await asyncio.gather(*running.values(), return_exceptions=True)
            self._assert_lease(run_id)
            self.store.connection.execute(
                "UPDATE runs SET status='interrupted',error=? WHERE run_id=?",
                (f"coordinator error: {type(exc).__name__}: {exc}", run_id),
            )
            raise
        finally:
            self.store.release_lease(run_id, owner)
            self._active.discard(run_id)
            self._owners.pop(run_id, None)
            self._run_semaphores.pop(run_id, None)
            self._wakeups.pop(run_id, None)
        return self.result(run_id)

    async def _turn(self, run_id: str, agent_id: str, config: RuntimeConfig) -> None:
        self._assert_lease(run_id)
        agent = self.store.agent(run_id, agent_id)
        if agent.pending_action is not None:
            if agent.pending_action.get("action") == "batch":
                await self._apply_batch(run_id, agent_id, agent.pending_action, config)
            else:
                await self._apply_action(
                    run_id,
                    agent_id,
                    agent.pending_action,
                    config,
                    recovering=agent.action_phase == "executing",
                )
            return
        if agent.action_phase != "generating":
            if agent.steps >= config.max_steps_per_agent:
                self._fail(
                    run_id, agent_id, "per-agent step limit reached without finish"
                )
                return
            self.store.update_agent(
                run_id,
                agent_id,
                status="running",
                steps=agent.steps + 1,
                action_phase="generating",
                backend_attempts=0,
            )
        else:
            self.store.update_agent(run_id, agent_id, status="running")
        agent = self.store.agent(run_id, agent_id)
        request_id = f"{run_id}.{agent_id}.{agent.steps}"
        if getattr(self.backend, "routes", False) is True:
            # Sub-agents may run on a model with a smaller context window.
            group = (
                config.driver_group
                if agent.parent_id is None
                else config.subagent_group
            )
            context_limit = getattr(self.backend, "context_limit", None)
            limit = context_limit(group) if callable(context_limit) else None
            if limit and limit < config.max_context_tokens:
                config = replace(
                    config,
                    max_context_tokens=limit,
                    max_output_tokens=min(config.max_output_tokens, limit // 4),
                )
        saved_input = self.store.generation_input(run_id, request_id)
        if saved_input is None:
            tool_specs = (
                self._native_catalog(run_id, agent_id, config)
                if self.native_mode
                else None
            )
            if self.native_mode:
                self._note_state(run_id, agent_id, config)
                if config.compact_history:
                    await self._maybe_compact(run_id, agent_id, config, tool_specs)
            history_length = len(self.store.messages(run_id, agent_id))
            messages = self._model_context(
                run_id,
                agent_id,
                config,
                schema_input_bound=self._schema_input_bound(tool_specs),
            )
            self.store.save_generation_input(
                run_id, request_id, messages, history_length, tool_specs=tool_specs
            )
        else:
            messages, history_length = saved_input
            tool_specs = self.store.generation_tool_specs(run_id, request_id)
        if tool_specs is not None and not self.native_mode:
            raise ValueError(
                "resume requires a native backend matching the persisted function catalog"
            )
        input_bound = self._input_bound(
            run_id, agent_id, messages
        ) + self._schema_input_bound(tool_specs)
        if input_bound + config.max_output_tokens > config.max_context_tokens:
            self._fail(
                run_id,
                agent_id,
                f"context limit reached ({input_bound} conservative input tokens + {config.max_output_tokens} output > {config.max_context_tokens}); full history is retained in the journal",
            )
            return
        reservation_input_bound = input_bound
        backend_options = self._backend_options(run_id, agent_id, config, messages)
        if any(message.media for message in messages):
            model_limit = getattr(self.backend, "max_model_length", None)
            if type(model_limit) is not int or model_limit <= 0:
                self._fail(
                    run_id,
                    agent_id,
                    "multimodal backend must advertise a bounded model context",
                )
                return
            if (
                getattr(self.backend, "supports_context_limits", False) is not True
                and model_limit > config.max_context_tokens
            ):
                self._fail(
                    run_id,
                    agent_id,
                    "multimodal backend must enforce the runtime's smaller request context limit",
                )
                return
            # Processor expansion is checked by the worker before inference.
            # Reserve its full context until measured usage refunds the excess.
            reservation_input_bound = max(
                input_bound, min(model_limit, config.max_context_tokens)
            )
        async with self._semaphore, self._run_semaphores[run_id]:
            while True:
                self._assert_lease(run_id)
                agent = self.store.agent(run_id, agent_id)
                if agent.backend_attempts >= config.max_backend_retries + 1:
                    self._fail(run_id, agent_id, "backend retry limit reached")
                    return
                attempt = agent.backend_attempts + 1
                reservation = request_id + f"/attempt-{attempt}"
                try:
                    self.store.reserve(
                        run_id,
                        reservation,
                        reservation_input_bound + config.max_output_tokens,
                        config.max_total_tokens,
                    )
                except BudgetError as exc:
                    self._fail(run_id, agent_id, str(exc))
                    return
                self.store.update_agent(run_id, agent_id, backend_attempts=attempt)
                self._requests[(run_id, agent_id)] = request_id
                try:
                    request = (
                        self.backend.generate_action(
                            messages,
                            config.max_output_tokens,
                            request_id,
                            tool_specs=tool_specs,
                            **backend_options,
                        )
                        if tool_specs is not None
                        else self.backend.generate(
                            messages,
                            config.max_output_tokens,
                            request_id,
                            **backend_options,
                        )
                    )
                    generation = await asyncio.wait_for(
                        request,
                        timeout=config.generation_timeout_seconds,
                    )
                    self._assert_lease(run_id)
                    if not isinstance(generation, Generation):
                        raise TypeError("backend must return Generation")
                    if (
                        not isinstance(generation.text, str)
                        or len(generation.text) > 32_768
                    ):
                        raise ValueError(
                            "backend response exceeds the 32768-character action limit"
                        )
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in (generation.input_tokens, generation.output_tokens)
                    ):
                        raise ValueError(
                            "backend token usage must contain nonnegative integers"
                        )
                    if generation.output_tokens > config.max_output_tokens:
                        raise ValueError("backend exceeded max_output_tokens")
                    if (
                        backend_options
                        and generation.input_tokens + config.max_output_tokens
                        > config.max_context_tokens
                    ):
                        raise ValueError("backend exceeded the requested context limit")
                    # Zero/missing counts use conservative estimates, rather
                    # than allowing a backend to defeat the token budget.
                    input_tokens = generation.input_tokens or reservation_input_bound
                    output_tokens = generation.output_tokens or min(
                        config.max_output_tokens, len(generation.text.encode("utf-8"))
                    )
                    self.store.settle(run_id, reservation, input_tokens, output_tokens)
                    self.store.prune_generation_inputs(run_id, agent_id, request_id)
                    self.store.event(
                        run_id,
                        agent_id,
                        "generation",
                        {
                            "request_id": request_id,
                            "attempt": attempt,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "finish_reason": generation.finish_reason,
                            "model": generation.model,
                            **(
                                {"metrics": generation.metrics}
                                if generation.metrics
                                else {}
                            ),
                        },
                    )
                    break
                except asyncio.CancelledError:
                    try:
                        self._assert_lease(run_id)
                    except LeaseError:
                        pass  # The new owner accounts for stale reservations.
                    else:
                        self.store.uncertain(run_id, reservation)
                    raise
                except LeaseError:
                    # Another coordinator owns recovery and journal mutation.
                    raise
                except Exception as exc:  # noqa: BLE001 - normalize failures from interchangeable untrusted backends.
                    if getattr(exc, "work_started", None) is False:
                        self.store.settle(run_id, reservation, 0, 0)
                    else:
                        self.store.uncertain(run_id, reservation)
                    retryable = getattr(
                        exc,
                        "retryable",
                        not isinstance(exc, (TypeError, ValueError, BudgetError)),
                    )
                    self.store.event(
                        run_id,
                        agent_id,
                        "backend_error",
                        {
                            "request_id": request_id,
                            "attempt": attempt,
                            "retryable": bool(retryable),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    if not retryable or attempt >= config.max_backend_retries + 1:
                        if getattr(exc, "work_started", None) is not False:
                            await self._cancel_request(run_id, request_id)
                            self._assert_lease(run_id)
                        self._fail(
                            run_id,
                            agent_id,
                            f"backend failed after {attempt} attempts: {type(exc).__name__}: {exc}",
                        )
                        return
                finally:
                    self._requests.pop((run_id, agent_id), None)
        self._assert_lease(run_id)
        with self.store.transaction():
            self.store.add_message(run_id, agent_id, "assistant", generation.text)
            response = {"text": generation.text}
            if generation.reasoning:
                # Kept for display only; it never re-enters the prompt.
                response["reasoning"] = generation.reasoning[-8000:]
            self.store.event(run_id, agent_id, "model_response", response)
        try:
            if tool_specs is not None:
                dialect = generation.dialect or getattr(
                    self.backend, "native_call_dialect", "json"
                )
                try:
                    actions = parse_native_actions(
                        generation.text,
                        tool_specs,
                        max_calls=4,
                        allow_xml_text=dialect == "qwen3_coder_xml",
                        dialect=dialect,
                    )
                except NoNativeCall:
                    # A reply without a tool call is the agent's final answer.
                    if not generation.text.strip():
                        raise ProtocolError(
                            "empty reply; call a tool or answer"
                        ) from None
                    actions = [
                        parse_action(
                            json.dumps(
                                {"action": "finish", "result": generation.text.strip()},
                                ensure_ascii=False,
                            ),
                            max_chars=max(32_768, len(generation.text) + 64),
                        )
                    ]
                if len(actions) > 1:
                    self._validate_batch(run_id, agent_id, actions, config)
                    action = {"action": "batch", "items": actions, "next_index": 0}
                else:
                    action = actions[0]
            else:
                action = parse_action(
                    generation.text, allowed_tools=self._allowed_tools(run_id, agent_id)
                )
        except ProtocolError as exc:
            self._action_error(run_id, agent_id, str(exc), config)
            return
        if (
            action["action"] == "finish"
            and len(self.store.messages(run_id, agent_id)) > history_length + 1
        ):
            self._action_error(
                run_id,
                agent_id,
                "new messages arrived during generation; review them in your next turn before finishing",
                config,
            )
            return
        with self.store.transaction():
            self.store.update_agent(
                run_id, agent_id, pending_action=action, action_phase="prepared"
            )
            self.store.event(run_id, agent_id, "action_prepared", action)
        if action["action"] == "batch":
            await self._apply_batch(run_id, agent_id, action, config)
        else:
            await self._apply_action(run_id, agent_id, action, config)
