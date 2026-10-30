"""Validating and applying an agent's actions, and checking completion."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from ..context import tool_actor
from ..protocol import (
    ProtocolError,
)
from ..store import (
    TERMINAL,
    LeaseError,
)
from .config import CompletionCheck, RuntimeConfig


class ActionMixin:
    def _action_error(
        self, run_id: str, agent_id: str, error: str, config: RuntimeConfig
    ) -> None:
        agent = self.store.agent(run_id, agent_id)
        count = agent.protocol_errors + 1
        if count >= config.max_protocol_errors:
            self._fail(run_id, agent_id, f"action repair limit reached: {error}")
            return
        with self.store.transaction():
            self.store.update_agent(run_id, agent_id, protocol_errors=count)
            self.store.complete_action(
                run_id,
                agent_id,
                "Action rejected: "
                + error
                + (
                    ". Fix the call and try again, or reply with your final answer as plain text if you're done."
                    if self.native_mode
                    else ". Choose ONE action from the CURRENT ACTION MENU. Direct registered tools use exactly action=tool_name and args={...}; put path inside args.path, not top-level name. Use no extra fields or fences. A task answer belongs in finish.result as a STRING, never an unwrapped object. Wait only for child outcomes not already delivered."
                ),
            )
            self.store.event(
                run_id,
                agent_id,
                "action_rejected",
                {"error": error, "consecutive_errors": count},
            )

    def _validate_batch(
        self,
        run_id: str,
        agent_id: str,
        items: list[dict[str, Any]],
        config: RuntimeConfig,
    ) -> None:
        if not 2 <= len(items) <= 4:
            raise ProtocolError("native batches require 2 to 4 calls")
        available = self._allowed_tools(run_id, agent_id)
        if all(item["action"] == "spawn" for item in items):
            agent = self.store.agent(run_id, agent_id)
            if agent.depth >= config.max_depth:
                raise ProtocolError("batch spawning exceeds hierarchy depth")
            children = {
                child.agent_id
                for child in self.store.agents(run_id)
                if child.parent_id == agent_id
            }
            new = 0
            for index, item in enumerate(items):
                if any(dep not in children for dep in item.get("dependencies", [])):
                    raise ProtocolError(
                        "batch dependencies must refer to existing direct children"
                    )
                if set(item.get("tools", [])) - available:
                    raise ProtocolError("batch child tools cannot exceed caller grants")
                if (
                    self.store.effect(
                        f"{run_id}/{agent_id}/{agent.steps}/effect/batch/{index}"
                    )
                    is None
                ):
                    new += 1
            if len(self.store.agents(run_id)) + new > config.max_agents:
                raise ProtocolError(
                    "full spawn batch exceeds remaining agent capacity; no new children created"
                )
        elif all(item["action"] == "tool" for item in items):
            safe = (self.replay_safe_tools | {"journal_read"}) - self.mutating_tools
            if any(
                item["name"] not in safe or item["name"] not in available
                for item in items
            ):
                raise ProtocolError(
                    "native tool batches permit only granted replay-safe read-only tools"
                )
            queries = [
                json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items
            ]
            if len(set(queries)) != len(queries):
                raise ProtocolError(
                    "duplicate identical batch queries add no evidence; choose distinct queries"
                )
        else:
            raise ProtocolError(
                "native batches must be all spawns or all replay-safe read-only tools; no mixed controls or edits"
            )

    async def _apply_batch(
        self, run_id: str, agent_id: str, batch: dict[str, Any], config: RuntimeConfig
    ) -> None:
        self._assert_lease(run_id)
        try:
            self._validate_batch(run_id, agent_id, batch["items"], config)
        except ProtocolError as exc:
            self._action_error(run_id, agent_id, str(exc), config)
            return
        step = self.store.agent(run_id, agent_id).steps
        for index, item in enumerate(batch["items"]):
            key = f"{run_id}/{agent_id}/{step}/effect/batch/{index}"
            if self.store.effect(key) is None:
                await self._apply_action(
                    run_id,
                    agent_id,
                    item,
                    config,
                    recovering=self.store.agent(run_id, agent_id).action_phase
                    == "batch",
                    effect_key=key,
                    preserve_pending=True,
                )
                current = self.store.agent(run_id, agent_id)
                if current.status in TERMINAL or current.pending_action is None:
                    return
            self._assert_lease(run_id)
            batch = dict(batch, next_index=index + 1)
            self.store.update_agent(
                run_id, agent_id, pending_action=batch, action_phase="batch"
            )
        warnings = [
            self.store.effect(
                f"{run_id}/{agent_id}/{step}/effect/batch/{index}/no_progress"
            )
            for index in range(len(batch["items"]))
        ]
        warning_text = "\n".join(
            value["feedback"] for value in warnings if value is not None
        )
        with self.store.transaction():
            self.store.complete_action(
                run_id,
                agent_id,
                f"Completed {len(batch['items'])} batch calls. Their individual outcomes are recorded above; use any failures as diagnostics.\n"
                + warning_text,
                feedback_role="user",
            )
            self.store.event(
                run_id,
                agent_id,
                "batch_completed",
                {"calls": len(batch["items"]), "step": step},
            )
        if warning_text:
            self._action_error(run_id, agent_id, warning_text, config)
        else:
            self.store.update_agent(run_id, agent_id, protocol_errors=0)

    async def _check_completion(
        self, run_id: str, agent_id: str, result: str, key: str, config: RuntimeConfig
    ) -> bool:
        if self.completion_validator is None or (
            self.completion_root_only and agent_id != "root"
        ):
            return True
        check_key, decision_key = (
            key + "/completion_check",
            key + "/completion_decision",
        )
        outcome = self.store.effect(check_key)
        if outcome is None:
            if self.store.agent(run_id, agent_id).action_phase == "validating":
                self._fail(
                    run_id,
                    agent_id,
                    "interrupted completion validation has unknown outcome; callback replay refused. Inspect the journal before starting a new checked task",
                )
                return False
            with self.store.transaction():
                self.store.update_agent(
                    run_id, agent_id, status="executing", action_phase="validating"
                )
                self.store.event(
                    run_id,
                    agent_id,
                    "completion_started",
                    {"step": self.store.agent(run_id, agent_id).steps},
                )
            actor_token = tool_actor.set((run_id, agent_id))
            try:
                check = await asyncio.wait_for(
                    self.completion_validator(run_id, agent_id, result),
                    timeout=config.completion_timeout_seconds,
                )
                if (
                    not isinstance(check, CompletionCheck)
                    or not isinstance(check.passed, bool)
                    or not isinstance(check.feedback, str)
                ):
                    raise TypeError(
                        "completion validator must return CompletionCheck with bool passed and string feedback"
                    )
                if len(check.feedback) > config.max_stored_tool_chars:
                    raise ValueError(
                        "completion validator feedback exceeds journal storage limit"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - operator validators need bounded, durable failure diagnostics.
                check = CompletionCheck(
                    False, f"Completion validator failed: {type(exc).__name__}: {exc}"
                )
            finally:
                tool_actor.reset(actor_token)
            self._assert_lease(run_id)
            outcome = {"passed": check.passed, "feedback": check.feedback}
            with self.store.transaction():
                self.store.record_effect(check_key, outcome)
                self.store.event(run_id, agent_id, "completion_checked", outcome)
        if outcome["passed"]:
            self.store.update_agent(run_id, agent_id, completion_passed=True)
            return True
        if self.store.effect(decision_key) is not None:
            return False
        rejected = self.store.agent(run_id, agent_id).completion_rejections + 1
        feedback = (
            outcome["feedback"] or "Configured acceptance criteria have not passed."
        )
        with self.store.transaction():
            self.store.record_effect(
                decision_key, {"passed": False, "rejections": rejected}
            )
            self.store.update_agent(
                run_id,
                agent_id,
                completion_rejections=rejected,
                completion_passed=False,
            )
            self.store.event(
                run_id,
                agent_id,
                "completion_rejected",
                {"rejections": rejected, "feedback": feedback},
            )
            message = (
                "Finish rejected by configured acceptance checks. Use the actual diagnostic to complete remaining work; no accepted final result has been recorded.\n"
                + self._context_excerpt(feedback, config.max_tool_result_chars)
            )
            failed = rejected >= config.max_completion_rejections
            self.store.complete_action(
                run_id,
                agent_id,
                message,
                status="failed" if failed else "runnable",
                feedback_role="tool" if self.native_mode else "user",
            )
            if failed:
                error = (
                    f"Completion criteria failed after {rejected} rejected finishes: "
                    + self._context_excerpt(feedback, config.max_tool_result_chars)
                )
                self.store.update_agent(run_id, agent_id, error=error)
                self.store.event(run_id, agent_id, "failed", {"error": error})
        return False

    async def _apply_action(
        self,
        run_id: str,
        agent_id: str,
        action: dict[str, Any],
        config: RuntimeConfig,
        *,
        recovering: bool = False,
        effect_key: str | None = None,
        preserve_pending: bool = False,
    ) -> None:
        self._assert_lease(run_id)
        agent = self.store.agent(run_id, agent_id)
        kind, key = (
            action["action"],
            effect_key or f"{run_id}/{agent_id}/{agent.steps}/effect",
        )
        agents = {item.agent_id: item for item in self.store.agents(run_id)}
        children = {
            item.agent_id: item
            for item in agents.values()
            if item.parent_id == agent_id
        }
        try:
            if kind == "spawn":
                if len(agents) >= config.max_agents:
                    raise ProtocolError(
                        "global agent limit reached; use existing agents or finish"
                    )
                if agent.depth >= config.max_depth:
                    raise ProtocolError(
                        "hierarchy depth limit reached; solve this assigned task directly"
                    )
                dependencies = action.get("dependencies", [])
                if any(dep not in children for dep in dependencies):
                    raise ProtocolError(
                        "dependencies must reference your existing direct children"
                    )
                caller_tools = self._allowed_tools(run_id, agent_id)
                requested = (
                    frozenset(action["tools"])
                    if "tools" in action
                    else caller_tools - self.mutating_tools
                )
                if requested - caller_tools:
                    raise ProtocolError(
                        "child tools must be a subset of your own granted tools; cannot escalate permissions"
                    )
                granted = requested | {"journal_read"}
                self.store.spawn(
                    run_id,
                    agent_id,
                    key,
                    action["task"],
                    action.get("role", "Complete the assigned task using evidence."),
                    dependencies,
                    self._system_prompt(
                        config,
                        task=action["task"],
                        allowed_tools=granted,
                        is_subagent=True,
                    ),
                    inherited_context=self._inherited_goals(run_id, agent_id),
                    tools=sorted(granted),
                    feedback_role="tool" if self.native_mode else "user",
                    preserve_pending=preserve_pending,
                )
            elif kind == "send":
                recipient = agents.get(action["to"])
                if recipient is None or recipient.status in TERMINAL:
                    raise ProtocolError("recipient must be an active agent in this run")
                if recipient.agent_id == agent_id:
                    raise ProtocolError(
                        "sending messages to yourself is not useful; use a tool or finish"
                    )
                with self.store.transaction():
                    if self.store.effect(key) is None:
                        self.store.add_message(
                            run_id,
                            recipient.agent_id,
                            "user",
                            f"Message from {agent_id}:\n{action['message']}",
                        )
                        self.store.record_effect(key, {"to": recipient.agent_id})
                        self.store.event(
                            run_id,
                            agent_id,
                            "send",
                            {"to": recipient.agent_id, "message": action["message"]},
                        )
                    self.store.complete_action(
                        run_id,
                        agent_id,
                        f"Message delivered durably to {recipient.agent_id}; it will be included in that agent's next model turn.",
                        feedback_role="tool" if self.native_mode else "user",
                    )
            elif kind == "wait":
                ids = action["agents"]
                if any(child not in children for child in ids):
                    raise ProtocolError("wait accepts only your direct child IDs")
                if all(
                    children[child].status in TERMINAL
                    and self.store.effect(f"{run_id}/{agent_id}/outcome/{child}")
                    is not None
                    for child in ids
                ):
                    raise ProtocolError(
                        "these terminal child outcomes were already delivered; synthesize them, finish, or do new useful work instead of waiting again"
                    )
                with self.store.transaction():
                    self.store.complete_action(
                        run_id,
                        agent_id,
                        "Waiting for child outcomes: " + ", ".join(ids),
                        status="waiting",
                        wait_for=ids,
                        feedback_role="tool" if self.native_mode else "user",
                    )
                    self.store.event(run_id, agent_id, "wait", {"agents": ids})
            elif kind == "finish":
                unsettled = [
                    child.agent_id
                    for child in children.values()
                    if child.status not in TERMINAL
                ]
                if unsettled:
                    raise ProtocolError(
                        "children still active: "
                        + ", ".join(unsettled)
                        + "; wait for their outcomes before synthesis"
                    )
                unseen = [
                    child.agent_id
                    for child in children.values()
                    if self.store.effect(
                        f"{run_id}/{agent_id}/outcome/{child.agent_id}"
                    )
                    is None
                ]
                if unseen:
                    raise ProtocolError(
                        "child outcomes have not been delivered: "
                        + ", ".join(unseen)
                        + "; use wait to read them before synthesis"
                    )
                if not await self._check_completion(
                    run_id, agent_id, action["result"], key, config
                ):
                    return
                with self.store.transaction():
                    self.store.update_agent(
                        run_id,
                        agent_id,
                        status="completed",
                        result=action["result"],
                        pending_action=None,
                        action_phase=None,
                    )
                    self.store.event(
                        run_id, agent_id, "finish", {"result": action["result"]}
                    )
            else:
                name = action["name"]
                if (
                    name not in self.tools and name != "journal_read"
                ) or name not in self._allowed_tools(run_id, agent_id):
                    raise ProtocolError(
                        "tool is not in this agent's granted allowlist: " + name
                    )
                replay_safe = name == "journal_read" or name in self.replay_safe_tools
                if recovering and not replay_safe:
                    self._fail(
                        run_id,
                        agent_id,
                        f"interrupted tool {name}: outcome unknown; automatic replay refused. Inspect the journal and external state before starting a new task",
                    )
                    return
                self.store.update_agent(
                    run_id,
                    agent_id,
                    status="executing",
                    action_phase="batch" if preserve_pending else "executing",
                )
                self.store.event(
                    run_id,
                    agent_id,
                    "tool_started",
                    {"name": name, "args": action["args"], "replay": recovering},
                )

                async def invoke() -> str:
                    actor_token = tool_actor.set((run_id, agent_id))
                    try:
                        return await invoke_with_context()
                    finally:
                        tool_actor.reset(actor_token)

                async def invoke_with_context() -> str:
                    function = self.tools.get(name)
                    if name == "journal_read":
                        value = self._read_journal(run_id, agent_id, action["args"])
                    elif inspect.iscoroutinefunction(function):
                        value = await function(action["args"])
                    else:
                        value = await asyncio.to_thread(function, action["args"])
                        if inspect.isawaitable(value):
                            value = await value
                    if not isinstance(value, str):
                        value = json.dumps(value, ensure_ascii=False)
                    if len(value) > config.max_stored_tool_chars:
                        raise ValueError(
                            f"tool result exceeds storage limit {config.max_stored_tool_chars}; narrow the query"
                        )
                    return value

                try:
                    value = await asyncio.wait_for(
                        invoke(), timeout=config.tool_timeout_seconds
                    )
                    self._assert_lease(run_id)
                except asyncio.CancelledError:
                    raise
                except LeaseError:
                    raise
                except Exception as exc:  # noqa: BLE001 - tool plugin failures need durable outcomes and bounded recovery.
                    # A timed-out writing tool may still have taken effect.
                    # Stop that agent; read-only failures can be handled by it.
                    if (
                        not replay_safe
                        and getattr(exc, "work_started", None) is not False
                    ):
                        self._fail(
                            run_id,
                            agent_id,
                            f"tool {name} failed; outcome may be unknown: {type(exc).__name__}: {exc}",
                        )
                        return
                    value = f"Tool failed: {type(exc).__name__}: {exc}"
                with self.store.transaction():
                    self.store.record_effect(key, {"name": name, "result": value})
                    self.store.event(
                        run_id,
                        agent_id,
                        "tool_result",
                        {"name": name, "args": action["args"], "result": value},
                    )
                    self.store.complete_action(
                        run_id,
                        agent_id,
                        f"Tool {name} result:\n"
                        + self._context_excerpt(value, config.max_tool_result_chars),
                        feedback_role="tool" if self.native_mode else "user",
                        preserve_pending=preserve_pending,
                    )
                repeats = self.store.identical_tool_results(
                    run_id,
                    agent_id,
                    name,
                    action["args"],
                    value,
                    config.max_identical_tool_results,
                )
                if repeats >= config.max_identical_tool_results:
                    hint = ""
                    if name == "journal_read":
                        try:
                            page = json.loads(value)
                            hint = f" Returned next_after_event_id={page.get('next_after_event_id')}; for a truncated event use its next_offset with limit=1."
                        except (ValueError, AttributeError):
                            hint = ""
                    feedback = (
                        f"The same {name} query returned the same result {repeats} times with no new evidence. Use the current diagnostic, choose a new query/cursor, or finish instead of repeating it."
                        + hint
                    )
                    if preserve_pending:
                        with self.store.transaction():
                            self.store.record_effect(
                                key + "/no_progress", {"feedback": feedback}
                            )
                            self.store.event(
                                run_id,
                                agent_id,
                                "batch_no_progress",
                                {"effect_key": key, "feedback": feedback},
                            )
                        return
                    self._action_error(
                        run_id,
                        agent_id,
                        feedback,
                        config,
                    )
                    return
            if not preserve_pending:
                self.store.update_agent(run_id, agent_id, protocol_errors=0)
        except ProtocolError as exc:
            self._action_error(run_id, agent_id, str(exc), config)
