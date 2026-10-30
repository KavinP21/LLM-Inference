"""Each model call's context: history, journal excerpts and compaction."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..native_tools import NoNativeCall, parse_native_actions
from ..protocol import (
    ChatMessage,
    ProtocolError,
    conservative_input_tokens,
)
from ..store import (
    BudgetError,
    LeaseError,
)
from .config import (
    COMPACT_AT,
    COMPACT_REQUEST,
    COMPACT_SUMMARY_CHARS,
    COMPACT_SUMMARY_TOKENS,
    COMPACT_TAIL,
    SUMMARY_PREFIX,
    RuntimeConfig,
)


class ContextMixin:
    @staticmethod
    def _byte_excerpt(text: str, byte_limit: int, marker: str) -> str:
        data = text.encode("utf-8")
        if len(data) <= byte_limit:
            return text
        marker_bytes = marker.encode("utf-8")
        if byte_limit < len(marker_bytes):
            return marker_bytes[:byte_limit].decode("utf-8", errors="ignore")
        room = byte_limit - len(marker_bytes)
        head, tail = (room + 1) // 2, room // 2
        return (
            data[:head].decode("utf-8", errors="ignore")
            + marker
            + (data[-tail:].decode("utf-8", errors="ignore") if tail else "")
        )

    def _input_bound(
        self, run_id: str, agent_id: str, messages: list[ChatMessage]
    ) -> int:
        """Upper bound on the prompt tokens ``messages`` will take.

        UTF-8 bytes are a safe bound but overcount code about 3.5x, which made
        agents drop history they had room for. When the backend has measured
        an earlier request of this agent whose messages are a prefix of these,
        only what was appended since needs the byte bound.
        """
        bound = conservative_input_tokens(messages)
        if not self.native_mode:
            return bound
        event = self.store.last_event(run_id, agent_id, "generation")
        saved = event and self.store.generation_input(
            run_id, event["payload"]["request_id"]
        )
        if not saved:
            return bound
        sent = saved[0]
        if len(sent) > len(messages) or any(
            old.role != new.role
            or old.content != new.content
            or [item["sha256"] for item in old.media]
            != [item["sha256"] for item in new.media]
            for old, new in zip(sent, messages)
        ):
            return bound
        measured = event["payload"]["input_tokens"] + conservative_input_tokens(
            messages[len(sent) :]
        )
        return min(bound, measured)

    def _history(
        self, run_id: str, agent_id: str, config: RuntimeConfig
    ) -> list[ChatMessage]:
        history = self.store.messages(run_id, agent_id)
        if self.native_mode:
            agent = self.store.agent(run_id, agent_id)
            history[0] = ChatMessage(
                "system",
                self._system_prompt(
                    config,
                    task=agent.task,
                    allowed_tools=self._allowed_tools(run_id, agent_id),
                    is_subagent=agent.parent_id is not None,
                ),
            )
        return history

    async def _maybe_compact(
        self,
        run_id: str,
        agent_id: str,
        config: RuntimeConfig,
        tool_specs: list[dict[str, Any]] | None,
    ) -> None:
        """Replace older messages with a summary the model writes, once the
        context is mostly full.

        Projection alone would rebuild the window on every step, so the prompt
        would change near its start and miss the prompt cache each time. A
        compaction changes it once; after that it grows by appending again.
        The summary request itself extends the cached prompt, so it only costs
        the summary's own tokens.
        """
        schema = self._schema_input_bound(tool_specs)
        budget = config.max_context_tokens - config.max_output_tokens - schema
        summary_tokens = min(COMPACT_SUMMARY_TOKENS, config.max_output_tokens)
        # Compact at COMPACT_AT, or sooner if what's left couldn't take one
        # more tool result plus the summary request: past that point there's
        # no room to compact and projection would take over.
        threshold = min(
            budget * COMPACT_AT,
            config.max_context_tokens
            - schema
            - summary_tokens
            - len(COMPACT_REQUEST.encode())
            - config.max_tool_result_chars,
        )
        history = self._history(run_id, agent_id, config)
        view = self._compacted(run_id, agent_id, history)
        if self._input_bound(run_id, agent_id, view) <= threshold:
            return
        floor = max(self._anchor_indices(run_id, agent_id, history)) + 1
        previous = self.store.last_event(run_id, agent_id, "context_compacted")
        if previous is not None:
            floor = max(floor, previous["payload"]["keep_from"])
        # Keep recent messages, starting at an assistant message so tool
        # results stay with their calls. Sizes are in bytes because nothing
        # measured them on their own yet.
        keep_from, size = len(history), 0
        for index in range(len(history) - 1, floor - 1, -1):
            size += len(history[index].content.encode("utf-8")) + 32
            if size > budget * COMPACT_TAIL:
                break
            if history[index].role == "assistant":
                keep_from = index
        request = [*view, ChatMessage("user", COMPACT_REQUEST)]
        if (
            keep_from <= floor
            or self._input_bound(run_id, agent_id, request) + schema + summary_tokens
            > config.max_context_tokens
        ):
            return  # Nothing to drop, or too late: projection handles it.
        agent = self.store.agent(run_id, agent_id)
        request_id = f"{run_id}.{agent_id}.{agent.steps}.compact"
        options = self._backend_options(run_id, agent_id, config, request)
        reservation = request_id + "/reservation"
        async with self._semaphore, self._run_semaphores[run_id]:
            self._assert_lease(run_id)
            try:
                self.store.reserve(
                    run_id,
                    reservation,
                    self._input_bound(run_id, agent_id, request)
                    + schema
                    + summary_tokens,
                    config.max_total_tokens,
                )
            except BudgetError:
                return
            self._requests[(run_id, agent_id)] = request_id
            try:
                generation = await asyncio.wait_for(
                    self.backend.generate_action(
                        request,
                        summary_tokens,
                        request_id,
                        tool_specs=tool_specs,
                        **options,
                    ),
                    timeout=config.generation_timeout_seconds,
                )
                self._assert_lease(run_id)
                self.store.settle(
                    run_id,
                    reservation,
                    generation.input_tokens,
                    generation.output_tokens,
                )
            except (asyncio.CancelledError, LeaseError):
                self.store.uncertain(run_id, reservation)
                raise
            except Exception as exc:  # noqa: BLE001 - compaction is best effort.
                self.store.uncertain(run_id, reservation)
                self.store.event(
                    run_id,
                    agent_id,
                    "compaction_failed",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            finally:
                self._requests.pop((run_id, agent_id), None)
        summary = generation.text.strip()
        try:
            parse_native_actions(
                summary, tool_specs or [], dialect=generation.dialect or "json"
            )
        except NoNativeCall:
            pass
        except ProtocolError:
            summary = ""  # A malformed call is not a summary either.
        else:
            summary = ""
        if not summary:
            self.store.event(
                run_id,
                agent_id,
                "compaction_failed",
                {"error": "the model did not reply with a plain-text summary"},
            )
            return
        self.store.event(
            run_id,
            agent_id,
            "context_compacted",
            {
                "keep_from": keep_from,
                "summary": summary[:COMPACT_SUMMARY_CHARS],
                "messages_summarized": keep_from - floor,
                "input_tokens": generation.input_tokens,
            },
        )

    def _backend_options(
        self,
        run_id: str,
        agent_id: str,
        config: RuntimeConfig,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if config.effort != "off":
            options["options"] = {"effort": config.effort}
        if getattr(self.backend, "routes", False) is True:
            # Root agents run on the driver model and sub-agents on the
            # sub-agent model; each agent sticks to the replica holding its
            # prompt cache.
            agent = self.store.agent(run_id, agent_id)
            options["group"] = (
                config.driver_group
                if agent.parent_id is None
                else config.subagent_group
            )
            options["affinity"] = f"{run_id}/{agent_id}"
        if any(message.media for message in messages) and (
            getattr(self.backend, "supports_context_limits", False) is True
        ):
            options["max_context_tokens"] = config.max_context_tokens
        return options

    def _anchor_indices(
        self, run_id: str, agent_id: str, history: list[ChatMessage]
    ) -> list[int]:
        """Messages every context starts with: the system prompt and the task."""
        return [0, 1]

    def _compacted(
        self, run_id: str, agent_id: str, history: list[ChatMessage]
    ) -> list[ChatMessage]:
        """The history as the model sees it after its latest compaction."""
        event = self.store.last_event(run_id, agent_id, "context_compacted")
        if event is None:
            return history
        payload = event["payload"]
        return [
            *(history[i] for i in self._anchor_indices(run_id, agent_id, history)),
            ChatMessage("user", SUMMARY_PREFIX + payload["summary"]),
            *history[payload["keep_from"] :],
        ]

    def _model_context(
        self,
        run_id: str,
        agent_id: str,
        config: RuntimeConfig,
        *,
        schema_input_bound: int = 0,
    ) -> list[ChatMessage]:
        history = self._history(run_id, agent_id, config)
        if self.native_mode:
            history = self._compacted(run_id, agent_id, history)
        # Native agents get state through persisted notes (see _note_state), so
        # their prompt only ever grows by appending.
        menu = "" if self.native_mode else self._legal_menu(run_id, agent_id, config)

        def with_menu(messages: list[ChatMessage]) -> list[ChatMessage]:
            if not menu:
                return messages
            # Keep the immutable original task unchanged.  Later feedback and
            # the current legal state share the final user message.
            if len(messages) > 2 and messages[-1].role == "user":
                return messages[:-1] + [
                    ChatMessage(
                        "user", messages[-1].content + "\n\n" + menu, messages[-1].media
                    )
                ]
            return messages + [ChatMessage("user", menu)]

        target = (
            config.max_context_tokens
            - config.max_output_tokens
            - schema_input_bound
            - len(menu.encode("utf-8"))
            - 34
        )
        if (
            self._input_bound(run_id, agent_id, history) <= target
            or not config.compact_history
        ):
            return with_menu(history)
        base = history[
            :2
        ]  # Always preserve the complete system and original assigned task.
        remaining = target - conservative_input_tokens(base)
        if remaining < 384:
            return with_menu(
                history
            )  # Caller fails explicitly if immutable task and control state cannot fit.
        receipts = []
        agent = self.store.agent(run_id, agent_id)
        current = [
            item
            for item in self.store.agents(run_id)
            if item.parent_id == agent_id or item.agent_id in agent.dependencies
        ]
        state = (
            "Current children/dependencies: "
            + ", ".join(f"{item.agent_id}={item.status}" for item in current)
            + "."
        )
        for event in self.store.recent_events(run_id, agent_id, limit=8):
            payload = json.dumps(event["payload"], ensure_ascii=False)
            receipts.append(f"event {event['id']} {event['kind']}: {payload[:160]}")
        marker = (
            "History projection: earlier messages are omitted from this model input; complete messages and evidence remain in the durable run journal. Use journal_read to recover referenced events before making claims from missing evidence. "
            + state
            + " Recent receipts (partial):\n"
            + "\n".join(receipts)
        )
        marker_budget = min(1_500, max(300, remaining // 3))
        marker = self._byte_excerpt(
            marker, marker_budget - 32, "\n[receipt list truncated; use journal_read]"
        )
        remaining -= len(marker.encode("utf-8")) + 32
        recent: list[ChatMessage] = []
        for message in reversed(history[2:]):
            size = len(message.content.encode("utf-8")) + 32
            if size <= remaining:
                recent.append(message)
                remaining -= size
            elif remaining >= 200:
                excerpt = self._byte_excerpt(
                    message.content,
                    remaining - 32,
                    "\n[TRUNCATED for model context; full evidence in journal_read]",
                )
                recent.append(ChatMessage(message.role, excerpt, message.media))
                break
            else:
                break
        projected = with_menu(
            base + [ChatMessage("user", marker)] + list(reversed(recent))
        )
        self.store.event(
            run_id,
            agent_id,
            "context_projected",
            {
                "original_messages": len(history),
                "projected_messages": len(projected),
                "conservative_input_tokens": conservative_input_tokens(projected),
                "source_preserved": True,
            },
        )
        return projected

    def _read_journal(self, run_id: str, agent_id: str, args: dict[str, Any]) -> str:
        allowed = {
            "agent_id",
            "after_event_id",
            "limit",
            "max_chars",
            "offset",
            "include_control",
        }
        if args.keys() - allowed:
            raise ValueError("unknown journal_read arguments")
        include_control = args.get("include_control", False)
        if not isinstance(include_control, bool):
            raise TypeError("include_control must be a boolean")
        target = args.get("agent_id", agent_id)
        if not isinstance(target, str):
            raise TypeError("agent_id must be a string")
        agent = self.store.agent(run_id, target)
        if target != agent_id and agent.parent_id != agent_id:
            raise ValueError(
                "journal_read may inspect only your own or a direct child's evidence"
            )
        values = {
            "after_event_id": (0, 0, 2**63 - 1),
            "limit": (3, 1, 5),
            "max_chars": (1500, 256, 4000),
            "offset": (0, 0, 2**31 - 1),
        }
        parsed = {}
        for key, (default, lower, upper) in values.items():
            value = args.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise ValueError(
                    f"{key} must be an integer between {lower} and {upper}"
                )
            parsed[key] = value
        if parsed["offset"] and parsed["limit"] != 1:
            raise ValueError(
                "offset requires limit=1; use after_event_id one less than the desired event ID"
            )
        records = []
        cursor = parsed["after_event_id"]
        evidence_kinds = {
            "tool_result",
            "send",
            "finish",
            "children_settled",
            "failed",
            "cancelled",
            "completion_checked",
            "completion_rejected",
        }
        # Scan a bounded number of actual rows, not only matching rows.  This
        # advances a cursor over metadata and recall copies even on empty pages.
        # Full audit mode preserves all rows and its existing paging behavior.
        for event in self.store.events(
            run_id,
            agent_id=target,
            after_id=parsed["after_event_id"],
            limit=parsed["limit"] if include_control else 64,
        ):
            cursor = event["id"]
            if not include_control and (
                event["kind"] not in evidence_kinds
                or (
                    event["kind"] == "tool_result"
                    and event["payload"].get("name") == "journal_read"
                )
            ):
                continue
            payload = json.dumps(event["payload"], ensure_ascii=False)
            start, end = parsed["offset"], parsed["offset"] + parsed["max_chars"]
            records.append(
                {
                    "event_id": event["id"],
                    "kind": event["kind"],
                    "payload_excerpt": payload[start:end],
                    "payload_chars": len(payload),
                    "offset": start,
                    "next_offset": end if end < len(payload) else None,
                }
            )
            if len(records) == parsed["limit"]:
                break
        return json.dumps(
            {
                "agent_id": target,
                "events": records,
                "next_after_event_id": cursor,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _context_excerpt(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        marker = (
            f"\n[TRUNCATED middle; full {len(text)} characters retained in journal]\n"
        )
        if len(marker) > limit:
            marker = "[TRUNCATED]"[:limit]
        room = limit - len(marker)
        head, tail = (room + 1) // 2, room // 2
        return text[:head] + marker + (text[-tail:] if tail else "")
