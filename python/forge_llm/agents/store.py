"""SQLite journal for one coordinator and durable, independently scoped agents."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import ChatMessage

TERMINAL = frozenset({"completed", "failed", "cancelled"})


class StoreError(RuntimeError):
    pass


class LeaseError(StoreError):
    pass


class BudgetError(StoreError):
    pass


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    run_id: str
    parent_id: str | None
    task: str
    role: str
    depth: int
    status: str
    steps: int
    protocol_errors: int
    backend_attempts: int
    dependencies: tuple[str, ...]
    wait_for: tuple[str, ...]
    result: str | None
    error: str | None
    pending_action: dict[str, Any] | None
    action_phase: str | None
    tools: tuple[str, ...] | None = None
    completion_rejections: int = 0
    completion_passed: bool | None = None


class AgentStore:
    """Synchronous journal methods are short transactions on the event-loop thread.

    A lease prevents two coordinators from replaying one run.  Generation
    reservations survive crashes; uncertain work stays charged to the budget.
    Actions that create children or deliver messages are idempotent by turn ID.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.connection.close()
            raise StoreError(f"unsupported journal version: {version}")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, status TEXT NOT NULL, config TEXT NOT NULL,
                created REAL NOT NULL, deadline REAL NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                uncertain_tokens INTEGER NOT NULL DEFAULT 0,
                reserved_tokens INTEGER NOT NULL DEFAULT 0,
                next_agent INTEGER NOT NULL DEFAULT 1,
                lease_owner TEXT, lease_until REAL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id),
                parent_id TEXT, task TEXT NOT NULL, role TEXT NOT NULL,
                depth INTEGER NOT NULL, status TEXT NOT NULL,
                steps INTEGER NOT NULL DEFAULT 0,
                protocol_errors INTEGER NOT NULL DEFAULT 0,
                backend_attempts INTEGER NOT NULL DEFAULT 0,
                dependencies TEXT NOT NULL DEFAULT '[]',
                wait_for TEXT NOT NULL DEFAULT '[]', result TEXT, error TEXT,
                pending_action TEXT, action_phase TEXT,
                PRIMARY KEY(run_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                created REAL NOT NULL,
                FOREIGN KEY(run_id, agent_id) REFERENCES agents(run_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                agent_id TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS effects (
                effect_key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reservations (
                key TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                amount INTEGER NOT NULL, state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generation_inputs (
                request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                messages TEXT NOT NULL, history_length INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_agent ON messages(run_id, agent_id, id);
            CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
            PRAGMA user_version=1;
        """)
        with self.transaction():
            columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(agents)")
            }
            if "tools" not in columns:
                self.connection.execute("ALTER TABLE agents ADD COLUMN tools TEXT")
            if "completion_rejections" not in columns:
                self.connection.execute(
                    "ALTER TABLE agents ADD COLUMN completion_rejections INTEGER NOT NULL DEFAULT 0"
                )
            if "completion_passed" not in columns:
                self.connection.execute(
                    "ALTER TABLE agents ADD COLUMN completion_passed INTEGER"
                )
            run_columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(runs)")
            }
            if "completion_required" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN completion_required INTEGER NOT NULL DEFAULT 0"
                )
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN completion_root_only INTEGER NOT NULL DEFAULT 1"
                )
            input_columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(generation_inputs)"
                )
            }
            if "tool_specs" not in input_columns:
                self.connection.execute(
                    "ALTER TABLE generation_inputs ADD COLUMN tool_specs TEXT"
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> AgentStore:  # noqa: PYI034 - stdlib typing.Self requires Python 3.11; Forge supports 3.10.
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def event(self, run_id: str, agent_id: str | None, kind: str, payload: Any) -> None:
        self.connection.execute(
            "INSERT INTO events(run_id,agent_id,kind,payload,created) VALUES(?,?,?,?,?)",
            (
                run_id,
                agent_id,
                kind,
                json.dumps(payload, ensure_ascii=False),
                time.time(),
            ),
        )

    def events(
        self,
        run_id: str,
        *,
        after_id: int = 0,
        agent_id: str | None = None,
        limit: int | None = None,
        kinds: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        where, values = "run_id=? AND id>?", [run_id, after_id]
        if agent_id is not None:
            where += " AND agent_id=?"
            values.append(agent_id)
        if kinds is not None:
            if not kinds:
                return []
            where += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            values.extend(kinds)
        sql = f"SELECT * FROM events WHERE {where} ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            values.append(limit)
        return [
            dict(row, payload=json.loads(row["payload"]))
            for row in self.connection.execute(sql, values)
        ]

    def recent_events(
        self, run_id: str, agent_id: str, *, limit: int = 12
    ) -> list[dict[str, Any]]:
        rows = list(
            self.connection.execute(
                "SELECT * FROM events WHERE run_id=? AND agent_id=? AND kind IN ('tool_result','spawn','send','children_settled','action_rejected') ORDER BY id DESC LIMIT ?",
                (run_id, agent_id, limit),
            )
        )
        return [dict(row, payload=json.loads(row["payload"])) for row in reversed(rows)]

    def recent_tool_results(
        self, run_id: str, agent_id: str, *, limit: int = 64
    ) -> list[dict[str, Any]]:
        """Bounded, newest-first receipts scoped to exactly one durable agent."""
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id=? AND agent_id=? AND kind='tool_result' ORDER BY id DESC LIMIT ?",
            (run_id, agent_id, limit),
        )
        return [dict(row, payload=json.loads(row["payload"])) for row in rows]

    def create_run(
        self,
        run_id: str,
        task: str,
        config: dict[str, Any],
        system: str,
        *,
        tools: list[str] | None = None,
        completion_required: bool = False,
        completion_root_only: bool = True,
    ) -> None:
        with self.transaction():
            now = time.time()
            self.connection.execute(
                "INSERT INTO runs(run_id,status,config,created,deadline) VALUES(?,?,?,?,?)",
                (
                    run_id,
                    "running",
                    json.dumps(config),
                    now,
                    now + config["deadline_seconds"],
                ),
            )
            self.connection.execute(
                "INSERT INTO agents(agent_id,run_id,task,role,depth,status) VALUES(?,?,?,?,?,?)",
                (
                    "root",
                    run_id,
                    task,
                    "Solve the user's task and synthesize evidence from delegated work.",
                    0,
                    "runnable",
                ),
            )
            self.add_message(run_id, "root", "system", system)
            self.add_message(run_id, "root", "user", task)
            if tools is not None:
                self.update_agent(run_id, "root", tools=tools)
            self.connection.execute(
                "UPDATE runs SET completion_required=?,completion_root_only=? WHERE run_id=?",
                (int(completion_required), int(completion_root_only), run_id),
            )
            self.event(run_id, "root", "run_created", {"task": task, "config": config})

    def run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown run: {run_id}")
        return dict(row, config=json.loads(row["config"]))

    def acquire_lease(self, run_id: str, owner: str, seconds: float) -> None:
        with self.transaction():
            run = self.run(run_id)
            now = time.time()
            if (
                run["lease_owner"] not in (None, owner)
                and (run["lease_until"] or 0) > now
            ):
                raise LeaseError(f"run {run_id} already has an active coordinator")
            self.connection.execute(
                "UPDATE runs SET lease_owner=?,lease_until=? WHERE run_id=?",
                (owner, now + seconds, run_id),
            )

    def renew_lease(self, run_id: str, owner: str, seconds: float) -> None:
        cursor = self.connection.execute(
            "UPDATE runs SET lease_until=? WHERE run_id=? AND lease_owner=? AND lease_until>?",
            (time.time() + seconds, run_id, owner, time.time()),
        )
        if cursor.rowcount != 1:
            raise LeaseError(f"coordinator lease lost for {run_id}")

    def release_lease(self, run_id: str, owner: str) -> None:
        self.connection.execute(
            "UPDATE runs SET lease_owner=NULL,lease_until=NULL WHERE run_id=? AND lease_owner=?",
            (run_id, owner),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> AgentRecord:
        return AgentRecord(
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            parent_id=row["parent_id"],
            task=row["task"],
            role=row["role"],
            depth=row["depth"],
            status=row["status"],
            steps=row["steps"],
            protocol_errors=row["protocol_errors"],
            backend_attempts=row["backend_attempts"],
            dependencies=tuple(json.loads(row["dependencies"])),
            wait_for=tuple(json.loads(row["wait_for"])),
            result=row["result"],
            error=row["error"],
            pending_action=json.loads(row["pending_action"])
            if row["pending_action"]
            else None,
            action_phase=row["action_phase"],
            tools=tuple(json.loads(row["tools"])) if row["tools"] is not None else None,
            completion_rejections=row["completion_rejections"],
            completion_passed=bool(row["completion_passed"])
            if row["completion_passed"] is not None
            else None,
        )

    def agent(self, run_id: str, agent_id: str) -> AgentRecord:
        row = self.connection.execute(
            "SELECT * FROM agents WHERE run_id=? AND agent_id=?", (run_id, agent_id)
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown agent: {agent_id}")
        return self._record(row)

    def agents(self, run_id: str) -> tuple[AgentRecord, ...]:
        return tuple(
            self._record(row)
            for row in self.connection.execute(
                "SELECT * FROM agents WHERE run_id=? ORDER BY rowid", (run_id,)
            )
        )

    def update_agent(self, run_id: str, agent_id: str, **changes: Any) -> None:
        allowed = {
            "status",
            "steps",
            "protocol_errors",
            "backend_attempts",
            "wait_for",
            "result",
            "error",
            "pending_action",
            "action_phase",
            "tools",
            "completion_rejections",
            "completion_passed",
        }
        if not changes or changes.keys() - allowed:
            raise StoreError("invalid agent update")
        fields, values = [], []
        for name, value in changes.items():
            fields.append(name + "=?")
            values.append(
                json.dumps(value, ensure_ascii=False)
                if name in {"wait_for", "pending_action", "tools"} and value is not None
                else value
            )
        self.connection.execute(
            f"UPDATE agents SET {','.join(fields)} WHERE run_id=? AND agent_id=?",
            (*values, run_id, agent_id),
        )

    def add_message(self, run_id: str, agent_id: str, role: str, content: str) -> None:
        self.connection.execute(
            "INSERT INTO messages(run_id,agent_id,role,content,created) VALUES(?,?,?,?,?)",
            (run_id, agent_id, role, content, time.time()),
        )

    def messages(self, run_id: str, agent_id: str) -> list[ChatMessage]:
        return [
            ChatMessage(row["role"], row["content"])
            for row in self.connection.execute(
                "SELECT role,content FROM messages WHERE run_id=? AND agent_id=? ORDER BY id",
                (run_id, agent_id),
            )
        ]

    def generation_input(
        self, run_id: str, request_id: str
    ) -> tuple[list[ChatMessage], int] | None:
        row = self.connection.execute(
            "SELECT messages,history_length FROM generation_inputs WHERE request_id=? AND run_id=?",
            (request_id, run_id),
        ).fetchone()
        if row is None:
            return None
        return [ChatMessage(**item) for item in json.loads(row["messages"])], row[
            "history_length"
        ]

    def save_generation_input(
        self,
        run_id: str,
        request_id: str,
        messages: list[ChatMessage],
        history_length: int,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO generation_inputs(request_id,run_id,messages,history_length,tool_specs) VALUES(?,?,?,?,?)",
            (
                request_id,
                run_id,
                json.dumps(
                    [{"role": item.role, "content": item.content} for item in messages],
                    ensure_ascii=False,
                ),
                history_length,
                json.dumps(tool_specs, ensure_ascii=False)
                if tool_specs is not None
                else None,
            ),
        )

    def generation_tool_specs(
        self, run_id: str, request_id: str
    ) -> list[dict[str, Any]] | None:
        row = self.connection.execute(
            "SELECT tool_specs FROM generation_inputs WHERE run_id=? AND request_id=?",
            (run_id, request_id),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def identical_tool_results(
        self,
        run_id: str,
        agent_id: str,
        name: str,
        args: dict[str, Any],
        result: str,
        limit: int,
    ) -> int:
        rows = self.connection.execute(
            "SELECT payload FROM events WHERE run_id=? AND agent_id=? AND kind='tool_result' ORDER BY id DESC LIMIT ?",
            (run_id, agent_id, 64),
        )
        count = 0
        for row in rows:
            payload = json.loads(row[0])
            if payload.get("name") != name or payload.get("args") != args:
                continue
            if payload.get("result") != result:
                break
            count += 1
            if count >= limit:
                break
        return count

    def effect(self, key: str) -> Any | None:
        row = self.connection.execute(
            "SELECT value FROM effects WHERE effect_key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def record_effect(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO effects(effect_key,value) VALUES(?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def spawn(
        self,
        run_id: str,
        parent_id: str,
        key: str,
        task: str,
        role: str,
        dependencies: list[str],
        system: str,
        inherited_context: str = "",
        tools: list[str] | None = None,
        feedback_role: str = "user",
        preserve_pending: bool = False,
    ) -> str:
        with self.transaction():
            existing = self.effect(key)
            if existing is not None:
                return str(existing["agent_id"])
            parent, run = self.agent(run_id, parent_id), self.run(run_id)
            agent_id = f"agent-{run['next_agent']:04d}"
            self.connection.execute(
                "UPDATE runs SET next_agent=next_agent+1 WHERE run_id=?", (run_id,)
            )
            self.connection.execute(
                "INSERT INTO agents(agent_id,run_id,parent_id,task,role,depth,status,dependencies) VALUES(?,?,?,?,?,?,?,?)",
                (
                    agent_id,
                    run_id,
                    parent_id,
                    task,
                    role,
                    parent.depth + 1,
                    "queued" if dependencies else "runnable",
                    json.dumps(dependencies),
                ),
            )
            self.add_message(run_id, agent_id, "system", system)
            self.add_message(
                run_id,
                agent_id,
                "user",
                inherited_context
                + f"\nParent: {parent_id}\nAssigned role: {role}\nYOUR ASSIGNED SCOPE: {task}",
            )
            if tools is not None:
                self.update_agent(run_id, agent_id, tools=tools)
            self.record_effect(key, {"agent_id": agent_id})
            self.event(
                run_id,
                parent_id,
                "spawn",
                {
                    "agent_id": agent_id,
                    "task": task,
                    "role": role,
                    "dependencies": dependencies,
                    "tools": tools,
                },
            )
            self.complete_action(
                run_id,
                parent_id,
                f"Spawned {agent_id}. It has its own task context. Use send to share additional evidence; wait before synthesis.",
                feedback_role=feedback_role,
                preserve_pending=preserve_pending,
            )
            return agent_id

    def complete_action(
        self,
        run_id: str,
        agent_id: str,
        feedback: str,
        *,
        status: str = "runnable",
        wait_for: list[str] | None = None,
        feedback_role: str = "user",
        preserve_pending: bool = False,
    ) -> None:
        self.add_message(run_id, agent_id, feedback_role, feedback)
        self.update_agent(
            run_id,
            agent_id,
            status=status,
            pending_action=self.agent(run_id, agent_id).pending_action
            if preserve_pending
            else None,
            action_phase="batch" if preserve_pending else None,
            wait_for=wait_for or [],
        )

    def reserve(self, run_id: str, key: str, amount: int, budget: int) -> None:
        with self.transaction():
            run = self.run(run_id)
            total = (
                run["input_tokens"]
                + run["output_tokens"]
                + run["uncertain_tokens"]
                + run["reserved_tokens"]
            )
            if total + amount > budget:
                raise BudgetError(
                    f"global token budget exhausted: {total} committed/reserved + {amount} requested > {budget}"
                )
            self.connection.execute(
                "INSERT INTO reservations(key,run_id,amount,state) VALUES(?,?,?,'reserved')",
                (key, run_id, amount),
            )
            self.connection.execute(
                "UPDATE runs SET reserved_tokens=reserved_tokens+? WHERE run_id=?",
                (amount, run_id),
            )

    def settle(
        self, run_id: str, key: str, input_tokens: int, output_tokens: int
    ) -> None:
        with self.transaction():
            row = self.connection.execute(
                "SELECT amount,state FROM reservations WHERE key=? AND run_id=?",
                (key, run_id),
            ).fetchone()
            if row is None or row["state"] != "reserved":
                raise StoreError("missing or already settled token reservation")
            if input_tokens + output_tokens > row["amount"]:
                raise BudgetError(
                    "backend reported more tokens than its conservative reservation"
                )
            self.connection.execute(
                "UPDATE reservations SET state='settled' WHERE key=?", (key,)
            )
            self.connection.execute(
                "UPDATE runs SET reserved_tokens=reserved_tokens-?,input_tokens=input_tokens+?,output_tokens=output_tokens+? WHERE run_id=?",
                (row["amount"], input_tokens, output_tokens, run_id),
            )

    def uncertain(self, run_id: str, key: str) -> None:
        with self.transaction():
            row = self.connection.execute(
                "SELECT amount,state FROM reservations WHERE key=? AND run_id=?",
                (key, run_id),
            ).fetchone()
            if row is not None and row["state"] == "reserved":
                self.connection.execute(
                    "UPDATE reservations SET state='uncertain' WHERE key=?", (key,)
                )
                self.connection.execute(
                    "UPDATE runs SET reserved_tokens=reserved_tokens-?,uncertain_tokens=uncertain_tokens+? WHERE run_id=?",
                    (row["amount"], row["amount"], run_id),
                )

    def recover(self, run_id: str) -> None:
        """Called only after obtaining the run lease; preserve pending actions."""
        for row in list(
            self.connection.execute(
                "SELECT key FROM reservations WHERE run_id=? AND state='reserved'",
                (run_id,),
            )
        ):
            self.uncertain(run_id, row[0])
        with self.transaction():
            self.connection.execute(
                "UPDATE agents SET status='runnable' WHERE run_id=? AND status IN ('running','executing')",
                (run_id,),
            )
            self.event(
                run_id,
                None,
                "recovered",
                {"uncertain_tokens": self.run(run_id)["uncertain_tokens"]},
            )

    def request_cancel(self, run_id: str) -> None:
        self.run(run_id)
        self.connection.execute(
            "UPDATE runs SET cancel_requested=1 WHERE run_id=?", (run_id,)
        )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        with self.transaction():
            self.connection.execute(
                "UPDATE runs SET status=?,error=? WHERE run_id=?",
                (status, error, run_id),
            )
            self.event(run_id, None, "run_finished", {"status": status, "error": error})
