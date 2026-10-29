"""Single-loop Studio lifecycle. HTTP threads must dispatch onto its owner loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ..agents.cli import create_pool, worker_identities
from ..agents.profiles import CompletionProfile
from ..agents.protocol import ChatMessage
from ..agents.runtime import AgentRuntime, RuntimeConfig
from ..agents.store import TERMINAL, AgentStore
from ..agents.tools import ToolValidationError, WorkspaceTools
from ..chat_formats import EFFORT_BUDGETS
from .commands import CommandTools
from .conversation import ConversationController, ConversationRuntime
from .demo import DemoBackend
from .extensions import MCPStdioClient
from .policy import ApprovalGate, workspace_preview

MUTATING = frozenset(
    {"write_file", "replace_text", "edit_lines", "save_artifact", "run_tests"}
)
SETTLED_RUNS = TERMINAL | {"completed_with_failures"}


class StudioWorkspaceTools(WorkspaceTools):
    def __init__(self, *args, state_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dir = state_dir

    def _path(self, value, *, artifact=False):
        path = super()._path(value, artifact=artifact)
        if not artifact and (
            path == self.state_dir or path.is_relative_to(self.state_dir)
        ):
            raise ToolValidationError(
                "Studio state is private; use journal_read for run evidence"
            )
        return path

    def _files(self, path):
        for candidate in super()._files(path):
            if not candidate.resolve().is_relative_to(self.state_dir):
                yield candidate


def _text(value: Any, label: str, limit: int = 12_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must contain 1 to {limit} characters")
    return value.strip()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


MODEL_NAMES = {
    "gemma-4-26b-a4b-mlx": "Gemma 4 26B A4B",
    "qwen3.5-4b-mlx": "Qwen3.5-4B",
    "qwen3-30b-a3b-mlx": "Qwen3-30B-A3B",
    "qwen2.5-7b-mlx": "Qwen2.5-7B",
}


def model_label(path: str) -> str:
    name = Path(path).name
    return MODEL_NAMES.get(name, name or "model")


def worker_state(health: dict) -> str:
    if "error" in health:
        return "error"
    if health.get("started") is False:
        return "starting" if health.get("starting") else "stopped"
    if not health.get("loaded", True):
        return "starting"
    return "busy" if health.get("active") else "idle"


class StudioService:
    """Each run gets isolated file receipts, artifacts, policy and completion checks."""

    def __init__(
        self,
        workspace: Path,
        state_dir: Path,
        *,
        config: dict | None = None,
        demo: bool = False,
        backend=None,
        skills=None,
        mcp_clients=(),
        hooks=None,
    ):
        self.workspace = workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        self.state_dir = state_dir.resolve()
        if self.workspace.is_relative_to(self.state_dir):
            raise ValueError("state directory cannot contain the workspace root")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.store = AgentStore(self.state_dir / "studio.sqlite")
        os.chmod(self.state_dir / "studio.sqlite", 0o600)
        self.store.connection.execute(
            "CREATE TABLE IF NOT EXISTS studio_manifests (run_id TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
        )
        self.config = json.loads(json.dumps(config or {}))
        self.demo, self.skills = demo, skills
        self.backend = backend or (DemoBackend(self.store) if demo else None)
        self.backend_status = "ready" if self.backend else "offline"
        self.backend_error: str | None = None
        self.identities: dict = (
            {"demo": {"model": "deterministic-demo"}} if demo else {}
        )
        self.jobs: dict[str, asyncio.Task] = {}
        self.runtimes: dict[str, AgentRuntime] = {}
        self.gates: dict[str, ApprovalGate] = {}
        self.mcp_clients = tuple(mcp_clients)
        self._mcp_configs = tuple(client.config for client in self.mcp_clients)
        self._restart_mcp = False
        self.hooks = hooks
        self.worker_status: list[dict] = []
        self.closing = False
        self._closed = False
        self.conversations = ConversationController(self)

    async def start_backend(self) -> dict:
        if self.closing:
            raise ValueError("Studio is shutting down")
        if self.backend_status == "loading":
            return {"status": "loading"}
        if self.backend_status == "ready":
            return {"status": "ready"}
        if not self.config.get("workers"):
            raise ValueError(
                "Select an operator worker configuration when starting Studio"
            )
        if self._restart_mcp:
            self.mcp_clients = tuple(
                MCPStdioClient(config) for config in self._mcp_configs
            )
            self._restart_mcp = False
        self.backend_status, self.backend_error = "loading", None

        async def load():
            pool = None
            try:
                pool = await create_pool(self.config)
                identities = await worker_identities(pool)
                for client in self.mcp_clients:
                    await client.start()
                self.backend, self.identities = pool, identities
                self.backend_status = "ready"
                self._status_task = asyncio.create_task(self._watch_workers())
            except BaseException as exc:
                # A partially started extension and pool are both owned here.
                # Clients are single-use; retry must construct fresh instances.
                await asyncio.gather(
                    *(client.close() for client in self.mcp_clients),
                    *((pool.close(),) if pool is not None else ()),
                    return_exceptions=True,
                )
                self.backend, self.identities = None, {}
                self._restart_mcp = True
                if not isinstance(exc, Exception):
                    self.backend_status = "offline"
                    raise
                self.backend_status = "error"
                self.backend_error = f"{type(exc).__name__}: {exc}"

        self._loader = asyncio.create_task(load())
        return {"status": "loading"}

    async def _watch_workers(self) -> None:
        """Refresh worker status for the UI every couple of seconds."""
        while not self.closing:
            await self._refresh_workers()
            await asyncio.sleep(2.0)

    async def _refresh_workers(self) -> None:
        backends = getattr(self.backend, "backends", None)
        if not backends:
            return
        groups = getattr(self.backend, "groups", {})
        status = []
        for name, backend in backends.items():
            try:
                health = backend.health()
                if asyncio.iscoroutine(health):
                    health = await asyncio.wait_for(health, 2.0)
            except Exception as exc:  # noqa: BLE001 - a stuck worker shows as unreachable.
                health = {"error": f"{type(exc).__name__}: {exc}"}
            stats = health.get("last_runner_stats") or {}
            status.append(
                {
                    "name": name,
                    "group": groups.get(name),
                    "model": model_label(health.get("model") or ""),
                    "state": worker_state(health),
                    "active": health.get("active", 0),
                    "requests": stats.get("requests", 0),
                    "memory_bytes": stats.get("active_memory_bytes"),
                    "weight_bytes": stats.get("weight_bytes"),
                    "cache_bytes": stats.get("cache_bytes"),
                    "cache_entries": stats.get("cache_entries"),
                    "prompt_tokens": stats.get("prompt_tokens", 0),
                    "reused_tokens": stats.get("reused_tokens", 0),
                    "generated_tokens": stats.get("generated_tokens", 0),
                    "draft_enabled": stats.get("draft_enabled"),
                    "context": health.get("max_model_length"),
                    "error": health.get("error"),
                }
            )
        self.worker_status = status

    def summary(self) -> dict:
        """The cheap part of state() that chat views need on every refresh."""
        label = (
            "Deterministic harness demo"
            if self.demo
            else ", ".join(model["label"] for model in self.models())
        )
        return {
            "workspace": str(self.workspace),
            "mode": "demo" if self.demo else "live",
            "backend": {
                "status": self.backend_status,
                "label": label or "No workers configured",
                "error": self.backend_error,
            },
        }

    def models(self) -> list[dict]:
        """Models a chat can be driven by: one entry per worker group."""
        choices = {}
        for worker in self.config.get("workers", []):
            group = worker.get("group")
            if group and group not in choices:
                choices[group] = {
                    "id": group,
                    "label": model_label(worker.get("model", "")),
                    "context": worker.get("max_model_length"),
                }
        return list(choices.values())

    def live(self) -> dict:
        """In-flight generations with the text decoded so far, plus workers."""
        partial = getattr(self.backend, "partial", None)
        generations = []
        for runtime in list(self.runtimes.values()):
            for (run_id, agent_id), request_id in list(runtime._requests.items()):
                generations.append(
                    {
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "request_id": request_id,
                        "text": (partial(request_id) if callable(partial) else None)
                        or "",
                    }
                )
        # Changes whenever anything in the journal or a chat changes, so the UI
        # knows when to refetch the chat it's showing.
        version = self.store.connection.execute(
            "SELECT (SELECT COALESCE(MAX(id), 0) FROM events), "
            "(SELECT COUNT(*) || ':' || COALESCE(MAX(updated), 0) FROM studio_chats), "
            "(SELECT COUNT(*) || ':' || COALESCE(SUM(LENGTH(status)), 0) "
            "FROM studio_chat_turns)"
        ).fetchone()
        return {
            "version": ":".join(str(part) for part in version),
            "generations": generations,
            "workers": self.worker_status,
            "backend": self.summary()["backend"],
        }

    def _manifest(self, run_id: str) -> dict:
        row = self.store.connection.execute(
            "SELECT manifest FROM studio_manifests WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("run was not created by Studio")
        return json.loads(row[0])

    def _runtime(
        self, run_id: str, manifest: dict, config: RuntimeConfig
    ) -> AgentRuntime:
        review = manifest["policy"] == "review"
        tools = StudioWorkspaceTools(
            self.workspace,
            self.state_dir / "artifacts" / run_id,
            allow_write=review,
            allow_tests=review,
            state_dir=self.state_dir,
        )
        gate = ApprovalGate(
            self.store,
            mode=manifest["policy"],
            run_id=run_id,
            timeout_seconds=300,
            preview=lambda name, args: workspace_preview(tools, name, args),
            hooks=self.hooks,
        )
        mapping = tools.mapping()
        if not review:
            mapping = {
                name: tool for name, tool in mapping.items() if name not in MUTATING
            }
        descriptions = {
            k: v for k, v in tools.tool_descriptions().items() if k in mapping
        }
        schemas = {k: tools.schemas[k] for k in mapping}
        safe = set(tools.replay_safe_tools)
        mutations = set(MUTATING)
        if review:
            commands = CommandTools(self.workspace)
            mapping.update(commands.mapping())
            descriptions.update(commands.tool_descriptions())
            schemas.update(commands.schemas)
            mutations.add("run_command")
        if self.skills:
            mapping.update(self.skills.mapping())
            descriptions.update(self.skills.tool_descriptions())
            schemas.update(self.skills.schemas)
            safe.update(self.skills.mapping())
        for client in self.mcp_clients:
            for name, tool in client.mapping().items():
                if name in mapping:
                    raise ValueError("extension tool name collision")
                # Server annotations are not trusted to grant read-only access.
                if review:
                    mapping[name] = tool
                    mutations.add(name)
            if review:
                descriptions.update(client.tool_descriptions())
                schemas.update(client.schemas)
        wrapped = {
            name: gate.wrap(name, tool, mutating=name in mutations)
            for name, tool in mapping.items()
        }
        validator = None
        if self.config.get("acceptance"):
            # Completion checks execute tools too. Use the same review and
            # hook boundary instead of silently running operator test files.
            if "run_tests" in wrapped:
                tools.run_tests = wrapped["run_tests"]
            validator = CompletionProfile(self.config["acceptance"], tools, self.store)
        runtime_class = (
            ConversationRuntime if manifest.get("conversation") else AgentRuntime
        )
        runtime = runtime_class(
            self.backend,
            self.store,
            wrapped,
            config,
            tool_descriptions=descriptions,
            tool_schemas=schemas,
            replay_safe_tools=safe,
            mutating_tools=mutations,
            completion_validator=validator,
        )
        self.runtimes[run_id], self.gates[run_id] = runtime, gate
        return runtime

    def _configuration(self, payload: dict) -> RuntimeConfig:
        """Resolve operator budgets once; conversations retain this exact configuration."""
        runtime_options = dict(self.config.get("runtime", {}))
        for key, default, low, high in (
            ("max_agents", 4, 1, 16),
            ("max_total_tokens", 128_000, 4000, 1_000_000),
            ("max_steps_per_agent", 20, 2, 64),
        ):
            value = payload.get(key, runtime_options.get(key, default))
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            runtime_options[key] = value
        runtime_options["tool_timeout_seconds"] = max(
            450, runtime_options.get("tool_timeout_seconds", 450)
        )
        if self.config.get("acceptance", {}).get("tests"):
            runtime_options["completion_timeout_seconds"] = max(
                390, runtime_options.get("completion_timeout_seconds", 390)
            )
        return RuntimeConfig(**runtime_options)

    async def create_run(
        self,
        payload: dict,
        *,
        parent_id: str | None = None,
        _history: list[ChatMessage] | None = None,
        _configuration: RuntimeConfig | None = None,
        _conversation: bool = False,
        _media=(),
    ) -> dict:
        if self.closing or self.backend_status != "ready":
            raise ValueError("Start the model before creating a mission")
        if not isinstance(payload, dict) or payload.keys() - {
            "task",
            "policy",
            "max_agents",
            "max_total_tokens",
            "max_steps_per_agent",
            "skill",
        }:
            raise ValueError("unsupported mission fields")
        if sum(not job.done() for job in self.jobs.values()) >= 4:
            raise ValueError("Four missions are already active; wait or stop one")
        task = _text(payload.get("task"), "task")
        policy = payload.get("policy", "read_only")
        if policy not in {"read_only", "review"}:
            raise ValueError("policy must be read_only or review")
        config = _configuration or self._configuration(payload)
        if config.tool_timeout_seconds < 450:
            # New turns need room for review (300s), command execution (120s),
            # and hooks/preview. Historical run recovery keeps its own config.
            config = replace(config, tool_timeout_seconds=450)
        skill = payload.get("skill")
        skill_body = ""
        if skill:
            if not self.skills:
                raise ValueError("no skill directories were configured")
            skill_body = json.loads(await self.skills.load_skill({"name": skill}))[
                "content"
            ]
        inherited = ""
        if parent_id:
            parent = self.store.agent(parent_id, "root")
            if self.store.run(parent_id)["status"] not in SETTLED_RUNS:
                raise ValueError(
                    "Use steering for an active mission; follow-up requires a settled mission"
                )
            inherited = (
                "\n\nPrevious mission result (reference evidence; verify before relying on it):\n"
                + (parent.result or parent.error or "No result")[:6000]
            )
        augmented = task + inherited
        if skill_body:
            augmented += (
                "\n\nOperator-selected skill guidance (does not grant tools):\n"
                + skill_body
            )
        # Protocol task bounds apply after adding inherited guidance.
        if len(augmented) > 12_000:
            raise ValueError(
                "Brief plus selected skill and previous result exceeds 12000 characters"
            )
        run_id = uuid.uuid4().hex
        manifest = {
            "version": 1,
            "workspace": str(self.workspace),
            "policy": policy,
            "mode": "demo" if self.demo else "live",
            "backend_config_sha256": _digest(self.config),
            "worker_identities": self.identities,
            "skill": skill,
            "extensions_sha256": self._extensions_digest(),
            "parent_id": parent_id,
            "brief": task,
            "conversation": _conversation,
        }
        runtime = self._runtime(run_id, manifest, config)
        self.store.connection.execute(
            "INSERT INTO studio_manifests VALUES(?,?)", (run_id, json.dumps(manifest))
        )
        self.store.create_run(
            run_id,
            augmented,
            asdict(config),
            runtime._system_prompt(config, task=augmented),
            tools=sorted({*runtime.tools, "journal_read"}),
            completion_required=runtime.completion_validator is not None,
            media=_media,
        )
        if _history:
            # Install real chat roles before the new request. The conversation
            # runtime also anchors this request in its immutable system scope.
            with self.store.transaction():
                self.store.connection.execute(
                    "DELETE FROM messages WHERE run_id=? AND agent_id='root' AND role='user'",
                    (run_id,),
                )
                for message in _history:
                    self.store.add_message(
                        run_id,
                        "root",
                        message.role,
                        message.content,
                        media=message.media,
                    )
                self.store.add_message(run_id, "root", "user", augmented, media=_media)
        self.store.event(
            run_id,
            "root",
            "studio_mission",
            {"brief": task, "policy": policy, "parent_id": parent_id},
        )
        self._launch(run_id, runtime)
        return {"run_id": run_id}

    def _extensions_digest(self) -> str:
        catalogs = []
        if self.skills:
            catalogs.append(self.skills.catalog())
        for client in self.mcp_clients:
            catalogs.append(
                {"config": asdict(client.config), "schemas": client.schemas}
            )
        return _digest(json.loads(json.dumps(catalogs, default=str)))

    def _launch(self, run_id: str, runtime: AgentRuntime) -> None:
        async def drive():
            try:
                await runtime.resume(run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.event(run_id, None, "studio_error", {"error": error})
                # Preserve interruption/recovery semantics on controller failures.
                self.store.connection.execute(
                    "UPDATE runs SET status='interrupted',error=? WHERE run_id=? AND status NOT IN ('completed','failed','cancelled')",
                    (error, run_id),
                )

        self.jobs[run_id] = asyncio.create_task(drive())

    async def resume(self, run_id: str) -> dict:
        if self.closing or self.backend_status != "ready":
            raise ValueError("Start the model before recovery")
        if self.jobs.get(run_id) and not self.jobs[run_id].done():
            raise ValueError("mission is already active")
        if sum(not job.done() for job in self.jobs.values()) >= 4:
            raise ValueError("Four missions are already active; wait or stop one")
        run, manifest = self.store.run(run_id), self._manifest(run_id)
        if run["status"] not in {"running", "interrupted"}:
            raise ValueError(
                "Only interrupted missions can recover; create a follow-up for terminal missions"
            )
        if run["deadline"] <= time.time():
            raise ValueError("Original mission deadline expired; create a follow-up")
        if (
            manifest["workspace"] != str(self.workspace)
            or manifest["mode"] != ("demo" if self.demo else "live")
            or manifest["backend_config_sha256"] != _digest(self.config)
            or manifest["worker_identities"] != self.identities
            or manifest["extensions_sha256"] != self._extensions_digest()
        ):
            raise ValueError(
                "Recovery requires the original workspace, backend identities and extension catalog"
            )
        runtime = self._runtime(run_id, manifest, RuntimeConfig(**run["config"]))
        self._launch(run_id, runtime)
        return {"run_id": run_id}

    async def cancel(self, run_id: str) -> dict:
        run = self.store.run(run_id)
        if run["status"] in SETTLED_RUNS:
            raise ValueError("mission has already settled")
        if run_id in self.runtimes:
            await self.runtimes[run_id].cancel(run_id)
        else:
            self.store.request_cancel(run_id)
            for agent in self.store.agents(run_id):
                if agent.status not in TERMINAL:
                    self.store.update_agent(
                        run_id,
                        agent.agent_id,
                        status="cancelled",
                        error="operator cancelled inactive mission",
                    )
            self.store.finish_run(
                run_id, "cancelled", "operator cancelled inactive mission"
            )
        return {"run_id": run_id, "cancel_requested": True}

    async def message(self, run_id: str, payload: dict) -> dict:
        if payload.keys() - {"message", "agent_id"}:
            raise ValueError("unsupported steering fields")
        message = _text(payload.get("message"), "message", 8000)
        agent_id = payload.get("agent_id", "root")
        if (
            self.store.run(run_id)["status"] in SETTLED_RUNS
            or self.store.agent(run_id, agent_id).status in TERMINAL
        ):
            raise ValueError("Steering requires an active agent")
        with self.store.transaction():
            self.store.add_message(
                run_id, agent_id, "user", "Operator steering:\n" + message
            )
            self.store.event(run_id, agent_id, "operator_message", {"message": message})
        return {"delivered": True, "visibility": "next model turn"}

    async def decide(self, approval_id: str, payload: dict) -> dict:
        if set(payload) != {"approved"} or type(payload["approved"]) is not bool:
            raise ValueError("approved must be a boolean")
        for run_id, gate in self.gates.items():
            if any(p["id"] == approval_id for p in gate.pending(run_id)):
                gate.decide(approval_id, payload["approved"])
                return {"decided": True}
        raise ValueError("approval is no longer pending")

    def _usage(self, run: dict) -> dict:
        result = {
            k: run[k]
            for k in (
                "input_tokens",
                "output_tokens",
                "uncertain_tokens",
                "reserved_tokens",
            )
        }
        result["total_tokens"] = sum(result.values())
        return result

    async def state(self) -> dict:
        rows = self.store.connection.execute(
            "SELECT run_id FROM runs ORDER BY created DESC LIMIT 100"
        ).fetchall()
        runs = []
        for row in rows:
            run = self.store.run(row[0])
            manifest = self._manifest(row[0])
            active = bool(self.jobs.get(row[0]) and not self.jobs[row[0]].done())
            runs.append(
                {
                    "run_id": row[0],
                    "task": manifest["brief"],
                    "status": run["status"]
                    if active or run["status"] in SETTLED_RUNS
                    else "interrupted",
                    "created": run["created"],
                    "config": run["config"],
                    "usage": self._usage(run),
                    "active": active,
                }
            )
        label = (
            "Deterministic harness demo"
            if self.demo
            else ", ".join(
                w.get("model", w.get("url", "worker"))
                for w in self.config.get("workers", [])
            )
        )
        return {
            "workspace": str(self.workspace),
            "mode": "demo" if self.demo else "live",
            "backend": {
                "status": self.backend_status,
                "label": label or "No workers configured",
                "error": self.backend_error,
            },
            "runs": runs,
            "skills": self.skills.catalog()["skills"] if self.skills else [],
            "skill_warnings": self.skills.catalog()["warnings"] if self.skills else [],
            "capabilities": {
                "subagents": True,
                "review": True,
                "recovery": True,
                "steering": True,
                "mcp_servers": len(self.mcp_clients),
            },
        }

    async def detail(self, run_id: str, *, export: bool = False) -> dict:
        run, manifest = self.store.run(run_id), self._manifest(run_id)
        active = bool(self.jobs.get(run_id) and not self.jobs[run_id].done())
        if not active and run["status"] not in SETTLED_RUNS:
            run = dict(run, status="interrupted")
        events = self.store.events(run_id)
        gate = self.gates.get(run_id)
        approvals = gate.pending(run_id) if gate else []
        artifacts = []
        directory = self.state_dir / "artifacts" / run_id
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(directory.resolve())
                ):
                    artifacts.append(
                        {
                            "path": str(path.relative_to(directory)),
                            "size": path.stat().st_size,
                        }
                    )
                    if len(artifacts) >= 200:
                        break
        root = self.store.agent(run_id, "root")
        return {
            "run": {
                k: run[k]
                for k in ("run_id", "status", "created", "deadline", "config", "error")
            },
            "task": manifest["brief"],
            "agents": [asdict(a) for a in self.store.agents(run_id)],
            "usage": self._usage(run),
            "events": events if export else events[-500:],
            "events_truncated": not export and len(events) > 500,
            "approvals": approvals,
            "output": root.result,
            "error": run["error"] or root.error,
            "active": active,
            "artifacts": artifacts,
            "lineage": {"parent_id": manifest.get("parent_id")},
            "manifest": manifest
            if export
            else {"policy": manifest["policy"], "skill": manifest["skill"]},
            "completion_verified": root.completion_passed is True,
        }

    async def dispatch(self, method: str, path: str, payload: dict) -> dict:
        # Import lazily: DeveloperWorkspace composes Studio's confined tools.
        from .developer import DeveloperWorkspace

        if DeveloperWorkspace.handles(method, path):
            if not hasattr(self, "developer"):
                self.developer = DeveloperWorkspace(self)
            return await self.developer.dispatch(method, path, payload)
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "chats"]:
            return await self.conversations.dispatch(method, parts, payload)
        if (
            method == "GET"
            and len(parts) >= 5
            and parts[:2] == ["api", "runs"]
            and parts[3] == "artifacts"
        ):
            run_id = parts[2]
            self._manifest(run_id)
            name = _text(unquote("/".join(parts[4:])), "artifact path", 1024)
            tools = StudioWorkspaceTools(
                self.workspace,
                self.state_dir / "artifacts" / run_id,
                state_dir=self.state_dir,
            )
            artifact = tools._path(name, artifact=True)
            if artifact.is_symlink():
                raise ValueError("artifact symlinks are unavailable")
            content = (await asyncio.to_thread(tools._safe_read, artifact)).decode(
                "utf-8"
            )
            return {"path": name, "content": content}
        if method == "GET" and path == "/api/live":
            return self.live()
        if method == "GET" and path == "/api/models":
            return {"models": self.models(), "efforts": list(EFFORT_BUDGETS)}
        if method == "GET" and path == "/api/state":
            return await self.state()
        if method == "POST" and path == "/api/backend/start":
            return await self.start_backend()
        if method == "POST" and path == "/api/runs":
            return await self.create_run(payload)
        if len(parts) in {3, 4} and parts[:2] == ["api", "runs"]:
            run_id = parts[2]
            if method == "GET" and len(parts) == 3:
                return await self.detail(run_id)
            if method == "GET" and parts[-1] == "export":
                return await self.detail(run_id, export=True)
            if method == "POST" and len(parts) == 4:
                operation = parts[3]
                if operation == "cancel":
                    return await self.cancel(run_id)
                if operation == "resume":
                    return await self.resume(run_id)
                if operation == "message":
                    return await self.message(run_id, payload)
                if operation == "fork":
                    original = self._manifest(run_id)
                    return await self.create_run(
                        {"task": payload.get("task"), "policy": original["policy"]},
                        parent_id=run_id,
                    )
        if method == "POST" and len(parts) == 3 and parts[:2] == ["api", "approvals"]:
            return await self.decide(parts[2], payload)
        raise ValueError("unknown Studio API route")

    async def close(self):
        if self._closed:
            return
        self.closing = True
        await self.conversations.close()
        if hasattr(self, "_status_task"):
            self._status_task.cancel()
        if hasattr(self, "_loader") and not self._loader.done():
            self._loader.cancel()
            await asyncio.gather(self._loader, return_exceptions=True)
        for job in self.jobs.values():
            if not job.done():
                job.cancel()
        await asyncio.gather(*self.jobs.values(), return_exceptions=True)
        try:
            outcomes = await asyncio.gather(
                *(client.close() for client in self.mcp_clients),
                *((self.backend.close(),) if self.backend is not None else ()),
                return_exceptions=True,
            )
        finally:
            self.store.close()
            self._closed = True
        failures = [
            outcome for outcome in outcomes if isinstance(outcome, BaseException)
        ]
        if failures:
            raise RuntimeError("Studio resource cleanup failed") from failures[0]
