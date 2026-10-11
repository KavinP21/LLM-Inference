"""Deterministic integration exercise. This backend does not perform inference."""

from __future__ import annotations

import asyncio
import json

from ..agents.protocol import Generation, conservative_input_tokens
from ..agents.store import TERMINAL, AgentStore


class DemoBackend:
    def __init__(self, store: AgentStore, delay: float = 0.35):
        self.store, self.delay = store, delay

    async def generate(self, messages, max_tokens, request_id):
        await asyncio.sleep(self.delay)
        run_id, agent_id, _ = request_id.split(".")
        agent = self.store.agent(run_id, agent_id)
        receipts = self.store.recent_tool_results(run_id, agent_id)
        tools = {e["payload"]["name"] for e in receipts}
        if "list_files" not in tools:
            action = {"action": "list_files", "args": {"path": ".", "limit": 40}}
        elif agent_id != "root":
            listing = next(e for e in receipts if e["payload"]["name"] == "list_files")
            files = json.loads(listing["payload"]["result"]).get("files", [])
            if "README.md" in files and "read_file" not in tools:
                action = {
                    "action": "read_file",
                    "args": {"path": "README.md", "max_lines": 20},
                }
            else:
                action = {
                    "action": "finish",
                    "result": f"Deterministic inspection: observed {len(files)} listed files. "
                    + (
                        "Read the first 20 README lines."
                        if "read_file" in tools
                        else "No README inspection performed."
                    ),
                }
        else:
            children = [a for a in self.store.agents(run_id) if a.parent_id == "root"]
            maximum = self.store.run(run_id)["config"]["max_agents"]
            if len(children) < min(2, maximum - 1):
                action = {
                    "action": "spawn",
                    "task": "Inspect the workspace file listing and first README lines. Report observed evidence only.",
                    "role": "Repository inspector"
                    if not children
                    else "Documentation inspector",
                }
            elif any(a.status not in TERMINAL for a in children):
                action = {"action": "wait", "agents": [a.agent_id for a in children]}
            elif (
                "save_artifact" in (agent.tools or ()) and "save_artifact" not in tools
            ):
                action = {
                    "action": "save_artifact",
                    "args": {
                        "path": "demo-inspection.md",
                        "content": "# Deterministic harness inspection\n\nThis report exercises real file tools, delegation, and approval. No model inference occurred.\n\n"
                        + "\n".join(a.result or a.error or a.status for a in children),
                    },
                }
            else:
                action = {
                    "action": "finish",
                    "result": "Harness exercise completed without model inference. File tools executed and "
                    + str(len(children))
                    + " child agents settled.\n\n"
                    + "\n\n".join(a.result or a.error or a.status for a in children)
                    + (
                        "\n\nArtifact request: "
                        + next(
                            e["payload"]["result"]
                            for e in receipts
                            if e["payload"]["name"] == "save_artifact"
                        )
                        if "save_artifact" in tools
                        else ""
                    ),
                }
        text = json.dumps(action)
        return Generation(
            text,
            min(
                conservative_input_tokens(messages),
                max(1, sum(len(m.content) for m in messages) // 4),
            ),
            max(1, len(text) // 4),
            model="deterministic-demo",
        )

    async def cancel(self, request_id):
        return None

    async def close(self):
        return None
