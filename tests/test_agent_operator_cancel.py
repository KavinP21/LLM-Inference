import asyncio
import json

import pytest
from instinct.agents.protocol import Generation
from instinct.agents.runtime import AgentRuntime, RuntimeConfig


def test_operator_stops_child_without_stopping_parent_or_sibling():
    class Backend:
        def __init__(self):
            self.waiting = asyncio.Event()
            self.cancelled = []

        async def generate(self, messages, max_tokens, request_id):
            _, agent, step = request_id.split(".")
            if agent == "root":
                actions = {
                    "1": {"action": "spawn", "task": "Long review"},
                    "2": {"action": "spawn", "task": "Short review"},
                    "3": {"action": "wait", "agents": ["agent-0001", "agent-0002"]},
                }
                action = actions.get(
                    step,
                    {
                        "action": "finish",
                        "result": "One reviewer stopped; other work completed.",
                    },
                )
            elif agent == "agent-0001":
                self.waiting.set()
                await asyncio.sleep(30)
                action = {"action": "finish", "result": "must not finish"}
            else:
                action = {"action": "finish", "result": "Short review completed."}
            return Generation(json.dumps(action), 1, 1)

        async def cancel(self, request_id):
            self.cancelled.append(request_id)

    async def check():
        backend = Backend()
        runtime = AgentRuntime(
            backend,
            config=RuntimeConfig(
                max_agents=3, max_steps_per_agent=8, max_total_tokens=128000
            ),
        )
        run = asyncio.create_task(runtime.run("Review", run_id="operator-cancel"))
        await asyncio.wait_for(backend.waiting.wait(), 3)
        await runtime.cancel_agent("operator-cancel", "agent-0001")
        result = await asyncio.wait_for(run, 5)
        agents = {agent.agent_id: agent for agent in result.agents}
        assert agents["agent-0001"].status == "cancelled"
        assert agents["agent-0002"].status == "completed"
        assert agents["root"].status == "completed"
        assert result.usage.reserved_tokens == 0
        assert any(
            event["kind"] == "agent_cancel_requested"
            for event in runtime.store.events("operator-cancel")
        )
        with pytest.raises(ValueError, match="active coordinator"):
            await runtime.cancel_agent("operator-cancel", "agent-0001")
        runtime.store.close()

    asyncio.run(check())


def test_stopping_parent_agent_cancels_its_executing_subtree():
    async def check():
        started, settled = asyncio.Event(), asyncio.Event()

        async def slow_tool(args):
            started.set()
            try:
                await asyncio.sleep(30)
                pytest.fail("cancelled tool continued executing")
            finally:
                settled.set()

        class Backend:
            async def generate(self, messages, max_tokens, request_id):
                _, agent, step = request_id.split(".")
                if agent == "root":
                    actions = {
                        "1": {
                            "action": "spawn",
                            "task": "Delegate a nested review",
                            "tools": ["slow_tool"],
                        },
                        "2": {"action": "wait", "agents": ["agent-0001"]},
                    }
                    action = actions.get(
                        step, {"action": "finish", "result": "The branch was stopped."}
                    )
                elif agent == "agent-0001":
                    action = (
                        {
                            "action": "spawn",
                            "task": "Execute the test tool",
                            "tools": ["slow_tool"],
                        }
                        if step == "1"
                        else {"action": "wait", "agents": ["agent-0002"]}
                    )
                else:
                    action = {"action": "tool", "name": "slow_tool", "args": {}}
                return Generation(json.dumps(action), 1, 1)

        runtime = AgentRuntime(
            Backend(),
            tools={"slow_tool": slow_tool},
            config=RuntimeConfig(
                max_agents=3, max_steps_per_agent=8, max_total_tokens=128000
            ),
        )
        run = asyncio.create_task(runtime.run("Nested review", run_id="nested-cancel"))
        await asyncio.wait_for(started.wait(), 3)
        await runtime.cancel_agent("nested-cancel", "agent-0001")
        result = await asyncio.wait_for(run, 5)
        assert settled.is_set()
        agents = {agent.agent_id: agent for agent in result.agents}
        assert agents["agent-0001"].status == agents["agent-0002"].status == "cancelled"
        assert agents["root"].status == "completed"
        assert result.usage.reserved_tokens == 0
        runtime.store.close()

    asyncio.run(check())
