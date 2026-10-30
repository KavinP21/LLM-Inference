"""Worker groups, prompt-cache affinity, lazy replicas and streamed progress."""

from __future__ import annotations

import asyncio
import time

from instinct.agents.backends import LazyBackend, WorkerConfig, WorkerPool
from instinct.agents.protocol import ChatMessage, Generation
from instinct.agents.runtime import AgentRuntime, RuntimeConfig

CHAT = [ChatMessage("user", "hello")]


class Replica:
    def __init__(self, name, delay=0.0):
        self.name, self.delay, self.calls = name, delay, []
        self.closed = False

    async def generate(self, messages, max_tokens, request_id, **kwargs):
        self.calls.append(request_id)
        await asyncio.sleep(self.delay)
        return Generation(self.name, 5, 1)

    async def cancel(self, request_id):
        pass

    async def health(self):
        return {"ready": True, "worker_id": self.name}

    async def close(self):
        self.closed = True


def test_groups_route_requests_to_their_replicas():
    driver, helper_a, helper_b = Replica("driver"), Replica("a"), Replica("b")
    pool = WorkerPool(
        {"driver": driver, "a": helper_a, "b": helper_b},
        groups={"driver": "driver", "a": "subagent", "b": "subagent"},
    )

    async def run():
        root = await pool.generate(CHAT, 1, "r1", group="driver")
        child = await pool.generate(CHAT, 1, "c1", group="subagent")
        anywhere = await pool.generate(CHAT, 1, "x1", group="missing")
        return root.text, child.text, anywhere.text

    root, child, anywhere = asyncio.run(run())
    assert root == "driver" and child in {"a", "b"} and anywhere
    assert driver.calls == ["r1"] + (["x1"] if anywhere == "driver" else [])


def test_affinity_keeps_an_agent_on_the_replica_with_its_cache():
    replicas = {name: Replica(name) for name in ("a", "b", "c")}
    pool = WorkerPool(replicas)

    async def run():
        first = await pool.generate(CHAT, 1, "s1", affinity="run/agent-1")
        later = [
            (await pool.generate(CHAT, 1, f"s{n}", affinity="run/agent-1")).text
            for n in range(2, 6)
        ]
        return first.text, later

    first, later = asyncio.run(run())
    assert later == [first] * 4


def test_busy_affinity_replica_does_not_block_other_work():
    slow, fast = Replica("slow", delay=0.2), Replica("fast")
    pool = WorkerPool({"slow": slow, "fast": fast})

    async def run():
        await pool.generate(CHAT, 1, "warm", worker="slow", affinity="agent")
        busy = asyncio.create_task(pool.generate(CHAT, 1, "busy", worker="slow"))
        await asyncio.sleep(0.01)
        started = time.monotonic()
        result = await pool.generate(CHAT, 1, "next", affinity="agent")
        elapsed = time.monotonic() - started
        await busy
        return result.text, elapsed

    text, elapsed = asyncio.run(run())
    assert text == "fast" and elapsed < 0.1


def lazy_config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3_5"}')
    (model / "tokenizer.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"\0" * 64)
    return WorkerConfig(
        str(model), str(model), backend="mlx", runner="chat", native_tool_calls=True
    )


def test_lazy_replica_starts_on_demand_and_stops_when_idle(tmp_path):
    started = []

    async def start():
        replica = Replica("lazy")
        started.append(replica)
        return replica

    async def run():
        lazy = LazyBackend(lazy_config(tmp_path), start, idle_seconds=0.05)
        identity = await lazy.health()
        assert identity["started"] is False and len(identity["model_data_sha256"]) == 64
        assert lazy.native_call_dialect == "qwen3_coder_xml"
        assert not started
        await lazy.generate(CHAT, 1, "one")
        assert lazy.started and len(started) == 1
        assert (await lazy.health())["started"] is True
        await asyncio.sleep(0.1)
        assert not lazy.started and started[0].closed
        await lazy.generate(CHAT, 1, "two")
        assert len(started) == 2
        await lazy.close()

    asyncio.run(run())


def test_pool_prefers_running_replicas_over_cold_lazy_ones(tmp_path):
    starts = []

    async def start():
        starts.append(1)
        return Replica("cold")

    async def run():
        lazy = LazyBackend(lazy_config(tmp_path), start, idle_seconds=None)
        pool = WorkerPool({"warm": Replica("warm"), "cold": lazy})
        result = await pool.generate(CHAT, 1, "r")
        await pool.close()
        return result.text

    assert asyncio.run(run()) == "warm" and not starts


def test_runtime_sends_root_and_child_agents_to_their_groups():
    seen = []

    class Recorder:
        routes = True
        supports_native_tools = True

        async def generate_action(
            self, messages, max_tokens, request_id, tool_specs, **kw
        ):
            seen.append((request_id.split(".")[1], kw.get("group"), kw.get("affinity")))
            if ".root.1" in request_id:
                return Generation(
                    '<tool_call>{"name": "spawn", "arguments": {"task": "Look"}}</tool_call>'
                )
            if ".root.2" in request_id:
                return Generation(
                    '<tool_call>{"name": "wait", "arguments": {"agents": ["agent-0001"]}}</tool_call>'
                )
            return Generation("done")

        async def cancel(self, request_id):
            pass

    config = RuntimeConfig(driver_group="driver", subagent_group="subagent")
    result = asyncio.run(
        AgentRuntime(Recorder(), config=config).run("Task", run_id="r")
    )
    assert result.status == "completed"
    assert ("root", "driver", "r/root") in seen
    assert ("agent-0001", "subagent", "r/agent-0001") in seen
