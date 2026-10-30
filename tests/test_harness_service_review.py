"""Regression checks for policy boundaries and owned-resource recovery."""

import asyncio
import sqlite3
import time

import pytest
from instinct.harness.extensions import MCPServerConfig
from instinct.harness.service import StudioService


def service_for(tmp_path, **kwargs):
    (tmp_path / "README.md").write_text("# Fixture\n")
    service = StudioService(tmp_path, tmp_path / ".instinct", demo=True, **kwargs)
    service.backend.delay = 0.01
    return service


async def pending(service, run_id):
    for _ in range(300):
        detail = await service.detail(run_id)
        if detail["approvals"]:
            return detail["approvals"][0]
        await asyncio.sleep(0.01)
    raise AssertionError("approval never became available")


def test_acceptance_tests_require_review_before_process_execution(tmp_path):
    marker = tmp_path / "acceptance-executed.txt"
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_acceptance.py").write_text(
        "from pathlib import Path\n\n"
        "def test_acceptance():\n"
        f"    Path({str(marker)!r}).write_text('executed')\n"
    )

    async def scenario():
        service = service_for(
            tmp_path, config={"acceptance": {"tests": ["tests/test_acceptance.py"]}}
        )
        try:
            run_id = (
                await service.create_run(
                    {"task": "Inspect", "policy": "review", "max_agents": 1}
                )
            )["run_id"]
            artifact = await pending(service, run_id)
            assert artifact["tool"] == "save_artifact"
            await service.decide(artifact["id"], {"approved": True})
            acceptance = await pending(service, run_id)
            assert acceptance["tool"] == "run_tests"
            assert acceptance["args"] == {"paths": ["tests/test_acceptance.py"]}
            assert not marker.exists()
            assert (
                service.store.run(run_id)["config"]["completion_timeout_seconds"] >= 390
            )
            await service.decide(acceptance["id"], {"approved": True})
            await asyncio.wait_for(service.jobs[run_id], 5)
            detail = await service.detail(run_id)
            assert marker.read_text() == "executed"
            assert detail["run"]["status"] == "completed", detail["error"]
            assert detail["completion_verified"] is True
        finally:
            await service.close()

    asyncio.run(scenario())


def test_read_only_acceptance_cannot_run_tests(tmp_path):
    async def scenario():
        service = service_for(
            tmp_path, config={"acceptance": {"tests": ["tests/test_operator.py"]}}
        )
        try:
            with pytest.raises(ValueError, match="allow-tests"):
                await service.create_run({"task": "Inspect", "policy": "read_only"})
            assert (await service.state())["runs"] == []
        finally:
            await service.close()

    asyncio.run(scenario())


def test_cancel_during_review_settles_promptly_without_artifact(tmp_path):
    async def scenario():
        service = service_for(tmp_path)
        try:
            run_id = (
                await service.create_run(
                    {"task": "Inspect", "policy": "review", "max_agents": 1}
                )
            )["run_id"]
            row = await pending(service, run_id)
            started = time.monotonic()
            await service.cancel(run_id)
            await asyncio.wait_for(service.jobs[run_id], 1.5)
            assert time.monotonic() - started < 1.5
            detail = await service.detail(run_id)
            assert detail["run"]["status"] == "cancelled"
            assert not detail["approvals"] and not detail["artifacts"]
            with pytest.raises(ValueError, match="no longer pending"):
                await service.decide(row["id"], {"approved": True})
        finally:
            await service.close()

    asyncio.run(scenario())


def test_completed_with_failures_stays_settled_and_supports_followup(tmp_path):
    async def scenario():
        service = service_for(tmp_path)
        try:
            run_id = (await service.create_run({"task": "Inspect", "max_agents": 1}))[
                "run_id"
            ]
            await asyncio.wait_for(service.jobs[run_id], 5)
            service.store.finish_run(run_id, "completed_with_failures")
            assert (await service.state())["runs"][0][
                "status"
            ] == "completed_with_failures"
            assert (await service.detail(run_id))["run"][
                "status"
            ] == "completed_with_failures"
            with pytest.raises(ValueError, match="already settled"):
                await service.cancel(run_id)
            with pytest.raises(ValueError, match="active agent"):
                await service.message(run_id, {"message": "late"})
            followup = (
                await service.dispatch(
                    "POST", f"/api/runs/{run_id}/fork", {"task": "Explain"}
                )
            )["run_id"]
            await asyncio.wait_for(service.jobs[followup], 5)
            assert (await service.detail(followup))["lineage"]["parent_id"] == run_id
        finally:
            await service.close()

    asyncio.run(scenario())


def test_recovery_enforces_concurrent_mission_limit(tmp_path):
    async def scenario():
        service = service_for(tmp_path)
        service.backend.delay = 1
        try:
            interrupted = []
            for _ in range(5):
                run_id = (
                    await service.create_run({"task": "Inspect", "max_agents": 1})
                )["run_id"]
                await asyncio.sleep(0.01)
                service.jobs[run_id].cancel()
                await asyncio.gather(service.jobs[run_id], return_exceptions=True)
                assert service.store.run(run_id)["status"] == "interrupted"
                interrupted.append(run_id)
            for run_id in interrupted[:4]:
                await service.resume(run_id)
            with pytest.raises(ValueError, match="Four missions"):
                await service.resume(interrupted[4])
        finally:
            await service.close()

    asyncio.run(scenario())


def test_state_directory_cannot_contain_workspace_or_change_parent_permissions(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="cannot contain"):
        StudioService(workspace, tmp_path, demo=True)
    assert tmp_path.stat().st_mode & 0o777 == 0o755


def test_partial_backend_load_disposes_extensions_and_retry_uses_fresh_clients(
    tmp_path, monkeypatch
):
    import instinct.harness.service as module

    pools, instances = [], []
    fail_once = True

    class Pool:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class Client:
        def __init__(self, config):
            self.config, self.schemas = config, {}
            self.closed, self.started = False, False
            instances.append(self)

        async def start(self):
            nonlocal fail_once
            assert not self.closed
            self.started = True
            if self.config.name == "second" and fail_once:
                fail_once = False
                raise RuntimeError("extension startup failed")

        async def close(self):
            self.closed = True

    async def create_pool(config):
        pool = Pool()
        pools.append(pool)
        return pool

    async def identities(pool):
        return {"fixture": {"model": "unit-test"}}

    monkeypatch.setattr(module, "create_pool", create_pool)
    monkeypatch.setattr(module, "worker_identities", identities)
    monkeypatch.setattr(module, "MCPStdioClient", Client)

    async def scenario():
        original = tuple(
            Client(MCPServerConfig(name, ("unused",), tmp_path))
            for name in ("first", "second")
        )
        service = StudioService(
            tmp_path,
            tmp_path / ".instinct",
            config={"workers": [{"model": "fixture"}]},
            mcp_clients=original,
        )
        try:
            assert (await service.start_backend())["status"] == "loading"
            await service._loader
            assert service.backend_status == "error" and service.backend is None
            assert all(client.closed for client in original) and pools[0].closed
            await service.start_backend()
            await service._loader
            assert service.backend_status == "ready"
            assert service.identities == {"fixture": {"model": "unit-test"}}
            assert all(
                client.started and not client.closed for client in service.mcp_clients
            )
            assert all(client not in original for client in service.mcp_clients)
        finally:
            await service.close()
        assert all(client.closed for client in instances)
        assert all(pool.closed for pool in pools)

    asyncio.run(scenario())


def test_shutdown_during_extension_start_disposes_partial_pool(tmp_path, monkeypatch):
    import instinct.harness.service as module

    entered = asyncio.Event()

    class Pool:
        closed = False

        async def close(self):
            self.closed = True

    class Client:
        config = MCPServerConfig("fixture", ("unused",), tmp_path)
        closed = False
        schemas = {}

        async def start(self):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    pool, client = Pool(), Client()

    async def create_pool(config):
        return pool

    async def identities(pool):
        return {}

    monkeypatch.setattr(module, "create_pool", create_pool)
    monkeypatch.setattr(module, "worker_identities", identities)

    async def scenario():
        service = StudioService(
            tmp_path,
            tmp_path / ".instinct",
            config={"workers": [{"model": "fixture"}]},
            mcp_clients=(client,),
        )
        await service.start_backend()
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(service.close(), 1)
        assert pool.closed and client.closed
        assert service.backend is None and service.backend_status == "offline"

    asyncio.run(scenario())


def test_teardown_failure_still_closes_other_resources_and_journal(tmp_path):
    class Client:
        def __init__(self, name, fail=False):
            self.config = MCPServerConfig(name, ("unused",), tmp_path)
            self.fail, self.closed = fail, False

        async def close(self):
            self.closed = True
            if self.fail:
                raise RuntimeError("close failed")

    class Backend:
        closed = False

        async def close(self):
            self.closed = True

    async def scenario():
        clients = (Client("first", fail=True), Client("second"))
        backend = Backend()
        service = StudioService(
            tmp_path, tmp_path / ".instinct", backend=backend, mcp_clients=clients
        )
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await service.close()
        assert backend.closed and all(client.closed for client in clients)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            service.store.connection.execute("SELECT 1")
        await service.close()

    asyncio.run(scenario())
