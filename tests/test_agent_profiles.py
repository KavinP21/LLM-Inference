import asyncio
import json
from types import SimpleNamespace

import pytest
from forge_llm.agents import AgentRuntime, AgentStore, RuntimeConfig, WorkspaceTools
from forge_llm.agents.profiles import CompletionProfile
from forge_llm.agents.protocol import Generation
from forge_llm.agents.tools import ToolValidationError


class Answer:
    async def generate(self, messages, max_tokens, request_id):
        return Generation(
            json.dumps({"action": "finish", "result": '{"ready":true}'}), 20, 20
        )


def test_explicit_format_acceptance_is_recorded(tmp_path):
    store = AgentStore()
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    profile = CompletionProfile(
        {
            "json_schema": {
                "type": "object",
                "properties": {"ready": {"type": "boolean"}},
                "required": ["ready"],
                "additionalProperties": False,
            }
        },
        tools,
        store,
    )
    result = asyncio.run(
        AgentRuntime(Answer(), store, completion_validator=profile).run(
            "Return the requested JSON."
        )
    )
    assert result.status == "completed"
    assert result.completion_verified


def test_empty_criteria_and_unauthorized_test_execution_fail_early(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    with pytest.raises(ValueError, match="enforceable"):
        CompletionProfile({}, tools, AgentStore())
    with pytest.raises(ValueError, match="allow-tests"):
        CompletionProfile({"tests": ["tests/test_code.py"]}, tools, AgentStore())


def test_failed_acceptance_does_not_mark_the_goal_verified(tmp_path):
    store = AgentStore()
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    profile = CompletionProfile({"min_children": 2}, tools, store)
    result = asyncio.run(
        AgentRuntime(
            Answer(),
            store,
            config=RuntimeConfig(max_completion_rejections=2),
            completion_validator=profile,
        ).run("Obtain two completed reviews.")
    )
    assert result.status == "failed"
    assert not result.completion_verified


def test_real_configured_test_check_uses_actual_code(tmp_path):
    async def run():
        (tmp_path / "test_value.py").write_text(
            "from value import VALUE\ndef test_value():\n    assert VALUE == 4\n"
        )
        (tmp_path / "value.py").write_text("VALUE = 5\n")
        store = AgentStore()
        store.create_run("case", "Task", vars(RuntimeConfig()), "system")
        tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_tests=True)
        profile = CompletionProfile({"tests": ["test_value.py"]}, tools, store)
        checked = await profile("case", "root", "done")
        assert not checked.passed
        assert "AssertionError" in checked.feedback
        (tmp_path / "value.py").write_text("VALUE = 4\n")
        assert (await profile("case", "root", "done")).passed

    asyncio.run(run())


@pytest.mark.parametrize(
    "answer", ['{"ready":true,"ready":false}', '{"ready":NaN}', '{"ready":1e999}']
)
def test_invalid_json_cannot_qualify_completion(tmp_path, answer):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    store = AgentStore()
    store.create_run("case", "Task", vars(RuntimeConfig()), "system")
    profile = CompletionProfile({"json_schema": {"type": "object"}}, tools, store)
    assert not asyncio.run(profile("case", "root", answer)).passed


def test_criterion_files_are_protected_and_external_changes_detected(tmp_path):
    async def run():
        criterion = tmp_path / "test_value.py"
        criterion.write_text("def test_value():\n    assert False\n")
        store = AgentStore()
        store.create_run("case", "Task", vars(RuntimeConfig()), "system")
        tools = WorkspaceTools(
            tmp_path, tmp_path / "artifacts", allow_write=True, allow_tests=True
        )
        profile = CompletionProfile({"tests": ["test_value.py"]}, tools, store)
        await tools.read_file({"path": "test_value.py"})
        with pytest.raises(ToolValidationError, match="protected"):
            await tools.write_file(
                {
                    "path": "test_value.py",
                    "content": "def test_fake():\n    assert True\n",
                }
            )
        criterion.write_text("def test_fake():\n    assert True\n")
        checked = await profile("case", "root", "done")
        assert not checked.passed
        assert "acceptance tests changed" in checked.feedback

    asyncio.run(run())


def test_cli_resume_rejects_changed_criterion_before_loading_models(
    tmp_path, monkeypatch, capsys
):
    from forge_llm.agents import cli
    from forge_llm.agents.backends import WorkerPool

    class Backend(Answer):
        def health(self):
            return {
                "model_data_sha256": "a" * 64,
                "model_config_sha256": "b" * 64,
                "tokenizer_signature": "c" * 64,
            }

        async def close(self):
            pass

    loaded = []

    async def create_pool(config):
        loaded.append(True)
        return WorkerPool({"fixture": Backend()})

    monkeypatch.setattr(cli, "create_pool", create_pool)
    (tmp_path / "test_acceptance.py").write_text(
        "def test_acceptance():\n    assert True\n"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workers": [{"model": "fixture.engine", "tokenizer": "fixture"}],
                "acceptance": {"tests": ["test_acceptance.py"]},
            }
        )
    )
    args = SimpleNamespace(
        command="run",
        state_dir=str(tmp_path / ".forge"),
        workspace=str(tmp_path),
        config=str(config),
        run_id="criterion-run",
        allow_write=True,
        allow_tests=True,
        quiet=True,
        task="Return the verified result",
        task_file=None,
    )
    assert asyncio.run(cli.execute(args)) == 0
    (tmp_path / "test_acceptance.py").write_text(
        "def test_acceptance():\n    assert False\n"
    )
    args.command = "resume"
    with pytest.raises(ValueError, match="original workspace"):
        asyncio.run(cli.execute(args))
    assert loaded == [True]
    capsys.readouterr()
