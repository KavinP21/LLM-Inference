from __future__ import annotations

import asyncio
import hashlib
import json
import os

import pytest
from forge_llm.agents.tools import WorkspaceTools


def test_read_search_and_bounded_listing(tmp_path):
    (tmp_path / "a.py").write_text("first\nneedle\nlast\n")
    (tmp_path / "b.py").write_text("needle twice\n")
    (tmp_path / ".env").write_text("needle SECRET\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "large.engine").write_text("needle\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    read = json.loads(
        asyncio.run(tools.read_file({"path": "a.py", "start_line": 2, "max_lines": 1}))
    )
    assert read["content"] == "2: needle"
    assert read["truncated"]
    assert (
        read["sha256"] == hashlib.sha256((tmp_path / "a.py").read_bytes()).hexdigest()
    )
    matches = json.loads(asyncio.run(tools.search({"query": "needle"})))
    assert [m["path"] for m in matches["matches"]] == ["a.py", "b.py"]
    listing = json.loads(asyncio.run(tools.list_files({"limit": 1})))
    assert listing == {"files": ["a.py"], "truncated": True}


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/etc/passwd",
        ".git/config",
        ".env",
        ".aws/credentials",
        "bad\x00name",
    ],
)
def test_unavailable_paths(tmp_path, path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    with pytest.raises(ValueError):
        asyncio.run(tools.read_file({"path": path}))


def test_symlinks_cannot_escape_workspace_or_artifacts(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (workspace / "link").symlink_to(outside)
    tools = WorkspaceTools(workspace, workspace / "artifacts", allow_write=True)
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(tools.read_file({"path": "link"}))
    tools.artifacts.mkdir()
    (tools.artifacts / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(tools.save_artifact({"path": "link", "content": "changed"}))
    assert outside.read_text() == "private"


def test_internal_symlink_cannot_bypass_private_path_policy(tmp_path):
    (tmp_path / ".env").write_text("private")
    (tmp_path / "public_alias").symlink_to(tmp_path / ".env")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    with pytest.raises(ValueError, match="private"):
        asyncio.run(tools.read_file({"path": "public_alias"}))
    with pytest.raises(ValueError, match="private"):
        asyncio.run(
            tools.write_file(
                {"path": "public_alias", "content": "x", "expected_sha256": "missing"}
            )
        )


def test_writes_require_opt_in_and_current_file_hash(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    assert "write_file" not in tools.mapping()
    with pytest.raises(PermissionError):
        asyncio.run(
            tools.write_file(
                {"path": "a.txt", "content": "x", "expected_sha256": "missing"}
            )
        )
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    result = json.loads(
        asyncio.run(
            tools.write_file(
                {"path": "a.txt", "content": "original", "expected_sha256": "missing"}
            )
        )
    )
    (tmp_path / "a.txt").write_text("human edit")
    with pytest.raises(ValueError, match="version changed"):
        asyncio.run(
            tools.write_file(
                {
                    "path": "a.txt",
                    "content": "agent edit",
                    "expected_sha256": result["sha256"],
                }
            )
        )
    assert (tmp_path / "a.txt").read_text() == "human edit"
    asyncio.run(
        tools.write_file(
            {
                "path": "a.txt",
                "content": "reviewed",
                "expected_sha256": hashlib.sha256(b"human edit").hexdigest(),
            }
        )
    )
    assert (tmp_path / "a.txt").read_text() == "reviewed"


def test_artifacts_are_new_files_and_writes_are_bounded(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", max_file_bytes=10)
    asyncio.run(
        tools.save_artifact({"path": "nested/report.md", "content": "evidence"})
    )
    assert (tools.artifacts / "nested/report.md").read_text() == "evidence"
    with pytest.raises(ValueError, match="version changed"):
        asyncio.run(
            tools.save_artifact({"path": "nested/report.md", "content": "overwrite"})
        )
    with pytest.raises(ValueError, match="write limit"):
        asyncio.run(tools.save_artifact({"path": "huge.md", "content": "x" * 11}))


def test_tool_argument_schema_and_binary_reads(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts")
    (tmp_path / "binary").write_bytes(b"abc\x00def")
    with pytest.raises(ValueError, match="binary"):
        asyncio.run(tools.read_file({"path": "binary"}))
    with pytest.raises(ValueError, match="expected arguments"):
        asyncio.run(tools.read_file({"path": "binary", "command": "rm -rf"}))
    with pytest.raises(ValueError, match="integer"):
        asyncio.run(tools.list_files({"limit": True}))


def test_tests_are_explicit_and_do_not_accept_shell_arguments(tmp_path):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_tests=True)
    with pytest.raises(ValueError, match="test files"):
        asyncio.run(tools.run_tests({"paths": ["-c print(1)"]}))
    (tmp_path / "test_pass.py").write_text("def test_pass():\n    assert 2 + 2 == 4\n")
    result = json.loads(asyncio.run(tools.run_tests({"paths": ["test_pass.py"]})))
    assert result["exit_code"] == 0
    assert "1 passed" in result["output"]


def test_test_timeout_kills_owned_process(tmp_path):
    tools = WorkspaceTools(
        tmp_path, tmp_path / "artifacts", allow_tests=True, test_timeout=0.3
    )
    (tmp_path / "test_sleep.py").write_text(
        "import time\ndef test_sleep():\n    time.sleep(20)\n"
    )
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(tools.run_tests({"paths": ["test_sleep.py"]}))


def test_narrow_replace_preserves_other_content_and_requires_unique_match(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("# keep this\ndef answer():\n    return 41\n# keep this too\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    asyncio.run(
        tools.replace_text(
            {
                "path": "module.py",
                "old_text": "return 41",
                "new_text": "return 42",
                "expected_sha256": digest,
            }
        )
    )
    assert (
        path.read_text()
        == "# keep this\ndef answer():\n    return 42\n# keep this too\n"
    )
    with pytest.raises(ValueError, match="exactly once"):
        asyncio.run(
            tools.replace_text(
                {
                    "path": "module.py",
                    "old_text": "keep this",
                    "new_text": "x",
                    "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO regression")
def test_fifo_reads_and_writes_fail_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    with pytest.raises(ValueError, match="regular files"):
        asyncio.run(tools.read_file({"path": "pipe"}))
    with pytest.raises(ValueError, match="regular files"):
        asyncio.run(
            tools.write_file(
                {"path": "pipe", "content": "x", "expected_sha256": "missing"}
            )
        )


def test_implicit_versions_are_agent_scoped_and_reject_stale_edits(tmp_path):
    from forge_llm.agents.context import tool_actor

    async def run():
        path = tmp_path / "module.py"
        path.write_text("old")
        tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
        one = tool_actor.set(("run", "one"))
        read = json.loads(await tools.read_file({"path": "module.py"}))
        tool_actor.reset(one)
        two = tool_actor.set(("run", "two"))
        with pytest.raises(ValueError, match="no version"):
            await tools.write_file({"path": "module.py", "content": "new"})
        await tools.read_file({"path": "module.py"})
        await tools.write_file({"path": "module.py", "content": "new"})
        with pytest.raises(ValueError, match="unavailable"):
            await tools.write_file(
                {
                    "path": "module.py",
                    "content": "wrong",
                    "read_receipt": read["read_receipt"],
                }
            )
        tool_actor.reset(two)
        one = tool_actor.set(("run", "one"))
        with pytest.raises(ValueError, match="version changed"):
            await tools.write_file({"path": "module.py", "content": "stale"})
        await tools.read_file({"path": "module.py"})
        await tools.replace_text(
            {"path": "module.py", "old_text": "new", "new_text": "reviewed"}
        )
        tool_actor.reset(one)
        assert path.read_text() == "reviewed"

    asyncio.run(run())


def test_invalid_python_edit_fails_before_mutation_with_useful_diagnostic(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def mean(values):\n    return sum(values) / (len(values) + 1)\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def run():
        await tools.read_file({"path": "module.py"})
        with pytest.raises(ValueError, match="Python syntax error") as error:
            await tools.replace_text(
                {
                    "path": "module.py",
                    "old_text": "return sum(values) / (len(values) + 1)",
                    "new_text": "return sum(values) / len(values) if values else raise ValueError('empty')",
                }
            )
        assert error.value.work_started is False
        assert "raise" in str(error.value)

    asyncio.run(run())
    assert "len(values) + 1" in path.read_text()


def test_multiline_replace_cannot_insert_top_level_return_and_can_correct_indentation(
    tmp_path,
):
    path = tmp_path / "stats.py"
    original = "def mean(values):\n    return sum(values) / (len(values) + 1)\n"
    path.write_text(original)
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def run():
        await tools.read_file({"path": "stats.py"})
        with pytest.raises(ValueError, match="return.*outside function") as error:
            await tools.replace_text(
                {
                    "path": "stats.py",
                    "old_text": "return sum(values) / (len(values) + 1)",
                    "new_text": "if not values: raise ValueError('empty')\nreturn sum(values) / len(values)",
                }
            )
        assert error.value.work_started is False
        assert "line 3" in str(error.value)
        assert "source: 'return sum(values) / len(values)'" in str(error.value)
        assert path.read_text() == original
        # The rejected write does not invalidate the last read; exact whitespace
        # lets the corrected edit use the same automatic compare-and-swap.
        await tools.replace_text(
            {
                "path": "stats.py",
                "old_text": "    return sum(values) / (len(values) + 1)",
                "new_text": "    if not values: raise ValueError('empty')\n    return sum(values) / len(values)",
            }
        )

    asyncio.run(run())
    assert (
        path.read_text()
        == "def mean(values):\n    if not values: raise ValueError('empty')\n    return sum(values) / len(values)\n"
    )


@pytest.mark.parametrize(
    "statement",
    ["return 7", "yield 7", "await operation()", "break", "continue", "nonlocal value"],
)
def test_python_preflight_rejects_context_errors_before_write(tmp_path, statement):
    path = tmp_path / "module.py"
    original = "value = 1\n"
    path.write_text(original)
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def run():
        await tools.read_file({"path": "module.py"})
        with pytest.raises(ValueError, match="Python syntax error") as error:
            await tools.write_file({"path": "module.py", "content": statement + "\n"})
        assert error.value.work_started is False
        assert path.read_text() == original

    asyncio.run(run())


def test_python_preflight_compiles_without_executing_module(tmp_path):
    marker = tmp_path / "executed.txt"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('must not execute')\n"
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    asyncio.run(tools.write_file({"path": "module.py", "content": source}))
    assert (tmp_path / "module.py").read_text() == source
    assert not marker.exists()


def test_native_edit_catalog_uses_implicit_version_but_explicit_api_remains_supported(
    tmp_path,
):
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    descriptions = tools.tool_descriptions()
    for name in ("write_file", "replace_text"):
        properties = descriptions[name]["parameters"]["properties"]
        assert not {"read_receipt", "expected_sha256"} & properties.keys()
        assert {"read_receipt", "expected_sha256"} <= tools.schemas[name][
            "properties"
        ].keys()
    replacement = descriptions["replace_text"]
    assert "EVERY replacement line" in replacement["description"]
    assert (
        "leading spaces"
        in replacement["parameters"]["properties"]["old_text"]["description"]
    )
    path = tmp_path / "config.txt"
    path.write_text("old\n")

    async def run():
        read = json.loads(await tools.read_file({"path": "config.txt"}))
        await tools.replace_text(
            {
                "path": "config.txt",
                "old_text": "old",
                "new_text": "new",
                "read_receipt": read["read_receipt"],
                "expected_sha256": read["sha256"],
            }
        )

    asyncio.run(run())
    assert path.read_text() == "new\n"


def test_redundant_matching_versions_are_valid_and_conflicts_are_rejected(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("original")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def run():
        read = json.loads(await tools.read_file({"path": "value.txt"}))
        with pytest.raises(ValueError, match="different file versions"):
            await tools.write_file(
                {
                    "path": "value.txt",
                    "content": "bad",
                    "read_receipt": read["read_receipt"],
                    "expected_sha256": "0" * 64,
                }
            )
        await tools.write_file(
            {
                "path": "value.txt",
                "content": "changed",
                "read_receipt": read["read_receipt"],
                "expected_sha256": read["sha256"],
            }
        )

    asyncio.run(run())
    assert path.read_text() == "changed"


def test_operator_protected_tests_cannot_be_written_or_replaced_after_read(tmp_path):
    (tmp_path / "tests").mkdir()
    path = tmp_path / "tests" / "test_acceptance.py"
    original = "def test_acceptance():\n    assert 2 + 2 == 4\n"
    path.write_text(original)
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    selected = ["tests/test_acceptance.py"]
    expected = {selected[0]: hashlib.sha256(original.encode()).hexdigest()}
    assert tools.protect_files(selected) == expected

    async def run():
        read = json.loads(await tools.read_file({"path": selected[0]}))
        with pytest.raises(ValueError, match="operator-protected") as error:
            await tools.write_file(
                {
                    "path": selected[0],
                    "content": "def test_acceptance():\n    assert True\n",
                    "expected_sha256": read["sha256"],
                }
            )
        assert error.value.work_started is False
        with pytest.raises(ValueError, match="operator-protected"):
            await tools.replace_text(
                {
                    "path": selected[0],
                    "old_text": "2 + 2 == 4",
                    "new_text": "True",
                    "read_receipt": read["read_receipt"],
                }
            )
        await tools.write_file({"path": "solution.py", "content": "answer = 4\n"})

    asyncio.run(run())
    assert path.read_text() == original
    assert tools.protected_file_hashes(selected) == expected
    assert (
        "protect_files" not in tools.mapping()
        and "protected_file_hashes" not in tools.mapping()
    )


def test_protected_canonical_target_cannot_be_edited_through_an_alias(tmp_path):
    target = tmp_path / "test_acceptance.py"
    target.write_text("assert True\n")
    (tmp_path / "alias.py").symlink_to(target)
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    tools.protect_files(["test_acceptance.py"])
    with pytest.raises(ValueError, match="operator-protected"):
        asyncio.run(
            tools.write_file(
                {
                    "path": "alias.py",
                    "content": "assert False\n",
                    "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        )
    assert target.read_text() == "assert True\n"


def test_protected_selected_name_remains_guarded_if_redirected_to_other_target(
    tmp_path,
):
    selected = tmp_path / "test_acceptance.py"
    selected.write_text("assert True\n")
    other = tmp_path / "other.py"
    other.write_text("value = 1\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    baseline = tools.protect_files(["test_acceptance.py"])
    selected.unlink()
    selected.symlink_to(other)
    assert tools.protected_file_hashes(["test_acceptance.py"]) != baseline
    with pytest.raises(ValueError, match="operator-protected"):
        asyncio.run(
            tools.write_file(
                {
                    "path": "test_acceptance.py",
                    "content": "value = 2\n",
                    "expected_sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
                }
            )
        )
    assert other.read_text() == "value = 1\n"


def test_protection_registration_is_bounded_confined_and_atomic(tmp_path):
    valid = tmp_path / "valid.txt"
    valid.write_text("original")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    assert tools.protect_files([]) == tools.protected_file_hashes([]) == {}
    for paths in (["../outside"], [str(valid)], ["valid.txt", "valid.txt"]):
        with pytest.raises(ValueError):
            tools.protect_files(paths)
    with pytest.raises(FileNotFoundError):
        tools.protect_files(["valid.txt", "missing.txt"])
    asyncio.run(
        tools.write_file(
            {
                "path": "valid.txt",
                "content": "changed",
                "expected_sha256": hashlib.sha256(b"original").hexdigest(),
            }
        )
    )
    assert valid.read_text() == "changed"
    too_small = WorkspaceTools(tmp_path, tmp_path / "artifacts", max_file_bytes=2)
    with pytest.raises(ValueError, match="read limit"):
        too_small.protect_files(["valid.txt"])


def test_forget_terminal_scheduler_history_preserves_counters():
    from forge_llm.runtime import IterationScheduler, KVBlockPool

    pool = KVBlockPool(4096, 8)
    scheduler = IterationScheduler(2, 32, 64, pool)
    request = scheduler.submit([1, 2], 3, ())
    with pytest.raises(RuntimeError, match="active"):
        scheduler.forget(request)
    scheduler.cancel(request)
    before = scheduler.stats()
    scheduler.forget(request)
    assert scheduler.stats() == before
    assert pool.stats()["allocated_blocks"] == 0
    with pytest.raises(KeyError):
        scheduler.request(request)
