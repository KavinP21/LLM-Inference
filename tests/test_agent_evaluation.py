"""The task grader must remain independent of agent-edited visible tests."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def evaluator(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "forge_evaluate_agents", directory / "evaluate_agents.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modified_visible_tests_cannot_manufacture_code_acceptance(
    tmp_path, monkeypatch
):
    module = evaluator(monkeypatch)
    (tmp_path / "tests").mkdir()
    (tmp_path / "stats.py").write_text("def mean(values):\n    return 0\n")
    (tmp_path / "tests/test_stats.py").write_text("def test_fake():\n    assert True\n")
    graded = module.grade("coding", tmp_path, SimpleNamespace(status="completed"))
    assert not graded["passed"]
    assert graded["independent_test_exit"] != 0
    (tmp_path / "stats.py").write_text(
        "def mean(values):\n    if not values:\n        raise ValueError('empty')\n    return sum(values) / len(values)\n"
    )
    assert module.grade("coding", tmp_path, SimpleNamespace(status="completed"))[
        "passed"
    ]


def test_research_grader_checks_types_and_unresolved_run(tmp_path, monkeypatch):
    module = evaluator(monkeypatch)
    import json

    answer = {
        "eligible_throughput": 64,
        "baseline_throughput": 32,
        "speedup": 2,
        "latency_increase_ms": 10,
        "memory_increase_mib": 256,
        "excluded_throughput": 128,
        "release_ready": False,
        "energy_measured": False,
    }
    result = SimpleNamespace(status="completed", output=json.dumps(answer))
    assert module.grade("research", tmp_path, result)["passed"]
    answer["release_ready"] = 0
    result.output = json.dumps(answer)
    assert not module.grade("research", tmp_path, result)["passed"]
    answer["release_ready"] = False
    result.output = json.dumps(answer)
    result.status = "failed"
    assert not module.grade("research", tmp_path, result)["passed"]


def test_completion_steering_does_not_reveal_additional_code_cases(
    tmp_path, monkeypatch
):
    from forge_llm.agents.store import AgentStore

    module = evaluator(monkeypatch)
    # A fabricated lookup satisfies the visible examples but fails general mean
    # behavior. Additional cases remain a separate final qualification gate.
    (tmp_path / "stats.py").write_text(
        "def mean(values):\n    if not values:\n        raise ValueError('empty')\n"
        "    return {(2, 4): 3, (1,): 1, (-2, 2): 0}[tuple(values)]\n"
    )
    with AgentStore() as store:
        check = module.completion_checker("coding", "single", tmp_path, store)
        assert asyncio.run(check("case", "root", "done")).passed
    final = module.grade("coding", tmp_path, SimpleNamespace(status="completed"))
    assert not final["passed"]
    assert "test_unseen_inputs" in final["independent_test_output"]
