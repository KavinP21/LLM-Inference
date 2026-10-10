#!/usr/bin/env python3
"""Actual-model coding/research evaluation with independent executable grading.

Scripted unit tests establish runtime mechanics; this evaluation measures model
decisions through the same worker and tool interfaces used by the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from forge_llm.agents import AgentRuntime, AgentStore, RuntimeConfig, WorkspaceTools
from forge_llm.agents.cli import create_pool, load_config
from forge_llm.agents.runtime import CompletionCheck
from forge_llm.benchmark import model_provenance, source_provenance
from gpu_guard import exclusive_gpu_workflow

CASES = {
    "coding": {
        "files": {
            "stats.py": "def mean(values):\n    return sum(values) / (len(values) + 1)\n",
            "tests/test_stats.py": "import pytest\nfrom stats import mean\n\ndef test_mean():\n    assert mean([2, 4]) == 3\n    assert mean([1]) == 1\n    assert mean([-2, 2]) == 0\n\ndef test_empty():\n    with pytest.raises(ValueError):\n        mean([])\n",
        },
        "task": "Fix stats.py so mean(values) computes the arithmetic mean and raises ValueError for an empty list. Read the source and tests, make a minimal edit, and run tests/test_stats.py. Cite the actual test result in your final answer.",
        "delegation": "First spawn two independent review agents: one inspects stats.py and proposes the minimal correction; the other inspects tests/test_stats.py and checks edge cases. Ask the children to report findings without editing files. Wait for both, then make and test the edit yourself.",
    },
    "research": {
        "files": {
            "report_a.md": "# Registered experiment A\n12 concurrent requests on Host A. Same model and prompt/output lengths. Baseline throughput: 32 tokens/s; candidate throughput: 64 tokens/s. Baseline p95 latency: 40 ms; candidate p95 latency: 50 ms. Baseline memory: 256 MiB; candidate memory: 512 MiB. All 12 outputs matched the reference.\n",
            "report_b.md": "# Diagnostic experiment B\nObserved throughput: 128 tokens/s. An unrelated GPU job overlapped the run, so this observation is excluded from performance claims. Energy was not measured. Final sustained-load qualification is still pending; no release-ready claim is justified.\n",
        },
        "task": "Read report_a.md and report_b.md. Compare the eligible throughput, latency and memory changes; distinguish the excluded observation. Finish with a JSON object with exactly these keys: eligible_throughput (number), baseline_throughput (number), speedup (number), latency_increase_ms (number), memory_increase_mib (number), excluded_throughput (number), release_ready (boolean), energy_measured (boolean). Base every value on the files.",
        "delegation": "First spawn one child to analyze report_a.md and one child to analyze report_b.md. Wait for both reports, then synthesize the required JSON yourself.",
    },
}


def grade(case: str, workspace: Path, result, *, extra_cases: bool = True) -> dict:
    if case == "coding":
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(workspace)
        # Grade against a fresh, operator-owned test copy outside the agent's
        # workspace. Editing the visible tests cannot manufacture a pass.
        with tempfile.TemporaryDirectory(
            prefix="forge-independent-grader-"
        ) as temporary:
            environment["PYTHONPYCACHEPREFIX"] = str(Path(temporary) / "pycache")
            test = Path(temporary) / "test_acceptance.py"
            extra = "\n@pytest.mark.parametrize('values', [[-4, 6, 13], [0, 0, 0], [0.5, 1.25, 3.75], [7, -1, 2, 8]])\ndef test_unseen_inputs(values):\n    assert mean(values) == pytest.approx(sum(values) / len(values))\n"
            test.write_text(
                CASES[case]["files"]["tests/test_stats.py"]
                + (extra if extra_cases else "")
            )
            command = [sys.executable, "-m", "pytest", "-q", "--", str(test)]
            checked = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        return {
            "passed": result.status == "completed" and checked.returncode == 0,
            "independent_test_exit": checked.returncode,
            "independent_test_output": checked.stdout + checked.stderr,
            "final_source": (workspace / "stats.py").read_text(),
        }
    expected = {
        "eligible_throughput": 64,
        "baseline_throughput": 32,
        "speedup": 2,
        "latency_increase_ms": 10,
        "memory_increase_mib": 256,
        "excluded_throughput": 128,
        "release_ready": False,
        "energy_measured": False,
    }
    try:
        actual = json.loads(result.output or "")
    except (ValueError, TypeError):
        actual = None
    correct = (
        isinstance(actual, dict)
        and actual.keys() == expected.keys()
        and all(
            actual[key] == value
            and (
                type(actual[key]) is bool
                if isinstance(value, bool)
                else type(actual[key]) in (int, float)
            )
            for key, value in expected.items()
        )
    )
    return {
        "passed": result.status == "completed" and correct,
        "expected": expected,
        "actual": actual,
    }


def completion_checker(case: str, mode: str, workspace: Path, store: AgentStore):
    """Objective checks steer unfinished work; factual answers remain withheld."""

    async def check(run_id: str, agent_id: str, result: str):
        if mode == "delegated":
            children = [
                a
                for a in store.agents(run_id)
                if a.parent_id == agent_id and a.status == "completed"
            ]
            if len(children) < 2:
                return CompletionCheck(
                    False,
                    f"The requested workflow requires two completed independent reviews. Only {len(children)} completed; carry out both assigned reviews before finishing.",
                )
        if case == "coding":
            checked = await asyncio.to_thread(
                grade,
                case,
                workspace,
                SimpleNamespace(status="completed"),
                extra_cases=False,
            )
            diagnostic = checked["independent_test_output"]
            if len(diagnostic) > 3000:
                diagnostic = (
                    diagnostic[:700]
                    + "\n[diagnostic middle omitted]\n"
                    + diagnostic[-2200:]
                )
            return CompletionCheck(
                checked["passed"],
                "Independent acceptance tests passed.\n" + diagnostic
                if checked["passed"]
                else "The requested code repair is unfinished. Independent acceptance tests still fail; correct the implementation using these real diagnostics.\n"
                + diagnostic,
            )
        # Validate requested structure/types only. Never reveal the grader's
        # expected numeric values or factual labels to the model.
        try:
            value = json.loads(result)
        except (ValueError, TypeError):
            return CompletionCheck(
                False,
                "The final answer must be a JSON object with the eight fields requested by the user, derived from both source reports.",
            )
        numeric = {
            "eligible_throughput",
            "baseline_throughput",
            "speedup",
            "latency_increase_ms",
            "memory_increase_mib",
            "excluded_throughput",
        }
        boolean = {"release_ready", "energy_measured"}
        if (
            not isinstance(value, dict)
            or value.keys() != numeric | boolean
            or any(type(value[k]) not in (int, float) for k in numeric)
            or any(type(value[k]) is not bool for k in boolean)
        ):
            return CompletionCheck(
                False,
                "Answer format is incomplete: use exactly the six requested numeric fields and the two requested boolean fields. Reread the task/source evidence; no factual values are supplied by this checker.",
            )
        return CompletionCheck(
            True,
            "Requested JSON format and workflow checks passed; factual grading remains independent.",
        )

    return check


async def evaluate(args):
    config = load_config(args.config)
    sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path("python/forge_llm/agents").glob("*.py")
    }
    started = time.monotonic()
    pool = await create_pool(config)
    startup = time.monotonic() - started
    rows = []
    try:
        for repetition in range(args.repetitions):
            modes = (
                ["single", "delegated"]
                if repetition % 2 == 0
                else ["delegated", "single"]
            )
            for case in args.cases:
                for mode in modes:
                    with tempfile.TemporaryDirectory(
                        prefix="forge-task-evaluation-"
                    ) as temporary:
                        workspace = Path(temporary)
                        for name, content in CASES[case]["files"].items():
                            path = workspace / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(content)
                        tools = WorkspaceTools(
                            workspace,
                            workspace / "artifacts",
                            allow_write=case == "coding",
                            allow_tests=case == "coding",
                        )
                        limits = {
                            **config.get("runtime", {}),
                            "max_agents": 1 if mode == "single" else 3,
                            "max_depth": 1,
                            "max_concurrent_generations": 1 if mode == "single" else 2,
                            "max_output_tokens": args.max_output_tokens,
                            "max_steps_per_agent": args.max_steps,
                            "deadline_seconds": args.deadline,
                            "max_tool_result_chars": 2000,
                        }
                        with AgentStore() as store:
                            runtime = AgentRuntime(
                                pool,
                                store,
                                tools.mapping(),
                                RuntimeConfig(**limits),
                                tool_descriptions=tools.tool_descriptions(),
                                replay_safe_tools=tools.replay_safe_tools,
                                completion_validator=completion_checker(
                                    case, mode, workspace, store
                                ),
                            )
                            task = CASES[case]["task"] + (
                                " " + CASES[case]["delegation"]
                                if mode == "delegated"
                                else " Solve this yourself without spawning children."
                            )
                            begin = time.monotonic()
                            result = await runtime.run(task)
                            elapsed = time.monotonic() - begin
                            grading = await asyncio.to_thread(
                                grade, case, workspace, result
                            )
                            events = store.events(result.run_id)
                            row = {
                                "case": case,
                                "mode": mode,
                                "repetition": repetition,
                                "elapsed_seconds": elapsed,
                                "result": asdict(result),
                                "grading": grading,
                                "spawned_agents": len(result.agents) - 1,
                                "model_calls": sum(
                                    e["kind"] == "generation" for e in events
                                ),
                                "events": events,
                            }
                            rows.append(row)
                            # Preserve each completed trial before the next long
                            # model call. A failed or interrupted run keeps evidence.
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            partial = args.output.with_suffix(
                                args.output.suffix + ".partial.jsonl"
                            )
                            with partial.open("a") as stream:
                                stream.write(json.dumps(row) + "\n")
                            print(
                                json.dumps(
                                    {
                                        key: row[key]
                                        for key in (
                                            "case",
                                            "mode",
                                            "elapsed_seconds",
                                            "spawned_agents",
                                            "model_calls",
                                        )
                                    }
                                    | {
                                        "passed": grading["passed"],
                                        "status": result.status,
                                    }
                                ),
                                flush=True,
                            )
    finally:
        await pool.close()
    identities = []
    for worker in config["workers"]:
        if "model" in worker and Path(worker["model"]).is_file():
            identities.append(model_provenance(Path(worker["model"])))
    current_sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path("python/forge_llm/agents").glob("*.py")
    }
    return {
        "schema": "forge_agent_task_evaluation_v1",
        "configuration": config,
        "startup_seconds": startup,
        "environment": source_provenance(),
        "model_provenance": identities,
        "source_sha256": sources,
        "source_changed_during_run": sources != current_sources,
        "cases": CASES,
        "rows": rows,
        "limitations": [
            "Two small controlled tasks; no general task-success claim.",
            "Single and delegated modes share task content, tools and global budgets, but delegated prompts request independent review.",
            "Protocol/control tests with scripted backends are separate evidence.",
            "Completion steering uses immutable copies of visible coding tests and requested JSON structure/child completion for research. Additional coding inputs are graded only after completion; research fact values remain withheld from the model.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--deadline", type=float, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a fresh output path is required")
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    with exclusive_gpu_workflow():
        report = asyncio.run(evaluate(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
