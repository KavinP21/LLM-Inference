"""Operator-selected, executable completion criteria for agent tasks.

Criteria check the delivered result and workspace. They do not guess a goal
from prompt keywords or supply a canned solution to the model.
"""

from __future__ import annotations

import json
from typing import Any

from .native_tools import _check_schema, _validate
from .protocol import _unique_object
from .store import AgentStore
from .tools import WorkspaceTools


def _reject_constant(value: str):
    raise ValueError(f"Nonfinite JSON value {value!r} is not permitted")


class CompletionProfile:
    def __init__(
        self, criteria: dict[str, Any], tools: WorkspaceTools, store: AgentStore
    ):
        if not isinstance(criteria, dict) or criteria.keys() - {
            "tests",
            "json_schema",
            "min_children",
            "required_artifacts",
        }:
            raise ValueError(
                "acceptance supports tests, json_schema, min_children and required_artifacts"
            )
        self.criteria = json.loads(json.dumps(criteria, allow_nan=False))
        self.tools, self.store = tools, store
        minimum = criteria.get("min_children", 0)
        if type(minimum) is not int or not 0 <= minimum <= 31:
            raise ValueError("min_children must be an integer between zero and 31")
        tests = criteria.get("tests", [])
        artifacts = criteria.get("required_artifacts", [])
        if (
            not tests
            and not artifacts
            and not minimum
            and "json_schema" not in criteria
        ):
            raise ValueError(
                "acceptance must contain at least one enforceable criterion"
            )
        for names, field in ((tests, "tests"), (artifacts, "required_artifacts")):
            if (
                not isinstance(names, list)
                or len(names) > 20
                or any(not isinstance(p, str) or not p for p in names)
            ):
                raise ValueError(f"{field} must contain at most 20 nonempty paths")
        if tests and not tools.allow_tests:
            raise ValueError(
                "acceptance tests require the operator's --allow-tests permission"
            )
        if "json_schema" in criteria:
            _check_schema(criteria["json_schema"])
        for name in artifacts:
            tools._path(name, artifact=True)
        for name in tests:
            tools._path(name)
        # Acceptance assertions belong to the operator. Keep their original
        # versions independent of agent edits, including across CLI resume.
        self.test_fingerprints = tools.protect_files(tests)

    def _tests_unchanged(self) -> bool:
        try:
            return (
                self.tools.protected_file_hashes(self.criteria.get("tests", []))
                == self.test_fingerprints
            )
        except (OSError, ValueError):
            return False

    async def __call__(self, run_id: str, agent_id: str, result: str):
        from .runtime import CompletionCheck

        minimum = self.criteria.get("min_children", 0)
        completed = [
            a
            for a in self.store.agents(run_id)
            if a.parent_id == agent_id and a.status == "completed"
        ]
        if len(completed) < minimum:
            return CompletionCheck(
                False,
                f"Completion requires {minimum} completed direct child tasks; only {len(completed)} completed. Finish the requested independent reviews before synthesis.",
            )
        if "json_schema" in self.criteria:
            try:
                value = json.loads(
                    result,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                # JSON's exponent syntax can also overflow into infinity.
                json.dumps(value, allow_nan=False)
                _validate(value, self.criteria["json_schema"], "final answer")
            except (ValueError, TypeError, RecursionError) as exc:
                return CompletionCheck(
                    False,
                    f"Final answer format check failed: {exc}. Return the requested structured answer from the source evidence.",
                )
        for name in self.criteria.get("required_artifacts", []):
            path = self.tools._path(name, artifact=True)
            if not path.is_file() or not path.stat().st_size:
                return CompletionCheck(
                    False, f"Required nonempty artifact {name!r} has not been saved."
                )
        tests = self.criteria.get("tests", [])
        if tests:
            if not self._tests_unchanged():
                return CompletionCheck(
                    False,
                    "Operator-selected acceptance tests changed or became unreadable. Restore their original versions; modifying assertions cannot qualify completion.",
                )
            report = json.loads(await self.tools.run_tests({"paths": tests}))
            if not self._tests_unchanged():
                return CompletionCheck(
                    False,
                    "Operator-selected acceptance tests changed during execution; this result cannot qualify completion.",
                )
            if report["exit_code"]:
                output = report["output"]
                diagnostic = (
                    output
                    if len(output) <= 3000
                    else output[:700]
                    + "\n[diagnostic middle omitted]\n"
                    + output[-2200:]
                )
                return CompletionCheck(
                    False,
                    f"Acceptance tests still fail (exit {report['exit_code']}). Correct the implementation and verify it before finishing.\n{diagnostic}",
                )
        return CompletionCheck(True, "Operator-selected completion checks passed.")
