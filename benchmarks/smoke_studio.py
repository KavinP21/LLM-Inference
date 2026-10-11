"""One real-model Studio delegation fixture; never a broad capability benchmark."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from forge_llm.harness.service import StudioService

ROOT = Path(__file__).resolve().parents[1]


async def execute(args):
    configuration = json.loads(Path(args.config).read_text())
    configuration["workers"] = configuration["workers"][:1]
    worker = configuration["workers"][0]
    worker["model"] = str(ROOT / worker["model"])
    worker["tokenizer"] = str(ROOT / worker["tokenizer"])
    configuration["runtime"].update(
        max_agents=2,
        max_concurrent_generations=1,
        max_steps_per_agent=12,
        max_output_tokens=384,
        max_total_tokens=128000,
        deadline_seconds=300,
    )
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_files = sorted((ROOT / "python/forge_llm/harness").rglob("*.py")) + sorted(
        (ROOT / "python/forge_llm/harness/web").glob("*")
    )
    source_identity = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
        if path.is_file()
    }
    recipe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="forge-studio-real-") as directory:
        workspace = Path(directory)
        (workspace / "README.md").write_text(
            "# Cedar Tools\n\nA fixture for the Forge Studio integration smoke test.\nIt contains no executable code and supports no claimed benchmarks.\n"
        )
        service = StudioService(
            workspace, ROOT / ".forge/studio-smoke", config=configuration
        )
        started = time.monotonic()
        try:
            await service.start_backend()
            while service.backend_status == "loading":
                await asyncio.sleep(1)
            if service.backend_status != "ready":
                raise RuntimeError(service.backend_error)
            print(
                "Local Qwen3.5 worker ready; submitting delegated inspection.",
                flush=True,
            )
            task = "Delegate exactly one child agent to read README.md and report its project title. Wait for the child result, then finish with the project title and say this is only an integration fixture. Use only observed file evidence. Do not create any files."
            run_id = (
                await service.create_run(
                    {
                        "task": task,
                        "policy": "read_only",
                        "max_agents": 2,
                        "max_steps_per_agent": 12,
                        "max_total_tokens": 128000,
                    }
                )
            )["run_id"]
            await asyncio.wait_for(service.jobs[run_id], 330)
            detail = await service.detail(run_id, export=True)
            children = [a for a in detail["agents"] if a["parent_id"] == "root"]
            reads = [
                e
                for e in detail["events"]
                if e["kind"] == "tool_result"
                and e["payload"].get("name") == "read_file"
            ]
            report = {
                "scope": "One local Qwen3.5-4B mission using actual model inference; not a broad capability evaluation",
                "source_revision": source_revision,
                "harness_file_sha256": source_identity,
                "recipe_sha256": recipe_sha256,
                "worker_configuration": configuration,
                "task": task,
                "fixture_readme": (workspace / "README.md").read_text(),
                "status": detail["run"]["status"],
                "output": detail["output"],
                "error": detail["error"],
                "agent_count": len(detail["agents"]),
                "completed_children": sum(a["status"] == "completed" for a in children),
                "read_receipts": len(reads),
                "usage": detail["usage"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "workers": service.identities,
                "event_kinds": [e["kind"] for e in detail["events"]],
                "passed": detail["run"]["status"] == "completed"
                and len(children) == 1
                and children[0]["status"] == "completed"
                and len(reads) >= 1
                and "Cedar" in (detail["output"] or ""),
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in [
                            "passed",
                            "status",
                            "output",
                            "error",
                            "agent_count",
                            "read_receipts",
                            "elapsed_seconds",
                        ]
                    }
                ),
                flush=True,
            )
        finally:
            await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "examples/agents-studio.json"))
    parser.add_argument(
        "--output", default=str(ROOT / "benchmark-results/studio-live-smoke.json")
    )
    asyncio.run(execute(parser.parse_args()))
