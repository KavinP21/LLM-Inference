"""Qualify real image transport, child inheritance and durable media on local replicas.

Run from the repository root with PYTHONPATH=python. The image must be a synthetic
fixture displaying FORGE 42. This is a narrow integration gate, not a general
visual-agent benchmark. Reports refuse overwrite and retain source identity.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from forge_llm.agents import (
    AgentRuntime,
    AgentStore,
    ChatMessage,
    CompletionCheck,
    RuntimeConfig,
    WorkspaceTools,
)
from forge_llm.agents.backends import ContextLengthError, GenerationCancelledError
from forge_llm.agents.cli import create_pool, load_config, worker_identities
from forge_llm.agents.media import media_from_file
from forge_llm.benchmark import source_provenance
from gpu_guard import exclusive_gpu_workflow


async def qualify(args, report):
    media = media_from_file(args.image, mime="image/png")
    report["media_sha256"] = media["sha256"]
    started = time.perf_counter()
    pool = await create_pool(load_config(args.config))
    report["startup_seconds"] = time.perf_counter() - started

    def record(case, passed, **details):
        report["rows"].append({"case": case, "passed": bool(passed), **details})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": case, "passed": bool(passed)}), flush=True)

    try:
        report["workers"] = await worker_identities(pool)
        message = ChatMessage(
            "user", "Read the large printed text. Return only that text.", (media,)
        )
        output = await pool.generate([message], 64, "q35-visual-http")
        record("image-http", "FORGE 42" in output.text.upper(), result=asdict(output))
        rejected = False
        try:
            await pool.generate(
                [message], 64, "q35-visual-context-rejection", max_context_tokens=128
            )
        except ContextLengthError:
            rejected = True
        record("expanded-context-rejection", rejected)

        with tempfile.TemporaryDirectory(prefix="forge-q35-visual-agents-") as tmp:
            workspace = Path(tmp)
            database = workspace / "journal.sqlite"
            tools = WorkspaceTools(workspace, workspace / "artifacts")
            with AgentStore(database) as store:

                async def completed_child(run_id, agent_id, text):
                    children = [
                        agent
                        for agent in store.agents(run_id)
                        if agent.parent_id == agent_id and agent.status == "completed"
                    ]
                    return CompletionCheck(
                        bool(children),
                        "Complete the requested independent child review before finishing."
                        if not children
                        else "Requested child review completed; content is graded separately.",
                    )

                runtime = AgentRuntime(
                    pool,
                    store,
                    tools.mapping(),
                    RuntimeConfig(
                        max_agents=2,
                        max_depth=1,
                        max_concurrent_generations=2,
                        max_output_tokens=256,
                        max_steps_per_agent=12,
                        max_total_tokens=128000,
                        max_context_tokens=24000,
                        deadline_seconds=180,
                    ),
                    tool_descriptions=tools.tool_descriptions(),
                    replay_safe_tools=tools.replay_safe_tools,
                    completion_validator=completed_child,
                )
                started = time.perf_counter()
                result = await runtime.run(
                    "Spawn one child to independently read the text in the attached "
                    "image. Wait for its review, then report only the printed text. "
                    "The child should simply read and return the text, without spawning.",
                    media=(media,),
                )
                children = [
                    agent for agent in result.agents if agent.parent_id == "root"
                ]
                record(
                    "visual-child-task",
                    result.status == "completed"
                    and result.completion_verified
                    and any(child.status == "completed" for child in children)
                    and "FORGE 42" in (result.output or "").upper(),
                    elapsed_seconds=time.perf_counter() - started,
                    result=asdict(result),
                    events=store.events(result.run_id),
                )
                run_id = result.run_id
                child_ids = [child.agent_id for child in children]
            # Reopen the actual database, rather than retaining in-memory media.
            with AgentStore(database) as reopened:
                retained = {
                    agent_id: [
                        blob["sha256"]
                        for message in reopened.messages(run_id, agent_id)
                        for blob in message.media
                    ]
                    for agent_id in ["root", *child_ids]
                }
                record(
                    "durable-child-media",
                    bool(child_ids)
                    and all(media["sha256"] in hashes for hashes in retained.values()),
                    retained_media=retained,
                    blob_count=reopened.connection.execute(
                        "SELECT COUNT(*) FROM media_blobs"
                    ).fetchone()[0],
                )

        # Cancel a real HTTP job, then check a fresh request on that same replica.
        worker = next(iter(pool.backends))
        request = "q35-http-cancel"
        task = asyncio.create_task(
            pool.generate(
                [ChatMessage("user", "List integers 1 through 10000, one per line.")],
                4096,
                request,
                worker=worker,
            )
        )
        await asyncio.sleep(0.4)
        await pool.cancel(request)
        cancelled = False
        try:
            await task
        except (GenerationCancelledError, asyncio.CancelledError):
            cancelled = True
        fresh = await pool.generate(
            [ChatMessage("user", "What is 17+23? Reply with only the number.")],
            32,
            "q35-http-after-cancel",
            worker=worker,
        )
        health = await pool.backends[worker].health()
        stats = health.get("last_runner_stats") or {}
        record(
            "http-cancel-restart",
            cancelled
            and "40" in fresh.text
            and stats.get("active_requests") == stats.get("kv_persistent_bytes") == 0,
            result=asdict(fresh),
            health=health,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/agents-qwen35.json")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; preserve old evidence and choose a fresh path"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "forge_qwen35_worker_qualification_v1",
        "source": source_provenance(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
        "method": "Synthetic FORGE 42 image; authenticated local-process workers; "
        "actual child execution and reopened durable journal; cancellation/restart. "
        "One physical Mac, no multi-host or broad visual-task quality claim.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    try:
        with exclusive_gpu_workflow():
            asyncio.run(qualify(args, report))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["source_after"] = source_provenance()
        report["source_changed_during_run"] = (
            report["source"]["runtime_source_sha256"]
            != report["source_after"]["runtime_source_sha256"]
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["source_changed_during_run"] or not all(
        row["passed"] for row in report["rows"]
    ):
        raise SystemExit("qualification failed; inspect retained report")


if __name__ == "__main__":
    main()
