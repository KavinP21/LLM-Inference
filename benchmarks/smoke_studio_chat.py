"""Real local chat startup, follow-up memory and tool-use integration fixture."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from instinct.harness.service import StudioService

ROOT = Path(__file__).resolve().parents[1]


async def settle(service, chat_id, count):
    async with asyncio.timeout(240):
        while True:
            detail = await service.conversations.detail(chat_id)
            turns = detail["turns"]
            if len(turns) >= count and turns[count - 1]["status"] in {
                "completed",
                "completed_with_failures",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return detail
            await asyncio.sleep(0.25)


async def execute(args):
    configuration = json.loads(Path(args.config).read_text())
    configuration["workers"] = configuration["workers"][:1]
    for name in ("model", "tokenizer"):
        path = Path(configuration["workers"][0][name])
        if not path.is_absolute():
            configuration["workers"][0][name] = str(ROOT / path)
    configuration["runtime"].update(
        max_agents=2,
        max_concurrent_generations=1,
        max_steps_per_agent=6,
        max_output_tokens=384,
        max_total_tokens=128000,
        deadline_seconds=240,
    )
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    sources = sorted((ROOT / "python/instinct/harness").rglob("*.py")) + sorted(
        (ROOT / "python/instinct/harness/web").glob("*")
    )
    source_identity = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
        if p.is_file()
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="instinct-chat-real-") as directory:
        workspace = Path(directory)
        (workspace / "README.md").write_text(
            "# Juniper Workshop\n\nAn isolated chat integration fixture, not a benchmark.\n"
        )
        service = StudioService(
            workspace, ROOT / ".instinct/studio-chat-smoke", config=configuration
        )
        try:
            first = await service.conversations.create(
                {
                    "message": "My project codename is Cedar. Reply with Cedar and nothing else. Do not use JSON.",
                    "policy": "read_only",
                    "request_id": uuid.uuid4().hex,
                }
            )
            print(
                "First prompt durably accepted before explicit model startup.",
                flush=True,
            )
            chat_id = first["chat_id"]
            await settle(service, chat_id, 1)
            await service.conversations.message(
                chat_id,
                {
                    "message": "What project codename did I give you? Reply with only the codename. Do not use JSON.",
                    "request_id": uuid.uuid4().hex,
                },
            )
            await settle(service, chat_id, 2)
            await service.conversations.message(
                chat_id,
                {
                    "message": "Read README.md and report its exact heading in one plain sentence.",
                    "request_id": uuid.uuid4().hex,
                },
            )
            detail = await settle(service, chat_id, 3)
            outputs = [turn.get("output") or "" for turn in detail["turns"]]
            reads = [
                event
                for turn in detail["turns"]
                for event in turn.get("events", [])
                if event["kind"] == "tool_result"
                and event["payload"].get("name") == "read_file"
            ]
            report = {
                "scope": "One local main-model chat, three sequential turns: automatic startup, conversational recall and a real file read. Not broad reliability evidence.",
                "source_revision": source_revision,
                "harness_file_sha256": source_identity,
                "recipe_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "workers": service.identities,
                "outputs": outputs,
                "turn_statuses": [turn["status"] for turn in detail["turns"]],
                "read_receipts": len(reads),
                "chat_count": len((await service.conversations.overview())["chats"]),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "passed": all(turn["status"] == "completed" for turn in detail["turns"])
                and "Cedar" in outputs[0]
                and "Cedar" in outputs[1]
                and "Juniper Workshop" in outputs[2]
                and len(reads) >= 1
                and all(not output.lstrip().startswith("{") for output in outputs),
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "passed",
                            "outputs",
                            "turn_statuses",
                            "read_receipts",
                            "elapsed_seconds",
                        )
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
        "--output", default=str(ROOT / "benchmark-results/studio-chat-smoke.json")
    )
    asyncio.run(execute(parser.parse_args()))
