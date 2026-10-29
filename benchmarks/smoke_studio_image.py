"""Real image upload, follow-up vision context and developer tools smoke."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from forge_llm.agents.media import media_from_file
from forge_llm.harness.service import StudioService
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


async def settle(service, chat_id, count):
    async with asyncio.timeout(240):
        while True:
            detail = await service.conversations.detail(chat_id)
            if len(detail["turns"]) >= count and detail["turns"][count - 1][
                "status"
            ] in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
                "completed_with_failures",
            }:
                return detail
            await asyncio.sleep(0.25)


async def execute(args):
    config = json.loads(Path(args.config).read_text())
    config["workers"] = config["workers"][:1]
    for field in ("model", "tokenizer"):
        value = Path(config["workers"][0][field])
        if not value.is_absolute():
            config["workers"][0][field] = str(ROOT / value)
    config["runtime"].update(
        max_agents=2,
        max_steps_per_agent=8,
        max_output_tokens=384,
        max_total_tokens=128000,
        deadline_seconds=240,
        max_concurrent_generations=1,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    files = sorted((ROOT / "python/forge_llm/harness").rglob("*.py")) + [
        ROOT / "python/forge_llm/agents/runtime.py",
        ROOT / "python/forge_llm/agents/store.py",
    ]
    identity = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="forge-image-smoke-") as directory:
        workspace = Path(directory)
        image = Image.new("RGB", (320, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 140, 160), fill=(255, 0, 0))
        draw.rectangle((180, 20, 300, 160), fill=(0, 0, 255))
        image_path = workspace / "two-colors.png"
        image.save(image_path)
        (workspace / "README.md").write_text("# Image and developer fixture\n")
        service = StudioService(
            workspace,
            ROOT / ".forge/studio-image-smoke" / uuid.uuid4().hex,
            config=config,
        )
        try:
            item = media_from_file(image_path, mime="image/png")
            first = await service.conversations.create(
                {
                    "message": "What color is the rectangle on the left in this image? Answer just the color, no JSON.",
                    "media": [item],
                    "media_names": ["two-colors.png"],
                    "policy": "review",
                    "request_id": uuid.uuid4().hex,
                }
            )
            print(
                "Image prompt accepted; starting the local main-model worker.",
                flush=True,
            )
            chat = first["chat_id"]
            await settle(service, chat, 1)
            await service.conversations.message(
                chat,
                {
                    "message": "What color is the rectangle on the right in that same image? Answer just the color, no JSON.",
                    "request_id": uuid.uuid4().hex,
                },
            )
            detail = await settle(service, chat, 2)
            outputs = [t.get("output") or "" for t in detail["turns"]]
            command_chat = await service.conversations.create(
                {
                    "message": "Run exactly python -c \"print('FORGE_COMMAND_OK')\" using run_command, then report its actual output. Do not edit files.",
                    "policy": "review",
                    "request_id": uuid.uuid4().hex,
                }
            )
            async with asyncio.timeout(120):
                while True:
                    command_detail = await service.conversations.detail(
                        command_chat["chat_id"]
                    )
                    turn = command_detail["turns"][0]
                    for approval in turn["approvals"]:
                        arguments = approval["args"]
                        argv = arguments.get("argv", [])
                        if (
                            approval["tool"] != "run_command"
                            or len(argv) != 3
                            or argv[0] not in {"python", "python3", sys.executable}
                            or argv[1] != "-c"
                            or argv[2]
                            not in {
                                "print('FORGE_COMMAND_OK')",
                                'print("FORGE_COMMAND_OK")',
                            }
                            or arguments.get("cwd", ".") != "."
                        ):
                            raise RuntimeError(
                                "Qualification refused an unexpected command proposal"
                            )
                        await service.decide(approval["id"], {"approved": True})
                    if turn["status"] in {
                        "completed",
                        "failed",
                        "cancelled",
                        "interrupted",
                        "completed_with_failures",
                    }:
                        break
                    await asyncio.sleep(0.1)
            command_receipts = [
                event
                for event in turn["events"]
                if event["kind"] == "tool_result"
                and event["payload"].get("name") == "run_command"
            ]
            successful_command = any(
                json.loads(event["payload"]["result"]).get("exit_code") == 0
                and "FORGE_COMMAND_OK"
                in json.loads(event["payload"]["result"]).get("output", "")
                for event in command_receipts
            )
            report = {
                "scope": "Two image turns plus one narrowly approved Python command through the local main model; not broad vision or coding capability evidence",
                "source_revision": revision,
                "file_sha256": identity,
                "image_sha256": item["sha256"],
                "image_description": "Synthetic white320x180PNG, redleft rectangle and blueright rectangle",
                "workers": service.identities,
                "statuses": [t["status"] for t in detail["turns"]],
                "outputs": outputs,
                "image_metadata_count": len(detail["turns"][0]["media"]),
                "command_status": turn["status"],
                "command_output": turn["output"],
                "successful_command_receipts": int(successful_command),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "passed": all(t["status"] == "completed" for t in detail["turns"])
                and "red" in outputs[0].lower()
                and "blue" in outputs[1].lower()
                and turn["status"] == "completed"
                and successful_command,
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
                            "statuses",
                            "outputs",
                            "image_metadata_count",
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
        "--output", default=str(ROOT / "benchmark-results/studio-image-smoke.json")
    )
    asyncio.run(execute(parser.parse_args()))
