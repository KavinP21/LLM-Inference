"""Exercise the real desktop model owner and Tk controls with a synthetic image.

Requires a macOS graphical session and the daily-driver and legacy 0.5B models.
Run from the repository root with PYTHONPATH=python. The image should display
FORGE 42; no real private photo is needed. Does not measure UI performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import forge_chat
from forge_chat import ForgeChat
from forge_llm.benchmark import source_provenance
from gpu_guard import exclusive_gpu_workflow


def qualify(args, report):
    app = ForgeChat()
    app.withdraw()

    def wait(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while not predicate():
            app.update()
            if time.monotonic() > deadline:
                raise TimeoutError(app.status_label.cget("text"))
            time.sleep(0.01)
        app.update()

    def generate(prompt, limit=96):
        app.max_tokens.delete(0, "end")
        app.max_tokens.insert(0, str(limit))
        app.prompt.delete("1.0", "end")
        app.prompt.insert("1.0", prompt)
        app._generate()
        assert app.busy
        wait(lambda: not app.busy)

    def record(name, passed):
        report["rows"].append(
            {
                "case": name,
                "passed": bool(passed),
                "status": app.status_label.cget("text"),
                "conversation": copy.deepcopy(app.conversation),
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": name, "passed": bool(passed)}), flush=True)

    try:
        wait(lambda: not app.loading)
        assert app.ready
        record("default-qwen35", app.profile.key == "qwen35" and not app.thinking.get())
        forge_chat.filedialog.askopenfilename = lambda **_: str(args.image.resolve())
        app._attach("image")
        generate("Read the large printed text and return only that text.")
        record(
            "image",
            bool(app.conversation)
            and "FORGE 42" in app.conversation[-1]["content"].upper(),
        )
        before = len(app.conversation)
        generate("What number was shown in the image? Answer only with the number.", 32)
        record(
            "visual-followup",
            len(app.conversation) == before + 2
            and "42" in app.conversation[-1]["content"],
        )
        before = len(app.conversation)
        app.thinking.set(True)
        generate("Prove the sum of the first n odd integers is n squared.", 1)
        record(
            "truncated-thinking",
            len(app.conversation) == before
            and "thinking" in app.status_label.cget("text").lower(),
        )
        app.thinking.set(False)
        before = len(app.conversation)
        app.max_tokens.delete(0, "end")
        app.max_tokens.insert(0, "4096")
        app.prompt.delete("1.0", "end")
        app.prompt.insert("1.0", "List all integers from 1 through 10000.")
        app._generate()
        app.after(200, app._stop)
        wait(lambda: not app.busy)
        record(
            "stop",
            len(app.conversation) == before
            and "stopped" in app.status_label.cget("text").lower(),
        )
        app._clear_chat()
        record(
            "clear-media",
            app.worker.media_store.total_bytes == 0 and not app.conversation,
        )
        generate("What is 17+23? Reply with only the number.", 32768)
        record(
            "output-budget-clamp",
            len(app.conversation) == 2
            and app.conversation[-1]["content"].strip() == "40",
        )
        before = len(app.conversation)
        draft = "This next draft must remain."
        app.max_tokens.set("32")
        app.prompt.delete("1.0", "end")
        app.prompt.insert("1.0", "What is 7+8? Reply with only the number.")
        app._generate()

        def compose():
            app.prompt.delete("1.0", "end")
            app.prompt.insert("1.0", draft)

        app.after(100, compose)
        wait(lambda: not app.busy)
        record(
            "preserve-composed-draft",
            len(app.conversation) == before + 2
            and app.prompt.get("1.0", "end-1c") == draft,
        )
        app.model_picker.current(1)
        app._select_model()
        wait(lambda: not app.loading)
        record("switch-to-native-05", app.ready and app.profile.key == "qwen05")
    finally:
        directory = Path(app.worker.media_store.directory.name)
        app._close()
        app.worker.thread.join(30)
        report["closed"] = (
            not app.worker.thread.is_alive()
            and app.worker.engine is None
            and not directory.exists()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh evidence path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "forge_qwen35_ui_qualification_v1",
        "source": source_provenance(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
        "method": "Actual Tk callbacks and actor-owned model execution; hidden window; "
        "synthetic image, visual follow-up, thinking truncation, stop, clear, model switch, shutdown.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    try:
        with exclusive_gpu_workflow():
            qualify(args, report)
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
    if (
        report["source_changed_during_run"]
        or not report.get("closed")
        or not all(row["passed"] for row in report["rows"])
    ):
        raise SystemExit("qualification failed; inspect retained report")


if __name__ == "__main__":
    main()
