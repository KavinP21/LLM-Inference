"""Markdown tables from a bench.sbatch results folder.

python tools/pace/summary.py benchmark-results/pace-bench-<job id>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _fmt(value, unit=""):
    return "–" if value is None else f"{value}{unit}"


def multi_user(folder: Path) -> list[str]:
    rows = []
    for path in sorted(folder.glob("multi-user-*.json")):
        match = re.match(r"multi-user-(.+)-(\d+)\.json", path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rates = [u["tok_per_s"] for u in data["users"]]
        gaps = data["gap_while_late_prefills_ms"]
        rows.append(
            (
                match[1],
                int(match[2]),
                f"| {match[1]} | {match[2]} | {min(rates)}–{max(rates)} | "
                f"{data['aggregate_tok_per_s']} | "
                f"{_fmt(data['late_time_to_first_token_s'], ' s')} | "
                f"{_fmt(gaps['p50'], ' ms')} / {_fmt(gaps['max'], ' ms')} | "
                f"{sum(data['leaked_pages'].values())} |",
            )
        )
    if not rows:
        return []
    return [
        "## Engine: users decoding together, then an 8K prompt arrives",
        "",
        "| Model | Users | Each user tok/s | Total tok/s | 8K prompt first token "
        "| Others' token gap meanwhile (p50 / max) | Pages leaked |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: |",
        *[row for _, _, row in sorted(rows)],
        "",
    ]


def serve_load(folder: Path) -> list[str]:
    lines = []
    for path in sorted(folder.glob("serve-load-*.json")):
        data = json.loads(path.read_text())
        name = path.stem.removeprefix("serve-load-")
        lines += [
            f"## Over HTTP: {name}, {data['turns']}-turn chats, "
            f"{data['document_words']}-word documents, up to {data['max_tokens']} "
            "tokens a reply",
            "",
            "| Users | Total tok/s | Each user decode tok/s (p50 / min) "
            "| First token, turn 1 (p50 / p95) | First token, later (p50 / p95) "
            "| Later prompts cached | Errors |",
            "| ---: | ---: | --- | --- | --- | ---: | ---: |",
        ]
        for level in data["levels"]:
            rate, first, later = (
                level["per_user_decode_tok_per_s"],
                level["ttft_first_turn_s"],
                level["ttft_later_turns_s"],
            )
            cached = level["later_turns_cached_fraction"]
            lines.append(
                f"| {level['users']} | {level['aggregate_tok_per_s']} | "
                f"{_fmt(rate['p50'])} / {_fmt(rate['min'])} | "
                f"{_fmt(first['p50'], ' s')} / {_fmt(first['p95'], ' s')} | "
                f"{_fmt(later['p50'], ' s')} / {_fmt(later['p95'], ' s')} | "
                f"{_fmt(None if cached is None else round(cached * 100, 1), '%')} | "
                f"{len(level['errors'])} |"
            )
        lines.append("")
    return lines


def long_context(folder: Path) -> list[str]:
    lines = []
    for path in sorted(folder.glob("long-context-*.json")):
        data = json.loads(path.read_text())
        first, follow = data["first_turn"], data["follow_up"]
        lines += [
            f"## Long context: {data['context_tokens']} tokens",
            "",
            "| Turn | First token | Prefilled tokens | Prefill tok/s | Decode tok/s |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| First | {first['ttft_ms'] / 1000:.1f} s | {first['prefilled_tokens']} | "
            f"{_fmt(first['prefill_tps'])} | {first['decode_tps']} |",
            f"| Follow-up | {follow['ttft_ms'] / 1000:.2f} s | "
            f"{follow['prefilled_tokens']} | {_fmt(follow['prefill_tps'])} | "
            f"{follow['decode_tps']} |",
            "",
        ]
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    lines = multi_user(folder) + serve_load(folder) + long_context(folder)
    print("\n".join(lines) if lines else f"no results in {folder}")


if __name__ == "__main__":
    main()
