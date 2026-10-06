from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric(result: dict, path: str) -> float:
    value: object = result
    for part in path.split("."):
        value = value[part]  # type: ignore[index]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize raw Forge benchmark JSON as Markdown"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/summary.md")
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("*.json")):
        payload = json.loads(path.read_text())
        if "aggregate" not in payload:
            continue
        config = payload["configuration"]
        rows.append(
            (
                payload.get("implementation", "unknown"),
                config.get(
                    "prompt_token_lengths", [config.get("prompt_length", "mixed")]
                )[0]
                if isinstance(config.get("prompt_token_lengths", []), list)
                else "mixed",
                config["output_length"],
                config["concurrency"],
                metric(payload, "aggregate.generated_tokens_per_second"),
                metric(payload, "aggregate.ttft_ms.p50"),
                metric(payload, "aggregate.ttft_ms.p95"),
                metric(payload, "aggregate.tpot_ms.p50"),
                metric(payload, "aggregate.tpot_ms.p95"),
            )
        )
    lines = [
        "# Benchmark summary",
        "",
        "Generated from raw JSON; environment and per-request observations remain in the source files.",
        "",
        "| Implementation | Prompt | Output | Concurrency | tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | TPOT p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.2f} | "
            f"{row[5]:.2f} | {row[6]:.2f} | {row[7]:.2f} | {row[8]:.2f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
