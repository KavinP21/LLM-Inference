"""Whole-file before/after snapshots supplement model-data header identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_llm.benchmark import source_provenance
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow


def snapshot(root):
    frozen_path = root / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    runtime = source_provenance()["runtime_source_sha256"]
    if runtime != frozen["environment"]["runtime_source_sha256"]:
        raise ValueError("runtime drift after freeze")
    paths = {frozen_path}
    for item in frozen["artifacts"].values():
        for field in ["source", "model", "rtn_control", "stats", "calibration_report"]:
            path = Path(item[field])
            paths.add(path)
            if field in {"source", "model", "rtn_control"}:
                paths.add(path.with_suffix(path.suffix + ".json"))
    return {
        "schema_version": 1,
        "runtime_source_sha256": runtime,
        "files": {
            str(p): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in sorted(paths)
        },
        "scope": "Complete file bytes, including architecture/descriptors; read-only and outside timed inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        report = snapshot(args.root)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    unchanged = baseline is None or (
        baseline["files"] == report["files"]
        and baseline["runtime_source_sha256"] == report["runtime_source_sha256"]
    )
    report["matches_baseline"] = unchanged
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"files": len(report["files"]), "matches_baseline": unchanged}))
    if not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
