"""Post-freeze final readback, including byte diagnostics and test evidence."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_cached_checkpoint import audit, load_frozen
from run_second_order_checkpoint import save


def byte_contract(report):
    observations = report["observations"]
    if len(observations) != 25 or [o["case_index"] for o in observations] != list(
        range(25)
    ):
        raise ValueError("incomplete byte-sensitive case coverage")
    total, equal = 0, 0
    for case in observations:
        rows = case["rows"]
        if [r["position"] for r in rows] != list(range(32)):
            raise ValueError("incomplete cached byte-sensitive positions")
        for row in rows:
            for field in ("native", "composed"):
                value = row[field]
                if (
                    np.dtype(value["dtype"]).kind != "f"
                    or len(value["shape"]) != 1
                    or not isinstance(value["shape"][0], int)
                    or isinstance(value["shape"][0], bool)
                    or value["shape"][0] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                ):
                    raise ValueError("invalid cached logit byte identity")
        count = sum(r["native"] == r["composed"] for r in rows)
        if count != case["byte_equal_rows"] or not case["private_cache_reclaimed"]:
            raise ValueError("byte-sensitive case summary/cleanup disagreement")
        total += len(rows)
        equal += count
    passed = total == equal == 800
    if (
        report["total_rows"] != total
        or report["byte_equal_rows"] != equal
        or report["passed"] != passed
        or not passed
    ):
        raise ValueError("byte-sensitive native/composed contract failed")
    return equal


def test_contract(path, expected_tests):
    suites = ET.parse(path).getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, "0"))
    if counts["tests"] != expected_tests or any(
        counts[key] for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("incomplete or failed final regression suite")
    return counts


def verify(root, expected_tests):
    frozen = load_frozen(root)
    index_path = root / "evidence-index.json"
    index = json.loads(index_path.read_text())
    recomputed = audit(root)
    files = {
        name: checksum
        for name, checksum in recomputed["files"].items()
        if name != "verification.json"
    }
    if (
        index["files"] != files
        or index["runtime_source_sha256"] != recomputed["runtime_source_sha256"]
        or index["strict_checkpoint_passed"] != recomputed["strict_checkpoint_passed"]
    ):
        raise ValueError("indexed evidence changed or disagrees with raw gate readback")
    before, after = [
        json.loads((root / name).read_text())
        for name in ("integrity-before.json", "integrity-after.json")
    ]
    if (
        len(before["files"]) != 17
        or before["files"] != after["files"]
        or before["runtime_source_sha256"] != after["runtime_source_sha256"]
        or not after["matches_baseline"]
    ):
        raise ValueError("frozen whole-file integrity changed")
    for name, expected in before["files"].items():
        path = Path(name)
        if (
            path.stat().st_size != expected["bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"frozen complete file changed: {path}")
    diagnostic_script = Path(__file__).with_name("verify_native_cached_bytes.py")
    rows = 0
    diagnostic_paths = list((root / "numerical-bytes").glob("*.json"))
    if len(diagnostic_paths) != 4:
        raise ValueError("incomplete supplemental byte diagnostics")
    for family, item in frozen["artifacts"].items():
        for corpus in (0, 1):
            path = root / "numerical-bytes" / f"{family}-corpus{corpus}.json"
            report = json.loads(path.read_text())
            quality = (
                root
                / "regression"
                / f"{family}-reconstruct-corpus{corpus}-quality.json"
            )
            if (
                report["configuration"]["quality_report_sha256"] != file_sha256(quality)
                or report["configuration"]["script_sha256"]
                != file_sha256(diagnostic_script)
                or report["configuration"]["decode_mode"] != "rowwise"
                or report["environment"]["runtime_source_sha256"]
                != index["runtime_source_sha256"]
                or report["model"]["model_data_sha256"]
                != item["model_provenance"]["model_data_sha256"]
            ):
                raise ValueError("byte diagnostic provenance changed")
            rows += byte_contract(report)
    test_path = root / "test-results.xml"
    return {
        "schema_version": 1,
        "environment": {"runtime_source_sha256": index["runtime_source_sha256"]},
        "audit_script_sha256": file_sha256(Path(__file__)),
        "evidence_index_sha256": file_sha256(index_path),
        "indexed_json_files_verified": len(files),
        "frozen_complete_files_verified": len(before["files"]),
        "native_composed_byte_equal_cached_rows": rows,
        "final_tests": test_contract(test_path, expected_tests),
        "test_results_sha256": file_sha256(test_path),
        "benchmark_results": recomputed["benchmark_results"],
        "cached_quality_rows": recomputed["cached_quality_rows"],
        "strict_checkpoint_passed": recomputed["strict_checkpoint_passed"],
        "readback_passed": True,
        "limitations": "Post-freeze audit, not fitting. Checks recorded raw observations and file identities, not missing hardware counters or general model quality. Strict quality failure is retained, not waived.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    if args.expected_tests <= 0:
        parser.error("expected-tests must be positive")
    with exclusive_gpu_workflow():
        report = verify(args.root, args.expected_tests)
    save(args.root / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
