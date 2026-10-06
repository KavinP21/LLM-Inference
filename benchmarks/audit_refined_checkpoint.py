"""Pre-registered, host-only final readback; never fitting or inference."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from forge_llm.model_file import ModelFile
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_refined_checkpoint import audit, load_contract
from run_second_order_checkpoint import save


def objective_contract(report, expected):
    """Check every selected matrix's independent readback and raw report binding."""
    observations = report["observations"]
    keys = [(o["family"], o["weight"]) for o in observations]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("incomplete or duplicate reconstruction coverage")
    rows, families = 0, {}
    for o in observations:
        binding = expected[o["family"], o["weight"]]
        for field in ("shape", "calibration_report_sha256", "stats_sha256"):
            if o[field] != binding[field]:
                raise ValueError("reconstruction provenance/shape mismatch")
        if (
            type(o["rows"]) is not int
            or o["rows"] != binding["shape"][0]
            or type(o["changed_scale_rows"]) is not int
            or not 0 <= o["changed_scale_rows"] <= o["rows"]
            or o["changed_scale_rows"] != binding["changed_scale_rows"]
            or type(o["changed_quantized_elements"]) is not int
            or not 0 <= o["changed_quantized_elements"] <= math.prod(o["shape"])
            or o["changed_quantized_elements"] != binding["changed_quantized_elements"]
            or type(o["positive_row_deltas_within_tolerance"]) is not int
            or not 0 <= o["positive_row_deltas_within_tolerance"] <= o["rows"]
            or o["row_bound_verified"] is not True
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", o[field])
                for field in ("packed_sha256", "scales_sha256")
            )
        ):
            raise ValueError("invalid reconstruction row/count/byte evidence")
        for field in (
            "baseline_objective",
            "refined_objective",
            "maximum_row_loss_delta",
        ):
            if type(o[field]) not in (int, float) or not math.isfinite(o[field]):
                raise ValueError("invalid reconstruction objective")
        baseline, actual = o["baseline_objective"], o["refined_objective"]
        tolerance = 1e-9 * max(1.0, baseline)
        if (
            min(baseline, actual) < 0
            or actual > baseline + tolerance
            or o["maximum_row_loss_delta"] > tolerance
            or not math.isclose(
                baseline, binding["baseline_objective"], rel_tol=1e-9, abs_tol=1e-9
            )
            or not math.isclose(
                actual, binding["refined_objective"], rel_tol=1e-9, abs_tol=1e-9
            )
        ):
            raise ValueError("reconstruction objective disagrees with calibration")
        rows += o["rows"]
        summary = families.setdefault(
            o["family"],
            {
                "matrices": 0,
                "rows": 0,
                "changed_scale_rows": 0,
                "changed_quantized_elements": 0,
                "baseline_objective": 0.0,
                "refined_objective": 0.0,
            },
        )
        summary["matrices"] += 1
        summary["rows"] += o["rows"]
        summary["changed_scale_rows"] += o["changed_scale_rows"]
        summary["changed_quantized_elements"] += o["changed_quantized_elements"]
        summary["baseline_objective"] += baseline
        summary["refined_objective"] += actual
    if (
        not observations
        or report["passed"] is not True
        or type(report["rows"]) is not int
        or report["rows"] != rows
        or type(report["matrices"]) is not int
        or report["matrices"] != len(observations)
    ):
        raise ValueError("reconstruction readback summary disagrees with observations")
    return families


def regression_contract(path, expected_tests):
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in ET.parse(path).getroot().iter("testsuite"):
        for key in counts:
            counts[key] += int(suite.attrib.get(key, "0"))
    if counts["tests"] != expected_tests or any(
        counts[key] for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("incomplete or failed final regression suite")
    return counts


def verify(root, expected_tests):
    contract = load_contract(root)
    recomputed = audit(root)
    saved = json.loads((root / "verification.json").read_text())
    if saved != recomputed:
        raise ValueError("saved stage verification disagrees with fresh raw readback")
    expected = {}
    for family, item in contract["families"].items():
        path = root / "calibration" / f"{family}.json"
        report = json.loads(path.read_text())
        local = {o["weight"]: o for o in report["local_objectives"]}
        with ModelFile(item["source"]) as source:
            for name in report["policy"]["quantized_weights"]:
                expected[family, name] = {
                    "shape": list(source.tensor_info(name).shape),
                    "changed_scale_rows": 0,
                    "changed_quantized_elements": local[name][
                        "refinement_changed_elements"
                    ],
                    "baseline_objective": local[name]["scale_aware_block_objective"],
                    "refined_objective": local[name]["calibrated_block_objective"],
                    "calibration_report_sha256": file_sha256(path),
                    "stats_sha256": report["policy"]["calibration_stats_sha256"],
                }
    diagnostic_path = root / "objective-verification.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if (
        diagnostic["contract_sha256"] != file_sha256(root / "contract.json")
        or diagnostic["environment"] != contract["environment"]
        or diagnostic["diagnostic_script_sha256"]
        != file_sha256(Path(__file__).with_name("verify_refined_objectives.py"))
    ):
        raise ValueError("independent reconstruction diagnostic provenance changed")
    test_path = root / "test-results.xml"
    return {
        "schema_version": 1,
        "environment": contract["environment"],
        "audit_script_sha256": file_sha256(Path(__file__)),
        "stage_verification_sha256": file_sha256(root / "verification.json"),
        "objective_verification_sha256": file_sha256(diagnostic_path),
        "test_results_sha256": file_sha256(test_path),
        "final_tests": regression_contract(test_path, expected_tests),
        "reconstruction_readback": objective_contract(diagnostic, expected),
        "pre_fit_files_verified": recomputed["pre_fit_files_verified"],
        "prior_complete_files_unchanged": recomputed["prior_complete_files_unchanged"],
        "captured_source_files_verified": recomputed["captured_source_files_verified"],
        "cached_calibration_rows": recomputed["cached_calibration_rows"],
        "calibration_stage_passed": recomputed["calibration_stage_passed"],
        "strict_checkpoint_passed": False,
        "held_out_inference_executed": False,
        "readback_passed": True,
        "limitations": "Pre-registered audit, not fitting or performance. Rechecks raw calibration, complete files, independent reconstruction records and JUnit. A passed evidence readback does not waive the failed exact-token gates or certify direct kernels, held-out quality, 32K, or CUDA for this method.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    if args.expected_tests <= 0:
        parser.error("expected-tests must be positive")
    target = args.root / "final-verification.json"
    if target.exists():
        raise FileExistsError(target)
    with exclusive_gpu_workflow():
        report = verify(args.root, args.expected_tests)
    save(target, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
