"""Pre-registered scale-aware calibration gate; no held-out work on failure."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import canonical_sha256, validate_policy
from forge_llm.scale_aware import DEFAULT_CONFIG, DEFAULT_SEARCH, POLICY_ALGORITHM
from forge_llm.second_order import CalibrationStats, file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_second_order_checkpoint import FAMILIES, checked_report, save

CALIBRATION = Path("benchmarks/scale-calibration-prompts.json")
REGRESSION = Path("benchmarks/scale-regression-prompts.json")
GUARDS = [
    Path("benchmarks") / name
    for name in [
        "quality-prompts.json",
        "calibration-prompts.json",
        "second-order-calibration-prompts.json",
        "second-order-regression-prompts.json",
        "cached-calibration-prompts.json",
        "cached-regression-prompts.json",
        "scale-regression-prompts.json",
    ]
]
TOOLS = [
    Path(__file__),
    Path("benchmarks/gpu_guard.py"),
    Path("benchmarks/run_second_order_checkpoint.py"),
]


def capture_sources(root):
    """Recover the dirty runtime, not just a Git commit and an unrecoverable hash."""
    repo = Path(__file__).resolve().parents[1]
    paths = set()
    for directory in ("python", "include", "src"):
        paths.update(
            p
            for p in (repo / directory).rglob("*")
            if p.suffix in {".py", ".h", ".hpp", ".cpp", ".cu", ".cuh"}
        )
    paths.update(
        p.resolve() for p in [*TOOLS, CALIBRATION, *GUARDS, Path("pyproject.toml")]
    )
    checksums = {}
    archive = root / "sources.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            name = str(path.relative_to(repo))
            data = path.read_bytes()
            bundle.writestr(name, data)
            checksums[name] = hashlib.sha256(data).hexdigest()
    return {"sha256": file_sha256(archive), "files_sha256": checksums}


def check_sources(root, expected):
    archive = root / "sources.zip"
    if file_sha256(archive) != expected["sha256"]:
        raise ValueError("captured source archive changed")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if (
            len(members) != len(expected["files_sha256"])
            or len({m.filename for m in members}) != len(members)
            or set(bundle.namelist()) != set(expected["files_sha256"])
            or sum(m.file_size for m in members) > 50 << 20
        ):
            raise ValueError("invalid captured source archive")
        for name, sha in expected["files_sha256"].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != sha:
                raise ValueError("captured source member changed")


def prior_bindings(root):
    index = json.loads((root / "evidence-index.json").read_text())
    bindings = {str(root / name): sha for name, sha in index["files"].items()}
    integrity = json.loads((root / "integrity-before.json").read_text())
    bindings.update({name: item["sha256"] for name, item in integrity["files"].items()})
    for name in ("evidence-index.json", "verification.json", "test-results.xml"):
        path = root / name
        bindings[str(path)] = file_sha256(path)
    check_bindings(bindings)
    return bindings


def check_bindings(bindings):
    for name, sha in bindings.items():
        if file_sha256(Path(name)) != sha:
            raise ValueError(f"frozen complete file changed: {name}")


def prepare(root, artifact_dir, previous):
    # Verify everything before creating experiment state; never clobber a run.
    families, files = {}, {}
    for family, (stem, tokenizer) in FAMILIES.items():
        source = Path("models") / f"{stem}.engine"
        with ModelFile(source) as model:
            if model.quantization:
                raise ValueError("scale experiment requires official FP16 sources")
        families[family] = {
            "source": str(source),
            "tokenizer": tokenizer,
            "source_model": model_provenance(source),
            "stats": str(artifact_dir / f"{family}-stats.npz"),
        }
        files[str(source)] = file_sha256(source)
        manifest = source.with_suffix(source.suffix + ".json")
        files[str(manifest)] = file_sha256(manifest)
    files.update({str(p): file_sha256(p) for p in [CALIBRATION, *GUARDS, *TOOLS]})
    old = prior_bindings(previous)
    if root.exists() or artifact_dir.exists():
        raise FileExistsError("fresh result and artifact directories are required")
    root.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    captured = capture_sources(root)
    save(
        root / "contract.json",
        {
            "schema_version": 1,
            "registered_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "environment": source_provenance(),
            "families": families,
            "files_sha256": files,
            "prior_files_sha256": old,
            "prior_evidence": str(previous),
            "source_archive": captured,
            "configuration": {
                "weight_method": "scale-aware",
                "quantizer_config": DEFAULT_CONFIG,
                "search_config": DEFAULT_SEARCH,
                "calibration_prompts": str(CALIBRATION),
                "fresh_regression_prompts": str(REGRESSION),
                "output_tokens": 32,
                "positions": list(range(32)),
                "minimum_projection_fraction": 0.25,
                "cosine_gate": 0.999,
                "decode_mode": "rowwise",
            },
            "stop_rule": "Both families must pass exact full-calibration outputs, all cached decisions, cosine 0.999 and cleanup. Failure saves complete calibration evidence and stops BEFORE export, held-out inference or performance claims. No retuning within this run.",
            "downstream_status": "Held-out validation, official packed exports and performance are a separate gated follow-up, never implied by calibration success.",
        },
    )


def load_contract(root):
    contract = json.loads((root / "contract.json").read_text())
    if (
        source_provenance()["runtime_source_sha256"]
        != contract["environment"]["runtime_source_sha256"]
    ):
        raise ValueError("runtime drift after pre-registration")
    check_bindings(contract["files_sha256"])
    check_bindings(contract["prior_files_sha256"])
    check_sources(root, contract["source_archive"])
    if (
        contract["configuration"]["quantizer_config"] != DEFAULT_CONFIG
        or contract["configuration"]["search_config"] != DEFAULT_SEARCH
    ):
        raise ValueError("pre-registered method/search configuration differs")
    return contract


def calibration_gates(report):
    """Recompute acceptance from cases, not calibration_passed flags."""
    cases = report["cases"]
    if len(cases) != 32 or report["configuration"]["positions"] != list(range(32)):
        raise ValueError("incomplete all-position calibration evidence")
    exact, changes, cosines = 0, 0, []
    for case in cases:
        reference, actual, teacher = [
            case[k]
            for k in ("fp16_tokens", "candidate_tokens", "teacher_forced_tokens")
        ]
        rows = case["cached_teacher_forced_logits"]
        if any(len(t) != 32 for t in (reference, actual, teacher)) or [
            r["generated_position"] for r in rows
        ] != list(range(32)):
            raise ValueError("truncated calibration continuation/positions")
        equal = reference == actual
        if (
            type(case["exact_greedy_parity"]) is not bool
            or type(case["private_cache_reclaimed"]) is not bool
            or case["exact_greedy_parity"] != equal
        ):
            raise ValueError("calibration case flags disagree with raw tokens")
        for p, row in enumerate(rows):
            if (
                type(row["argmax_equal"]) is not bool
                or row["argmax_equal"] != (reference[p] == teacher[p])
                or not math.isfinite(row["cosine"])
                or not -1.000001 <= row["cosine"] <= 1.000001
            ):
                raise ValueError("cached decision/numerical observations disagree")
        exact += equal
        changes += sum(not r["argmax_equal"] for r in rows)
        cosines.extend(r["cosine"] for r in rows)
    local = report["local_objectives"]
    if not local or len({o["weight"] for o in local}) != len(local):
        raise ValueError("incomplete local reconstruction evidence")
    for o in local:
        for name in (
            "rtn_block_objective",
            "fixed_scale_block_objective",
            "calibrated_block_objective",
        ):
            if not math.isfinite(o[name]) or o[name] < 0:
                raise ValueError("invalid local reconstruction objective")
        if (
            type(o["no_worse_fixed_scale_row_objective"]) is not bool
            or not o["no_worse_fixed_scale_row_objective"]
            or o["calibrated_block_objective"] > o["fixed_scale_block_objective"]
        ):
            raise ValueError(
                "new method worsened the baseline reconstruction objective"
            )
    policy, summary = report["policy"], report["summary"]
    if any(
        type(value) is not bool
        for value in [
            report["cache_reclaimed"],
            policy["calibration_passed"],
            summary["calibration_passed"],
        ]
    ):
        raise ValueError("invalid calibration acceptance flag types")
    inventory = {o["weight"]: o["fp16_bytes"] for o in local}
    if any(type(v) is not int or v <= 0 for v in inventory.values()):
        raise ValueError("invalid eligible projection bytes")
    validate_policy(policy, report["source_model"]["model_data_sha256"], set(inventory))
    fraction = sum(inventory[n] for n in policy["quantized_weights"]) / sum(
        inventory.values()
    )
    minimum = min(cosines)
    if (
        summary["exact_greedy_cases"] != exact
        or summary["total_cases"] != 32
        or summary["minimum_logit_cosine"] != minimum
        or summary["cached_top1_changes"] != changes
        or report["search"]["summary"]["top1_changes"] != changes
        or report["search"]["summary"]["observations"] != 1024
        or policy["quantized_projection_fraction"] != fraction
        or policy["algorithm"] != POLICY_ALGORITHM
    ):
        raise ValueError("calibration summary/policy disagrees with raw observations")
    gates = {
        "exact_free_continuations": exact == 32,
        "exact_cached_decisions": changes == 0,
        "numerical": minimum >= 0.999
        and report["search"]["summary"]["minimum_logit_cosine"] >= 0.999,
        "cache_reclaimed": report["cache_reclaimed"]
        and all(c["private_cache_reclaimed"] for c in cases),
        "minimum_projection_fraction": fraction >= 0.25,
        "local_reconstruction_bound": True,
    }
    gates["passed"] = all(gates.values())
    if (
        policy["calibration_passed"] != gates["passed"]
        or summary["calibration_passed"] != gates["passed"]
    ):
        raise ValueError("calibration acceptance flags disagree with raw gates")
    return gates


def calibrate_phase(run, root):
    contract = load_contract(root)
    directory = root / "calibration"
    directory.mkdir(exist_ok=True)
    reports, files, durations = {}, {}, {}
    for family, item in contract["families"].items():
        output, stats = directory / f"{family}.json", Path(item["stats"])
        if output.exists() or stats.exists():
            raise FileExistsError(
                "partial/completed calibration preserved; use audit or a new run, never overwrite"
            )
        started = time.perf_counter()
        report = checked_report(
            run,
            [
                sys.executable,
                "-m",
                "forge_llm.calibrate_cached",
                "--source",
                item["source"],
                "--tokenizer",
                item["tokenizer"],
                "--prompts",
                str(CALIBRATION),
                "--held-out",
                *map(str, GUARDS),
                "--stats",
                str(stats),
                "--output",
                str(output),
                "--weight-method",
                "scale-aware",
            ],
            output,
            ("policy", "calibration_passed"),
        )
        durations[family] = time.perf_counter() - started
        reports[family] = calibration_gates(report)
        if (
            report["environment"]["runtime_source_sha256"]
            != contract["environment"]["runtime_source_sha256"]
        ):
            raise ValueError("calibration runtime differs from pre-registration")
        files.update({str(p): file_sha256(p) for p in (output, stats)})
    load_contract(root)  # Inputs and all previous evidence must STILL be unchanged.
    eligible = all(r["passed"] for r in reports.values())
    save(
        root / "calibration-checkpoint.json",
        {
            "schema_version": 1,
            "environment": contract["environment"],
            "contract_sha256": file_sha256(root / "contract.json"),
            "files_sha256": files,
            "gates": reports,
            "calibration_stage_passed": eligible,
            "state": "eligible_for_held_out_validation"
            if eligible
            else "rejected_at_calibration",
            "held_out_inference_executed": False,
            "official_packed_exports_created": False,
            "performance_matrix_executed": False,
            "strict_checkpoint_passed": False,
            "offline_calibration_seconds": durations,
            "scope": "This stage ends at calibration. Local loss improvement and eligibility are not a complete quality/performance certification.",
        },
    )


def audit(root):
    contract = load_contract(root)
    checkpoint = json.loads((root / "calibration-checkpoint.json").read_text())
    if file_sha256(root / "contract.json") != checkpoint["contract_sha256"]:
        raise ValueError("pre-registration changed after calibration")
    check_bindings(checkpoint["files_sha256"])
    gates, rows, selected_gains = {}, 0, {}
    for family, item in contract["families"].items():
        path = root / "calibration" / f"{family}.json"
        report = json.loads(path.read_text())
        gates[family] = calibration_gates(report)
        with ModelFile(item["source"]) as source:
            stats = CalibrationStats(
                Path(item["stats"]),
                source,
                expected_sha=report["policy"]["calibration_stats_sha256"],
            )
            names = {n for n in source.tensors if n.endswith("_proj.weight")}
            if (
                set(
                    report["policy"]["quantized_weights"]
                    + report["policy"]["retained_fp16"]
                )
                != names
            ):
                raise ValueError("official projection inventory mismatch")
            if (
                stats.config != contract["configuration"]["quantizer_config"]
                or report["policy"]["search_config"]
                != contract["configuration"]["search_config"]
            ):
                raise ValueError("policy/statistics differ from the registered method")
        if (
            report["environment"]["runtime_source_sha256"]
            != contract["environment"]["runtime_source_sha256"]
            or report["source_model"] != item["source_model"]
            or report["configuration"]["calibration_prompts"]
            != json.loads(CALIBRATION.read_text())
            or report["policy"]["calibration_corpus_sha256"]
            != canonical_sha256(json.loads(CALIBRATION.read_text()))
        ):
            raise ValueError("source/corpus/runtime provenance mismatch")
        with ModelFile(item["source"]) as source:
            names = {n for n in source.tensors if n.endswith("_proj.weight")}
            if {o["weight"]: o["fp16_bytes"] for o in report["local_objectives"]} != {
                n: source.tensor_info(n).nbytes for n in names
            }:
                raise ValueError("eligible-byte inventory differs from source tensors")
        if [c["prompt"] for c in report["cases"]] != json.loads(
            CALIBRATION.read_text()
        ) or report["configuration"]["tokenizer"] != item["tokenizer"]:
            raise ValueError("case/tokenizer provenance mismatch")
        selected = set(report["policy"]["quantized_weights"])
        objectives = [o for o in report["local_objectives"] if o["weight"] in selected]
        selected_gains[family] = {
            "fixed_scale_objective": sum(
                o["fixed_scale_block_objective"] for o in objectives
            ),
            "scale_aware_objective": sum(
                o["calibrated_block_objective"] for o in objectives
            ),
            "changed_scale_rows": sum(o["changed_scale_rows"] for o in objectives),
        }
        rows += 1024
    passed = all(r["passed"] for r in gates.values())
    if gates != checkpoint["gates"] or passed != checkpoint["calibration_stage_passed"]:
        raise ValueError("checkpoint gates disagree with raw calibration")
    if any(
        checkpoint[k]
        for k in (
            "held_out_inference_executed",
            "official_packed_exports_created",
            "performance_matrix_executed",
            "strict_checkpoint_passed",
        )
    ):
        raise ValueError("calibration-only evidence cannot certify downstream stages")
    for name in ("regression", "matrix", "held-out"):
        if (root / name).exists():
            raise ValueError(
                "downstream results do not belong in the stopped calibration run"
            )
    for directory in {
        Path(item["stats"]).parent for item in contract["families"].values()
    }:
        if list(directory.glob("*.engine")):
            raise ValueError(
                "calibration-only stage contains an unexpected official export"
            )
    return {
        "schema_version": 1,
        "environment": contract["environment"],
        "contract_sha256": file_sha256(root / "contract.json"),
        "calibration_checkpoint_sha256": file_sha256(
            root / "calibration-checkpoint.json"
        ),
        "calibration_files_verified": len(checkpoint["files_sha256"]),
        "pre_fit_files_verified": len(contract["files_sha256"]),
        "prior_complete_files_unchanged": len(contract["prior_files_sha256"]),
        "captured_source_files_verified": len(
            contract["source_archive"]["files_sha256"]
        ),
        "source_archive_sha256": contract["source_archive"]["sha256"],
        "cached_calibration_rows": rows,
        "gates": gates,
        "selected_local_objectives": selected_gains,
        "calibration_stage_passed": passed,
        "strict_checkpoint_passed": False,
        "held_out_inference_executed": False,
        "readback_passed": True,
        "limitations": "Whole-file hashes and recorded cases validate this calibration stage only. Per-row bound assertions are backed by independent operator tests, not a language-quality guarantee. No held-out/performance/RTX/counter evidence is supplied.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/scale-aware"))
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("benchmarks/results/cached-calibration-m3-max-2026-10-01"),
    )
    parser.add_argument(
        "--phase", choices=("all", "prepare", "calibrate", "audit"), default="all"
    )
    args = parser.parse_args()
    with exclusive_gpu_workflow() as run:
        if args.phase in ("all", "prepare"):
            prepare(args.output_dir, args.artifact_dir, args.previous)
        if args.phase in ("all", "calibrate"):
            calibrate_phase(run, args.output_dir)
        if args.phase in ("all", "audit"):
            report = audit(args.output_dir)
            save(args.output_dir / "verification.json", report)
            print(json.dumps(report, indent=2), flush=True)
            if not report["calibration_stage_passed"]:
                raise SystemExit(1)
        elif (
            args.phase == "calibrate"
            and not json.loads(
                (args.output_dir / "calibration-checkpoint.json").read_text()
            )["calibration_stage_passed"]
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
