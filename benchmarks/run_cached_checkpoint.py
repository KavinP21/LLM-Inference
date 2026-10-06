"""Fresh cached-calibration checkpoint, serialized and frozen before regression."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.cached_policy import DEFAULT_SEARCH
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_second_order_checkpoint import FAMILIES, checked_report, save

CALIBRATION = Path("benchmarks/cached-calibration-prompts.json")
CORPORA = [
    Path("benchmarks/quality-prompts.json"),
    Path("benchmarks/cached-regression-prompts.json"),
]
GUARDS = CORPORA + [Path("benchmarks/second-order-regression-prompts.json")]
TOOLS = [
    Path("benchmarks") / n
    for n in [
        "run_cached_checkpoint.py",
        "run_second_order_checkpoint.py",
        "gpu_guard.py",
        "run_int8_matrix.py",
        "validate_batch_numerics.py",
        "validate_batch_boundaries.py",
        "check_second_order_integrity.py",
        "probe_int8_numerics.py",
    ]
]


def calibrate_phase(run, root, artifact_dir):
    directory = root / "calibration"
    directory.mkdir()
    artifacts = {}
    for family, (stem, tokenizer) in FAMILIES.items():
        source = Path("models") / f"{stem}.engine"
        stats = artifact_dir / f"{family}-stats.npz"
        output = directory / f"{family}-calibration.json"
        report = checked_report(
            run,
            [
                sys.executable,
                "-m",
                "forge_llm.calibrate_cached",
                "--source",
                str(source),
                "--tokenizer",
                tokenizer,
                "--prompts",
                str(CALIBRATION),
                "--held-out",
                *map(str, GUARDS),
                "--stats",
                str(stats),
                "--output",
                str(output),
            ],
            output,
            ("policy", "calibration_passed"),
        )
        model = artifact_dir / f"{stem}-int8-cached.engine"
        rtn = artifact_dir / f"{stem}-int8-cached-rtn-control.engine"
        run(
            [
                sys.executable,
                "-m",
                "forge_llm.quantization",
                str(source),
                str(model),
                "--policy",
                str(output),
                "--calibration-stats",
                str(stats),
            ],
            check=True,
        )
        command = [
            sys.executable,
            "-m",
            "forge_llm.quantization",
            str(source),
            str(rtn),
        ]
        for name in report["policy"]["retained_fp16"]:
            command += ["--retain-fp16", name]
        run(command, check=True)
        artifacts[family] = {
            "source": str(source),
            "tokenizer": tokenizer,
            "model": str(model),
            "rtn_control": str(rtn),
            "stats": str(stats),
            "calibration_report": str(output),
            "model_provenance": model_provenance(model),
            "rtn_control_provenance": model_provenance(rtn),
            "policy_sha256": report["policy"]["policy_sha256"],
            "calibration_passed": report["policy"]["calibration_passed"],
            "calibration_stats_sha256": file_sha256(stats),
        }
    files = set()
    for item in artifacts.values():
        for key in ["source", "model", "rtn_control", "stats", "calibration_report"]:
            path = Path(item[key])
            files.add(path)
            if key in {"source", "model", "rtn_control"}:
                files.add(path.with_suffix(path.suffix + ".json"))
    save(
        root / "frozen.json",
        {
            "schema_version": 1,
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": source_provenance(),
            "artifacts": artifacts,
            "files_sha256": {str(p): file_sha256(p) for p in sorted(files)},
            "corpora_sha256": {str(p): file_sha256(p) for p in [CALIBRATION, *GUARDS]},
            "tools_sha256": {str(p): file_sha256(p) for p in TOOLS},
            "search_config": DEFAULT_SEARCH,
            "contract": "Both policies and whole files frozen before ANY held-out inference; 25% floor; all 32 cached positions; rowwise; no post-regression retuning.",
        },
    )


def load_frozen(root):
    frozen = json.loads((root / "frozen.json").read_text())
    if (
        frozen["environment"]["runtime_source_sha256"]
        != source_provenance()["runtime_source_sha256"]
    ):
        raise RuntimeError("runtime changed after calibration freeze")
    for field in ["files_sha256", "corpora_sha256", "tools_sha256"]:
        for name, checksum in frozen[field].items():
            if file_sha256(Path(name)) != checksum:
                raise RuntimeError(f"frozen {field} changed: {name}")
    return frozen


def quality_command(item, model, corpus, output, mode, stats):
    command = [
        sys.executable,
        "-m",
        "forge_llm.validate_int8",
        "--source",
        item["source"],
        "--model",
        model,
        "--tokenizer",
        item["tokenizer"],
        "--prompts",
        str(corpus),
        "--output",
        str(output),
        "--int8-mode",
        mode,
        "--decode-mode",
        "rowwise",
        "--cached-logits",
    ]
    return command + (["--calibration-stats", item["stats"]] if stats else [])


def validate_phase(run, root):
    frozen = load_frozen(root)
    directory = root / "regression"
    directory.mkdir()
    quality, controls, batching, resources = {}, {}, {}, {}
    for family, item in frozen["artifacts"].items():
        for mode in ["metal", "reconstruct"]:
            for index, corpus in enumerate(CORPORA):
                key = f"{family}-{mode}-corpus{index}"
                output = directory / f"{key}-quality.json"
                report = checked_report(
                    run,
                    quality_command(item, item["model"], corpus, output, mode, True),
                    output,
                    ("gates", "strict_checkpoint_passed"),
                )
                quality[key] = {"summary": report["summary"], "gates": report["gates"]}
            key = f"{family}-{mode}"
            batch_path = directory / f"{key}-batch.json"
            report = checked_report(
                run,
                [
                    sys.executable,
                    "benchmarks/validate_batch_numerics.py",
                    "--model",
                    item["model"],
                    "--quality",
                    str(directory / f"{key}-corpus0-quality.json"),
                    "--int8-mode",
                    mode,
                    "--output",
                    str(batch_path),
                ],
                batch_path,
                ("gates", "checkpoint_passed"),
            )
            batching[key] = report["gates"]
            boundary = directory / f"{key}-boundary.json"
            report = checked_report(
                run,
                [
                    sys.executable,
                    "benchmarks/validate_batch_boundaries.py",
                    "--model",
                    item["model"],
                    "--int8-mode",
                    mode,
                    "--output",
                    str(boundary),
                ],
                boundary,
                ("gates", "checkpoint_passed"),
            )
            resources[f"{key}-boundary"] = report["gates"]["checkpoint_passed"]
            long_path = directory / f"{key}-32k.json"
            run(
                [
                    sys.executable,
                    "-m",
                    "forge_llm.long_context",
                    "--model",
                    item["model"],
                    "--context-lengths",
                    "32768",
                    "--output-tokens",
                    "2",
                    "--kv-cache-mib",
                    "1024",
                    "--int8-mode",
                    mode,
                    "--decode-mode",
                    "rowwise",
                    "--output",
                    str(long_path),
                ],
                check=True,
            )
            long = json.loads(long_path.read_text())
            resources[f"{key}-32k"] = (
                bool(long["observations"])
                and long["all_reclaimed"]
                and all(
                    o["peak_pool_blocks"] == o["peak_materialized_blocks"] == 2048
                    and o["decode_iterations"] == 1
                    and o["prompt_tokens"] == 32766
                    and o["output_tokens"] == 2
                    for o in long["observations"]
                )
            )
        for index, corpus in enumerate(CORPORA):
            key = f"{family}-rtn-corpus{index}"
            output = directory / f"{key}-quality.json"
            report = checked_report(
                run,
                quality_command(
                    item, item["rtn_control"], corpus, output, "reconstruct", False
                ),
                output,
                ("gates", "strict_checkpoint_passed"),
            )
            controls[key] = {"summary": report["summary"], "gates": report["gates"]}
    checkpoint = {
        "quality": quality,
        "same_partition_rtn_controls": controls,
        "batching": batching,
        "resources": resources,
        "calibration_passed": all(
            a["calibration_passed"] for a in frozen["artifacts"].values()
        ),
        "strict_checkpoint_passed": all(
            q["gates"]["strict_checkpoint_passed"] for q in quality.values()
        )
        and all(b["checkpoint_passed"] for b in batching.values())
        and all(resources.values())
        and all(a["calibration_passed"] for a in frozen["artifacts"].values()),
        "scope": "Cached logits at all 32 positions; exact greedy unchanged; controls observational; no Transformers/CUDA certification.",
    }
    save(directory / "checkpoint.json", checkpoint)
    for family, item in frozen["artifacts"].items():
        output = directory / f"{family}-projection-numerics.json"
        checked_report(
            run,
            [
                sys.executable,
                "benchmarks/probe_int8_numerics.py",
                "--model",
                item["model"],
                "--output",
                str(output),
            ],
            output,
            ("gates", "passed"),
        )


def benchmark_phase(run, root):
    frozen = load_frozen(root)
    directory = root / "matrix"
    directory.mkdir()
    for family, item in frozen["artifacts"].items():
        run(
            [
                sys.executable,
                "benchmarks/run_int8_matrix.py",
                "--source",
                item["source"],
                "--model",
                item["model"],
                "--output-dir",
                str(directory / family),
                "--include-reconstruct",
                "--decode-mode",
                "rowwise",
                "--warmups",
                "5",
                "--repetitions",
                "3",
            ],
            check=True,
        )


def audit(root):
    frozen = load_frozen(root)
    from forge_llm import cached_evidence

    if len(list((root / "calibration").glob("*-calibration.json"))) != 2:
        raise ValueError("incomplete calibration evidence")

    counts = {
        "*-quality.json": 12,
        "*-batch.json": 4,
        "*-boundary.json": 4,
        "*-32k.json": 4,
    }
    for pattern, expected in counts.items():
        if len(list((root / "regression").glob(pattern))) != expected:
            raise ValueError(f"incomplete evidence: {pattern}")
    strict = all(a["calibration_passed"] for a in frozen["artifacts"].values())
    for item in frozen["artifacts"].values():
        calibration = json.loads(Path(item["calibration_report"]).read_text())
        cases = calibration["cases"]
        passed = (
            len(cases) == 32
            and all(c["fp16_tokens"] == c["candidate_tokens"] for c in cases)
            and all(c["private_cache_reclaimed"] for c in cases)
            and calibration["cache_reclaimed"]
            and calibration["search"]["summary"]["top1_changes"] == 0
            and min(
                m["cosine"] for c in cases for m in c["cached_teacher_forced_logits"]
            )
            >= 0.999
        )
        if (
            passed != item["calibration_passed"]
            or passed != calibration["policy"]["calibration_passed"]
        ):
            raise ValueError("saved calibration gates disagree with raw observations")
    for path in (root / "regression").glob("*-quality.json"):
        report = json.loads(path.read_text())
        if (
            report["configuration"]["teacher_forcing_path"] != "cached_paged_decode"
            or report["configuration"]["teacher_forced_positions"] != list(range(32))
            or report["summary"]["teacher_forced_logit_rows"] != 800
            or report["summary"]["total_cases"] != 25
            or report["summary"]["output_tokens_per_case"] != 32
            or len(report["cases"]) != 25
        ):
            raise ValueError(f"invalid cached quality evidence: {path}")
        gates = cached_evidence.quality(report)
        family, mode, corpus_label = path.name.removesuffix("-quality.json").split("-")
        item = frozen["artifacts"][family]
        corpus = CORPORA[int(corpus_label.removeprefix("corpus"))]
        expected = item[
            "rtn_control_provenance" if mode == "rtn" else "model_provenance"
        ]
        if (
            report["quantized_model"]["model_data_sha256"]
            != expected["model_data_sha256"]
            or [c["prompt"] for c in report["cases"]] != json.loads(corpus.read_text())
            or report["configuration"]["decode_mode"] != "rowwise"
            or report["configuration"]["int8_mode"]
            != ("reconstruct" if mode == "rtn" else mode)
            or (
                mode != "rtn"
                and report["derivation"]["calibration_stats_sha256"]
                != item["calibration_stats_sha256"]
            )
        ):
            raise ValueError(
                f"quality artifact/corpus/recipe identity mismatch: {path}"
            )
        if "-rtn-" not in path.name:
            strict &= gates["strict_checkpoint_passed"]
    for path in (root / "regression").glob("*-batch.json"):
        strict &= cached_evidence.batching(json.loads(path.read_text()))[
            "checkpoint_passed"
        ]
    for path in (root / "regression").glob("*-boundary.json"):
        strict &= cached_evidence.boundary(json.loads(path.read_text()))[
            "checkpoint_passed"
        ]
    checkpoint = json.loads((root / "regression/checkpoint.json").read_text())
    diagnostics = list((root / "regression").glob("*-projection-numerics.json"))
    if len(diagnostics) != 2:
        raise ValueError("incomplete projection diagnostics")
    for path in diagnostics:
        report = json.loads(path.read_text())
        observations = report["observations"]
        if not observations or not all(
            o["reconstruction_bitwise_equal"] and o["native_projection_bitwise_equal"]
            for o in observations
        ):
            raise ValueError("reconstruction/numerical contract failed")
    for path in (root / "regression").glob("*-32k.json"):
        gate = cached_evidence.long_context(json.loads(path.read_text()))
        if gate != checkpoint["resources"][path.name.removesuffix(".json")]:
            raise ValueError("resource checkpoint disagrees with raw observations")
        strict &= gate
    if strict != checkpoint["strict_checkpoint_passed"]:
        raise ValueError(
            "checkpoint does not agree with independent gate recomputation"
        )
    matrices = [
        p
        for p in (root / "matrix").glob("*/*-p*-c*.json")
        if not p.name.startswith("workload-")
    ]
    if len(matrices) != 32:
        raise ValueError("controlled benchmark matrix must contain 32 results")
    seen = set()
    for path in matrices:
        report = json.loads(path.read_text())
        config, stats = report["configuration"], report["engine_stats"]
        key = (
            path.parent.name,
            config["int8_mode"],
            tuple(config["prompt_token_lengths"]),
            config["concurrency"],
        )
        # FP16 and dequantize can both label int8_mode=dequantize; bind the model too.
        key += (report["model"]["model_data_sha256"],)
        if (
            key in seen
            or config["warmups"] != 5
            or config["repetitions"] != 3
            or config["output_length"] != 32
            or config["decode_mode"] != "rowwise"
            or len(report["requests"]) != config["concurrency"] * 3
            or stats["kv_cache"]["allocated_blocks"]
            or stats["kv_cache"]["reserved_blocks"]
            or stats["kv_device_bytes"]
        ):
            raise ValueError(f"invalid benchmark configuration/cache: {path}")
        seen.add(key)
    expected = set()
    for family, item in frozen["artifacts"].items():
        source_identity = json.loads(Path(item["calibration_report"]).read_text())[
            "source_model"
        ]["model_data_sha256"]
        for mode in ["auto", "dequantize", "metal", "reconstruct"]:
            identity = (
                source_identity
                if mode == "auto"
                else item["model_provenance"]["model_data_sha256"]
            )
            for length in [128, 1024]:
                for concurrency in [1, 8]:
                    expected.add(
                        (family, mode, (length,) * concurrency, concurrency, identity)
                    )
    if seen != expected:
        raise ValueError(
            "benchmark matrix identities/shapes do not match the frozen contract"
        )
    files = sorted(p for p in root.rglob("*.json") if p.name != "evidence-index.json")
    for path in files:
        report = json.loads(path.read_text())
        if (
            isinstance(report, dict)
            and report.get("environment")
            and report["environment"].get("runtime_source_sha256")
            != frozen["environment"]["runtime_source_sha256"]
        ):
            raise ValueError(f"mixed runtime provenance: {path}")
    return {
        "schema_version": 1,
        "files": {str(p.relative_to(root)): file_sha256(p) for p in files},
        "runtime_source_sha256": frozen["environment"]["runtime_source_sha256"],
        "strict_checkpoint_passed": strict,
        "benchmark_results": len(matrices),
        "cached_quality_rows": 12 * 800,
        "scope": "Checksums and frozen whole files, not unavailable profiler counters or CUDA measurements.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("models/cached-calibration")
    )
    parser.add_argument(
        "--phase",
        choices=["all", "calibrate", "validate", "benchmark", "audit"],
        default="all",
    )
    args = parser.parse_args()
    root = args.output_dir
    if args.phase in {"all", "calibrate"}:
        root.mkdir(parents=True)
    with exclusive_gpu_workflow() as run:
        if args.phase in {"all", "calibrate"}:
            calibrate_phase(run, root, args.artifact_dir)
        if args.phase in {"all", "validate"}:
            validate_phase(run, root)
        if args.phase in {"all", "benchmark"}:
            benchmark_phase(run, root)
        if args.phase in {"all", "audit"}:
            index = audit(root)
            save(root / "evidence-index.json", index)
            print(json.dumps(index, indent=2), flush=True)
            if not index["strict_checkpoint_passed"]:
                raise SystemExit(1)
        elif (
            args.phase == "validate"
            and not json.loads((root / "regression/checkpoint.json").read_text())[
                "strict_checkpoint_passed"
            ]
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
