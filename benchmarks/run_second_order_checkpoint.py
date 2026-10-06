"""Serialized, frozen-before-regression second-order INT8 checkpoint.

Saved quality failures are evidence, never success. Exceptions, absent reports,
source drift, and inconsistent exit codes abort the workflow. Calibration of
BOTH families finishes before any held-out output is requested.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow

FAMILIES = {
    "qwen": ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    "gemma": ("gemma-3-1b-it", "google/gemma-3-1b-it"),
}
CORPORA = [
    Path("benchmarks/quality-prompts.json"),
    Path("benchmarks/second-order-regression-prompts.json"),
]


def checked_report(run, command, output: Path, gate_path: tuple[str, ...]):
    result = run(command, check=False)
    if result.returncode not in {0, 1} or not output.is_file():
        raise RuntimeError(f"execution failed without valid evidence: {output}")
    report = json.loads(output.read_text())
    value = report
    for key in gate_path:
        value = value[key]
    if not isinstance(value, bool) or value != (result.returncode == 0):
        raise RuntimeError(f"gate and exit status disagree: {output}")
    return report


def save(path: Path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def calibrate_phase(run, root: Path, artifact_dir: Path):
    directory = root / "calibration"
    directory.mkdir()
    artifacts = {}
    for family, (stem, tokenizer) in FAMILIES.items():
        source = Path("models") / f"{stem}.engine"
        stats = artifact_dir / f"{family}-stats.npz"
        output = directory / f"{family}-calibration.json"
        print(f"Calibration only: {family}", flush=True)
        report = checked_report(
            run,
            [
                sys.executable,
                "-m",
                "forge_llm.calibrate_second_order",
                "--source",
                str(source),
                "--tokenizer",
                tokenizer,
                "--stats",
                str(stats),
                "--output",
                str(output),
            ],
            output,
            ("policy", "calibration_passed"),
        )
        policy = report["policy"]
        model = artifact_dir / f"{stem}-int8-second-order.engine"
        rtn = artifact_dir / f"{stem}-int8-second-order-rtn-control.engine"
        # CPU-only export; isolated from timed inference and still lease-owned.
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
        for name in policy["retained_fp16"]:
            command += ["--retain-fp16", name]
        run(command, check=True)
        artifacts[family] = {
            "source": str(source),
            "tokenizer": tokenizer,
            "model": str(model),
            "rtn_control": str(rtn),
            "stats": str(stats),
            "calibration_report": str(output),
            "calibration_report_sha256": file_sha256(output),
            "calibration_stats_sha256": file_sha256(stats),
            "model_provenance": model_provenance(model),
            "rtn_control_provenance": model_provenance(rtn),
            "policy_sha256": policy["policy_sha256"],
            "calibration_passed": policy["calibration_passed"],
        }
    frozen = {
        "schema_version": 1,
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": source_provenance(),
        "artifacts": artifacts,
        "corpora_sha256": {
            str(p): file_sha256(p)
            for p in [
                Path("benchmarks/second-order-calibration-prompts.json"),
                *CORPORA,
            ]
        },
        "driver_sha256": file_sha256(Path(__file__)),
        "contract": "Both policies and stats frozen before any regression; no post-regression tuning; FP16 defaults unchanged.",
    }
    save(root / "frozen.json", frozen)


def load_frozen(root: Path):
    frozen = json.loads((root / "frozen.json").read_text())
    if (
        frozen["environment"]["runtime_source_sha256"]
        != source_provenance()["runtime_source_sha256"]
    ):
        raise RuntimeError("runtime changed after calibration freeze")
    if frozen["driver_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("checkpoint driver changed after calibration freeze")
    for path, checksum in frozen["corpora_sha256"].items():
        if file_sha256(Path(path)) != checksum:
            raise RuntimeError("corpus changed after freeze")
    for item in frozen["artifacts"].values():
        for field in ["calibration_report", "stats"]:
            checksum = item[
                "calibration_report_sha256"
                if field == "calibration_report"
                else "calibration_stats_sha256"
            ]
            if file_sha256(Path(item[field])) != checksum:
                raise RuntimeError("policy or covariance archive changed after freeze")
        for field in ["model", "rtn_control"]:
            expected = item[
                "model_provenance" if field == "model" else "rtn_control_provenance"
            ]
            if (
                model_provenance(Path(item[field]))["model_data_sha256"]
                != expected["model_data_sha256"]
            ):
                raise RuntimeError("frozen model artifact changed")
    return frozen


def validate_phase(run, root: Path):
    frozen = load_frozen(root)
    directory = root / "regression"
    directory.mkdir()
    quality, resources, batching, controls = {}, {}, {}, {}
    for family, item in frozen["artifacts"].items():
        for mode in ["metal", "reconstruct"]:
            for corpus_index, corpus in enumerate(CORPORA):
                key = f"{family}-{mode}-corpus{corpus_index}"
                output = directory / f"{key}-quality.json"
                report = checked_report(
                    run,
                    [
                        sys.executable,
                        "-m",
                        "forge_llm.validate_int8",
                        "--source",
                        item["source"],
                        "--model",
                        item["model"],
                        "--tokenizer",
                        item["tokenizer"],
                        "--calibration-stats",
                        item["stats"],
                        "--decode-mode",
                        "rowwise",
                        "--int8-mode",
                        mode,
                        "--prompts",
                        str(corpus),
                        "--output",
                        str(output),
                    ],
                    output,
                    ("gates", "strict_checkpoint_passed"),
                )
                quality[key] = {"summary": report["summary"], "gates": report["gates"]}
            batch_path = directory / f"{family}-{mode}-batch.json"
            report = checked_report(
                run,
                [
                    sys.executable,
                    "benchmarks/validate_batch_numerics.py",
                    "--model",
                    item["model"],
                    "--quality",
                    str(directory / f"{family}-{mode}-corpus0-quality.json"),
                    "--int8-mode",
                    mode,
                    "--output",
                    str(batch_path),
                ],
                batch_path,
                ("gates", "checkpoint_passed"),
            )
            batching[f"{family}-{mode}"] = report["gates"]
            boundary = directory / f"{family}-{mode}-boundary.json"
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
            resources[f"{family}-{mode}-boundary"] = report["gates"][
                "checkpoint_passed"
            ]
            long_path = directory / f"{family}-{mode}-32k.json"
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
            resources[f"{family}-{mode}-32k"] = long["all_reclaimed"] and all(
                o["peak_pool_blocks"] == o["peak_materialized_blocks"] == 2048
                and o["decode_iterations"] == 1
                and o["prompt_tokens"] == 32766
                and o["output_tokens"] == 2
                for o in long["observations"]
            )
        for corpus_index, corpus in enumerate(CORPORA):
            key = f"{family}-rtn-corpus{corpus_index}"
            output = directory / f"{key}-quality.json"
            report = checked_report(
                run,
                [
                    sys.executable,
                    "-m",
                    "forge_llm.validate_int8",
                    "--source",
                    item["source"],
                    "--model",
                    item["rtn_control"],
                    "--tokenizer",
                    item["tokenizer"],
                    "--decode-mode",
                    "rowwise",
                    "--int8-mode",
                    "reconstruct",
                    "--prompts",
                    str(corpus),
                    "--output",
                    str(output),
                ],
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
        "scope": "Two frozen calibrated artifacts; rowwise consistency is separate from quantization quality. RTN control quality is an A/B observation, not an acceptance requirement.",
    }
    save(directory / "checkpoint.json", checkpoint)
    return checkpoint


def benchmark_phase(run, root: Path):
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


def audit(root: Path):
    frozen = load_frozen(root)
    checkpoint = json.loads((root / "regression/checkpoint.json").read_text())
    for directory, pattern, expected in [
        ("calibration", "*-calibration.json", 2),
        ("regression", "*-quality.json", 12),
        ("regression", "*-batch.json", 4),
        ("regression", "*-boundary.json", 4),
        ("regression", "*-32k.json", 4),
    ]:
        if len(list((root / directory).glob(pattern))) != expected:
            raise ValueError(f"incomplete evidence set: {directory}/{pattern}")
    matrices = [
        p
        for p in (root / "matrix").glob("*/*-p*-c*.json")
        if not p.name.startswith("workload-")
    ]
    if len(matrices) != 32:
        raise ValueError("controlled benchmark matrix must contain 32 results")
    for path in matrices:
        report = json.loads(path.read_text())
        config, stats = report["configuration"], report["engine_stats"]
        if (
            config["warmups"] != 5
            or config["repetitions"] != 3
            or config["output_length"] != 32
            or config["decode_mode"] != "rowwise"
            or len(report["requests"]) != config["concurrency"] * 3
            or stats["kv_cache"]["allocated_blocks"] != 0
            or stats["kv_cache"]["reserved_blocks"] != 0
            or stats["kv_device_bytes"] != 0
        ):
            raise ValueError(f"invalid controlled workload or leaked cache: {path}")
    files = sorted(root.rglob("*.json"))
    reports = [p for p in files if not p.name.startswith("workload-")]
    for path in reports:
        report = json.loads(path.read_text())
        environment = report.get("environment")
        if (
            environment
            and environment.get("runtime_source_sha256")
            != frozen["environment"]["runtime_source_sha256"]
        ):
            raise ValueError(f"mixed runtime provenance: {path}")
    return {
        "schema_version": 1,
        "runtime_source_sha256": frozen["environment"]["runtime_source_sha256"],
        "files": {str(p.relative_to(root)): file_sha256(p) for p in files},
        "calibration_archives": {
            a["stats"]: a["calibration_stats_sha256"]
            for a in frozen["artifacts"].values()
        },
        "strict_checkpoint_passed": checkpoint["strict_checkpoint_passed"],
        "regression_quality_reports": len(
            list((root / "regression").glob("*-quality.json"))
        ),
        "benchmark_results": len(
            [
                p
                for p in (root / "matrix").glob("*/*-p*-c*.json")
                if not p.name.startswith("workload-")
            ]
        ),
        "note": "Checksums bind raw evidence; no absent profiler counters or unexecuted RTX results are certified.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("models/second-order")
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
        elif args.phase == "validate":
            if not json.loads((root / "regression/checkpoint.json").read_text())[
                "strict_checkpoint_passed"
            ]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
