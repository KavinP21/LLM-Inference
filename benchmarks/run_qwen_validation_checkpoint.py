"""Sealed Qwen-only downstream validation; never fit on held-out observations.

The coordinate-refined two-family run remains stopped. This separate contract
authorizes export of its calibration-eligible Qwen policy, not Gemma promotion.
Only native FP16-rounded reconstruction/rowwise decode is acceptance-bearing.
Direct Metal reductions are a separately reported diagnostic contract.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

from forge_llm import cached_evidence
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import validate_corpora
from forge_llm.refined import DEFAULT_CONFIG, POLICY_ALGORITHM
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_refined_checkpoint import (
    calibration_gates,
    check_bindings,
    check_sources,
    prior_bindings,
)
from run_second_order_checkpoint import checked_report, save

CORPUS = Path("benchmarks/refined-regression-prompts.json")
CONFIG = {
    "family": "qwen",
    "output_tokens": 32,
    "positions": list(range(32)),
    "cosine_gate": 0.999,
    "minimum_projection_fraction": 0.25,
    "decode_mode": "rowwise",
    "acceptance_mode": "reconstruct",
    "diagnostic_mode": "metal",
    "batch_concurrencies": [2, 8, 16],
    "boundary_prompt_lengths": [15, 16, 17, 31, 32, 33, 128, 512, 513],
    "long_context": 32768,
    "long_output_tokens": 2,
    "prompt_lengths": [128, 1024],
    "benchmark_concurrencies": [1, 8],
    "benchmark_modes": ["auto", "reconstruct", "dequantize"],
    "warmups": 5,
    "repetitions": 3,
}
TOOLS = [
    Path(__file__),
    Path(".github/workflows/qwen-validation.yml"),
    *[
        Path("benchmarks") / name
        for name in [
            "gpu_guard.py",
            "run_second_order_checkpoint.py",
            "run_refined_checkpoint.py",
            "audit_refined_checkpoint.py",
            "validate_batch_numerics.py",
            "validate_batch_boundaries.py",
            "verify_native_cached_bytes.py",
        ]
    ],
]


def read(path):
    return json.loads(Path(path).read_text())


def capture_sources(root):
    repo = Path(__file__).resolve().parents[1]
    paths = {p.resolve() for p in [*TOOLS, CORPUS, Path("pyproject.toml")]}
    for directory in ("python", "include", "src", "tests"):
        paths.update(
            p
            for p in (repo / directory).rglob("*")
            if p.suffix in {".py", ".h", ".hpp", ".cpp", ".cu", ".cuh"}
        )
    checksums = {}
    with zipfile.ZipFile(root / "sources.zip", "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            name, data = str(path.relative_to(repo)), path.read_bytes()
            bundle.writestr(name, data)
            checksums[name] = hashlib.sha256(data).hexdigest()
    return {
        "sha256": file_sha256(root / "sources.zip"),
        "files_sha256": checksums,
    }


def prepare(root, artifact_dir, previous):
    from audit_refined_checkpoint import verify

    if root.exists() or artifact_dir.exists():
        raise FileExistsError("fresh result and artifact directories are required")
    old_final = read(previous / "final-verification.json")
    if verify(previous, old_final["final_tests"]["tests"]) != old_final:
        raise ValueError("prior refined final readback changed")
    old_contract = read(previous / "contract.json")
    calibration_path = previous / "calibration/qwen.json"
    calibration = read(calibration_path)
    policy = calibration["policy"]
    if not calibration_gates(calibration)["passed"]:
        raise ValueError("Qwen policy is not calibration eligible")
    if policy["algorithm"] != POLICY_ALGORITHM:
        raise ValueError("wrong sealed policy method")
    # Only an overlap guard: do not infer semantic independence from it.
    validate_corpora(read(CORPUS), [c["prompt"] for c in calibration["cases"]])
    if len(read(CORPUS)) != 25:
        raise ValueError("held-out contract requires exactly 25 prompts")
    item = old_contract["families"]["qwen"]
    files = {str(p): file_sha256(p) for p in [CORPUS, *TOOLS]}
    prior = prior_bindings(previous)
    # Includes the preserved prose clarification, not a rewrite of old evidence.
    root.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    save(
        root / "contract.json",
        {
            "schema_version": 1,
            "registered_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "environment": source_provenance(),
            "configuration": CONFIG,
            "corpus": str(CORPUS),
            "corpus_prompts": read(CORPUS),
            "previous": str(previous),
            "files_sha256": files,
            "prior_files_sha256": prior,
            "source_archive": capture_sources(root),
            "item": {
                **item,
                "calibration_report": str(calibration_path),
                "policy": policy,
                "model": str(artifact_dir / "qwen2.5-0.5b-int8-refined.engine"),
            },
            "stop_rule": "Complete both registered quality reports and the native/composed byte control. Native quality or byte failure stops BEFORE batching, 32K or timing. Resource failure stops BEFORE timing. No fitting, policy changes, corpus replacements or gate relaxation; interrupted stages are preserved, never overwritten.",
            "scope": "Qwen only, native reconstruction with rowwise projections. Direct Metal quality is diagnostic and cannot certify this execution mode. References are the identical Forge FP16 artifact, not a new Transformers or general-accuracy certification. Gemma remains FP16. No app/default change or automatic promotion.",
        },
    )


def load_contract(root):
    contract = read(root / "contract.json")
    if contract["configuration"] != CONFIG:
        raise ValueError("registered downstream configuration changed")
    if (
        source_provenance()["runtime_source_sha256"]
        != contract["environment"]["runtime_source_sha256"]
    ):
        raise ValueError("runtime drift after pre-registration")
    check_bindings(contract["files_sha256"])
    check_bindings(contract["prior_files_sha256"])
    check_sources(root, contract["source_archive"])
    if read(CORPUS) != contract["corpus_prompts"] or contract["corpus"] != str(CORPUS):
        raise ValueError("registered corpus changed")
    item = contract["item"]
    calibration = read(item["calibration_report"])
    if (
        calibration["policy"] != item["policy"]
        or not calibration_gates(calibration)["passed"]
    ):
        raise ValueError("sealed calibration policy changed or failed")
    return contract


def export_phase(run, root):
    contract = load_contract(root)
    if (root / "frozen.json").exists():
        raise FileExistsError("completed export is immutable")
    item = contract["item"]
    model = Path(item["model"])
    if model.exists() or model.with_suffix(".engine.json").exists():
        raise FileExistsError("partial/completed export preserved; never overwrite")
    run(
        [
            sys.executable,
            "-m",
            "forge_llm.quantization",
            item["source"],
            item["model"],
            "--policy",
            item["calibration_report"],
            "--calibration-stats",
            item["stats"],
        ],
        check=True,
    )
    load_contract(root)
    manifest = model.with_suffix(".engine.json")
    save(
        root / "frozen.json",
        {
            "contract_sha256": file_sha256(root / "contract.json"),
            "files_sha256": {str(p): file_sha256(p) for p in (model, manifest)},
            "model_provenance": model_provenance(model),
            "exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    load_frozen(root)


def load_frozen(root):
    contract = load_contract(root)
    frozen = read(root / "frozen.json")
    if frozen["contract_sha256"] != file_sha256(root / "contract.json"):
        raise ValueError("contract changed after export")
    item = contract["item"]
    model = Path(item["model"])
    if set(frozen["files_sha256"]) != {
        str(model),
        str(model.with_suffix(".engine.json")),
    }:
        raise ValueError("incomplete frozen artifact bindings")
    check_bindings(frozen["files_sha256"])
    manifest = read(model.with_suffix(".engine.json"))
    if (
        manifest["precision_policy"] != item["policy"]
        or manifest["calibration_stats_sha256"]
        != item["policy"]["calibration_stats_sha256"]
        or manifest["quantization_method"] != "coordinate_refined_scale_aware_v1"
        or model_provenance(model) != frozen["model_provenance"]
        or manifest["source_data_sha256"] != item["source_model"]["model_data_sha256"]
        or manifest["quantized_projection_fraction"] < 0.25
        or frozen["model_provenance"]["model_file_bytes"]
        >= item["source_model"]["model_file_bytes"]
    ):
        raise ValueError("export recipe/model identity differs from sealed policy")
    with ModelFile(model) as packed:
        if set(packed.quantization) != set(item["policy"]["quantized_weights"]):
            raise ValueError("exported precision partition changed")
    return contract, frozen


def quality_readback(report, contract, frozen, mode):
    """Raw all-position coverage, finite metrics, exact token flags and provenance."""
    item = contract["item"]
    config = report["configuration"]
    if (
        config["teacher_forcing_path"] != "cached_paged_decode"
        or config["teacher_forced_positions"] != list(range(32))
        or config["cosine_gate"] != 0.999
        or config["int8_mode"] != mode
        or config["decode_mode"] != "rowwise"
        or config["tokenizer"] != item["tokenizer"]
        or config["seed"] != 17
        or config["prompts_sha256"]
        != hashlib.sha256(
            json.dumps(contract["corpus_prompts"], ensure_ascii=False).encode()
        ).hexdigest()
        or [c["prompt"] for c in report["cases"]] != contract["corpus_prompts"]
        or len(report["cases"]) != 25
        or report["summary"]["teacher_forced_logit_rows"] != 800
        or report["environment"]["runtime_source_sha256"]
        != contract["environment"]["runtime_source_sha256"]
        or report["source_model"] != item["source_model"]
        or report["quantized_model"] != frozen["model_provenance"]
    ):
        raise ValueError("held-out coverage/source/corpus/runtime mismatch")
    derivation = report["derivation"]
    if (
        derivation["method"] != "coordinate_refined_scale_aware_v1"
        or derivation["calibration_stats_sha256"]
        != item["policy"]["calibration_stats_sha256"]
        or derivation["quantizer_config"] != DEFAULT_CONFIG
        or derivation["all_packed_values_and_scales_recomputed"] is not True
        or derivation["retained_source_tensors_identical"] is not True
    ):
        raise ValueError("exact export derivation is missing or changed")
    layers = report["layers"]
    if any(type(v) is not bool for v in report["gates"].values()):
        raise ValueError("invalid saved gate flag types")
    if len(layers) != len(item["policy"]["quantized_weights"]) or {
        o["weight"] for o in layers
    } != set(item["policy"]["quantized_weights"]):
        raise ValueError("incomplete projection comparisons")
    for layer in layers:
        if (
            not math.isfinite(layer["cosine_vs_fp16"])
            or not -1.000001 <= layer["cosine_vs_fp16"] <= 1.000001
        ):
            raise ValueError("invalid projection cosine")
        for name in ("relative_l2_error_vs_fp16", "max_abs_error_vs_dequantized"):
            if not math.isfinite(layer[name]) or layer[name] < 0:
                raise ValueError("invalid projection error")
    for case in report["cases"]:
        for field in (
            "exact_greedy_parity",
            "metal_dequantize_greedy_parity",
            "private_cache_reclaimed",
        ):
            if type(case[field]) is not bool:
                raise ValueError("invalid quality boolean")
        for name in ("input_ids", "fp16_tokens", "int8_tokens", "dequantize_tokens"):
            if not case[name] or any(type(t) is not int or t < 0 for t in case[name]):
                raise ValueError("invalid token observations")
        for row in case["teacher_forced_logits"]:
            if type(row["bitwise_kernel_logits"]) is not bool:
                raise ValueError("invalid byte comparison flag")
            for field in ("int8_vs_fp16", "metal_vs_dequantize"):
                value = row[field]
                if (
                    type(value["argmax_equal"]) is not bool
                    or not math.isfinite(value["cosine"])
                    or not -1.000001 <= value["cosine"] <= 1.000001
                ):
                    raise ValueError("invalid cached numerical observation")
                for name in (
                    "relative_l2_error",
                    "max_absolute_error",
                    "reference_margin",
                ):
                    if not math.isfinite(value[name]) or value[name] < 0:
                        raise ValueError("invalid cached error/margin")
                for name in ("reference_top_two", "actual_top_two"):
                    top = value[name]
                    if (
                        len(top) != 2
                        or top[0]["token"] == top[1]["token"]
                        or any(
                            type(t["token"]) is not int
                            or t["token"] < 0
                            or not math.isfinite(t["logit"])
                            for t in top
                        )
                        or top[0]["logit"] < top[1]["logit"]
                    ):
                        raise ValueError("invalid cached top-two evidence")
                # argsort tie order need not match first-index greedy argmax.
                unambiguous = all(
                    value[n][0]["logit"] > value[n][1]["logit"]
                    for n in ("reference_top_two", "actual_top_two")
                )
                if unambiguous and value["argmax_equal"] != (
                    value["reference_top_two"][0]["token"]
                    == value["actual_top_two"][0]["token"]
                ):
                    raise ValueError("cached argmax flag disagrees with top-two tokens")
            top = row["int8_vs_fp16"]["reference_top_two"]
            if (
                top[0]["logit"] > top[1]["logit"]
                and case["fp16_tokens"][row["generated_position"]] != top[0]["token"]
            ):
                raise ValueError("source cached token differs from free generation")
    if type(report["cache_reclaimed"]) is not bool:
        raise ValueError("invalid cleanup flag")
    return cached_evidence.quality(report)


def byte_readback(control):
    observations = control["observations"]
    if len(observations) != 25 or [o["case_index"] for o in observations] != list(
        range(25)
    ):
        raise ValueError("incomplete byte case coverage")
    equal = 0
    for case in observations:
        if [r["position"] for r in case["rows"]] != list(range(32)):
            raise ValueError("incomplete byte position coverage")
        if type(case["private_cache_reclaimed"]) is not bool:
            raise ValueError("invalid byte cleanup flag")
        for row in case["rows"]:
            for name in ("native", "composed"):
                value = row[name]
                if (
                    value["dtype"] not in {"<f2", "<f4", "<f8"}
                    or len(value["shape"]) != 1
                    or type(value["shape"][0]) is not int
                    or value["shape"][0] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                ):
                    raise ValueError("invalid byte logit identity")
        count = sum(r["native"] == r["composed"] for r in case["rows"])
        if type(case["byte_equal_rows"]) is not int or case["byte_equal_rows"] != count:
            raise ValueError("byte summary disagrees with rows")
        equal += count
    passed = equal == 800 and all(o["private_cache_reclaimed"] for o in observations)
    if (
        type(control["passed"]) is not bool
        or control["total_rows"] != 800
        or control["byte_equal_rows"] != equal
        or control["passed"] != passed
    ):
        raise ValueError("byte acceptance flag disagrees")
    return passed


def byte_control(contract, frozen, quality_path):
    import numpy as np
    from forge_llm.cached_decode import replay_cached
    from forge_llm.mlx_engine import MlxEngine
    from verify_native_cached_bytes import identity

    observations = []
    with ExitStack() as ownership:
        engines = [
            ownership.enter_context(
                MlxEngine(
                    contract["item"]["model"],
                    max_model_length=2048,
                    kv_cache_bytes=512 << 20,
                    int8_mode=m,
                    decode_mode="rowwise",
                )
            )
            for m in ("reconstruct", "dequantize")
        ]
        for index, case in enumerate(read(quality_path)["cases"]):
            replays = [
                replay_cached(e, case["input_ids"], case["fp16_tokens"])
                for e in engines
            ]
            rows = [
                {
                    "position": p,
                    "native": identity(np.asarray(a)),
                    "composed": identity(np.asarray(b)),
                }
                for p, (a, b) in enumerate(
                    zip(replays[0]["logits"], replays[1]["logits"], strict=True)
                )
            ]
            observations.append(
                {
                    "case_index": index,
                    "rows": rows,
                    "byte_equal_rows": sum(r["native"] == r["composed"] for r in rows),
                    "private_cache_reclaimed": all(
                        r["cache_reclaimed"] for r in replays
                    ),
                }
            )
    total = sum(len(o["rows"]) for o in observations)
    equal = sum(o["byte_equal_rows"] for o in observations)
    return {
        "environment": source_provenance(),
        "model": frozen["model_provenance"],
        "quality_report_sha256": file_sha256(quality_path),
        "observations": observations,
        "total_rows": total,
        "byte_equal_rows": equal,
        "passed": total == equal == 800
        and all(o["private_cache_reclaimed"] for o in observations),
        "scope": "Signed-zero-sensitive native/composed cached bytes, not direct Metal equivalence or source quality.",
    }


def quality_phase(run, root):
    contract, frozen = load_frozen(root)
    directory = root / "quality"
    directory.mkdir()  # Partial stages cannot be silently rerun.
    gates = {}
    item = contract["item"]
    for mode in ("reconstruct", "metal"):
        output = directory / f"{mode}.json"
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
                "--prompts",
                str(CORPUS),
                "--output",
                str(output),
                "--int8-mode",
                mode,
                "--decode-mode",
                "rowwise",
                "--cached-logits",
                "--output-tokens",
                "32",
                "--calibration-stats",
                item["stats"],
            ],
            output,
            ("gates", "strict_checkpoint_passed"),
        )
        gates[mode] = quality_readback(report, contract, frozen, mode)
    control = byte_control(contract, frozen, directory / "reconstruct.json")
    save(directory / "native-composed-bytes.json", control)
    load_frozen(root)
    save(
        root / "quality-checkpoint.json",
        {
            "contract_sha256": file_sha256(root / "contract.json"),
            "files_sha256": {
                str(p): file_sha256(p) for p in sorted(directory.glob("*.json"))
            },
            "gates": gates,
            "native_composed_bytes": control["passed"],
            "quality_passed": gates["reconstruct"]["strict_checkpoint_passed"]
            and control["passed"],
            "direct_metal_is_diagnostic": True,
        },
    )


def quality_checkpoint(root):
    contract, frozen = load_frozen(root)
    checkpoint = read(root / "quality-checkpoint.json")
    directory = root / "quality"
    expected = {
        str(directory / f"{name}.json")
        for name in ("reconstruct", "metal", "native-composed-bytes")
    }
    if (
        set(checkpoint["files_sha256"]) != expected
        or {str(p) for p in directory.iterdir()} != expected
    ):
        raise ValueError("unexpected or incomplete quality files")
    check_bindings(checkpoint["files_sha256"])
    gates = {
        mode: quality_readback(read(directory / f"{mode}.json"), contract, frozen, mode)
        for mode in ("reconstruct", "metal")
    }
    control = read(directory / "native-composed-bytes.json")
    if (
        control["model"] != frozen["model_provenance"]
        or control["environment"]["runtime_source_sha256"]
        != contract["environment"]["runtime_source_sha256"]
        or control["quality_report_sha256"]
        != file_sha256(directory / "reconstruct.json")
    ):
        raise ValueError("byte control provenance changed")
    byte_readback(control)
    passed = gates["reconstruct"]["strict_checkpoint_passed"] and control["passed"]
    if (
        checkpoint["contract_sha256"] != file_sha256(root / "contract.json")
        or checkpoint["gates"] != gates
        or checkpoint["native_composed_bytes"] != control["passed"]
        or checkpoint["quality_passed"] != passed
        or checkpoint["direct_metal_is_diagnostic"] is not True
    ):
        raise ValueError("quality checkpoint disagrees with raw observations")
    return passed, gates


def resource_phase(run, root):
    if not quality_checkpoint(root)[0]:
        raise ValueError("quality failure forbids batching/resources")
    contract, _ = load_frozen(root)
    directory = root / "resources"
    directory.mkdir()
    item = contract["item"]
    for filename, command in [
        (
            "batch.json",
            [
                sys.executable,
                "benchmarks/validate_batch_numerics.py",
                "--model",
                item["model"],
                "--quality",
                str(root / "quality/reconstruct.json"),
                "--int8-mode",
                "reconstruct",
            ],
        ),
        (
            "boundary.json",
            [
                sys.executable,
                "benchmarks/validate_batch_boundaries.py",
                "--model",
                item["model"],
                "--int8-mode",
                "reconstruct",
            ],
        ),
    ]:
        output = directory / filename
        checked_report(
            run,
            command + ["--output", str(output)],
            output,
            ("gates", "checkpoint_passed"),
        )
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
            "reconstruct",
            "--decode-mode",
            "rowwise",
            "--output",
            str(directory / "32k.json"),
        ],
        check=True,
    )
    load_frozen(root)
    save(
        root / "resource-checkpoint.json",
        {
            "files_sha256": {str(p): file_sha256(p) for p in directory.glob("*.json")},
            "passed": resource_readback(root),
        },
    )


def resource_readback(root):
    contract, frozen = load_frozen(root)
    directory = root / "resources"
    if {p.name for p in directory.iterdir()} != {
        "batch.json",
        "boundary.json",
        "32k.json",
    }:
        raise ValueError("incomplete resource coverage")
    batch, boundary, long = [
        read(directory / n) for n in ("batch.json", "boundary.json", "32k.json")
    ]
    for report in (batch, boundary, long):
        if (
            report["model"] != frozen["model_provenance"]
            or report["environment"]["runtime_source_sha256"]
            != contract["environment"]["runtime_source_sha256"]
            or report["configuration"]["int8_mode"] != "reconstruct"
        ):
            raise ValueError("resource provenance mismatch")
    if (
        batch["configuration"]["historical_report_sha256"]
        != file_sha256(root / "quality/reconstruct.json")
        or batch["configuration"]["concurrencies"] != CONFIG["batch_concurrencies"]
        or boundary["configuration"]["prompt_lengths"]
        != CONFIG["boundary_prompt_lengths"]
        or long["configuration"]["context_lengths"] != [32768]
        or long["configuration"]["output_tokens"] != 2
        or long["configuration"]["decode_mode"] != "rowwise"
    ):
        raise ValueError("resource workloads differ from registration")
    return all(
        [
            cached_evidence.batching(batch)["checkpoint_passed"],
            cached_evidence.boundary(boundary)["checkpoint_passed"],
            cached_evidence.long_context(long),
            all(o["blocks_reclaimed"] for o in long["observations"]),
        ]
    )


def resource_checkpoint(root):
    if not quality_checkpoint(root)[0]:
        raise ValueError("quality failure forbids downstream evidence")
    report = read(root / "resource-checkpoint.json")
    expected = {
        str(root / "resources" / n) for n in ("batch.json", "boundary.json", "32k.json")
    }
    if set(report["files_sha256"]) != expected:
        raise ValueError("incomplete resource bindings")
    check_bindings(report["files_sha256"])
    passed = resource_readback(root)
    if report["passed"] != passed:
        raise ValueError("resource flags disagree with raw observations")
    return passed


def benchmark_phase(run, root):
    if not resource_checkpoint(root):
        raise ValueError("resource failure forbids timing")
    contract, _ = load_frozen(root)
    directory = root / "matrix"
    directory.mkdir()
    for index, (length, concurrency) in enumerate(
        (p, c)
        for p in CONFIG["prompt_lengths"]
        for c in CONFIG["benchmark_concurrencies"]
    ):
        workload = directory / f"workload-p{length}-c{concurrency}.json"
        save(workload, [[42] * length for _ in range(concurrency)])
        modes = CONFIG["benchmark_modes"]
        for mode in modes[index % 3 :] + modes[: index % 3]:
            model = contract["item"]["source" if mode == "auto" else "model"]
            run(
                [
                    sys.executable,
                    "-m",
                    "forge_llm.benchmark",
                    "--model",
                    model,
                    "--backend",
                    "mlx",
                    "--prompts",
                    str(workload),
                    "--output",
                    str(directory / f"{mode}-p{length}-c{concurrency}.json"),
                    "--max-model-length",
                    "2048",
                    "--max-sequences",
                    str(concurrency),
                    "--kv-cache-mib",
                    "1024",
                    "--int8-mode",
                    mode,
                    "--decode-mode",
                    "rowwise",
                    "--output-length",
                    "32",
                    "--warmups",
                    "5",
                    "--repetitions",
                    "3",
                ],
                check=True,
            )
    load_frozen(root)
    save(
        root / "matrix-checkpoint.json",
        {
            "files_sha256": {str(p): file_sha256(p) for p in directory.glob("*.json")},
            "results": matrix_readback(root),
        },
    )


def matrix_readback(root):
    contract, frozen = load_frozen(root)
    directory = root / "matrix"
    names = {
        f"{m}-p{p}-c{c}.json"
        for m in ["workload", *CONFIG["benchmark_modes"]]
        for p in CONFIG["prompt_lengths"]
        for c in CONFIG["benchmark_concurrencies"]
    }
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("incomplete or unexpected benchmark matrix")
    count = 0
    for length in CONFIG["prompt_lengths"]:
        for concurrency in CONFIG["benchmark_concurrencies"]:
            workload = directory / f"workload-p{length}-c{concurrency}.json"
            if read(workload) != [[42] * length for _ in range(concurrency)]:
                raise ValueError("benchmark workload changed")
            for mode in CONFIG["benchmark_modes"]:
                report = read(directory / f"{mode}-p{length}-c{concurrency}.json")
                config = report["configuration"]
                expected_model = (
                    contract["item"]["source_model"]
                    if mode == "auto"
                    else frozen["model_provenance"]
                )
                if (
                    report["model"] != expected_model
                    or report["environment"]["runtime_source_sha256"]
                    != contract["environment"]["runtime_source_sha256"]
                    or config["warmups"] != 5
                    or config["repetitions"] != 3
                    or config["output_length"] != 32
                    or config["decode_mode"] != "rowwise"
                    or config["int8_mode"] != mode
                    or config["concurrency"] != concurrency
                    or config["prompt_token_lengths"] != [length] * concurrency
                    or config["workload_sha256"] != file_sha256(workload)
                    or len(report["requests"]) != 3 * concurrency
                    or not cached_evidence.reclaimed(report["engine_stats"])
                ):
                    raise ValueError("benchmark provenance/workload/cleanup mismatch")
                for request in report["requests"]:
                    if request["output_tokens"] != 32:
                        raise ValueError("truncated timed output")
                count += 1
    return count


def audit(root):
    contract, frozen = load_frozen(root)
    passed, gates = quality_checkpoint(root)
    resources, matrices = False, 0
    downstream = [
        "resources",
        "resource-checkpoint.json",
        "matrix",
        "matrix-checkpoint.json",
    ]
    if not passed:
        if any((root / name).exists() for name in downstream):
            raise ValueError("quality rejection must stop before resources/timing")
        state = "rejected_at_held_out_quality"
    else:
        resources = resource_checkpoint(root)
        if not resources:
            if any(
                (root / name).exists() for name in ("matrix", "matrix-checkpoint.json")
            ):
                raise ValueError("resource rejection must stop before timing")
            state = "rejected_at_resources"
        else:
            index = read(root / "matrix-checkpoint.json")
            check_bindings(index["files_sha256"])
            matrices = matrix_readback(root)
            if index["results"] != matrices:
                raise ValueError("matrix index disagrees with raw evidence")
            state = "native_contract_validated"
    return {
        "schema_version": 1,
        "contract_sha256": file_sha256(root / "contract.json"),
        "environment": contract["environment"],
        "state": state,
        "quality_gates": gates,
        "cached_quality_rows": 1600,
        "native_composed_byte_rows": read(root / "quality/native-composed-bytes.json")[
            "byte_equal_rows"
        ],
        "prior_complete_files_unchanged": len(contract["prior_files_sha256"]),
        "registered_files_verified": len(contract["files_sha256"]),
        "captured_source_files_verified": len(
            contract["source_archive"]["files_sha256"]
        ),
        "quality_passed": passed,
        "resource_stage_executed": passed,
        "resource_stage_passed": resources,
        "performance_matrix_executed": bool(matrices),
        "benchmark_results": matrices,
        "strict_checkpoint_passed": passed and resources and matrices == 12,
        "readback_passed": True,
        "exported_model": frozen["model_provenance"],
        "scope": contract["scope"],
    }


def final_audit(root, expected_tests):
    from audit_refined_checkpoint import regression_contract

    report = audit(root)
    if read(root / "verification.json") != report:
        raise ValueError("saved verification disagrees with fresh readback")
    index = read(root / "evidence-index.json")
    paths = {
        str(p.relative_to(root)): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in {"evidence-index.json", "final-verification.json"}
    }
    if index["files_sha256"] != paths:
        raise ValueError("indexed final evidence changed or unexpected files added")
    return {
        **report,
        "final_tests": regression_contract(root / "test-results.xml", expected_tests),
        "test_results_sha256": file_sha256(root / "test-results.xml"),
        "evidence_index_sha256": file_sha256(root / "evidence-index.json"),
        "final_complete_files_verified": len(paths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("models/refined-qwen-validation")
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("benchmarks/results/refined-m3-max-2026-10-02"),
    )
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "prepare",
            "export",
            "quality",
            "resources",
            "benchmark",
            "audit",
            "final-audit",
        ),
        default="all",
    )
    parser.add_argument("--expected-tests", type=int)
    args = parser.parse_args()
    root = args.output_dir
    with exclusive_gpu_workflow() as run:
        if args.phase in ("all", "prepare"):
            prepare(root, args.artifact_dir, args.previous)
        if args.phase in ("all", "export"):
            export_phase(run, root)
        if args.phase in ("all", "quality"):
            quality_phase(run, root)
        if args.phase == "resources" or (
            args.phase == "all" and quality_checkpoint(root)[0]
        ):
            resource_phase(run, root)
        if args.phase == "benchmark" or (
            args.phase == "all"
            and quality_checkpoint(root)[0]
            and resource_checkpoint(root)
        ):
            benchmark_phase(run, root)
        if args.phase in ("all", "audit"):
            report = audit(root)
            save(root / "verification.json", report)
        elif args.phase == "final-audit":
            if args.expected_tests is None or args.expected_tests <= 0:
                parser.error("final-audit requires positive expected-tests")
            save(
                root / "evidence-index.json",
                {
                    "files_sha256": {
                        str(p.relative_to(root)): file_sha256(p)
                        for p in root.rglob("*")
                        if p.is_file()
                        and p.name
                        not in {"evidence-index.json", "final-verification.json"}
                    },
                },
            )
            report = final_audit(root, args.expected_tests)
            save(root / "final-verification.json", report)
        else:
            return
        print(json.dumps(report, indent=2), flush=True)
        if not report["strict_checkpoint_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
