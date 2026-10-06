from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from forge_llm.validate_int8 import quality_gates


def driver(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "qwen_validation_driver", repo / "benchmarks/run_qwen_validation_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_quality(mod):
    contract = {
        "environment": {"runtime_source_sha256": "runtime"},
        "corpus_prompts": [f"held out {i}" for i in range(25)],
        "item": {
            "tokenizer": "offline",
            "source_model": {"identity": "source"},
            "policy": {
                "calibration_stats_sha256": "stats",
                "quantized_weights": ["weight"],
            },
        },
    }
    frozen = {"model_provenance": {"identity": "candidate"}}
    comparison = {
        "cosine": 1.0,
        "argmax_equal": True,
        "relative_l2_error": 0.0,
        "max_absolute_error": 0.0,
        "reference_margin": 1.0,
        "reference_top_two": [{"token": 1, "logit": 2.0}, {"token": 2, "logit": 1.0}],
        "actual_top_two": [{"token": 1, "logit": 2.0}, {"token": 2, "logit": 1.0}],
    }
    report = {
        "environment": contract["environment"],
        "source_model": contract["item"]["source_model"],
        "quantized_model": frozen["model_provenance"],
        "configuration": {
            "teacher_forcing_path": "cached_paged_decode",
            "teacher_forced_positions": list(range(32)),
            "cosine_gate": 0.999,
            "int8_mode": "reconstruct",
            "decode_mode": "rowwise",
            "tokenizer": "offline",
            "seed": 17,
            "prompts_sha256": hashlib.sha256(
                json.dumps(contract["corpus_prompts"], ensure_ascii=False).encode()
            ).hexdigest(),
        },
        "summary": {
            "minimum_layer_cosine": 1.0,
            "minimum_logit_cosine": 1.0,
            "teacher_forced_top1_agreement": 1.0,
            "exact_greedy_cases": 25,
            "total_cases": 25,
            "output_tokens_per_case": 32,
            "metal_dequantize_exact_cases": 25,
            "minimum_metal_dequantize_logit_cosine": 1.0,
            "teacher_forced_logit_rows": 800,
            "bitwise_kernel_logit_rows": 800,
        },
        "derivation": {
            "method": "coordinate_refined_scale_aware_v1",
            "calibration_stats_sha256": "stats",
            "quantizer_config": mod.DEFAULT_CONFIG,
            "all_packed_values_and_scales_recomputed": True,
            "retained_source_tensors_identical": True,
        },
        "cache_reclaimed": True,
        "layers": [
            {
                "weight": "weight",
                "cosine_vs_fp16": 1.0,
                "relative_l2_error_vs_fp16": 0.0,
                "max_abs_error_vs_dequantized": 0.0,
            }
        ],
        "cases": [
            {
                "prompt": prompt,
                "input_ids": [3],
                "fp16_tokens": [1] * 32,
                "int8_tokens": [1] * 32,
                "dequantize_tokens": [1] * 32,
                "exact_greedy_parity": True,
                "metal_dequantize_greedy_parity": True,
                "matching_prefix": 32,
                "private_cache_reclaimed": True,
                "teacher_forced_logits": [
                    {
                        "generated_position": p,
                        "int8_vs_fp16": copy.deepcopy(comparison),
                        "metal_vs_dequantize": copy.deepcopy(comparison),
                        "bitwise_kernel_logits": True,
                    }
                    for p in range(32)
                ],
            }
            for prompt in contract["corpus_prompts"]
        ],
    }
    report["gates"] = quality_gates(report["summary"], True)
    return contract, frozen, report


def test_native_acceptance_does_not_certify_direct_metal(monkeypatch):
    mod = driver(monkeypatch)
    contract, frozen, report = fixture_quality(mod)
    assert mod.CONFIG["acceptance_mode"] == "reconstruct"
    assert mod.CONFIG["diagnostic_mode"] == "metal"
    assert "metal" not in mod.CONFIG["benchmark_modes"]
    assert mod.quality_readback(report, contract, frozen, "reconstruct")[
        "strict_checkpoint_passed"
    ]
    with pytest.raises(ValueError, match="mismatch"):
        mod.quality_readback(report, contract, frozen, "metal")


@pytest.mark.parametrize(
    "mutation",
    [
        "position",
        "tokens",
        "summary",
        "corpus",
        "runtime",
        "model",
        "mode",
        "method",
        "stats",
        "layer",
        "nan",
        "error",
        "argmax",
        "top_two",
        "reference_token",
        "bool",
        "gate",
        "threshold",
        "cleanup",
    ],
)
def test_raw_quality_readback_rejects_tampering_and_sparse_coverage(
    monkeypatch, mutation
):
    mod = driver(monkeypatch)
    contract, frozen, report = fixture_quality(mod)
    row = report["cases"][0]["teacher_forced_logits"][0]
    if mutation == "position":
        report["cases"][0]["teacher_forced_logits"].pop()
    elif mutation == "tokens":
        report["cases"][0]["int8_tokens"].pop()
    elif mutation == "summary":
        report["summary"]["exact_greedy_cases"] = 24
    elif mutation == "corpus":
        report["cases"][0]["prompt"] = "replacement"
    elif mutation == "runtime":
        report["environment"] = {"runtime_source_sha256": "different"}
    elif mutation == "model":
        report["quantized_model"] = {"identity": "different"}
    elif mutation == "mode":
        report["configuration"]["decode_mode"] = "batched"
    elif mutation == "method":
        report["derivation"]["method"] = "rtn_v1"
    elif mutation == "stats":
        report["derivation"]["calibration_stats_sha256"] = "different"
    elif mutation == "layer":
        report["layers"] = []
    elif mutation == "nan":
        row["int8_vs_fp16"]["cosine"] = float("nan")
    elif mutation == "error":
        row["int8_vs_fp16"]["relative_l2_error"] = -1
    elif mutation == "argmax":
        row["int8_vs_fp16"]["argmax_equal"] = False
    elif mutation == "top_two":
        row["int8_vs_fp16"]["actual_top_two"][1]["token"] = 1
    elif mutation == "reference_token":
        report["cases"][0]["fp16_tokens"][0] = 3
    elif mutation == "bool":
        report["cases"][0]["exact_greedy_parity"] = 1
    elif mutation == "gate":
        report["gates"]["numerical"] = False
    elif mutation == "threshold":
        report["configuration"]["cosine_gate"] = 0.998
    else:
        report["cache_reclaimed"] = False
    with pytest.raises(ValueError):
        mod.quality_readback(report, contract, frozen, "reconstruct")


def test_complete_quality_rejection_is_evidence_not_an_exception(monkeypatch):
    mod = driver(monkeypatch)
    contract, frozen, report = fixture_quality(mod)
    for case in report["cases"]:
        case["int8_tokens"][0] = case["dequantize_tokens"][0] = 3
        case["matching_prefix"] = 0
        case["exact_greedy_parity"] = False
    report["summary"]["exact_greedy_cases"] = 0
    report["gates"] = quality_gates(report["summary"], True)
    gates = mod.quality_readback(report, contract, frozen, "reconstruct")
    assert not gates["strict_checkpoint_passed"]
    assert gates["numerical"] and gates["kernel_exact_greedy"]


def test_top_two_ties_do_not_change_first_index_runtime_argmax(monkeypatch):
    mod = driver(monkeypatch)
    contract, frozen, report = fixture_quality(mod)
    row = report["cases"][0]["teacher_forced_logits"][0]
    row["int8_vs_fp16"]["reference_top_two"] = [
        {"token": 2, "logit": 2.0},
        {"token": 1, "logit": 2.0},
    ]
    row["int8_vs_fp16"]["reference_margin"] = 0.0
    assert mod.quality_readback(report, contract, frozen, "reconstruct")[
        "strict_checkpoint_passed"
    ]


def test_byte_readback_accepts_complete_rejection_and_rejects_bad_identities(
    monkeypatch,
):
    mod = driver(monkeypatch)
    identity = {"dtype": "<f2", "shape": [32], "sha256": "a" * 64}
    report = {
        "total_rows": 800,
        "byte_equal_rows": 800,
        "passed": True,
        "observations": [
            {
                "case_index": i,
                "private_cache_reclaimed": True,
                "byte_equal_rows": 32,
                "rows": [
                    {
                        "position": p,
                        "native": copy.deepcopy(identity),
                        "composed": copy.deepcopy(identity),
                    }
                    for p in range(32)
                ],
            }
            for i in range(25)
        ],
    }
    assert mod.byte_readback(report)
    negative = copy.deepcopy(report)
    negative["observations"][0]["rows"][0]["native"]["sha256"] = "b" * 64
    negative["observations"][0]["byte_equal_rows"] = 31
    negative["byte_equal_rows"] = 799
    negative["passed"] = False
    assert mod.byte_readback(negative) is False
    for field, value in [
        ("dtype", "object"),
        ("shape", [True]),
        ("sha256", "not a digest"),
    ]:
        bad = copy.deepcopy(negative)
        bad["observations"][0]["rows"][0]["native"][field] = value
        with pytest.raises(ValueError, match="identity"):
            mod.byte_readback(bad)


def test_cpu_export_freeze_and_exact_derivation_for_real_tiny_artifact(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from forge_llm.calibrate_second_order import projection_groups
    from forge_llm.format import write_engine
    from forge_llm.model_file import ModelFile
    from forge_llm.precision_policy import seal_policy
    from forge_llm.quantization import quantize_model
    from forge_llm.refined import ALGORITHM, DEFAULT_SEARCH
    from forge_llm.second_order import block_moments, write_calibration_stats
    from forge_llm.validate_int8 import verify_derivation
    from test_quantization import tiny_artifact

    mod = driver(monkeypatch)
    source, packed, stats = [
        tmp_path / n for n in ("source.engine", "candidate.engine", "stats.npz")
    ]
    # At very tiny widths, 256-byte scale alignment exceeds weight savings.
    # Use a still-small, valid fixture large enough for the artifact-size gate.
    base = tmp_path / "base.engine"
    tiny_artifact(base)
    with ModelFile(base) as model:
        tensors = {}
        for name in model.tensors:
            values = np.repeat(model.tensor_numpy(name), 8, axis=0)
            if values.ndim == 2:
                values = np.repeat(values, 8, axis=1)
            tensors[name] = values
        config = replace(
            model.config,
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            head_dim=32,
            query_pre_attn_scalar=32.0,
        )
        write_engine(
            source, config, tensors, {"lm_head.weight": "model.embed_tokens.weight"}
        )
    groups = projection_groups(2)
    mapping = {name: key for key, names in groups.items() for name in names}
    with ModelFile(source) as model:
        moments = {
            key: block_moments(np.eye(model.tensor_info(names[0]).shape[1]), 64)
            for key, names in groups.items()
        }
        stats_sha = write_calibration_stats(
            stats,
            {
                "schema_version": 1,
                "algorithm": ALGORITHM,
                "source_data_sha256": model.data_sha256,
                "quantizer_config": mod.DEFAULT_CONFIG,
                "weight_to_moments": mapping,
            },
            moments,
        )
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": mod.POLICY_ALGORITHM,
                "source_data_sha256": model.data_sha256,
                "quantizer_config": mod.DEFAULT_CONFIG,
                "search_config": DEFAULT_SEARCH,
                "calibration_stats_sha256": stats_sha,
                "quantized_weights": sorted(mapping),
                "retained_fp16": [],
                "calibration_passed": True,
            }
        )
    quantize_model(source, packed, policy=policy, calibration_stats=stats)
    mod.save(tmp_path / "contract.json", {"registered": True})
    contract = {
        "item": {
            "model": str(packed),
            "source": str(source),
            "stats": str(stats),
            "source_model": mod.model_provenance(source),
            "policy": policy,
        }
    }
    monkeypatch.setattr(mod, "load_contract", lambda root: contract)
    mod.save(
        tmp_path / "frozen.json",
        {
            "contract_sha256": mod.file_sha256(tmp_path / "contract.json"),
            "files_sha256": {
                str(p): mod.file_sha256(p)
                for p in (packed, packed.with_suffix(".engine.json"))
            },
            "model_provenance": mod.model_provenance(packed),
        },
    )
    assert mod.load_frozen(tmp_path)[0] == contract
    with ModelFile(source) as original, ModelFile(packed) as candidate:
        assert verify_derivation(original, candidate, stats)[
            "all_packed_values_and_scales_recomputed"
        ]
    with packed.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ValueError, match="file changed"):
        mod.load_frozen(tmp_path)


def test_failed_quality_forbids_resource_and_timing_calls(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    monkeypatch.setattr(mod, "quality_checkpoint", lambda root: (False, {}))
    called = []
    for function in (mod.resource_phase, mod.benchmark_phase):
        with pytest.raises(ValueError, match="quality failure"):
            function(lambda *a, **k: called.append(a), tmp_path)
    assert not called and not list(tmp_path.iterdir())


def test_failed_resource_forbids_benchmark_calls(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    monkeypatch.setattr(mod, "resource_checkpoint", lambda root: False)
    called = []
    with pytest.raises(ValueError, match="resource failure"):
        mod.benchmark_phase(lambda *a, **k: called.append(a), tmp_path)
    assert not called and not list(tmp_path.iterdir())


def test_partial_export_and_quality_are_preserved(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    model = tmp_path / "candidate.engine"
    model.write_bytes(b"partial evidence")
    contract = {"item": {"model": str(model)}}
    monkeypatch.setattr(mod, "load_contract", lambda root: contract)
    with pytest.raises(FileExistsError, match="preserved"):
        mod.export_phase(lambda *a, **k: pytest.fail("must not execute"), tmp_path)
    monkeypatch.setattr(mod, "load_frozen", lambda root: (contract, {}))
    (tmp_path / "quality").mkdir()
    with pytest.raises(FileExistsError):
        mod.quality_phase(lambda *a, **k: pytest.fail("must not execute"), tmp_path)
    assert model.read_bytes() == b"partial evidence"


def test_new_registration_refuses_existing_directories_before_reading_old_run(
    tmp_path, monkeypatch
):
    mod = driver(monkeypatch)
    with pytest.raises(FileExistsError):
        mod.prepare(tmp_path, tmp_path / "models", tmp_path / "missing")


def test_source_capture_includes_new_driver_tests_and_runtime(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    captured = mod.capture_sources(tmp_path)
    assert "benchmarks/run_qwen_validation_checkpoint.py" in captured["files_sha256"]
    assert "tests/test_qwen_validation_driver.py" in captured["files_sha256"]
    assert "python/forge_llm/refined.py" in captured["files_sha256"]
    mod.check_sources(tmp_path, captured)
    with pytest.raises(FileExistsError):
        mod.capture_sources(tmp_path)


def test_live_contract_rejects_policy_corpus_runtime_and_binding_drift(
    tmp_path, monkeypatch
):
    from test_refined_driver import fixture_report

    mod = driver(monkeypatch)
    corpus, current, old, calibration = [
        tmp_path / n for n in ("corpus.json", "input", "old", "calibration.json")
    ]
    mod.save(corpus, [f"held out {i}" for i in range(25)])
    current.write_bytes(b"fixed tool")
    old.write_bytes(b"preserved evidence")
    report = fixture_report()
    mod.save(calibration, report)
    contract = {
        "configuration": copy.deepcopy(mod.CONFIG),
        "corpus": str(corpus),
        "corpus_prompts": mod.read(corpus),
        "environment": {"runtime_source_sha256": "runtime"},
        "files_sha256": {str(p): mod.file_sha256(p) for p in (corpus, current)},
        "prior_files_sha256": {str(old): mod.file_sha256(old)},
        "source_archive": mod.capture_sources(tmp_path),
        "item": {"calibration_report": str(calibration), "policy": report["policy"]},
    }
    mod.save(tmp_path / "contract.json", contract)
    monkeypatch.setattr(mod, "CORPUS", corpus)
    monkeypatch.setattr(
        mod, "source_provenance", lambda: {"runtime_source_sha256": "runtime"}
    )
    assert mod.load_contract(tmp_path) == contract
    for path in (corpus, current, old):
        data = path.read_bytes()
        path.write_bytes(b"drift")
        with pytest.raises(ValueError, match="file changed"):
            mod.load_contract(tmp_path)
        path.write_bytes(data)
    changed = copy.deepcopy(contract)
    changed["configuration"]["cosine_gate"] = 0.998
    (tmp_path / "contract.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration changed"):
        mod.load_contract(tmp_path)
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    report["policy"]["policy_sha256"] = "different"
    calibration.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="sealed calibration"):
        mod.load_contract(tmp_path)


def test_final_index_detects_changed_and_unexpected_evidence(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    verification = {"readback_passed": True}
    monkeypatch.setattr(mod, "audit", lambda root: verification)
    mod.save(tmp_path / "verification.json", verification)
    (tmp_path / "test-results.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>'
    )
    mod.save(
        tmp_path / "evidence-index.json",
        {
            "files_sha256": {
                str(p.relative_to(tmp_path)): mod.file_sha256(p)
                for p in tmp_path.iterdir()
                if p.name != "evidence-index.json"
            },
        },
    )
    assert mod.final_audit(tmp_path, 1)["final_complete_files_verified"] == 2
    (tmp_path / "unexpected").write_bytes(b"unregistered")
    with pytest.raises(ValueError, match="indexed final evidence"):
        mod.final_audit(tmp_path, 1)


def test_readback_rejects_downstream_work_after_quality_failure(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    contract = {
        "environment": {},
        "prior_files_sha256": {},
        "files_sha256": {},
        "source_archive": {"files_sha256": {}},
        "scope": "test",
    }
    frozen = {"model_provenance": {}}
    monkeypatch.setattr(mod, "load_frozen", lambda root: (contract, frozen))
    monkeypatch.setattr(mod, "quality_checkpoint", lambda root: (False, {}))
    (tmp_path / "matrix").mkdir()
    with pytest.raises(ValueError, match="rejection must stop"):
        mod.audit(tmp_path)
