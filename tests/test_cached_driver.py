from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from forge_llm import cached_evidence
from forge_llm.cached_policy import ALGORITHM, DEFAULT_SEARCH
from forge_llm.format import write_engine
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import seal_policy, validate_policy
from forge_llm.quantization import quantize_model
from forge_llm.validate_int8 import quality_gates, verify_derivation
from test_quantization import tiny_artifact
from test_second_order import fixture_stats


def driver(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "cached_driver", root / "benchmarks/run_cached_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("retain_alias", [True, False])
def test_projection_alias_conversion_has_independent_valid_storage(
    tmp_path, retain_alias
):
    base, source, output = [
        tmp_path / n for n in ["base.engine", "alias.engine", "packed.engine"]
    ]
    tiny_artifact(base)
    q, o = ["model.layers.0.self_attn." + n + "_proj.weight" for n in ["q", "o"]]
    with ModelFile(base) as original:
        tensors = {n: original.tensor_numpy(n).copy() for n in original.tensors}
        tensors[o] = tensors[q]
        write_engine(
            source,
            original.config,
            tensors,
            {o: q, "lm_head.weight": "model.embed_tokens.weight"},
        )
    stats = tmp_path / "stats.npz"
    with ModelFile(source) as original:
        metadata, _, checksum = fixture_stats(original, stats)
        names = set(metadata["weight_to_moments"])
        retained = [q] if retain_alias else []
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": ALGORITHM,
                "source_data_sha256": original.data_sha256,
                "quantizer_config": metadata["quantizer_config"],
                "calibration_stats_sha256": checksum,
                "search_config": DEFAULT_SEARCH,
                "retained_fp16": retained,
                "quantized_weights": sorted(names - set(retained)),
            }
        )
        assert validate_policy(policy, original.data_sha256, names) == retained
        with pytest.raises(ValueError, match="search"):
            validate_policy(
                seal_policy({**policy, "search_config": {}}),
                original.data_sha256,
                names,
            )
        quantize_model(source, output, policy=policy, calibration_stats=stats)
        with ModelFile(output) as packed:
            assert packed.tensor_info(q).offset != packed.tensor_info(o).offset
            assert verify_derivation(original, packed, stats)[
                "all_packed_values_and_scales_recomputed"
            ]


@pytest.mark.parametrize("field", ["files_sha256", "corpora_sha256", "tools_sha256"])
def test_freeze_binds_complete_file_bytes_and_all_inputs(tmp_path, monkeypatch, field):
    mod = driver(monkeypatch)
    path = tmp_path / "sample.json"
    mod.save(path, {"metadata": "original"})
    frozen = {
        "environment": {"runtime_source_sha256": "same"},
        "files_sha256": {},
        "corpora_sha256": {},
        "tools_sha256": {},
    }
    frozen[field] = {str(path): mod.file_sha256(path)}
    mod.save(tmp_path / "frozen.json", frozen)
    monkeypatch.setattr(
        mod, "source_provenance", lambda: {"runtime_source_sha256": "same"}
    )
    assert mod.load_frozen(tmp_path) == frozen
    path.write_text('{"metadata": "changed"}')
    with pytest.raises(RuntimeError, match="frozen"):
        mod.load_frozen(tmp_path)


def test_incomplete_evidence_and_stale_runtime_fail_closed(tmp_path, monkeypatch):
    mod = driver(monkeypatch)
    mod.save(
        tmp_path / "frozen.json", {"environment": {"runtime_source_sha256": "old"}}
    )
    monkeypatch.setattr(
        mod, "source_provenance", lambda: {"runtime_source_sha256": "new"}
    )
    with pytest.raises(RuntimeError, match="runtime changed"):
        mod.load_frozen(tmp_path)
    monkeypatch.setattr(mod, "load_frozen", lambda _: {})
    with pytest.raises(ValueError, match="incomplete"):
        mod.audit(tmp_path)


def test_new_quality_command_never_uses_prefill_proxy(monkeypatch):
    mod = driver(monkeypatch)
    command = mod.quality_command(
        {"source": "source", "tokenizer": "local", "stats": "frozen"},
        "model",
        Path("corpus"),
        Path("output"),
        "reconstruct",
        True,
    )
    assert "--cached-logits" in command and command[-2:] == [
        "--calibration-stats",
        "frozen",
    ]


def test_raw_workload_tampering_cannot_be_promoted_into_pass():
    refs = [{"tokens": [1], "logits": [{"sha256": "good"}]}]
    work = {
        "requests": [
            {
                "case_index": 0,
                "output_budget": 1,
                "tokens": [1],
                "logits": [{"sha256": "good"}],
                "exact_tokens": True,
                "exact_logit_rows": 1,
                "total_logit_rows": 1,
                "events_exact": True,
            }
        ],
        "summary": {
            "cases": 1,
            "exact_token_cases": 1,
            "exact_logit_rows": 1,
            "total_logit_rows": 1,
            "events_exact": True,
            "cache_reclaimed": True,
        },
        "engine_stats": {
            "kv_cache": {"allocated_blocks": 0, "reserved_blocks": 0},
            "kv_device_bytes": 0,
        },
    }
    assert all(cached_evidence.workload(work, refs).values())
    for mutation in ["hash", "tokens", "cache", "duplicate"]:
        changed = copy.deepcopy(work)
        if mutation == "hash":
            changed["requests"][0]["logits"][0]["sha256"] = "different"
        elif mutation == "tokens":
            changed["requests"][0]["tokens"] = [2]
        elif mutation == "cache":
            changed["engine_stats"]["kv_device_bytes"] = 1
        else:
            changed["requests"] *= 2
        with pytest.raises(ValueError):
            cached_evidence.workload(changed, refs)


def test_long_resource_check_is_not_vacuously_true():
    assert not cached_evidence.long_context({"observations": [], "all_reclaimed": True})


def test_byte_sensitive_replay_identity_catches_signed_zero_and_dtype(monkeypatch):
    driver(monkeypatch)
    from verify_native_cached_bytes import identity

    positive = np.array([0.0, 1.0], dtype=np.float32)
    negative = np.array([-0.0, 1.0], dtype=np.float32)
    assert np.array_equal(positive, negative)  # numeric equality is not byte equality
    assert identity(positive) != identity(negative)
    assert identity(positive) != identity(positive.astype(np.float16))
    assert identity(positive) == identity(positive.copy())
    with pytest.raises(ValueError):
        identity(np.array([np.nan]))


def test_quality_gate_recomputed_from_every_cached_row():
    rows = [
        {
            "generated_position": p,
            "int8_vs_fp16": {"cosine": 1.0, "argmax_equal": True},
            "metal_vs_dequantize": {"cosine": 1.0},
            "bitwise_kernel_logits": True,
        }
        for p in range(32)
    ]
    case = {
        "fp16_tokens": [1] * 32,
        "int8_tokens": [1] * 32,
        "dequantize_tokens": [1] * 32,
        "matching_prefix": 32,
        "exact_greedy_parity": True,
        "metal_dequantize_greedy_parity": True,
        "teacher_forced_logits": rows,
        "private_cache_reclaimed": True,
    }
    summary = {
        "total_cases": 1,
        "output_tokens_per_case": 32,
        "minimum_layer_cosine": 1.0,
        "minimum_logit_cosine": 1.0,
        "teacher_forced_top1_agreement": 1.0,
        "exact_greedy_cases": 1,
        "metal_dequantize_exact_cases": 1,
        "minimum_metal_dequantize_logit_cosine": 1.0,
        "teacher_forced_logit_rows": 32,
        "bitwise_kernel_logit_rows": 32,
    }
    report = {
        "cases": [case],
        "summary": summary,
        "layers": [{"cosine_vs_fp16": 1.0}],
        "cache_reclaimed": True,
        "configuration": {"cosine_gate": 0.999},
        "gates": quality_gates(summary, True),
    }
    assert cached_evidence.quality(report)["strict_checkpoint_passed"]
    for mutation in ["tokens", "logit", "truncated", "cache"]:
        changed = copy.deepcopy(report)
        if mutation == "tokens":
            changed["cases"][0]["int8_tokens"][3] = 2
        elif mutation == "logit":
            changed["cases"][0]["teacher_forced_logits"][4]["int8_vs_fp16"][
                "cosine"
            ] = 0.5
        elif mutation == "truncated":
            changed["cases"][0]["teacher_forced_logits"].pop()
        else:
            changed["cache_reclaimed"] = False
        with pytest.raises(ValueError):
            cached_evidence.quality(changed)
