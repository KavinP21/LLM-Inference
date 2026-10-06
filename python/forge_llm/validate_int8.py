"""Quality regression against identical FP16 source weights, never a speed benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .benchmark import model_provenance, source_provenance
from .mlx_engine import MlxEngine
from .model_file import ModelFile
from .quantization import quantize_per_channel


def compare_logits(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected, actual = expected.astype(np.float64), actual.astype(np.float64)
    norm = np.linalg.norm(expected)
    denominator = norm * np.linalg.norm(actual)
    indices = np.argsort(expected)[-2:][::-1]
    actual_indices = np.argsort(actual)[-2:][::-1]
    return {
        "cosine": float(np.dot(expected.ravel(), actual.ravel()) / denominator)
        if denominator
        else 1.0
        if np.array_equal(expected, actual)
        else 0.0,
        "relative_l2_error": float(np.linalg.norm(expected - actual) / norm)
        if norm
        else float(np.linalg.norm(actual)),
        "max_absolute_error": float(np.max(np.abs(expected - actual))),
        "argmax_equal": int(np.argmax(expected)) == int(np.argmax(actual)),
        "reference_margin": float(expected[indices[0]] - expected[indices[1]]),
        "reference_top_two": [
            {"token": int(i), "logit": float(expected[i])} for i in indices
        ],
        "actual_top_two": [
            {"token": int(i), "logit": float(actual[i])} for i in actual_indices
        ],
    }


def quality_gates(summary: dict, reclaimed: bool, cosine_gate: float = 0.999) -> dict:
    if not 0.999 <= cosine_gate <= 1:
        raise ValueError("cosine gate cannot be weaker than 0.999")
    gates = {
        "numerical": summary["minimum_layer_cosine"] >= cosine_gate
        and summary["minimum_logit_cosine"] >= cosine_gate,
        "exact_greedy": summary["exact_greedy_cases"] == summary["total_cases"],
        "cache_reclaimed": reclaimed,
        "kernel_numerical": summary["minimum_metal_dequantize_logit_cosine"]
        >= cosine_gate,
        "kernel_exact_greedy": summary["metal_dequantize_exact_cases"]
        == summary["total_cases"],
    }
    gates["strict_checkpoint_passed"] = all(gates.values())
    return gates


def verify_derivation(original, packed, calibration_stats: Path | None = None) -> dict:
    """Recompute every packed value and scale; never trust an export label."""
    if original.quantization or not packed.quantization:
        raise ValueError("validation requires FP16 source and INT8 candidate")
    calibrated = None
    if calibration_stats is not None:
        from .second_order import CalibrationStats

        calibrated = CalibrationStats(calibration_stats, original)
    for name, spec in packed.quantization.items():
        if original.tensor_info(name).shape != packed.tensor_info(name).shape:
            raise ValueError("source and quantized tensor shapes differ")
        if calibrated is None:
            q, s = quantize_per_channel(original.tensor_numpy(name))
        else:
            q, s, _ = calibrated.quantize(name, original.tensor_numpy(name))
        if not np.array_equal(q, packed.tensor_numpy(name)) or not np.array_equal(
            s, packed.tensor_numpy(spec.scale_name)
        ):
            raise ValueError(
                "INT8 artifact does not derive from the supplied source and quantization recipe"
            )
    for name in original.tensors:
        if name not in packed.quantization and not np.array_equal(
            original.tensor_numpy(name), packed.tensor_numpy(name)
        ):
            raise ValueError(f"retained source tensor differs: {name}")
    return {
        "method": calibrated.method if calibrated else "rtn_v1",
        "calibration_stats_sha256": calibrated.sha256 if calibrated else None,
        "quantizer_config": calibrated.config if calibrated else None,
        "all_packed_values_and_scales_recomputed": True,
        "retained_source_tensors_identical": True,
    }


def validate(
    source: Path,
    quantized: Path,
    tokenizer_name: str,
    prompts: list[str],
    count: int,
    positions: list[int],
    cosine_gate: float = 0.999,
    int8_mode: str = "metal",
    calibration_stats: Path | None = None,
    decode_mode: str = "batched",
    cached_teacher_forcing: bool = False,
) -> dict:
    from transformers import AutoTokenizer

    if (
        count <= 0
        or not prompts
        or not positions
        or min(positions) < 0
        or max(positions) >= count
    ):
        raise ValueError("invalid quality workload")
    if cached_teacher_forcing and positions != sorted(set(positions)):
        raise ValueError("cached positions must be sorted and unique")
    if not 0.999 <= cosine_gate <= 1:
        raise ValueError("cosine gate cannot be weaker than 0.999")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    with ModelFile(source) as metadata:
        context_limit = min(2048, metadata.config.max_position_embeddings)
    options = {
        "max_model_length": context_limit,
        "max_num_sequences": 4,
        "kv_cache_bytes": 512 << 20,
        "decode_mode": decode_mode,
    }
    ownership = ExitStack()
    try:
        reference = ownership.enter_context(MlxEngine(source, **options))
        candidate = ownership.enter_context(
            MlxEngine(quantized, int8_mode=int8_mode, **options)
        )
        baseline = ownership.enter_context(
            MlxEngine(quantized, int8_mode="dequantize", **options)
        )
    except Exception:
        ownership.close()
        raise
    layers, cases = [], []
    mx = reference.model.mx
    try:
        original, packed = reference.model.file, candidate.model.file

        def signature(config):
            values = asdict(config)
            values.update(
                head_dim=config.attention_head_dim,
                rope_local_theta=config.rope_local_theta or config.rope_theta,
                query_pre_attn_scalar=config.query_pre_attn_scalar
                or config.attention_head_dim,
            )
            return values

        if signature(original.config) != signature(packed.config):
            raise ValueError("FP16 and INT8 architecture configurations differ")
        derivation = verify_derivation(original, packed, calibration_stats)
        mx.random.seed(17)
        for name in packed.quantization:
            x = mx.random.normal((4, packed.tensor_info(name).shape[1])).astype(
                mx.float16
            )
            expected = np.asarray(reference.model._linear(x, name), dtype=np.float32)
            actual = np.asarray(candidate.model._linear(x, name), dtype=np.float32)
            dequantized = np.asarray(baseline.model._linear(x, name), dtype=np.float32)
            numerator = np.dot(
                expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
            )
            cosine = float(
                numerator
                / (
                    np.linalg.norm(expected.astype(np.float64))
                    * np.linalg.norm(actual.astype(np.float64))
                )
            )
            layers.append(
                {
                    "weight": name,
                    "cosine_vs_fp16": cosine,
                    "relative_l2_error_vs_fp16": float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    ),
                    "max_abs_error_vs_dequantized": float(
                        np.max(np.abs(actual - dequantized))
                    ),
                }
            )
        for index, prompt in enumerate(prompts):
            ids = tokenizer(prompt, add_special_tokens=True).input_ids
            expected = reference.generate(ids, count, eos_token_ids=[])
            actual = candidate.generate(ids, count, eos_token_ids=[])
            dense = baseline.generate(ids, count, eos_token_ids=[])
            observations = []
            cached_rows = None
            if cached_teacher_forcing:
                from .cached_decode import replay_cached

                cached_rows = [
                    replay_cached(e, ids, expected, positions)
                    for e in (reference, candidate, baseline)
                ]
                if cached_rows[0]["greedy_tokens"] != expected:
                    raise RuntimeError(
                        "source cached replay differs from greedy generation"
                    )
            for position in positions:
                if cached_rows is None:
                    context = ids + expected[:position]
                    fp16_logits = np.array(reference.debug_prefill_logits(context))
                    int8_logits = np.array(candidate.debug_prefill_logits(context))
                    dequant_logits = np.array(baseline.debug_prefill_logits(context))
                else:
                    row_index = positions.index(position)
                    fp16_logits, int8_logits, dequant_logits = [
                        r["logits"][row_index] for r in cached_rows
                    ]
                observations.append(
                    {
                        "generated_position": position,
                        "int8_vs_fp16": compare_logits(fp16_logits, int8_logits),
                        "metal_vs_dequantize": compare_logits(
                            dequant_logits, int8_logits
                        ),
                        "bitwise_kernel_logits": bool(
                            np.array_equal(dequant_logits, int8_logits)
                        ),
                    }
                )
            prefix = next(
                (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), count
            )
            divergence = None
            if prefix < count and prefix not in positions:
                if cached_teacher_forcing:
                    rows = [
                        replay_cached(e, ids, expected, [prefix])
                        for e in (reference, candidate)
                    ]
                    divergence = compare_logits(
                        rows[0]["logits"][0], rows[1]["logits"][0]
                    )
                else:
                    context = ids + expected[:prefix]
                    divergence = compare_logits(
                        np.array(reference.debug_prefill_logits(context)),
                        np.array(candidate.debug_prefill_logits(context)),
                    )
            cases.append(
                {
                    "prompt": prompt,
                    "input_ids": ids,
                    "fp16_tokens": expected,
                    "int8_tokens": actual,
                    "dequantize_tokens": dense,
                    "exact_greedy_parity": expected == actual,
                    "metal_dequantize_greedy_parity": dense == actual,
                    "matching_prefix": prefix,
                    "teacher_forced_logits": observations,
                    "first_divergence_logits": divergence,
                    "private_cache_reclaimed": all(
                        r["cache_reclaimed"] for r in cached_rows
                    )
                    if cached_rows
                    else None,
                }
            )
            print(
                json.dumps(
                    {
                        "case": index,
                        "matching_prefix": prefix,
                        "minimum_cosine": min(
                            o["int8_vs_fp16"]["cosine"] for o in observations
                        ),
                        "metal_dequantize_greedy_parity": dense == actual,
                    }
                ),
                flush=True,
            )
        all_logits = [
            o["int8_vs_fp16"] for case in cases for o in case["teacher_forced_logits"]
        ]
        kernel_logits = [
            o["metal_vs_dequantize"]
            for case in cases
            for o in case["teacher_forced_logits"]
        ]
        summary = {
            "minimum_layer_cosine": min(item["cosine_vs_fp16"] for item in layers),
            "minimum_logit_cosine": min(item["cosine"] for item in all_logits),
            "teacher_forced_top1_agreement": sum(
                item["argmax_equal"] for item in all_logits
            )
            / len(all_logits),
            "exact_greedy_cases": sum(case["exact_greedy_parity"] for case in cases),
            "total_cases": len(cases),
            "output_tokens_per_case": count,
            "metal_dequantize_exact_cases": sum(
                case["metal_dequantize_greedy_parity"] for case in cases
            ),
            "minimum_metal_dequantize_logit_cosine": min(
                item["cosine"] for item in kernel_logits
            ),
            "teacher_forced_logit_rows": len(all_logits),
            "bitwise_kernel_logit_rows": sum(
                o["bitwise_kernel_logits"]
                for c in cases
                for o in c["teacher_forced_logits"]
            ),
        }
        reclaimed = all(
            engine.stats()["kv_cache"]["allocated_blocks"] == 0
            and engine.stats()["kv_device_bytes"] == 0
            and engine.stats()["kv_cache"]["reserved_blocks"] == 0
            for engine in [reference, candidate, baseline]
        ) and (
            not cached_teacher_forcing
            or all(c["private_cache_reclaimed"] for c in cases)
        )
        return {
            "schema_version": 1,
            "purpose": "INT8 quality regression against the identical FP16 artifact; not Transformers certification",
            "environment": {
                "platform": platform.platform(),
                "device": mx.device_info(),
                "native_build": reference.build_info(),
                **source_provenance(),
            },
            "source_model": model_provenance(source),
            "quantized_model": model_provenance(quantized),
            "configuration": {
                "tokenizer": tokenizer_name,
                "seed": 17,
                "cosine_gate": cosine_gate,
                "teacher_forced_positions": positions,
                "int8_mode": int8_mode,
                "decode_mode": decode_mode,
                "teacher_forcing_path": "cached_paged_decode"
                if cached_teacher_forcing
                else "whole_prompt_prefill",
                "prompts_sha256": hashlib.sha256(
                    json.dumps(prompts, ensure_ascii=False).encode()
                ).hexdigest(),
            },
            "summary": summary,
            "derivation": derivation,
            "gates": quality_gates(summary, reclaimed, cosine_gate),
            "cache_reclaimed": reclaimed,
            "layers": layers,
            "cases": cases,
        }
    finally:
        ownership.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--prompts", type=Path, default=Path("benchmarks/quality-prompts.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument(
        "--int8-mode", choices=["metal", "reconstruct"], default="metal"
    )
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 1, 7, 15, 31])
    parser.add_argument("--calibration-stats", type=Path)
    parser.add_argument(
        "--cached-logits",
        action="store_true",
        help="observe every generated position on the actual paged decode path",
    )
    parser.add_argument(
        "--decode-mode", choices=["batched", "rowwise"], default="batched"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite quality evidence: {args.output}")
    result = validate(
        args.source,
        args.model,
        args.tokenizer,
        json.loads(args.prompts.read_text()),
        args.output_tokens,
        list(range(args.output_tokens)) if args.cached_logits else args.positions,
        int8_mode=args.int8_mode,
        calibration_stats=args.calibration_stats,
        decode_mode=args.decode_mode,
        cached_teacher_forcing=args.cached_logits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2))
    if not result["gates"]["strict_checkpoint_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
