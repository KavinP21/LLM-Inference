"""Offline single-projection ablation and source-bound mixed-precision selection.

No loss/scale fitting or held-out output is used. This is a conservative whole-
matrix retention experiment, not GPTQ/AWQ or a general quality certification.
"""

from __future__ import annotations

import argparse
import json
import platform
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

import numpy as np

from .benchmark import model_provenance, source_provenance
from .mlx_engine import MlxEngine
from .model_file import ModelFile
from .precision_policy import (
    canonical_sha256,
    rank_sensitivities,
    retained_for_fraction,
    seal_policy,
    validate_corpora,
)
from .quantization import dequantize_per_channel, quantize_per_channel
from .validate_int8 import compare_logits


@contextmanager
def replaced_weights(model, replacements: dict):
    """Offline-only perturbation; restore ownership even if execution fails."""
    previous = {name: model.weights[name] for name in replacements}
    try:
        model.weights.update(replacements)
        yield
    finally:
        model.weights.update(previous)


def cache_reclaimed(engine) -> bool:
    stats = engine.stats()
    return (
        stats["kv_cache"]["allocated_blocks"] == 0
        and stats["kv_cache"]["reserved_blocks"] == 0
        and stats["kv_device_bytes"] == 0
    )


def calibrate(
    source: Path,
    tokenizer_name: str,
    prompts: list[str],
    held_out: list[str],
    *,
    output_tokens: int = 32,
    positions: tuple[int, ...] = (0, 7, 31),
    probe_count: int = 8,
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
) -> dict:
    validate_corpora(prompts, held_out)
    if (
        output_tokens <= 0
        or not positions
        or min(positions) < 0
        or max(positions) >= output_tokens
        or probe_count <= 0
        or not fractions
        or fractions[0] != 1.0
        or any(not 0.25 <= f <= 1 for f in fractions)
        or any(a <= b for a, b in pairwise(fractions))
    ):
        raise ValueError("invalid calibration workload; INT8 floor is 25%")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    input_ids = [tokenizer(p, add_special_tokens=True).input_ids for p in prompts]
    held_ids = {
        tuple(tokenizer(p, add_special_tokens=True).input_ids) for p in held_out
    }
    if any(tuple(ids) in held_ids for ids in input_ids):
        raise ValueError("tokenized calibration and held-out prompts overlap")
    with ModelFile(source) as metadata:
        context_limit = min(2048, metadata.config.max_position_embeddings)
    if any(not ids or len(ids) + output_tokens > context_limit for ids in input_ids):
        raise ValueError("calibration prompt is outside the supported context")
    with MlxEngine(
        source, max_model_length=context_limit, kv_cache_bytes=512 << 20
    ) as engine:
        if engine.model.file.quantization:
            raise ValueError("calibration requires a validated FP16 source")
        mx, file = engine.model.mx, engine.model.file
        names = sorted(
            name
            for name in file.tensors
            if name.startswith("model.layers.") and name.endswith("_proj.weight")
        )
        if not names:
            raise ValueError("no supported projection matrices")
        goldens = []
        for prompt, ids in zip(prompts, input_ids):
            tokens = engine.generate(ids, output_tokens, eos_token_ids=[])
            contexts = [ids + tokens[:position] for position in positions]
            goldens.append(
                {
                    "prompt": prompt,
                    "input_ids": ids,
                    "tokens": tokens,
                    "contexts": contexts,
                    "logits": [
                        np.array(engine.debug_prefill_logits(c)) for c in contexts
                    ],
                }
            )
        all_probes = [
            (context, logits)
            for golden in goldens
            for context, logits in zip(golden["contexts"], golden["logits"])
        ]
        # Deterministic coverage across the entire corpus and teacher-forced positions.
        probe_indices = sorted(
            set(
                np.linspace(
                    0, len(all_probes) - 1, min(probe_count, len(all_probes)), dtype=int
                ).tolist()
            )
        )
        probes = [all_probes[index] for index in probe_indices]
        reconstructed, observations = {}, []
        for index, name in enumerate(names):
            q, scales = quantize_per_channel(file.tensor_numpy(name))
            reconstructed[name] = mx.array(dequantize_per_channel(q, scales))
            mx.eval(reconstructed[name])
            with replaced_weights(engine.model, {name: reconstructed[name]}):
                metrics = [
                    compare_logits(
                        expected, np.array(engine.debug_prefill_logits(context))
                    )
                    for context, expected in probes
                ]
            observation = {
                "weight": name,
                "fp16_bytes": file.tensor_info(name).nbytes,
                "top1_changes": sum(not metric["argmax_equal"] for metric in metrics),
                "mean_squared_relative_logit_error": float(
                    np.mean([metric["relative_l2_error"] ** 2 for metric in metrics])
                ),
                "minimum_logit_cosine": min(metric["cosine"] for metric in metrics),
                "probes": metrics,
            }
            observations.append(observation)
            print(
                json.dumps(
                    {
                        "ablation": index + 1,
                        "total": len(names),
                        "weight": name,
                        "top1_changes": observation["top1_changes"],
                        "minimum_cosine": observation["minimum_logit_cosine"],
                    }
                ),
                flush=True,
            )
        ranked = rank_sensitivities(observations)
        total_bytes = sum(item["fp16_bytes"] for item in ranked)
        trials = []
        for fraction in fractions:
            retained = retained_for_fraction(ranked, fraction)
            quantized = sorted(set(names) - set(retained))
            with replaced_weights(
                engine.model, {n: reconstructed[n] for n in quantized}
            ):
                cases = []
                for golden in goldens:
                    actual = engine.generate(
                        golden["input_ids"], output_tokens, eos_token_ids=[]
                    )
                    metrics = [
                        compare_logits(
                            expected, np.array(engine.debug_prefill_logits(context))
                        )
                        for context, expected in zip(
                            golden["contexts"], golden["logits"]
                        )
                    ]
                    cases.append(
                        {
                            "prompt": golden["prompt"],
                            "input_ids": golden["input_ids"],
                            "fp16_tokens": golden["tokens"],
                            "candidate_tokens": actual,
                            "exact_greedy_parity": actual == golden["tokens"],
                            "teacher_forced_logits": metrics,
                        }
                    )
            minimum = min(
                m["cosine"] for case in cases for m in case["teacher_forced_logits"]
            )
            exact = sum(case["exact_greedy_parity"] for case in cases)
            reclaimed = cache_reclaimed(engine)
            trial = {
                "requested_quantized_projection_fraction": fraction,
                "quantized_projection_fraction": sum(
                    file.tensor_info(n).nbytes for n in quantized
                )
                / total_bytes,
                "retained_fp16": retained,
                "quantized_weights": quantized,
                "exact_greedy_cases": exact,
                "total_cases": len(cases),
                "minimum_logit_cosine": minimum,
                "cache_reclaimed": reclaimed,
                "calibration_passed": exact == len(cases)
                and minimum >= 0.999
                and reclaimed,
                "cases": cases,
            }
            trials.append(trial)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in trial.items()
                        if k not in {"cases", "retained_fp16", "quantized_weights"}
                    }
                ),
                flush=True,
            )
            if trial["calibration_passed"]:
                break
        # No unseen regression is consulted when selecting a policy, even on failure.
        selected = trials[-1]
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": "single_projection_logit_ablation_v1",
                "source_data_sha256": file.data_sha256,
                "calibration_corpus_sha256": canonical_sha256(prompts),
                "held_out_guard_sha256": canonical_sha256(held_out),
                "retained_fp16": selected["retained_fp16"],
                "quantized_weights": selected["quantized_weights"],
                "quantized_projection_fraction": selected[
                    "quantized_projection_fraction"
                ],
                "calibration_passed": selected["calibration_passed"],
                "execution_contract": "FP16-rounded reconstruction with native MLX GEMM; not direct INT8 reduction",
            }
        )
        return {
            "schema_version": 1,
            "purpose": "calibration-only whole-projection precision selection; held-out regression required",
            "environment": {
                "platform": platform.platform(),
                "native_build": engine.build_info(),
                **source_provenance(),
            },
            "source_model": model_provenance(source),
            "configuration": {
                "tokenizer": tokenizer_name,
                "output_tokens": output_tokens,
                "positions": positions,
                "probe_indices": probe_indices,
                "quantized_fraction_candidates": fractions,
                "minimum_fraction": 0.25,
                "calibration_prompts": prompts,
                "held_out_used_for": "overlap guard only",
                "calibration_corpus_sha256": canonical_sha256(prompts),
                "held_out_guard_sha256": canonical_sha256(held_out),
            },
            "ranking": ranked,
            "trials": trials,
            "policy": policy,
            "cache_reclaimed": cache_reclaimed(engine),
            "limitations": [
                "Single-matrix sensitivities ignore interactions; the combined policy is tested separately.",
                "25% floor is a scope guard, not a promise of memory savings or quality.",
                "Offline reconstructed FP16 copies are allowed here only, never retained by the runtime.",
                "No calibration success is a held-out or task-quality certification.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--prompts", type=Path, default=Path("benchmarks/calibration-prompts.json")
    )
    parser.add_argument(
        "--held-out", type=Path, default=Path("benchmarks/quality-prompts.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite calibration evidence: {args.output}"
        )
    result = calibrate(
        args.source,
        args.tokenizer,
        json.loads(args.prompts.read_text()),
        json.loads(args.held_out.read_text()),
        probe_count=args.probe_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["policy"], indent=2, sort_keys=True))
    if not result["policy"]["calibration_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
