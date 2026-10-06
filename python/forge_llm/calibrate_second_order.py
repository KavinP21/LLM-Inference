"""Calibration-only covariance fitting and interaction-aware precision selection.

Held-out prompts are used solely for overlap rejection. Policies and covariance
archives are frozen before any regression output is inspected. All fitting is
offline; no dense reconstructed copy is retained by the inference runtime.
"""

from __future__ import annotations

import argparse
import json
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .benchmark import model_provenance, source_provenance
from .calibrate_int8 import cache_reclaimed, replaced_weights
from .mlx_engine import MlxEngine
from .model_file import ModelFile
from .precision_policy import canonical_sha256, seal_policy, validate_corpora
from .quantization import dequantize_per_channel
from .second_order import (
    ALGORITHM,
    DEFAULT_CONFIG,
    block_moments,
    quantize_second_order,
    validate_config,
    write_calibration_stats,
)
from .validate_int8 import compare_logits


def projection_groups(layer_count: int) -> dict[str, list[str]]:
    groups = {}
    for layer in range(layer_count):
        prefix = f"model.layers.{layer}."
        for group, suffixes in [
            ("qkv", ["self_attn.q", "self_attn.k", "self_attn.v"]),
            ("attention_out", ["self_attn.o"]),
            ("gate_up", ["mlp.gate", "mlp.up"]),
            ("down", ["mlp.down"]),
        ]:
            groups[f"cov_{layer}_{group}"] = [
                prefix + s + "_proj.weight" for s in suffixes
            ]
    return groups


@contextmanager
def capture_inputs(model, groups: dict, rows_per_prompt: int, accumulate):
    """Capture one input per shared group without changing execution or ownership."""
    original = model._linear
    first_names = {names[0]: key for key, names in groups.items()}
    seen = set()

    def captured(inputs, name, *args, **kwargs):
        if name in first_names:
            key = first_names[name]
            if key in seen:
                raise RuntimeError(
                    "calibration requires one unchunked prefill per prompt"
                )
            seen.add(key)
            indices = np.linspace(
                0, inputs.shape[0] - 1, min(rows_per_prompt, inputs.shape[0]), dtype=int
            )
            # Materialize only sampled rows, not full sequence activations on CPU.
            selected = np.asarray(inputs[model.mx.array(indices.tolist())]).astype(
                np.float64
            )
            accumulate(key, selected)
        return original(inputs, name, *args, **kwargs)

    model._linear = captured
    try:
        yield seen
    finally:
        model._linear = original


def joint_score(metrics: list[dict], quantized_bytes: int) -> tuple:
    """Lexicographic combined top-1 changes then squared logit error per byte."""
    if quantized_bytes <= 0 or not metrics:
        raise ValueError("joint scoring needs observations and positive savings")
    return (
        sum(not m["argmax_equal"] for m in metrics),
        float(np.mean([m["relative_l2_error"] ** 2 for m in metrics]))
        / quantized_bytes,
    )


def calibrate(
    source: Path,
    tokenizer_name: str,
    prompts: list[str],
    held_out: list[str],
    stats_path: Path,
    *,
    output_tokens: int = 32,
    positions: tuple[int, ...] = (0, 7, 31),
    probe_count: int = 8,
    candidate_pool: int = 12,
    rows_per_prompt: int = 8,
    config: dict | None = None,
) -> dict:
    config = dict(DEFAULT_CONFIG if config is None else config)
    validate_config(config)
    validate_corpora(prompts, held_out)
    if (
        output_tokens <= 0
        or not positions
        or min(positions) < 0
        or max(positions) >= output_tokens
        or probe_count <= 0
        or candidate_pool <= 0
        or rows_per_prompt <= 0
    ):
        raise ValueError("invalid second-order calibration workload")
    if stats_path.exists():
        raise FileExistsError(
            f"refusing to overwrite calibration statistics: {stats_path}"
        )
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
        source,
        max_model_length=context_limit,
        kv_cache_bytes=512 << 20,
        decode_mode="rowwise",
    ) as engine:
        model, mx, file = engine.model, engine.model.mx, engine.model.file
        if file.quantization:
            raise ValueError("calibration requires an FP16 source")
        groups = projection_groups(file.config.num_hidden_layers)
        mapping = {n: key for key, names in groups.items() for n in names}
        if set(mapping) != {n for n in file.tensors if n.endswith("_proj.weight")}:
            raise ValueError("unsupported projection grouping")
        goldens, moments, row_counts = [], {}, {}

        def accumulate(key, values):
            count = len(values)
            contribution = block_moments(values, config["block_size"]) * count
            if key not in moments:
                moments[key] = contribution
                row_counts[key] = count
            else:
                moments[key] += contribution
                row_counts[key] += count

        for index, (prompt, ids) in enumerate(zip(prompts, input_ids)):
            tokens = engine.generate(ids, output_tokens, eos_token_ids=[])
            contexts = [ids + tokens[:position] for position in positions]
            logits = [np.array(engine.debug_prefill_logits(c)) for c in contexts]
            with capture_inputs(model, groups, rows_per_prompt, accumulate) as seen:
                engine.debug_prefill_logits(ids + tokens[:-1])
            if seen != set(groups):
                raise RuntimeError("not every projection activation was observed")
            goldens.append(
                {
                    "prompt": prompt,
                    "input_ids": ids,
                    "tokens": tokens,
                    "contexts": contexts,
                    "logits": logits,
                }
            )
            print(
                json.dumps({"calibration_reference": index + 1, "total": len(prompts)}),
                flush=True,
            )
        for key in moments:
            moments[key] /= row_counts[key]
        stats_metadata = {
            "schema_version": 1,
            "algorithm": ALGORITHM,
            "source_data_sha256": file.data_sha256,
            "quantizer_config": config,
            "calibration_corpus_sha256": canonical_sha256(prompts),
            "calibration_token_ids_sha256": canonical_sha256(input_ids),
            "weight_to_moments": mapping,
            "sampled_rows": row_counts,
            "activation_source": "FP16 source; cross-block covariance discarded",
        }
        stats_sha = write_calibration_stats(stats_path, stats_metadata, moments)
        reconstructed, local = {}, []
        for name in sorted(mapping):
            q, scales, objective = quantize_second_order(
                file.tensor_numpy(name), moments[mapping[name]], config
            )
            reconstructed[name] = mx.array(dequantize_per_channel(q, scales))
            mx.eval(reconstructed[name])
            local.append(
                {
                    "weight": name,
                    "fp16_bytes": file.tensor_info(name).nbytes,
                    **objective,
                }
            )
        # Local covariance ranks only the bounded candidate pool. Actual selection
        # scores each candidate together with EVERY already selected perturbation.
        local.sort(
            key=lambda o: (o["relative_block_objective"] / o["fp16_bytes"], o["weight"])
        )
        all_probes = [
            (c, logits)
            for g in goldens
            for c, logits in zip(g["contexts"], g["logits"])
        ]
        probe_indices = sorted(
            set(
                np.linspace(
                    0, len(all_probes) - 1, min(probe_count, len(all_probes)), dtype=int
                ).tolist()
            )
        )
        probes = [all_probes[i] for i in probe_indices]
        selected, selection = set(), []
        total_bytes = sum(o["fp16_bytes"] for o in local)
        selected_bytes = 0
        while selected_bytes < total_bytes * 0.25:
            candidates = [o for o in local if o["weight"] not in selected][
                :candidate_pool
            ]
            trials = []
            for item in candidates:
                combined = selected | {item["weight"]}
                with replaced_weights(model, {n: reconstructed[n] for n in combined}):
                    metrics = [
                        compare_logits(
                            expected, np.array(engine.debug_prefill_logits(context))
                        )
                        for context, expected in probes
                    ]
                score = joint_score(metrics, selected_bytes + item["fp16_bytes"])
                trials.append(
                    {
                        "weight": item["weight"],
                        "score": list(score),
                        "minimum_logit_cosine": min(m["cosine"] for m in metrics),
                    }
                )
            best = min(trials, key=lambda t: (*t["score"], t["weight"]))
            selected.add(best["weight"])
            selected_bytes += file.tensor_info(best["weight"]).nbytes
            step = {
                "step": len(selected),
                "selected": best["weight"],
                "quantized_projection_fraction": selected_bytes / total_bytes,
                "trials": trials,
            }
            selection.append(step)
            print(
                json.dumps({k: v for k, v in step.items() if k != "trials"}), flush=True
            )
        with replaced_weights(model, {n: reconstructed[n] for n in selected}):
            cases = []
            for golden in goldens:
                actual = engine.generate(
                    golden["input_ids"], output_tokens, eos_token_ids=[]
                )
                metrics = [
                    compare_logits(
                        expected, np.array(engine.debug_prefill_logits(context))
                    )
                    for context, expected in zip(golden["contexts"], golden["logits"])
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
        minimum = min(m["cosine"] for c in cases for m in c["teacher_forced_logits"])
        exact = sum(c["exact_greedy_parity"] for c in cases)
        reclaimed = cache_reclaimed(engine)
        passed = exact == len(cases) and minimum >= 0.999 and reclaimed
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": ALGORITHM,
                "source_data_sha256": file.data_sha256,
                "quantizer_config": config,
                "calibration_stats_sha256": stats_sha,
                "calibration_corpus_sha256": canonical_sha256(prompts),
                "held_out_guard_sha256": canonical_sha256(held_out),
                "retained_fp16": sorted(set(mapping) - selected),
                "quantized_weights": sorted(selected),
                "quantized_projection_fraction": selected_bytes / total_bytes,
                "calibration_passed": passed,
                "execution_contract": "FP16-rounded reconstruction; rowwise decode; native MLX prefill GEMM",
            }
        )
        return {
            "schema_version": 1,
            "purpose": "calibration-only block-diagonal error compensation and joint precision selection",
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
                "candidate_pool": candidate_pool,
                "rows_per_prompt": rows_per_prompt,
                "minimum_fraction": 0.25,
                "calibration_prompts": prompts,
                "held_out_used_for": "overlap guard only",
                "stats_path": str(stats_path),
                "quantizer_config": config,
            },
            "local_objectives": local,
            "joint_selection": selection,
            "summary": {
                "exact_greedy_cases": exact,
                "total_cases": len(cases),
                "minimum_logit_cosine": minimum,
                "calibration_passed": passed,
            },
            "cases": cases,
            "policy": policy,
            "cache_reclaimed": reclaimed,
            "limitations": [
                "Not full GPTQ: fixed scales, block-diagonal covariance, FP16 source activations.",
                "Bounded greedy shortlist can miss better projection combinations.",
                "Teacher-forced prefill logits are a selection proxy; cached greedy generation is checked separately.",
                "Local objective improvement and calibration parity do not certify held-out quality.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/second-order-calibration-prompts.json"),
    )
    parser.add_argument(
        "--held-out",
        type=Path,
        nargs="+",
        default=[
            Path("benchmarks/quality-prompts.json"),
            Path("benchmarks/second-order-regression-prompts.json"),
        ],
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite calibration report: {args.output}"
        )
    held_out = [p for file in args.held_out for p in json.loads(file.read_text())]
    result = calibrate(
        args.source,
        args.tokenizer,
        json.loads(args.prompts.read_text()),
        held_out,
        args.stats,
        probe_count=args.probe_count,
        candidate_pool=args.candidate_pool,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
    if not result["policy"]["calibration_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
