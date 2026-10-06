"""Calibration-only cached decode coverage and bounded joint INT8 policy repair."""

from __future__ import annotations

import argparse
import json
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .benchmark import model_provenance, source_provenance
from .cached_decode import replay_cached
from .cached_policy import ALGORITHM, DEFAULT_SEARCH, select_and_repair, validate_search
from .calibrate_int8 import cache_reclaimed, replaced_weights
from .calibrate_second_order import projection_groups
from .mlx_engine import MlxEngine
from .model_file import ModelFile
from .precision_policy import canonical_sha256, seal_policy, validate_corpora
from .quantization import dequantize_per_channel
from .second_order import (
    ALGORITHM as STATS_ALGORITHM,
    DEFAULT_CONFIG,
    block_moments,
    quantize_second_order,
    validate_config,
    write_calibration_stats,
)
from .validate_int8 import compare_logits


@contextmanager
def capture_cached_inputs(model, groups: dict, rows_per_call: int):
    """One group observation per call, including every cached decode position."""
    if type(rows_per_call) is not int or rows_per_call <= 0:
        raise ValueError("activation sample count must be positive")
    original = model._linear
    first_names = {names[0]: key for key, names in groups.items()}
    samples = {key: [] for key in groups}

    def captured(inputs, name, *args, **kwargs):
        if name in first_names:
            indices = np.linspace(
                0, inputs.shape[0] - 1, min(rows_per_call, inputs.shape[0]), dtype=int
            )
            samples[first_names[name]].append(
                np.array(inputs[model.mx.array(indices.tolist())], dtype=np.float64)
            )
        return original(inputs, name, *args, **kwargs)

    model._linear = captured
    try:
        yield samples
    finally:
        model._linear = original


def cached_metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    """Vectorized summaries, without expensive whole-vocabulary sorting in search."""
    if reference.ndim != 2 or reference.shape != actual.shape or not reference.size:
        raise ValueError("cached logit arrays must have equal nonempty matrix shapes")
    expected, values = reference.astype(np.float64), actual.astype(np.float64)
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite cached logits")
    norm2 = np.einsum("ij,ij->i", expected, expected)
    other2 = np.einsum("ij,ij->i", values, values)
    denominator = np.sqrt(norm2 * other2)
    dots = np.einsum("ij,ij->i", expected, values)
    cosine = np.divide(
        dots, denominator, out=np.zeros_like(dots), where=denominator != 0
    )
    zero = denominator == 0
    cosine[zero] = np.all(expected[zero] == values[zero], axis=1)
    error = expected - values
    error2 = np.einsum("ij,ij->i", error, error)
    relative2 = np.divide(error2, norm2, out=error2.copy(), where=norm2 != 0)
    return {
        "observations": len(reference),
        "top1_changes": int(
            np.count_nonzero(np.argmax(reference, axis=1) != np.argmax(actual, axis=1))
        ),
        "minimum_logit_cosine": float(np.min(cosine)),
        "mean_squared_relative_error": float(np.mean(relative2)),
    }


def calibrate(
    source: Path,
    tokenizer_name: str,
    prompts: list[str],
    held_out: list[str],
    stats_path: Path,
    *,
    output_tokens: int = 32,
    rows_per_call: int = 8,
    config: dict | None = None,
    search_config: dict | None = None,
    weight_method: str = "fixed-scale",
) -> dict:
    quantizer = quantize_second_order
    stats_algorithm, policy_algorithm = STATS_ALGORITHM, ALGORITHM
    default_config, default_search = DEFAULT_CONFIG, DEFAULT_SEARCH
    validate_quantizer = validate_config
    if weight_method == "coordinate-refined":
        from . import refined

        quantizer = refined.quantize_refined
        stats_algorithm, policy_algorithm = refined.ALGORITHM, refined.POLICY_ALGORITHM
        default_config, default_search = refined.DEFAULT_CONFIG, refined.DEFAULT_SEARCH
        validate_quantizer = refined.validate_config
    elif weight_method == "scale-aware":
        from . import scale_aware

        quantizer = scale_aware.quantize_scale_aware
        stats_algorithm, policy_algorithm = (
            scale_aware.ALGORITHM,
            scale_aware.POLICY_ALGORITHM,
        )
        default_config, default_search = (
            scale_aware.DEFAULT_CONFIG,
            scale_aware.DEFAULT_SEARCH,
        )
        validate_quantizer = scale_aware.validate_config
    elif weight_method != "fixed-scale":
        raise ValueError("unsupported cached calibration weight method")
    config = dict(default_config if config is None else config)
    search_config = dict(default_search if search_config is None else search_config)
    validate_quantizer(config)
    validate_search(search_config)
    validate_corpora(prompts, held_out)
    if type(output_tokens) is not int or not 1 <= output_tokens <= 128:
        raise ValueError("cached calibration output count must be in 1..128")
    if type(rows_per_call) is not int or rows_per_call <= 0:
        raise ValueError("activation sample count must be positive")
    if stats_path.exists():
        raise FileExistsError(stats_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    input_ids = [tokenizer(p, add_special_tokens=True).input_ids for p in prompts]
    held_ids = {
        tuple(tokenizer(p, add_special_tokens=True).input_ids) for p in held_out
    }
    if len({tuple(ids) for ids in input_ids}) != len(input_ids):
        raise ValueError("duplicate tokenized calibration prompts")
    if any(tuple(ids) in held_ids for ids in input_ids):
        raise ValueError("tokenized calibration and held-out prompts overlap")
    with ModelFile(source) as metadata:
        limit = min(2048, metadata.config.max_position_embeddings)
    if any(not ids or len(ids) + output_tokens > limit for ids in input_ids):
        raise ValueError("cached calibration context exceeds supported limit")
    with MlxEngine(
        source, max_model_length=limit, kv_cache_bytes=512 << 20, decode_mode="rowwise"
    ) as engine:
        model, mx, file = engine.model, engine.model.mx, engine.model.file
        if file.quantization:
            raise ValueError("cached calibration requires an FP16 source")
        groups = projection_groups(file.config.num_hidden_layers)
        mapping = {n: key for key, names in groups.items() for n in names}
        if set(mapping) != {n for n in file.tensors if n.endswith("_proj.weight")}:
            raise ValueError("unsupported projection grouping")
        storage = [
            (file.tensor_info(n).offset, file.tensor_info(n).shape) for n in mapping
        ]
        if len(set(storage)) != len(storage):
            raise ValueError(
                "projection aliases are unsupported for calibrated byte accounting"
            )
        goldens, moments, row_counts = [], {}, {}
        for index, (prompt, ids) in enumerate(zip(prompts, input_ids)):
            tokens = engine.generate(ids, output_tokens, eos_token_ids=[])
            with capture_cached_inputs(model, groups, rows_per_call) as samples:
                replay = replay_cached(engine, ids, tokens)
            if replay["greedy_tokens"] != tokens:
                raise RuntimeError(
                    "FP16 cached teacher forcing differs from scheduler generation"
                )
            for key, chunks in samples.items():
                if not chunks:
                    raise RuntimeError("not every projection group was observed")
                values = np.concatenate(chunks)
                contribution = block_moments(values, config["block_size"]) * len(values)
                if key not in moments:
                    moments[key], row_counts[key] = contribution, len(values)
                else:
                    moments[key] += contribution
                    row_counts[key] += len(values)
            goldens.append(
                {
                    "prompt": prompt,
                    "input_ids": ids,
                    "tokens": tokens,
                    "logits": replay["logits"],
                }
            )
            print(
                json.dumps({"cached_reference": index + 1, "total": len(prompts)}),
                flush=True,
            )
        for key in moments:
            moments[key] /= row_counts[key]
        stats_sha = write_calibration_stats(
            stats_path,
            {
                "schema_version": 1,
                "algorithm": stats_algorithm,
                "source_data_sha256": file.data_sha256,
                "quantizer_config": config,
                "calibration_corpus_sha256": canonical_sha256(prompts),
                "calibration_token_ids_sha256": canonical_sha256(input_ids),
                "weight_to_moments": mapping,
                "sampled_rows": row_counts,
                "activation_source": "FP16 chunked prompt and every cached decode position; block-diagonal covariance",
            },
            moments,
        )
        reconstructed, local = {}, []
        for name in sorted(mapping):
            q, scales, objective = quantizer(
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
        probe_indices = sorted(
            set(
                np.linspace(
                    0,
                    len(prompts) - 1,
                    min(search_config["probe_cases"], len(prompts)),
                    dtype=int,
                ).tolist()
            )
        )
        evaluation_count = 0

        def evaluate(names, full):
            nonlocal evaluation_count
            indices = range(len(goldens)) if full else probe_indices
            observations = []
            with replaced_weights(model, {n: reconstructed[n] for n in names}):
                for i in indices:
                    g = goldens[i]
                    replay = replay_cached(engine, g["input_ids"], g["tokens"])
                    observations.append(cached_metrics(g["logits"], replay["logits"]))
            count = sum(o["observations"] for o in observations)
            result = {
                "observations": count,
                "top1_changes": sum(o["top1_changes"] for o in observations),
                "minimum_logit_cosine": min(
                    o["minimum_logit_cosine"] for o in observations
                ),
                "mean_squared_relative_error": sum(
                    o["mean_squared_relative_error"] * o["observations"]
                    for o in observations
                )
                / count,
            }
            evaluation_count += 1
            print(
                json.dumps(
                    {
                        "policy_evaluation": evaluation_count,
                        "full_calibration": full,
                        "quantized_matrices": len(names),
                        **result,
                    }
                ),
                flush=True,
            )
            return result

        search = select_and_repair(local, evaluate, search_config)
        selected, cases = set(search["quantized_weights"]), []
        with replaced_weights(model, {n: reconstructed[n] for n in selected}):
            for g in goldens:
                replay = replay_cached(engine, g["input_ids"], g["tokens"])
                actual = engine.generate(
                    g["input_ids"], output_tokens, eos_token_ids=[]
                )
                metrics = [
                    {"generated_position": p, **compare_logits(a, b)}
                    for p, (a, b) in enumerate(
                        zip(g["logits"], replay["logits"], strict=True)
                    )
                ]
                cases.append(
                    {
                        "prompt": g["prompt"],
                        "input_ids": g["input_ids"],
                        "fp16_tokens": g["tokens"],
                        "candidate_tokens": actual,
                        "teacher_forced_tokens": replay["greedy_tokens"],
                        "exact_greedy_parity": actual == g["tokens"],
                        "cached_teacher_forced_logits": metrics,
                        "private_cache_reclaimed": replay["cache_reclaimed"],
                    }
                )
        exact = sum(c["exact_greedy_parity"] for c in cases)
        minimum = min(
            m["cosine"] for c in cases for m in c["cached_teacher_forced_logits"]
        )
        reclaimed = cache_reclaimed(engine) and all(
            c["private_cache_reclaimed"] for c in cases
        )
        passed = (
            exact == len(cases)
            and search["summary"]["top1_changes"] == 0
            and minimum >= 0.999
            and reclaimed
        )
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": policy_algorithm,
                "source_data_sha256": file.data_sha256,
                "quantizer_config": config,
                "calibration_stats_sha256": stats_sha,
                "calibration_corpus_sha256": canonical_sha256(prompts),
                "held_out_guard_sha256": canonical_sha256(held_out),
                "retained_fp16": sorted(set(mapping) - selected),
                "quantized_weights": sorted(selected),
                "quantized_projection_fraction": search[
                    "quantized_projection_fraction"
                ],
                "calibration_passed": passed,
                "search_config": search_config,
                "execution_contract": "native FP16-rounded reconstruction; rowwise cached decode; chunked prefill; greedy argmax unchanged",
            }
        )
        return {
            "schema_version": 1,
            "purpose": __doc__,
            "environment": {
                "platform": platform.platform(),
                "native_build": engine.build_info(),
                **source_provenance(),
            },
            "source_model": model_provenance(source),
            "configuration": {
                "tokenizer": tokenizer_name,
                "output_tokens": output_tokens,
                "probe_case_indices": probe_indices,
                "positions": list(range(output_tokens)),
                "search_config": search_config,
                "rows_per_call": rows_per_call,
                "minimum_fraction": 0.25,
                "calibration_prompts": prompts,
                "held_out_used_for": "normalized/tokenized overlap rejection only",
                "stats_path": str(stats_path),
                "quantizer_config": config,
            },
            "local_objectives": local,
            "search": search,
            "summary": {
                "exact_greedy_cases": exact,
                "total_cases": len(cases),
                "minimum_logit_cosine": minimum,
                "calibration_passed": passed,
                "cached_top1_changes": search["summary"]["top1_changes"],
                "accepted_repairs": sum(r["accepted"] for r in search["repairs"]),
            },
            "cases": cases,
            "policy": policy,
            "cache_reclaimed": reclaimed,
            "limitations": [
                (
                    "Block-diagonal covariance and fixed RTN scales, not full GPTQ or AWQ."
                    if weight_method == "fixed-scale"
                    else (
                        "Bounded activation-weighted scale grid and block-diagonal covariance, not full GPTQ or AWQ; reconstruction bounds are not quality guarantees."
                        if weight_method == "scale-aware"
                        else "Bounded integer-coordinate descent with fixed fitted scales and block-diagonal covariance; not full GPTQ/AWQ or a language-quality guarantee."
                    )
                ),
                "Bounded shortlists/single-matrix swaps may miss better policies or stop at a local minimum.",
                "Calibration uses source activations, not sequential quantized-layer activation recollection.",
                "Exact calibration decisions do not certify unseen prompts, near ties, or direct-kernel reductions.",
                "Offline dense trial weights are temporary and never change the packed runtime format or FP16 defaults.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/cached-calibration-prompts.json"),
    )
    parser.add_argument("--held-out", type=Path, nargs="+", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-method",
        choices=["fixed-scale", "scale-aware", "coordinate-refined"],
        default="fixed-scale",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = calibrate(
        args.source,
        args.tokenizer,
        json.loads(args.prompts.read_text()),
        [p for file in args.held_out for p in json.loads(file.read_text())],
        args.stats,
        weight_method=args.weight_method,
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
