"""Portable, source-bound mixed-precision selection contracts (no MLX imports)."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def validate_corpora(calibration: list[str], held_out: list[str]) -> None:
    """Fail closed on overlap after Unicode/case/whitespace normalization.

    This guards direct leakage, not semantic similarity or benchmark overfitting.
    Neither the held-out text nor its outputs participate in precision ranking.
    """

    def normalized(prompts):
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("corpora must be nonempty lists of prompt strings")
        if any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("every corpus prompt must be a nonempty string")
        values = [
            " ".join(unicodedata.normalize("NFKC", p).casefold().split())
            for p in prompts
        ]
        if len(set(values)) != len(values):
            raise ValueError("duplicate normalized prompts in corpus")
        return set(values)

    if normalized(calibration) & normalized(held_out):
        raise ValueError("calibration and held-out corpora overlap")


def rank_sensitivities(observations: list[dict]) -> list[dict]:
    """Greedy heuristic: top-1 changes first, then squared error per saved byte.

    Single-projection ablations do not model interactions and are not GPTQ/AWQ.
    Selection must subsequently be tested with all chosen perturbations together.
    """
    names = set()
    ranked = []
    for item in observations:
        name = item["weight"]
        size, changes, loss = (
            item["fp16_bytes"],
            item["top1_changes"],
            item["mean_squared_relative_logit_error"],
        )
        if (
            not isinstance(name, str)
            or name in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(changes, int)
            or isinstance(changes, bool)
            or changes < 0
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("invalid sensitivity observation")
        names.add(name)
        ranked.append({**item, "error_per_fp16_byte": loss / size})
    if not ranked:
        raise ValueError("no sensitivity observations")
    return sorted(
        ranked,
        key=lambda item: (
            -item["top1_changes"],
            -item["error_per_fp16_byte"],
            item["weight"],
        ),
    )


def retained_for_fraction(ranked: list[dict], fraction: float) -> list[str]:
    """Retain sensitive matrices without crossing the INT8-byte fraction floor.

    Whole-matrix granularity may leave more than the requested fraction in INT8.
    The fraction refers to eligible projection FP16 bytes, not total model bytes.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("quantized projection fraction must be in (0, 1]")
    total = remaining = sum(item["fp16_bytes"] for item in ranked)
    if total <= 0:
        raise ValueError("no projection bytes")
    retained = []
    for item in ranked:
        if remaining - item["fp16_bytes"] >= fraction * total:
            retained.append(item["weight"])
            remaining -= item["fp16_bytes"]
    return sorted(retained)


def seal_policy(policy: dict) -> dict:
    payload = {k: v for k, v in policy.items() if k != "policy_sha256"}
    return {**payload, "policy_sha256": canonical_sha256(payload)}


def validate_policy(
    policy: dict, source_sha: str, projection_names: set[str]
) -> list[str]:
    """Validate integrity and identity. A policy checksum is NOT a quality certificate."""
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("unsupported precision policy")
    if policy.get("algorithm") not in {
        "single_projection_logit_ablation_v1",
        "block_second_order_joint_forward_v1",
        "block_second_order_cached_repair_v1",
    }:
        raise ValueError("unsupported precision selection algorithm")
    if policy["algorithm"] in {
        "block_second_order_joint_forward_v1",
        "block_second_order_cached_repair_v1",
    }:
        from .second_order import validate_config

        validate_config(policy.get("quantizer_config"))
        checksum = policy.get("calibration_stats_sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ValueError("invalid calibration statistics checksum")
        if policy["algorithm"] in {
            "block_second_order_cached_repair_v1",
            }:
            from .cached_policy import validate_search

            validate_search(policy.get("search_config"))
    if policy.get("policy_sha256") != seal_policy(policy)["policy_sha256"]:
        raise ValueError("precision policy checksum mismatch")
    if policy.get("source_data_sha256") != source_sha:
        raise ValueError("precision policy belongs to a different source artifact")
    retained, quantized = policy.get("retained_fp16"), policy.get("quantized_weights")
    for names in [retained, quantized]:
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("invalid precision policy tensor list")
    if (
        set(retained) & set(quantized)
        or set(retained) | set(quantized) != projection_names
        or not quantized
    ):
        raise ValueError("precision policy must partition supported projections")
    return retained
