"""Bounded per-output-channel scale fitting; existing W8A16 format and kernels.

Only offline source activation covariance influences fitting. The guaranteed
objective is the block-diagonal FP16-rounded reconstruction loss, NOT task
quality, exact greedy output, or speed. Ties retain the old recipe exactly.
"""

from __future__ import annotations

import math

import numpy as np

from .quantization import dequantize_per_channel, quantize_per_channel
from .second_order import (
    DEFAULT_CONFIG as BASE_CONFIG,
)
from .second_order import (
    quantize_second_order,
    validate_moments,
)
from .second_order import (
    validate_config as validate_base_config,
)

ALGORITHM = "scale_aware_block_second_order_v1"
POLICY_ALGORITHM = "scale_aware_cached_repair_v1"
DEFAULT_CONFIG = {
    **BASE_CONFIG,
    "scale_factors": [1.0, 0.995, 0.99, 0.98, 0.96, 0.92, 0.85],
    "row_chunk_size": 256,
}
DEFAULT_SEARCH = {
    "probe_cases": 4,
    "candidate_pool": 4,
    "repair_rounds": 2,
    "removal_pool": 2,
    "full_trial_count": 2,
}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("invalid scale-aware configuration")
    validate_base_config({k: config[k] for k in BASE_CONFIG})
    factors = config["scale_factors"]
    if (
        not isinstance(factors, list)
        or not 1 <= len(factors) <= 16
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0.5 <= v <= 1
            for v in factors
        )
        or factors[0] != 1.0
        or any(a <= b for a, b in zip(factors, factors[1:]))
        or type(config["row_chunk_size"]) is not int
        or not 1 <= config["row_chunk_size"] <= 4096
    ):
        raise ValueError("invalid scale-aware configuration")


def row_objective(weight, packed, scales, moments, block_size, row_chunk_size):
    """Actual FP16-rounded quadratic loss, bounded temporary rows per block."""
    losses = np.zeros(len(weight), dtype=np.float64)
    for begin in range(0, len(weight), row_chunk_size):
        end = min(begin + row_chunk_size, len(weight))
        for index, start in enumerate(range(0, weight.shape[1], block_size)):
            stop = min(start + block_size, weight.shape[1])
            h = moments[index, : stop - start, : stop - start]
            expected = weight[begin:end, start:stop].astype(np.float64)
            actual = dequantize_per_channel(
                packed[begin:end, start:stop], scales[begin:end]
            ).astype(np.float64)
            error = expected - actual
            losses[begin:end] += np.maximum(np.sum((error @ h) * error, axis=1), 0)
    if not np.all(np.isfinite(losses)):
        raise ValueError("nonfinite scale-aware reconstruction objective")
    return losses


def quantize_scale_aware(weight, moments, config=None):
    """Search weighted RTN scales, compensate once, then fallback per whole row.

    Baseline is the old block-compensated recipe on THESE SAME source moments.
    Scales are chosen by a bounded grid of activation-weighted RTN objectives.
    A final full-row comparison with the compensated baseline prevents scale
    changes from degrading that baseline objective. No mixed-scale row/block
    merging is permitted: one FP32 scale describes each complete output row.
    """
    config = DEFAULT_CONFIG if config is None else config
    validate_config(config)
    weight = np.asarray(weight)
    if weight.dtype != np.float16:
        raise ValueError("scale-aware fitting requires FP16 source weights")
    base = {k: config[k] for k in BASE_CONFIG}
    rtn, original_scales = quantize_per_channel(weight)
    validate_moments(moments, weight.shape[1], config["block_size"])
    old_q, old_scales, old_metrics = quantize_second_order(weight, moments, base)
    args = moments, config["block_size"], config["row_chunk_size"]
    old_loss = row_objective(weight, old_q, old_scales, *args)
    rtn_loss = row_objective(weight, rtn, original_scales, *args)
    best_loss, best_scales = rtn_loss.copy(), original_scales.copy()
    zero_rows = np.all(weight == 0, axis=1)
    for factor in config["scale_factors"][1:]:
        scales = (original_scales * np.float32(factor)).astype("<f4")
        scales[zero_rows] = np.float32(1)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("scale-aware scale is not representable in FP32")
        packed = np.clip(
            np.rint(weight.astype(np.float32) / scales[:, None]), -127, 127
        ).astype(np.int8)
        losses = row_objective(weight, packed, scales, *args)
        better = losses < best_loss
        best_loss[better], best_scales[better] = losses[better], scales[better]
    candidate_q, candidate_scales, candidate_metrics = quantize_second_order(
        weight, moments, base, scales=best_scales
    )
    candidate_loss = row_objective(weight, candidate_q, candidate_scales, *args)
    accepted = candidate_loss < old_loss
    packed, scales = old_q.copy(), old_scales.copy()
    packed[accepted], scales[accepted] = (
        candidate_q[accepted],
        candidate_scales[accepted],
    )
    final_loss = row_objective(weight, packed, scales, *args)
    if np.any(final_loss > old_loss):
        raise RuntimeError("scale-aware fallback violated the reconstruction bound")
    return (
        packed,
        scales,
        {
            "rtn_block_objective": float(rtn_loss.sum()),
            "fixed_scale_block_objective": float(old_loss.sum()),
            "calibrated_block_objective": float(final_loss.sum()),
            "relative_block_objective": old_metrics["relative_block_objective"]
            * float(final_loss.sum())
            / max(float(old_loss.sum()), 1e-30),
            "changed_quantized_elements": int(np.count_nonzero(packed != rtn)),
            "changed_scale_rows": int(np.count_nonzero(scales != original_scales)),
            "accepted_candidate_rows": int(np.count_nonzero(accepted)),
            "baseline_fallback_rows": int(np.count_nonzero(~accepted)),
            "scale_grid_candidates": len(config["scale_factors"]),
            "fixed_scale_fallback_row_blocks": old_metrics["rtn_fallback_row_blocks"],
            "candidate_fallback_row_blocks": candidate_metrics[
                "rtn_fallback_row_blocks"
            ],
            "no_worse_fixed_scale_row_objective": bool(np.all(final_loss <= old_loss)),
        },
    )
