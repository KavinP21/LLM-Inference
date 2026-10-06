"""Bounded integer-coordinate descent on FP16-rounded block reconstruction.

Offline fitting only: fixed row scales, existing signed W8 format and kernels.
This is a block-diagonal local-loss bound, not an exact-token or speed guarantee.
"""

from __future__ import annotations

import numpy as np

from .quantization import dequantize_per_channel, quantize_per_channel
from .scale_aware import (
    DEFAULT_CONFIG as SCALE_CONFIG,
)
from .scale_aware import (
    DEFAULT_SEARCH as DEFAULT_SEARCH,
)
from .scale_aware import (
    quantize_scale_aware,
    row_objective,
)
from .scale_aware import (
    validate_config as validate_scale_config,
)

ALGORITHM = "coordinate_refined_scale_aware_v1"
POLICY_ALGORITHM = "coordinate_refined_cached_repair_v1"
DEFAULT_CONFIG = {**SCALE_CONFIG, "refinement_sweeps": 2}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("invalid refined quantization configuration")
    validate_scale_config({k: config[k] for k in SCALE_CONFIG})
    if (
        type(config["refinement_sweeps"]) is not int
        or not 1 <= config["refinement_sweeps"] <= 4
    ):
        raise ValueError("invalid refinement sweep configuration")


def quantize_refined(weight, moments, config=None):
    """Two-neighbor integer descent, with complete block/row fallback.

    e = W - FP16(q*s). A candidate reconstruction change d in column j changes
    its quadratic by -2*d*(e@H)[j] + d*d*H[j,j]. The actual FP16 candidate value
    determines d; a nominal exact scale is not substituted. Each sweep visits
    stable activation order once. Strict improvements only, decrement wins exact
    neighbor ties, and unchanged values win zero-loss ties. Never wrap int8.

    FP64 incremental scores rank proposals; freshly recomputed complete block
    losses authorize changes at every sweep. A final whole-row fallback guards
    against accumulated rounding error. Row scales never change in refinement.
    """
    config = DEFAULT_CONFIG if config is None else config
    validate_config(config)
    scale_config = {k: config[k] for k in SCALE_CONFIG}
    baseline_q, scales, baseline_metrics = quantize_scale_aware(
        weight, moments, scale_config
    )
    weight = np.asarray(weight)
    packed = baseline_q.copy()
    block, chunk = config["block_size"], config["row_chunk_size"]
    moves = sweep_fallbacks = 0
    for index, start in enumerate(range(0, weight.shape[1], block)):
        stop = min(start + block, weight.shape[1])
        h = moments[index, : stop - start, : stop - start]
        order = (
            np.argsort(-np.diag(h), kind="stable")
            if config["activation_order"]
            else np.arange(len(h))
        )
        for begin in range(0, len(weight), chunk):
            end = min(begin + chunk, len(weight))
            w = weight[begin:end, start:stop].astype(np.float64)
            q = packed[begin:end, start:stop].copy()
            row_scales = scales[begin:end]
            for _ in range(config["refinement_sweeps"]):
                before = q.copy()
                dense = dequantize_per_channel(q, row_scales).astype(np.float64)
                error = w - dense
                product = error @ h
                before_loss = np.maximum(np.sum(product * error, axis=1), 0)
                sweep_moves = 0
                for column in order:
                    current = q[:, column].astype(np.int16)
                    minus = np.maximum(current - 1, -127).astype(np.int8)
                    plus = np.minimum(current + 1, 127).astype(np.int8)
                    values = [
                        dequantize_per_channel(candidate[:, None], row_scales)[
                            :, 0
                        ].astype(np.float64)
                        for candidate in (minus, plus)
                    ]
                    deltas = [value - dense[:, column] for value in values]
                    losses = [
                        -2 * d * product[:, column] + d * d * h[column, column]
                        for d in deltas
                    ]
                    choose_minus = (losses[0] < 0) & (losses[0] <= losses[1])
                    choose_plus = (losses[1] < 0) & (losses[1] < losses[0])
                    changed = choose_minus | choose_plus
                    if not np.any(changed):
                        continue
                    d = np.where(
                        choose_minus, deltas[0], np.where(choose_plus, deltas[1], 0)
                    )
                    q[:, column] = np.where(
                        choose_minus, minus, np.where(choose_plus, plus, current)
                    ).astype(np.int8)
                    dense[:, column] += d
                    product -= d[:, None] * h[column, :]
                    sweep_moves += int(np.count_nonzero(changed))
                error = w - dequantize_per_channel(q, row_scales).astype(np.float64)
                after_loss = np.maximum(np.sum((error @ h) * error, axis=1), 0)
                accept = after_loss < before_loss
                sweep_fallbacks += int(
                    np.count_nonzero(~accept & np.any(q != before, axis=1))
                )
                q[~accept] = before[~accept]
                moves += sweep_moves
                # No early-stopping threshold or extra sweeps after inspecting results.
            packed[begin:end, start:stop] = q
    args = moments, block, chunk
    baseline_loss = row_objective(weight, baseline_q, scales, *args)
    candidate_loss = row_objective(weight, packed, scales, *args)
    accept = candidate_loss < baseline_loss
    packed[~accept] = baseline_q[~accept]
    actual_loss = row_objective(weight, packed, scales, *args)
    if np.any(actual_loss > baseline_loss):
        raise RuntimeError("refinement violated its scale-aware row bound")
    old_total, new_total = float(baseline_loss.sum()), float(actual_loss.sum())
    return (
        packed,
        scales,
        {
            **baseline_metrics,
            "scale_aware_block_objective": old_total,
            "calibrated_block_objective": new_total,
            "relative_block_objective": baseline_metrics["relative_block_objective"]
            * new_total
            / max(old_total, 1e-30),
            "refinement_sweeps": config["refinement_sweeps"],
            "proposed_coordinate_moves": moves,
            "sweep_row_block_fallbacks": sweep_fallbacks,
            "refinement_changed_elements": int(np.count_nonzero(packed != baseline_q)),
            "changed_quantized_elements": int(
                np.count_nonzero(packed != quantize_per_channel(weight)[0])
            ),
            "refinement_accepted_rows": int(np.count_nonzero(accept)),
            "no_worse_scale_aware_row_objective": bool(
                np.all(actual_loss <= baseline_loss)
            ),
        },
    )
