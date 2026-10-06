"""Portable bounded joint search; evaluators see calibration data only."""

from __future__ import annotations

import math

ALGORITHM = "block_second_order_cached_repair_v1"
DEFAULT_SEARCH = {
    "probe_cases": 4,
    "candidate_pool": 8,
    "repair_rounds": 3,
    "removal_pool": 4,
    "full_trial_count": 4,
}


def validate_search(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_SEARCH):
        raise ValueError("invalid cached search configuration")
    if any(type(v) is not int or not 1 <= v <= 32 for v in config.values()):
        raise ValueError("cached search bounds must be integers in 1..32")


def score(summary: dict) -> tuple:
    changes, total = summary["top1_changes"], summary["observations"]
    cosine, loss = (
        summary["minimum_logit_cosine"],
        summary["mean_squared_relative_error"],
    )
    if (
        type(changes) is not int
        or type(total) is not int
        or not 0 <= changes <= total
        or total <= 0
        or not math.isfinite(cosine)
        or not -1.000001 <= cosine <= 1.000001
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise ValueError("invalid cached policy observations")
    return changes, max(0.0, 0.999 - cosine), loss


def select_and_repair(local: list[dict], evaluate, config: dict | None = None) -> dict:
    """Forward shortlist search, then strictly improving full-corpus swaps.

    evaluate(frozenset(names), full: bool) returns an aggregated cached-logit
    summary. Probe rankings never authorize a repair: every accepted swap must
    strictly improve the full calibration score. No held-out callback exists.
    Whole-matrix swaps keep the 25% eligible-byte floor throughout repair.
    """
    config = dict(DEFAULT_SEARCH if config is None else config)
    validate_search(config)
    by_name = {o["weight"]: o for o in local}
    if (
        not local
        or len(by_name) != len(local)
        or any(not isinstance(n, str) or not n for n in by_name)
        or any(
            type(o["fp16_bytes"]) is not int
            or o["fp16_bytes"] <= 0
            or not math.isfinite(o["relative_block_objective"])
            or o["relative_block_objective"] < 0
            for o in local
        )
    ):
        raise ValueError("invalid cached-search projection inventory")
    ordered = sorted(
        by_name,
        key=lambda n: (
            by_name[n]["relative_block_objective"] / by_name[n]["fp16_bytes"],
            n,
        ),
    )
    total = sum(o["fp16_bytes"] for o in local)

    def size(names):
        return sum(by_name[n]["fp16_bytes"] for n in names)

    cache, evaluations = {}, []

    def trial(names, full):
        names = frozenset(names)
        key = names, full
        if key not in cache:
            summary = dict(evaluate(names, full))
            score(summary)  # reject NaN/invalid counts before choosing a policy
            cache[key] = summary
            evaluations.append(
                {
                    "quantized_weights": sorted(names),
                    "full_calibration": full,
                    "summary": summary,
                }
            )
        return cache[key]

    selected, forward = frozenset(), []
    while size(selected) * 4 < total:
        candidates = [n for n in ordered if n not in selected][
            : config["candidate_pool"]
        ]
        trials = [(n, trial(selected | {n}, False)) for n in candidates]
        best, _ = min(trials, key=lambda t: (*score(t[1]), t[0]))
        selected |= {best}
        forward.append(
            {"selected": best, "quantized_projection_fraction": size(selected) / total}
        )
    seed = sorted(selected)
    current = trial(selected, True)
    repairs = []
    for round_index in range(config["repair_rounds"]):
        if score(current)[:2] == (0, 0.0):
            break
        removable = sorted(
            selected,
            key=lambda n: (
                -by_name[n]["relative_block_objective"] / by_name[n]["fp16_bytes"],
                n,
            ),
        )[: config["removal_pool"]]
        additions = [n for n in ordered if n not in selected][
            : config["candidate_pool"]
        ]
        proposals = []
        for removed in removable:
            for added in additions:
                names = (selected - {removed}) | {added}
                if size(names) * 4 >= total:
                    proposals.append((removed, added, names, trial(names, False)))
        shortlist = sorted(proposals, key=lambda t: (*score(t[3]), t[0], t[1]))[
            : config["full_trial_count"]
        ]
        full_trials = [
            (r, a, names, trial(names, True)) for r, a, names, _ in shortlist
        ]
        best = min(full_trials, key=lambda t: (*score(t[3]), t[0], t[1]), default=None)
        accepted = best is not None and score(best[3]) < score(current)
        repairs.append(
            {
                "round": round_index + 1,
                "proposal_count": len(proposals),
                "full_trials": [
                    {"removed": r, "added": a, "summary": s}
                    for r, a, _, s in full_trials
                ],
                "accepted": accepted,
                "before": current,
            }
        )
        if not accepted:
            break
        selected, current = best[2], best[3]
        repairs[-1].update(removed=best[0], added=best[1], after=current)
    return {
        "quantized_weights": sorted(selected),
        "seed_weights": seed,
        "quantized_projection_fraction": size(selected) / total,
        "forward_selection": forward,
        "repairs": repairs,
        "evaluations": evaluations,
        "summary": current,
        "evaluation_bound": len(local) * config["candidate_pool"]
        + 1
        + config["repair_rounds"]
        * (
            config["removal_pool"] * config["candidate_pool"]
            + config["full_trial_count"]
        ),
    }
