"""Post-registration host-only reconstruction readback, never fitting or timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from forge_llm.second_order import CalibrationStats, file_sha256, quantize_second_order
from forge_llm.scale_aware import ALGORITHM
from forge_llm.model_file import ModelFile
from gpu_guard import exclusive_gpu_workflow
from run_scale_checkpoint import calibration_gates, load_contract
from run_second_order_checkpoint import save


def independent_loss(weight, packed, scales, moments, block):
    """Independent FP16 reconstruction and einsum quadratic, not fitting helpers."""
    values = np.zeros(len(weight), dtype=np.float64)
    for begin in range(0, len(weight), 256):
        end = min(begin + 256, len(weight))
        for i, start in enumerate(range(0, weight.shape[1], block)):
            stop = min(start + block, weight.shape[1])
            reconstructed = (
                (
                    packed[begin:end, start:stop].astype(np.float32)
                    * scales[begin:end, None]
                )
                .astype(np.float16)
                .astype(np.float64)
            )
            error = weight[begin:end, start:stop].astype(np.float64) - reconstructed
            h = moments[i, : stop - start, : stop - start]
            values[begin:end] += np.maximum(
                np.einsum("ri,ij,rj->r", error, h, error, optimize=False), 0
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite independent reconstruction objective")
    return values


def verify(root):
    contract = load_contract(root)
    observations = []
    for family, item in contract["families"].items():
        path = root / "calibration" / f"{family}.json"
        report = json.loads(path.read_text())
        calibration_gates(
            report
        )  # A valid failed calibration is permitted, not waived.
        local = {o["weight"]: o for o in report["local_objectives"]}
        with ModelFile(item["source"]) as source:
            recipe = CalibrationStats(
                Path(item["stats"]),
                source,
                expected_sha=report["policy"]["calibration_stats_sha256"],
            )
            if recipe.algorithm != ALGORITHM:
                raise ValueError("wrong recipe for scale objective diagnosis")
            base_config = {
                k: recipe.config[k]
                for k in ("block_size", "damping", "activation_order")
            }
            for name in report["policy"]["quantized_weights"]:
                weight = source.tensor_numpy(name)
                moments = recipe.arrays[recipe.metadata["weight_to_moments"][name]]
                q, scales, _ = recipe.quantize(name, weight)
                old_q, old_scales, _ = quantize_second_order(
                    weight, moments, base_config
                )
                actual = independent_loss(
                    weight, q, scales, moments, recipe.config["block_size"]
                )
                baseline = independent_loss(
                    weight, old_q, old_scales, moments, recipe.config["block_size"]
                )
                delta = actual - baseline
                # Independent FP64 quadratic evaluation may order sums differently.
                tolerance = 1e-9 * np.maximum(1.0, baseline)
                if np.any(delta > tolerance) or not np.allclose(
                    [baseline.sum(), actual.sum()],
                    [
                        local[name]["fixed_scale_block_objective"],
                        local[name]["calibrated_block_objective"],
                    ],
                    rtol=1e-9,
                    atol=1e-9,
                ):
                    raise ValueError(
                        "independent objective disagrees with the fitted reconstruction bound"
                    )
                if (
                    int(np.count_nonzero(scales != old_scales))
                    != local[name]["changed_scale_rows"]
                ):
                    raise ValueError(
                        "reported scale changes disagree with derived coefficients"
                    )
                observations.append(
                    {
                        "family": family,
                        "weight": name,
                        "rows": len(weight),
                        "shape": list(weight.shape),
                        "baseline_objective": float(baseline.sum()),
                        "scale_aware_objective": float(actual.sum()),
                        "maximum_row_loss_delta": float(delta.max()),
                        "positive_row_deltas_within_tolerance": int(
                            np.count_nonzero(delta > 0)
                        ),
                        "row_bound_verified": True,
                        "changed_scale_rows": int(
                            np.count_nonzero(scales != old_scales)
                        ),
                        "packed_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                        "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
                        "calibration_report_sha256": file_sha256(path),
                        "stats_sha256": recipe.sha256,
                    }
                )
        print(
            json.dumps(
                {
                    "family": family,
                    "matrices": sum(o["family"] == family for o in observations),
                    "passed": True,
                }
            ),
            flush=True,
        )
    return {
        "schema_version": 1,
        "environment": contract["environment"],
        "contract_sha256": file_sha256(root / "contract.json"),
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "observations": observations,
        "matrices": len(observations),
        "rows": sum(o["rows"] for o in observations),
        "passed": bool(observations)
        and all(o["row_bound_verified"] for o in observations),
        "limitations": "Post-registration, calibration-only host diagnosis. Recomputes all selected rows/coefficients and independent FP64 quadratics; verification tolerance 1e-9 for changed reduction order is not a relaxed inference gate. No packed official model is exported or evaluated. No language-quality/performance/direct-kernel certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    target = args.root / "objective-verification.json"
    if target.exists():
        raise FileExistsError(target)
    with exclusive_gpu_workflow():
        report = verify(args.root)
    save(target, report)
    print(
        json.dumps(
            {
                "matrices": report["matrices"],
                "rows": report["rows"],
                "passed": report["passed"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
