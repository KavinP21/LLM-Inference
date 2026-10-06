"""Separate reconstruction identity from projection evaluation; not fitting/timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from forge_llm.backends.int8_kernels import Int8LinearKernels
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.quantization import dequantize_per_channel
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow


def probe(path):
    observations = []
    rng = np.random.default_rng(7321)
    with MlxEngine(
        path, max_model_length=2048, int8_mode="reconstruct", decode_mode="rowwise"
    ) as engine:
        model, mx = engine.model, engine.model.mx
        kernels = Int8LinearKernels(mx)
        if not model.file.quantization:
            raise ValueError("projection diagnostics require an INT8 artifact")
        for name, spec in model.file.quantization.items():
            packed, scales = model.weights[name], model.weights[spec.scale_name]
            reconstructed = kernels.reconstruct(packed, scales)
            composed = (packed.astype(mx.float32) * scales[:, None]).astype(mx.float16)
            independent = dequantize_per_channel(
                model.file.tensor_numpy(name), model.file.tensor_numpy(spec.scale_name)
            )
            same = bool(
                np.array_equal(np.array(reconstructed), independent)
                and np.array_equal(np.array(composed), independent)
            )
            for rows in (1, 2, 8):
                x = mx.array(
                    rng.normal(0, 0.3, (rows, packed.shape[1])).astype(np.float16)
                )
                native = mx.concatenate(
                    [x[i : i + 1] @ reconstructed.T for i in range(rows)], axis=0
                )
                baseline = mx.concatenate(
                    [x[i : i + 1] @ composed.T for i in range(rows)], axis=0
                )
                direct = mx.concatenate(
                    [kernels.linear(x[i : i + 1], packed, scales) for i in range(rows)],
                    axis=0,
                )
                native, baseline, direct = [
                    np.array(v) for v in (native, baseline, direct)
                ]
                error = direct.astype(np.float64) - baseline.astype(np.float64)
                observations.append(
                    {
                        "weight": name,
                        "rows": rows,
                        "shape": list(packed.shape),
                        "reconstruction_bitwise_equal": same,
                        "native_projection_bitwise_equal": bool(
                            np.array_equal(native, baseline)
                        ),
                        "direct_projection_bitwise_equal": bool(
                            np.array_equal(direct, baseline)
                        ),
                        "direct_different_elements": int(
                            np.count_nonzero(direct != baseline)
                        ),
                        "direct_max_absolute_error": float(np.max(np.abs(error))),
                        "direct_relative_l2_error": float(
                            np.linalg.norm(error)
                            / np.linalg.norm(baseline.astype(np.float64))
                        ),
                    }
                )
        build = engine.build_info()
    gates = {
        "all_reconstructed_values_match_numpy": all(
            o["reconstruction_bitwise_equal"] for o in observations
        ),
        "all_native_projections_match_composed": all(
            o["native_projection_bitwise_equal"] for o in observations
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "model": model_provenance(path),
        "environment": {"native_build": build, **source_provenance()},
        "configuration": {
            "seed": 7321,
            "row_counts": [1, 2, 8],
            "decode_mode": "rowwise",
            "script_sha256": file_sha256(Path(__file__)),
        },
        "observations": observations,
        "gates": gates,
        "direct_bitwise_projection_cases": sum(
            o["direct_projection_bitwise_equal"] for o in observations
        ),
        "limitations": "Synthetic same-input projections isolate evaluation differences, not exact Metal compiler instructions, all possible inputs, model quality, or measured performance. Direct mismatches are retained and never drive calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        report = probe(args.model)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "gates": report["gates"],
                "direct_bitwise_projection_cases": report[
                    "direct_bitwise_projection_cases"
                ],
                "total": len(report["observations"]),
            }
        ),
        flush=True,
    )
    if not report["gates"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
