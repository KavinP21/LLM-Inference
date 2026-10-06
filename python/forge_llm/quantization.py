"""Deterministic weight-only INT8 conversion; no ML framework required."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .format import QuantizationSpec, write_engine, write_manifest
from .model_contract import validate_model_weights
from .model_file import ModelFile
from .precision_policy import canonical_sha256, validate_policy


def quantize_per_channel(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest-even symmetric INT8 with per-row FP32 scales.

    Zero channels use scale=1. Embeddings, norms, biases and the LM head stay
    FP16. Calibration, activation quantization and clipping are not performed.
    """
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or not all(weight.shape):
        raise ValueError("INT8 quantization requires a nonempty matrix")
    if not np.all(np.isfinite(weight)):
        raise ValueError("cannot quantize non-finite weights")
    maximum = np.max(np.abs(weight), axis=1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(127))
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("INT8 scale is not representable in FP32")
    packed = np.clip(np.rint(weight / scales[:, None]), -127, 127).astype(np.int8)
    return packed, scales.astype("<f4")


def dequantize_per_channel(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (weight.astype(np.float32) * scales[:, None]).astype(np.float16)


def quantize_model(
    source: str | Path,
    output: str | Path,
    *,
    retain_fp16: list[str] | None = None,
    policy: dict | None = None,
    calibration_stats: str | Path | None = None,
) -> dict:
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("quantized output must not overwrite the source artifact")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    manifest_path = output.with_suffix(output.suffix + ".json")
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing manifest: {manifest_path}"
        )
    with ModelFile(source) as file:
        validate_model_weights(file)
        if file.quantization:
            raise ValueError("conversion requires an FP16 source artifact")
        projection_names = {
            name
            for name in file.tensors
            if name.startswith("model.layers.") and name.endswith("_proj.weight")
        }
        if policy is not None:
            if retain_fp16 is not None:
                raise ValueError("use either a precision policy or explicit retention")
            retain_fp16 = validate_policy(policy, file.data_sha256, projection_names)
        calibrated = None
        second_order = policy is not None and policy["algorithm"] in {
            "block_second_order_joint_forward_v1",
            "block_second_order_cached_repair_v1",
        }
        if second_order:
            from .second_order import CalibrationStats

            if calibration_stats is None:
                raise ValueError(
                    "second-order policy requires frozen calibration statistics"
                )
            calibrated = CalibrationStats(
                Path(calibration_stats),
                file,
                expected_sha=policy["calibration_stats_sha256"],
            )
            if calibrated.config != policy["quantizer_config"]:
                raise ValueError("policy and calibration configuration differ")
            expected_algorithm = "block_second_order_joint_forward_v1"
            if calibrated.algorithm != expected_algorithm:
                raise ValueError("policy and calibration method differ")
        elif calibration_stats is not None:
            raise ValueError("calibration statistics require a second-order policy")
        retained = set(retain_fp16 or [])
        if len(retained) != len(retain_fp16 or []) or not retained <= projection_names:
            raise ValueError("retained FP16 names must be unique supported projections")
        projection_bytes = sum(file.tensor_info(n).nbytes for n in projection_names)
        quantized_source_bytes = sum(
            file.tensor_info(n).nbytes for n in projection_names - retained
        )
        if policy is not None and quantized_source_bytes < 0.25 * projection_bytes:
            raise ValueError(
                "calibration policy violates the 25% INT8 projection floor"
            )
        tensors, aliases, seen, specs = {}, {}, {}, {}
        max_error = 0.0
        for name, info in file.tensors.items():
            key = (info.offset, info.shape, info.dtype.str)
            tensors[name] = file.tensor_numpy(name).copy()
            # A source alias need not remain an alias after independent fitting or
            # mixed-precision conversion. Keep ordinary embedding/head aliases,
            # but give projection recipes their own storage (and dtype).
            if (
                key in seen
                and name not in projection_names
                and seen[key] not in projection_names
            ):
                aliases[name] = seen[key]
            else:
                seen[key] = name
            if name in projection_names - retained:
                if calibrated is None:
                    packed, scales = quantize_per_channel(tensors[name])
                else:
                    packed, scales, _ = calibrated.quantize(name, tensors[name])
                max_error = max(
                    max_error,
                    float(
                        np.max(
                            np.abs(
                                packed.astype(np.float32) * scales[:, None]
                                - tensors[name]
                            )
                        )
                    ),
                )
                scale_name = name + ".int8_scale"
                tensors[name], tensors[scale_name] = packed, scales
                specs[name] = QuantizationSpec(scale_name)
        if not specs:
            raise ValueError("source has no supported projection matrices")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".int8-", dir=output.parent)
        os.close(descriptor)
        try:
            result = write_engine(Path(temporary), file.config, tensors, aliases, specs)
            with ModelFile(temporary) as check:
                validate_model_weights(check)
            # Link rather than replace: never clobber a concurrently created output.
            os.link(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        result.update(
            path=str(output),
            quantization="symmetric_int8_per_output_channel",
            quantized_matrices=len(specs),
            source_data_sha256=file.data_sha256,
            source_bytes=source.stat().st_size,
            max_weight_absolute_error=max_error,
            retained_fp16="embeddings, LM head, norms, biases",
            retained_fp16_projections=sorted(retained),
            quantized_projection_fraction=quantized_source_bytes / projection_bytes,
            fp16_source_projection_bytes=projection_bytes,
            quantized_source_projection_bytes=quantized_source_bytes,
            precision_policy=policy,
            precision_policy_sha256=canonical_sha256(policy) if policy else None,
            quantization_method=calibrated.method if calibrated else "rtn_v1",
            calibration_stats_sha256=calibrated.sha256 if calibrated else None,
            quality_status="experimental; calibration is not held-out certification",
        )
    write_manifest(manifest_path, result, str(source))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a validated FP16 Forge artifact to W8A16"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--retain-fp16", action="append", help="exact projection name")
    choice.add_argument(
        "--policy", type=Path, help="source-bound calibration report/policy"
    )
    parser.add_argument(
        "--calibration-stats",
        type=Path,
        help="frozen source-bound NPZ for a second-order policy",
    )
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text()) if args.policy else None
    if policy is not None and "policy" in policy:
        policy = policy["policy"]
    print(
        json.dumps(
            quantize_model(
                args.source,
                args.output,
                retain_fp16=args.retain_fp16,
                policy=policy,
                calibration_stats=args.calibration_stats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
