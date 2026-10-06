"""Portable block-diagonal, activation-calibrated W8A16 error compensation.

This is not full GPTQ: correlations across blocks are discarded, scales use
the original per-row max, and activations come from the FP16 source model.
All objectives use the actual FP16-rounded reconstruction consumed by Forge.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .quantization import dequantize_per_channel, quantize_per_channel

ALGORITHM = "block_second_order_joint_forward_v1"
DEFAULT_CONFIG = {"block_size": 64, "damping": 0.01, "activation_order": True}


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("invalid second-order configuration")
    block, damping = config["block_size"], config["damping"]
    if (
        isinstance(block, bool)
        or not isinstance(block, int)
        or not 1 <= block <= 256
        or isinstance(damping, bool)
        or not isinstance(damping, (int, float))
        or not math.isfinite(damping)
        or not 0 < damping <= 1
        or not isinstance(config["activation_order"], bool)
    ):
        raise ValueError("invalid second-order configuration")


def block_moments(inputs: np.ndarray, block_size: int) -> np.ndarray:
    validate_config({**DEFAULT_CONFIG, "block_size": block_size})
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("calibration inputs must be a finite nonempty matrix")
    padded = np.pad(x, ((0, 0), (0, (-x.shape[1]) % block_size)))
    blocks = padded.reshape(len(x), -1, block_size).transpose(1, 0, 2)
    return blocks.transpose(0, 2, 1) @ blocks / len(x)


def validate_moments(moments: np.ndarray, width: int, block_size: int) -> None:
    expected = ((width + block_size - 1) // block_size, block_size, block_size)
    if (
        moments.dtype != np.float64
        or moments.shape != expected
        or not np.isfinite(moments).all()
        or not np.allclose(moments, moments.transpose(0, 2, 1), rtol=0, atol=1e-10)
    ):
        raise ValueError("invalid calibration covariance")
    for block in moments:
        tolerance = max(1.0, float(np.max(np.abs(block)))) * 1e-9
        if float(np.linalg.eigvalsh(block)[0]) < -tolerance:
            raise ValueError("calibration covariance is not positive semidefinite")
    padding = (-width) % block_size
    if padding and np.any(moments[-1, -padding:, :]):
        raise ValueError("nonzero covariance padding")


def quantize_second_order(
    weight: np.ndarray,
    moments: np.ndarray,
    config: dict,
    *,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compensate rounding error inside each block; reject worse rows per block.

    Optional offline scales must be finite FP32 channel scales no larger than
    the original max-based scales. With no override, the old recipe is unchanged.
    The row/block fallback guarantees no worse *block-diagonal calibration*
    reconstruction objective than RTN. It does not guarantee output quality.
    """
    validate_config(config)
    rtn, original_scales = quantize_per_channel(weight)
    if scales is None:
        scales = original_scales
    else:
        scales = np.asarray(scales)
        if (
            scales.dtype != np.float32
            or scales.shape != original_scales.shape
            or not np.all(np.isfinite(scales) & (scales > 0))
            or np.any(scales > original_scales)
        ):
            raise ValueError("invalid fixed second-order scales")
        rtn = np.clip(
            np.rint(np.asarray(weight, dtype=np.float32) / scales[:, None]), -127, 127
        ).astype(np.int8)
    weight = np.asarray(weight, dtype=np.float64)
    block_size = config["block_size"]
    validate_moments(moments, weight.shape[1], block_size)
    packed = rtn.copy()
    reference = dequantize_per_channel(rtn, scales).astype(np.float64)
    total_reference = total_candidate = total_signal = 0.0
    fallbacks = 0
    for index, start in enumerate(range(0, weight.shape[1], block_size)):
        stop = min(start + block_size, weight.shape[1])
        h = moments[index, : stop - start, : stop - start]
        original = weight[:, start:stop]
        permutation = (
            np.argsort(-np.diag(h), kind="stable")
            if config["activation_order"]
            else np.arange(len(h))
        )
        ordered_h = h[np.ix_(permutation, permutation)].copy()
        damp = max(float(np.mean(np.diag(h))) * config["damping"], 1e-12)
        ordered_h.flat[:: len(h) + 1] += damp
        # Upper R with R.T @ R = inverse(H), not Cholesky(H) itself.
        inverse = np.linalg.solve(ordered_h, np.eye(len(h)))
        inverse = (inverse + inverse.T) * 0.5
        factor = np.linalg.cholesky(inverse).T
        working = original[:, permutation].copy()
        candidate = np.empty_like(rtn[:, start:stop])
        for column in range(len(h)):
            rounded = np.clip(np.rint(working[:, column] / scales), -127, 127).astype(
                np.int8
            )
            candidate[:, permutation[column]] = rounded
            reconstructed = dequantize_per_channel(rounded[:, None], scales)[:, 0]
            error = (working[:, column] - reconstructed) / factor[column, column]
            working[:, column + 1 :] -= error[:, None] * factor[column, column + 1 :]
        candidate_dense = dequantize_per_channel(candidate, scales).astype(np.float64)

        def loss(values, original=original, h=h):
            error = original - values
            return np.maximum(np.sum((error @ h) * error, axis=1), 0)

        old_loss, new_loss = loss(reference[:, start:stop]), loss(candidate_dense)
        fallback = new_loss > old_loss
        candidate[fallback] = rtn[:, start:stop][fallback]
        fallbacks += int(np.count_nonzero(fallback))
        packed[:, start:stop] = candidate
        total_reference += float(old_loss.sum())
        total_candidate += float(np.minimum(new_loss, old_loss).sum())
        total_signal += float(np.sum((original @ h) * original))
    return (
        packed,
        scales,
        {
            "rtn_block_objective": total_reference,
            "calibrated_block_objective": total_candidate,
            "relative_block_objective": total_candidate / max(total_signal, 1e-30),
            "changed_quantized_elements": int(np.count_nonzero(packed != rtn)),
            "rtn_fallback_row_blocks": fallbacks,
        },
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_header(archive, member):
    with archive.open(member) as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError("unsupported calibration NPY format")
        if (
            dtype.hasobject
            or handle.tell() + math.prod(shape) * dtype.itemsize != member.file_size
        ):
            raise ValueError("invalid calibration NPY payload size or dtype")
        return shape, dtype


def write_calibration_stats(path: Path, metadata: dict, arrays: dict) -> str:
    """No pickle, no overwrite; source-bound metadata is inside the archive."""
    path = Path(path)
    if "metadata" in arrays:
        raise ValueError("reserved calibration array key")
    payload = np.frombuffer(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".calibration-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, metadata=payload, **arrays)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return file_sha256(path)


class CalibrationStats:
    """Strictly bounded, source-bound NPZ reader used only by offline tools."""

    def __init__(self, path: Path, source, *, expected_sha: str | None = None):
        self.sha256 = file_sha256(path)
        if expected_sha is not None and self.sha256 != expected_sha:
            raise ValueError("calibration statistics checksum mismatch")
        names = {
            n
            for n in source.tensors
            if n.startswith("model.layers.") and n.endswith("_proj.weight")
        }
        # Bound even untrusted ZIP sizes before NumPy decompresses any member.
        maximum = (1 << 20) + sum(
            ((source.tensor_info(n).shape[1] + 255) // 256) * 256 * 256 * 8 + 4096
            for n in names
        )
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) > len(names) + 1
                or len({m.filename for m in members}) != len(members)
                or sum(m.file_size for m in members) > maximum
                or "metadata.npy" not in archive.namelist()
                or archive.getinfo("metadata.npy").file_size > 1 << 20
            ):
                raise ValueError("invalid or oversized calibration archive")
            shape, dtype = npy_header(archive, archive.getinfo("metadata.npy"))
            if dtype != np.uint8 or len(shape) != 1:
                raise ValueError("invalid calibration metadata")
            raw = np.load(io.BytesIO(archive.read("metadata.npy")), allow_pickle=False)
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise ValueError("invalid calibration metadata")
            self.metadata = json.loads(raw.tobytes())
            metadata = self.metadata
            if (
                not isinstance(metadata, dict)
                or metadata.get("schema_version") != 1
                or metadata.get("source_data_sha256") != source.data_sha256
                or metadata.get("algorithm")
                not in {
                    ALGORITHM,
                }
                or "quantizer_config" not in metadata
                or "weight_to_moments" not in metadata
            ):
                raise ValueError(
                    "calibration statistics belong to a different source or method"
                )
            self.config = metadata["quantizer_config"]
            self.algorithm = metadata["algorithm"]
            self.method = "block_second_order_v1"
            validate_config(self.config)
            mapping = metadata["weight_to_moments"]
            if not isinstance(mapping, dict) or set(mapping) != names:
                raise ValueError("calibration must cover every supported projection")
            if any(
                not isinstance(k, str) or not k.startswith("cov_")
                for k in mapping.values()
            ):
                raise ValueError("invalid calibration covariance key")
            keys = set(mapping.values())
            if set(archive.namelist()) != {k + ".npy" for k in keys} | {"metadata.npy"}:
                raise ValueError("unexpected calibration archive members")
        self.arrays = {}
        with zipfile.ZipFile(path) as archive:
            for key in sorted(keys):
                widths = {
                    source.tensor_info(n).shape[1]
                    for n, k in mapping.items()
                    if k == key
                }
                if len(widths) != 1:
                    raise ValueError(
                        "shared calibration covariance has incompatible widths"
                    )
                width = next(iter(widths))
                block = self.config["block_size"]
                expected_shape = ((width + block - 1) // block, block, block)
                member = archive.getinfo(key + ".npy")
                if member.file_size > math.prod(expected_shape) * 8 + 4096:
                    raise ValueError("oversized calibration covariance member")
                # Check the NPY shape before allocating even a bounded ZIP payload.
                shape, dtype = npy_header(archive, member)
                if shape != expected_shape or dtype != np.float64:
                    raise ValueError("invalid calibration covariance shape or dtype")
                self.arrays[key] = np.load(
                    io.BytesIO(archive.read(member)), allow_pickle=False
                )
        for name, key in mapping.items():
            validate_moments(
                self.arrays[key],
                source.tensor_info(name).shape[1],
                self.config["block_size"],
            )

    def quantize(self, name: str, weight: np.ndarray):
        quantizer = quantize_second_order
        return quantizer(
            weight, self.arrays[self.metadata["weight_to_moments"][name]], self.config
        )
