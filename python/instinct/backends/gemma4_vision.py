"""Gemma 4's image encoder: pixels to soft tokens in the language model's space.

An image is resized (keeping its aspect ratio) so it splits into at most
2,520 patches of 16 x 16 pixels, with sides a multiple of 48. The patches go
through 27 transformer blocks with 2-D rotary positions (x on the first half
of each head, y on the second), are averaged over 3 x 3 neighbourhoods into at
most 280 soft tokens, and are projected to the text model's width. Each image
is encoded on its own, so no padding or attention mask is needed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .base import Weights


def load_image(
    path: str | Path, *, patch_size: int = 16, pool: int = 3, max_tokens: int = 280
) -> np.ndarray:
    """Read and resize an image to ``[3, height, width]`` floats in [0, 1]."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    max_patches = max_tokens * pool * pool
    side = pool * patch_size
    factor = math.sqrt(max_patches * patch_size * patch_size / (height * width))
    target_h = int(math.floor(factor * height / side)) * side
    target_w = int(math.floor(factor * width / side)) * side
    longest = (max_patches // pool**2) * side
    if target_h == 0 and target_w == 0:
        raise ValueError("the image is too small to encode")
    if target_h == 0:
        target_h, target_w = side, min(int(width // height) * side, longest)
    elif target_w == 0:
        target_h, target_w = min(int(height // width) * side, longest), side
    if (target_w, target_h) != (width, height):
        image = image.resize((target_w, target_h), resample=Image.BICUBIC)
    return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)


class Gemma4Vision:
    def __init__(
        self, mx: Any, weights: Weights, config: dict[str, Any], text_hidden: int
    ) -> None:
        import mlx.nn as nn

        self.mx, self.nn, self.w = mx, nn, weights
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.layers = config["num_hidden_layers"]
        self.patch = config["patch_size"]
        self.pool = config["pooling_kernel_size"]
        self.max_tokens = config.get("default_output_length", 280)
        self.eps = config.get("rms_norm_eps", 1e-6)
        self.base = config["rope_parameters"]["rope_theta"]
        self.standardize = config.get("standardize", False)
        self.text_hidden = text_hidden
        half = self.head_dim // 2 // 2  # frequencies per spatial axis
        exponents = (2.0 / (2 * half)) * mx.arange(0, half).astype(mx.float32)
        self.timescale = mx.power(self.base, exponents)

    def soft_tokens(self, pixels: np.ndarray) -> int:
        _, height, width = pixels.shape
        return (height // self.patch) * (width // self.patch) // self.pool**2

    def _norm(self, x: Any, name: str | None) -> Any:
        weight = None if name is None else self.w[name]
        return self.mx.fast.rms_norm(x, weight, self.eps)

    def _rope(self, x: Any, cos: Any, sin: Any) -> Any:
        """Rotate the x half and the y half of each head independently."""
        mx = self.mx
        parts = []
        width = x.shape[-1] // 2
        for axis in range(2):
            part = x[..., axis * width : (axis + 1) * width]
            half = width // 2
            rotated = mx.concatenate([-part[..., half:], part[..., :half]], axis=-1)
            parts.append(part * cos[axis] + rotated * sin[axis])
        return mx.concatenate(parts, axis=-1)

    def encode(self, pixels: np.ndarray) -> Any:
        """Soft-token embeddings for one image, ``[tokens, text_hidden]``."""
        mx, w = self.mx, self.w
        _, height, width = pixels.shape
        p = self.patch
        rows, cols = height // p, width // p
        patches = (
            pixels.reshape(3, rows, p, cols, p)
            .transpose(1, 3, 2, 4, 0)
            .reshape(rows * cols, -1)
        )
        x = w.linear(
            "vision_tower.patch_embedder.input_proj",
            mx.array(2 * (patches - 0.5)).astype(mx.bfloat16),
        )
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows), indexing="xy")
        xs, ys = mx.array(xs.flatten()), mx.array(ys.flatten())
        table = w["vision_tower.patch_embedder.position_embedding_table"]
        x = x + table[0][xs] + table[1][ys]

        cos, sin = [], []
        for positions in (xs, ys):
            angles = positions[:, None].astype(mx.float32) / self.timescale
            cos.append(
                mx.concatenate([mx.cos(angles)] * 2, axis=-1)[:, None, :].astype(
                    x.dtype
                )
            )
            sin.append(
                mx.concatenate([mx.sin(angles)] * 2, axis=-1)[:, None, :].astype(
                    x.dtype
                )
            )
        n = rows * cols
        for layer in range(self.layers):
            pre = f"vision_tower.encoder.layers.{layer}."
            h = self._norm(x, pre + "input_layernorm.weight")
            q = w.linear(pre + "self_attn.q_proj.linear", h).reshape(
                n, self.heads, self.head_dim
            )
            k = w.linear(pre + "self_attn.k_proj.linear", h).reshape(
                n, self.heads, self.head_dim
            )
            v = w.linear(pre + "self_attn.v_proj.linear", h).reshape(
                n, self.heads, self.head_dim
            )
            q = self._rope(self._norm(q, pre + "self_attn.q_norm.weight"), cos, sin)
            k = self._rope(self._norm(k, pre + "self_attn.k_norm.weight"), cos, sin)
            v = self._norm(v, None)
            out = mx.fast.scaled_dot_product_attention(
                q.transpose(1, 0, 2)[None],
                k.transpose(1, 0, 2)[None],
                v.transpose(1, 0, 2)[None],
                scale=1.0,
            )
            out = out[0].transpose(1, 0, 2).reshape(n, -1)
            x = x + self._norm(
                w.linear(pre + "self_attn.o_proj.linear", out),
                pre + "post_attention_layernorm.weight",
            )
            h = self._norm(x, pre + "pre_feedforward_layernorm.weight")
            h = self.nn.gelu_approx(
                w.linear(pre + "mlp.gate_proj.linear", h)
            ) * w.linear(pre + "mlp.up_proj.linear", h)
            x = x + self._norm(
                w.linear(pre + "mlp.down_proj.linear", h),
                pre + "post_feedforward_layernorm.weight",
            )

        # Average 3 x 3 neighbourhoods, row by row, into the soft tokens.
        k = self.pool
        x = (
            x.reshape(rows // k, k, cols // k, k, -1)
            .astype(mx.float32)
            .mean(axis=(1, 3))
        )
        x = (x.reshape(-1, self.hidden) * self.hidden**0.5).astype(mx.bfloat16)
        if self.standardize:
            x = (x - w["vision_tower.std_bias"]) * w["vision_tower.std_scale"]
        x = self._norm(x, None)
        return w.linear("embed_vision.embedding_projection", x)
