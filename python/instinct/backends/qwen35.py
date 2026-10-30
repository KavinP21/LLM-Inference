"""Qwen3.5 text model (the 4B hybrid) from an MLX checkpoint.

Three of every four layers are Gated DeltaNet linear attention: a short causal
convolution, then a recurrent matrix memory per head that is decayed, read,
corrected toward each new value and read again. Their state is a fixed size
per sequence, kept in the engine's state slots. Every fourth layer is ordinary
GQA attention with a sigmoid output gate, over the paged cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paged_kv import KVGroup, StateGroup
from .base import load_checkpoint, mlx, text_config
from .metal_kernels import gated_delta


class Qwen35Model:
    model_type = "qwen3_5"

    def __init__(self, path: str | Path) -> None:
        self.mx = mx = mlx()
        import mlx.nn as nn

        self.nn = nn
        config, self.weights = load_checkpoint(
            path, keep=lambda name: name.startswith("model.")
        )
        c = self.config = text_config(config)
        self.dtype = mx.bfloat16
        self.vocab_size = c["vocab_size"]
        self.max_position_embeddings = c["max_position_embeddings"]
        eos = c.get("eos_token_id", config.get("eos_token_id"))
        self.eos_token_ids = tuple(eos if isinstance(eos, list) else [eos])
        self.eps = c["rms_norm_eps"]
        self.heads = c["num_attention_heads"]
        self.head_dim = c["head_dim"]
        rope = c.get("rope_parameters") or {}
        self.rope_base = rope.get("rope_theta", c.get("rope_theta", 10000.0))
        self.rope_dims = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        self.k_heads = c["linear_num_key_heads"]
        self.v_heads = c["linear_num_value_heads"]
        self.k_dim = c["linear_key_head_dim"]
        self.v_dim = c["linear_value_head_dim"]
        self.kernel = c["linear_conv_kernel_dim"]
        self.conv_width = 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
        interval = c["full_attention_interval"]
        layers = range(c["num_hidden_layers"])
        self.linear = [(i + 1) % interval != 0 for i in layers]
        self.kv_groups = [
            KVGroup(
                "full",
                tuple(i for i in layers if not self.linear[i]),
                c["num_key_value_heads"],
                self.head_dim,
            )
        ]
        self.state_groups = [
            StateGroup(
                "delta",
                tuple(i for i in layers if self.linear[i]),
                (
                    (self.kernel - 1, self.conv_width),
                    (self.v_heads, self.v_dim, self.k_dim),
                ),
                (mx.bfloat16, mx.float32),
            )
        ]
        # Depthwise conv taps as [kernel, channels], float32 for the sum.
        self.taps = {
            i: self.weights[f"model.layers.{i}.linear_attn.conv1d.weight"][
                :, :, 0
            ].T.astype(mx.float32)
            for i in layers
            if self.linear[i]
        }
        self.decay_rate = {
            i: mx.exp(
                self.weights[f"model.layers.{i}.linear_attn.A_log"].astype(mx.float32)
            )
            for i in layers
            if self.linear[i]
        }
        mx.eval(*self.taps.values(), *self.decay_rate.values())

    def _norm(self, x: Any, name: str | None, eps: float | None = None) -> Any:
        weight = None if name is None else self.weights[name]
        return self.mx.fast.rms_norm(x, weight, self.eps if eps is None else eps)

    def _attention(self, batch: Any, i: int, h: Any) -> Any:
        mx, w, n = self.mx, self.weights, batch.count
        p = f"model.layers.{i}.self_attn."
        q, gate = mx.split(
            w.linear(p + "q_proj", h).reshape(n, self.heads, 2 * self.head_dim),
            2,
            axis=-1,
        )
        q = self._norm(q, p + "q_norm.weight")
        k = self._norm(
            w.linear(p + "k_proj", h).reshape(n, -1, self.head_dim), p + "k_norm.weight"
        )
        v = w.linear(p + "v_proj", h).reshape(n, -1, self.head_dim)
        q = batch.rope(q, self.rope_dims, base=self.rope_base)
        k = batch.rope(k, self.rope_dims, base=self.rope_base)
        out = batch.attend(i, q, k, v, scale=self.head_dim**-0.5)
        return w.linear(p + "o_proj", out * mx.sigmoid(gate.reshape(n, -1)))

    def _delta(self, batch: Any, i: int, h: Any) -> Any:
        mx, w, nn = self.mx, self.weights, self.nn
        p = f"model.layers.{i}.linear_attn."
        rows = 1 if batch.prefill else batch.count  # sequences in this call
        steps = batch.count // rows
        mixed = w.linear(p + "in_proj_qkv", h).reshape(rows, steps, -1)
        z = w.linear(p + "in_proj_z", h).reshape(rows, steps, self.v_heads, self.v_dim)
        b = w.linear(p + "in_proj_b", h).reshape(rows, steps, self.v_heads)
        a = w.linear(p + "in_proj_a", h).reshape(rows, steps, self.v_heads)
        conv_state, memory = batch.state(i)

        # Causal depthwise convolution over the carried-in tail and this chunk.
        window = mx.concatenate([conv_state.astype(mixed.dtype), mixed], axis=1)
        taps = self.taps[i]
        conv = sum(
            window[:, j : j + steps].astype(mx.float32) * taps[j]
            for j in range(self.kernel)
        )
        conv = nn.silu(conv).astype(mixed.dtype)
        key_width = self.k_heads * self.k_dim
        q = conv[..., :key_width].reshape(rows, steps, self.k_heads, self.k_dim)
        k = conv[..., key_width : 2 * key_width].reshape(
            rows, steps, self.k_heads, self.k_dim
        )
        v = conv[..., 2 * key_width :].reshape(rows, steps, self.v_heads, self.v_dim)
        q = self.k_dim**-1.0 * mx.fast.rms_norm(q, None, 1e-6)
        k = self.k_dim**-0.5 * mx.fast.rms_norm(k, None, 1e-6)
        decay = mx.exp(
            -self.decay_rate[i]
            * nn.softplus(a.astype(mx.float32) + w[p + "dt_bias"].astype(mx.float32))
        )
        strength = mx.sigmoid(b.astype(mx.float32))
        out, memory = gated_delta(mx, q, k, v, decay, strength, memory)
        batch.set_state(i, [window[:, -(self.kernel - 1) :], memory])

        normed = self._norm(out, p + "norm.weight")
        gated = (nn.silu(z.astype(mx.float32)) * normed.astype(mx.float32)).astype(
            h.dtype
        )
        return w.linear(p + "out_proj", gated.reshape(batch.count, -1))

    def forward(self, batch: Any) -> Any:
        mx, w = self.mx, self.weights
        x = w.embed("model.embed_tokens", batch.tokens)
        for i, linear in enumerate(self.linear):
            p = f"model.layers.{i}."
            h = self._norm(x, p + "input_layernorm.weight")
            x = x + (
                self._delta(batch, i, h) if linear else self._attention(batch, i, h)
            )
            h = self._norm(x, p + "post_attention_layernorm.weight")
            gate = w.linear(p + "mlp.gate_proj", h)
            x = x + w.linear(
                p + "mlp.down_proj", self.nn.silu(gate) * w.linear(p + "mlp.up_proj", h)
            )
        if batch.prefill:
            x = x[-1:]
        x = self._norm(x, "model.norm.weight")
        group_size, bits = w.quantization["model.embed_tokens"]
        return mx.quantized_matmul(
            x,
            w["model.embed_tokens.weight"],
            w["model.embed_tokens.scales"],
            w["model.embed_tokens.biases"],
            transpose=True,
            group_size=group_size,
            bits=bits,
        ).astype(mx.float32)

    def close(self) -> None:
        self.weights.clear()
