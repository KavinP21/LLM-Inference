"""Gemma 4 text model (the 26B A4B mixture of experts) from an MLX checkpoint.

Thirty layers: five of every six attend over a 1,024-token sliding window
with eight 256-wide KV heads, the sixth over the whole context with two
512-wide KV heads whose values come from the key projection. Every layer
runs a dense GeGLU MLP next to a 128-expert, top-8 mixture of experts. The
experts are 4-bit; the batch's tokens are sorted by expert so each expert's
weights are read once per step however many requests chose it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paged_kv import KVGroup
from .base import load_checkpoint, mlx, text_config


class Gemma4Model:
    model_type = "gemma4"

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
        eos = config.get("eos_token_id", c.get("eos_token_id"))
        self.eos_token_ids = tuple(eos if isinstance(eos, list) else [eos])
        self.hidden = c["hidden_size"]
        self.heads = c["num_attention_heads"]
        self.eps = c["rms_norm_eps"]
        self.layer_types = c["layer_types"]
        self.top_k = c["top_k_experts"]
        self.softcap = c.get("final_logit_softcapping") or 0.0
        self.k_eq_v = bool(c.get("attention_k_eq_v"))
        self.embed_scale = self.hidden**0.5
        self.sliding_dim = c["head_dim"]
        self.full_dim = c.get("global_head_dim") or c["head_dim"]
        full_heads = (
            c["num_global_key_value_heads"] if self.k_eq_v else c["num_key_value_heads"]
        )
        layers = range(c["num_hidden_layers"])
        self.kv_groups = [
            KVGroup(
                "sliding",
                tuple(i for i in layers if self.layer_types[i] == "sliding_attention"),
                c["num_key_value_heads"],
                self.sliding_dim,
                c["sliding_window"],
            ),
            KVGroup(
                "full",
                tuple(i for i in layers if self.layer_types[i] == "full_attention"),
                full_heads,
                self.full_dim,
            ),
        ]
        rope = c["rope_parameters"]
        self.sliding_base = rope["sliding_attention"]["rope_theta"]
        # Full layers rotate only the first quarter of their dimensions, at
        # frequencies taken relative to the full width ("proportional" RoPE).
        full = rope["full_attention"]
        rotated = 2 * int(full.get("partial_rotary_factor", 1.0) * self.full_dim // 2)
        exponents = mx.arange(0, rotated, 2, dtype=mx.float32) / self.full_dim
        periods = full.get("factor", 1.0) * mx.power(full["rope_theta"], exponents)
        unrotated = mx.full(
            (self.full_dim // 2 - rotated // 2,), mx.inf, dtype=mx.float32
        )
        self.full_freqs = mx.concatenate([periods, unrotated])
        self.router_scales = {}
        for i in layers:
            p = f"model.layers.{i}.router."
            self.router_scales[i] = self.weights[p + "scale"] * self.hidden**-0.5
        mx.eval(self.full_freqs, *self.router_scales.values())
        self._geglu = mx.compile(
            lambda gate, x: nn.gelu_approx(gate) * x, shapeless=True
        )

    def _norm(self, x: Any, name: str | None) -> Any:
        weight = None if name is None else self.weights[name]
        return self.mx.fast.rms_norm(x, weight, self.eps)

    def _experts(self, p: str, x: Any, indices: Any, weights: Any) -> Any:
        mx, w = self.mx, self.weights
        k = indices.shape[-1]
        x = mx.expand_dims(x, (-2, -3))
        order = inverse = None
        flat = indices
        if indices.size >= 64:
            # Group tokens by expert so gather_qmm walks each expert once.
            flat = indices.flatten()
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            x = x.flatten(0, -3)[order // k]
            flat = flat[order]
        sorted_ = order is not None

        def project(name: str, values: Any) -> Any:
            group_size, bits = w.quantization[p + name]
            return mx.gather_qmm(
                values,
                w[p + name + ".weight"],
                w[p + name + ".scales"],
                w[p + name + ".biases"],
                rhs_indices=flat,
                transpose=True,
                group_size=group_size,
                bits=bits,
                sorted_indices=sorted_,
            )

        hidden = self._geglu(project("gate_proj", x), project("up_proj", x))
        out = project("down_proj", hidden)
        if sorted_:
            out = mx.unflatten(out[inverse], 0, indices.shape)
        return (out.squeeze(-2) * weights[..., None]).sum(-2)

    def forward(self, batch: Any) -> Any:
        mx, w = self.mx, self.weights
        n = batch.count
        x = w.embed("model.embed_tokens", batch.tokens) * self.embed_scale
        for i, kind in enumerate(self.layer_types):
            p = f"model.layers.{i}."
            sliding = kind == "sliding_attention"
            dim = self.sliding_dim if sliding else self.full_dim
            h = self._norm(x, p + "input_layernorm.weight")
            q = w.linear(p + "self_attn.q_proj", h).reshape(n, self.heads, dim)
            q = self._norm(q, p + "self_attn.q_norm.weight")
            k = w.linear(p + "self_attn.k_proj", h).reshape(n, -1, dim)
            v = (
                k
                if (self.k_eq_v and not sliding)
                else w.linear(p + "self_attn.v_proj", h).reshape(n, -1, dim)
            )
            k = self._norm(k, p + "self_attn.k_norm.weight")
            v = self._norm(v, None)
            if sliding:
                q = batch.rope(q, dim, base=self.sliding_base)
                k = batch.rope(k, dim, base=self.sliding_base)
            else:
                q = batch.rope(q, dim, freqs=self.full_freqs)
                k = batch.rope(k, dim, freqs=self.full_freqs)
            attention = batch.attend(i, q, k, v, scale=1.0)
            h = self._norm(
                w.linear(p + "self_attn.o_proj", attention),
                p + "post_attention_layernorm.weight",
            )
            x = x + h

            dense = self._norm(x, p + "pre_feedforward_layernorm.weight")
            dense = w.linear(
                p + "mlp.down_proj",
                self._geglu(
                    w.linear(p + "mlp.gate_proj", dense),
                    w.linear(p + "mlp.up_proj", dense),
                ),
            )
            dense = self._norm(dense, p + "post_feedforward_layernorm_1.weight")
            routed = mx.fast.rms_norm(x, self.router_scales[i], self.eps)
            scores = w.linear(p + "router.proj", routed)
            top = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            gates = mx.softmax(mx.take_along_axis(scores, top, axis=-1), axis=-1)
            gates = gates * w[p + "router.per_expert_scale"][top]
            experts = self._norm(x, p + "pre_feedforward_layernorm_2.weight")
            experts = self._experts(p + "experts.switch_glu.", experts, top, gates)
            experts = self._norm(experts, p + "post_feedforward_layernorm_2.weight")
            x = x + self._norm(dense + experts, p + "post_feedforward_layernorm.weight")
            x = x * w[p + "layer_scalar"]
        if batch.prefill:
            x = x[-1:]
        x = self._norm(x, "model.norm.weight")
        group_size, bits = w.quantization["model.embed_tokens"]
        logits = mx.quantized_matmul(
            x,
            w["model.embed_tokens.weight"],
            w["model.embed_tokens.scales"],
            w["model.embed_tokens.biases"],
            transpose=True,
            group_size=group_size,
            bits=bits,
        ).astype(mx.float32)
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    def close(self) -> None:
        self.weights.clear()
