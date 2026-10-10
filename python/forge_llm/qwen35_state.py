"""Portable resource geometry for the pinned Qwen3.5-4B hybrid cache.

This describes upstream MLX-VLM state, not Forge quantization or kernels.
Persistent buffers and speculative history are accounted separately from model
weights and temporary forward-pass arrays. No MLX imports occur in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen35StateGeometry:
    attention_layers: int = 8
    recurrent_layers: int = 24
    kv_heads: int = 4
    attention_head_dim: int = 256
    value_heads: int = 32
    key_heads: int = 16
    key_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel: int = 4
    activation_bytes: int = 2

    @classmethod
    def from_text_config(cls, config: dict[str, Any]) -> Qwen35StateGeometry:
        """Reject layouts outside the explicitly supported dense 4B checkpoint."""
        expected = {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
        }
        if not isinstance(config, dict) or any(
            type(config.get(name)) is not int or config[name] != value
            for name, value in expected.items()
        ):
            raise ValueError("unsupported Qwen3.5-4B hybrid cache geometry")
        layer_types = [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(32)
        ]
        # Converted MLX configs may omit the expanded list; the validated
        # interval above still defines the exact same ordered hybrid layout.
        if config.get("layer_types", layer_types) != layer_types:
            raise ValueError("unsupported Qwen3.5 hybrid layer ordering")
        if config.get("mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("Qwen3.5 recurrent state must use FP32")
        return cls()

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            self.attention_layers
            * 2
            * self.kv_heads
            * self.attention_head_dim
            * self.activation_bytes
        )

    @property
    def recurrent_bytes(self) -> int:
        return (
            self.recurrent_layers
            * self.value_heads
            * self.key_head_dim
            * self.value_head_dim
            * 4
        )

    @property
    def conv_channels(self) -> int:
        return (
            2 * self.key_heads * self.key_head_dim
            + self.value_heads * self.value_head_dim
        )

    @property
    def conv_bytes_per_token(self) -> int:
        return self.recurrent_layers * self.conv_channels * self.activation_bytes

    @property
    def conv_bytes(self) -> int:
        return (self.conv_kernel - 1) * self.conv_bytes_per_token

    @property
    def fixed_bytes(self) -> int:
        return self.recurrent_bytes + self.conv_bytes

    def speculative_extra_bytes(self, block_length: int) -> int:
        """Conservative extra retained buffers while one block is uncommitted.

        MLX-VLM 0.7.6 keeps the pre-block SSM, L-1 intermediate states and
        final state. Above the ordinary one-state allocation, that adds L
        states. Convolution records retain their concatenated causal input;
        allow another materialized current window as well. Attention KV is
        already covered by the request's rounded maximum-output reservation.
        """
        if type(block_length) is not int or not 1 <= block_length <= 9:
            raise ValueError("verification block length must be between 1 and 9")
        return (
            block_length * self.recurrent_bytes
            + self.conv_bytes
            + (block_length + self.conv_kernel - 1) * self.conv_bytes_per_token
        )


def retained_array_bytes(value: Any) -> int:
    """Count logical bytes of distinct retained array objects in a state tree.

    Repeated references to the same array object are counted once. Views may
    overcount shared storage or hide a larger backing allocation: callers must
    include their full owning sources during transactions and detach selected
    slices before releasing those sources' reservation. This accepts MLX arrays
    or portable test doubles without importing MLX.
    """
    seen: set[int] = set()

    def count(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, dict):
            return sum(count(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(count(child) for child in item)
        nbytes = getattr(item, "nbytes", None)
        if type(nbytes) is int:
            if nbytes < 0:
                raise ValueError("negative retained array size")
            return nbytes
        return 0

    return count(value)


def retained_cache_bytes(caches: list[Any]) -> int:
    """Include recurrent transaction records, which ordinary cache.nbytes omits."""
    return retained_array_bytes(
        [
            {
                "keys": getattr(cache, "keys", None),
                "values": getattr(cache, "values", None),
                "arrays": getattr(cache, "cache", None),
                "transaction": getattr(cache, "_speculation", None),
            }
            for cache in caches
        ]
    )
