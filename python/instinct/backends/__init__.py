"""Model adapters and Metal kernels for the MLX engine."""

from .base import mlx_build_info
from .factory import load_model

__all__ = ["load_model", "mlx_build_info"]
