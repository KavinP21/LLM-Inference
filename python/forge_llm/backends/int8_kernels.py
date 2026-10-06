"""W8A16 linear primitives with no persistent dequantized weight copy."""

from __future__ import annotations


class Int8LinearKernels:
    """One SIMD group per output channel, vectorized coalesced INT8 loads.

    FP32 accumulation; reconstructed weights round to FP16 to match the
    dequantized MLX GEMM baseline. Four channels share a 128-thread group.
    This GEMV-style kernel is for small decode batches, not prefill GEMMs.
    """

    max_rows = 16

    def __init__(self, mx) -> None:
        self.mx = mx
        self.reconstruction_kernel = mx.fast.metal_kernel(
            name="forge_int8_reconstruct",
            input_names=["weight", "scales"],
            output_names=["out"],
            source="""
                uint index = thread_position_in_grid.x;
                if (index >= N * K) return;
                out[index] = half(float(weight[index]) * scales[index / K]);
            """,
        )
        self.kernel = mx.fast.metal_kernel(
            name="forge_int8_linear",
            input_names=["x", "weight", "scales"],
            output_names=["out"],
            source="""
                uint lane = thread_position_in_grid.x;
                uint channel = thread_position_in_grid.y;
                if (channel >= N) return;
                float scale = scales[channel];
                float sums[M];
                for (uint row = 0; row < M; ++row) sums[row] = 0.0f;
                for (uint base = lane * 4; base < K; base += 128) {
                    if (K % 4 == 0) {
                        char4 packed = reinterpret_cast<const device char4*>(weight)[channel * (K / 4) + base / 4];
                        float4 w = float4(half4(float4(packed) * scale));
                        for (uint row = 0; row < M; ++row) {
                            float4 activation = float4(reinterpret_cast<const device half4*>(x)[row * (K / 4) + base / 4]);
                            sums[row] += dot(activation, w);
                        }
                    } else {
                        for (uint i = 0; i < 4 && base + i < K; ++i) {
                            uint k = base + i;
                            float w = float(half(float(weight[channel * K + k]) * scale));
                            for (uint row = 0; row < M; ++row)
                                sums[row] += float(x[row * K + k]) * w;
                        }
                    }
                }
                for (uint row = 0; row < M; ++row) {
                    float sum = simd_sum(sums[row]);
                    if (lane == 0) out[row * N + channel] = half(sum);
                }
            """,
        )

    def linear(self, x, weight, scales):
        mx = self.mx
        if (
            x.dtype != mx.float16
            or weight.dtype != mx.int8
            or scales.dtype != mx.float32
        ):
            raise ValueError(
                "INT8 Metal linear requires FP16 activations, INT8 weights and FP32 scales"
            )
        if x.ndim < 1 or weight.ndim != 2 or scales.shape != (weight.shape[0],):
            raise ValueError("invalid INT8 linear ranks or scales")
        if not x.shape[-1] or not all(weight.shape) or not x.size:
            raise ValueError("INT8 linear dimensions must be nonempty")
        rows = x.size // x.shape[-1]
        n, k = weight.shape
        if x.shape[-1] != k or not 0 < rows <= self.max_rows:
            raise ValueError(
                "INT8 Metal linear supports 1..16 rows with matching input width"
            )
        return self.kernel(
            inputs=[x, weight, scales],
            template=[("K", k), ("N", n), ("M", rows)],
            grid=(32, (n + 3) // 4 * 4, 1),
            threadgroup=(32, 4, 1),
            output_shapes=[(*x.shape[:-1], n)],
            output_dtypes=[mx.float16],
        )[0]

    def reconstruct(self, weight, scales):
        """One-pass ephemeral FP16 reconstruction without FP32 matrix temporaries."""
        mx = self.mx
        if weight.dtype != mx.int8 or scales.dtype != mx.float32 or weight.ndim != 2:
            raise ValueError(
                "INT8 reconstruction requires INT8 weights and FP32 scales"
            )
        n, k = weight.shape
        if not n or not k or scales.shape != (n,):
            raise ValueError("invalid INT8 reconstruction shape")
        return self.reconstruction_kernel(
            inputs=[weight, scales],
            template=[("N", n), ("K", k)],
            grid=(n * k, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[weight.shape],
            output_dtypes=[mx.float16],
        )[0]
