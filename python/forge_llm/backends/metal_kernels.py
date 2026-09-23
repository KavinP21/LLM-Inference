"""Shape-specialized custom Metal kernels used by the MLX Qwen runtime."""

from __future__ import annotations

from typing import Any


class MetalKernelSuite:
    """Own JIT kernel definitions and their shape-specialized launch contracts."""

    def __init__(self, mx: Any) -> None:
        self.mx = mx
        self._residual_rms_norm_kernel = mx.fast.metal_kernel(
            name="forge_residual_rms_norm",
            input_names=["residual", "branch", "weight", "epsilon"],
            output_names=["residual_out", "normalized"],
            source=r"""
                uint row = threadgroup_position_in_grid.x;
                uint lane = thread_index_in_threadgroup;
                uint base = row * HIDDEN;
                float sum_squares = 0.0f;

                for (uint column = lane; column < HIDDEN;
                     column += threads_per_threadgroup.x) {
                    uint index = base + column;
                    float value = float(residual[index]) + float(branch[index]);
                    residual_out[index] = T(value);
                    sum_squares += value * value;
                }

                sum_squares = simd_sum(sum_squares);
                threadgroup float partials[32];
                if (thread_index_in_simdgroup == 0) {
                    partials[simdgroup_index_in_threadgroup] = sum_squares;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simdgroup_index_in_threadgroup == 0) {
                    float total = lane < simdgroups_per_threadgroup
                        ? partials[lane] : 0.0f;
                    total = simd_sum(total);
                    if (lane == 0) {
                        partials[0] = metal::rsqrt(
                            total / float(HIDDEN) + epsilon[0]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inverse_rms = partials[0];
                for (uint column = lane; column < HIDDEN;
                     column += threads_per_threadgroup.x) {
                    uint index = base + column;
                    normalized[index] = T(
                        float(residual_out[index]) * inverse_rms
                        * float(weight[column]));
                }
            """,
        )
        self._swiglu_kernel = mx.fast.metal_kernel(
            name="forge_swiglu",
            input_names=["gate", "up"],
            output_names=["output"],
            source=r"""
                uint index = thread_position_in_grid.x;
                if (index < SIZE) {
                    float value = float(gate[index]);
                    float silu = value / (1.0f + metal::exp(-value));
                    output[index] = T(silu * float(up[index]));
                }
            """,
        )
        self._rope_write_prefill_kernel = mx.fast.metal_kernel(
            name="forge_rope_write_prefill",
            input_names=["query", "key", "value", "pages", "frequencies", "metadata"],
            output_names=["query_out", "key_out", "pages_out"],
            source=r"""
                uint index = thread_position_in_grid.x;
                uint token_count = query_shape[0];
                uint query_elements = token_count * Q_HEADS * HEAD_DIM;
                uint page_elements = pages_shape[0] * 2 * BLOCK_TOKENS
                    * KV_HEADS * HEAD_DIM;
                int start_position = metadata[0];
                int first_logical_block = metadata[1];
                constexpr uint half_dim = HEAD_DIM / 2;

                if (index < query_elements) {
                    uint dimension = index % HEAD_DIM;
                    uint item = index / HEAD_DIM;
                    uint token = item / Q_HEADS;
                    uint partner = dimension < half_dim
                        ? index + half_dim : index - half_dim;
                    float angle = float(start_position + int(token))
                        * frequencies[dimension % half_dim];
                    float rotated = dimension < half_dim
                        ? -float(query[partner]) : float(query[partner]);
                    query_out[index] = T(
                        float(query[index]) * metal::cos(angle)
                        + rotated * metal::sin(angle));
                }

                if (index < page_elements) {
                    constexpr uint head_stride = HEAD_DIM;
                    constexpr uint token_stride = KV_HEADS * HEAD_DIM;
                    constexpr uint plane_stride = BLOCK_TOKENS * token_stride;
                    constexpr uint page_stride = 2 * plane_stride;
                    uint page = index / page_stride;
                    uint within_page = index % page_stride;
                    uint plane = within_page / plane_stride;
                    uint within_plane = within_page % plane_stride;
                    uint token_offset = within_plane / token_stride;
                    uint within_token = within_plane % token_stride;
                    uint kv_head = within_token / head_stride;
                    uint dimension = within_token % head_stride;
                    int logical_position =
                        (first_logical_block + int(page)) * BLOCK_TOKENS
                        + int(token_offset);
                    bool appended = logical_position >= start_position
                        && logical_position < start_position + int(token_count);

                    if (!appended) {
                        pages_out[index] = pages[index];
                    } else {
                        uint token = uint(logical_position - start_position);
                        uint source = (token * KV_HEADS + kv_head) * HEAD_DIM
                            + dimension;
                        if (plane == 0) {
                            uint partner = dimension < half_dim
                                ? source + half_dim : source - half_dim;
                            float angle = float(logical_position)
                                * frequencies[dimension % half_dim];
                            float rotated = dimension < half_dim
                                ? -float(key[partner]) : float(key[partner]);
                            T result = T(
                                float(key[source]) * metal::cos(angle)
                                + rotated * metal::sin(angle));
                            pages_out[index] = result;
                            key_out[source] = result;
                        } else {
                            pages_out[index] = value[source];
                        }
                    }
                }
            """,
        )
        self._rope_write_decode_kernel = mx.fast.metal_kernel(
            name="forge_rope_write_decode",
            input_names=["query", "key", "value", "pages", "frequencies", "positions"],
            output_names=["query_out", "key_out", "pages_out"],
            source=r"""
                uint index = thread_position_in_grid.x;
                uint batch_size = query_shape[0];
                uint query_elements = batch_size * Q_HEADS * HEAD_DIM;
                uint page_elements = batch_size * 2 * BLOCK_TOKENS
                    * KV_HEADS * HEAD_DIM;
                constexpr uint half_dim = HEAD_DIM / 2;

                if (index < query_elements) {
                    uint dimension = index % HEAD_DIM;
                    uint item = index / HEAD_DIM;
                    uint batch = item / Q_HEADS;
                    uint partner = dimension < half_dim
                        ? index + half_dim : index - half_dim;
                    float angle = float(positions[batch])
                        * frequencies[dimension % half_dim];
                    float rotated = dimension < half_dim
                        ? -float(query[partner]) : float(query[partner]);
                    query_out[index] = T(
                        float(query[index]) * metal::cos(angle)
                        + rotated * metal::sin(angle));
                }

                if (index < page_elements) {
                    constexpr uint head_stride = HEAD_DIM;
                    constexpr uint token_stride = KV_HEADS * HEAD_DIM;
                    constexpr uint plane_stride = BLOCK_TOKENS * token_stride;
                    constexpr uint page_stride = 2 * plane_stride;
                    uint batch = index / page_stride;
                    uint within_page = index % page_stride;
                    uint plane = within_page / plane_stride;
                    uint within_plane = within_page % plane_stride;
                    uint token_offset = within_plane / token_stride;
                    uint within_token = within_plane % token_stride;
                    uint kv_head = within_token / head_stride;
                    uint dimension = within_token % head_stride;
                    uint source = (batch * KV_HEADS + kv_head) * HEAD_DIM
                        + dimension;

                    if (token_offset != uint(positions[batch]) % BLOCK_TOKENS) {
                        pages_out[index] = pages[index];
                    } else if (plane == 0) {
                        uint partner = dimension < half_dim
                            ? source + half_dim : source - half_dim;
                        float angle = float(positions[batch])
                            * frequencies[dimension % half_dim];
                        float rotated = dimension < half_dim
                            ? -float(key[partner]) : float(key[partner]);
                        T result = T(
                            float(key[source]) * metal::cos(angle)
                            + rotated * metal::sin(angle));
                        pages_out[index] = result;
                        key_out[source] = result;
                    } else {
                        pages_out[index] = value[source];
                    }
                }
            """,
        )
        self._paged_decode_attention_kernel = mx.fast.metal_kernel(
            name="forge_paged_decode_attention",
            input_names=[
                "query",
                "pages",
                "block_tables",
                "lengths",
                "scale",
                "softcap",
            ],
            output_names=["output"],
            source=r"""
                uint lane = thread_index_in_simdgroup;
                uint query_head = thread_position_in_grid.y;
                uint batch = thread_position_in_grid.z;
                constexpr uint segments = (HEAD_DIM + 31) / 32;
                constexpr uint groups = Q_HEADS / KV_HEADS;
                constexpr uint token_stride = KV_HEADS * HEAD_DIM;
                constexpr uint plane_stride = BLOCK_TOKENS * token_stride;
                constexpr uint page_stride = 2 * plane_stride;
                uint kv_head = query_head / groups;
                uint table_width = block_tables_shape[1];
                uint token_count = uint(lengths[batch]);
                float running_max = -INFINITY;
                float running_sum = 0.0f;
                float accumulators[8];
                for (uint segment = 0; segment < segments; ++segment) {
                    accumulators[segment] = 0.0f;
                }

                uint first_token = WINDOW_SIZE > 0 && token_count > WINDOW_SIZE
                    ? token_count - WINDOW_SIZE : 0;
                for (uint token = first_token; token < token_count; ++token) {
                    uint logical_block = token / BLOCK_TOKENS;
                    uint token_offset = token % BLOCK_TOKENS;
                    uint page = uint(
                        block_tables[batch * table_width + logical_block]);
                    uint key_base = page * page_stride
                        + token_offset * token_stride + kv_head * HEAD_DIM;
                    float score = 0.0f;
                    for (uint dimension = lane; dimension < HEAD_DIM;
                         dimension += 32) {
                        uint query_index =
                            (batch * Q_HEADS + query_head) * HEAD_DIM + dimension;
                        score += float(query[query_index])
                            * float(pages[key_base + dimension]);
                    }
                    score = simd_sum(score) * scale[0];
                    if (USE_SOFTCAP) {
                        score = metal::tanh(score / softcap[0]) * softcap[0];
                    }
                    float next_max = metal::max(running_max, score);
                    float previous_scale = metal::exp(running_max - next_max);
                    float probability = metal::exp(score - next_max);
                    uint value_base = key_base + plane_stride;
                    for (uint segment = 0; segment < segments; ++segment) {
                        uint dimension = lane + segment * 32;
                        if (dimension < HEAD_DIM) {
                            accumulators[segment] =
                                accumulators[segment] * previous_scale
                                + probability * float(pages[value_base + dimension]);
                        }
                    }
                    running_sum = running_sum * previous_scale + probability;
                    running_max = next_max;
                }

                uint output_base =
                    (batch * Q_HEADS + query_head) * HEAD_DIM;
                for (uint segment = 0; segment < segments; ++segment) {
                    uint dimension = lane + segment * 32;
                    if (dimension < HEAD_DIM) {
                        output[output_base + dimension] = T(
                            accumulators[segment] / running_sum);
                    }
                }
            """,
        )

    def residual_rms_norm(
        self, residual: Any, branch: Any, weight: Any, epsilon: float
    ) -> tuple[Any, Any]:
        if residual.shape != branch.shape:
            raise ValueError("residual fusion inputs must have identical shapes")
        if len(residual.shape) < 1 or int(residual.shape[-1]) != int(weight.shape[0]):
            raise ValueError("RMSNorm weight does not match the hidden dimension")
        hidden = int(residual.shape[-1])
        rows = residual.size // hidden
        eps = self.mx.array([epsilon], dtype=self.mx.float32)
        outputs = self._residual_rms_norm_kernel(
            inputs=[residual, branch, weight, eps],
            template=[("T", residual.dtype), ("HIDDEN", hidden)],
            grid=(rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[residual.shape, residual.shape],
            output_dtypes=[residual.dtype, residual.dtype],
        )
        return outputs[0], outputs[1]

    def swiglu(self, gate: Any, up: Any) -> Any:
        if gate.shape != up.shape:
            raise ValueError("SwiGLU inputs must have identical shapes")
        return self._swiglu_kernel(
            inputs=[gate, up],
            template=[("T", gate.dtype), ("SIZE", gate.size)],
            grid=(gate.size, 1, 1),
            threadgroup=(min(256, gate.size), 1, 1),
            output_shapes=[gate.shape],
            output_dtypes=[gate.dtype],
        )[0]

    def rope_write_prefill(
        self,
        query: Any,
        key: Any,
        value: Any,
        pages: Any,
        frequencies: Any,
        *,
        start_position: int,
        first_logical_block: int,
        block_tokens: int,
    ) -> tuple[Any, Any, Any]:
        token_count, query_heads, head_dim = (int(item) for item in query.shape)
        if tuple(key.shape) != tuple(value.shape) or int(key.shape[0]) != token_count:
            raise ValueError("RoPE/cache-write K/V inputs do not align")
        kv_heads = int(key.shape[1])
        if int(key.shape[2]) != head_dim or head_dim % 2:
            raise ValueError("RoPE requires matching even head dimensions")
        metadata = self.mx.array(
            [start_position, first_logical_block], dtype=self.mx.int32
        )
        grid_size = max(query.size, pages.size)
        outputs = self._rope_write_prefill_kernel(
            inputs=[query, key, value, pages, frequencies, metadata],
            template=[
                ("T", query.dtype),
                ("Q_HEADS", query_heads),
                ("KV_HEADS", kv_heads),
                ("HEAD_DIM", head_dim),
                ("BLOCK_TOKENS", block_tokens),
            ],
            grid=(grid_size, 1, 1),
            threadgroup=(min(256, grid_size), 1, 1),
            output_shapes=[query.shape, key.shape, pages.shape],
            output_dtypes=[query.dtype, key.dtype, pages.dtype],
        )
        return outputs[0], outputs[1], outputs[2]

    def rope_write_decode(
        self,
        query: Any,
        key: Any,
        value: Any,
        pages: Any,
        frequencies: Any,
        *,
        positions: Any,
        block_tokens: int,
    ) -> tuple[Any, Any, Any]:
        batch_size, query_heads, head_dim = (int(item) for item in query.shape)
        if tuple(key.shape) != tuple(value.shape) or int(key.shape[0]) != batch_size:
            raise ValueError("batched RoPE/cache-write K/V inputs do not align")
        kv_heads = int(key.shape[1])
        if int(key.shape[2]) != head_dim or tuple(pages.shape[:1]) != (batch_size,):
            raise ValueError("batched RoPE/cache-write shapes do not align")
        grid_size = max(query.size, pages.size)
        outputs = self._rope_write_decode_kernel(
            inputs=[query, key, value, pages, frequencies, positions],
            template=[
                ("T", query.dtype),
                ("Q_HEADS", query_heads),
                ("KV_HEADS", kv_heads),
                ("HEAD_DIM", head_dim),
                ("BLOCK_TOKENS", block_tokens),
            ],
            grid=(grid_size, 1, 1),
            threadgroup=(min(256, grid_size), 1, 1),
            output_shapes=[query.shape, key.shape, pages.shape],
            output_dtypes=[query.dtype, key.dtype, pages.dtype],
        )
        return outputs[0], outputs[1], outputs[2]

    def paged_decode_attention(
        self,
        query: Any,
        pages: Any,
        block_tables: Any,
        lengths: Any,
        *,
        block_tokens: int,
        scale: float,
        window_size: int = 0,
        softcap: float = 0.0,
    ) -> Any:
        if len(query.shape) != 3 or len(pages.shape) != 5:
            raise ValueError("paged decode attention received invalid ranks")
        batch, query_heads, head_dim = (int(item) for item in query.shape)
        kv_heads = int(pages.shape[3])
        if (
            int(pages.shape[1]) != 2
            or int(pages.shape[2]) != block_tokens
            or int(pages.shape[4]) != head_dim
            or query_heads % kv_heads
            or head_dim > 256
        ):
            raise ValueError("paged decode attention shapes are unsupported")
        return self._paged_decode_attention_kernel(
            inputs=[
                query,
                pages,
                block_tables,
                lengths,
                self.mx.array([scale], dtype=self.mx.float32),
                self.mx.array([softcap], dtype=self.mx.float32),
            ],
            template=[
                ("T", query.dtype),
                ("Q_HEADS", query_heads),
                ("KV_HEADS", kv_heads),
                ("HEAD_DIM", head_dim),
                ("BLOCK_TOKENS", block_tokens),
                ("WINDOW_SIZE", max(0, int(window_size))),
                ("USE_SOFTCAP", int(softcap > 0.0)),
            ],
            grid=(32, query_heads, batch),
            threadgroup=(32, 1, 1),
            output_shapes=[query.shape],
            output_dtypes=[query.dtype],
        )[0]
