"""Offline cached teacher forcing using the real paged prefill/decode operators.

No MLX import at module load. A replay owns a private reservation and device
store, never mutates the engine's scheduler, and is not a performance sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from .paged_kv import MlxPagedKVStore
from .runtime import KVBlockPool


def _integers(values: Iterable[int], label: str) -> list[int]:
    result = list(values)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{label} must contain integers, not coercible values")
    return [int(v) for v in result]


def replay_cached(
    engine,
    input_ids: Iterable[int],
    teacher_tokens: Iterable[int],
    positions: Iterable[int] | None = None,
) -> dict:
    """Predict each source continuation token without following candidate errors.

    Position 0 is the final chunked-prefill row. Position p>0 feeds source token
    p-1 to decode_paged_batch with its exact absolute position and growing page
    table. All intermediate operations are materialized, even unobserved rows.
    The final teacher token is a target, not an input. Returned logits retain
    their actual output dtype; no FP32 re-evaluation or tie manipulation occurs.
    """
    prompt = _integers(input_ids, "prompt")
    tokens = _integers(teacher_tokens, "teacher tokens")
    observed = (
        list(range(len(tokens)))
        if positions is None
        else _integers(positions, "positions")
    )
    vocab = engine.model.config.vocab_size
    if (
        not prompt
        or not tokens
        or not observed
        or any(t < 0 or t >= vocab for t in prompt + tokens)
        or len(prompt) + len(tokens) > engine.max_model_length
        or observed != sorted(set(observed))
        or observed[0] < 0
        or observed[-1] >= len(tokens)
    ):
        raise ValueError("invalid cached replay tokens, context, or positions")
    if engine.cache_pool.allocations:
        raise ValueError("offline cached replay requires an idle engine")
    model, mx = engine.model, engine.model.mx
    pool = KVBlockPool(
        engine.cache_pool.total_blocks * engine.bytes_per_block,
        engine.bytes_per_token,
        engine.block_tokens,
        layout="paged",
    )
    if not pool.reserve(0, len(prompt) + len(tokens)):
        raise ValueError("cached replay exceeds configured KV capacity")
    store = MlxPagedKVStore(
        mx,
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.head_dim,
        block_tokens=engine.block_tokens,
    )
    rows, predictions = [], []
    wanted = set(observed)
    try:
        for start in range(0, len(prompt), engine.prefill_chunk_size):
            end = min(start + engine.prefill_chunk_size, len(prompt))
            pool.ensure_tokens(0, end)
            logits = model.forward_paged_chunk(
                prompt[start:end],
                start_position=start,
                block_table=pool.block_table(0),
                store=store,
            )
            mx.eval(logits)
        for position in range(len(tokens)):
            if position:
                absolute = len(prompt) + position - 1
                pool.ensure_tokens(0, absolute + 1)
                logits = model.decode_paged_batch(
                    [tokens[position - 1]],
                    positions=[absolute],
                    block_tables=[pool.block_table(0)],
                    store=store,
                )
            row = logits[-1]
            predictions.append(int(mx.argmax(row).item()))
            if position in wanted:
                rows.append(np.array(row))
        result = {
            "positions": observed,
            "logits": np.stack(rows),
            "greedy_tokens": predictions,
            "kv_peak_materialized_blocks": store.peak_allocated_blocks,
            "kv_peak_allocated_blocks": pool.peak_allocated_blocks,
        }
    finally:
        store.clear()
        pool.release(0)
    result["cache_reclaimed"] = (
        store.allocated_bytes == 0
        and store.allocated_blocks == 0
        and pool.stats()["allocated_blocks"] == pool.stats()["reserved_blocks"] == 0
    )
    if not result["cache_reclaimed"]:
        raise RuntimeError("cached replay leaked its private KV reservation")
    return result
