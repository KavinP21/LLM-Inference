"""Per-request sampling for a decode batch.

Each request in a batch brings its own settings, so a greedy agent step and a
temperature-0.7 chat can share one forward pass. Filters run on the top
``CANDIDATES`` tokens, sorted once per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CANDIDATES = 1024


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and 0.0 <= self.temperature <= 10.0):
            raise ValueError("temperature must be between 0 and 10")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1


GREEDY = SamplingParams()


def sample(
    mx: Any, logits: Any, params: list[SamplingParams], steps: list[int]
) -> list[int]:
    """Pick one token per row of ``logits`` (``[batch, vocab]``, float32).

    ``steps`` counts tokens each request has generated, which with its seed
    makes seeded sampling reproducible regardless of batch composition.
    """
    tokens = [0] * len(params)
    greedy = [i for i, p in enumerate(params) if p.greedy]
    if greedy:
        picked = mx.argmax(
            logits[mx.array(greedy)] if len(greedy) < len(params) else logits, axis=-1
        )
        for i, token in zip(greedy, picked.tolist()):
            tokens[i] = int(token)
    sampled = [i for i, p in enumerate(params) if not p.greedy]
    if not sampled:
        return tokens
    rows = logits[mx.array(sampled)] if len(sampled) < len(params) else logits
    count = min(CANDIDATES, int(rows.shape[-1]))
    order = mx.argpartition(-rows, kth=count - 1, axis=-1)[:, :count]
    candidates = mx.take_along_axis(rows, order, axis=-1)
    ranked = mx.argsort(-candidates, axis=-1)
    order = mx.take_along_axis(order, ranked, axis=-1)
    candidates = mx.take_along_axis(candidates, ranked, axis=-1)
    chosen = [params[i] for i in sampled]
    temperature = mx.array([[p.temperature] for p in chosen], mx.float32)
    scaled = candidates / temperature
    probs = mx.softmax(scaled, axis=-1)
    keep = mx.ones(scaled.shape, dtype=mx.bool_)
    ranks = mx.arange(count)[None, :]
    top_k = np.array([p.top_k or count for p in chosen], dtype=np.int32)[:, None]
    keep = keep & (ranks < mx.array(top_k))
    min_p = mx.array([[p.min_p] for p in chosen], mx.float32)
    keep = keep & (probs >= min_p * probs[:, :1])
    # A token survives top-p while the mass before it is still under top_p.
    before = mx.cumsum(probs, axis=-1) - probs
    keep = keep & (before < mx.array([[p.top_p] for p in chosen], mx.float32))
    masked = mx.where(keep, scaled, -mx.inf)
    unseeded = [n for n, p in enumerate(chosen) if p.seed is None]
    picks: dict[int, int] = {}
    if unseeded:
        index = mx.array(unseeded)
        draws = mx.random.categorical(masked[index], axis=-1)
        ids = mx.take_along_axis(order[index], draws[:, None], axis=-1)[:, 0]
        picks.update(zip(unseeded, ids.tolist()))
    for n, p in enumerate(chosen):
        if p.seed is not None:
            key = mx.random.key((p.seed * 1_000_003 + steps[sampled[n]]) % (1 << 63))
            draw = mx.random.categorical(masked[n], key=key)
            picks[n] = int(order[n, draw].item())
    for n, i in enumerate(sampled):
        tokens[i] = int(picks[n])
    return tokens
