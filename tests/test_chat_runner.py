"""ChatRunner and the worker backend on the real Qwen3.5 checkpoint.

Skips unless the model is downloaded and MLX is available.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("transformers")

from instinct.agents.backends import LocalModelBackend, WorkerConfig
from instinct.agents.protocol import ChatMessage
from instinct.chat_runner import ChatRunner

MODEL = Path(__file__).resolve().parents[1] / "models" / "qwen3.5-4b-mlx"
pytestmark = pytest.mark.skipif(
    not (MODEL / "config.json").is_file(), reason="Qwen3.5 isn't downloaded"
)


@pytest.fixture(scope="module")
def runner():
    with ChatRunner(
        MODEL, max_num_sequences=4, max_context_tokens=16384, kv_cache_bytes=2 << 30
    ) as runner:
        yield runner


def test_follow_up_turn_continues_the_session(runner):
    convo = [
        {"role": "user", "content": "What is 12 times 12? Reply with the number only."}
    ]
    first = runner.generate(convo, max_tokens=32)
    assert "144" in first.text and first.finish_reason == "stop"
    convo += [
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": "Add one. Number only."},
    ]
    second = runner.generate(convo, max_tokens=32)
    assert "145" in second.text
    assert second.metrics["cache_source"] == "session"
    assert second.cached_tokens >= first.input_tokens


def test_thinking_budget_closes_the_thought(runner):
    result = runner.generate(
        [
            {
                "role": "user",
                "content": "Count from 1 to 400 in your head, then say done.",
            }
        ],
        max_tokens=1600,
        effort="low",
    )
    # The low budget is 1,024 thinking tokens; the runner forces the close.
    assert result.reasoning and result.reasoning_finished


def test_cancel_stops_generation(runner):
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    with pytest.raises(InterruptedError):
        runner.generate(
            [{"role": "user", "content": "Write a very long essay about rivers."}],
            max_tokens=4000,
            cancelled=flag.is_set,
        )
    assert runner.engine.stats()["running"] == 0


def test_worker_backend_serves_requests_concurrently():
    config = WorkerConfig(
        str(MODEL),
        str(MODEL),
        backend="mlx",
        max_model_length=8192,
        kv_cache_bytes=1 << 30,
        max_pending=4,
    )
    backend = LocalModelBackend(config)

    async def run():
        await backend.start()
        prompts = ["Name a color.", "Name a fruit.", "Name a planet."]
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                backend.generate([ChatMessage("user", text)], 48, f"request-{i}")
                for i, text in enumerate(prompts)
            )
        )
        together = time.perf_counter() - started
        health = backend.health()
        await backend.close()
        return results, together, health

    results, _, health = asyncio.run(run())
    assert all(result.text for result in results)
    assert health["runner"] == "engine" and health["family"] == "qwen3_5"
    assert health["last_runner_stats"]["requests"] >= 3
