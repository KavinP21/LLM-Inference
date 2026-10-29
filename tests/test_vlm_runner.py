"""Runner checks against the real local checkpoints; skipped when they're absent."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "models/qwen3.5-4b-mlx"
GEMMA = ROOT / "models/gemma-4-26b-a4b-mlx"
GEMMA_DRAFT = ROOT / "models/gemma-4-26b-a4b-assistant"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def runner(path, **options):
    if not (path / "config.json").is_file():
        pytest.skip(f"{path.name} is not downloaded")
    from forge_llm.vlm_runner import VlmRunner

    return VlmRunner(path, **options)


@pytest.fixture(scope="module")
def qwen():
    loaded = runner(QWEN, max_context_tokens=8192)
    yield loaded
    loaded.close()


SYSTEM = {"role": "system", "content": "You answer briefly. " * 80}


def test_follow_up_turns_reuse_the_session_cache(qwen):
    messages = [SYSTEM, {"role": "user", "content": "Name a color."}]
    first = qwen.generate(messages, max_tokens=24)
    messages += [
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": "Name another one."},
    ]
    second = qwen.generate(messages, max_tokens=24)
    assert second.metrics["cache_source"] == "session"
    assert second.cached_tokens >= first.input_tokens
    assert second.metrics["prefilled_tokens"] < 40
    assert second.text


def test_new_conversation_restores_the_shared_prefix(qwen):
    qwen.generate([SYSTEM, {"role": "user", "content": "Name a fruit."}], max_tokens=8)
    result = qwen.generate(
        [SYSTEM, {"role": "user", "content": "Name a tree."}], max_tokens=8
    )
    assert result.metrics["cache_source"] == "snapshot"
    assert result.cached_tokens > 200


def test_edited_history_falls_back_to_a_fresh_prompt(qwen):
    base = [{"role": "user", "content": "Say hello."}]
    reply = qwen.generate(base, max_tokens=8)
    edited = [
        {"role": "user", "content": "Say goodbye."},
        {"role": "assistant", "content": reply.text},
        {"role": "user", "content": "Again."},
    ]
    assert qwen.generate(edited, max_tokens=8).metrics["cache_source"] != "session"


def test_cancellation_drops_the_partial_session(qwen):
    messages = [{"role": "user", "content": "Count from one to fifty."}]
    with pytest.raises(InterruptedError):
        qwen.generate(messages, max_tokens=200, cancelled=lambda: True)
    assert qwen.generate(messages, max_tokens=8).text


def test_thinking_budget_closes_reasoning(qwen):
    result = qwen.generate(
        [{"role": "user", "content": "Is 91 prime? Answer yes or no."}],
        max_tokens=1500,
        effort="low",
    )
    assert result.reasoning and result.reasoning_finished
    assert "no" in result.text.lower()


def test_context_limit_is_enforced(qwen):
    with pytest.raises(ValueError, match="context"):
        qwen.generate([{"role": "user", "content": "x " * 9000}], max_tokens=16)


def test_gemma_tool_round_trip_with_drafting():
    gemma = runner(GEMMA, draft_path=GEMMA_DRAFT if GEMMA_DRAFT.is_dir() else None)
    try:
        messages = [
            {"role": "system", "content": "Use the tools to answer."},
            {
                "role": "user",
                "content": "What does README.md say the project is called?",
            },
        ]
        call = gemma.generate(messages, max_tokens=120, tools=TOOLS)
        assert call.dialect == "gemma4"
        assert "<|tool_call>call:read_file" in call.text
        messages += [
            {"role": "assistant", "content": call.text},
            {"role": "tool", "content": "# Forge LLM\nAn inference engine."},
        ]
        answer = gemma.generate(messages, max_tokens=60, tools=TOOLS)
        assert answer.metrics["cache_source"] == "session"
        assert "Forge" in answer.text
    finally:
        gemma.close()
