"""How ChatRunner builds each prompt from earlier requests' cached tokens.

A stand-in tokenizer with a Qwen-shaped template (one token per character,
turn markers whole) keeps this free of MLX and model files.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
from instinct.chat_formats import QWEN35
from instinct.chat_runner import (
    MIN_CHECKPOINT_TOKENS,
    ChatRunner,
    _identity,
    _render_key,
    _Session,
)

START, END = "<|im_start|>", "<|im_end|>"
MARKERS = {START: 1, END: 2}
KEY = _render_key(None, False)
SYSTEM = {"role": "system", "content": "Answer carefully. " * 20}


class Tokenizer:
    chat_template = "qwen-like"
    added_tokens_decoder = {
        token: SimpleNamespace(content=text, special=True)
        for text, token in MARKERS.items()
    }

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking, **_
    ):
        text = "".join(f"{START}{m['role']}\n{m['content']}{END}\n" for m in messages)
        if add_generation_prompt:
            text += f"{START}assistant\n<think>\n"
            if not enable_thinking:
                text += "\n</think>\n\n"
        return text

    def encode(self, text, add_special_tokens=False):
        ids, at = [], 0
        while at < len(text):
            marker = next((m for m in MARKERS if text.startswith(m, at)), None)
            if marker:
                ids.append(MARKERS[marker])
                at += len(marker)
            else:
                ids.append(3 + ord(text[at]))
                at += 1
        return ids


def _runner(max_sessions=64):
    runner = ChatRunner.__new__(ChatRunner)
    runner.tokenizer = Tokenizer()
    runner.model_type = "qwen3_5"
    runner.family = QWEN35
    runner.max_sessions = max_sessions
    runner._lock = threading.Lock()
    runner._setup_prompts()
    return runner


def _request(runner, messages, reply, generated=None):
    """What ``generate`` does around the model: build the prompt, then keep
    the session. ``generated`` is the reply's raw text, thinking included."""
    ids, text, source, checkpoints, cut = runner._prepare(messages, None, False, KEY)
    output = runner._encode(reply if generated is None else generated) + [2]
    runner._remember(
        _Session(
            [_identity(m) for m in messages],
            _identity({"role": "assistant", "content": reply}),
            KEY,
            np.asarray(ids + output, dtype=np.int32),
            END,
            cut,
        )
    )
    return SimpleNamespace(ids=ids, source=source, checkpoints=checkpoints, cut=cut)


def test_the_turn_after_a_tool_loop_branches_after_the_users_message():
    runner = _runner()
    ask = {"role": "user", "content": "Find the bug in parse()."}
    first = _request(runner, [SYSTEM, ask], '<tool_call>{"name": "read"}')
    assert first.source == "cold"
    # The cut sits after the user's message, before the generation prompt.
    opening = runner._encode(f"{START}assistant\n<think>\n\n</think>\n\n")
    assert first.cut == len(first.ids) - len(opening) >= MIN_CHECKPOINT_TOKENS
    assert first.cut in first.checkpoints

    history = [
        SYSTEM,
        ask,
        {"role": "assistant", "content": '<tool_call>{"name": "read"}'},
    ]
    step = _request(
        runner, history + [{"role": "tool", "content": "x = 1"}], "Off by one."
    )
    assert step.source == "session" and step.cut == 0  # Tool results don't cut.

    # The chat keeps only the final answer, so the next turn can't continue
    # the tool loop. It picks up at the end of the user's first message.
    turn = [SYSTEM, ask, {"role": "assistant", "content": "Off by one."}]
    turn.append({"role": "user", "content": "Fix it."})
    second = _request(runner, turn, "Done.")
    assert second.source == "branch"
    assert second.ids[: first.cut] == first.ids[: first.cut]
    assert second.ids == runner._encode(runner.render(turn, None, False))
    assert second.cut and second.checkpoints == [second.cut]


def test_continuing_a_reply_beats_branching_before_it():
    runner = _runner()
    ask = {"role": "user", "content": "Name a prime."}
    thought = "<think>\nTwo is prime.\n</think>\n\n7"
    first = _request(runner, [SYSTEM, ask], "7", generated=thought)
    turn = [SYSTEM, ask, {"role": "assistant", "content": "7"}]
    turn.append({"role": "user", "content": "Another."})
    second = _request(runner, turn, "11")
    # The reply as generated, thinking and all, is what the engine cached.
    assert second.source == "session"
    assert second.ids[: len(first.ids)] == first.ids
    assert (
        second.ids[len(first.ids) : len(first.ids) + 1] == runner._encode(thought)[:1]
    )


def test_branch_points_outlast_a_long_tool_loop():
    runner = _runner(max_sessions=3)
    ask = {"role": "user", "content": "Refactor the loader."}
    _request(runner, [SYSTEM, ask], "call 0")
    history = [SYSTEM, ask, {"role": "assistant", "content": "call 0"}]
    for step in range(10):
        history.append({"role": "tool", "content": f"result {step}"})
        _request(runner, history, f"call {step + 1}")
        history.append({"role": "assistant", "content": f"call {step + 1}"})
    turn = [SYSTEM, ask, {"role": "assistant", "content": "Refactored."}]
    turn.append({"role": "user", "content": "Thanks, now test it."})
    assert _request(runner, turn, "ok").source == "branch"


def test_an_edited_message_starts_cold():
    runner = _runner()
    _request(runner, [SYSTEM, {"role": "user", "content": "Version one."}], "a")
    edited = _request(
        runner, [SYSTEM, {"role": "user", "content": "Version two."}], "b"
    )
    assert edited.source == "cold"
