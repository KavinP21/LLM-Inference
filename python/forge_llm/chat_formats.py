"""Model-family chat details that don't need MLX: thinking markers, native tool
call syntax, history conversion and incremental prompt rendering.

Everything here is plain Python so it can be tested without a GPU.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# Reasoning effort -> thinking token budget. None means "thinking off";
# 0 means thinking on with no budget beyond max_tokens.
EFFORT_BUDGETS: dict[str, int | None] = {
    "off": None,
    "low": 1024,
    "medium": 4096,
    "high": 0,
}


@dataclass(frozen=True)
class ChatFamily:
    name: str
    dialect: str
    think_open: str
    think_close: str
    # Marker that closes an assistant turn in the chat template.
    turn_end: str
    # Sampling the model card recommends, used unless the request overrides it.
    temperature: float
    top_p: float
    top_k: int


GEMMA4 = ChatFamily(
    "gemma4", "gemma4", "<|channel>thought\n", "<channel|>", "<turn|>", 1.0, 0.95, 64
)
QWEN35 = ChatFamily(
    "qwen3_5", "qwen3_coder_xml", "<think>", "</think>", "<|im_end|>", 0.7, 0.8, 20
)
GENERIC = ChatFamily(
    "generic", "json", "<think>", "</think>", "<|im_end|>", 0.7, 0.9, 40
)


def family_for(model_type: str) -> ChatFamily:
    if model_type.startswith("gemma4"):
        return GEMMA4
    if model_type.startswith("qwen3_5"):
        return QWEN35
    return GENERIC


def thinking_budget(effort: str) -> int | None:
    try:
        return EFFORT_BUDGETS[effort]
    except KeyError:
        raise ValueError(f"effort must be one of {sorted(EFFORT_BUDGETS)}") from None


def split_thinking(text: str, family: ChatFamily) -> tuple[str, str, bool]:
    """Return (answer, reasoning, reasoning_finished)."""
    start = text.find(family.think_open)
    if start == -1 or text[:start].strip():
        return text.strip(), "", True
    body = text[start + len(family.think_open) :]
    end = _thinking_end(body, family.think_close)
    if end == -1:
        return "", body.strip(), False
    answer = body[end + len(family.think_close) :]
    return answer.strip(), body[:end].strip(), True


def _thinking_end(body: str, close: str) -> int:
    # Thinking about a file that contains the close marker quotes it, and the
    # tokenizer turns the quote into the real token. Skip occurrences that are
    # immediately closed off by a quote or punctuation, which a real end of
    # thinking never is.
    first = position = body.find(close)
    while position != -1:
        following = body[position + len(close) : position + len(close) + 1]
        if not following or following not in "\"'`),;]":
            return position
        position = body.find(close, position + 1)
    return first


# --- Gemma 4 native tool calls ------------------------------------------------

GEMMA_CALL_OPEN = "<|tool_call>"
GEMMA_CALL_CLOSE = "<tool_call|>"
GEMMA_STRING = '<|"|>'


class GemmaSyntaxError(ValueError):
    pass


class _GemmaArgs:
    """Parser for Gemma 4's argument syntax: ``{key:<|"|>text<|"|>,n:3,xs:[...]}``."""

    def __init__(self, text: str):
        self.text, self.i = text, 0

    def parse(self) -> Any:
        value = self.value()
        if self.i != len(self.text):
            raise GemmaSyntaxError(f"unexpected text after arguments at {self.i}")
        return value

    def value(self) -> Any:
        text = self.text
        if text.startswith(GEMMA_STRING, self.i):
            end = text.find(GEMMA_STRING, self.i + len(GEMMA_STRING))
            if end == -1:
                raise GemmaSyntaxError("unterminated string")
            value = text[self.i + len(GEMMA_STRING) : end]
            self.i = end + len(GEMMA_STRING)
            return value
        if self.i >= len(text):
            raise GemmaSyntaxError("missing value")
        char = text[self.i]
        if char == "{":
            return self.object()
        if char == "[":
            return self.array()
        match = re.compile(r"[^,\]\}]+").match(text, self.i)
        if not match:
            raise GemmaSyntaxError(f"bad value at {self.i}")
        self.i = match.end()
        word = match.group().strip()
        if word in ("true", "false"):
            return word == "true"
        if word in ("null", "None"):
            return None
        try:
            return json.loads(word)
        except ValueError:
            raise GemmaSyntaxError(f"bad literal {word!r}") from None

    def object(self) -> dict:
        self.i += 1
        result: dict[str, Any] = {}
        if self.text.startswith("}", self.i):
            self.i += 1
            return result
        while True:
            if self.text.startswith(GEMMA_STRING, self.i):
                key = self.value()
            else:
                colon = self.text.find(":", self.i)
                if colon == -1:
                    raise GemmaSyntaxError("object key without a value")
                key = self.text[self.i : colon].strip()
                self.i = colon
            if (
                not isinstance(key, str)
                or not key
                or not self.text.startswith(":", self.i)
            ):
                raise GemmaSyntaxError(f"bad object key at {self.i}")
            if key in result:
                raise GemmaSyntaxError(f"duplicate key {key!r}")
            self.i += 1
            result[key] = self.value()
            if self.text.startswith(",", self.i):
                self.i += 1
            elif self.text.startswith("}", self.i):
                self.i += 1
                return result
            else:
                raise GemmaSyntaxError(f"expected , or }} at {self.i}")

    def array(self) -> list:
        self.i += 1
        items: list[Any] = []
        if self.text.startswith("]", self.i):
            self.i += 1
            return items
        while True:
            items.append(self.value())
            if self.text.startswith(",", self.i):
                self.i += 1
            elif self.text.startswith("]", self.i):
                self.i += 1
                return items
            else:
                raise GemmaSyntaxError(f"expected , or ] at {self.i}")


def parse_gemma_arguments(body: str) -> dict:
    value = _GemmaArgs(body).parse()
    if not isinstance(value, dict):
        raise GemmaSyntaxError("call arguments must be an object")
    return value


def _raw_gemma_calls(text: str) -> tuple[list[tuple[str, str]], str]:
    """Split text into ``(name, "{...}")`` call bodies and the text around them."""
    calls, rest, position = [], [], 0
    while True:
        start = text.find(GEMMA_CALL_OPEN, position)
        if start == -1:
            rest.append(text[position:])
            break
        rest.append(text[position:start])
        end = text.find(GEMMA_CALL_CLOSE, start)
        if end == -1:
            raise GemmaSyntaxError("unterminated tool call")
        segment = text[start + len(GEMMA_CALL_OPEN) : end]
        brace = segment.find("{")
        if not segment.startswith("call:") or brace == -1 or not segment.endswith("}"):
            raise GemmaSyntaxError("tool call must look like call:name{...}")
        name = segment[len("call:") : brace]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", name):
            raise GemmaSyntaxError(f"bad function name {name!r}")
        calls.append((name, segment[brace:]))
        position = end + len(GEMMA_CALL_CLOSE)
    return calls, "".join(rest).strip()


def gemma_calls(text: str) -> tuple[list[dict], str]:
    """Extract ``{"name", "arguments"}`` calls and the text left around them."""
    raw, rest = _raw_gemma_calls(text)
    return [
        {"name": name, "arguments": parse_gemma_arguments(body)} for name, body in raw
    ], rest


def to_gemma_messages(messages: Sequence[dict]) -> list[dict]:
    """Rewrite raw assistant call text into the structured form Gemma's
    template expects, and name each following tool result after its call.

    Argument bodies are kept as the exact text the model produced, so a
    re-render reproduces the model's own tokens.
    """
    converted: list[dict] = []
    pending: list[tuple[str, str]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and isinstance(message.get("content"), str):
            content = message["content"]
            try:
                raw, _ = _raw_gemma_calls(content)
            except GemmaSyntaxError:
                raw = []
            raw = [(name, body[1:-1]) for name, body in raw]
            if raw:
                calls = [
                    {
                        "id": f"call_{len(converted)}_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": body},
                    }
                    for index, (name, body) in enumerate(raw)
                ]
                pending = [(call["id"], call["function"]["name"]) for call in calls]
                converted.append(
                    {"role": "assistant", "content": "", "tool_calls": calls}
                )
                continue
            pending = []
            converted.append(dict(message))
        elif role == "tool":
            call_id, name = pending.pop(0) if pending else ("", "tool")
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": message["content"],
                }
            )
        else:
            pending = []
            converted.append(dict(message))
    return converted


# --- Incremental rendering ---------------------------------------------------

_SENTINEL = "\x00\x01forge\x01\x00"


def incremental_suffix(
    render: Callable[[list[dict], bool], str],
    previous: list[dict],
    current: list[dict],
    last_token_text: str,
) -> str | None:
    """Text to append to the cached token stream to reach ``current``.

    ``previous`` is the conversation the cache was built from, ending with the
    assistant reply the model generated. The cache ends at that reply's stop
    token. Rendering ``previous`` followed by a dummy user turn and comparing
    it with the real next render isolates the template's canonical text after
    the reply; everything from the stop token onwards is appended. Returns
    None when ``current`` doesn't extend ``previous`` or the template can't be
    split this way, in which case the caller prefills from scratch.
    """
    if len(current) <= len(previous) or current[: len(previous)] != previous:
        return None
    probe = render(previous + [{"role": "user", "content": _SENTINEL}], True)
    full = render(current, True)
    shared = 0
    limit = min(len(probe), len(full))
    while shared < limit and probe[shared] == full[shared]:
        shared += 1
    common = full[:shared]
    cut = common.rfind(last_token_text) if last_token_text else -1
    if cut == -1:
        return None
    # The stop token must close the previous reply, not appear inside the new turn.
    reply = previous[-1].get("content")
    if isinstance(reply, str) and reply and common.rfind(reply[-64:]) > cut:
        return None
    return common[cut + len(last_token_text) :] + full[shared:]
