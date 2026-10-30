"""OpenAI-compatible ``/v1/models`` and ``/v1/chat/completions``.

Any OpenAI client (a phone app, an IDE, another machine's tools) can talk to
Instinct's models through this, and its requests join the same batches as the
app's own agents. Requests are translated into worker generations; replies
come back as ``chat.completion`` objects or, with ``stream: true``, as
server-sent ``chat.completion.chunk`` events ending in ``data: [DONE]``.

Thinking is reported as ``reasoning_content``. Tool calls are read from each
model's own call syntax (Gemma 4's ``<|tool_call>`` blocks, Qwen's XML, or
JSON) and returned as OpenAI ``tool_calls``; tool history sent back is written
in the same syntax for the model.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..agents.media import MAX_MEDIA_ITEMS, validate_media
from ..agents.protocol import ChatMessage, Generation
from ..chat_formats import (
    GEMMA_STRING,
    ChatFamily,
    family_for,
    gemma_calls,
    split_thinking,
)

MAX_OUTPUT_TOKENS = 32768
EFFORTS = {
    "none": "off",
    "minimal": "off",
    "off": "off",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp|bmp));base64,(.+)$", re.DOTALL)


class OpenAIError(Exception):
    def __init__(
        self, status: int, message: str, kind: str = "invalid_request_error"
    ) -> None:
        super().__init__(message)
        self.status, self.kind = status, kind

    def body(self) -> dict[str, Any]:
        return {"error": {"message": str(self), "type": self.kind, "code": None}}


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    stream: bool
    include_usage: bool
    tools: list[dict] | None
    options: dict[str, Any] = field(default_factory=dict)


def _number(body: dict, name: str, low: float, high: float) -> float | None:
    value = body.get(name)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
    ):
        raise OpenAIError(400, f"{name} must be a number between {low} and {high}")
    return float(value)


def _text(content: Any, role: str) -> tuple[str, tuple[dict, ...]]:
    """Message content as text plus image media."""
    if content is None:
        return "", ()
    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        raise OpenAIError(400, f"{role} content must be a string or a list of parts")
    texts, media = [], []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "image_url" and role == "user":
            url = (part.get("image_url") or {}).get("url", "")
            match = _DATA_URL.match(url) if isinstance(url, str) else None
            if match is None:
                raise OpenAIError(
                    400, "images must be base64 data URLs (data:image/png;base64,...)"
                )
            try:
                data = base64.b64decode(match.group(2), validate=True)
            except (ValueError, binascii.Error):
                raise OpenAIError(400, "an image isn't valid base64") from None
            media.append(
                {
                    "type": "image",
                    "mime": match.group(1),
                    "data": base64.b64encode(data).decode("ascii"),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        else:
            raise OpenAIError(400, f"unsupported content part in a {role} message")
    if len(media) > MAX_MEDIA_ITEMS:
        raise OpenAIError(400, f"at most {MAX_MEDIA_ITEMS} images per message")
    try:
        return "\n".join(texts), validate_media(media)
    except (TypeError, ValueError) as exc:
        raise OpenAIError(400, str(exc)) from None


def _call_text(call: dict, dialect: str) -> str:
    """An OpenAI tool call written in the model's own syntax."""
    function = call.get("function") or {}
    name = function.get("name")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except ValueError:
        raise OpenAIError(400, "tool call arguments must be a JSON object") from None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise OpenAIError(400, "tool calls need a function name and object arguments")
    if dialect == "gemma4":

        def value(item: Any) -> str:
            if isinstance(item, str):
                return f"{GEMMA_STRING}{item}{GEMMA_STRING}"
            if isinstance(item, dict):
                return "{" + ",".join(f"{k}:{value(v)}" for k, v in item.items()) + "}"
            if isinstance(item, list):
                return "[" + ",".join(value(v) for v in item) + "]"
            return json.dumps(item)

        return f"<|tool_call>call:{name}{value(arguments)}<tool_call|>"
    if dialect == "qwen3_coder_xml":
        parameters = "".join(
            f"<parameter={key}>\n{item if isinstance(item, str) else json.dumps(item)}\n</parameter>\n"
            for key, item in arguments.items()
        )
        return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def parse_request(body: Any, *, dialect: str, default_model: str) -> ChatRequest:
    if not isinstance(body, dict):
        raise OpenAIError(400, "the request body must be a JSON object")
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise OpenAIError(400, "messages must be a non-empty list")
    if body.get("n", 1) != 1:
        raise OpenAIError(400, "only n=1 is supported")
    messages: list[ChatMessage] = []
    for message in raw:
        role = message.get("role") if isinstance(message, dict) else None
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise OpenAIError(
                400, "message roles must be system, user, assistant or tool"
            )
        text, media = _text(message.get("content"), role)
        if role == "assistant" and message.get("tool_calls"):
            calls = "\n".join(
                _call_text(call, dialect) for call in message["tool_calls"]
            )
            text = f"{text}\n{calls}".strip()
        messages.append(ChatMessage(role, text, media))
    tokens = body.get("max_completion_tokens", body.get("max_tokens", 4096))
    if type(tokens) is not int or not 1 <= tokens <= MAX_OUTPUT_TOKENS:
        raise OpenAIError(
            400, f"max_tokens must be an integer from 1 to {MAX_OUTPUT_TOKENS}"
        )
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or len(tools) > 32:
            raise OpenAIError(400, "tools must be a list of at most 32 functions")
        normalized = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(
                function.get("name"), str
            ):
                raise OpenAIError(400, "each tool needs a function with a name")
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "description": function.get("description") or "",
                        "parameters": function.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        tools = normalized if body.get("tool_choice") != "none" else None
    options: dict[str, Any] = {}
    effort = body.get("reasoning_effort")
    if effort is not None:
        if effort not in EFFORTS:
            raise OpenAIError(400, "reasoning_effort must be none, low, medium or high")
        options["effort"] = EFFORTS[effort]
    for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = _number(body, name, low, high)
        if value is not None:
            options[name] = value
    if body.get("top_k") is not None:
        if type(body["top_k"]) is not int or body["top_k"] < 0:
            raise OpenAIError(400, "top_k must be a non-negative integer")
        options["top_k"] = body["top_k"]
    if body.get("seed") is not None:
        if type(body["seed"]) is not int:
            raise OpenAIError(400, "seed must be an integer")
        options["seed"] = body["seed"]
    stream_options = body.get("stream_options") or {}
    model = body.get("model") or default_model
    if not isinstance(model, str):
        raise OpenAIError(400, "model must be a string")
    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=tokens,
        stream=bool(body.get("stream", False)),
        include_usage=bool(
            isinstance(stream_options, dict) and stream_options.get("include_usage")
        ),
        tools=tools,
        options=options,
    )


def tool_calls(text: str, dialect: str) -> tuple[str, list[dict]]:
    """Split a reply into its text and OpenAI-shaped tool calls."""
    found: list[tuple[str, dict]] = []
    rest = text
    if dialect == "gemma4":
        try:
            calls, rest = gemma_calls(text)
            found = [(call["name"], call["arguments"]) for call in calls]
        except ValueError:
            return text, []
    else:
        pieces = []
        for match in re.finditer(
            r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
        ):
            body = match.group(1)
            function = re.fullmatch(
                r"<function=([A-Za-z_][\w\-]*)>(.*)</function>", body, re.DOTALL
            )
            if function:
                arguments = {}
                for parameter in re.finditer(
                    r"<parameter=([\w\-]+)>\n?(.*?)\n?</parameter>",
                    function.group(2),
                    re.DOTALL,
                ):
                    raw = parameter.group(2)
                    try:
                        arguments[parameter.group(1)] = json.loads(raw)
                    except ValueError:
                        arguments[parameter.group(1)] = raw
                found.append((function.group(1), arguments))
            else:
                try:
                    call = json.loads(body)
                    found.append((call["name"], call.get("arguments") or {}))
                except (ValueError, KeyError, TypeError):
                    continue
            pieces.append(match.group(0))
        for piece in pieces:
            rest = rest.replace(piece, "")
        rest = rest.strip()
    return rest, [
        {
            "id": "call_" + secrets.token_hex(12),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        for name, arguments in found
    ]


def usage(generation: Generation) -> dict[str, Any]:
    metrics = generation.metrics or {}
    return {
        "prompt_tokens": generation.input_tokens,
        "completion_tokens": generation.output_tokens,
        "total_tokens": generation.input_tokens + generation.output_tokens,
        "prompt_tokens_details": {
            "cached_tokens": int(metrics.get("reused_tokens") or 0)
        },
    }


class Completion:
    """One chat completion: drives the generation and shapes the replies."""

    def __init__(self, request: ChatRequest, *, model: str, family: ChatFamily) -> None:
        self.request, self.model, self.family = request, model, family
        self.id = "chatcmpl-" + secrets.token_hex(12)
        self.created = int(time.time())
        self._sent_answer = self._sent_reasoning = 0
        # Qwen's template opens the thought itself, so its stream starts inside it.
        self._opens = (
            family.name == "qwen3_5" and request.options.get("effort", "off") != "off"
        )

    def _chunk(
        self, delta: dict, finish: str | None = None, usage_: dict | None = None
    ) -> dict:
        chunk = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": []
            if usage_ is not None and not delta
            else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage_ is not None:
            chunk["usage"] = usage_
        return chunk

    def deltas(self, partial: str) -> list[dict]:
        """Chunks for the text produced since the last call. With tools the
        answer is held back, since it may turn out to be a call."""
        if self._opens and not partial.startswith(self.family.think_open):
            partial = self.family.think_open + partial
        answer, reasoning, _ = split_thinking(partial, self.family)
        chunks = []
        if len(reasoning) > self._sent_reasoning:
            chunks.append(
                self._chunk({"reasoning_content": reasoning[self._sent_reasoning :]})
            )
            self._sent_reasoning = len(reasoning)
        if (
            not self.request.tools
            and len(answer) > self._sent_answer
            and answer.startswith(answer[: self._sent_answer])
        ):
            chunks.append(self._chunk({"content": answer[self._sent_answer :]}))
            self._sent_answer = len(answer)
        return chunks

    def _finish(self, generation: Generation) -> tuple[str, list[dict], str]:
        text, calls = generation.text, []
        if self.request.tools:
            text, calls = tool_calls(
                generation.text, generation.dialect or self.family.dialect
            )
        finish = (
            "tool_calls"
            if calls
            else ("length" if generation.finish_reason == "length" else "stop")
        )
        return text, calls, finish

    def final_chunks(self, generation: Generation) -> list[dict]:
        text, calls, finish = self._finish(generation)
        chunks = []
        if generation.reasoning and len(generation.reasoning) > self._sent_reasoning:
            chunks.append(
                self._chunk(
                    {"reasoning_content": generation.reasoning[self._sent_reasoning :]}
                )
            )
        if text and len(text) > self._sent_answer:
            chunks.append(self._chunk({"content": text[self._sent_answer :]}))
        if calls:
            chunks.append(
                self._chunk(
                    {
                        "tool_calls": [
                            {"index": i, **call} for i, call in enumerate(calls)
                        ]
                    }
                )
            )
        chunks.append(self._chunk({}, finish))
        if self.request.include_usage:
            chunks.append(self._chunk({}, usage_=usage(generation)))
        return chunks

    def response(self, generation: Generation) -> dict:
        text, calls, finish = self._finish(generation)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": text or (None if calls else ""),
        }
        if generation.reasoning:
            message["reasoning_content"] = generation.reasoning
        if calls:
            message["tool_calls"] = calls
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(generation),
        }


async def run_completion(
    completion: Completion,
    generate: Callable[[str], Awaitable[Generation]],
    partial: Callable[[str], str | None],
    emit: Callable[[dict], None] | None,
) -> Generation:
    """Run ``generate`` and, when streaming, ``emit`` chunks as text arrives."""
    request_id = "openai-" + secrets.token_hex(8)
    task = asyncio.ensure_future(generate(request_id))
    try:
        if emit is not None:
            emit(completion._chunk({"role": "assistant", "content": ""}))
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                for chunk in completion.deltas(partial(request_id) or ""):
                    emit(chunk)
        return await task
    finally:
        if not task.done():
            task.cancel()


def family_from(identity: dict[str, Any]) -> ChatFamily:
    return family_for(str(identity.get("family") or identity.get("model_type") or ""))


def _clean_schema(schema: Any) -> Any:
    """Keep the JSON Schema keywords the native tool validator understands."""
    from ..agents.native_tools import _SCHEMA_KEYS

    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            value = {name: _clean_schema(item) for name, item in value.items()}
        elif key in {"items", "additionalProperties"} and isinstance(value, dict):
            value = _clean_schema(value)
        elif key in {"anyOf", "oneOf"} and isinstance(value, list):
            value = [_clean_schema(item) for item in value]
        cleaned[key] = value
    return cleaned


def native_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tool definitions as the workers' closed native schemas."""
    from ..agents.native_tools import validate_tool_specs

    specs = []
    for tool in tools:
        function = tool["function"]
        parameters = _clean_schema(function["parameters"])
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters["additionalProperties"] = False
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function["description"],
                    "parameters": parameters,
                },
            }
        )
    try:
        return validate_tool_specs(specs)
    except ValueError as exc:
        raise OpenAIError(400, f"unsupported tool definition: {exc}") from None


class OpenAIEndpoint:
    """``/v1`` on the app server: model ids are its workers' names, or their
    checkpoint folder names; the default is the driver."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def _workers(self) -> list[dict[str, Any]]:
        return [
            w
            for w in self.service.config.get("workers", [])
            if isinstance(w, dict) and w.get("name")
        ]

    def models(self) -> dict[str, Any]:
        created = int(time.time())
        ids = []
        for worker in self._workers():
            ids.append(worker["name"])
            folder = str(worker.get("model", "")).rstrip("/").rsplit("/", 1)[-1]
            if folder and folder not in ids:
                ids.append(folder)
        return {
            "object": "list",
            "data": [
                {"id": i, "object": "model", "created": created, "owned_by": "instinct"}
                for i in ids
            ],
        }

    def resolve(self, model: str | None) -> tuple[str | None, dict[str, Any]]:
        workers = self._workers()
        driver = self.service.config.get("runtime", {}).get("driver_group")
        for worker in workers:
            folder = str(worker.get("model", "")).rstrip("/").rsplit("/", 1)[-1]
            if model in {worker["name"], folder} or (
                not model and worker.get("group") == driver
            ):
                return worker["name"], self.service.identities.get(worker["name"], {})
        if not model and workers:
            return workers[0]["name"], self.service.identities.get(
                workers[0]["name"], {}
            )
        raise OpenAIError(404, f"no model named {model!r}", "not_found_error")

    async def chat(self, body: Any, emit: Callable[[dict], None] | None) -> dict | None:
        backend = self.service.backend
        if self.service.demo:
            raise OpenAIError(
                503, "the app is in demo mode, without models", "server_error"
            )
        if backend is None:
            raise OpenAIError(503, "the models aren't loaded yet", "server_error")
        requested = body.get("model") if isinstance(body, dict) else None
        worker, identity = self.resolve(requested)
        routed = {"worker": worker} if worker and hasattr(backend, "backends") else {}
        return await chat_completion(
            body,
            emit,
            backend=backend,
            identity=identity,
            model=worker or "",
            routed=routed,
        )


async def chat_completion(
    body: Any,
    emit: Callable[[dict], None] | None,
    *,
    backend: Any,
    identity: dict[str, Any],
    model: str,
    routed: dict[str, Any] | None = None,
) -> dict | None:
    """Serve one chat completion with ``backend`` (a worker or the pool)."""
    dialect = str(identity.get("native_call_dialect") or "json")
    request = parse_request(body, dialect=dialect, default_model=model)
    tools = native_tools(request.tools) if request.tools else None
    routed = routed or {}

    async def generate(request_id: str) -> Generation:
        if tools:
            return await backend.generate_action(
                request.messages,
                request.max_tokens,
                request_id,
                tools,
                options=request.options,
                **routed,
            )
        return await backend.generate(
            request.messages,
            request.max_tokens,
            request_id,
            options=request.options,
            **routed,
        )

    partial = getattr(backend, "partial", None)
    if isinstance(partial, dict):
        partial = partial.get
    completion = Completion(request, model=request.model, family=family_from(identity))
    generation = await run_completion(
        completion,
        generate,
        partial if callable(partial) else (lambda _: None),
        emit if request.stream else None,
    )
    if request.stream:
        for chunk in completion.final_chunks(generation):
            emit(chunk)
        return None
    return completion.response(generation)


def serve(
    handler: Any,
    loop: asyncio.AbstractEventLoop,
    body: Any,
    chat: Callable[..., Awaitable[dict | None]],
) -> None:
    """Answer an HTTP handler's completion request, as JSON or as SSE."""
    import concurrent.futures
    import queue

    def respond(status: int, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)

    def failure(error: BaseException) -> OpenAIError:
        return (
            error
            if isinstance(error, OpenAIError)
            else OpenAIError(500, str(error), "server_error")
        )

    if not (isinstance(body, dict) and body.get("stream")):
        try:
            result = asyncio.run_coroutine_threadsafe(chat(body, None), loop).result()
        except (Exception, concurrent.futures.CancelledError) as exc:  # noqa: BLE001 - reported to the client.
            error = failure(exc)
            return respond(error.status, error.body())
        return respond(200, result)
    chunks: queue.Queue = queue.Queue()
    future = asyncio.run_coroutine_threadsafe(chat(body, chunks.put), loop)
    future.add_done_callback(lambda _: chunks.put(None))
    started = False
    try:
        while (chunk := chunks.get()) is not None:
            if not started:
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.send_header("Cache-Control", "no-store")
                handler.end_headers()
                started = True
            handler.wfile.write(
                f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
            )
            handler.wfile.flush()
        error = future.exception()
        if error is not None:
            if not started:
                return respond(failure(error).status, failure(error).body())
            handler.wfile.write(
                f"data: {json.dumps(failure(error).body())}\n\n".encode()
            )
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        future.cancel()
