"""The OpenAI-compatible /v1 API and its API keys."""

import asyncio
import base64
import http.client
import json
import threading

import pytest
from instinct.agents.protocol import Generation
from instinct.chat_formats import GEMMA4, QWEN35
from instinct.harness.openai_api import (
    Completion,
    OpenAIError,
    parse_request,
    tool_calls,
)
from instinct.harness.server import AppServer
from instinct.harness.service import AppService

PNG = base64.b64encode(bytes.fromhex("89504e470d0a1a0a")).decode()


def test_requests_become_worker_messages():
    request = parse_request(
        {
            "model": "gemma-4",
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{PNG}"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {
                                "name": "look",
                                "arguments": '{"path": "x.py", "n": 2}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "ok"},
            ],
            "max_completion_tokens": 64,
            "reasoning_effort": "low",
            "temperature": 0.5,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        dialect="gemma4",
        default_model="gemma-4",
    )
    roles = [m.role for m in request.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert request.messages[1].media[0]["mime"] == "image/png"
    assert (
        request.messages[2].content
        == '<|tool_call>call:look{path:<|"|>x.py<|"|>,n:2}<tool_call|>'
    )
    assert request.options == {"effort": "low", "temperature": 0.5}
    assert request.max_tokens == 64 and request.stream and request.include_usage
    for bad in (
        {"messages": []},
        {"messages": [{"role": "robot", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x"}}
                    ],
                }
            ]
        },
        {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"},
    ):
        with pytest.raises(OpenAIError):
            parse_request(bad, dialect="json", default_model="m")


@pytest.mark.parametrize(
    "dialect, text",
    [
        (
            "gemma4",
            'Let me check.<|tool_call>call:look{path:<|"|>a.py<|"|>,n:2}<tool_call|>',
        ),
        (
            "qwen3_coder_xml",
            "Let me check.\n<tool_call>\n<function=look>\n<parameter=path>\na.py\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>",
        ),
        (
            "json",
            'Let me check.\n<tool_call>\n{"name": "look", "arguments": {"path": "a.py", "n": 2}}\n</tool_call>',
        ),
    ],
)
def test_tool_calls_come_back_in_openai_shape(dialect, text):
    rest, calls = tool_calls(text, dialect)
    assert rest == "Let me check."
    assert calls[0]["function"]["name"] == "look"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py", "n": 2}


def test_streaming_separates_reasoning_from_the_answer():
    request = parse_request(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"},
        dialect="gemma4",
        default_model="g",
    )
    completion = Completion(request, model="g", family=GEMMA4)
    deltas = completion.deltas("<|channel>thought\nWeighing it")
    deltas += completion.deltas("<|channel>thought\nWeighing it<channel|>Hello")
    sent = [c["choices"][0]["delta"] for c in deltas]
    assert {"reasoning_content": "Weighing it"} in sent and {"content": "Hello"} in sent
    final = completion.final_chunks(
        Generation("Hello there", 5, 3, "stop", reasoning="Weighing it")
    )
    assert final[0]["choices"][0]["delta"] == {"content": " there"}
    assert final[-1]["choices"][0]["finish_reason"] == "stop"

    # Qwen's template opens the thought, so the stream starts inside it.
    qwen = Completion(
        parse_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            },
            dialect="qwen3_coder_xml",
            default_model="q",
        ),
        model="q",
        family=QWEN35,
    )
    assert qwen.deltas("thinking")[0]["choices"][0]["delta"] == {
        "reasoning_content": "thinking"
    }


class Workers:
    """Stands in for the worker pool: replies in two steps so streams have deltas."""

    def __init__(self):
        self.live = {}
        self.calls = []

    def partial(self, request_id):
        return self.live.get(request_id)

    async def generate(self, messages, max_tokens, request_id, *, options=None):
        self.calls.append((messages, options))
        self.live[request_id] = "Hello"
        await asyncio.sleep(0.12)
        self.live[request_id] = "Hello world"
        await asyncio.sleep(0.12)
        return Generation(
            "Hello world", 12, 2, "stop", "gemma", "gemma4", "", {"reused_tokens": 4}
        )

    async def generate_action(
        self, messages, max_tokens, request_id, tools, *, options=None
    ):
        self.calls.append((messages, tools))
        return Generation(
            '<|tool_call>call:look{path:<|"|>a.py<|"|>}<tool_call|>',
            12,
            8,
            "stop",
            "gemma",
            "gemma4",
        )

    async def close(self):
        pass


@pytest.fixture
def server(tmp_path):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    workers = Workers()

    async def build():
        service = AppService(
            tmp_path,
            tmp_path / ".instinct",
            config={
                "workers": [
                    {
                        "name": "gemma-4",
                        "group": "driver",
                        "model": "models/gemma-4-26b-a4b-mlx",
                    }
                ],
                "runtime": {"driver_group": "driver"},
            },
            backend=workers,
            keys_file=tmp_path / "keys.json",
        )
        service.identities = {
            "gemma-4": {"native_call_dialect": "gemma4", "family": "gemma4"}
        }
        return service

    service = asyncio.run_coroutine_threadsafe(build(), loop).result()
    app = AppServer(("127.0.0.1", 0), service, loop)
    serving = threading.Thread(target=app.serve_forever, daemon=True)
    serving.start()
    key = service.api_keys.create("phone")["key"]
    yield app, service, workers, key
    app.shutdown()
    asyncio.run_coroutine_threadsafe(service.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)


def call(app, method, path, body=None, key=None):
    connection = http.client.HTTPConnection("127.0.0.1", app.server_port, timeout=10)
    headers = {
        "Content-Type": "application/json",
        "Host": f"127.0.0.1:{app.server_port}",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    connection.request(
        method, path, json.dumps(body) if body is not None else None, headers
    )
    response = connection.getresponse()
    return response.status, response.read().decode()


def test_models_and_completions_need_a_key(server):
    app, service, workers, key = server
    assert call(app, "GET", "/v1/models")[0] == 401
    assert call(app, "GET", "/v1/models", key="sk-instinct-wrong")[0] == 401
    status, text = call(app, "GET", "/v1/models", key=key)
    assert status == 200
    assert [m["id"] for m in json.loads(text)["data"]] == [
        "gemma-4",
        "gemma-4-26b-a4b-mlx",
    ]
    status, text = call(
        app,
        "POST",
        "/v1/chat/completions",
        {
            "model": "gemma-4-26b-a4b-mlx",
            "messages": [{"role": "user", "content": "hi"}],
        },
        key,
    )
    reply = json.loads(text)
    assert status == 200 and reply["object"] == "chat.completion"
    assert reply["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello world",
    }
    assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert (
        call(
            app,
            "POST",
            "/v1/chat/completions",
            {"model": "nope", "messages": [{"role": "user", "content": "x"}]},
            key,
        )[0]
        == 404
    )
    listed = asyncio.run_coroutine_threadsafe(
        service.dispatch("GET", "/api/keys", {}), app.loop
    ).result()
    assert listed["keys"][0]["name"] == "phone" and "key" not in listed["keys"][0]
    asyncio.run_coroutine_threadsafe(
        service.dispatch("POST", f"/api/keys/{listed['keys'][0]['id']}/delete", {}),
        app.loop,
    ).result()
    assert call(app, "GET", "/v1/models", key=key)[0] == 401


def test_streaming_and_tool_calls(server):
    app, _, workers, key = server
    status, text = call(
        app,
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        key,
    )
    events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    assert status == 200 and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]
    )
    assert content == "Hello world"
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    tool = {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "examples": [{"path": "a.py"}],
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    }
    status, text = call(
        app,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "read a.py"}], "tools": [tool]},
        key,
    )
    message = json.loads(text)["choices"][0]
    assert message["finish_reason"] == "tool_calls"
    assert json.loads(message["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "a.py"
    }
    spec = workers.calls[-1][1][0]["function"]["parameters"]
    assert spec["additionalProperties"] is False and "examples" not in spec
