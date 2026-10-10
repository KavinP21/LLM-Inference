"""Portable media transport, persistence and Qwen3.5 actor integration checks."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from forge_llm.agents.backends import LocalForgeBackend, WorkerConfig, validate_request
from forge_llm.agents.media import materialize_media, media_from_file, validate_media
from forge_llm.agents.protocol import ChatMessage
from forge_llm.agents.runtime import AgentRuntime
from forge_llm.agents.store import AgentStore
from forge_llm.runtime import TokenEvent


@pytest.fixture
def attachment(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"operator-owned media fixture")
    return media_from_file(path, mime="image/png")


@pytest.mark.parametrize("defect", ["sha256", "data", "mime", "type", "extra"])
def test_media_rejects_corruption_and_unknown_fields(attachment, defect):
    changed = dict(attachment)
    if defect == "sha256":
        changed[defect] = "0" * 64
    elif defect == "data":
        changed[defect] = "invalid*base64"
    elif defect == "mime":
        changed[defect] = "application/x-python"
    elif defect == "type":
        changed[defect] = "audio"
    else:
        changed[defect] = "ignored"
    with pytest.raises(ValueError):
        validate_media([changed])


def test_media_is_copied_and_materialized_only_to_private_request_files(
    attachment, tmp_path
):
    accepted = validate_request(
        [ChatMessage("user", "inspect", [attachment])], 8, "job"
    )
    attachment["mime"] = "changed after acceptance"
    assert accepted[0].media[0]["mime"] == "image/png"
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    parts = materialize_media(accepted[0].media, directory)
    path = Path(parts[0]["image"])
    assert path.read_bytes() == b"operator-owned media fixture"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="user role"):
        validate_request(
            [ChatMessage("assistant", "untrusted", accepted[0].media)], 8, "job"
        )


def test_media_journal_deduplicates_bytes_and_restores_generation_snapshot(
    attachment, tmp_path
):
    from forge_llm.agents.runtime import RuntimeConfig

    path = tmp_path / "journal.sqlite"
    with AgentStore(path) as store:
        store.create_run("run", "task", vars(RuntimeConfig()), "system")
        for _ in range(3):
            store.add_message("run", "root", "user", "image", media=[attachment])
        messages = store.messages("run", "root")
        store.save_generation_input("run", "request", messages, len(messages))
        assert (
            store.connection.execute("SELECT count(*) FROM media_blobs").fetchone()[0]
            == 1
        )
        encoded = store.connection.execute(
            "SELECT messages FROM generation_inputs"
        ).fetchone()[0]
        assert attachment["data"] not in encoded
    with AgentStore(path) as store:
        restored, count = store.generation_input("run", "request")
        assert count == 5 and restored[-1].media == (attachment,)


def test_invalid_run_media_fails_before_creating_a_run():
    runtime = AgentRuntime(SimpleNamespace())
    with pytest.raises(ValueError):
        asyncio.run(runtime.run("task", media=[{}]))
    assert (
        runtime.store.connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    )


def test_spawn_inherits_operator_media_without_duplicating_blob(attachment):
    from forge_llm.agents.runtime import RuntimeConfig

    with AgentStore() as store:
        store.create_run("run", "inspect image", vars(RuntimeConfig()), "system")
        store.add_message("run", "root", "user", "operator image", media=[attachment])
        child = store.spawn(
            "run", "root", "spawn-1", "Review the image", "reviewer", [], "system"
        )
        assert store.messages("run", child)[-1].media == (attachment,)
        assert (
            store.connection.execute("SELECT count(*) FROM media_blobs").fetchone()[0]
            == 1
        )


def test_media_reserves_advertised_context_before_inference(attachment):
    from forge_llm.agents.protocol import Generation
    from forge_llm.agents.runtime import RuntimeConfig

    class Native:
        supports_native_tools = True
        max_model_length = 32768
        calls = 0

        async def generate_action(self, messages, maximum, request_id, tool_specs):
            self.calls += 1
            assert messages[1].media == (attachment,)
            reserved = runtime.store.run(request_id.split(".")[0])["reserved_tokens"]
            assert reserved == self.max_model_length + maximum
            return Generation(
                '<tool_call>{"name":"finish","arguments":{"result":"observed"}}</tool_call>',
                2000,
                30,
            )

    backend = Native()
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_total_tokens=40000))
    result = asyncio.run(runtime.run("Read the image", media=[attachment]))
    assert result.status == "completed" and backend.calls == 1
    assert result.usage.input_tokens == 2000 and result.usage.reserved_tokens == 0
    backend = Native()
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_total_tokens=32000))
    result = asyncio.run(runtime.run("Read the image", media=[attachment]))
    assert result.status == "failed" and backend.calls == 0


class Tokenizer:
    chat_template = "fixture"
    eos_token_id = 248044


class Engine:
    max_model_length = 32768
    backend = "mlx"
    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=248320, eos_token_id=248044, model_type="qwen3_5_text"
        )
    )
    metadata = {"runner": "qwen35", "runner_version": "0.7.6"}
    checkpoint_identity = {}

    def __init__(self):
        self.prepared = None
        self.paths = []
        self.forgotten = False

    def set_request_guard(self, guard):
        self.guard = guard

    def prepare_chat(self, turns, **options):
        self.guard()
        self.prepared = (turns, options)
        for turn in turns:
            if isinstance(turn["content"], list):
                for part in turn["content"]:
                    if part["type"] == "image":
                        self.paths.append(part["image"])
                        assert Path(part["image"]).is_file()
        return SimpleNamespace(input_tokens=17)

    def submit(self, prepared, maximum, eos):
        assert prepared.input_tokens == 17 and eos == [248044]
        return 1

    def step(self):
        return [TokenEvent(1, 248044, True)]

    def response_metadata(self, tokens, **options):
        return SimpleNamespace(text="Observed the operator's image")

    def forget(self, request):
        self.forgotten = True

    def stats(self):
        return {"active_requests": 0}

    def close(self):
        pass


def test_qwen35_actor_uses_full_processor_and_removes_request_media(
    attachment, tmp_path
):
    engine = Engine()
    backend = LocalForgeBackend(
        WorkerConfig(
            str(tmp_path),
            str(tmp_path),
            runner="qwen35",
            backend="mlx",
            max_model_length=32768,
            kv_cache_bytes=2 << 30,
        ),
        engine_factory=lambda: engine,
        tokenizer_factory=Tokenizer,
    )

    async def run():
        try:
            result = await backend.generate(
                [ChatMessage("user", "inspect", (attachment,))], 16, "image-request"
            )
            assert result.text == "Observed the operator's image"
            assert result.input_tokens == 17 and result.output_tokens == 1
        finally:
            await backend.close()

    asyncio.run(run())
    assert engine.forgotten
    assert engine.prepared[1]["enable_thinking"] is False
    assert all(not Path(path).exists() for path in engine.paths)
