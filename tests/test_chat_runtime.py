"""Portable chat profile and engine ownership checks; no Tk or GPU execution."""

import importlib
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("forge_chat_runtime")


class FakeTokenizer:
    eos_token_id = 151645

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered"

    def __call__(self, _rendered, **_kwargs):
        return SimpleNamespace(input_ids=[1, 2, 3])

    def decode(self, tokens, **_kwargs):
        return f"answer {tokens}"


class FakeEngine:
    max_model_length = 8

    def __init__(self):
        self.owner = threading.get_ident()
        self.closed = False
        self.actions = []
        self.started = self.release = None

    def record(self, action):
        assert threading.get_ident() == self.owner
        assert not self.closed
        self.actions.append(action)

    def build_info(self):
        self.record("info")
        return {"device": "Fake Metal"}

    def submit(self, _tokens, **kwargs):
        self.record(("submit", kwargs))
        return 1

    def step(self):
        self.record("step")
        if self.started is not None:
            self.started.set()
            assert self.release.wait(2)
            return []
        return [SimpleNamespace(request_id=1, token=17, finished=True)]

    def cancel(self, request_id):
        self.record(("cancel", request_id))

    def forget(self, request_id):
        self.record(("forget", request_id))

    def close(self):
        self.record("close")
        self.closed = True


def stop(worker):
    worker.close()
    worker.thread.join(2)
    assert not worker.thread.is_alive()


def test_qwen35_default_retains_native_profiles_and_7b_32k_capacity(chat):
    daily, small, large = chat.CHAT_PROFILES
    assert daily.key == "qwen35"
    assert daily.supports_media and daily.supports_thinking
    assert daily.runner == "qwen35"
    assert small.key == "qwen05"
    assert small.kv_cache_bytes == 512 << 20
    assert small.decode_mode == "batched"
    assert "reconstructed FP16" in large.label
    assert large.max_model_length == 32768
    assert large.decode_mode == "rowwise"
    assert 28 * 2 * 4 * 128 * 2 * 32768 <= large.kv_cache_bytes


def test_profile_uses_matching_offline_tokenizer_and_native_engine(
    chat, tmp_path, monkeypatch
):
    profile = chat.CHAT_PROFILES[2]
    models = tmp_path / "models"
    models.mkdir()
    (models / profile.artifact).write_bytes(b"fixture")
    calls = []

    def tokenizer_load(source, **kwargs):
        calls.append(("tokenizer", source, kwargs))
        return "tokenizer"

    def engine_load(path, **kwargs):
        calls.append(("engine", path, kwargs))
        return "engine"

    monkeypatch.setitem(
        sys.modules, "forge_llm", SimpleNamespace(create_engine=engine_load)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load)),
    )
    assert chat.load_native_profile(tmp_path, profile) == ("engine", "tokenizer")
    assert calls[0] == ("tokenizer", profile.tokenizer, {"local_files_only": True})
    assert calls[1][1] == models / profile.artifact
    assert calls[1][2]["kv_cache_bytes"] == 2 << 30
    assert calls[1][2]["max_model_length"] == 32768
    assert calls[1][2]["decode_mode"] == "rowwise"


def test_reload_closes_previous_engine_before_loading_and_preserves_dialogue(
    chat, tmp_path
):
    engines, tokenizers = [], []

    def loader(_root, _profile):
        assert all(engine.closed for engine in engines)
        engine, tokenizer = FakeEngine(), FakeTokenizer()
        engines.append(engine)
        tokenizers.append(tokenizer)
        return engine, tokenizer

    events = queue.Queue()
    worker = chat.ChatWorker(tmp_path, events, loader=loader)
    try:
        worker.load(chat.CHAT_PROFILES[1])
        assert events.get(timeout=2)[0] == "ready"
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
        worker.generate(messages, 64)
        assert events.get(timeout=2) == (
            "response",
            (messages[-1], chat.ChatReply("answer [17]")),
        )
        assert tokenizers[0].messages == messages
        submission = next(
            action for action in engines[0].actions if isinstance(action, tuple)
        )
        assert submission[1]["max_new_tokens"] == 5
        assert submission[1]["eos_token_ids"] == [151645]
        assert engines[0].actions[-1] == ("forget", 1)
        worker.load(chat.CHAT_PROFILES[2])
        assert events.get(timeout=2)[0] == "ready"
        assert engines[0].closed
    finally:
        stop(worker)
    assert engines[1].closed


def test_window_shutdown_waits_for_owned_step_then_cancels_and_closes(chat, tmp_path):
    started, release = threading.Event(), threading.Event()
    engines = []

    def loader(_root, _profile):
        engine = FakeEngine()
        engine.started, engine.release = started, release
        engines.append(engine)
        return engine, FakeTokenizer()

    events = queue.Queue()
    worker = chat.ChatWorker(tmp_path, events, loader=loader)
    try:
        worker.load(chat.CHAT_PROFILES[1])
        assert events.get(timeout=2)[0] == "ready"
        worker.generate([{"role": "user", "content": "prompt"}], 2)
        assert started.wait(2)
        worker.close()
        assert not engines[0].closed
        release.set()
    finally:
        release.set()
        stop(worker)
    assert engines[0].actions[-3:] == [("cancel", 1), ("forget", 1), "close"]
    assert events.empty()


def test_close_during_load_reclaims_new_engine_without_ready_event(chat, tmp_path):
    started, release = threading.Event(), threading.Event()
    engines = []

    def loader(_root, _profile):
        started.set()
        assert release.wait(2)
        engine = FakeEngine()
        engines.append(engine)
        return engine, FakeTokenizer()

    events = queue.Queue()
    worker = chat.ChatWorker(tmp_path, events, loader=loader)
    try:
        worker.load(chat.CHAT_PROFILES[2])
        assert started.wait(2)
        worker.close()
        release.set()
    finally:
        release.set()
        stop(worker)
    assert engines[0].closed
    assert events.empty()


def test_daily_profile_loads_full_local_vlm_without_native_artifact_conversion(
    chat, tmp_path, monkeypatch
):
    daily = chat.CHAT_PROFILES[0]
    model = tmp_path / "models" / daily.artifact
    model.mkdir(parents=True)
    calls = []

    def create(path, **options):
        calls.append((path, options))
        return SimpleNamespace(tokenizer="processor tokenizer")

    monkeypatch.setitem(
        sys.modules, "forge_llm.qwen35_engine", SimpleNamespace(Qwen35Engine=create)
    )
    engine, tokenizer = chat.load_native_profile(tmp_path, daily)
    assert tokenizer == engine.tokenizer
    assert calls[0][0] == model
    assert calls[0][1] == {
        "max_model_length": 32768,
        "kv_cache_bytes": 2 << 30,
        "prefill_chunk_size": 256,
    }


def test_final_history_keeps_visual_input_and_excludes_reasoning_or_interrupted_answers(
    chat,
):
    history = []
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown?"},
            {"type": "image", "image": "/local/fixture.png"},
        ],
    }
    for reply in (
        chat.ChatReply("", "unfinished", "length", True, False),
        chat.ChatReply("partial", finish_reason="cancelled"),
    ):
        assert not chat.record_final_reply(history, user, reply)
        assert history == []
    assert chat.record_final_reply(
        history, user, chat.ChatReply("A red circle.", "private history-free reasoning")
    )
    assert history == [user, {"role": "assistant", "content": "A red circle."}]
    user["content"][0]["text"] = "mutated"
    assert history[0]["content"][0]["text"] == "What is shown?"


def test_media_staging_is_stable_private_bounded_and_released(chat, tmp_path):
    source = tmp_path / "picture.png"
    source.write_bytes(b"original pixels")
    store = chat.ChatMediaStore(max_file_bytes=32, max_total_bytes=32)
    try:
        part = store.stage(source, "image")
        staged = Path(part["image"])
        source.write_bytes(b"changed pixels")
        assert staged.read_bytes() == b"original pixels"
        assert staged.stat().st_mode & 0o077 == 0
        assert store.names[str(staged)] == "picture.png"
        source.write_bytes(b"x" * 33)
        with pytest.raises(ValueError):
            store.stage(source, "image")
        store.discard(part)
        assert not staged.exists()
        assert store.total_bytes == 0
        with pytest.raises(ValueError):
            store.discard({"type": "image", "image": str(source)})
        assert source.exists()
        store.clear()
        assert not staged.exists()
        assert store.total_bytes == 0
    finally:
        store.close()
    assert not Path(store.directory.name).exists()


def test_vlm_owner_receives_full_visual_history_thinking_and_stop(chat, tmp_path):
    started, release = threading.Event(), threading.Event()
    engines = []

    class VlmEngine(FakeEngine):
        def generate_chat(
            self, messages, max_new_tokens, *, enable_thinking, cancelled
        ):
            self.record("generate_chat")
            self.seen = (messages, max_new_tokens, enable_thinking)
            started.set()
            assert release.wait(2)
            if cancelled():
                raise InterruptedError("Qwen3.5 generation cancelled")
            return chat.ChatReply("visual answer", "reasoning")

    def loader(_root, _profile):
        engine = VlmEngine()
        engines.append(engine)
        return engine, FakeTokenizer()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/local/staged.png"},
                {"type": "text", "text": "Describe this."},
            ],
        },
        {"role": "assistant", "content": "A circle."},
        {"role": "user", "content": "What color was it?"},
    ]
    events = queue.Queue()
    worker = chat.ChatWorker(tmp_path, events, loader=loader)
    try:
        worker.load(chat.CHAT_PROFILES[0])
        assert events.get(timeout=2)[0] == "ready"
        worker.generate(messages, 2048, enable_thinking=True)
        assert started.wait(2)
        worker.stop()
        assert not engines[0].closed
        release.set()
        kind, (_, reply) = events.get(timeout=2)
        assert kind == "response" and reply.finish_reason == "cancelled"
        assert engines[0].seen == (messages, 2048, True)
        worker.generate(messages, 32)
        assert events.get(timeout=2)[1][1].text == "visual answer"
        assert not engines[0].closed
    finally:
        release.set()
        stop(worker)
    assert engines[0].closed


def test_native_profile_does_not_silently_drop_visual_parts(chat, tmp_path):
    events = queue.Queue()
    worker = chat.ChatWorker(
        tmp_path, events, loader=lambda *_: (FakeEngine(), FakeTokenizer())
    )
    try:
        worker.load(chat.CHAT_PROFILES[1])
        assert events.get(timeout=2)[0] == "ready"
        worker.generate(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "/local/input.png"}],
                }
            ],
            8,
        )
        assert events.get(timeout=2)[0] == "error"
    finally:
        stop(worker)


@pytest.fixture
def headless_ui(chat, monkeypatch):
    """Exercise actual widget-state methods with inert widgets, without a Tk app."""
    import importlib.util

    fake_tk = SimpleNamespace(
        Tk=object,
        filedialog=SimpleNamespace(),
        messagebox=SimpleNamespace(),
        ttk=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    path = Path(__file__).resolve().parents[1] / "tools" / "forge_chat.py"
    spec = importlib.util.spec_from_file_location("forge_chat_headless", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Widget:
        def __init__(self):
            self.options = {}

        def configure(self, **kwargs):
            self.options.update(kwargs)

        def delete(self, *_args):
            pass

    ui = SimpleNamespace(
        ready=True,
        busy=True,
        loading=False,
        closing=False,
        profile=chat.CHAT_PROFILES[0],
        conversation=[],
        attachments=[],
        last_reasoning="",
        events=queue.Queue(),
        context_limit=32768,
        device_name="Fake Metal",
        after=lambda *_: None,
    )
    for name in (
        "model_picker",
        "load_button",
        "clear_button",
        "max_tokens",
        "generate_button",
        "stop_button",
        "image_button",
        "video_button",
        "remove_media_button",
        "thinking_button",
        "prompt",
    ):
        setattr(ui, name, Widget())
    ui._set_controls = lambda: module.ForgeChat._set_controls(ui)
    ui._drain_events = lambda: module.ForgeChat._drain_events(ui)
    ui._set_status = lambda text, _color: setattr(ui, "status", text)
    ui._ready_status = lambda: "Ready"
    ui._update_attachments = lambda: None
    ui._render_conversation = lambda: None
    ui._write_output = lambda text: setattr(ui, "display", text)
    return module, ui


def test_controls_block_model_and_media_changes_during_generation(headless_ui):
    module, ui = headless_ui
    module.ForgeChat._set_controls(ui)
    for name in (
        "model_picker",
        "image_button",
        "video_button",
        "thinking_button",
        "generate_button",
    ):
        assert getattr(ui, name).options["state"] == "disabled"
    assert ui.stop_button.options["state"] == "normal"
    ui.busy = False
    module.ForgeChat._set_controls(ui)
    assert ui.model_picker.options["state"] == "readonly"
    assert ui.image_button.options["state"] == "normal"
    assert ui.stop_button.options["state"] == "disabled"


def test_truncated_thinking_has_visible_status_without_false_history_entry(
    chat, headless_ui
):
    module, ui = headless_ui
    user = {"role": "user", "content": "Solve this."}
    ui.events.put(
        (
            "response",
            (user, chat.ChatReply("", "unfinished reasoning", "length", True, False)),
        )
    )
    module.ForgeChat._drain_events(ui)
    assert ui.conversation == []
    assert "no final answer" in ui.status
    assert ui.last_reasoning == "unfinished reasoning"
    assert not ui.busy
    ui.busy = True
    ui.events.put(("response", (user, chat.ChatReply("The answer.", "reasoning"))))
    module.ForgeChat._drain_events(ui)
    assert ui.conversation == [user, {"role": "assistant", "content": "The answer."}]


def test_stopped_generation_does_not_append_partial_reply(chat, headless_ui):
    module, ui = headless_ui
    ui.events.put(
        (
            "response",
            (
                {"role": "user", "content": "Question."},
                chat.ChatReply("partial", finish_reason="cancelled"),
            ),
        )
    )
    module.ForgeChat._drain_events(ui)
    assert ui.conversation == []
    assert ui.status.startswith("Stopped")
    assert not ui.busy
