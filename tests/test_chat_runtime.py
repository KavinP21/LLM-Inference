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


def test_7b_profile_keeps_small_default_and_covers_native_32k_kv(chat):
    small, large = chat.CHAT_PROFILES
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
    profile = chat.CHAT_PROFILES[1]
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
        worker.load(chat.CHAT_PROFILES[0])
        assert events.get(timeout=2)[0] == "ready"
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
        worker.generate(messages, 64)
        assert events.get(timeout=2) == ("response", ("next", "answer [17]"))
        assert tokenizers[0].messages == messages
        submission = next(
            action for action in engines[0].actions if isinstance(action, tuple)
        )
        assert submission[1]["max_new_tokens"] == 5
        assert submission[1]["eos_token_ids"] == [151645]
        assert engines[0].actions[-1] == ("forget", 1)
        worker.load(chat.CHAT_PROFILES[1])
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
        worker.load(chat.CHAT_PROFILES[0])
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
        worker.load(chat.CHAT_PROFILES[1])
        assert started.wait(2)
        worker.close()
        release.set()
    finally:
        release.set()
        stop(worker)
    assert engines[0].closed
    assert events.empty()
