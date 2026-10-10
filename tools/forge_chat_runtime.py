"""Native chat profiles and a single owner for model loading and generation."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChatProfile:
    key: str
    label: str
    artifact: str
    tokenizer: str
    kv_cache_bytes: int
    max_model_length: int = 32768
    decode_mode: str = "batched"


CHAT_PROFILES = (
    ChatProfile(
        "qwen05",
        "Qwen2.5-0.5B-Instruct",
        "qwen2.5-0.5b.engine",
        "Qwen/Qwen2.5-0.5B-Instruct",
        512 << 20,
    ),
    ChatProfile(
        "qwen7b",
        "Qwen2.5-7B-Instruct · reconstructed FP16",
        "qwen2.5-7b-mlx-reconstructed.engine",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        2 << 30,
        decode_mode="rowwise",
    ),
)


def load_native_profile(root: Path, profile: ChatProfile):
    """Use an existing Forge artifact and an offline cached tokenizer only."""
    from forge_llm import create_engine
    from transformers import AutoTokenizer

    model_path = root / "models" / profile.artifact
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            profile.tokenizer, local_files_only=True
        )
    except OSError as exc:
        raise RuntimeError(
            f"Tokenizer is not cached locally: {profile.tokenizer}. "
            "Prepare the matching tokenizer before loading this model."
        ) from exc
    engine = create_engine(
        model_path,
        backend="mlx",
        max_num_sequences=1,
        max_model_length=profile.max_model_length,
        kv_cache_bytes=profile.kv_cache_bytes,
        prefill_chunk_size=512,
        decode_mode=profile.decode_mode,
    )
    return engine, tokenizer


class ChatWorker:
    """Own engine operations on one thread, including reload and shutdown."""

    def __init__(self, root: Path, events: queue.Queue, *, loader=load_native_profile):
        self.root, self.events, self.loader = root, events, loader
        self.commands: queue.Queue = queue.Queue()
        self.stopping = threading.Event()
        self.engine = self.tokenizer = None
        self.thread = threading.Thread(target=self._serve, name="forge-chat-model")
        self.thread.start()

    def load(self, profile: ChatProfile) -> None:
        if not self.stopping.is_set():
            self.commands.put(("load", profile))

    def generate(self, messages: list[dict[str, str]], max_tokens: int) -> None:
        if not self.stopping.is_set():
            self.commands.put(("generate", ([dict(m) for m in messages], max_tokens)))

    def close(self) -> None:
        self.stopping.set()
        self.commands.put(("close", None))

    def _close_engine(self) -> None:
        engine, self.engine = self.engine, None
        self.tokenizer = None
        if engine is not None:
            engine.close()

    def _serve(self) -> None:
        try:
            while not self.stopping.is_set():
                action, payload = self.commands.get()
                if self.stopping.is_set():
                    break
                try:
                    if action == "load":
                        self._close_engine()
                        self.engine, self.tokenizer = self.loader(self.root, payload)
                        if not self.stopping.is_set():
                            self.events.put(
                                (
                                    "ready",
                                    (
                                        payload,
                                        int(self.engine.max_model_length),
                                        str(
                                            self.engine.build_info().get(
                                                "device", "Metal"
                                            )
                                        ),
                                    ),
                                )
                            )
                    elif action == "generate":
                        messages, max_tokens = payload
                        response = self._generate(messages, max_tokens)
                        if not self.stopping.is_set():
                            self.events.put(
                                ("response", (messages[-1]["content"], response))
                            )
                except Exception as exc:
                    if action == "load":
                        self._close_engine()
                    if not self.stopping.is_set():
                        self.events.put(("error", str(exc)))
        finally:
            self._close_engine()

    def _generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        if self.engine is None or self.tokenizer is None:
            raise RuntimeError("Load a model before generating.")
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(rendered, return_tensors=None).input_ids
        available_tokens = self.engine.max_model_length - len(tokens)
        if available_tokens <= 0:
            raise ValueError(
                f"This conversation is too long for the {self.engine.max_model_length:,}-token "
                "context. Press Clear chat to start a new conversation."
            )
        if self.stopping.is_set():
            return ""
        request_id = self.engine.submit(
            tokens,
            max_new_tokens=min(max_tokens, available_tokens),
            eos_token_ids=[self.tokenizer.eos_token_id],
        )
        output = []
        finished = False
        try:
            while not self.stopping.is_set():
                for event in self.engine.step():
                    if event.request_id == request_id:
                        output.append(event.token)
                        finished = event.finished
                if finished:
                    return self.tokenizer.decode(
                        output, skip_special_tokens=True
                    ).strip()
            return ""
        finally:
            if not finished:
                self.engine.cancel(request_id)
            self.engine.forget(request_id)
