"""Native chat profiles and a single owner for model loading and generation."""

from __future__ import annotations

import copy
import os
import queue
import tempfile
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
    runner: str = "forge"
    supports_media: bool = False
    supports_thinking: bool = False


@dataclass(frozen=True)
class ChatReply:
    text: str
    reasoning: str = ""
    finish_reason: str = "eos"
    truncated: bool = False
    thinking_complete: bool = True


def record_final_reply(conversation: list[dict], user: dict, reply: ChatReply) -> bool:
    """Only final answers enter model history; media references remain on user turns."""
    if (
        reply.finish_reason == "cancelled"
        or not reply.thinking_complete
        or not reply.text
    ):
        return False
    conversation.extend(
        [copy.deepcopy(user), {"role": "assistant", "content": reply.text}]
    )
    return True


class ChatMediaStore:
    """Private stable copies keep visual inputs unchanged across conversation turns."""

    def __init__(self, *, max_file_bytes=64 << 20, max_total_bytes=128 << 20):
        self.directory = tempfile.TemporaryDirectory(prefix="forge-chat-media-")
        self.max_file_bytes, self.max_total_bytes = max_file_bytes, max_total_bytes
        self.total_bytes = 0
        self.serial = 0
        self.names: dict[str, str] = {}
        self.sizes: dict[str, int] = {}

    def stage(self, source: Path, kind: str) -> dict:
        suffixes = {
            "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
            "video": {".mp4", ".mov", ".m4v", ".webm"},
        }
        source = Path(source).resolve(strict=True)
        if (
            kind not in suffixes
            or source.suffix.lower() not in suffixes[kind]
            or not source.is_file()
        ):
            raise ValueError("Choose a supported local image or video file.")
        size = source.stat().st_size
        if (
            not 0 < size <= self.max_file_bytes
            or self.total_bytes + size > self.max_total_bytes
        ):
            raise ValueError(
                "Attachment exceeds the local media budget. Clear chat to release earlier attachments."
            )
        self.serial += 1
        destination = (
            Path(self.directory.name) / f"{self.serial}{source.suffix.lower()}"
        )
        try:
            with source.open("rb") as original, destination.open("xb") as staged:
                before = os.fstat(original.fileno())
                copied = 0
                while chunk := original.read(
                    min(1 << 20, self.max_file_bytes + 1 - copied)
                ):
                    copied += len(chunk)
                    if (
                        copied > self.max_file_bytes
                        or self.total_bytes + copied > self.max_total_bytes
                    ):
                        raise ValueError(
                            "Attachment grew beyond the local media budget."
                        )
                    staged.write(chunk)
                after = os.fstat(original.fileno())
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ) or copied != size:
                    raise ValueError(
                        "Attachment changed while copying; choose it again."
                    )
            destination.chmod(0o600)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self.total_bytes += size
        self.names[str(destination)] = source.name
        self.sizes[str(destination)] = size
        return {"type": kind, kind: str(destination)}

    def discard(self, part: dict) -> None:
        """Release an unused pending copy; callers retain copies referenced by history."""
        path = part.get(part.get("type"))
        if path not in self.sizes:
            raise ValueError("Attachment is not owned by this chat.")
        Path(path).unlink(missing_ok=True)
        self.total_bytes -= self.sizes.pop(path)
        self.names.pop(path)

    def clear(self) -> None:
        for file in Path(self.directory.name).iterdir():
            file.unlink()
        self.total_bytes = 0
        self.names.clear()
        self.sizes.clear()

    def close(self) -> None:
        self.directory.cleanup()


CHAT_PROFILES = (
    ChatProfile(
        "qwen35",
        "Qwen3.5-4B · MLX 4-bit · text, images and video",
        "qwen3.5-4b-mlx",
        "",
        2 << 30,
        runner="qwen35",
        supports_media=True,
        supports_thinking=True,
    ),
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
    model_path = root / "models" / profile.artifact
    if profile.runner == "qwen35":
        from forge_llm.qwen35_engine import Qwen35Engine

        if not model_path.is_dir():
            raise FileNotFoundError(f"Local Qwen3.5 checkpoint not found: {model_path}")
        engine = Qwen35Engine(
            model_path,
            max_model_length=profile.max_model_length,
            kv_cache_bytes=profile.kv_cache_bytes,
            prefill_chunk_size=256,
        )
        return engine, engine.tokenizer
    from forge_llm import create_engine
    from transformers import AutoTokenizer

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
        self.cancel_requested = threading.Event()
        self.media_store = ChatMediaStore()
        self.engine = self.tokenizer = None
        self.thread = threading.Thread(target=self._serve, name="forge-chat-model")
        self.thread.start()

    def load(self, profile: ChatProfile) -> None:
        if not self.stopping.is_set():
            self.commands.put(("load", profile))

    def generate(
        self, messages: list[dict], max_tokens: int, *, enable_thinking=False
    ) -> None:
        if not self.stopping.is_set():
            self.cancel_requested.clear()
            self.commands.put(
                ("generate", (copy.deepcopy(messages), max_tokens, enable_thinking))
            )

    def stop(self) -> None:
        self.cancel_requested.set()

    def _cancelled(self) -> bool:
        return self.stopping.is_set() or self.cancel_requested.is_set()

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
                        messages, max_tokens, thinking = payload
                        response = self._generate(messages, max_tokens, thinking)
                        if not self.stopping.is_set():
                            self.events.put(("response", (messages[-1], response)))
                except Exception as exc:
                    if action == "load":
                        self._close_engine()
                    if not self.stopping.is_set():
                        self.events.put(("error", str(exc)))
        finally:
            try:
                self._close_engine()
            finally:
                self.media_store.close()

    def _generate(
        self, messages: list[dict], max_tokens: int, enable_thinking: bool
    ) -> ChatReply:
        if self.engine is None or self.tokenizer is None:
            raise RuntimeError("Load a model before generating.")
        if hasattr(self.engine, "generate_chat"):
            try:
                result = self.engine.generate_chat(
                    messages,
                    max_new_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    cancelled=self._cancelled,
                )
            except InterruptedError:
                if not self._cancelled():
                    raise
                return ChatReply("", finish_reason="cancelled")
            return ChatReply(
                result.text,
                result.reasoning,
                result.finish_reason,
                result.truncated,
                result.thinking_complete,
            )
        if enable_thinking or any(
            not isinstance(message.get("content"), str) for message in messages
        ):
            raise ValueError(
                "This native chat profile supports text without a thinking-mode switch."
            )
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
        if self._cancelled():
            return ChatReply("", finish_reason="cancelled")
        request_id = self.engine.submit(
            tokens,
            max_new_tokens=min(max_tokens, available_tokens),
            eos_token_ids=[self.tokenizer.eos_token_id],
        )
        output = []
        finished = False
        try:
            while not self._cancelled():
                for event in self.engine.step():
                    if event.request_id == request_id:
                        output.append(event.token)
                        finished = event.finished
                if finished:
                    return ChatReply(
                        self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    )
            return ChatReply("", finish_reason="cancelled")
        finally:
            if not finished:
                self.engine.cancel(request_id)
            self.engine.forget(request_id)
