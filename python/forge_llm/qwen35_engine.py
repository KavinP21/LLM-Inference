"""Actor-owned Qwen3.5-4B text/image/video adapter for MLX-VLM 0.7.6.

This uses the pinned upstream model and processors, not Forge Metal kernels.
The state limit covers full attention buffers and recurrent/convolution state.
Weights, decoded media, embeddings and transient GPU arrays have separate
bounded inputs but are not included in that persistent-state limit.
"""

from __future__ import annotations

import hashlib
import math
import platform
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .qwen35_checkpoint import Qwen35Checkpoint, inspect_checkpoint
from .runtime import TokenEvent

PINNED_MLX_VLM = "0.7.6"
KV_STEP = 256
KV_BYTES_PER_TOKEN = 32768
RECURRENT_BYTES = 24 * 32 * 128 * 128 * 4
CONV_BYTES = 24 * 3 * 8192 * 2
FIXED_STATE_BYTES = RECURRENT_BYTES + CONV_BYTES


@dataclass(frozen=True)
class PreparedChat:
    input_ids: tuple[int, ...]
    inputs: Mapping[str, Any] = field(repr=False, compare=False)
    enable_thinking: bool = False
    media_fingerprints: tuple[tuple[str, str], ...] = ()
    owner: int = field(default=0, repr=False)

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class ChatResult:
    text: str
    reasoning: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    truncated: bool
    thinking_complete: bool
    tokens: tuple[int, ...] = field(default=(), repr=False)


@dataclass
class Qwen35Runtime:
    """Injectable processor/model boundary; constructing a fake never imports MLX."""

    model: Any
    processor: Any
    tokenizer: Any
    make_cache: Callable[[], list[Any]]
    prepare: Callable[[str, list[str], list[str], dict[str, int]], dict[str, Any]]
    embed: Callable[[dict[str, Any]], dict[str, Any]]
    forward: Callable[[Sequence[int], list[Any], dict[str, Any]], Any]
    greedy: Callable[[Any], list[int]]
    evaluate: Callable[[Any], None]
    synchronize: Callable[[], None]
    clear_cache: Callable[[], None]
    reset_positions: Callable[[], None]
    runner_version: str = PINNED_MLX_VLM
    mlx_version: str = "unknown"
    kv_type: type | None = None
    arrays_type: type | None = None
    mx: Any = None
    sample: Callable[[Any, Sequence[int], dict[str, Any]], list[int]] | None = None
    detach_array: Callable[[Any], Any] | None = None


def _load_runtime(checkpoint: Qwen35Checkpoint) -> Qwen35Runtime:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Qwen3.5 MLX-VLM requires Apple Silicon Metal")
    installed = version("mlx-vlm")
    if installed != PINNED_MLX_VLM:
        raise RuntimeError(
            f"Qwen3.5 adapter requires mlx-vlm=={PINNED_MLX_VLM}; installed {installed}"
        )
    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream
    from mlx_vlm.models.cache import ArraysCache, KVCache
    from mlx_vlm.sample_utils import make_logits_processors, make_sampler
    from mlx_vlm.utils import load_model, load_processor, prepare_inputs

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("Qwen3.5 requires an available Metal GPU")
    model = None
    try:
        model = load_model(
            checkpoint.path,
            lazy=False,
            strict=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        processor = load_processor(
            checkpoint.path,
            add_detokenizer=False,
            local_files_only=True,
            trust_remote_code=False,
        )
        if not checkpoint.unchanged():
            raise RuntimeError("Qwen3.5 checkpoint changed during loading")
        tokenizer = getattr(processor, "tokenizer", processor)
        if not getattr(tokenizer, "chat_template", None) and not getattr(
            processor, "chat_template", None
        ):
            raise ValueError("Qwen3.5 processor requires its trained chat template")
    except BaseException:
        model = None
        mx.synchronize(generation_stream)
        mx.clear_cache()
        raise

    def reset() -> None:
        model.language_model._position_ids = None
        model.language_model._rope_deltas = None

    def prepare(prompt, images, videos, limits):
        for component_name in ("image_processor", "video_processor"):
            component = getattr(processor, component_name, None)
            if component is not None:
                component.max_pixels = (
                    limits["image_pixels"]
                    if component_name == "image_processor"
                    else limits["video_pixels"]
                )
                if component_name == "video_processor":
                    component.max_frames = limits["video_frames"]
                    component.min_frames = 2
        return prepare_inputs(
            processor,
            images=images or None,
            videos=videos or None,
            prompts=prompt,
            padding=False,
            add_special_tokens=False,
            max_frames=limits["video_frames"],
            min_frames=2,
            fps=2.0,
            frame_factor=2,
            return_tensors="mlx",
        )

    def embed(inputs):
        selected = dict(inputs)
        ids = selected.pop("input_ids")
        mask = selected.pop("attention_mask", None)
        pixels = selected.pop("pixel_values", None)
        video_pixels = selected.get("pixel_values_videos")
        with mx.stream(generation_stream):
            if pixels is not None and video_pixels is not None:
                # Upstream's single-modality helper selects one grid. Encode
                # each tower input and scatter into its own token positions.
                embeddings = model.language_model.model.embed_tokens(ids)
                dtype = model.vision_tower.patch_embed.proj.weight.dtype
                for values, grid, token in (
                    (
                        pixels,
                        selected["image_grid_thw"],
                        model.config.image_token_index,
                    ),
                    (
                        video_pixels,
                        selected["video_grid_thw"],
                        model.config.video_token_index,
                    ),
                ):
                    features, _ = model.vision_tower(values.astype(dtype), grid)
                    embeddings, _ = model.merge_input_ids_with_image_features(
                        features, embeddings, ids, token, token
                    )
                positions, deltas = model.language_model.get_rope_index(
                    ids, selected["image_grid_thw"], selected["video_grid_thw"], mask
                )
                return {
                    "inputs_embeds": embeddings,
                    "position_ids": positions,
                    "rope_deltas": deltas,
                }
            features = model.get_input_embeddings(ids, pixels, mask=mask, **selected)
            return {
                key: value
                for key, value in features.to_dict().items()
                if value is not None
            }

    def forward(tokens, caches, options):
        with mx.stream(generation_stream):
            return model.language_model(
                mx.array([list(tokens)], dtype=mx.int32), cache=caches, **options
            )

    def greedy(output):
        if not mx.all(mx.isfinite(output.logits)).item():
            raise RuntimeError("Qwen3.5 returned nonfinite logits")
        return mx.argmax(output.logits, axis=-1).reshape(-1).tolist()

    samplers = {}

    def sample(output, history, options):
        key = (
            options["temperature"],
            options["top_p"],
            options["top_k"],
            options["presence_penalty"],
        )
        if key not in samplers:
            samplers[key] = (
                make_sampler(temp=key[0], top_p=key[1], top_k=key[2]),
                make_logits_processors(
                    presence_penalty=key[3], presence_context_size=262144
                ),
            )
        sampler, processors = samplers[key]
        logits = output.logits[:, -1, :]
        if not mx.all(mx.isfinite(logits)).item():
            raise RuntimeError("Qwen3.5 returned nonfinite logits")
        with mx.stream(generation_stream):
            for process in processors:
                logits = process(mx.array(list(history)), logits)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return sampler(logprobs).reshape(-1).tolist()

    return Qwen35Runtime(
        model,
        processor,
        tokenizer,
        model.language_model.make_cache,
        prepare,
        embed,
        forward,
        greedy,
        mx.eval,
        lambda: mx.synchronize(generation_stream),
        mx.clear_cache,
        reset,
        installed,
        version("mlx"),
        KVCache,
        ArraysCache,
        mx,
        sample,
        mx.array,
    )


def _float_bytes(value: Any, *, size: int) -> bool:
    dtype = value.dtype
    actual = getattr(dtype, "size", getattr(dtype, "itemsize", None))
    name = getattr(dtype, "name", str(dtype)).lower()
    return actual == size and (
        "float32" in name if size == 4 else "float16" in name or "bfloat16" in name
    )


def split_response(
    text: str,
    *,
    enable_thinking: bool,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    tokens=(),
) -> ChatResult:
    """Only the completed final channel becomes chat history or a tool action."""
    if enable_thinking:
        content = text[len("<think>") :] if text.startswith("<think>") else text
        if "</think>" not in content:
            return ChatResult(
                "",
                content,
                finish_reason,
                input_tokens,
                output_tokens,
                True,
                False,
                tuple(tokens),
            )
        reasoning, final = content.split("</think>", 1)
        complete = True
    elif text.startswith("<think>"):
        if "</think>" not in text:
            return ChatResult(
                "",
                text[len("<think>") :],
                finish_reason,
                input_tokens,
                output_tokens,
                True,
                False,
                tuple(tokens),
            )
        reasoning, final = text[len("<think>") :].split("</think>", 1)
        complete = True
    else:
        reasoning, final, complete = "", text, True
    return ChatResult(
        final.strip(),
        reasoning.strip(),
        finish_reason,
        input_tokens,
        output_tokens,
        finish_reason == "length",
        complete,
        tuple(tokens),
    )


class Qwen35Engine:
    """One request per actor; fresh multimodal positions and caches each time."""

    backend = "mlx"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_model_length: int = 32768,
        kv_cache_bytes: int = 2 << 30,
        prefill_chunk_size: int = 256,
        media_bytes_limit: int = 64 << 20,
        max_images: int = 4,
        max_videos: int = 1,
        max_video_frames: int = 32,
        image_pixels_limit: int = 1048576,
        video_pixels_limit: int = 2097152,
        prepared_bytes_limit: int = 256 << 20,
        max_video_seconds: float = 120.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        speculative: dict[str, Any] | None = None,
        _runtime_loader: Callable = _load_runtime,
        _checkpoint_loader: Callable = inspect_checkpoint,
    ) -> None:
        values = (
            max_model_length,
            kv_cache_bytes,
            prefill_chunk_size,
            media_bytes_limit,
            max_images,
            max_videos,
            max_video_frames,
            image_pixels_limit,
            video_pixels_limit,
            prepared_bytes_limit,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Qwen3.5 resource limits must be positive integers")
        if prefill_chunk_size > 2048 or (
            KV_STEP % prefill_chunk_size and prefill_chunk_size % KV_STEP
        ):
            raise ValueError(
                "prefill_chunk_size must divide 256 or be a multiple of 256 up to 2048"
            )
        if (
            max_images > 16
            or max_videos > 4
            or not 2 <= max_video_frames <= 128
            or max_video_frames % 2
        ):
            raise ValueError("media counts or video frame limit exceed bounded limits")
        if (
            any(
                type(v) not in {int, float} or not math.isfinite(v)
                for v in (max_video_seconds, temperature, top_p, presence_penalty)
            )
            or not 0 < max_video_seconds <= 120
            or not 0 <= temperature <= 2
            or not 0 < top_p <= 1
            or not -2 <= presence_penalty <= 2
            or type(top_k) is not int
            or not 0 <= top_k <= 248320
        ):
            raise ValueError("invalid Qwen3.5 sampling or video duration limits")
        if speculative is not None and (temperature != 0 or presence_penalty != 0):
            raise ValueError(
                "Qwen3.5 speculative decoding currently requires unmodified greedy sampling"
            )
        if speculative is not None and (
            not isinstance(speculative, dict)
            or set(speculative) - {"draft_tokens", "verification_mode", "adaptive"}
            or type(speculative.get("draft_tokens", 4)) is not int
            or not 1 <= speculative.get("draft_tokens", 4) <= 8
            or speculative.get("verification_mode", "block")
            not in {"block", "sequential"}
            or type(speculative.get("adaptive", True)) is not bool
        ):
            raise ValueError("invalid Qwen3.5 speculative configuration")
        self._owner = threading.get_ident()
        self.checkpoint = _checkpoint_loader(model_dir)
        text_config = self.checkpoint.config["text_config"]
        if max_model_length > text_config["max_position_embeddings"]:
            raise ValueError("Qwen3.5 context exceeds the checkpoint limit")
        if kv_cache_bytes < FIXED_STATE_BYTES + KV_STEP * KV_BYTES_PER_TOKEN:
            raise ValueError("state budget cannot fit one Qwen3.5 cache allocation")
        self.max_model_length, self.kv_cache_bytes = max_model_length, kv_cache_bytes
        self.prefill_chunk_size = prefill_chunk_size
        self.media_bytes_limit, self.max_images, self.max_videos = (
            media_bytes_limit,
            max_images,
            max_videos,
        )
        self.prepared_bytes_limit = prepared_bytes_limit
        self.max_video_seconds = max_video_seconds
        self._sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }
        self._media_limits = {
            "image_pixels": image_pixels_limit,
            "video_pixels": video_pixels_limit,
            "video_frames": max_video_frames,
        }
        self._runtime = _runtime_loader(self.checkpoint)
        self.model, self.processor, self.tokenizer = (
            self._runtime.model,
            self._runtime.processor,
            self._runtime.tokenizer,
        )
        self.config = SimpleNamespace(**text_config)
        self.checkpoint_identity = self.checkpoint.identity
        self._eos = frozenset({text_config["eos_token_id"], 248046})
        self._next_id, self._active, self._terminal = 1, None, None
        self._cache, self._output, self._history = [], [], []
        self._prepared = None
        self._pending = None
        self._maximum = self._reservation = self._extra_reserved = 0
        self._current_bytes = self._peak_bytes = self._peak_extra_bytes = 0
        self._prefill_calls = self._decode_calls = 0
        self._guard = lambda: None
        self._closed = False
        self._spec_options = dict(speculative) if speculative is not None else None
        self._speculator = None
        self._spec_stats = None
        self._last_response = None
        self.metadata = {
            "runner": "mlx_vlm",
            "runner_version": self._runtime.runner_version,
            "mlx_version": self._runtime.mlx_version,
            "checkpoint_format": "mlx_vlm_safetensors",
            "weight_storage_bytes": self.checkpoint.weight_bytes,
            "model_type": "qwen3_5",
            "kv_layers": 8,
            "recurrent_layers": 24,
            "kv_heads": 4,
            "kv_head_dim": 256,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "recurrent_state_bytes": RECURRENT_BYTES,
            "conv_state_bytes": CONV_BYTES,
            "kv_accounting": "persistent_full_KV_plus_recurrent_and_conv; excludes weights, media, embeddings and transient arrays",
        }

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Qwen3.5 engine operations belong to their actor thread")
        if self._closed:
            raise RuntimeError("Qwen3.5 engine is closed")

    def set_request_guard(self, guard: Callable[[], None]) -> None:
        self._check_owner()
        if not callable(guard):
            raise TypeError("request guard must be callable")
        self._guard = guard

    def _media(self, path: Any, kind: str) -> tuple[str, str, tuple[int, int, int]]:
        if not isinstance(path, (str, Path)) or "://" in str(path):
            raise ValueError("media must be explicit local file paths")
        selected = Path(path).expanduser().resolve(strict=True)
        before = selected.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= self.media_bytes_limit
        ):
            raise ValueError("media must be a bounded regular local file")
        digest = hashlib.sha256()
        with selected.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                self._guard()
                digest.update(chunk)
        if kind == "image":
            from PIL import Image

            with Image.open(selected) as image:
                if (
                    image.width <= 0
                    or image.height <= 0
                    or image.width * image.height > 16_777_216
                ):
                    raise ValueError("decoded image dimensions exceed the media limit")
        else:
            import cv2

            capture = cv2.VideoCapture(str(selected))
            try:
                if not capture.isOpened():
                    raise ValueError("video could not be opened")
                width, height = (
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                )
                frames, fps = (
                    capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    capture.get(cv2.CAP_PROP_FPS),
                )
                if (
                    not all(
                        math.isfinite(v) and v > 0 for v in (width, height, frames, fps)
                    )
                    or width * height > 2_097_152
                    or frames / fps > self.max_video_seconds
                    or frames > 10000
                ):
                    raise ValueError(
                        "video dimensions or duration exceed the media limit"
                    )
            finally:
                capture.release()
        after = selected.stat()
        signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        if signature != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("media changed during preparation")
        return str(selected), digest.hexdigest(), signature

    def prepare_chat(
        self,
        messages: Sequence[Any],
        *,
        tools=None,
        enable_thinking=False,
        images=(),
        videos=(),
    ) -> PreparedChat:
        self._check_owner()
        if self._active is not None or self._terminal is not None:
            raise RuntimeError(
                "finish and forget the previous request before preparing chat"
            )
        try:
            return self._prepare_chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                images=images,
                videos=videos,
            )
        except BaseException:
            try:
                self._runtime.synchronize()
            finally:
                self._prepared = None
                self._runtime.reset_positions()
                self._guard = lambda: None
                self._runtime.clear_cache()
            raise

    def _prepare_chat(
        self,
        messages: Sequence[Any],
        *,
        tools=None,
        enable_thinking: bool = False,
        images: Sequence[str] = (),
        videos: Sequence[str] = (),
    ) -> PreparedChat:
        self._check_owner()
        self._guard()
        if self._active is not None or self._terminal is not None:
            raise RuntimeError(
                "finish and forget the previous request before preparing chat"
            )
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be a boolean")
        turns, media, image_paths, video_paths = [], [], [], []
        total_media = 0
        for message in messages:
            role = message.get("role") if isinstance(message, Mapping) else message.role
            content = (
                message.get("content")
                if isinstance(message, Mapping)
                else message.content
            )
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError("invalid Qwen3.5 chat role")
            if isinstance(content, str):
                rendered_content = content
            elif isinstance(content, (list, tuple)) and role == "user":
                rendered_content = []
                for part in content:
                    if not isinstance(part, Mapping) or part.get("type") not in {
                        "text",
                        "image",
                        "video",
                    }:
                        raise ValueError("unsupported chat content part")
                    kind = part["type"]
                    if kind == "text":
                        if not isinstance(part.get("text"), str):
                            raise ValueError("text parts require text")
                        rendered_content.append({"type": "text", "text": part["text"]})
                    else:
                        path, digest, signature = self._media(part.get(kind), kind)
                        total_media += signature[1]
                        media.append((path, digest, signature))
                        (image_paths if kind == "image" else video_paths).append(path)
                        rendered_content.append({"type": kind, kind: path})
            else:
                raise ValueError("chat content must be text or user media parts")
            turns.append({"role": role, "content": rendered_content})
        if images or videos:
            if not turns or turns[-1]["role"] != "user":
                raise ValueError("explicit attachments require a final user turn")
            last = turns[-1]["content"]
            last = (
                [{"type": "text", "text": last}]
                if isinstance(last, str)
                else list(last)
            )
            for kind, paths in (("image", images), ("video", videos)):
                for given in paths:
                    path, digest, signature = self._media(given, kind)
                    total_media += signature[1]
                    media.append((path, digest, signature))
                    (image_paths if kind == "image" else video_paths).append(path)
                    last.append({"type": kind, kind: path})
            turns[-1]["content"] = last
        if (
            not turns
            or len(turns) > 256
            or sum(len(str(t["content"]).encode()) for t in turns) > 1 << 20
        ):
            raise ValueError("chat history is empty or exceeds its bounded size")
        if (
            len(image_paths) > self.max_images
            or len(video_paths) > self.max_videos
            or total_media > self.media_bytes_limit
        ):
            raise ValueError("chat attachments exceed the media budget")
        if tools is not None:
            from .agents.native_tools import validate_tool_specs

            tools = validate_tool_specs(tools)
        template = getattr(
            self.processor, "apply_chat_template", self.tokenizer.apply_chat_template
        )
        kwargs = {"tools": tools} if tools is not None else {}
        prompt = template(
            turns,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        if not isinstance(prompt, str) or len(prompt.encode()) > 2 << 20:
            raise ValueError("chat template produced an invalid or excessive prompt")
        self._guard()
        inputs = self._runtime.prepare(
            prompt, image_paths, video_paths, self._media_limits
        )
        self._guard()
        ids = inputs.get("input_ids")
        shape = getattr(ids, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != 1:
            raise ValueError("processor must return one tokenized conversation")
        tokens = tuple(ids.tolist()[0])
        if not tokens or any(
            type(t) is not int or not 0 <= t < self.config.vocab_size for t in tokens
        ):
            raise ValueError("processor produced invalid token IDs")
        if len(tokens) >= self.max_model_length:
            raise ValueError("expanded multimodal prompt exceeds the context limit")
        allowed = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }
        inputs = {k: v for k, v in inputs.items() if k in allowed and v is not None}
        if (
            sum(getattr(v, "nbytes", 0) for v in inputs.values())
            > self.prepared_bytes_limit
        ):
            raise ValueError("processed media exceeds the preparation memory limit")
        for path, _, signature in media:
            now = Path(path).stat()
            if signature != (now.st_ino, now.st_size, now.st_mtime_ns):
                raise ValueError("media changed while the processor was reading it")
        self._prepared = PreparedChat(
            tokens,
            inputs,
            enable_thinking,
            tuple((p, digest) for p, digest, _ in media),
            id(self),
        )
        return self._prepared

    def _observe_cache(self) -> None:
        if len(self._cache) != 32:
            raise RuntimeError("Qwen3.5 cache layer count changed")
        if self._runtime.detach_array is not None:
            # Contiguous tails can retain a whole prefill projection. Copy the
            # small history before charging its physical persistent payload.
            detached = []
            for i, cache in enumerate(self._cache):
                if (i + 1) % 4 and cache.cache[0] is not None:
                    selected = self._runtime.detach_array(cache.cache[0])
                    detached.append((cache, selected))
            self._runtime.evaluate([value for _, value in detached])
            for cache, value in detached:
                cache.cache[0] = value
        total = 0
        offsets = []
        for i, cache in enumerate(self._cache):
            if (i + 1) % 4 == 0:
                if (
                    self._runtime.kv_type is not None
                    and type(cache) is not self._runtime.kv_type
                ):
                    raise RuntimeError("Qwen3.5 full-attention cache type changed")
                if getattr(cache, "step", None) != KV_STEP:
                    raise RuntimeError("Qwen3.5 KV allocation step changed")
                offsets.append(cache.offset)
                for value in (cache.keys, cache.values):
                    if value is None:
                        continue
                    if (
                        not _float_bytes(value, size=2)
                        or len(value.shape) != 4
                        or (value.shape[0], value.shape[1], value.shape[3])
                        != (1, 4, 256)
                        or type(value.shape[2]) is not int
                        or value.shape[2] <= 0
                        or value.shape[2] < cache.offset
                    ):
                        raise RuntimeError("Qwen3.5 K/V dtype or geometry changed")
                    total += value.nbytes
            else:
                if (
                    self._runtime.arrays_type is not None
                    and type(cache) is not self._runtime.arrays_type
                ):
                    raise RuntimeError("Qwen3.5 recurrent cache type changed")
                if len(cache.cache) != 2:
                    raise RuntimeError("Qwen3.5 recurrent cache slot count changed")
                for value, shape, dtype_size in zip(
                    cache.cache, ((1, 3, 8192), (1, 32, 128, 128)), (2, 4)
                ):
                    if value is None:
                        continue
                    if tuple(value.shape) != shape or not _float_bytes(
                        value, size=dtype_size
                    ):
                        raise RuntimeError(
                            "Qwen3.5 recurrent state dtype or geometry changed"
                        )
                    total += value.nbytes
        if (
            any(type(offset) is not int or offset < 0 for offset in offsets)
            or len(set(offsets)) != 1
        ):
            raise RuntimeError("Qwen3.5 attention layer positions diverged")
        if offsets[0] > 0 and any(
            (cache.keys is None or cache.values is None)
            if (i + 1) % 4 == 0
            else any(value is None for value in cache.cache)
            for i, cache in enumerate(self._cache)
        ):
            raise RuntimeError("Qwen3.5 populated cache is missing layer state")
        self._current_bytes = total
        self._peak_bytes = max(self._peak_bytes, total)
        if (
            total > self._reservation
            or total + self._extra_reserved > self.kv_cache_bytes
        ):
            raise RuntimeError("Qwen3.5 persistent state exceeded its reservation")
        from .qwen35_state import retained_cache_bytes

        retained = retained_cache_bytes(self._cache)
        if (
            retained > self._reservation + self._extra_reserved
            or retained > self.kv_cache_bytes
        ):
            raise RuntimeError(
                "Qwen3.5 retained transaction state exceeded its reservation"
            )
        self._guard()

    def _cached_tokens(self) -> int:
        return self._cache[3].offset

    def _reserve_history(self, extra_bytes: int) -> Callable[[], None]:
        if (
            type(extra_bytes) is not int
            or extra_bytes < 0
            or self._reservation + self._extra_reserved + extra_bytes
            > self.kv_cache_bytes
        ):
            raise ValueError("speculative history exceeds the persistent-state budget")
        self._extra_reserved += extra_bytes
        self._peak_extra_bytes = max(self._peak_extra_bytes, self._extra_reserved)
        released = False

        def release():
            nonlocal released
            if not released:
                self._extra_reserved -= extra_bytes
                released = True

        return release

    def _forward(self, tokens: Sequence[int], *, speculative_verify: bool = False):
        self._guard()
        options = {"speculative_verify": speculative_verify}
        if not speculative_verify:
            options["logits_to_keep"] = 1
        output = self._runtime.forward(tokens, self._cache, options)
        self._decode_calls += 1
        return output

    def _sample(self, output) -> list[int]:
        if (
            self._sampling["temperature"] == 0
            and self._sampling["presence_penalty"] == 0
        ):
            return self._runtime.greedy(output)
        if self._runtime.sample is None:
            raise RuntimeError("runtime does not implement configured sampling")
        return self._runtime.sample(
            output, self._history + self._output, self._sampling
        )

    def submit(
        self,
        input_ids: Sequence[int] | PreparedChat,
        max_new_tokens: int,
        eos_token_ids: Sequence[int] = (),
    ) -> int:
        self._check_owner()
        self._guard()
        if self._active is not None or self._terminal is not None:
            raise RuntimeError("finish and forget the previous Qwen3.5 request")
        prepared = input_ids if isinstance(input_ids, PreparedChat) else None
        if prepared is not None and (
            prepared.owner != id(self) or prepared is not self._prepared
        ):
            raise ValueError("prepared prompt does not belong to this engine")
        tokens = list(prepared.input_ids if prepared else input_ids)
        if (
            prepared is not None
            and tuple(prepared.inputs["input_ids"].tolist()[0]) != prepared.input_ids
        ):
            raise ValueError("prepared token IDs changed after processing")
        if (
            not tokens
            or any(
                type(t) is not int or not 0 <= t < self.config.vocab_size
                for t in tokens
            )
            or type(max_new_tokens) is not int
            or max_new_tokens <= 0
        ):
            raise ValueError("invalid Qwen3.5 token IDs or generation limit")
        if len(tokens) + max_new_tokens > self.max_model_length:
            raise ValueError("prompt plus output exceeds Qwen3.5 context")
        reservation = (
            math.ceil((len(tokens) + max_new_tokens) / KV_STEP)
            * KV_STEP
            * KV_BYTES_PER_TOKEN
            + FIXED_STATE_BYTES
        )
        if reservation > self.kv_cache_bytes:
            raise ValueError(
                "prompt plus output exceeds Qwen3.5 persistent-state budget"
            )
        eos = list(eos_token_ids) if eos_token_ids else list(self._eos)
        if any(type(t) is not int or not 0 <= t < self.config.vocab_size for t in eos):
            raise ValueError("invalid Qwen3.5 EOS token IDs")
        self._eos = frozenset(eos)
        self._reservation, self._maximum = reservation, max_new_tokens
        self._peak_bytes = self._current_bytes = self._extra_reserved = (
            self._peak_extra_bytes
        ) = 0
        self._prefill_calls = self._decode_calls = 0
        self._last_response = self._spec_stats = None
        self._active, self._next_id = self._next_id, self._next_id + 1
        self._history, self._output = tokens, []
        try:
            self._runtime.reset_positions()
            self._cache = self._runtime.make_cache()
            self._observe_cache()
            inputs = dict(prepared.inputs) if prepared else {"input_ids": None}
            if prepared:
                features = self._runtime.embed(inputs)
                self._guard()
                embeddings = features.pop("inputs_embeds")
                if (
                    tuple(embeddings.shape) != (1, len(tokens), 2560)
                    or embeddings.nbytes > self.prepared_bytes_limit
                ):
                    raise ValueError(
                        "model produced excessive or invalid prompt embeddings"
                    )
            else:
                embeddings, features = None, {}
            # Match the released generator's chunk policy: cache the prefix
            # without logits, then run one final prompt token. The model does
            # not implement logits_to_keep, so evaluating a whole final chunk
            # would materialize a large and unnecessary vocabulary matrix.
            chunks = [
                (start, min(start + self.prefill_chunk_size, len(tokens) - 1))
                for start in range(0, len(tokens) - 1, self.prefill_chunk_size)
            ]
            chunks.append((len(tokens) - 1, len(tokens)))
            for start, end in chunks:
                self._guard()
                options = dict(features)
                options["skip_logits"] = end < len(tokens)
                if embeddings is not None:
                    options["inputs_embeds"] = embeddings[:, start:end]
                output = self._runtime.forward(tokens[start:end], self._cache, options)
                self._runtime.evaluate([c.state for c in self._cache])
                self._prefill_calls += 1
                self._observe_cache()
            predictions = self._sample(output)
            if not predictions:
                raise RuntimeError("Qwen3.5 returned no prefill prediction")
            self._pending = predictions[-1]
            if self._spec_options is not None:
                from .qwen35_speculative import Qwen35Speculator, UpstreamQwen35Target
                from .qwen35_state import Qwen35StateGeometry

                geometry = Qwen35StateGeometry.from_text_config(
                    self.checkpoint.config["text_config"]
                )
                target = UpstreamQwen35Target(
                    self._runtime.mx,
                    caches=self._cache,
                    forward=self._forward,
                    observe_cache=self._observe_cache,
                    reserve_history=self._reserve_history,
                    cached_tokens=self._cached_tokens,
                    geometry=geometry,
                    vocab_size=self.config.vocab_size,
                )
                self._speculator = Qwen35Speculator(target, **self._spec_options)
                self._speculator.start(tokens, max_new_tokens)
            return self._active
        except BaseException:
            self._release()
            self._terminal = None
            raise

    def step(self) -> list[TokenEvent]:
        self._check_owner()
        if self._active is None:
            raise RuntimeError("Qwen3.5 engine has no active request")
        request_id = self._active
        try:
            self._guard()
            if not self._output:
                additions = [self._pending]
            elif self._speculator is not None:
                additions = self._speculator.next_block(
                    self._history + self._output,
                    self._maximum - len(self._output),
                    self._eos,
                    cancelled=lambda: False,
                )
            else:
                output = self._forward([self._output[-1]])
                additions = self._sample(output)
                self._observe_cache()
            if (
                not additions
                or len(additions) > self._maximum - len(self._output)
                or any(
                    type(t) is not int or not 0 <= t < self.config.vocab_size
                    for t in additions
                )
            ):
                raise RuntimeError("Qwen3.5 decoder violated its output contract")
            events = []
            for token in additions:
                self._output.append(token)
                finished = token in self._eos or len(self._output) == self._maximum
                events.append(TokenEvent(request_id, token, finished))
                if finished:
                    self._release()
                    break
            return events
        except BaseException:
            if self._active is not None:
                self._release()
            raise

    def _release(self) -> None:
        request_id = self._active
        try:
            if self._speculator is not None:
                self._spec_stats = self._speculator.stats.to_dict()
                self._speculator.finish()
        finally:
            try:
                self._runtime.synchronize()
            finally:
                for i, cache in enumerate(self._cache):
                    if (i + 1) % 4 == 0:
                        cache.keys = cache.values = None
                        cache.offset = 0
                    else:
                        cache.cache = [None, None]
                        cache.left_padding = cache.lengths = None
                        if hasattr(cache, "_speculation"):
                            cache._speculation = None
                self._cache.clear()
                self._speculator = self._prepared = self._pending = None
                self._active, self._terminal = None, request_id
                self._current_bytes = self._reservation = self._extra_reserved = 0
                self._runtime.reset_positions()
                self._guard = lambda: None
                self._runtime.clear_cache()

    def cancel(self, request_id: int) -> None:
        self._check_owner()
        if request_id == self._active:
            self._release()
        elif request_id != self._terminal:
            raise KeyError("unknown Qwen3.5 request")

    def forget(self, request_id: int) -> None:
        self._check_owner()
        if request_id == self._active:
            raise RuntimeError("cannot forget an active Qwen3.5 request")
        if request_id != self._terminal:
            raise KeyError("unknown Qwen3.5 request")
        self._terminal = None
        self._history, self._output = [], []

    def generate(
        self, input_ids, max_new_tokens: int, eos_token_ids=(), *, cancelled=None
    ) -> list[int]:
        self._check_owner()

        def guard():
            if cancelled is not None and cancelled():
                raise InterruptedError("Qwen3.5 generation cancelled")

        self.set_request_guard(guard)
        request_id = self.submit(input_ids, max_new_tokens, eos_token_ids)
        output = []
        try:
            finished = False
            while not finished:
                for event in self.step():
                    output.append(event.token)
                    finished = event.finished
            return output
        finally:
            if self._active is not None:
                self.cancel(request_id)
            self.forget(request_id)

    def response_metadata(
        self,
        tokens: Sequence[int],
        *,
        enable_thinking: bool = False,
        input_tokens: int = 0,
    ) -> ChatResult:
        selected = list(tokens)
        reason = "stop" if selected and selected[-1] in self._eos else "length"
        decoded = selected[:-1] if reason == "stop" else selected
        text = self.tokenizer.decode(decoded, skip_special_tokens=False)
        result = split_response(
            text,
            enable_thinking=enable_thinking,
            finish_reason=reason,
            input_tokens=input_tokens,
            output_tokens=len(selected),
            tokens=selected,
        )
        self._last_response = result
        return result

    def generate_chat(
        self,
        messages,
        max_new_tokens: int,
        *,
        enable_thinking: bool = False,
        cancelled=None,
        tools=None,
        images=(),
        videos=(),
    ) -> ChatResult:
        self._check_owner()

        def guard():
            if cancelled is not None and cancelled():
                raise InterruptedError("Qwen3.5 generation cancelled")

        self.set_request_guard(guard)
        try:
            prepared = self.prepare_chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                images=images,
                videos=videos,
            )
            output = self.generate(prepared, max_new_tokens, cancelled=cancelled)
            return self.response_metadata(
                output,
                enable_thinking=enable_thinking,
                input_tokens=prepared.input_tokens,
            )
        finally:
            self._guard = lambda: None
            self._prepared = None

    def stats(self) -> dict[str, Any]:
        return {
            "active_requests": int(self._active is not None),
            "kv_reserved_bytes": self._reservation,
            "kv_persistent_bytes": self._current_bytes,
            "last_kv_peak_bytes": self._peak_bytes,
            "speculative_reserved_bytes": self._extra_reserved,
            "last_speculative_history_peak_bytes": self._peak_extra_bytes,
            "prefill_calls": self._prefill_calls,
            "decode_calls": self._decode_calls,
            "speculative": self._spec_stats,
        }

    def build_info(self) -> dict[str, Any]:
        return {
            **self.metadata,
            **self.checkpoint_identity,
            "max_model_length": self.max_model_length,
            "state_budget_bytes": self.kv_cache_bytes,
            "supports_images": True,
            "supports_video": True,
            "supports_audio": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._check_owner()
        try:
            if self._active is not None:
                self._release()
            self._runtime.synchronize()
        finally:
            self._runtime.reset_positions()
            self.model = self.processor = self.tokenizer = None
            self._runtime.model = self._runtime.processor = self._runtime.tokenizer = (
                None
            )
            self._runtime.forward = lambda *args: None
            self._runtime.embed = lambda *args: {}
            self._runtime.prepare = lambda *args: {}
            self._runtime.make_cache = list
            self._runtime.reset_positions = lambda: None
            self._prepared = None
            self._history, self._output = [], []
            self._terminal = None
            self._guard = lambda: None
            self._closed = True
            self._runtime.clear_cache()
