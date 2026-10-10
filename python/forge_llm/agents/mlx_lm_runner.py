"""Optional pinned MLX-LM replica adapter; not Forge kernels or quantization.

The execution surface mirrors Forge's actor-owned submit/step/cancel/forget
contract. Local Qwen3-MoE and dense Qwen2/Qwen2.5 7B affine 4-bit checkpoints
require validated FP16/BF16 KV dimensions. The budget covers full KV buffers,
not weights, temporary forward-pass arrays, or total process memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import stat
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..runtime import TokenEvent
from .backends import BackendError, ContextLengthError, RequestValidationError

PINNED_MLX_LM = "0.28.3"
KV_STEP = 256


@dataclass(frozen=True)
class KVGeometry:
    layers: int
    heads: int
    head_dim: int
    hidden_size: int

    @property
    def bytes_per_token(self) -> int:
        return self.layers * 2 * self.heads * self.head_dim * 2

    @property
    def scale_shape(self) -> list[int]:
        return [self.heads * self.head_dim, self.hidden_size // 64]


def _geometry(config: dict[str, Any]) -> KVGeometry:
    family = config.get("model_type")
    if family == "qwen3_moe":
        expected = {
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2048,
        }
    elif family == "qwen2":
        expected = {
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
        }
    else:
        raise ValueError(
            "mlx_lm runner supports validated Qwen3-MoE or dense Qwen2/Qwen2.5 7B checkpoints"
        )
    if any(
        type(config.get(key)) is not int or config[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("checkpoint does not match a validated Qwen geometry")
    if family == "qwen2":
        # The pinned Qwen2 adapter computes this from hidden size / Q heads.
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        if "head_dim" in config and (
            type(config["head_dim"]) is not int or config["head_dim"] != head_dim
        ):
            raise ValueError(
                "dense checkpoint head_dim disagrees with the attention geometry"
            )
    else:
        head_dim = config["head_dim"]
    enabled = config.get("use_sliding_window", False)
    if type(enabled) is not bool or enabled:
        raise ValueError("enabled or malformed sliding-window attention is unsupported")
    window = config.get("sliding_window")
    if window is not None:
        if family != "qwen2" or config.get("use_sliding_window") is not False:
            raise ValueError(
                "nonnull sliding_window requires explicitly disabled dense attention metadata"
            )
        if type(window) is not int or window <= 0:
            raise ValueError("invalid disabled sliding_window metadata")
    return KVGeometry(
        config["num_hidden_layers"],
        config["num_key_value_heads"],
        head_dim,
        config["hidden_size"],
    )


@dataclass(frozen=True)
class LocalCheckpoint:
    path: Path
    config: dict[str, Any]
    identity: dict[str, str]
    weight_bytes: int
    signatures: tuple[tuple[Path, int, int, int], ...]
    geometry: KVGeometry

    def unchanged(self) -> bool:
        try:
            expected = {
                p.name for p, *_ in self.signatures if p.suffix == ".safetensors"
            }
            if {p.name for p in self.path.glob("model*.safetensors")} != expected:
                return False
            return all(
                (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                == (inode, size, modified)
                for p, inode, size, modified in self.signatures
            )
        except OSError:
            return False


def inspect_checkpoint(path: str | Path) -> LocalCheckpoint:
    """CPU-only local validation and full-byte fingerprints of loaded shards."""
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("mlx_lm requires an existing local checkpoint directory")
    config_path = directory / "config.json"
    if not stat.S_ISREG(config_path.stat().st_mode):
        raise ValueError("checkpoint configuration must be a regular local file")
    with config_path.open("rb") as stream:
        raw_config = stream.read((1 << 20) + 1)
    if len(raw_config) > 1 << 20:
        raise ValueError("checkpoint configuration exceeds 1 MiB")
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration must be an object")
    geometry = _geometry(config)
    for field in ("vocab_size", "max_position_embeddings"):
        if type(config.get(field)) is not int or not 0 < config[field] <= 1_000_000:
            raise ValueError(f"invalid checkpoint {field}")
    if (
        type(config.get("eos_token_id")) is not int
        or not 0 <= config["eos_token_id"] < config["vocab_size"]
    ):
        raise ValueError("invalid checkpoint EOS token")
    quantization = config.get("quantization", {})
    if (
        not isinstance(quantization, dict)
        or type(quantization.get("bits")) is not int
        or quantization.get("bits") != 4
        or type(quantization.get("group_size")) is not int
        or quantization.get("group_size") != 64
        or quantization.get("mode", "affine") != "affine"
    ):
        raise ValueError(
            "mlx_lm runner requires the local affine 4-bit checkpoint format"
        )
    files = sorted(directory.glob("model*.safetensors"))
    if not files:
        raise ValueError("checkpoint contains no local model safetensors shards")
    if len(files) > 128:
        raise ValueError("checkpoint contains too many weight shards")
    tensors: dict[str, dict] = {}
    manifest = []
    signatures = []
    for selected in [config_path, *files]:
        before = selected.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint files must be regular local files")
        digest = hashlib.sha256()
        with selected.open("rb") as stream:
            if selected.suffix == ".safetensors":
                raw_size = stream.read(8)
                if len(raw_size) != 8:
                    raise ValueError("truncated safetensors header")
                size = struct.unpack("<Q", raw_size)[0]
                if not 1 <= size <= 16 << 20 or size + 8 > before.st_size:
                    raise ValueError("invalid safetensors header size")
                header = json.loads(stream.read(size))
                if not isinstance(header, dict):
                    raise ValueError("invalid safetensors tensor metadata")
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    if (
                        name in tensors
                        or not isinstance(info, dict)
                        or info.get("dtype") not in {"BF16", "F16", "U32"}
                    ):
                        raise ValueError(
                            "unsupported/duplicate checkpoint tensor or floating dtype"
                        )
                    tensors[name] = info
                stream.seek(0)
            while chunk := stream.read(4 << 20):
                digest.update(chunk)
        after = selected.stat()
        signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        if signature != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("checkpoint changed while hashing")
        signatures.append((selected, *signature))
        manifest.append(
            {
                "name": selected.name,
                "sha256": digest.hexdigest(),
                "bytes": before.st_size,
            }
        )
    # Establish the actual two-byte K/V projection output path, rather than
    # assuming a torch_dtype label describes the safetensors contents.
    for layer in range(geometry.layers):
        for projection in ("k_proj", "v_proj"):
            name = f"model.layers.{layer}.self_attn.{projection}.scales"
            info = tensors.get(name, {})
            shape = info.get("shape")
            if (
                info.get("dtype") not in {"BF16", "F16"}
                or not isinstance(shape, list)
                or any(type(value) is not int for value in shape)
                or shape != geometry.scale_shape
            ):
                raise ValueError(
                    "checkpoint K/V projection dimensions or dtype are unsupported"
                )
    identity = {
        "model_data_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    }
    return LocalCheckpoint(
        directory,
        config,
        identity,
        sum(p.stat().st_size for p in files),
        tuple(signatures),
        geometry,
    )


@dataclass
class _Runtime:
    model: Any
    make_cache: Callable[[], list[Any]]
    generate: Callable[..., Iterator[tuple[int, Any]]]
    synchronize: Callable[[], None]
    clear_cache: Callable[[], None]
    runner_version: str
    mlx_version: str


def _load_runtime(checkpoint: LocalCheckpoint) -> _Runtime:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("mlx_lm runner requires an Apple Silicon Metal GPU")
    try:
        installed = version("mlx-lm")
        mlx_version = version("mlx")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "install the optional mlx-lm==0.28.3 runner without upgrading the configured MLX runtime"
        ) from exc
    if installed != PINNED_MLX_LM:
        raise RuntimeError(
            f"mlx_lm runner requires version {PINNED_MLX_LM}; installed {installed}"
        )
    import mlx.core as mx

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("mlx_lm runner requires an available Metal device")
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from mlx_lm.utils import load_model

    # load_model takes a Path directly and never calls the repository downloader.
    try:
        model, loaded_config = load_model(checkpoint.path, lazy=False, strict=True)
    except BaseException:
        mx.synchronize(generation_stream)
        mx.clear_cache()
        raise
    if loaded_config != checkpoint.config or not checkpoint.unchanged():
        del model
        mx.clear_cache()
        raise RuntimeError("checkpoint changed during model loading")

    def make_cache() -> list[Any]:
        caches = make_prompt_cache(model)
        if len(caches) != checkpoint.geometry.layers or any(
            type(cache) is not KVCache or cache.step != KV_STEP for cache in caches
        ):
            raise BackendError("unexpected MLX-LM cache implementation")
        return caches

    def generate(tokens, **options):
        return generate_step(mx.array(tokens), model, **options)

    return _Runtime(
        model,
        make_cache,
        generate,
        lambda: mx.synchronize(generation_stream),
        mx.clear_cache,
        installed,
        mlx_version,
    )


class MlxLmRunner:
    """One actor-owned full-context greedy runner with no retained conversations."""

    backend = "mlx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_model_length: int,
        kv_cache_bytes: int,
        prefill_step_size: int = 256,
        _runtime_loader: Callable[[LocalCheckpoint], _Runtime] = _load_runtime,
    ) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (max_model_length, kv_cache_bytes, prefill_step_size)
        ):
            raise ValueError("runner resource limits must be positive integers")
        if prefill_step_size > 2048:
            raise ValueError("prefill_step_size cannot exceed 2048")
        if KV_STEP % prefill_step_size != 0 and prefill_step_size % KV_STEP != 0:
            raise ValueError(
                "prefill_step_size must divide 256 or be a multiple of 256 to preserve KV allocation bounds"
            )
        self.checkpoint = inspect_checkpoint(model_path)
        if max_model_length > self.checkpoint.config["max_position_embeddings"]:
            raise ValueError("runtime context exceeds checkpoint context")
        self.max_model_length, self.kv_cache_bytes = max_model_length, kv_cache_bytes
        self.prefill_step_size = prefill_step_size
        self.bytes_per_token = self.checkpoint.geometry.bytes_per_token
        self._runtime = _runtime_loader(self.checkpoint)
        self.model = SimpleNamespace(config=SimpleNamespace(**self.checkpoint.config))
        self.checkpoint_identity = self.checkpoint.identity
        self.metadata = {
            "runner": "mlx_lm",
            "runner_version": self._runtime.runner_version,
            "mlx_version": self._runtime.mlx_version,
            "checkpoint_format": "mlx_lm_safetensors",
            "weight_storage_bytes": self.checkpoint.weight_bytes,
            "kv_bytes_per_token": self.bytes_per_token,
            "model_type": self.checkpoint.config["model_type"],
            "kv_layers": self.checkpoint.geometry.layers,
            "kv_heads": self.checkpoint.geometry.heads,
            "kv_head_dim": self.checkpoint.geometry.head_dim,
            "kv_allocation_step": KV_STEP,
            "kv_accounting": "persistent_full_KV_only; excludes transient arrays and weights",
        }
        self._active: int | None = None
        self._terminal: int | None = None
        self._next_id = 1
        self._iterator = None
        self._cache: list[Any] = []
        self._guard: Callable[[], None] = lambda: None
        self._max_tokens = self._produced = self._reservation = 0
        self._eos: frozenset[int] = frozenset()
        self._current_bytes = self._peak_bytes = 0
        self._closed = False

    def set_request_guard(self, guard: Callable[[], None]) -> None:
        self._guard = guard

    def _observe_cache(self, *_: Any) -> None:
        geometry = self.checkpoint.geometry
        if len(self._cache) != geometry.layers:
            raise BackendError("MLX-LM K/V layer count changed")
        total = 0
        for cache in self._cache:
            for value in (cache.keys, cache.values):
                if value is None:
                    continue
                if (
                    value.dtype.size != 2
                    or len(value.shape) != 4
                    or any(
                        type(dimension) is not int or dimension <= 0
                        for dimension in value.shape
                    )
                    or (value.shape[0], value.shape[1], value.shape[3])
                    != (1, geometry.heads, geometry.head_dim)
                ):
                    raise BackendError("MLX-LM K/V dtype or dimensions changed")
                total += value.nbytes
        self._current_bytes = total
        self._peak_bytes = max(self._peak_bytes, total)
        if total > self._reservation or total > self.kv_cache_bytes:
            raise BackendError(
                "MLX-LM persistent KV allocation exceeded its reservation"
            )
        self._guard()

    def submit(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> int:
        if self._closed:
            raise BackendError("runner is closed")
        if self._active is not None or self._terminal is not None:
            raise BackendError(
                "finish and forget the previous runner request before admission"
            )
        tokens = list(input_ids)
        if (
            not tokens
            or type(max_new_tokens) is not int
            or max_new_tokens <= 0
            or any(
                type(t) is not int or not 0 <= t < self.checkpoint.config["vocab_size"]
                for t in tokens
            )
        ):
            raise RequestValidationError("invalid runner token IDs or token budget")
        eos = list(eos_token_ids)
        if any(
            type(t) is not int or not 0 <= t < self.checkpoint.config["vocab_size"]
            for t in eos
        ):
            raise RequestValidationError("invalid runner EOS token IDs")
        maximum = len(tokens) + max_new_tokens
        reservation = math.ceil(maximum / KV_STEP) * KV_STEP * self.bytes_per_token
        if maximum > self.max_model_length:
            raise ContextLengthError(
                "runner prompt and output exceed the context limit"
            )
        if reservation > self.kv_cache_bytes:
            raise ContextLengthError(
                f"runner needs {reservation} persistent KV bytes; budget is {self.kv_cache_bytes}"
            )
        self._guard()
        self._cache = self._runtime.make_cache()
        self._reservation, self._peak_bytes, self._current_bytes = reservation, 0, 0
        self._max_tokens, self._produced = max_new_tokens, 0
        self._eos = frozenset(eos)
        self._active = self._next_id
        self._next_id += 1
        try:
            self._iterator = self._runtime.generate(
                tokens,
                max_tokens=max_new_tokens,
                prompt_cache=self._cache,
                prefill_step_size=self.prefill_step_size,
                prompt_progress_callback=self._observe_cache,
            )
        except BaseException:
            self._release()
            self._terminal = None  # Failed admission returned no request ID.
            raise
        return self._active

    def step(self) -> list[TokenEvent]:
        if self._closed or self._active is None:
            raise BackendError("runner has no active request")
        request_id = self._active
        try:
            self._guard()
            token, _ = next(self._iterator)
            self._observe_cache()
            if (
                type(token) is not int
                or not 0 <= token < self.checkpoint.config["vocab_size"]
            ):
                raise BackendError("MLX-LM returned an invalid token")
            self._produced += 1
            finished = token in self._eos or self._produced == self._max_tokens
            if finished:
                self._release()
            return [TokenEvent(request_id, token, finished)]
        except BaseException:
            if self._active is not None:
                self._release()
            raise

    def _release(self) -> None:
        request_id, iterator = self._active, self._iterator
        try:
            if iterator is not None:
                iterator.close()
        finally:
            try:
                self._runtime.synchronize()
            finally:
                for cache in self._cache:
                    cache.keys = cache.values = None
                    cache.offset = 0
                self._cache.clear()
                self._iterator = self._active = None
                self._terminal = request_id
                self._current_bytes = self._reservation = 0
                self._guard = lambda: None
                self._runtime.clear_cache()

    def cancel(self, request_id: int) -> None:
        if request_id == self._active:
            self._release()
        elif request_id != self._terminal:
            raise KeyError("unknown runner request")

    def forget(self, request_id: int) -> None:
        if request_id == self._active:
            raise RuntimeError("cannot forget an active runner request")
        if request_id != self._terminal:
            raise KeyError("unknown runner request")
        self._terminal = None

    def stats(self) -> dict[str, int]:
        return {
            "kv_reserved_bytes": self._reservation,
            "kv_persistent_bytes": self._current_bytes,
            "last_kv_peak_bytes": self._peak_bytes,
            "active_requests": int(self._active is not None),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._active is not None:
                self._release()
        finally:
            try:
                self._runtime.synchronize()
            finally:
                self._terminal = None
                self._guard = lambda: None
                self._runtime.model = None
                self._runtime.generate = lambda *args, **kwargs: iter(())
                self._runtime.make_cache = list
                self._closed = True
                self._runtime.clear_cache()
