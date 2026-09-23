from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import build_info, create_engine
from .format import HEADER, MAGIC


@dataclass
class RequestMetric:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float | None


class GpuSampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="nvidia-smi-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.samples:
            return {
                "sample_count": 0,
                "gpu_utilization_mean_percent": None,
                "gpu_utilization_peak_percent": None,
                "memory_peak_mib": None,
                "power_mean_watts": None,
                "power_peak_watts": None,
            }
        utilization = [item["utilization"] for item in self.samples]
        memory = [item["memory"] for item in self.samples]
        power = [item["power"] for item in self.samples]
        return {
            "sample_count": len(self.samples),
            "gpu_utilization_mean_percent": statistics.fmean(utilization),
            "gpu_utilization_peak_percent": max(utilization),
            "memory_peak_mib": max(memory),
            "power_mean_watts": statistics.fmean(power),
            "power_peak_watts": max(power),
        }

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                line = subprocess.run(
                    command, check=False, capture_output=True, text=True, timeout=2
                ).stdout.splitlines()[0]
                utilization, memory, power = (
                    float(value.strip()) for value in line.split(",")
                )
                self.samples.append(
                    {"utilization": utilization, "memory": memory, "power": power}
                )
            except (
                FileNotFoundError,
                subprocess.TimeoutExpired,
                IndexError,
                ValueError,
            ):
                return
            self._stop.wait(self.interval_seconds)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def command_output(command: list[str]) -> str | None:
    try:
        return (
            subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def model_provenance(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("model file is shorter than its fixed header")
    magic, version, _, _, _, _, data_digest = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("model file has invalid magic")
    return {
        "model_format_version": version,
        "model_data_sha256": data_digest.hex(),
        "model_file_bytes": path.stat().st_size,
    }


def run_once(
    engine: object, prompts: list[list[int]], output_length: int, eos_ids: list[int]
) -> tuple[list[RequestMetric], float, dict[str, float | int]]:
    submitted: dict[int, int] = {}
    first_token: dict[int, int] = {}
    finished: dict[int, int] = {}
    output_counts: dict[int, int] = {}
    prompt_sizes: dict[int, int] = {}
    start = time.perf_counter_ns()
    peak_cache = {
        "allocated_blocks": 0,
        "reserved_blocks": 0,
        "internal_fragmentation_tokens": 0,
        "occupancy": 0.0,
    }
    for prompt in prompts:
        request_id = engine.submit(prompt, output_length, eos_ids)
        submitted[request_id] = time.perf_counter_ns()
        output_counts[request_id] = 0
        prompt_sizes[request_id] = len(prompt)
    while len(finished) < len(prompts):
        for event in engine.step():
            now = time.perf_counter_ns()
            first_token.setdefault(event.request_id, now)
            output_counts[event.request_id] += 1
            if event.finished:
                finished[event.request_id] = now
        cache = engine.stats()["kv_cache"]
        for key, peak in peak_cache.items():
            peak_cache[key] = max(peak, cache[key])
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    metrics = []
    for request_id, submitted_at in submitted.items():
        ttft = (first_token[request_id] - submitted_at) / 1e6
        e2e = (finished[request_id] - submitted_at) / 1e6
        count = output_counts[request_id]
        metrics.append(
            RequestMetric(
                request_id,
                prompt_sizes[request_id],
                count,
                ttft,
                e2e,
                (e2e - ttft) / (count - 1) if count > 1 else None,
            )
        )
    return metrics, wall_ms, peak_cache


def benchmark(
    model_path: Path,
    tokenizer_name: str,
    prompt_file: Path,
    output_length: int,
    warmups: int,
    repetitions: int,
    max_sequences: int,
    max_model_length: int,
    kv_cache_mib: int,
    stop_on_eos: bool = False,
    backend: str = "auto",
    mlx_kernel_mode: str = "full",
    capture_path: Path | None = None,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    payload = json.loads(prompt_file.read_text())
    if not isinstance(payload, list):
        raise TypeError("prompt file must be a JSON array")
    if all(isinstance(item, str) for item in payload):
        prompts = [
            tokenizer(text, add_special_tokens=True).input_ids for text in payload
        ]
    elif all(
        isinstance(item, list)
        and item
        and all(isinstance(token, int) for token in item)
        for item in payload
    ):
        prompts = payload
    else:
        raise ValueError("prompts must be all strings or all non-empty token-id arrays")
    prompts = [
        tokens for tokens in prompts if len(tokens) + output_length <= max_model_length
    ]
    if not prompts or len(prompts) > max_sequences:
        raise ValueError("prompt count must be between 1 and max-sequences")
    engine_options = {
        "max_num_sequences": max_sequences,
        "max_model_length": max_model_length,
        "kv_cache_bytes": kv_cache_mib << 20,
    }
    if backend == "mlx" or (backend == "auto" and platform.system() == "Darwin"):
        if mlx_kernel_mode not in {"baseline", "fused", "full"}:
            raise ValueError("unknown MLX kernel mode")
        engine_options["custom_metal"] = mlx_kernel_mode != "baseline"
        engine_options["metal_paged_attention"] = mlx_kernel_mode == "full"
    engine = create_engine(model_path, backend=backend, **engine_options)
    eos = [int(tokenizer.eos_token_id)] if stop_on_eos else []
    for _ in range(warmups):
        run_once(engine, prompts, min(output_length, 4), eos)

    all_metrics: list[RequestMetric] = []
    walls = []
    cache_peaks = []
    gpu_sampler = GpuSampler()
    gpu_sampler.start()
    try:
        for repetition in range(repetitions):
            capturing = capture_path is not None and repetition == 0
            if capturing:
                if getattr(engine, "backend", "cuda") != "mlx":
                    raise ValueError("Metal capture requires the MLX backend")
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    engine.model.mx.metal.start_capture(str(capture_path))
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Metal capture failed; relaunch with MTL_CAPTURE_ENABLED=1"
                    ) from exc
            try:
                metrics, wall, cache_peak = run_once(
                    engine, prompts, output_length, eos
                )
            finally:
                if capturing:
                    engine.model.mx.metal.stop_capture()
            all_metrics.extend(metrics)
            walls.append(wall)
            cache_peaks.append(cache_peak)
    finally:
        gpu_metrics = gpu_sampler.stop()
    ttft = [item.ttft_ms for item in all_metrics]
    tpot = [item.tpot_ms for item in all_metrics if item.tpot_ms is not None]
    e2e = [item.e2e_ms for item in all_metrics]
    output_tokens = sum(item.output_tokens for item in all_metrics)
    prompt_tokens = sum(item.prompt_tokens for item in all_metrics)
    total_seconds = sum(walls) / 1000.0
    result = {
        "schema_version": 1,
        "implementation": (
            f"forge-mlx-fp16-paged-{mlx_kernel_mode}"
            if getattr(engine, "backend", "cuda") == "mlx"
            else "forge-cuda-fp16-paged"
        ),
        "configuration": {
            "model_path": str(model_path),
            "tokenizer": tokenizer_name,
            "output_length": output_length,
            "warmups": warmups,
            "repetitions": repetitions,
            "concurrency": len(prompts),
            "prompt_token_lengths": [len(prompt) for prompt in prompts],
            "max_model_length": max_model_length,
            "kv_cache_mib": kv_cache_mib,
            "stop_on_eos": stop_on_eos,
            "backend": getattr(engine, "backend", "cuda"),
            "mlx_kernel_mode": mlx_kernel_mode,
            "capture_path": str(capture_path) if capture_path is not None else None,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_build": engine.build_info()
            if hasattr(engine, "build_info")
            else build_info(),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": command_output(["nvcc", "--version"]),
        },
        "model": model_provenance(model_path),
        "aggregate": {
            "generated_tokens_per_second": output_tokens / total_seconds,
            "prompt_tokens_per_second": prompt_tokens / total_seconds,
            "ttft_ms": {
                "p50": percentile(ttft, 0.5),
                "p95": percentile(ttft, 0.95),
                "p99": percentile(ttft, 0.99),
            },
            "tpot_ms": {
                "p50": percentile(tpot, 0.5),
                "p95": percentile(tpot, 0.95),
                "p99": percentile(tpot, 0.99),
            },
            "e2e_ms": {
                "p50": percentile(e2e, 0.5),
                "p95": percentile(e2e, 0.95),
                "p99": percentile(e2e, 0.99),
            },
        },
        "gpu_samples": gpu_metrics,
        "peak_kv_cache": {
            key: max(item[key] for item in cache_peaks) for key in cache_peaks[0]
        },
        "engine_stats": engine.stats(),
        "requests": [asdict(item) for item in all_metrics],
    }
    if hasattr(engine, "close"):
        engine.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Forge LLM continuous batching"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "cuda", "mlx"], default="auto")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON array of strings or pre-tokenized integer arrays",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--max-model-length", type=int, default=2048)
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="allow EOS to shorten requests (disabled for fixed-length benchmarks)",
    )
    parser.add_argument(
        "--mlx-kernel-mode",
        choices=["baseline", "fused", "full"],
        default="full",
        help="select the MLX custom-kernel ablation",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="optional Xcode .gputrace path for the first measured MLX trial",
    )
    args = parser.parse_args()
    result = benchmark(
        args.model,
        args.tokenizer,
        args.prompts,
        args.output_length,
        args.warmups,
        args.repetitions,
        args.max_sequences,
        args.max_model_length,
        args.kv_cache_mib,
        args.stop_on_eos,
        args.backend,
        args.mlx_kernel_mode,
        args.capture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
