#!/usr/bin/env python3
"""Reproduce bounded Qwen3.5 text, visual, thinking and lifecycle checks.

Uses the ordinary pinned MLX-VLM generator as the text numerical reference and
authored local image/video fixtures for functional checks. Optional sampling is
a separate finite-output smoke check, not an accuracy or distribution claim.
Use qualify_qwen35_speculative.py for transaction correctness and speed trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from forge_llm.benchmark import source_provenance  # noqa: E402 - direct checkout CLI
from forge_llm.qwen35_engine import Qwen35Engine  # noqa: E402
from gpu_guard import exclusive_gpu_workflow  # noqa: E402
from qualify_qwen35_speculative import Report, cleanup_passed  # noqa: E402


def make_fixtures(directory, font_path):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    directory.mkdir(parents=True, exist_ok=False)
    font = (
        ImageFont.truetype(str(font_path), 64)
        if font_path is not None and font_path.is_file()
        else ImageFont.load_default(size=64)
    )
    image = Image.new("RGB", (640, 240), "white")
    ImageDraw.Draw(image).text((40, 70), "FORGE 42", font=font, fill="black")
    image_path = directory / "text.png"
    image.save(image_path)
    video_path = directory / "sequence.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (256, 256)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the authored video fixture")
    try:
        for index in range(8):
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            frame[:] = (0, 0, 255) if index < 4 else (255, 0, 0)
            writer.write(frame)
    finally:
        writer.release()
    return image_path, video_path


def ordinary_reference(engine, prepared, max_tokens, prefill_chunk_size):
    from mlx_vlm.generate.ar import generate_step

    inputs = dict(prepared.inputs)
    ids = inputs.pop("input_ids")
    pixels, mask = inputs.pop("pixel_values", None), inputs.pop("attention_mask", None)
    engine._runtime.reset_positions()
    caches = engine._runtime.make_cache()
    iterator = generate_step(
        ids,
        engine.model,
        pixels,
        mask,
        max_tokens=max_tokens,
        temperature=0.0,
        prompt_cache=caches,
        prefill_step_size=prefill_chunk_size,
        **inputs,
    )
    output = []
    try:
        for token, _ in iterator:
            output.append(int(token))
            if int(token) in engine._eos:
                break
    finally:
        try:
            iterator.close()
            engine._runtime.synchronize()
        finally:
            for cache in caches:
                if hasattr(cache, "cache"):
                    cache.cache = [None, None]
                else:
                    cache.keys = cache.values = None
                    cache.offset = 0
            caches.clear()
            engine._runtime.reset_positions()
            engine._runtime.clear_cache()
    return output


def parse_colors(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate video-result JSON key")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("nonfinite video-result JSON constant")

    selected = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", selected, re.DOTALL)
    if fenced is not None:
        selected = fenced.group(1)
    try:
        return json.loads(
            selected, object_pairs_hook=unique, parse_constant=reject_constant
        )
    except ValueError:
        return None


def normalize_colors(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"first", "last"}
        or not all(isinstance(color, str) for color in value.values())
    ):
        return None
    return {key: color.lower() for key, color in value.items()}


def full_context_run(engine):
    previous_eos = engine._eos
    try:
        full = engine.generate([1234] * (engine.max_model_length - 2), 2, [0])
    finally:
        # Keep this synthetic probe's EOS policy local even for versions that
        # retain the last request's EOS default.
        engine._eos = previous_eos
    stats = engine.stats()
    return {
        "passed": len(full) == 2 and cleanup_passed(stats),
        "prompt_tokens": engine.max_model_length - 2,
        "output_tokens": len(full),
        "actual": full,
        "synthetic_token_id": 1234,
        "eos_token_ids": [0],
        "peak_logical_persistent_state_bytes": stats["last_kv_peak_bytes"],
        "stats": stats,
        "scope": "Execution and resource cleanup at the configured context limit only; no long-context semantic-quality claim.",
    }


def qualify(engine, args, report, fixtures):
    first_ids, first_reference = None, None
    cases = {
        "short": "What is 17+23? Answer with the number only.",
        "code": "Write a Python function returning the square of its integer input. Return code only.",
        "long": (
            "A record contains three fields: name Ada; color blue; number 42.\n"
            * args.long_records
        )
        + "Return the name from the records only.",
    }
    for name, prompt in cases.items():
        prepared = engine.prepare_chat([{"role": "user", "content": prompt}])
        reference = ordinary_reference(
            engine, prepared, args.parity_tokens, args.prefill_chunk_size
        )
        actual = engine.generate(prepared, args.parity_tokens)
        if first_ids is None:
            first_ids, first_reference = prepared.input_ids, reference
        report.row(
            "parity-" + name,
            {
                "passed": reference == actual and cleanup_passed(engine.stats()),
                "prompt": prompt,
                "prompt_tokens": prepared.input_tokens,
                "reference": reference,
                "actual": actual,
                "stats": engine.stats(),
            },
        )
    image_path, video_path = fixtures
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": "Read the large printed text. Return only that text.",
                },
            ],
        }
    ]
    answer = engine.generate_chat(image_messages, args.visual_tokens)
    report.row(
        "image-ocr",
        {
            "passed": "FORGE 42" in answer.text.upper()
            and cleanup_passed(engine.stats()),
            "text": answer.text,
            "stats": engine.stats(),
        },
    )
    follow = [
        *image_messages,
        {"role": "assistant", "content": answer.text},
        {
            "role": "user",
            "content": "What number was printed in the image? Answer with the number only.",
        },
    ]
    answer = engine.generate_chat(follow, args.parity_tokens)
    report.row(
        "image-followup",
        {
            "passed": answer.text.strip() == "42" and cleanup_passed(engine.stats()),
            "text": answer.text,
            "stats": engine.stats(),
        },
    )
    video_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {
                    "type": "text",
                    "text": "Which solid color is shown first, and which solid color is shown last? Return JSON with keys first and last.",
                },
            ],
        }
    ]
    answer = engine.generate_chat(video_messages, args.visual_tokens)
    colors = parse_colors(answer.text)
    normalized = normalize_colors(colors)
    correct = normalized == {"first": "red", "last": "blue"}
    report.row(
        "video-order",
        {
            "passed": correct and cleanup_passed(engine.stats()),
            "text": answer.text,
            "parsed": colors,
            "normalized": normalized,
            "stats": engine.stats(),
        },
    )
    mixed = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "video", "video": str(video_path)},
                {
                    "type": "text",
                    "text": "Read the image text and describe the first and last colors in the video. Answer briefly.",
                },
            ],
        }
    ]
    answer = engine.generate_chat(mixed, args.visual_tokens)
    text = answer.text.lower()
    correct = (
        "forge 42" in text
        and "red" in text
        and "blue" in text
        and text.index("red") < text.index("blue")
    )
    report.row(
        "mixed-image-video",
        {
            "passed": correct and cleanup_passed(engine.stats()),
            "text": answer.text,
            "stats": engine.stats(),
        },
    )
    answer = engine.generate_chat(
        [{"role": "user", "content": cases["short"]}],
        args.thinking_tokens,
        enable_thinking=True,
    )
    report.row(
        "thinking",
        {
            "passed": answer.thinking_complete
            and answer.text.strip() == "40"
            and cleanup_passed(engine.stats()),
            "text": answer.text,
            "reasoning_present": bool(answer.reasoning),
            "thinking_complete": answer.thinking_complete,
            "stats": engine.stats(),
        },
    )
    answer = engine.generate_chat(
        [
            {
                "role": "user",
                "content": "Prove that the sum of the first n odd integers is n squared.",
            }
        ],
        1,
        enable_thinking=True,
    )
    report.row(
        "thinking-truncation",
        {
            "passed": not answer.thinking_complete
            and not answer.text
            and cleanup_passed(engine.stats()),
            "text": answer.text,
            "truncated": answer.truncated,
            "thinking_complete": answer.thinking_complete,
            "stats": engine.stats(),
        },
    )
    request = engine.submit([1234] * args.cancel_prompt_tokens, args.parity_tokens)
    engine.step()
    engine.cancel(request)
    engine.forget(request)
    cleared = engine.stats()
    restarted = engine.generate(first_ids, args.parity_tokens)
    report.row(
        "post-prefill-cancel-and-restart",
        {
            "passed": cleanup_passed(cleared) and restarted == first_reference,
            "restart_actual": restarted,
            "reference": first_reference,
            "stats": cleared,
        },
    )
    if args.full_context_check:
        report.row(
            "full-configured-context-execution",
            full_context_run(engine),
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new evidence file; refuses overwrite",
    )
    result.add_argument("--fixtures-dir", type=Path)
    result.add_argument(
        "--font",
        type=Path,
        default=Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    result.add_argument("--max-model-length", type=int, default=32768)
    result.add_argument("--state-budget-bytes", type=int, default=2 << 30)
    result.add_argument("--prefill-chunk-size", type=int, default=256)
    result.add_argument("--parity-tokens", type=int, default=32)
    result.add_argument("--visual-tokens", type=int, default=96)
    result.add_argument("--thinking-tokens", type=int, default=256)
    result.add_argument("--long-records", type=int, default=380)
    result.add_argument("--cancel-prompt-tokens", type=int, default=8191)
    result.add_argument(
        "--sampling-smoke",
        action="store_true",
        help="reload once for temp=.7/top_p=.8/top_k=20/presence_penalty=1.5 smoke",
    )
    result.add_argument("--sampling-seed", type=int, default=42)
    result.add_argument(
        "--full-context-check",
        action="store_true",
        help="execution-only synthetic context_limit−2 prompt plus two outputs",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if any(
        value <= 0
        for value in (
            args.parity_tokens,
            args.visual_tokens,
            args.thinking_tokens,
            args.long_records,
            args.cancel_prompt_tokens,
        )
    ):
        raise ValueError("qualification lengths must be positive")
    fixtures_dir = args.fixtures_dir or args.output.with_name(
        args.output.stem + "-fixtures"
    )
    payload = {
        "schema": "forge_qwen35_qualification_v2",
        "source": source_provenance(),
        "implementation_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/qualify_qwen35.py",
                "benchmarks/qualify_qwen35_speculative.py",
                "python/forge_llm/qwen35_engine.py",
                "python/forge_llm/qwen35_checkpoint.py",
            )
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": version("mlx"),
            "mlx_vlm": version("mlx-vlm"),
        },
        "rows": [],
        "passed": False,
        "method": "Direct ordinary pinned MLX-VLM generate_step text reference; same affine4-bit weights shared sequentially, fresh mixed caches. Authored image/video recognition checks are functional evidence, not multimodal numerical parity. Sampling, when enabled, checks finite bounded nonempty output only. No original-FP16 parity, full advertised context, broad task quality or speed claim.",
    }
    report = Report(args.output, payload)
    settings = {
        "max_model_length": args.max_model_length,
        "kv_cache_bytes": args.state_budget_bytes,
        "prefill_chunk_size": args.prefill_chunk_size,
    }
    try:
        with exclusive_gpu_workflow():
            fixtures = make_fixtures(fixtures_dir, args.font)
            payload["fixtures"] = {
                path.name: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in fixtures
            }
            with closing(Qwen35Engine(args.model, **settings)) as engine:
                payload["checkpoint_identity"], payload["engine_metadata"] = (
                    engine.checkpoint_identity,
                    engine.build_info(),
                )
                qualify(engine, args, report, fixtures)
            if args.sampling_smoke:
                sampling = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                }
                with closing(
                    Qwen35Engine(args.model, **settings, **sampling)
                ) as engine:
                    engine._runtime.mx.random.seed(args.sampling_seed)
                    answer = engine.generate_chat(
                        [
                            {
                                "role": "user",
                                "content": "Write one sentence explaining what a cache does.",
                            }
                        ],
                        args.visual_tokens,
                    )
                    report.row(
                        "sampling-smoke",
                        {
                            "passed": bool(answer.text.strip())
                            and 0 < answer.output_tokens <= args.visual_tokens
                            and cleanup_passed(engine.stats()),
                            "text": answer.text,
                            "output_tokens": answer.output_tokens,
                            "sampling": sampling,
                            "seed": args.sampling_seed,
                            "stats": engine.stats(),
                        },
                    )
            else:
                payload["sampling"] = "not executed; use --sampling-smoke"
        payload["passed"] = all(row["passed"] for row in payload["rows"])
    except BaseException as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)[:4096]}
        report.save()
        raise
    report.save()
    if not payload["passed"]:
        raise SystemExit("Qwen3.5 core qualification failed; raw evidence preserved")


if __name__ == "__main__":
    main()
