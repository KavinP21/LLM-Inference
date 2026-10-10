#!/usr/bin/env python3
"""Qualify ordinary versus speculative multimodal decoding with real n-gram drafts."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from forge_llm.benchmark import source_provenance  # noqa: E402
from forge_llm.qwen35_engine import Qwen35Engine  # noqa: E402
from gpu_guard import exclusive_gpu_workflow  # noqa: E402
from qualify_qwen35_speculative import Report, cleanup_passed  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    from importlib.util import find_spec

    base = Path(find_spec("mlx_vlm").origin).parent
    files = [
        Path(__file__).resolve(),
        *[
            ROOT / name
            for name in (
                "python/forge_llm/qwen35_engine.py",
                "python/forge_llm/qwen35_speculative.py",
                "python/forge_llm/qwen35_state.py",
                "python/forge_llm/qwen35_checkpoint.py",
                "benchmarks/qualify_qwen35_speculative.py",
                "benchmarks/gpu_guard.py",
            )
        ],
        *[
            base / name
            for name in (
                "models/qwen3_5/language.py",
                "models/qwen3_5/speculative_verifier.py",
                "models/qwen3_5/gated_delta.py",
                "models/cache.py",
                "speculative/cache_state.py",
            )
        ],
    ]
    return {
        "runtime": source_provenance(),
        "files_sha256": {str(path): digest(path) for path in files},
    }


def trial(engine, messages, enabled, args):
    engine._spec_options = (
        {
            "draft_tokens": args.draft_tokens,
            "verification_mode": "block",
            "adaptive": True,
        }
        if enabled
        else None
    )
    prepared = engine.prepare_chat(messages, enable_thinking=False)
    grids = {
        name: prepared.inputs[name].tolist()
        for name in ("image_grid_thw", "video_grid_thw")
        if name in prepared.inputs
    }
    started = time.perf_counter()
    request = engine.submit(prepared, args.output_tokens)
    language = engine.model.language_model
    deltas = getattr(language, "_rope_deltas", None)
    position_ids = getattr(language, "_position_ids", None)
    position_state = {
        "rope_deltas": None if deltas is None else deltas.tolist(),
        "stored_position_ids_shape": None
        if position_ids is None
        else list(position_ids.shape),
        "prefill_cache_tokens": engine._cache[3].offset,
    }
    output = []
    try:
        finished = False
        while not finished:
            for event in engine.step():
                output.append(event.token)
                finished = event.finished
    finally:
        if engine._active is not None:
            engine.cancel(request)
        engine.forget(request)
    elapsed = time.perf_counter() - started
    answer = engine.response_metadata(output, input_tokens=prepared.input_tokens)
    return {
        "speculative": enabled,
        "prompt_ids": list(prepared.input_ids),
        "prompt_tokens": prepared.input_tokens,
        "media_fingerprints": list(prepared.media_fingerprints),
        "grids": grids,
        "position_state_after_prefill": position_state,
        "tokens": output,
        "output_tokens": len(output),
        "text": answer.text,
        "generation_seconds_one_observation": elapsed,
        "stats": engine.stats(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/qwen3.5-4b-mlx")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--state-budget-bytes", type=int, default=2 << 30)
    args = parser.parse_args()
    if args.output_tokens < 8:
        raise ValueError("visual speculation requires at least eight output tokens")
    before = sources()
    fixture_hashes = {str(path): digest(path) for path in (args.image, args.video)}
    payload = {
        "schema": "forge_qwen35_multimodal_speculative_parity_v1",
        "source_before": before,
        "script_sha256_before": digest(Path(__file__).resolve()),
        "fixtures_before": fixture_hashes,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {"mlx": version("mlx"), "mlx_vlm": version("mlx-vlm")},
        "rows": [],
        "passed": False,
        "method": "Each ordinary/speculative request freshly prepares the same local image/video through the actual processor, embeddings and mRoPE positions; fresh hybrid caches, same loaded quantized weights. Only real NGramDraft proposals are used. Bounded greedy output token parity and real verification activity are the gates. Generation durations are single observations, not a speed benchmark. No multimodal semantic accuracy, universal numerical parity or original-FP16 claim.",
    }
    report = Report(args.output, payload)
    try:
        with exclusive_gpu_workflow():
            with closing(
                Qwen35Engine(args.model, kv_cache_bytes=args.state_budget_bytes)
            ) as engine:
                checkpoint = engine.checkpoint
                payload["checkpoint_identity"], payload["engine_metadata"] = (
                    engine.checkpoint_identity,
                    engine.build_info(),
                )
                cases = [
                    (
                        "image",
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": str(args.image)},
                                    {
                                        "type": "text",
                                        "text": "Read the image and output its printed text exactly twenty times, one per line. Do not add explanations.",
                                    },
                                ],
                            }
                        ],
                    ),
                    (
                        "video",
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": str(args.video)},
                                    {
                                        "type": "text",
                                        "text": "Observe the video. Print the name of its first solid color and then the name of its last solid color. Repeat that same two-color sequence twenty times, one sequence per line, without explanation.",
                                    },
                                ],
                            }
                        ],
                    ),
                ]
                for name, messages in cases:
                    ordinary = trial(engine, messages, False, args)
                    speculative = trial(engine, messages, True, args)
                    stats = speculative["stats"]["speculative"]
                    positions = [
                        entry["position_state_after_prefill"]
                        for entry in (ordinary, speculative)
                    ]
                    deltas_present = all(
                        entry["rope_deltas"] is not None for entry in positions
                    )
                    passed = (
                        ordinary["tokens"] == speculative["tokens"]
                        and ordinary["prompt_ids"] == speculative["prompt_ids"]
                        and ordinary["media_fingerprints"]
                        == speculative["media_fingerprints"]
                        and ordinary["grids"] == speculative["grids"]
                        and bool(ordinary["grids"])
                        and positions[0] == positions[1]
                        and deltas_present
                        and stats["verification_blocks"] > 0
                        and stats["accepted_draft_tokens"] > 0
                        and all(
                            cleanup_passed(entry["stats"])
                            for entry in (ordinary, speculative)
                        )
                    )
                    report.row(
                        "multimodal-speculative-" + name,
                        {
                            "passed": passed,
                            "messages": messages,
                            "canonical_token_parity": ordinary["tokens"]
                            == speculative["tokens"],
                            "real_verification_blocks": stats["verification_blocks"],
                            "accepted_draft_tokens": stats["accepted_draft_tokens"],
                            "ordinary": ordinary,
                            "speculative": speculative,
                        },
                    )
                payload["checkpoint_unchanged"] = checkpoint.unchanged()
    except BaseException as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)[:4096]}
        raise
    finally:
        payload["source_after"] = sources()
        payload["script_sha256_after"] = digest(Path(__file__).resolve())
        payload["fixtures_after"] = {
            str(path): digest(path) for path in (args.image, args.video)
        }
        payload["source_unchanged"] = (
            before["files_sha256"] == payload["source_after"]["files_sha256"]
            and all(
                before["runtime"][key] == payload["source_after"]["runtime"][key]
                for key in ("git_commit", "runtime_source_sha256")
            )
            and payload["script_sha256_before"] == payload["script_sha256_after"]
        )
        payload["fixtures_unchanged"] = fixture_hashes == payload["fixtures_after"]
        payload["passed"] = (
            len(payload["rows"]) == 2
            and all(row["passed"] for row in payload["rows"])
            and payload["source_unchanged"]
            and payload["fixtures_unchanged"]
            and payload.get("checkpoint_unchanged", False)
            and "error" not in payload
        )
        report.save()
    if not payload["passed"]:
        raise SystemExit(
            "Multimodal speculative qualification failed; raw evidence preserved"
        )


if __name__ == "__main__":
    main()
