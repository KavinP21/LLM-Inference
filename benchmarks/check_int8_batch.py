"""Frozen-policy arrival/cancellation regression against saved independent outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.runtime import SequenceState
from forge_llm.validate_int8 import compare_logits
from gpu_guard import exclusive_gpu_workflow


@contextmanager
def trace_step_logits(engine, enabled):
    """Diagnostic-only host copies; never enable for performance measurements."""
    data = {}
    if not enabled:
        yield data
        return
    next_schedule = engine.scheduler.next
    decode, prefill = engine.model.decode_paged_batch, engine.model.forward_paged_chunk
    schedule = None

    def traced_next():
        nonlocal schedule
        data.clear()
        schedule = next_schedule()
        return schedule

    def traced_decode(*args, **kwargs):
        logits = decode(*args, **kwargs)
        for index, request_id in enumerate(schedule.decode):
            data[request_id] = np.array(logits[index])
        return logits

    def traced_prefill(*args, **kwargs):
        logits = prefill(*args, **kwargs)
        data[schedule.prefill] = np.array(logits[-1])
        return logits

    engine.scheduler.next = traced_next
    engine.model.decode_paged_batch, engine.model.forward_paged_chunk = (
        traced_decode,
        traced_prefill,
    )
    try:
        yield data
    finally:
        engine.scheduler.next = next_schedule
        engine.model.decode_paged_batch, engine.model.forward_paged_chunk = (
            decode,
            prefill,
        )


def check(
    model: Path,
    quality_path: Path,
    mode: str,
    *,
    fp16_source: bool = False,
    diagnose: bool = False,
) -> dict:
    quality = json.loads(quality_path.read_text())
    identity = model_provenance(model)
    comparison_model = "source_model" if fp16_source else "quantized_model"
    token_field = "fp16_tokens" if fp16_source else "int8_tokens"
    if identity["model_data_sha256"] != quality[comparison_model]["model_data_sha256"]:
        raise ValueError("independent outputs belong to a different artifact")
    if not fp16_source and quality["configuration"]["int8_mode"] != mode:
        raise ValueError("independent outputs use a different execution mode")
    if (
        source_provenance()["runtime_source_sha256"]
        != quality["environment"]["runtime_source_sha256"]
    ):
        raise ValueError("runtime changed since independent output validation")
    cases = quality["cases"]
    budgets = [1, 4, 8, 16, 32]
    with (
        MlxEngine(
            model,
            int8_mode="auto" if fp16_source else mode,
            max_model_length=2048,
            max_num_sequences=8,
            kv_cache_bytes=512 << 20,
        ) as engine,
        trace_step_logits(engine, diagnose) as raw_logits,
    ):
        cancelled = engine.submit(cases[0]["input_ids"], 32)
        engine.step()
        allocated_before_cancel = engine.stats()["kv_cache"]["allocated_blocks"]
        engine.cancel(cancelled)
        cancellation_reclaimed = (
            engine.stats()["kv_cache"]["allocated_blocks"] == 0
            and engine.stats()["kv_cache"]["reserved_blocks"] == 0
            and engine.stats()["kv_device_bytes"] == 0
        )
        submitted, iterations, pending = [], 0, 0
        event_counts = {}
        request_cases, divergences = {}, {}
        while pending < len(cases) or any(
            engine.scheduler.request(request_id).state is not SequenceState.COMPLETED
            for request_id, _ in submitted
        ):
            # Insert a new request every other iteration, while earlier requests
            # remain active. Short and long outputs cause dynamic batch removal.
            if pending < len(cases) and iterations % 2 == 0:
                count = min(
                    budgets[pending % len(budgets)], len(cases[pending][token_field])
                )
                request_id = engine.submit(
                    cases[pending]["input_ids"], count, eos_token_ids=[]
                )
                submitted.append((request_id, count))
                request_cases[request_id] = pending
                event_counts[request_id] = 0
                pending += 1
            for event in engine.step():
                if event.request_id not in event_counts:
                    raise RuntimeError("unexpected or cancelled request executed")
                event_counts[event.request_id] += 1
                if diagnose and event.request_id not in divergences:
                    case_index = request_cases[event.request_id]
                    output = engine.scheduler.request(event.request_id).output
                    position = len(output) - 1
                    expected = cases[case_index][token_field]
                    if (
                        output[:-1] == expected[:position]
                        and output[-1] != expected[position]
                    ):
                        divergences[event.request_id] = (
                            case_index,
                            position,
                            raw_logits[event.request_id],
                        )
            iterations += 1
            if iterations > 4096:
                raise RuntimeError("arrival trace failed to drain")
        observations = []
        for case, (request_id, count) in zip(cases, submitted):
            actual = engine.scheduler.request(request_id).output
            expected = case[token_field][:count]
            observations.append(
                {
                    "request_id": request_id,
                    "prompt_tokens": len(case["input_ids"]),
                    "requested_output_tokens": count,
                    "independent_tokens": expected,
                    "batched_tokens": actual,
                    "exact_parity": actual == expected,
                    "events_exact": event_counts[request_id] == count,
                }
            )
        stats = engine.stats()
        reclaimed = (
            stats["kv_cache"]["allocated_blocks"] == 0
            and stats["kv_cache"]["reserved_blocks"] == 0
            and stats["kv_device_bytes"] == 0
        )
        diagnostics = []
        for case_index, position, actual_logits in divergences.values():
            with (
                MlxEngine(
                    model,
                    int8_mode="auto" if fp16_source else mode,
                    max_model_length=2048,
                    kv_cache_bytes=512 << 20,
                ) as independent,
                trace_step_logits(independent, True) as reference_logits,
            ):
                request_id = independent.submit(
                    cases[case_index]["input_ids"], position + 1
                )
                while (
                    independent.scheduler.request(request_id).state
                    is not SequenceState.COMPLETED
                ):
                    independent.step()
                diagnostics.append(
                    {
                        "case_index": case_index,
                        "position": position,
                        "independent_output_reproduced": independent.scheduler.request(
                            request_id
                        ).output
                        == cases[case_index][token_field][: position + 1],
                        "batched_vs_independent_logits": compare_logits(
                            reference_logits[request_id], actual_logits
                        ),
                    }
                )
        return {
            "schema_version": 1,
            "purpose": "arrival/batching/cancellation correctness against independent same-artifact outputs; not cross-artifact parity or timing",
            "model": identity,
            "environment": {
                "platform": platform.platform(),
                "native_build": engine.build_info(),
                **source_provenance(),
            },
            "configuration": {
                "int8_mode": mode,
                "fp16_source_baseline": fp16_source,
                "logit_diagnostics_enabled": diagnose,
                "max_num_sequences": 8,
                "insert_every_iterations": 2,
                "output_budgets": budgets,
                "quality_report_sha256": hashlib.sha256(
                    quality_path.read_bytes()
                ).hexdigest(),
                "checker_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            "cancellation": {
                "blocks_before": allocated_before_cancel,
                "reclaimed": cancellation_reclaimed,
            },
            "summary": {
                "exact_cases": sum(o["exact_parity"] for o in observations),
                "total_cases": len(observations),
                "iterations": iterations,
                "cache_reclaimed": reclaimed,
                "passed": reclaimed
                and cancellation_reclaimed
                and all(o["exact_parity"] and o["events_exact"] for o in observations),
            },
            "requests": observations,
            "logit_diagnostics": diagnostics,
            "engine_stats": stats,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--mode", choices=["metal", "reconstruct"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp16-source", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="copy and independently replay first-divergence logits; never a speed run",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite batch evidence: {args.output}")
    with exclusive_gpu_workflow():
        result = check(
            args.model,
            args.quality,
            args.mode,
            fp16_source=args.fp16_source,
            diagnose=args.diagnose,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
