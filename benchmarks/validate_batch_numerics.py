"""Independent-versus-continuous-batch logits/tokens; diagnostic, not timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.runtime import SequenceState
from gpu_guard import exclusive_gpu_workflow


def fingerprint(logits):
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("nonfinite or malformed logits")
    top = np.argpartition(values, -2)[-2:]
    top = top[np.argsort(values[top])[::-1]]
    return {
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "argmax": int(np.argmax(values)),
        "top_two": [{"token": int(i), "value": float(values[i])} for i in top],
    }


@contextmanager
def observe_logits(engine):
    """Copy once per emitted row after the adapter's normal materialization."""
    data = {}
    next_schedule = engine.scheduler.next
    decode, prefill = engine.model.decode_paged_batch, engine.model.forward_paged_chunk
    schedule = None

    def next_iteration():
        nonlocal schedule
        data.clear()
        schedule = next_schedule()
        return schedule

    def decode_step(*args, **kwargs):
        logits = decode(*args, **kwargs)
        for row, request_id in enumerate(schedule.decode):
            data[request_id] = fingerprint(np.array(logits[row]))
        return logits

    def prefill_step(*args, **kwargs):
        logits = prefill(*args, **kwargs)
        request = engine.scheduler.request(schedule.prefill)
        if kwargs["start_position"] + len(args[0]) == len(request.prompt):
            data[schedule.prefill] = fingerprint(np.array(logits[-1]))
        return logits

    engine.scheduler.next = next_iteration
    engine.model.decode_paged_batch = decode_step
    engine.model.forward_paged_chunk = prefill_step
    try:
        yield data
    finally:
        engine.scheduler.next = next_schedule
        engine.model.decode_paged_batch = decode
        engine.model.forward_paged_chunk = prefill


def reclaimed(stats):
    return (
        all(
            stats["kv_cache"][name] == 0
            for name in ("allocated_blocks", "reserved_blocks")
        )
        and stats["kv_device_bytes"] == 0
    )


def replay(engine, cases, references, *, staggered=False):
    """All cases run, including waiting requests; no dropped or truncated work."""
    ids, pending, iterations = {}, 0, 0
    observations = []
    logs = {}
    event_counts = {}
    peak_batch = 0
    with observe_logits(engine) as current:
        while pending < len(cases) or any(
            engine.scheduler.request(r).state is not SequenceState.COMPLETED
            for r in ids
        ):
            if not staggered or iterations % 2 == 0:
                live = sum(
                    engine.scheduler.request(r).state is not SequenceState.COMPLETED
                    for r in ids
                )
                while pending < len(cases) and live < engine.max_num_sequences:
                    budget = [1, 4, 8, 16, 32][pending % 5] if staggered else 32
                    r = engine.submit(cases[pending]["input_ids"], budget)
                    ids[r] = (pending, budget)
                    logs[r], event_counts[r] = [], 0
                    pending += 1
                    live += 1
                    if staggered:
                        break
            events = engine.step()
            peak_batch = max(peak_batch, len(events))
            for event in events:
                if event.request_id not in ids:
                    raise RuntimeError("unexpected/cancelled request emitted a token")
                logs[event.request_id].append(current[event.request_id])
                event_counts[event.request_id] += 1
            iterations += 1
            if iterations > 4096:
                raise RuntimeError("arrival workload failed to drain")
    for request_id, (index, count) in ids.items():
        reference = references[index]
        tokens = engine.scheduler.request(request_id).output
        logit_matches = [
            a["sha256"] == b["sha256"]
            for a, b in zip(logs[request_id], reference["logits"][:count], strict=True)
        ]
        observations.append(
            {
                "case_index": index,
                "request_id": request_id,
                "output_budget": count,
                "tokens": tokens,
                "exact_tokens": tokens == reference["tokens"][:count],
                "exact_logit_rows": sum(logit_matches),
                "total_logit_rows": count,
                "events_exact": event_counts[request_id] == count,
                "logits": logs[request_id],
            }
        )
    stats = engine.stats()
    return {
        "configuration": {
            "max_num_sequences": engine.max_num_sequences,
            "staggered": staggered,
            "insert_every_iterations": 2 if staggered else 0,
            "decode_mode": engine.model.decode_mode,
        },
        "summary": {
            "cases": len(cases),
            "exact_token_cases": sum(o["exact_tokens"] for o in observations),
            "exact_logit_rows": sum(o["exact_logit_rows"] for o in observations),
            "total_logit_rows": sum(o["total_logit_rows"] for o in observations),
            "events_exact": all(o["events_exact"] for o in observations),
            "cache_reclaimed": reclaimed(stats),
            "iterations": iterations,
            "peak_emitted_requests_per_iteration": peak_batch,
        },
        "requests": observations,
        "engine_stats": stats,
    }


def validate(model_path, quality_path, int8_mode, *, concurrency=(2, 8, 16)):
    # Saved tokens are a historical-default regression, NOT fresh references.
    # Fresh independent logits/outputs below use the same current source hash
    # as every mode/pattern; never bypass the old checker's source-hash guard.
    quality = json.loads(quality_path.read_text())
    model = model_provenance(model_path)
    source = int8_mode == "auto"
    key, field = (
        ("source_model", "fp16_tokens")
        if source
        else ("quantized_model", "int8_tokens")
    )
    if model["model_data_sha256"] != quality[key]["model_data_sha256"]:
        raise ValueError("historical report belongs to a different artifact")
    if not source and quality["configuration"]["int8_mode"] != int8_mode:
        raise ValueError("historical report uses a different INT8 mode")
    cases = quality["cases"]
    options = {
        "max_model_length": 2048,
        "kv_cache_bytes": 512 << 20,
        "int8_mode": int8_mode,
    }
    references, cancellation = [], []
    with (
        MlxEngine(model_path, max_num_sequences=1, **options) as engine,
        observe_logits(engine) as current,
    ):
        for case in cases:
            request_id = engine.submit(case["input_ids"], 32)
            logits = []
            for _ in range(4096):
                for event in engine.step():
                    logits.append(current[event.request_id])
                if (
                    engine.scheduler.request(request_id).state
                    is SequenceState.COMPLETED
                ):
                    break
            else:
                raise RuntimeError("independent request failed to complete")
            tokens = engine.scheduler.request(request_id).output
            references.append(
                {
                    "input_ids": case["input_ids"],
                    "tokens": tokens,
                    "logits": logits,
                    "historical_tokens_unchanged": tokens == case[field],
                }
            )
        reference_reclaimed = reclaimed(engine.stats())
        build = engine.build_info()
    modes = {}
    for mode in ("batched", "rowwise"):
        workloads = []
        for width, staggered in [(c, False) for c in concurrency] + [(8, True)]:
            with MlxEngine(
                model_path, max_num_sequences=width, decode_mode=mode, **options
            ) as engine:
                cancelled = engine.submit(cases[0]["input_ids"], 32)
                engine.step()
                before = engine.stats()["kv_cache"]["allocated_blocks"]
                engine.cancel(cancelled)
                cancellation.append(
                    {
                        "decode_mode": mode,
                        "concurrency": width,
                        "blocks_before": before,
                        "reclaimed": reclaimed(engine.stats()),
                    }
                )
                workloads.append(replay(engine, cases, references, staggered=staggered))
            print(
                f"{model_path.name} {mode} c={width} staggered={staggered}: {workloads[-1]['summary']}",
                flush=True,
            )
        summaries = [w["summary"] for w in workloads]
        modes[mode] = {
            "workloads": workloads,
            "gates": {
                "exact_tokens": all(
                    s["exact_token_cases"] == len(cases) for s in summaries
                ),
                "exact_logits": all(
                    s["exact_logit_rows"] == s["total_logit_rows"] for s in summaries
                ),
                "events_exact": all(s["events_exact"] for s in summaries),
                "cache_reclaimed": all(s["cache_reclaimed"] for s in summaries),
            },
        }
    gates = {
        "historical_default_tokens_unchanged": all(
            r["historical_tokens_unchanged"] for r in references
        ),
        "reference_reclaimed": reference_reclaimed,
        "cancellation_reclaimed": all(
            c["blocks_before"] > 0 and c["reclaimed"] for c in cancellation
        ),
        **{f"rowwise_{k}": v for k, v in modes["rowwise"]["gates"].items()},
    }
    gates["checkpoint_passed"] = all(gates.values())
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "model": model,
        "configuration": {
            "int8_mode": int8_mode,
            "output_tokens": 32,
            "concurrencies": list(concurrency),
            "historical_report_sha256": hashlib.sha256(
                quality_path.read_bytes()
            ).hexdigest(),
            "historical_report_runtime_sha256": quality["environment"][
                "runtime_source_sha256"
            ],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": {"native_build": build, **source_provenance()},
        "references": references,
        "modes": modes,
        "cancellation": cancellation,
        "gates": gates,
        "limitations": "Same-artifact greedy and bitwise numerical consistency on this hardware/corpus, not Transformers or quantization quality certification. All logit host copies are correctness-only overhead.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument(
        "--int8-mode", choices=["auto", "metal", "reconstruct"], default="auto"
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[2, 8, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.concurrencies or min(args.concurrencies) < 1:
        parser.error("concurrency must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        result = validate(
            args.model, args.quality, args.int8_mode, concurrency=args.concurrencies
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["gates"], indent=2))
    if not result["gates"]["checkpoint_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
