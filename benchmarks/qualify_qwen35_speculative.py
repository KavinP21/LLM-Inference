#!/usr/bin/env python3
"""Qualify opt-in Qwen3.5 hybrid speculation against ordinary greedy decode.

Fault-injection drafts use recorded ordinary outputs only for rollback checks.
Timed trials use the real history n-gram provider, never an oracle draft. The
script writes raw outputs, failures and source identity and exits nonzero when
any correctness check fails. Existing evidence files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from forge_llm.benchmark import source_provenance  # noqa: E402 - direct checkout CLI
from forge_llm.qwen35_engine import Qwen35Engine  # noqa: E402
from forge_llm.qwen35_state import retained_cache_bytes  # noqa: E402
from gpu_guard import exclusive_gpu_workflow  # noqa: E402

WORKLOADS = {
    "copy": "Repeat exactly four times, without introduction: alpha beta gamma delta epsilon zeta eta theta. alpha beta gamma delta epsilon zeta eta theta.",
    "code": "Write a Python function computing arithmetic mean and raising ValueError for empty input. Return code only.",
    "arithmetic": "List the first twenty square numbers separated by commas. Output nothing else.",
}
CONTROL_PROMPT = (
    "Print the integers from 1 through 100 separated by spaces. Output nothing else."
)


class QualificationCancellation(InterruptedError):
    pass


class ForcedPrefixDraft:
    """Inject exactly one wrong proposal per recorded ordinary prefix."""

    def __init__(self, reference, prompt_length, keep, vocab_size, rounds):
        self.reference, self.prompt_length = list(reference), prompt_length
        self.keep, self.vocab_size, self.maximum_rounds = keep, vocab_size, rounds
        self.blocks = []

    def propose(self, history, maximum, check):
        check()
        if len(self.blocks) == self.maximum_rounds:
            return []
        produced = len(history) - self.prompt_length
        values = self.reference[produced : produced + maximum]
        if len(values) != maximum:
            raise RuntimeError("ordinary reference cannot cover the forced block")
        if self.keep < len(values):
            values[self.keep] = (values[self.keep] + 1) % self.vocab_size
        self.blocks.append({"produced": produced, "proposals": list(values)})
        return values

    def finish(self):
        pass


def options(args, *, adaptive=False, mode="block"):
    return {
        "draft_tokens": args.draft_tokens,
        "verification_mode": mode,
        "adaptive": adaptive,
    }


def cleanup_passed(stats):
    return all(
        stats[name] == 0
        for name in (
            "active_requests",
            "kv_reserved_bytes",
            "kv_persistent_bytes",
            "speculative_reserved_bytes",
        )
    )


def active_recurrent_snapshot(engine):
    """Return proof that every recurrent layer has a complete live timeline."""
    if engine._extra_reserved <= 0 or len(engine._cache) != 32:
        return None
    layers = []
    offsets = []
    for index, cache in enumerate(engine._cache):
        if (index + 1) % 4 == 0:
            offsets.append(cache.offset)
            continue
        transaction = getattr(cache, "_speculation", None)
        if transaction is None:
            return None
        records = transaction.get("records", {})
        recurrent, window = records.get(1), records.get(0)
        if (
            not recurrent
            or not window
            or recurrent[0] != "states"
            or window[0] != "window"
        ):
            return None
        states, final = recurrent[1:3]
        length = transaction["length"]
        if (
            states is None
            or tuple(states.shape) != (1, length - 1, 32, 128, 128)
            or tuple(final.shape) != (1, 32, 128, 128)
            or "float32" not in str(states.dtype)
            or "float32" not in str(final.dtype)
        ):
            return None
        layers.append(
            {
                "layer": index,
                "block_length": length,
                "intermediate_shape": list(states.shape),
                "intermediate_bytes": states.nbytes,
                "final_bytes": final.nbytes,
                "conv_source_shape": list(window[1].shape),
                "conv_width": window[2],
            }
        )
    if len(layers) != 24 or len(set(offsets)) != 1:
        return None
    return {
        "recurrent_layers": len(layers),
        "layers": layers,
        "attention_offset": offsets[0],
        "extra_reserved_bytes": engine._extra_reserved,
        "retained_logical_bytes": retained_cache_bytes(engine._cache),
    }


def forced_run(engine, ids, reference, keep, args):
    engine._spec_options = options(args)
    request = engine.submit(ids, args.check_tokens)
    draft = ForcedPrefixDraft(
        reference, len(ids), keep, engine.config.vocab_size, args.forced_rounds
    )
    engine._speculator.draft = draft
    output, capacities = [], []
    try:
        finished = False
        while not finished:
            events = engine.step()
            output.extend(event.token for event in events)
            finished = events[-1].finished
            if engine._active is not None:
                capacities.append(engine._cache[3].keys.shape[2])
    finally:
        if engine._active is not None:
            engine.cancel(request)
        engine.forget(request)
    stats = engine.stats()
    expected = keep * args.forced_rounds
    return {
        "passed": (
            output == reference
            and len(draft.blocks) == args.forced_rounds
            and stats["speculative"]["verification_blocks"] == args.forced_rounds
            and stats["speculative"]["accepted_draft_tokens"] == expected
            and cleanup_passed(stats)
        ),
        "expected_accepted": expected,
        "forced_blocks": draft.blocks,
        "actual": output,
        "reference": reference,
        "stats": stats,
        "kv_capacities": sorted(set(capacities)),
    }


def cancellation_run(engine, ids, reference, args):
    engine._spec_options = options(args)
    request = engine.submit(ids, args.check_tokens)
    engine._speculator.draft = ForcedPrefixDraft(
        reference, len(ids), args.draft_tokens, engine.config.vocab_size, 1
    )
    first = engine.step()
    observed, caught = [], False

    def guard():
        snapshot = active_recurrent_snapshot(engine)
        if snapshot is not None:
            observed.append(snapshot)
            raise QualificationCancellation("cancel inside complete hybrid transaction")

    engine.set_request_guard(guard)
    try:
        finished = False
        while not finished:
            events = engine.step()
            finished = events[-1].finished
    except QualificationCancellation:
        caught = True
    finally:
        if engine._active is not None:
            engine.cancel(request)
        engine.forget(request)
    cleared = engine.stats()
    cache_cleared = not engine._cache
    engine.set_request_guard(lambda: None)
    engine._spec_options = None
    restarted = engine.generate(ids, args.check_tokens)
    return {
        "passed": caught
        and bool(observed)
        and all(
            item["attention_offset"] == len(ids) + args.draft_tokens + 1
            for item in observed
        )
        and cache_cleared
        and cleanup_passed(cleared)
        and restarted == reference,
        "cancellation_observed": caught,
        "cache_owner_cleared": cache_cleared,
        "live_transactions": observed,
        "emitted_before_cancellation": [event.token for event in first],
        "restart_actual": restarted,
        "reference": reference,
        "stats": cleared,
        "note": "Verification statistics count completed rounds; the aborted round is proven by the live 24-layer state records above.",
    }


class Report:
    def __init__(self, output, payload):
        self.output, self.payload = output, payload
        output.parent.mkdir(parents=True, exist_ok=True)
        # Claim once so a rerun cannot alter another experiment's evidence.
        with output.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def save(self):
        self.output.write_text(
            json.dumps(self.payload, indent=2, allow_nan=False) + "\n"
        )

    def row(self, case, values):
        self.payload["rows"].append({"case": case, **values})
        self.save()
        print(
            json.dumps(
                {
                    "case": case,
                    "passed": values["passed"],
                    "speedup": values.get("speedup"),
                }
            ),
            flush=True,
        )


def finalize_provenance(payload):
    """Reject source changes; a newly written evidence file may change dirtiness."""
    after = source_provenance()
    current = {}
    for name in payload["implementation_source_sha256"]:
        try:
            current[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        except OSError:
            current[name] = None
    payload["source_after"] = after
    payload["implementation_source_sha256_after"] = current
    payload["source_unchanged"] = (
        all(
            payload["source"].get(key) == after.get(key)
            for key in ("git_commit", "runtime_source_sha256")
        )
        and current == payload["implementation_source_sha256"]
    )
    payload["passed"] = payload["passed"] and payload["source_unchanged"]


def timing_run(engine, ids, args):
    reference = None
    warmups, trials = [], []
    for warmup in range(args.warmups):
        for enabled in (False, True):
            engine._spec_options = options(args, adaptive=True) if enabled else None
            output = engine.generate(ids, args.output_tokens)
            if reference is None:
                reference = output
            warmups.append(
                {
                    "warmup": warmup,
                    "speculative": enabled,
                    "tokens": output,
                    "canonical_token_parity": output == reference,
                    "stats": engine.stats(),
                }
            )
    for repeat in range(args.repetitions):
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            engine._spec_options = options(args, adaptive=True) if enabled else None
            begin = time.perf_counter()
            output = engine.generate(ids, args.output_tokens)
            elapsed = time.perf_counter() - begin
            trials.append(
                {
                    "repeat": repeat,
                    "speculative": enabled,
                    "seconds": elapsed,
                    "tokens": output,
                    "output_tokens": len(output),
                    "canonical_token_parity": output == reference,
                    "stats": engine.stats(),
                }
            )
    seconds = {
        str(enabled): [
            row["seconds"] for row in trials if row["speculative"] == enabled
        ]
        for enabled in (False, True)
    }
    baseline, speculative = (
        statistics.median(seconds[str(enabled)]) for enabled in (False, True)
    )
    proposed = sum(
        row["stats"]["speculative"]["draft_tokens"]
        for row in trials
        if row["speculative"]
    )
    accepted = sum(
        row["stats"]["speculative"]["accepted_draft_tokens"]
        for row in trials
        if row["speculative"]
    )
    return {
        "passed": all(
            row["canonical_token_parity"] and cleanup_passed(row["stats"])
            for row in warmups + trials
        ),
        "reference": reference,
        "warmups": warmups,
        "trials": trials,
        "baseline_median_seconds": baseline,
        "speculative_median_seconds": speculative,
        "speedup": baseline / speculative,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
    }


def qualify(engine, args, report):
    ids = engine.prepare_chat([{"role": "user", "content": CONTROL_PROMPT}]).input_ids
    engine._spec_options = None
    reference = engine.generate(ids, args.check_tokens)
    report.payload["control_prompt"] = CONTROL_PROMPT
    report.payload["control_prompt_tokens"] = list(ids)
    if len(reference) != args.check_tokens:
        raise RuntimeError(
            "control prompt ended before the fixed rollback-check budget"
        )
    for keep in range(args.draft_tokens + 1):
        report.row(
            f"repeated-prefix-{keep}", forced_run(engine, ids, reference, keep, args)
        )
    engine._spec_options = options(args, mode="sequential")
    fallback = engine.generate(ids, args.check_tokens)
    report.row(
        "canonical-fallback",
        {
            "passed": fallback == reference
            and cleanup_passed(engine.stats())
            and engine.stats()["speculative"]["verification_blocks"] == 0
            and engine.stats()["speculative"]["draft_tokens"] == 0
            and engine.stats()["decode_calls"] == len(fallback) - 1,
            "actual": fallback,
            "reference": reference,
            "stats": engine.stats(),
        },
    )
    report.row(
        "active-transaction-cancel-and-restart",
        cancellation_run(engine, ids, reference, args),
    )
    fill = engine.tokenizer.encode(" padding", add_special_tokens=False)[0]
    for length in args.boundary_prompts:
        if len(ids) >= length:
            raise ValueError("boundary prompt is shorter than the control template")
        synthetic = [fill] * (length - len(ids)) + list(ids)
        engine._spec_options = None
        baseline = engine.generate(synthetic, args.check_tokens)
        if len(baseline) != args.check_tokens:
            raise RuntimeError(
                "synthetic boundary prompt ended before the check budget"
            )
        values = forced_run(
            engine, synthetic, baseline, min(1, args.draft_tokens), args
        )
        values["prompt_tokens"] = length
        values["synthetic_padding_token"] = fill
        values["uneven_kv_capacity_observed"] = any(
            capacity % 256 for capacity in values["kv_capacities"]
        )
        values["passed"] = values["passed"] and values["uneven_kv_capacity_observed"]
        report.row(f"allocation-boundary-{length}", values)
    for name, prompt in WORKLOADS.items():
        ids = engine.prepare_chat([{"role": "user", "content": prompt}]).input_ids
        values = timing_run(engine, ids, args)
        values.update({"prompt": prompt, "prompt_tokens": list(ids)})
        report.row("timing-" + name, values)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model",
        type=Path,
        required=True,
        help="existing full local MLX-VLM checkpoint; never downloads",
    )
    result.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSON evidence path; existing paths are refused",
    )
    result.add_argument("--max-model-length", type=int, default=32768)
    result.add_argument("--state-budget-bytes", type=int, default=2 << 30)
    result.add_argument("--prefill-chunk-size", type=int, default=256)
    result.add_argument("--draft-tokens", type=int, default=4, choices=range(1, 9))
    result.add_argument("--check-tokens", type=int, default=24)
    result.add_argument("--forced-rounds", type=int, default=3)
    result.add_argument("--boundary-prompts", type=int, nargs="+", default=[255, 511])
    result.add_argument("--output-tokens", type=int, default=64)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repetitions", type=int, default=3)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.warmups < 1
        or args.repetitions < 3
        or args.forced_rounds < 2
        or args.check_tokens
        < 1 + (args.draft_tokens + 1) * args.forced_rounds + args.draft_tokens
        or args.output_tokens < 1
        or any(length < 1 for length in args.boundary_prompts)
    ):
        raise ValueError(
            "need positive budgets, at least one warmup, three repetitions and two fully covered forced blocks"
        )
    source = source_provenance()
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "benchmarks/qualify_qwen35_speculative.py",
            "python/forge_llm/qwen35_engine.py",
            "python/forge_llm/qwen35_speculative.py",
            "python/forge_llm/qwen35_state.py",
            "python/forge_llm/qwen35_checkpoint.py",
        )
    }
    payload = {
        "schema": "forge_qwen35_speculative_qualification_v2",
        "source": source,
        "implementation_source_sha256": hashes,
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
        "method": "Ordinary greedy engine baseline, same loaded affine4-bit weights and fresh mixed caches. Oracle outputs are used only for forced rollback fault injection. Timings use real n-grams, separate untimed warmups per mode/workload, and alternating order; include prefill/decode/request cleanup, exclude model loading/tokenization/media preparation. Synthetic padding tests allocation only. Reserved history bytes are not measured Metal memory. No universal parity, original-FP16, task-quality or speed claim.",
    }
    report = Report(args.output, payload)
    try:
        with exclusive_gpu_workflow():
            with closing(
                Qwen35Engine(
                    args.model,
                    max_model_length=args.max_model_length,
                    kv_cache_bytes=args.state_budget_bytes,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            ) as engine:
                payload["checkpoint_identity"], payload["engine_metadata"] = (
                    engine.checkpoint_identity,
                    engine.build_info(),
                )
                qualify(engine, args, report)
        payload["passed"] = bool(payload["rows"]) and all(
            row["passed"] for row in payload["rows"]
        )
    except BaseException as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)[:4096]}
        raise
    finally:
        finalize_provenance(payload)
        report.save()
    if not payload["passed"]:
        raise SystemExit(
            "Qwen3.5 speculative qualification failed; raw evidence preserved"
        )


if __name__ == "__main__":
    main()
