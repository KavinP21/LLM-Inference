"""Readback checks for cached checkpoint evidence, independent of saved gates."""

from __future__ import annotations

from .validate_int8 import quality_gates


def reclaimed(stats):
    return (
        stats["kv_cache"]["allocated_blocks"] == 0
        and stats["kv_cache"]["reserved_blocks"] == 0
        and stats["kv_device_bytes"] == 0
    )


def quality(report):
    cases, summary = report["cases"], report["summary"]
    count = summary["output_tokens_per_case"]
    if not cases or len(cases) != summary["total_cases"] or count != 32:
        raise ValueError("incomplete cached quality corpus")
    rows = []
    for case in cases:
        reference, actual, dense = [
            case[k] for k in ("fp16_tokens", "int8_tokens", "dequantize_tokens")
        ]
        if any(len(t) != count for t in (reference, actual, dense)):
            raise ValueError("truncated greedy token sequences")
        prefix = next(
            (i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), count
        )
        observations = case["teacher_forced_logits"]
        if (
            [o["generated_position"] for o in observations] != list(range(count))
            or case["exact_greedy_parity"] != (reference == actual)
            or case["metal_dequantize_greedy_parity"] != (actual == dense)
            or case["matching_prefix"] != prefix
        ):
            raise ValueError("saved quality case disagrees with raw observations")
        rows.extend(observations)
    values = {
        "minimum_layer_cosine": min(o["cosine_vs_fp16"] for o in report["layers"]),
        "minimum_logit_cosine": min(o["int8_vs_fp16"]["cosine"] for o in rows),
        "teacher_forced_top1_agreement": sum(
            o["int8_vs_fp16"]["argmax_equal"] for o in rows
        )
        / len(rows),
        "exact_greedy_cases": sum(c["fp16_tokens"] == c["int8_tokens"] for c in cases),
        "metal_dequantize_exact_cases": sum(
            c["int8_tokens"] == c["dequantize_tokens"] for c in cases
        ),
        "minimum_metal_dequantize_logit_cosine": min(
            o["metal_vs_dequantize"]["cosine"] for o in rows
        ),
        "teacher_forced_logit_rows": len(rows),
        "bitwise_kernel_logit_rows": sum(o["bitwise_kernel_logits"] for o in rows),
    }
    if any(summary[k] != v for k, v in values.items()):
        raise ValueError("saved quality summary disagrees with raw observations")
    cleanup = report["cache_reclaimed"] and all(
        c["private_cache_reclaimed"] for c in cases
    )
    gates = quality_gates(summary, cleanup, report["configuration"]["cosine_gate"])
    if gates != report["gates"]:
        raise ValueError("saved quality gates disagree with recomputed gates")
    return gates


def workload(work, references):
    requests, summary = work["requests"], work["summary"]
    if not requests or len(requests) != len(references):
        raise ValueError("incomplete batch workload")
    if sorted(r["case_index"] for r in requests) != list(range(len(references))):
        raise ValueError("duplicate or missing batch request")
    tokens, rows, total, events = 0, 0, 0, True
    for request in requests:
        ref, count = references[request["case_index"]], request["output_budget"]
        exact = request["tokens"] == ref["tokens"][:count]
        if len(request["logits"]) != count or not 1 <= count <= 32:
            raise ValueError("truncated batch logits")
        equal = sum(
            a["sha256"] == b["sha256"]
            for a, b in zip(request["logits"], ref["logits"][:count], strict=True)
        )
        if (
            request["exact_tokens"] != exact
            or request["exact_logit_rows"] != equal
            or request["total_logit_rows"] != count
        ):
            raise ValueError("batch request flags disagree with raw observations")
        tokens += exact
        rows += equal
        total += count
        events &= request["events_exact"]
    cleanup = reclaimed(work["engine_stats"])
    values = {
        "cases": len(requests),
        "exact_token_cases": tokens,
        "exact_logit_rows": rows,
        "total_logit_rows": total,
        "events_exact": events,
        "cache_reclaimed": cleanup,
    }
    if any(summary[k] != v for k, v in values.items()):
        raise ValueError("batch summary disagrees with raw observations")
    return {
        "exact_tokens": tokens == len(requests),
        "exact_logits": rows == total,
        "events_exact": events,
        "cache_reclaimed": cleanup,
    }


def batching(report):
    references = report["references"]
    rowwise = report["modes"]["rowwise"]
    works = rowwise["workloads"]
    configs = [w["configuration"] for w in works]
    if [(c["max_num_sequences"], c["staggered"]) for c in configs] != [
        (2, False),
        (8, False),
        (16, False),
        (8, True),
    ]:
        raise ValueError("incomplete arrival matrix")
    verified = [workload(w, references) for w in works]
    mode_gates = {k: all(w[k] for w in verified) for k in verified[0]}
    if mode_gates != rowwise["gates"]:
        raise ValueError("rowwise gates disagree with raw observations")
    gates = {
        "historical_default_tokens_unchanged": all(
            r["historical_tokens_unchanged"] for r in references
        ),
        "reference_reclaimed": report["gates"]["reference_reclaimed"],
        "cancellation_reclaimed": all(
            c["blocks_before"] > 0 and c["reclaimed"] for c in report["cancellation"]
        ),
        **{f"rowwise_{k}": v for k, v in mode_gates.items()},
    }
    gates["checkpoint_passed"] = all(gates.values())
    if gates != report["gates"]:
        raise ValueError("batch gates disagree with raw observations")
    return gates


def boundary(report):
    if report["configuration"]["prompt_lengths"] != [
        15,
        16,
        17,
        31,
        32,
        33,
        128,
        512,
        513,
    ]:
        raise ValueError("incomplete boundary workload")
    gates = {
        "independent_reclaimed": report["gates"]["independent_reclaimed"],
        **workload(report["batched"], report["references"]),
    }
    gates["checkpoint_passed"] = all(gates.values())
    if gates != report["gates"]:
        raise ValueError("boundary gates disagree with raw observations")
    return gates


def long_context(report):
    observations = report["observations"]
    return (
        len(observations) == 1
        and report["all_reclaimed"]
        and all(
            o["peak_pool_blocks"] == o["peak_materialized_blocks"] == 2048
            and o["decode_iterations"] == 1
            and o["prompt_tokens"] == 32766
            and o["output_tokens"] == 2
            for o in observations
        )
    )
