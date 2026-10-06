"""Frozen FP16 prefix/COW validation on Qwen and Gemma; timing only after parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import datetime
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from forge_llm.benchmark import model_provenance, percentile, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.model_file import ModelFile
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_refined_checkpoint import check_bindings, check_sources
from run_second_order_checkpoint import FAMILIES, save

CONFIG = {
    "families": ["qwen", "gemma"],
    "decode_mode": "rowwise",
    "output_tokens": 32,
    "chunk_sizes": [127, 512],
    "prompt_lengths": [
        15,
        16,
        17,
        31,
        32,
        33,
        127,
        128,
        129,
        511,
        512,
        513,
        1023,
        1024,
        1025,
    ],
    "text_corpus": "benchmarks/quality-prompts.json",
    "text_cases": 25,
    "long_prompt": 32766,
    "long_outputs": 2,
    "concurrency": 8,
    "benchmark_prompt_lengths": [128, 1024],
    "benchmark_concurrencies": [1, 8],
    "benchmark_modes": ["disabled", "cold", "primed"],
    "warmups": 5,
    "repetitions": 3,
}
TOOLS = [
    Path(__file__),
    Path("benchmarks/benchmark_prefix.py"),
    Path("benchmarks/gpu_guard.py"),
    Path(".github/workflows/prefix-cache.yml"),
    Path("docs/prefix-cache.md"),
]


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    from transformers import AutoTokenizer

    if root.exists():
        raise FileExistsError("fresh prefix result directory required")
    prior = Path("benchmarks/results/qwen-validation-m3-max-2026-10-02")
    old = read(prior / "contract.json")
    frozen = read(prior / "frozen.json")
    bindings = {
        **old["prior_files_sha256"],
        **old["files_sha256"],
        **frozen["files_sha256"],
    }
    bindings.update({str(p): file_sha256(p) for p in prior.rglob("*") if p.is_file()})
    check_bindings(bindings)
    # Prior live audits were run BEFORE this milestone changed runtime sources.
    # Their captured source/results remain immutable; do not forge a new live
    # runtime identity into a historical checkpoint.
    files = {str(p): file_sha256(p) for p in [*TOOLS, Path(CONFIG["text_corpus"])]}
    models = {}
    for family, (stem, tokenizer) in FAMILIES.items():
        model = Path("models") / f"{stem}.engine"
        with ModelFile(model) as artifact:
            vocab = artifact.config.vocab_size
        encoder = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        models[family] = {
            "path": str(model),
            "tokenizer": tokenizer,
            "provenance": model_provenance(model),
            "vocab_size": vocab,
            "text_input_ids": [
                encoder(p, add_special_tokens=True).input_ids
                for p in read(CONFIG["text_corpus"])
            ],
        }
        files[str(model)] = file_sha256(model)
        files[str(model.with_suffix(".engine.json"))] = file_sha256(
            model.with_suffix(".engine.json")
        )
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    paths = {
        p.resolve()
        for p in [*TOOLS, Path(CONFIG["text_corpus"]), Path("pyproject.toml")]
    }
    for directory in ("python", "include", "src", "tests"):
        paths.update(
            p
            for p in (repo / directory).rglob("*")
            if p.suffix in {".py", ".h", ".hpp", ".cpp", ".cu", ".cuh"}
        )
    members = {}
    with zipfile.ZipFile(root / "sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            name, data = str(path.relative_to(repo)), path.read_bytes()
            archive.writestr(name, data)
            members[name] = hashlib.sha256(data).hexdigest()
    save(
        root / "contract.json",
        {
            "schema_version": 1,
            "registered_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "configuration": CONFIG,
            "models": models,
            "environment": source_provenance(),
            "files_sha256": files,
            "prior_files_sha256": bindings,
            "source_archive": {
                "sha256": file_sha256(root / "sources.zip"),
                "files_sha256": members,
            },
            "stop_rule": "All registered bitwise/token/resource checks must pass before any timing. Freeze source, models, workloads and gates; no post-result tuning. Preserve negative or partial evidence; never overwrite.",
            "scope": "MLX FP16/rowwise only. Model/execution identity and namespaces are engine-local; reference is same-artifact uncached Forge, not new Transformers semantic certification. Prefix default disabled. Old INT8 rejections and CUDA validation boundary unchanged.",
        },
    )


def contract(root):
    value = read(root / "contract.json")
    if (
        value["configuration"] != CONFIG
        or value["environment"]["runtime_source_sha256"]
        != source_provenance()["runtime_source_sha256"]
    ):
        raise ValueError("prefix registration/runtime changed")
    check_bindings(value["files_sha256"])
    check_bindings(value["prior_files_sha256"])
    check_sources(root, value["source_archive"])
    return value


def fingerprint(row):
    row = np.asarray(row)
    if row.ndim != 1 or row.dtype.kind != "f" or not np.isfinite(row).all():
        raise ValueError("invalid prefix logit observation")
    return {
        "dtype": row.dtype.str,
        "shape": list(row.shape),
        "sha256": hashlib.sha256(row.tobytes()).hexdigest(),
        "argmax": int(np.argmax(row)),
    }


@contextmanager
def observe(engine):
    rows = {}
    prefill, decode, forward = (
        engine._prefill_chunk,
        engine.model.decode_paged_batch,
        engine.model.forward_paged_chunk,
    )
    work = {"prefill_tokens": 0, "decode_rows": 0}

    def chunk(request_id):
        result = prefill(request_id)
        if result[1] == len(engine.scheduler.request(request_id).prompt):
            rows.setdefault(request_id, []).append(fingerprint(np.array(result[0][-1])))
        return result

    def batch(*a, **k):
        ids = tuple(engine.scheduler.running)
        logits = decode(*a, **k)
        for request_id, row in zip(ids, logits, strict=True):
            rows.setdefault(request_id, []).append(fingerprint(np.array(row)))
        work["decode_rows"] += len(ids)
        return logits

    def prompt(*a, **k):
        work["prefill_tokens"] += len(a[0])
        return forward(*a, **k)

    (
        engine._prefill_chunk,
        engine.model.decode_paged_batch,
        engine.model.forward_paged_chunk,
    ) = chunk, batch, prompt
    try:
        yield rows, work
    finally:
        (
            engine._prefill_chunk,
            engine.model.decode_paged_batch,
            engine.model.forward_paged_chunk,
        ) = prefill, decode, forward


def run_request(engine, prompt, count, namespace=""):
    with observe(engine) as (rows, work):
        request = engine.submit(prompt, count, cache_namespace=namespace)
        for _ in range(4096):
            engine.step()
            if engine.scheduler.request(request).state.value == "completed":
                break
        else:
            raise RuntimeError("prefix request failed to drain")
    tokens = engine.scheduler.request(request).output
    if len(tokens) != count or len(rows[request]) != count:
        raise ValueError("truncated prefix continuation")
    return {
        "tokens": tokens,
        "logits": rows[request],
        "work": work,
        "stats": engine.stats(),
    }


def matches(a, b):
    return a["tokens"] == b["tokens"] and a["logits"] == b["logits"]


def continuation_readback(value, count, vocab, *, work=True):
    if len(value["tokens"]) != count or len(value["logits"]) != count:
        raise ValueError("truncated prefix continuation")
    for row, token in zip(value["logits"], value["tokens"], strict=True):
        if (
            type(token) is not int
            or not 0 <= token < vocab
            or row["argmax"] != token
            or type(row["argmax"]) is not int
            or row["shape"] != [vocab]
            or row["dtype"] != "<f4"
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
        ):
            raise ValueError("inconsistent prefix logit identity/token")
    if work and (
        value["work"]["decode_rows"] != count - 1
        or type(value["work"]["prefill_tokens"]) is not int
        or value["work"]["prefill_tokens"] < 0
    ):
        raise ValueError("invalid executed prefix work")


def cleanup_records(report, expected):
    stats = report["cleanup_stats"]
    if set(stats) != set(expected) or type(report["cleanup"]) is not bool:
        raise ValueError("incomplete prefix cleanup coverage")
    actual = all(cleanup_readback(s) for s in stats.values())
    if actual != report["cleanup"]:
        raise ValueError("prefix cleanup flag disagrees with raw accounting")
    return actual


def quality_readback(report, item, chunk):
    observations = report["observations"]
    vocab = item["vocab_size"]
    expected = [
        [(17 + i * 104729 + index * 113) % vocab for i in range(length)]
        for index, length in enumerate(CONFIG["prompt_lengths"])
    ] + item["text_input_ids"]
    if (
        len(expected) != 40
        or len(observations) != len(expected)
        or [o["case_index"] for o in observations] != list(range(len(expected)))
        or report["model"] != item["provenance"]
        or report["chunk_size"] != chunk
    ):
        raise ValueError("incomplete prefix quality coverage")
    for o, prompt in zip(observations, expected, strict=True):
        branch = prompt[:-1] + [(prompt[-1] + 1) % vocab]
        if o["prompt"] != prompt or o["branch_prompt"] != branch:
            raise ValueError("prefix workload drift")
        for mode in ("reference", "cold", "warm", "branch_reference", "branch"):
            continuation_readback(o[mode], 32, vocab)
        if any(
            o[mode]["work"]["prefill_tokens"] != len(prompt)
            for mode in ("reference", "cold", "branch_reference")
        ):
            raise ValueError("missing uncached prefill work")
    passed = cleanup_records(report, ("baseline", "cached")) and all(
        matches(o["reference"], o["cold"])
        and matches(o["reference"], o["warm"])
        and matches(o["branch_reference"], o["branch"])
        and o["warm"]["work"]["prefill_tokens"] == 0
        for o in observations
    )
    if type(report["passed"]) is not bool or report["passed"] != passed:
        raise ValueError("prefix quality flag disagrees with raw observations")
    return passed


def stress_readback(report, item):
    vocab = item["vocab_size"]
    root = [(i * 104729 + 37) % vocab for i in range(513)]
    prompts = [root if i % 2 == 0 else root[:-1] + [(i + 2) % vocab] for i in range(8)]
    if (
        report["model"] != item["provenance"]
        or report["prompts"] != prompts
        or len(report["references"]) != 8
        or len(report["results"]) != 8
    ):
        raise ValueError("incomplete shared batch coverage")
    for a, b in zip(report["references"], report["results"], strict=True):
        continuation_readback(a, 32, vocab)
        continuation_readback(b, 32, vocab, work=False)
    isolated = report["namespace_misses_after"] == report["namespace_misses_before"] + 1
    if type(report["isolated"]) is not bool or isolated != report["isolated"]:
        raise ValueError("namespace flag disagrees with raw counters")
    if report["work"]["decode_rows"] != 8 * 31:
        raise ValueError("incomplete shared decode work")
    passed = (
        cleanup_records(report, ("baseline", "cached", "pressure"))
        and all(
            matches(a, b)
            for a, b in zip(report["references"], report["results"], strict=True)
        )
        and isolated
        and report["cow_copies"] > 0
        and report["pressure_stats"]["prefix_cache"]["evictions"] > 0
    )
    if type(report["passed"]) is not bool or report["passed"] != passed:
        raise ValueError("stress flag disagrees with raw observations")
    return passed


def long_readback(report, item):
    if (
        report["model"] != item["provenance"]
        or report["prompt_tokens"] != CONFIG["long_prompt"]
    ):
        raise ValueError("incomplete 32K prefix coverage")
    for mode in ("reference", "cold", "warm"):
        continuation_readback(report[mode], 2, item["vocab_size"])
    if any(report[k]["work"]["prefill_tokens"] != 32766 for k in ("reference", "cold")):
        raise ValueError("missing 32K uncached work")
    passed = (
        cleanup_records(report, ("baseline", "cached"))
        and matches(report["reference"], report["cold"])
        and matches(report["reference"], report["warm"])
        and report["warm"]["work"] == {"prefill_tokens": 0, "decode_rows": 1}
    )
    if type(report["passed"]) is not bool or report["passed"] != passed:
        raise ValueError("32K flag disagrees with raw observations")
    return passed


def reclaimed(engine):
    return cleanup_readback(engine.stats())


def cleanup_readback(stats):
    return (
        stats["kv_cache"]["allocated_blocks"]
        == stats["kv_cache"]["reserved_blocks"]
        == stats["kv_device_bytes"]
        == 0
        and stats["scheduler"]["running"] == stats["scheduler"]["waiting"] == 0
    )


def quality_model(item, chunk):
    prompts = item["text_input_ids"]
    options = dict(
        max_model_length=2048,
        kv_cache_bytes=512 << 20,
        prefill_chunk_size=chunk,
        decode_mode="rowwise",
    )
    observations = []
    with (
        MlxEngine(item["path"], **options) as baseline,
        MlxEngine(item["path"], prefix_cache_bytes=64 << 20, **options) as cached,
    ):
        synthetic = [
            [
                (17 + i * 104729 + index * 113) % baseline.model.config.vocab_size
                for i in range(length)
            ]
            for index, length in enumerate(CONFIG["prompt_lengths"])
        ]
        for index, prompt in enumerate(synthetic + prompts):
            cached.clear_prefix_cache()
            reference = run_request(baseline, prompt, 32)
            cold = run_request(cached, prompt, 32)
            warm = run_request(cached, prompt, 32)
            branch = prompt[:-1] + [(prompt[-1] + 1) % baseline.model.config.vocab_size]
            branch_reference = run_request(baseline, branch, 32)
            branch_cached = run_request(cached, branch, 32)
            observations.append(
                {
                    "case_index": index,
                    "prompt": prompt,
                    "branch_prompt": branch,
                    "reference": reference,
                    "cold": cold,
                    "warm": warm,
                    "branch_reference": branch_reference,
                    "branch": branch_cached,
                }
            )
            if (index + 1) % 10 == 0:
                print(
                    f"{item['tokenizer']} chunk={chunk}: {index + 1}/40 cases",
                    flush=True,
                )
        cached.clear_prefix_cache()
        cleanup = reclaimed(baseline) and reclaimed(cached)
        cleanup_stats = {"baseline": baseline.stats(), "cached": cached.stats()}
    passed = cleanup and all(
        matches(o["reference"], o["cold"])
        and matches(o["reference"], o["warm"])
        and matches(o["branch_reference"], o["branch"])
        and o["warm"]["work"]["prefill_tokens"] == 0
        for o in observations
    )
    return {
        "model": item["provenance"],
        "chunk_size": chunk,
        "observations": observations,
        "cleanup": cleanup,
        "cleanup_stats": cleanup_stats,
        "passed": passed,
    }


def stress_model(item):
    options = dict(
        max_model_length=2048,
        max_num_sequences=8,
        kv_cache_bytes=512 << 20,
        prefill_chunk_size=127,
        decode_mode="rowwise",
    )
    with (
        MlxEngine(item["path"], **options) as baseline,
        MlxEngine(item["path"], prefix_cache_bytes=64 << 20, **options) as cached,
    ):
        vocab = cached.model.config.vocab_size
        root = [(i * 104729 + 37) % vocab for i in range(513)]
        prompts = [
            root if i % 2 == 0 else root[:-1] + [(i + 2) % vocab] for i in range(8)
        ]
        references = [run_request(baseline, p, 32) for p in prompts]
        cached.generate(root, 1)
        cancelled = cached.submit(root, 32)
        cached.cancel(cancelled)  # Waiting hit owns references but emits nothing.
        with observe(cached) as (rows, work):
            ids = [cached.submit(p, 32) for p in prompts]
            cached.clear_prefix_cache()  # Active/pending hits survive pin eviction.
            for _ in range(4096):
                cached.step()
                if all(
                    cached.scheduler.request(r).state.value == "completed" for r in ids
                ):
                    break
            else:
                raise RuntimeError("shared batch failed to drain")
        results = [
            {"tokens": cached.scheduler.request(r).output, "logits": rows[r]}
            for r in ids
        ]
        # Isolation is measured separately, before clearing its entries.
        cached.clear_prefix_cache()
        cached.generate(root, 1, cache_namespace="tenant A")
        misses = cached.stats()["prefix_cache"]["misses"]
        cached.generate(root, 1, cache_namespace="tenant B")
        isolated = cached.stats()["prefix_cache"]["misses"] == misses + 1
        cached.clear_prefix_cache()
        cleanup = reclaimed(baseline) and reclaimed(cached)
        cleanup_stats = {"baseline": baseline.stats(), "cached": cached.stats()}
        cow = cached.cache_pool.cow_copies
        page_bytes = cached.bytes_per_block
    # LRU pressure with a small budget exercises actual device reclamation.
    with MlxEngine(
        item["path"],
        prefix_cache_bytes=4 * page_bytes,
        prefix_cache_max_entries=2,
        **options,
    ) as pressure:
        for i in range(5):
            pressure.generate([(j + 100 * i) % vocab for j in range(33)], 2)
        pressure_stats = pressure.stats()
        pressure.clear_prefix_cache()
        pressure_clean = reclaimed(pressure)
        cleanup_stats["pressure"] = pressure.stats()
    return {
        "model": item["provenance"],
        "prompts": prompts,
        "references": references,
        "results": results,
        "work": work,
        "isolated": isolated,
        "namespace_misses_before": misses,
        "namespace_misses_after": cached.stats()["prefix_cache"]["misses"],
        "cow_copies": cow,
        "pressure_stats": pressure_stats,
        "cleanup": cleanup and pressure_clean,
        "cleanup_stats": cleanup_stats,
        "passed": all(matches(a, b) for a, b in zip(references, results, strict=True))
        and isolated
        and cow > 0
        and pressure_stats["prefix_cache"]["evictions"] > 0
        and cleanup
        and pressure_clean,
    }


def long_model(item):
    options = dict(
        max_model_length=32768,
        kv_cache_bytes=2 << 30,
        prefill_chunk_size=512,
        decode_mode="rowwise",
    )
    with (
        MlxEngine(item["path"], **options) as baseline,
        MlxEngine(item["path"], prefix_cache_bytes=1 << 30, **options) as cached,
    ):
        prompt = [
            (17 + i * 104729) % baseline.model.config.vocab_size for i in range(32766)
        ]
        reference, cold, warm = (
            run_request(baseline, prompt, 2),
            run_request(cached, prompt, 2),
            run_request(cached, prompt, 2),
        )
        cached.clear_prefix_cache()
        cleanup = reclaimed(baseline) and reclaimed(cached)
        cleanup_stats = {"baseline": baseline.stats(), "cached": cached.stats()}
    return {
        "model": item["provenance"],
        "prompt_tokens": 32766,
        "reference": reference,
        "cold": cold,
        "warm": warm,
        "cleanup": cleanup,
        "cleanup_stats": cleanup_stats,
        "passed": cleanup
        and matches(reference, cold)
        and matches(reference, warm)
        and warm["work"] == {"prefill_tokens": 0, "decode_rows": 1},
    }


def validate(root):
    frozen = contract(root)
    directory = root / "validation"
    directory.mkdir()
    gates = {}
    for family, item in frozen["models"].items():
        for chunk in CONFIG["chunk_sizes"]:
            key = f"{family}-chunk{chunk}"
            report = quality_model(item, chunk)
            save(directory / f"{key}.json", report)
            gates[key] = report["passed"]
            print(f"{key}: {report['passed']}", flush=True)
        for name, check in (("stress", stress_model), ("32k", long_model)):
            report = check(item)
            save(directory / f"{family}-{name}.json", report)
            gates[f"{family}-{name}"] = report["passed"]
            print(f"{family}-{name}: {report['passed']}", flush=True)
    contract(root)
    save(
        root / "validation-checkpoint.json",
        {
            "contract_sha256": file_sha256(root / "contract.json"),
            "gates": gates,
            "passed": all(gates.values()),
            "files_sha256": {str(p): file_sha256(p) for p in directory.glob("*.json")},
        },
    )


def validation_readback(root):
    frozen = contract(root)
    checkpoint = read(root / "validation-checkpoint.json")
    expected_files = {
        str(root / "validation" / f"{family}-{name}.json")
        for family in CONFIG["families"]
        for name in ("chunk127", "chunk512", "stress", "32k")
    }
    if (
        set(checkpoint["files_sha256"]) != expected_files
        or {str(p) for p in (root / "validation").iterdir()} != expected_files
    ):
        raise ValueError("incomplete validation file coverage")
    check_bindings(checkpoint["files_sha256"])
    if checkpoint["contract_sha256"] != file_sha256(root / "contract.json"):
        raise ValueError("prefix registration binding changed")
    gates = {}
    for family, item in frozen["models"].items():
        for chunk in CONFIG["chunk_sizes"]:
            key = f"{family}-chunk{chunk}"
            report = read(root / "validation" / f"{key}.json")
            gates[key] = quality_readback(report, item, chunk)
        stress = read(root / "validation" / f"{family}-stress.json")
        gates[f"{family}-stress"] = stress_readback(stress, item)
        long = read(root / "validation" / f"{family}-32k.json")
        gates[f"{family}-32k"] = long_readback(long, item)
    if (
        type(checkpoint["passed"]) is not bool
        or any(type(v) is not bool for v in checkpoint["gates"].values())
        or checkpoint["gates"] != gates
        or checkpoint["passed"] != all(gates.values())
    ):
        raise ValueError("prefix checkpoint disagrees with raw gates")
    return all(gates.values())


def benchmark(run, root):
    if not validation_readback(root):
        raise ValueError("prefix parity failure forbids timing")
    frozen = contract(root)
    directory = root / "matrix"
    directory.mkdir()
    for family, item in frozen["models"].items():
        for index, (length, concurrency) in enumerate(
            (p, c) for p in (128, 1024) for c in (1, 8)
        ):
            modes = CONFIG["benchmark_modes"]
            for mode in modes[index % 3 :] + modes[: index % 3]:
                print(
                    f"Timing {family} {mode} prompt={length} concurrency={concurrency}",
                    flush=True,
                )
                run(
                    [
                        sys.executable,
                        "benchmarks/benchmark_prefix.py",
                        "--model",
                        item["path"],
                        "--mode",
                        mode,
                        "--prompt-length",
                        str(length),
                        "--concurrency",
                        str(concurrency),
                        "--output",
                        str(
                            directory / f"{family}-{mode}-p{length}-c{concurrency}.json"
                        ),
                    ],
                    check=True,
                )
    contract(root)


def benchmark_readback(report, item, mode, length, concurrency, environment):
    config = report["configuration"]
    expected = {
        "mode": mode,
        "prompt_length": length,
        "concurrency": concurrency,
        "output_tokens": 32,
        "warmups": 5,
        "repetitions": 3,
        "decode_mode": "rowwise",
        "prefill_chunk_size": 512,
        "workload": [(17 + i * 104729) % item["vocab_size"] for i in range(length)],
        "prefix_cache_bytes": 0 if mode == "disabled" else 64 << 20,
        "priming_outside_timing": mode == "primed",
    }
    if (
        config != expected
        or report["model"] != item["provenance"]
        or any(report["environment"][k] != v for k, v in environment.items())
        or len(report["requests"]) != concurrency * 3
        or [t["trial"] for t in report["trials"]] != [0, 1, 2]
    ):
        raise ValueError("invalid prefix benchmark provenance/workload")
    for trial in report["trials"]:
        rows = [r for r in report["requests"] if r["trial"] == trial["trial"]]
        if (
            len(rows) != concurrency
            or len({r["request_id"] for r in rows}) != concurrency
            or not math.isfinite(trial["seconds"])
            or trial["seconds"] <= 0
            or not math.isclose(
                trial["wall_ms"], 1000 * trial["seconds"], rel_tol=1e-12
            )
        ):
            raise ValueError("invalid raw prefix trial time/count")
        for row in rows:
            if (
                row["prompt_tokens"] != length
                or row["output_tokens"] != 32
                or not all(
                    math.isfinite(row[k]) for k in ("ttft_ms", "tpot_ms", "e2e_ms")
                )
                or not 0 <= row["ttft_ms"] <= row["e2e_ms"] <= trial["wall_ms"]
                or not math.isclose(
                    row["tpot_ms"],
                    (row["e2e_ms"] - row["ttft_ms"]) / 31,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("invalid raw prefix latency/TPOT")
        if (
            trial["after"]["scheduler"]["completed"]
            - trial["before"]["scheduler"]["completed"]
            != concurrency
        ):
            raise ValueError("prefix benchmark dropped admitted requests")
        if mode != "disabled":
            before, after = (
                trial["before"]["prefix_cache"],
                trial["after"]["prefix_cache"],
            )
            if after["reused_tokens"] - before["reused_tokens"] != (
                length * concurrency if mode == "primed" else 0
            ) or after["hits"] - before["hits"] != (
                concurrency if mode == "primed" else 0
            ):
                raise ValueError("prefix timing mode does not match actual cache work")
        if (
            trial["after"]["kv_cache"]["reserved_blocks"]
            or trial["after"]["scheduler"]["running"]
            or trial["after"]["scheduler"]["waiting"]
        ):
            raise ValueError("prefix benchmark left admitted work alive")
    if not cleanup_readback(report["cleared_stats"]):
        raise ValueError("prefix benchmark failed cleanup")
    requests = report["requests"]
    aggregate = {
        "generated_tokens_per_second": concurrency
        * 3
        * 32
        / sum(t["seconds"] for t in report["trials"]),
        **{
            name: {
                f"p{int(q * 100)}": percentile([r[name] for r in requests], q)
                for q in (0.5, 0.95, 0.99)
            }
            for name in ("ttft_ms", "tpot_ms", "e2e_ms")
        },
    }
    if aggregate != report["aggregate"]:
        raise ValueError("prefix aggregate disagrees with raw measurements")
    return aggregate


def audit(root):
    frozen = contract(root)
    passed = validation_readback(root)
    matrix = root / "matrix"
    if not passed and matrix.exists():
        raise ValueError("timing after rejected prefix validation")
    results = []
    if passed:
        expected = {
            f"{f}-{m}-p{p}-c{c}.json"
            for f in CONFIG["families"]
            for m in CONFIG["benchmark_modes"]
            for p in (128, 1024)
            for c in (1, 8)
        }
        if not matrix.exists() or {p.name for p in matrix.iterdir()} != expected:
            raise ValueError("incomplete controlled prefix matrix")
        for path in sorted(matrix.iterdir()):
            report = read(path)
            family, mode, prompt, concurrency = path.stem.split("-")
            aggregate = benchmark_readback(
                report,
                frozen["models"][family],
                mode,
                int(prompt[1:]),
                int(concurrency[1:]),
                frozen["environment"],
            )
            results.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "aggregate": aggregate,
                }
            )
    return {
        "schema_version": 1,
        "environment": frozen["environment"],
        "contract_sha256": file_sha256(root / "contract.json"),
        "validation_passed": passed,
        "milestone_passed": passed and len(results) == 24,
        "matrix": results,
        "matrix_results": len(results),
        "prior_files_unchanged": len(frozen["prior_files_sha256"]),
        "captured_source_members": len(frozen["source_archive"]["files_sha256"]),
        "readback_passed": True,
        "scope": frozen["scope"],
    }


def evidence_paths(root):
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in {"evidence-index.json", "final-verification.json"}
    }


def final_audit(root, expected_tests):
    from audit_refined_checkpoint import regression_contract

    report = audit(root)
    if report != read(root / "verification.json"):
        raise ValueError("saved prefix verification disagrees with fresh readback")
    paths = evidence_paths(root)
    if paths != read(root / "evidence-index.json")["files_sha256"]:
        raise ValueError("prefix final evidence changed or unexpected files added")
    return {
        **report,
        "final_tests": regression_contract(root / "test-results.xml", expected_tests),
        "test_results_sha256": file_sha256(root / "test-results.xml"),
        "evidence_index_sha256": file_sha256(root / "evidence-index.json"),
        "final_complete_files_verified": len(paths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "prepare", "validate", "benchmark", "audit", "final-audit"),
        default="all",
    )
    parser.add_argument("--expected-tests", type=int)
    args = parser.parse_args()
    root = args.output_dir
    with exclusive_gpu_workflow() as run:
        if args.phase in ("all", "prepare"):
            prepare(root)
        if args.phase in ("all", "validate"):
            validate(root)
        if args.phase == "benchmark" or (
            args.phase == "all" and validation_readback(root)
        ):
            benchmark(run, root)
        if args.phase == "final-audit":
            if args.expected_tests is None or args.expected_tests <= 0:
                parser.error("final-audit requires positive expected-tests")
            save(root / "evidence-index.json", {"files_sha256": evidence_paths(root)})
            report = final_audit(root, args.expected_tests)
            save(root / "final-verification.json", report)
            print(json.dumps(report, indent=2), flush=True)
        elif args.phase in ("all", "audit"):
            report = audit(root)
            save(root / "verification.json", report)
            print(json.dumps(report, indent=2), flush=True)
            if not report["milestone_passed"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
