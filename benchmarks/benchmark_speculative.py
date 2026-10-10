#!/usr/bin/env python3
"""Matched, alternating-order speculative benchmark with independent parity.

No oracle/replay draft: all proposals come from prompt/history n-grams. Writes
raw tokens, timings, acceptance and execution provenance even when parity fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.speculative import NGramDraft, NoDraft, SpeculativeEngine
from gpu_guard import exclusive_gpu_workflow


def run(args):
    import mlx.core as mx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    runtime_root = Path(__file__).resolve().parents[1] / "python" / "forge_llm"
    implementation_hashes = {
        name: hashlib.sha256((runtime_root / name).read_bytes()).hexdigest()
        for name in (
            "speculative.py",
            "paged_kv.py",
            "backends/mlx.py",
            "backends/gemma3.py",
        )
    }
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        raise ValueError("prompts must be a nonempty JSON list of text tasks")
    encoded = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if args.chat
        else tokenizer.encode(text)
        for text in prompts
    ]
    encoded = [
        list(ids["input_ids"] if hasattr(ids, "keys") else ids) for ids in encoded
    ]
    settings = {
        "max_model_length": args.max_model_length,
        "kv_cache_bytes": args.kv_cache_bytes,
        "prefill_chunk_size": args.prefill_chunk_size,
    }
    # The existing continuous engine is an independent one-request canonical
    # decode reference. Its token outputs are collected before timed trials.
    with MlxEngine(args.model, decode_mode="rowwise", **settings) as canonical:
        references = [canonical.generate(ids, args.output_length) for ids in encoded]
    trials = []
    providers = {"baseline": NoDraft, "ngram": NGramDraft}
    with SpeculativeEngine(
        args.model,
        draft_tokens=args.draft_tokens,
        adaptive=not args.fixed_budget,
        **settings,
    ) as engine:
        for _ in range(args.warmups):
            for provider in providers.values():
                engine.draft = provider()
                for prompt in encoded:
                    engine.generate(prompt, args.output_length)
        for repetition in range(args.repetitions):
            variants = (
                list(providers) if repetition % 2 == 0 else list(reversed(providers))
            )
            for index, prompt in enumerate(encoded):
                for variant in variants:
                    engine.draft = providers[variant]()
                    result = engine.generate_result(prompt, args.output_length)
                    trials.append(
                        {
                            "repetition": repetition,
                            "prompt_index": index,
                            "variant": variant,
                            "tokens": result.tokens,
                            "text": tokenizer.decode(result.tokens),
                            "canonical_token_parity": result.tokens
                            == references[index],
                            "finish_reason": result.finish_reason,
                            "stats": result.stats.to_dict(),
                        }
                    )
        info = mx.device_info()
    summaries = []
    for index, text in enumerate(prompts):
        groups = {
            variant: [
                t
                for t in trials
                if t["variant"] == variant and t["prompt_index"] == index
            ]
            for variant in providers
        }
        baseline = statistics.median(
            t["stats"]["elapsed_seconds"] for t in groups["baseline"]
        )
        spec = statistics.median(t["stats"]["elapsed_seconds"] for t in groups["ngram"])
        proposed = sum(t["stats"]["draft_tokens"] for t in groups["ngram"])
        accepted = sum(t["stats"]["accepted_draft_tokens"] for t in groups["ngram"])
        summaries.append(
            {
                "prompt_index": index,
                "prompt": text,
                "prompt_tokens": len(encoded[index]),
                "baseline_median_seconds": baseline,
                "ngram_median_seconds": spec,
                "speedup": baseline / spec,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "canonical_token_parity": all(
                    t["canonical_token_parity"]
                    for group in groups.values()
                    for t in group
                ),
            }
        )
    return {
        "schema": "forge_speculative_benchmark_v1",
        "model": str(args.model),
        "model_provenance": model_provenance(args.model),
        "source_provenance": source_provenance(),
        "implementation_source_sha256_at_start": implementation_hashes,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": __import__("mlx.core").core.__version__,
            "metal": info,
        },
        "configuration": {
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "output_length": args.output_length,
            "draft_tokens": args.draft_tokens,
            "adaptive": not args.fixed_budget,
            "verification_mode": "native_fp16_block",
            "chat": args.chat,
            "max_model_length": args.max_model_length,
            "prefill_chunk_size": args.prefill_chunk_size,
            "kv_cache_bytes": args.kv_cache_bytes,
            "timing_order": "alternating_baseline_ngram",
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_json_sha256": hashlib.sha256(
                (args.tokenizer / "tokenizer.json").read_bytes()
            ).hexdigest(),
            "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        },
        "canonical_outputs": references,
        "summary": summaries,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="local tokenizer directory"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).with_name("speculative-prompts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--max-model-length", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--kv-cache-bytes", type=int, default=512 << 20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--fixed-budget", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--require-token-parity", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a fresh output path is required")
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    with exclusive_gpu_workflow():
        result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2))
    if args.require_token_parity and not all(
        row["canonical_token_parity"] for row in result["summary"]
    ):
        raise SystemExit("canonical greedy token parity failed; raw evidence saved")


if __name__ == "__main__":
    main()
