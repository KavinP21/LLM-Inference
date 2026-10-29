from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

from .benchmark import command_output, percentile


def one_trial(model: object, encoded: dict, output_length: int) -> dict[str, float]:
    import torch

    input_ids = encoded["input_ids"].cuda(non_blocking=True)
    attention_mask = encoded["attention_mask"].cuda(non_blocking=True)
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        next_tokens = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
    torch.cuda.synchronize()
    first = time.perf_counter_ns()
    for _ in range(1, output_length):
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device="cuda",
                    dtype=attention_mask.dtype,
                ),
            ),
            dim=1,
        )
        with torch.inference_mode():
            output = model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            next_tokens = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
    torch.cuda.synchronize()
    ended = time.perf_counter_ns()
    ttft_ms = (first - started) / 1e6
    e2e_ms = (ended - started) / 1e6
    return {
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": (e2e_ms - ttft_ms) / (output_length - 1)
        if output_length > 1
        else math.nan,
    }


def benchmark_reference(
    model_name: str,
    prompt_file: Path,
    output_length: int,
    warmups: int,
    repetitions: int,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = json.loads(prompt_file.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt file must be a non-empty JSON array")
    if all(isinstance(item, str) for item in prompts):
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    elif all(
        isinstance(item, list)
        and item
        and all(isinstance(token, int) for token in item)
        for item in prompts
    ):
        width = max(len(item) for item in prompts)
        padded = [
            [tokenizer.pad_token_id] * (width - len(item)) + item for item in prompts
        ]
        masks = [[0] * (width - len(item)) + [1] * len(item) for item in prompts]
        encoded = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
    else:
        raise ValueError("prompts must be all strings or all non-empty token-id arrays")
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )
    for _ in range(warmups):
        one_trial(model, encoded, min(output_length, 4))
    trials = [one_trial(model, encoded, output_length) for _ in range(repetitions)]
    ttft = [trial["ttft_ms"] for trial in trials]
    tpot = [trial["tpot_ms"] for trial in trials]
    e2e = [trial["e2e_ms"] for trial in trials]
    batch = len(prompts)
    return {
        "schema_version": 1,
        "implementation": "transformers-fp16-manual-cache-loop",
        "configuration": {
            "model": model_name,
            "output_length": output_length,
            "warmups": warmups,
            "repetitions": repetitions,
            "concurrency": batch,
            "prompt_token_lengths": encoded["attention_mask"].sum(dim=1).tolist(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "aggregate": {
            "generated_tokens_per_second": batch
            * output_length
            * repetitions
            / (sum(e2e) / 1000),
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
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Transformers FP16 comparison workload"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
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
    args = parser.parse_args()
    result = benchmark_reference(
        args.model, args.prompts, args.output_length, args.warmups, args.repetitions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
