from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import create_engine


def compare(model_file: Path, model_name: str, prompts: list[str], max_new_tokens: int,
            backend: str = "auto", reference_device: str = "auto") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if reference_device == "auto":
        if torch.cuda.is_available():
            reference_device = "cuda"
        elif torch.backends.mps.is_available():
            reference_device = "mps"
        else:
            reference_device = "cpu"
    reference_dtype = torch.float16 if reference_device in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reference = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=reference_dtype
    ).to(reference_device).eval()
    engine = create_engine(model_file, backend=backend)
    cases = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids[0].tolist()
        reference_input = encoded.input_ids.to(reference_device)
        with torch.inference_mode():
            reference_logits = reference(reference_input).logits[0, -1].float().cpu().numpy()
            expected = reference.generate(
                reference_input, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.0, use_cache=True,
            )[0, len(input_ids):].cpu().tolist()
        actual = engine.generate(input_ids, max_new_tokens, [tokenizer.eos_token_id])
        forge_logits = np.asarray(engine.debug_prefill_logits(input_ids), dtype=np.float32)
        cosine = float(np.dot(reference_logits, forge_logits) /
                       (np.linalg.norm(reference_logits) * np.linalg.norm(forge_logits)))
        common = min(len(expected), len(actual))
        matches = sum(expected[i] == actual[i] for i in range(common))
        cases.append({
            "prompt": prompt, "reference": expected, "forge": actual,
            "matching_prefix": next((i for i in range(common) if expected[i] != actual[i]), common),
            "token_agreement": matches / max(len(expected), len(actual), 1),
            "logit_cosine_similarity": cosine,
            "logit_max_absolute_error": float(np.max(np.abs(reference_logits - forge_logits))),
            "top1_agrees": int(reference_logits.argmax()) == int(forge_logits.argmax()),
        })
    result = {"model": model_name, "backend": getattr(engine, "backend", "cuda"),
              "reference_device": reference_device,
              "max_new_tokens": max_new_tokens, "cases": cases,
              "all_exact": all(case["reference"] == case["forge"] for case in cases),
              "all_logits_accepted": all(case["logit_cosine_similarity"] >= 0.999 and
                                           case["top1_agrees"] for case in cases)}
    if hasattr(engine, "close"):
        engine.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Forge greedy output with Transformers")
    parser.add_argument("--engine-model", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--backend", choices=["auto", "cuda", "mlx"], default="auto")
    parser.add_argument("--reference-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts.json"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.engine_model, args.model, json.loads(args.prompts.read_text()),
                     args.max_new_tokens, args.backend, args.reference_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["all_exact"] or not result["all_logits_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
