"""Official-artifact bitwise batch replay across page, chunk and sliding boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.runtime import SequenceState
from gpu_guard import exclusive_gpu_workflow
from validate_batch_numerics import observe_logits, reclaimed, replay


def validate(model_path, int8_mode):
    lengths = [15, 16, 17, 31, 32, 33, 128, 512, 513]
    options = {
        "max_model_length": 1024,
        "kv_cache_bytes": 1 << 30,
        "prefill_chunk_size": 128,
        "int8_mode": int8_mode,
    }
    references = []
    with (
        MlxEngine(model_path, max_num_sequences=1, **options) as independent,
        observe_logits(independent) as current,
    ):
        vocab = independent.model.config.vocab_size
        cases = [
            {"input_ids": [(42 + i * 104729) % vocab for i in range(n)]}
            for n in lengths
        ]
        for case in cases:
            r = independent.submit(case["input_ids"], 32)
            logs = []
            for _ in range(1024):
                for event in independent.step():
                    logs.append(current[event.request_id])
                if independent.scheduler.request(r).state is SequenceState.COMPLETED:
                    break
            else:
                raise RuntimeError("independent boundary case failed to drain")
            references.append(
                {"tokens": independent.scheduler.request(r).output, "logits": logs}
            )
        independent_reclaimed = reclaimed(independent.stats())
    with MlxEngine(
        model_path, max_num_sequences=16, decode_mode="rowwise", **options
    ) as batched:
        result = replay(batched, cases, references)
        build = batched.build_info()
    summary = result["summary"]
    gates = {
        "independent_reclaimed": independent_reclaimed,
        "exact_tokens": summary["exact_token_cases"] == len(cases),
        "exact_logits": summary["exact_logit_rows"] == summary["total_logit_rows"],
        "events_exact": summary["events_exact"],
        "cache_reclaimed": summary["cache_reclaimed"],
    }
    gates["checkpoint_passed"] = all(gates.values())
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "model": model_provenance(model_path),
        "environment": {"native_build": build, **source_provenance()},
        "configuration": {
            "prompt_lengths": lengths,
            "output_tokens": 32,
            "int8_mode": int8_mode,
            "prefill_chunk_size": 128,
            "maximum_concurrency": 16,
            "input_ids_sha256": hashlib.sha256(
                json.dumps(cases, sort_keys=True).encode()
            ).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_checker_sha256": hashlib.sha256(
                Path(__file__).with_name("validate_batch_numerics.py").read_bytes()
            ).hexdigest(),
        },
        "cases": cases,
        "references": references,
        "batched": result,
        "gates": gates,
        "limitation": "Synthetic token IDs and bitwise same-artifact numerics, not semantic quality or timing. Global/local Gemma attention crosses the 512-token sliding boundary.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--int8-mode", choices=["auto", "metal", "reconstruct"], default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        result = validate(args.model, args.int8_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["gates"], indent=2))
    if not result["gates"]["checkpoint_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
