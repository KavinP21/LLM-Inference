"""Cached, common-prefix kernel divergence replay; diagnostics, never fitting/timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.paged_kv import MlxPagedKVStore
from forge_llm.second_order import file_sha256
from forge_llm.validate_int8 import compare_logits
from gpu_guard import exclusive_gpu_workflow
from trace_batch_numerics import difference, snapshot, stages


def divergences(report):
    result = []
    for index, case in enumerate(report["cases"]):
        metal, dense = case["int8_tokens"], case["dequantize_tokens"]
        if len(metal) != len(dense) or not metal:
            raise ValueError("malformed diagnostic token sequences")
        position = next(
            (p for p, (a, b) in enumerate(zip(metal, dense)) if a != b), None
        )
        if position is not None:
            result.append((index, position))
    return result


def replay(model_path, quality_path):
    quality = json.loads(quality_path.read_text())
    identity = model_provenance(model_path)
    if (
        identity["model_data_sha256"] != quality["quantized_model"]["model_data_sha256"]
        or quality["configuration"]["int8_mode"] != "metal"
        or quality["configuration"]["decode_mode"] != "rowwise"
        or quality["environment"]["runtime_source_sha256"]
        != source_provenance()["runtime_source_sha256"]
    ):
        raise ValueError(
            "diagnostic report does not match the frozen runtime/artifact/mode"
        )
    observations = []
    for index, position in divergences(quality):
        case = quality["cases"][index]
        snapshots, logits_by_mode = {}, {}
        for mode in ["metal", "reconstruct", "dequantize"]:
            with MlxEngine(
                model_path,
                max_model_length=2048,
                kv_cache_bytes=1 << 30,
                int8_mode=mode,
                decode_mode="rowwise",
            ) as engine:
                model = engine.model
                store = MlxPagedKVStore(
                    model.mx,
                    num_layers=model.config.num_hidden_layers,
                    num_kv_heads=model.config.num_key_value_heads,
                    head_dim=model.head_dim,
                )
                table = tuple(range((len(case["input_ids"]) + position + 15) // 16))
                try:
                    if position == 0:
                        raise ValueError(
                            "prefill divergence requires a separate prefill-stage diagnostic"
                        )
                    prefill = model.forward_paged_chunk(
                        case["input_ids"],
                        start_position=0,
                        block_table=table,
                        store=store,
                    )
                    np.asarray(prefill[-1])
                    for step in range(1, position + 1):
                        if step == position:
                            with stages(model) as (records, _):
                                logits = model.decode_paged_batch(
                                    [case["int8_tokens"][step - 1]],
                                    positions=[len(case["input_ids"]) + step - 1],
                                    block_tables=[table],
                                    store=store,
                                )
                            logits_by_mode[mode] = np.array(logits[0])
                            snapshots[mode] = snapshot(records, 0)
                        else:
                            logits = model.decode_paged_batch(
                                [case["int8_tokens"][step - 1]],
                                positions=[len(case["input_ids"]) + step - 1],
                                block_tables=[table],
                                store=store,
                            )
                            model.mx.eval(logits)
                    build = engine.build_info()
                finally:
                    store.clear()
        comparisons = []
        for baseline in ["reconstruct", "dequantize"]:
            stage_differences = []
            for reference, actual in zip(
                snapshots[baseline], snapshots["metal"], strict=True
            ):
                if reference["stage"] != actual["stage"]:
                    raise ValueError("diagnostic stage orders differ")
                stage_differences.append(
                    {
                        "stage": reference["stage"],
                        "input": difference(reference["input"], actual["input"]),
                        "outputs": [
                            difference(a, b)
                            for a, b in zip(
                                reference["outputs"], actual["outputs"], strict=True
                            )
                        ],
                    }
                )
            first = next(
                (
                    s
                    for s in stage_differences
                    if any(not o["exact"] for o in s["outputs"])
                ),
                None,
            )
            comparisons.append(
                {
                    "baseline": baseline,
                    "cached_logits": compare_logits(
                        logits_by_mode[baseline], logits_by_mode["metal"]
                    ),
                    "first_different_stage": first,
                    "stages": stage_differences,
                }
            )
        argmax = {
            mode: int(np.argmax(logits)) for mode, logits in logits_by_mode.items()
        }
        matches_saved = (
            argmax["metal"] == case["int8_tokens"][position]
            and argmax["dequantize"] == case["dequantize_tokens"][position]
        )
        observations.append(
            {
                "case_index": index,
                "generated_position": position,
                "common_prefix_tokens": case["int8_tokens"][:position],
                "argmax": argmax,
                "matches_saved_free_generation_decision": matches_saved,
                "comparisons": comparisons,
            }
        )
        print(
            json.dumps(
                {
                    "case": index,
                    "position": position,
                    "argmax": argmax,
                    "matches_saved": matches_saved,
                }
            ),
            flush=True,
        )
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "model": identity,
        "environment": {
            "native_build": build if observations else None,
            **source_provenance(),
        },
        "configuration": {
            "quality_report_sha256": file_sha256(quality_path),
            "script_sha256": file_sha256(Path(__file__)),
            "decode_mode": "rowwise",
        },
        "observations": observations,
        "reproduced_all_saved_kernel_divergences": all(
            o["matches_saved_free_generation_decision"] for o in observations
        ),
        "limitations": "Manual cached common-prefix replay with retained stage arrays; not scheduler, latency, quality certification, or post-regression calibration data.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        report = replay(args.model, args.quality)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if not report["reproduced_all_saved_kernel_divergences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
