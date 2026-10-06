"""Diagnostic-only teacher-forced decode stage replay; never timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from forge_llm.benchmark import model_provenance, source_provenance
from forge_llm.mlx_engine import MlxEngine
from forge_llm.paged_kv import MlxPagedKVStore
from gpu_guard import exclusive_gpu_workflow


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("stage shapes do not align")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "exact": bool(np.array_equal(a, b)),
        "changed_elements": int(np.count_nonzero(delta)),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


@contextmanager
def stages(model):
    """Retain lazy arrays, copying only after normal decode materialization.

    No extra evaluation barriers are inserted into the original execution path.
    References are diagnostic overhead and must not be used for benchmarks.
    """
    records, originals = [], {}

    def wrap(name, original):
        def traced(*args, **kwargs):
            result = original(*args, **kwargs)
            label = name
            if name in {"_linear", "_rms_norm"}:
                label += ":" + args[1]
            elif "layer" in kwargs:
                label += ":" + str(kwargs["layer"])
            elif name == "_paged_decode_attention_batch":
                label += ":" + str(args[1])
            outputs = result[:2] if name == "_rope_write_decode" else (result,)
            records.append((label, name, args, kwargs, outputs))
            return result

        return traced

    for name in (
        "_embed",
        "_rms_norm",
        "_linear",
        "_rope_write_decode",
        "_paged_decode_attention_batch",
        "_gelu_tanh",
    ):
        if hasattr(model, name):
            originals[name] = getattr(model, name)
            setattr(model, name, wrap(name, originals[name]))
    try:
        yield records, originals
    finally:
        for name, original in originals.items():
            setattr(model, name, original)


def snapshot(records, row):
    return [
        {
            "stage": label,
            "input": np.array(args[0][row : row + 1]),
            "input_shape": list(args[0].shape),
            "outputs": [np.array(value[row : row + 1]) for value in outputs],
        }
        for label, _, args, _, outputs in records
    ]


def trace(model_path, quality_path, indices, batch_size, steps, decode_mode="batched"):
    quality = json.loads(quality_path.read_text())
    identity = model_provenance(model_path)
    if identity["model_data_sha256"] != quality["source_model"]["model_data_sha256"]:
        raise ValueError("diagnostic inputs belong to a different source artifact")
    cases = quality["cases"]
    reports = []
    with MlxEngine(
        model_path,
        max_model_length=2048,
        kv_cache_bytes=1 << 30,
        decode_mode=decode_mode,
    ) as engine:
        model = engine.model
        for index in indices:
            selected = [cases[index]] + [
                case for i, case in enumerate(cases) if i != index
            ][: batch_size - 1]
            stores = [
                MlxPagedKVStore(
                    model.mx,
                    num_layers=model.config.num_hidden_layers,
                    num_kv_heads=model.config.num_key_value_heads,
                    head_dim=model.head_dim,
                )
                for _ in range(2)
            ]
            tables = []
            next_page = 0
            try:
                for case in selected:
                    pages = (len(case["input_ids"]) + steps + 15) // 16
                    tables.append(tuple(range(next_page, next_page + pages)))
                    next_page += pages
                    model.forward_paged_chunk(
                        case["input_ids"],
                        start_position=0,
                        block_table=tables[-1],
                        store=stores[1],
                    )
                model.forward_paged_chunk(
                    selected[0]["input_ids"],
                    start_position=0,
                    block_table=tables[0],
                    store=stores[0],
                )
                for position in range(1, steps + 1):
                    tokens = [case["fp16_tokens"][position - 1] for case in selected]
                    positions = [
                        len(case["input_ids"]) + position - 1 for case in selected
                    ]
                    with stages(model) as (records, _):
                        independent = model.decode_paged_batch(
                            tokens[:1],
                            positions=positions[:1],
                            block_tables=tables[:1],
                            store=stores[0],
                        )
                    reference_stages = snapshot(records, 0)
                    with stages(model) as (records, originals):
                        batched = model.decode_paged_batch(
                            tokens,
                            positions=positions,
                            block_tables=tables,
                            store=stores[1],
                        )
                    actual_stages = snapshot(records, 0)
                    comparisons = []
                    for ref, actual, record in zip(
                        reference_stages, actual_stages, records, strict=True
                    ):
                        label, name, args, kwargs, _outputs = record
                        if label != ref["stage"]:
                            raise ValueError("stage execution order changed")
                        item = {
                            "stage": label,
                            "batch_input_shape": actual["input_shape"],
                            "input": difference(ref["input"], actual["input"]),
                            "outputs": [
                                difference(a, b)
                                for a, b in zip(
                                    ref["outputs"], actual["outputs"], strict=True
                                )
                            ],
                        }
                        # Isolate local shape dependence on the SAME materialized input,
                        # instead of attributing propagated upstream error to this operation.
                        if name in {"_linear", "_rms_norm", "_gelu_tanh"}:
                            row_input = model.mx.array(actual["input"])
                            replay = originals[name](row_input, *args[1:], **kwargs)
                            item["same_input_single_row_replay"] = difference(
                                np.array(replay), actual["outputs"][0]
                            )
                        comparisons.append(item)
                    first = next(
                        (
                            i
                            for i in comparisons
                            if any(not o["exact"] for o in i["outputs"])
                        ),
                        None,
                    )
                    reports.append(
                        {
                            "case_index": index,
                            "generated_position": position,
                            "teacher_forced": True,
                            "independent_argmax": int(
                                model.mx.argmax(independent[0]).item()
                            ),
                            "batched_argmax": int(model.mx.argmax(batched[0]).item()),
                            "first_different_stage": first,
                            "stages": comparisons,
                        }
                    )
            finally:
                for store in stores:
                    store.clear()
        build = engine.build_info()
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "model": identity,
        "environment": {"native_build": build, **source_provenance()},
        "configuration": {
            "case_indices": indices,
            "batch_size": batch_size,
            "decode_steps": steps,
            "decode_mode": decode_mode,
            "input_report_sha256": hashlib.sha256(
                quality_path.read_bytes()
            ).hexdigest(),
            "input_report_runtime_sha256": quality["environment"][
                "runtime_source_sha256"
            ],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Historical tokens are teacher-forcing inputs, not current independent-output certification. Diagnostic local replay includes host copies; no timings.",
        },
        "replays": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=[22, 24])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--decode-mode", choices=["batched", "rowwise"], default="batched"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 2 or args.steps < 1:
        parser.error("batch-size >= 2 and steps >= 1 are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        result = trace(
            args.model,
            args.quality,
            args.cases,
            args.batch_size,
            args.steps,
            args.decode_mode,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for replay in result["replays"]:
        print(json.dumps({k: v for k, v in replay.items() if k != "stages"}))


if __name__ == "__main__":
    main()
