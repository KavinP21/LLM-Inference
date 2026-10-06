"""Post-freeze diagnosis: byte-sensitive cached reconstruction identity, no fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from forge_llm.benchmark import source_provenance
from forge_llm.cached_decode import replay_cached
from forge_llm.mlx_engine import MlxEngine
from forge_llm.second_order import file_sha256
from gpu_guard import exclusive_gpu_workflow
from run_cached_checkpoint import load_frozen
from run_second_order_checkpoint import save


def identity(values):
    if values.ndim != 1 or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("byte comparison requires a finite floating logit row")
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def verify(root):
    frozen = load_frozen(root)
    directory = root / "numerical-bytes"
    directory.mkdir()
    reports = {}
    for family, item in frozen["artifacts"].items():
        for corpus in (0, 1):
            quality_path = (
                root
                / "regression"
                / f"{family}-reconstruct-corpus{corpus}-quality.json"
            )
            quality = json.loads(quality_path.read_text())
            observations = []
            with ExitStack() as ownership:
                engines = [
                    ownership.enter_context(
                        MlxEngine(
                            item["model"],
                            max_model_length=2048,
                            kv_cache_bytes=1 << 30,
                            int8_mode=m,
                            decode_mode="rowwise",
                        )
                    )
                    for m in ("reconstruct", "dequantize")
                ]
                for index, case in enumerate(quality["cases"]):
                    replays = [
                        replay_cached(e, case["input_ids"], case["fp16_tokens"])
                        for e in engines
                    ]
                    rows = [
                        {"position": p, "native": identity(a), "composed": identity(b)}
                        for p, (a, b) in enumerate(
                            zip(replays[0]["logits"], replays[1]["logits"], strict=True)
                        )
                    ]
                    observations.append(
                        {
                            "case_index": index,
                            "rows": rows,
                            "byte_equal_rows": sum(
                                r["native"] == r["composed"] for r in rows
                            ),
                            "private_cache_reclaimed": all(
                                r["cache_reclaimed"] for r in replays
                            ),
                        }
                    )
                build = engines[0].build_info()
            total = sum(len(o["rows"]) for o in observations)
            equal = sum(o["byte_equal_rows"] for o in observations)
            passed = total == equal == 800 and all(
                o["private_cache_reclaimed"] for o in observations
            )
            key = f"{family}-corpus{corpus}"
            report = {
                "schema_version": 1,
                "purpose": __doc__,
                "environment": {"native_build": build, **source_provenance()},
                "model": item["model_provenance"],
                "configuration": {
                    "quality_report_sha256": file_sha256(quality_path),
                    "script_sha256": file_sha256(Path(__file__)),
                    "decode_mode": "rowwise",
                },
                "observations": observations,
                "total_rows": total,
                "byte_equal_rows": equal,
                "passed": passed,
                "limitations": "Additional post-freeze diagnostic only. Includes dtype, shape, and signed-zero-sensitive bytes; never fitting, timing, source-quality certification, or direct-kernel certification.",
            }
            save(directory / f"{key}.json", report)
            reports[key] = passed
            print(
                json.dumps(
                    {
                        "contract": key,
                        "byte_equal_rows": equal,
                        "total_rows": total,
                        "passed": passed,
                    }
                ),
                flush=True,
            )
    return all(reports.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with exclusive_gpu_workflow():
        passed = verify(args.root)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
