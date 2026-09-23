from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical Forge benchmark matrix"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "cuda", "mlx"], default="auto")
    parser.add_argument(
        "--mlx-kernel-mode",
        choices=["baseline", "fused", "full"],
        default="full",
        help=(
            "MLX ablation: fallback ops, custom fusions, or fusions plus "
            "paged attention"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[32, 128, 512, 1024]
    )
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-model-length", type=int, default=2048)
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    parser.add_argument(
        "--token-id",
        type=int,
        default=42,
        help="non-special vocabulary token used by deterministic synthetic prompts",
    )
    parser.add_argument("--with-reference", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload_dir = args.output_dir / "workloads"
    workload_dir.mkdir(exist_ok=True)
    for prompt_length in args.prompt_lengths:
        for concurrency in args.concurrencies:
            workload = workload_dir / f"p{prompt_length}-c{concurrency}.json"
            workload.write_text(
                json.dumps(
                    [[args.token_id] * prompt_length for _ in range(concurrency)]
                )
                + "\n"
            )
            for output_length in args.output_lengths:
                stem = f"p{prompt_length}-o{output_length}-c{concurrency}"
                common = [
                    "--prompts",
                    str(workload),
                    "--output-length",
                    str(output_length),
                    "--warmups",
                    str(args.warmups),
                    "--repetitions",
                    str(args.repetitions),
                ]
                run(
                    [
                        sys.executable,
                        "-m",
                        "forge_llm.benchmark",
                        "--model",
                        str(args.model),
                        "--backend",
                        args.backend,
                        "--mlx-kernel-mode",
                        args.mlx_kernel_mode,
                        "--tokenizer",
                        args.tokenizer,
                        "--output",
                        str(args.output_dir / f"forge-{stem}.json"),
                        "--max-sequences",
                        str(max(args.concurrencies)),
                        "--max-model-length",
                        str(args.max_model_length),
                        "--kv-cache-mib",
                        str(args.kv_cache_mib),
                        *common,
                    ]
                )
                if args.with_reference:
                    run(
                        [
                            sys.executable,
                            "-m",
                            "forge_llm.benchmark_reference",
                            "--model",
                            args.tokenizer,
                            "--output",
                            str(args.output_dir / f"transformers-{stem}.json"),
                            *common,
                        ]
                    )


if __name__ == "__main__":
    main()
