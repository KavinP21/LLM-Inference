# RTX/WSL2 setup and first validation

The NVIDIA machine is the only source of CUDA correctness and performance results. Use Ubuntu
24.04 under WSL2, keep the Windows NVIDIA driver current, and install the CUDA 12.8 toolkit inside
WSL without installing a second Linux display driver.

## One-time setup

From an elevated Windows terminal, install/update WSL and Ubuntu, then reboot if requested:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Inside Ubuntu, install the build prerequisites and the NVIDIA CUDA 12.8 WSL toolkit from NVIDIA's
WSL repository:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev python3-venv git
```

Do not install an Ubuntu `nvidia-driver-*` package in WSL. GPU access comes from the Windows driver.
Install Nsight Systems CLI, Nsight Compute, and Compute Sanitizer with the CUDA toolkit. In the
Windows NVIDIA Control Panel, enable access to GPU performance counters before using Nsight
Compute.

## Verify and build

```bash
./tools/verify_wsl_environment.sh
cmake --preset cuda-release
cmake --build --preset cuda-release -j
ctest --preset cuda-release
./tools/run_sanitizer.sh build/cuda-release
```

Then build the Python package in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[export,test]'
```

## First model validation

```bash
mkdir -p models correctness-results
forge-export Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b.engine
build/cuda-release/forge-inspect-model models/qwen2.5-0.5b.engine
forge-correctness \
  --engine-model models/qwen2.5-0.5b.engine \
  --prompts benchmarks/prompts.json \
  --max-new-tokens 32 \
  --output correctness-results/qwen2.5-0.5b.json
```

Do not begin performance tuning until:

- all CUDA unit tests pass;
- Compute Sanitizer reports no memory or race errors;
- prefill logit cosine similarity is at least `0.999` with matching top-1 tokens; and
- all fixed prompts produce exact greedy-token parity.

If parity fails, reduce to a one-token prompt and inspect in this order: row-major cuBLASLt output,
Q/K/V bias, RoPE pairing, GQA head mapping, causal bounds, RMSNorm accumulation, then weight names.

## First performance capture

Use a release build and keep all other GPU workloads closed:

```bash
mkdir -p profiles benchmark-results
nsys profile --trace=cuda,nvtx,cublas --sample=none \
  -o profiles/forge-p128-o32-c1 \
  python -m forge_llm.benchmark \
    --model models/qwen2.5-0.5b.engine \
    --prompts benchmark-results/workloads/p128-c1.json \
    --output benchmark-results/forge-p128-o32-c1.json
```

Run a short matrix first (`--prompt-lengths 32 128 --output-lengths 32 --concurrencies 1 4`) before
committing a full weekend to the canonical grid.

