# Benchmarking on Georgia Tech's PACE ICE

[tools/pace/](../tools/pace/) runs the engine on one H100 of PACE's
Instructional Cluster Environment. Gemma 4 and Qwen3.5 run on MLX's CUDA backend
([nvidia.md](nvidia.md)), inside an Apptainer container. Three batch jobs do
the work, in order:

| Job | Hardware | Time | What it does |
| --- | --- | --- | --- |
| `setup.sbatch` | 8 CPUs, no GPU | up to 2 h | Pulls the container image, builds the Python environment, downloads the two models (about 19 GB) |
| `check.sbatch` | 1 H100 | up to 45 min | Checks each MLX op on the GPU, runs the engine tests and the real-Qwen tests, then gets known answers from both models |
| `bench.sbatch` | 1 H100 | up to 4 h | Several users at once: on the engine, over the `/v1` API, then one long-context run |

Each job stops at the first failure, so a broken setup never holds the GPU
longer than it takes to find out.

## Running it

From a login node (`ssh <gt-username>@login-ice.pace.gatech.edu`, on the GT
VPN when off campus):

```bash
git clone https://github.com/KavinP21/LLM-Inference.git
cd LLM-Inference
sbatch tools/pace/setup.sbatch
```

Wait for it to finish (`squeue -u $USER`; the log is
`instinct-setup-<job>.out`), then:

```bash
sbatch tools/pace/check.sbatch
```

If its log ends with "All checks passed", start the benchmarks:

```bash
sbatch tools/pace/bench.sbatch
```

Results land in `benchmark-results/pace-<check|bench>-<job id>/`:

- the raw JSON
- `summary.md` with the tables
- the worker logs
- `gpu.csv`, GPU utilization and memory every 5 s

Copy them back with `scp` or Globus. Two knobs:

```bash
sbatch --export=ALL,USERS=1,4,16 tools/pace/bench.sbatch           # concurrency levels
sbatch --export=ALL,LONG_CONTEXT=262144 tools/pace/bench.sbatch     # the full context
```

## What the benchmarks measure

- **Engine level** (`benchmarks/multi_user.py`). Each user decodes 256 tokens
  with different sampling. Once they're a quarter in, an 8K-token prompt
  arrives. It reports each user's decode rate, how long the other users' tokens
  stall while that prompt prefills, the prompt's time to first token, and that
  every K/V page comes back. It runs at 1, 2, 4, 8 and 16 users for each model.
- **Over HTTP** (`benchmarks/serve_load.py`). One worker per model listens on
  `127.0.0.1` with up to 16 requests in flight. Each user holds a three-turn
  chat about its own 1,500-word document and streams 256-token replies. It
  reports total throughput, each user's decode rate, time to first token for
  the first and later turns, and how much of each later prompt came from the
  prefix cache.
- **Long context** (`benchmarks/long_context.py`). A 128K-token prompt on
  Gemma 4, then a follow-up turn. This shows the cost of the first prefill, the
  cost of a turn once the prefix is cached, and decode speed at that depth.

## PACE rules and how the scripts follow them

From PACE's [policies](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042198),
[ICE resources](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095),
[Slurm on ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096),
[storage on ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094)
and [Apptainer at PACE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043878):

- **Login nodes are for submitting jobs and light work.** PACE ends processes
  that use 90% of a CPU for over 30 minutes, or 20% of memory. Every script
  sources `env.sh`, which refuses to run outside a Slurm job. On a login node
  you only clone the repository and run `sbatch`.
- **Containers run on compute nodes only.** Apptainer is PACE's supported
  container platform, and pulling, building and running all happen on compute
  nodes. The image is pulled inside `setup.sbatch`.
- **Large downloads stay off the login nodes.** The models are downloaded in
  the setup job, which asks for no GPU, so no GPU hours go to downloading.
- **Partitions are assigned automatically, so jobs don't request one.** The
  scripts don't either.
- **One H100, with the memory PACE recommends.** The jobs request
  `--gres=gpu:H100:1` and `--mem-per-gpu=224G` (PACE's suggested value for the
  HGX H100 nodes) with 8 CPUs. Nothing asks for a whole node or more GPUs.
- **GPU jobs are capped at 16 hours and 16 GPU-hours, and a user may have 32
  GPU-hours running at once.** The jobs ask for 45 minutes and 4 hours and end
  as soon as their work does. Nothing sleeps or keeps the GPU reserved.
- **Home holds 30 GB and scratch 300 GB.** The image, the environment, the
  models and every cache (pip, Hugging Face, Apptainer, MLX kernels) go to
  `~/scratch/instinct`. The job copies the models to its local disk
  (`$TMPDIR`), which PACE suggests for heavy I/O and clears when the job ends.
  Scratch files untouched for 120 days are deleted at the end of the semester.
- **No services for other people.** The benchmark's model server listens only
  on `127.0.0.1`, on a free port. It takes a random token made in the job and
  never written anywhere, and it stops when the job does. Don't use the
  server's `--host 0.0.0.0` LAN mode on PACE.
- **Software installs live in your own space.** Everything is installed in a
  venv on scratch from public pip packages. The models are public Hugging Face
  checkpoints at pinned revisions: Gemma 4 26B A4B QAT and Qwen3.5-4B. The
  abliterated Gemma isn't downloaded there.
- **Use must follow Georgia Tech's IT policies.** That means the
  [Acceptable Use Policy and Cyber Security Policy](https://b.gatech.edu/it-policies).
  ICE is an instructional resource: you get access through a course, and
  PACE directs students' questions to their instructors and TAs. **Check with
  your instructor or TA that benchmarking this project fits how your course
  uses ICE.** The scripts can't decide that for you. For research use beyond a
  course, PACE's Phoenix cluster is the place.

## If something goes wrong

- **`check_mlx.py` reports a failed op.** For a fused op (`fused_*`), the engine
  already falls back to its plain version. `engine_picks` in the report shows
  which ones it uses. The required ops (quantized matmul, `gather_qmm`,
  indexing, sampling) have no fallback, so a failure there means MLX's CUDA
  build lacks something this engine needs. The report records the error and the
  versions.
- **Kernel compilation errors from MLX.** Try `INSTINCT_KERNELS=plain`, which
  also turns off `mx.compile`. Check that `nvidia-cuda-runtime-cu12` is in
  `pip-freeze.txt`; it supplies the headers MLX compiles with.
- **The driver is too old for the CUDA 12.9 libraries.** Add `-C cuda13` or
  `-C cuda12` to pick nodes by driver. MLX's CUDA 12 build needs driver 550.54
  or newer.
- **A job waits a long time.** H100s are popular. `pace-check-queue ice-gpu`
  shows what's free. Replacing `--gres=gpu:H100:1` with `-G 1 -C HX00` also
  accepts an H200.
