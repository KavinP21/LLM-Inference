"""Running on an NVIDIA PC: which models fit the card, the GPU memory check,
the engine option that sizes recurrent-state slots, and signing in from a
browser (Windows' browser, under WSL)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from instinct import nvidia
from instinct.agents.backends import memory
from instinct.agents.backends.config import WorkerConfig
from instinct.harness import cli as harness_cli

GB, GIB = nvidia.GB, nvidia.GIB


def test_an_rtx_3070_ti_runs_qwen_for_everything(tmp_path):
    config = nvidia.profile(8 * GIB, root=tmp_path)  # Models not downloaded.
    (qwen,) = config["workers"]
    assert qwen["model"].endswith("qwen3.5-4b-mlx") and qwen["group"] == "driver"
    assert config["runtime"]["subagent_group"] == "driver"
    assert qwen["engine_options"]["kv_bits"] == 8
    assert (
        qwen["max_pending"] == 4 and qwen["engine_options"]["prefix_cache_entries"] == 8
    )
    assert qwen["max_model_length"] == 131072
    # Everything MLX may hold, plus the reserve, fits the card.
    limit = qwen["engine_options"]["memory_limit_bytes"]
    assert limit + nvidia.RESERVED_BYTES <= 8 * GIB
    assert limit >= (
        nvidia.QWEN_BYTES
        + qwen["kv_cache_bytes"]
        + 12 * nvidia.QWEN_SLOT_BYTES
        + nvidia.QWEN_ACTIVATION_BYTES
    )
    report = memory.check(config)
    assert report["gpu"]["total_bytes"] <= report["gpu"]["limit_bytes"]
    assert nvidia.models_for(8 * GIB) == ["qwen3.5-4b"]


@pytest.mark.parametrize(
    "gib,models,subagent",
    [
        (6, ["qwen"], "driver"),
        (12, ["qwen"], "driver"),
        (16, ["qwen"], "driver"),
        (20, ["qwen"], "driver"),  # Just short of Gemma's 21 GB.
        (24, ["gemma-4"], "driver"),
        (48, ["gemma-4", "qwen"], "subagent"),
        (80, ["gemma-4", "qwen"], "subagent"),
    ],
)
def test_bigger_cards_get_gemma(tmp_path, monkeypatch, gib, models, subagent):
    config = nvidia.profile(gib * GIB, root=tmp_path, host_bytes=32 * GB)
    assert [w["name"] for w in config["workers"]] == models
    assert config["runtime"]["subagent_group"] == subagent
    assert not any("abliterated" in w["model"] for w in config["workers"])
    report = memory.check(config)
    assert report["gpu"]["total_bytes"] <= (gib * GIB - nvidia.RESERVED_BYTES) * 1.001
    assert (
        config["runtime"]["max_context_tokens"]
        < config["workers"][0]["max_model_length"]
    )
    assert _definitions(config, monkeypatch)  # Every worker is a valid WorkerConfig.


def _definitions(config, monkeypatch):
    """The WorkerConfigs the pool would start, without starting anything."""
    import asyncio

    from instinct.agents import cli

    seen = []

    class Process:
        @classmethod
        async def start(cls, definition, **_):
            seen.append(definition)
            return cls()

        async def close(self):
            pass

    monkeypatch.setattr(cli, "LocalProcessBackend", Process)
    asyncio.run(asyncio.run(cli.create_pool(json.loads(json.dumps(config)))).close())
    return seen


def test_small_cards_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="at least 6 GB"):
        nvidia.profile(4 * GIB)


def test_nvidia_smi_output_is_parsed():
    info = nvidia.parse_smi("NVIDIA GeForce RTX 3070 Ti, 8192, 8.6, 581.57\n")
    assert info == {
        "name": "NVIDIA GeForce RTX 3070 Ti",
        "memory_bytes": 8 * GIB,
        "compute_capability": "8.6",
        "driver": "581.57",
    }
    assert nvidia.parse_smi("No devices were found\n") is None


def test_profile_command_writes_a_config_that_loads(tmp_path):
    from instinct.agents.cli import load_config

    output = tmp_path / "nvidia.json"
    assert (
        nvidia.main(
            [
                "profile",
                "--gpu-gb",
                "8.59",
                "--root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    loaded = load_config(output)
    assert loaded["memory"]["gpu_gb"] < 8.59
    assert json.loads(output.read_text())["workers"][0]["name"] == "qwen"


def test_gpu_memory_is_checked_separately_from_host_memory(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    with open(model / "model.safetensors", "wb") as stream:
        stream.truncate(3 * GB)
    worker = {
        "name": "q",
        "model": str(model),
        "tokenizer": str(model),
        "kv_cache_bytes": 2 * GB,
        "engine_options": {"memory_limit_bytes": 6 * GB},
    }
    # The explicit limit counts, on the GPU; the host pays only the process.
    report = memory.check({"workers": [worker], "memory": {"max_gb": 4, "gpu_gb": 7}})
    assert report["gpu"]["workers"] == {"q": 6 * GB}
    assert report["workers"] == {"q": memory.PROCESS_BYTES}
    with pytest.raises(ValueError, match="GPU memory"):
        memory.check({"workers": [worker], "memory": {"gpu_gb": 5}})
    with pytest.raises(ValueError, match="gpu_gb"):
        memory.check({"workers": [worker], "memory": {"gpu_gb": "8"}})
    # Without gpu_gb (a Mac's unified memory) nothing changes.
    del worker["engine_options"]
    mac = memory.check({"workers": [worker]})
    assert (
        "gpu" not in mac
        and mac["workers"]["q"] == memory.mlx_limit(worker) + memory.PROCESS_BYTES
    )


def test_prefix_cache_entries_reach_the_engine(tmp_path, monkeypatch):
    base = dict(model=str(tmp_path), tokenizer=str(tmp_path), backend="mlx")
    WorkerConfig(**base, engine_options={"prefix_cache_entries": 8})
    for bad in (0, 2000, "8", True):
        with pytest.raises(ValueError, match="prefix_cache_entries"):
            WorkerConfig(**base, engine_options={"prefix_cache_entries": bad})
    seen = {}

    class Engine:
        def __init__(self, path, **options):
            seen.update(options)
            raise RuntimeError("stop here")

    transformers = pytest.importorskip("transformers")
    import instinct.chat_runner as chat_runner

    class Tokenizer:
        eos_token_id = 1
        chat_template = "{{ messages }}"

    monkeypatch.setattr(chat_runner, "MlxEngine", Engine)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    with pytest.raises(RuntimeError, match="stop here"):
        chat_runner.ChatRunner(tmp_path, prefix_cache_entries=8)
    assert seen["prefix_cache_entries"] == 8
    seen.clear()
    with pytest.raises(RuntimeError, match="stop here"):
        chat_runner.ChatRunner(tmp_path)
    assert "prefix_cache_entries" not in seen  # The engine's default stays.


def test_wsl_opens_the_link_in_windows_browser(monkeypatch):
    runs = []
    monkeypatch.setattr(harness_cli, "in_wsl", lambda: True)
    monkeypatch.setattr(
        harness_cli.shutil, "which", lambda name: name if name == "cmd.exe" else None
    )

    def run(command, **kwargs):
        runs.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(harness_cli.subprocess, "run", run)
    assert harness_cli.open_browser("http://127.0.0.1:8765/#token=abc")
    assert runs == [["cmd.exe", "/c", "start", "", "http://127.0.0.1:8765/#token=abc"]]
    monkeypatch.setattr(harness_cli.shutil, "which", lambda name: None)
    assert not harness_cli.open_browser("http://127.0.0.1:8765/#token=abc")


def test_open_is_only_for_running_without_the_mac_app(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["instinct", "--demo", "--app", "--open"])
    with pytest.raises(SystemExit) as exited:
        harness_cli.main()
    assert exited.value.code == 2


def test_the_launcher_script_parses():
    script = Path(__file__).resolve().parents[1] / "tools/nvidia/instinct.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    usage = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert usage.returncode == 2 and "setup" in usage.stdout and "start" in usage.stdout
