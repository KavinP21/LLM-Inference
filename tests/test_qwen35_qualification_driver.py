"""Portable qualification-driver checks, without loading MLX or a checkpoint."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("qualify_qwen35_speculative")


def test_evidence_file_is_claimed_once_and_old_result_is_preserved(driver, tmp_path):
    path = tmp_path / "result.json"
    report = driver.Report(path, {"rows": [], "passed": False})
    report.row("check", {"passed": True, "tokens": [1, 2]})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        driver.Report(path, {"rows": [], "passed": False})
    assert path.read_bytes() == original
    assert json.loads(original)["rows"][0]["tokens"] == [1, 2]


def test_forced_draft_tracks_current_output_and_changes_only_requested_position(driver):
    reference = list(range(100))
    draft = driver.ForcedPrefixDraft(reference, 8, 2, 1000, 2)
    first = draft.propose([10] * 9, 4, lambda: False)
    assert first == [1, 2, 4, 4]
    second = draft.propose([10] * 13, 4, lambda: False)
    assert second == [5, 6, 8, 8]
    assert draft.propose([10] * 17, 4, lambda: False) == []
    assert [block["produced"] for block in draft.blocks] == [1, 5]


def live_engine():
    def array(shape, bytes_per_element):
        size = bytes_per_element
        for dimension in shape:
            size *= dimension
        return SimpleNamespace(
            shape=shape,
            nbytes=size,
            dtype="float32" if bytes_per_element == 4 else "bfloat16",
        )

    caches = []
    for index in range(32):
        if (index + 1) % 4 == 0:
            caches.append(SimpleNamespace(offset=100, keys=None, values=None))
        else:
            states = array((1, 4, 32, 128, 128), 4)
            final = array((1, 32, 128, 128), 4)
            source = array((1, 8, 8192), 2)
            caches.append(
                SimpleNamespace(
                    cache=[None, final],
                    _speculation={
                        "length": 5,
                        "initial_state": [None, None],
                        "records": {
                            0: ("window", source, 3),
                            1: ("states", states, final),
                        },
                    },
                )
            )
    return SimpleNamespace(_extra_reserved=255983616, _cache=caches)


def test_cancellation_proof_requires_every_live_fp32_recurrent_timeline(driver):
    engine = live_engine()
    proof = driver.active_recurrent_snapshot(engine)
    assert proof["recurrent_layers"] == 24
    assert proof["attention_offset"] == 100
    assert proof["layers"][0]["intermediate_bytes"] == 8 << 20
    engine._cache[0]._speculation = None
    assert driver.active_recurrent_snapshot(engine) is None
    engine = live_engine()
    engine._cache[0]._speculation["records"][1][1].dtype = "bfloat16"
    assert driver.active_recurrent_snapshot(engine) is None
    engine = live_engine()
    engine._extra_reserved = 0
    assert driver.active_recurrent_snapshot(engine) is None


def test_timed_trials_warm_each_mode_and_alternate_order(driver):
    class Engine:
        _spec_options = None

        def __init__(self):
            self.variants = []

        def generate(self, ids, count):
            self.variants.append(self._spec_options is not None)
            return [1, 2, 3]

        def stats(self):
            return {
                "active_requests": 0,
                "kv_reserved_bytes": 0,
                "kv_persistent_bytes": 0,
                "speculative_reserved_bytes": 0,
                "speculative": {"draft_tokens": 2, "accepted_draft_tokens": 1},
            }

    engine = Engine()
    args = SimpleNamespace(warmups=1, repetitions=3, output_tokens=64, draft_tokens=4)
    report = driver.timing_run(engine, [8], args)
    assert report["passed"]
    assert engine.variants == [False, True, False, True, True, False, False, True]
    assert len(report["warmups"]) == 2 and len(report["trials"]) == 6
    assert all(trial["output_tokens"] == 3 for trial in report["trials"])
    assert report["proposed_tokens"] == 6 and report["accepted_tokens"] == 3


def test_parity_failure_is_reported_instead_of_hidden_in_speed_summary(driver):
    class Engine:
        _spec_options = None

        def generate(self, ids, count):
            return [2] if self._spec_options is not None else [1]

        def stats(self):
            return {
                "active_requests": 0,
                "kv_reserved_bytes": 0,
                "kv_persistent_bytes": 0,
                "speculative_reserved_bytes": 0,
                "speculative": {"draft_tokens": 1, "accepted_draft_tokens": 0},
            }

    args = SimpleNamespace(warmups=1, repetitions=3, output_tokens=64, draft_tokens=4)
    report = driver.timing_run(Engine(), [8], args)
    assert not report["passed"]
    assert all(
        not trial["canonical_token_parity"]
        for trial in report["trials"]
        if trial["speculative"]
    )


def test_provenance_ignores_evidence_dirtiness_but_rejects_changed_code(
    driver, tmp_path, monkeypatch
):
    import hashlib

    path = tmp_path / "runtime.py"
    path.write_text("original")
    source = {
        "git_commit": "head",
        "runtime_source_sha256": "runtime",
        "git_dirty": False,
    }
    payload = {
        "source": source,
        "implementation_source_sha256": {
            "runtime.py": hashlib.sha256(path.read_bytes()).hexdigest()
        },
        "passed": True,
    }
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(
        driver, "source_provenance", lambda: {**source, "git_dirty": True}
    )
    driver.finalize_provenance(payload)
    assert payload["passed"] and payload["source_unchanged"]
    path.write_text("changed while running")
    driver.finalize_provenance(payload)
    assert not payload["passed"] and not payload["source_unchanged"]


def test_provenance_rejects_runtime_or_commit_change(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    payload = {
        "source": {"git_commit": "head", "runtime_source_sha256": "runtime"},
        "implementation_source_sha256": {},
        "passed": True,
    }
    monkeypatch.setattr(
        driver,
        "source_provenance",
        lambda: {"git_commit": "head", "runtime_source_sha256": "changed"},
    )
    driver.finalize_provenance(payload)
    assert not payload["passed"]


def test_video_grader_rejects_ambiguous_and_nonfinite_json(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    core = importlib.import_module("qualify_qwen35")
    assert core.parse_colors('{"first":"red","last":"blue"}') == {
        "first": "red",
        "last": "blue",
    }
    assert core.parse_colors('{"first":"red","first":"blue","last":"blue"}') is None
    assert core.parse_colors('{"first":NaN,"last":"blue"}') is None
    fenced = '  ```json\n{"first":"Red","last":"Blue"}\n```  '
    assert core.parse_colors(fenced) == {"first": "Red", "last": "Blue"}
    assert core.normalize_colors(core.parse_colors(fenced)) == {
        "first": "red",
        "last": "blue",
    }
    assert core.parse_colors('Here are colors: {"first":"red","last":"blue"}') is None
    assert (
        core.normalize_colors({"first": "red", "last": "blue", "extra": True}) is None
    )
    assert core.normalize_colors({"first": "red", "last": 3}) is None


@pytest.mark.parametrize("fail", [False, True])
def test_full_context_probe_is_bounded_and_restores_eos_policy(monkeypatch, fail):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    core = importlib.import_module("qualify_qwen35")

    class Engine:
        max_model_length = 8
        _eos = frozenset({248044, 248046})

        def generate(self, ids, count, eos):
            assert ids == [1234] * 6 and count == 2 and eos == [0]
            self._eos = frozenset(eos)
            if fail:
                raise RuntimeError("execution failure")
            return [3, 4]

        def stats(self):
            return {
                "active_requests": 0,
                "kv_reserved_bytes": 0,
                "kv_persistent_bytes": 0,
                "speculative_reserved_bytes": 0,
                "last_kv_peak_bytes": 100,
            }

    engine = Engine()
    if fail:
        with pytest.raises(RuntimeError, match="execution failure"):
            core.full_context_run(engine)
    else:
        report = core.full_context_run(engine)
        assert report["passed"] and report["prompt_tokens"] == 6
        assert report["peak_logical_persistent_state_bytes"] == 100
    assert engine._eos == frozenset({248044, 248046})
