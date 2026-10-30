from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from instinct.format import ModelConfig, write_engine
from instinct.mlx_engine import MlxEngine


def _tiny_gemma3_artifact(path: Path):
    torch.manual_seed(17)
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        sliding_window=4,
        sliding_window_pattern=2,
        query_pre_attn_scalar=32,
        rms_norm_eps=1e-6,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    reference = transformers.Gemma3ForCausalLM(config).eval()
    tensors = {
        name: tensor.detach().cpu().numpy().astype(np.float16)
        for name, tensor in reference.state_dict().items()
    }
    reference.load_state_dict(
        {
            name: torch.from_numpy(array.astype(np.float32))
            for name, array in tensors.items()
        }
    )
    write_engine(
        path,
        ModelConfig.from_huggingface(config),
        tensors,
        aliases={"lm_head.weight": "model.embed_tokens.weight"},
    )
    return reference


def _reference_logits(model, tokens: list[int]) -> np.ndarray:
    with torch.inference_mode():
        output = model(torch.tensor([tokens], dtype=torch.long)).logits[0, -1]
    return output.detach().cpu().numpy().astype(np.float32)


def _reference_generate(model, prompt: list[int], count: int) -> list[int]:
    output = list(prompt)
    for _ in range(count):
        output.append(int(np.argmax(_reference_logits(model, output))))
    return output


@pytest.fixture
def tiny_gemma3(tmp_path: Path):
    path = tmp_path / "tiny-gemma3.engine"
    return path, _tiny_gemma3_artifact(path)


def _engine(path, **options):
    settings = dict(
        max_num_sequences=2,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=3,
        prefill_burst_size=3,
        page_size=4,
    )
    settings.update(options)
    return MlxEngine(path, **settings)


def test_gemma3_prefill_matches_transformers(tiny_gemma3) -> None:
    path, reference = tiny_gemma3
    prompt = [2, 7, 11, 5, 19, 23, 3, 29, 31, 13]
    expected = _reference_logits(reference, prompt)
    with _engine(path) as engine:
        actual = engine.prefill_logits(prompt)
        assert engine.stats()["model_type"] == "gemma3_text"
    cosine = float(
        np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    assert cosine >= 0.999
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)


def test_gemma3_greedy_decode_matches_transformers(tiny_gemma3) -> None:
    path, reference = tiny_gemma3
    prompts = [[2, 7, 11, 5, 19, 23], [2, 3, 5, 7, 9, 11, 13, 17, 19]]
    expected = [_reference_generate(reference, p, 6)[len(p) :] for p in prompts]
    with _engine(path) as engine:
        ids = [engine.submit(p, 6) for p in prompts]
        while engine.has_work():
            engine.step()
        assert [engine.request(i).output for i in ids] == expected
        assert all(p["used_pages"] == 0 for p in engine.stats()["kv_pools"].values())
