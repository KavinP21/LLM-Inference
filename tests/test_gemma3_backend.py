from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from forge_llm.backends.gemma3 import MlxGemma3Model
from forge_llm.format import ModelConfig, write_engine
from forge_llm.mlx_engine import MlxEngine


def _tiny_gemma3_artifact(path: Path):
    torch.manual_seed(17)
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        sliding_window_pattern=2,
        query_pre_attn_scalar=8,
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


def test_gemma3_prefill_matches_transformers(tiny_gemma3) -> None:
    path, reference = tiny_gemma3
    prompt = [2, 7, 11, 5, 19, 23, 3, 29, 31, 13]
    expected = _reference_logits(reference, prompt)
    with MlxEngine(
        path,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=4,
        custom_metal=False,
    ) as engine:
        actual = np.asarray(engine.debug_prefill_logits(prompt), dtype=np.float32)
        assert isinstance(engine.model, MlxGemma3Model)
        assert engine.stats()["model_type"] == "gemma3_text"

    cosine = float(
        np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    assert cosine >= 0.999
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)


def test_gemma3_greedy_decode_matches_transformers(tiny_gemma3) -> None:
    path, reference = tiny_gemma3
    prompt = [2, 7, 11, 5, 19, 23]
    expected = _reference_generate(reference, prompt, 6)[len(prompt) :]
    with MlxEngine(
        path,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=3,
        custom_metal=False,
    ) as engine:
        actual = engine.generate(prompt, max_new_tokens=6)
        stats = engine.stats()

    assert actual == expected
    assert stats["kv_cache"]["allocated_blocks"] == 0
    assert stats["kv_device_bytes"] == 0


def test_gemma3_custom_metal_matches_fallback_and_batches(tiny_gemma3) -> None:
    path, _ = tiny_gemma3
    prompts = [[2, 4, 6, 8, 10], [2, 3, 5, 7, 9, 11, 13]]
    with MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=3,
        custom_metal=False,
    ) as fallback:
        expected = [fallback.generate(prompt, 4) for prompt in prompts]

    with MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=3,
        custom_metal=True,
        metal_paged_attention=True,
    ) as custom:
        request_ids = [custom.submit(prompt, 4) for prompt in prompts]
        while any(
            custom.scheduler.request(request_id).state.value != "completed"
            for request_id in request_ids
        ):
            custom.step()
        actual = [
            custom.scheduler.request(request_id).output for request_id in request_ids
        ]
        stats = custom.stats()

    assert actual == expected
    assert stats["kv_cache"]["allocated_blocks"] == 0
    assert stats["kv_device_bytes"] == 0
