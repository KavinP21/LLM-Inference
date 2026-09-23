from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from forge_llm.export import export_model
from forge_llm.model_contract import validate_model_weights
from forge_llm.model_file import ModelFile


def test_local_qwen2_checkpoint_still_exports_and_validates(tmp_path: Path) -> None:
    torch.manual_seed(13)
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "qwen-checkpoint"
    transformers.Qwen2ForCausalLM(config).save_pretrained(checkpoint)

    output = tmp_path / "qwen.engine"
    result = export_model(str(checkpoint), output)

    assert result["format_version"] == 2
    assert result["model_type"] == "qwen2"
    with ModelFile(output) as artifact:
        validate_model_weights(artifact)
        assert artifact.config.model_type == "qwen2"
        assert (
            artifact.tensor_info("lm_head.weight").offset
            == artifact.tensor_info("model.embed_tokens.weight").offset
        )


def test_local_gemma3_checkpoint_exports_and_validates(tmp_path: Path) -> None:
    torch.manual_seed(31)
    config = transformers.Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=4,
        sliding_window_pattern=2,
        query_pre_attn_scalar=8,
        tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "checkpoint"
    transformers.Gemma3ForCausalLM(config).save_pretrained(checkpoint)

    output = tmp_path / "gemma3.engine"
    result = export_model(str(checkpoint), output)

    assert result["format_version"] == 2
    assert result["model_type"] == "gemma3_text"
    assert output.with_suffix(".engine.json").is_file()
    with ModelFile(output) as artifact:
        validate_model_weights(artifact)
        assert artifact.config.model_type == "gemma3_text"
        assert artifact.config.attention_head_dim == 8
        assert artifact.config.is_sliding_layer(0)
        assert not artifact.config.is_sliding_layer(1)
        assert (
            artifact.tensor_info("lm_head.weight").offset
            == artifact.tensor_info("model.embed_tokens.weight").offset
        )
