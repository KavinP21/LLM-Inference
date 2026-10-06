import numpy as np
import pytest
from forge_llm.validate_int8 import compare_logits, quality_gates


def test_logit_comparison_reports_real_argmax_and_margin():
    metric = compare_logits(np.array([1.0, 2.0, 1.9]), np.array([1.0, 1.8, 2.1]))
    assert not metric["argmax_equal"]
    assert metric["reference_margin"] == pytest.approx(0.1)
    assert metric["max_absolute_error"] == pytest.approx(0.2)
    assert metric["reference_top_two"][0]["token"] == 1
    assert metric["actual_top_two"][0]["token"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("minimum_layer_cosine", 0.998),
        ("minimum_logit_cosine", 0.998),
        ("minimum_metal_dequantize_logit_cosine", 0.998),
        ("exact_greedy_cases", 24),
        ("metal_dequantize_exact_cases", 24),
    ],
)
def test_failed_quality_gates_are_not_silently_accepted(field, value):
    summary = {
        "minimum_layer_cosine": 1.0,
        "minimum_logit_cosine": 1.0,
        "minimum_metal_dequantize_logit_cosine": 1.0,
        "exact_greedy_cases": 25,
        "metal_dequantize_exact_cases": 25,
        "total_cases": 25,
    }
    assert quality_gates(summary, True)["strict_checkpoint_passed"]
    summary[field] = value
    assert not quality_gates(summary, True)["strict_checkpoint_passed"]
    with pytest.raises(ValueError):
        quality_gates(summary, True, cosine_gate=0.99)


def test_cache_leak_fails_an_otherwise_exact_run():
    summary = {
        "minimum_layer_cosine": 1.0,
        "minimum_logit_cosine": 1.0,
        "minimum_metal_dequantize_logit_cosine": 1.0,
        "exact_greedy_cases": 25,
        "metal_dequantize_exact_cases": 25,
        "total_cases": 25,
    }
    assert not quality_gates(summary, False)["strict_checkpoint_passed"]
