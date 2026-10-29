from __future__ import annotations

import json

import pytest
from forge_llm.agents.native_tools import (
    NoNativeCall,
    build_tool_specs,
    parse_native_actions,
)
from forge_llm.chat_formats import (
    GEMMA4,
    QWEN35,
    GemmaSyntaxError,
    family_for,
    gemma_calls,
    incremental_suffix,
    split_thinking,
    thinking_budget,
    to_gemma_messages,
)

READ = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["path"],
    "additionalProperties": False,
}


def test_family_detection_and_effort_budgets():
    assert family_for("gemma4") is GEMMA4
    assert family_for("gemma4_text") is GEMMA4
    assert family_for("qwen3_5") is QWEN35
    assert family_for("llama").dialect == "json"
    assert thinking_budget("off") is None
    assert thinking_budget("high") == 0
    with pytest.raises(ValueError):
        thinking_budget("max")


def test_gemma_arguments_handle_nesting_and_braces_inside_strings():
    calls, rest = gemma_calls(
        'Checking.<|tool_call>call:read_file{limit:20,path:<|"|>a}b.py<|"|>,'
        "flags:[true,null,{x:1.5}]}<tool_call|>"
    )
    assert rest == "Checking."
    assert calls == [
        {
            "name": "read_file",
            "arguments": {
                "limit": 20,
                "path": "a}b.py",
                "flags": [True, None, {"x": 1.5}],
            },
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        '<|tool_call>call:read_file{path:<|"|>x<tool_call|>',
        "<|tool_call>call:read_file{path:x,path:y}<tool_call|>",
        "<|tool_call>read_file{}<tool_call|>",
        '<|tool_call>call:read_file{path:<|"|>x<|"|>}',
    ],
)
def test_gemma_rejects_malformed_calls(text):
    with pytest.raises(GemmaSyntaxError):
        gemma_calls(text)


def test_split_thinking_for_both_families():
    assert split_thinking("<|channel>thought\nhmm<channel|>Answer", GEMMA4) == (
        "Answer",
        "hmm",
        True,
    )
    assert split_thinking("<think>x</think>\n\nY", QWEN35) == ("Y", "x", True)
    assert split_thinking("<think>partial", QWEN35) == ("", "partial", False)
    assert split_thinking("Plain answer", QWEN35) == ("Plain answer", "", True)
    # A quoted marker inside the thinking doesn't end it.
    quoted = '<think>Qwen closes with "</think>" here.</think>\n\nAnswer'
    assert split_thinking(quoted, QWEN35) == (
        "Answer",
        'Qwen closes with "</think>" here.',
        True,
    )


def test_gemma_history_keeps_raw_arguments_and_names_tool_results():
    converted = to_gemma_messages(
        [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": 'Let me look.<|tool_call>call:read_file{path:<|"|>a.py<|"|>}<tool_call|>',
            },
            {"role": "tool", "content": "contents"},
        ]
    )
    call = converted[1]["tool_calls"][0]
    assert converted[1]["content"] == ""
    assert call["function"] == {"name": "read_file", "arguments": 'path:<|"|>a.py<|"|>'}
    assert converted[2] == {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": "read_file",
        "content": "contents",
    }


def qwen_like(messages, generation_prompt):
    """Mimics Qwen3.5: the empty think block appears only on the final turn."""
    out = []
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    for index, message in enumerate(messages):
        out.append(f"<|im_start|>{message['role']}\n")
        if message["role"] == "assistant" and index > last_user:
            out.append("<think>\n\n</think>\n\n")
        out.append(message["content"] + "<|im_end|>\n")
    if generation_prompt:
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)


def test_incremental_suffix_appends_only_new_turns():
    previous = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
    ]
    current = previous + [{"role": "user", "content": "U2"}]
    suffix = incremental_suffix(qwen_like, previous, current, "<|im_end|>")
    assert (
        suffix
        == "\n<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_incremental_suffix_refuses_edited_history_or_missing_stop():
    previous = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
    ]
    edited = [
        {"role": "user", "content": "U1 edited"},
        *previous[1:],
        {"role": "user", "content": "U2"},
    ]
    assert incremental_suffix(qwen_like, previous, edited, "<|im_end|>") is None
    assert incremental_suffix(qwen_like, previous, previous, "<|im_end|>") is None
    current = previous + [{"role": "user", "content": "U2"}]
    assert incremental_suffix(qwen_like, previous, current, "<missing>") is None


def specs():
    return build_tool_specs({"read_file": READ}, stable=True)


def test_gemma_dialect_parses_into_canonical_actions():
    actions = parse_native_actions(
        'Reading it.<|tool_call>call:read_file{path:<|"|>README.md<|"|>}<tool_call|>',
        specs(),
        dialect="gemma4",
    )
    assert actions == [
        {"action": "tool", "name": "read_file", "args": {"path": "README.md"}}
    ]
    with pytest.raises(Exception, match="not currently advertised"):
        parse_native_actions(
            "<|tool_call>call:delete_everything{}<tool_call|>",
            specs(),
            dialect="gemma4",
        )


@pytest.mark.parametrize("dialect", ["json", "qwen3_coder_xml", "gemma4"])
def test_plain_text_is_reported_as_no_call(dialect):
    with pytest.raises(NoNativeCall):
        parse_native_actions("The answer is 42.", specs(), dialect=dialect)


def test_stable_catalog_does_not_depend_on_current_agents():
    empty = build_tool_specs({"read_file": READ}, stable=True)
    busy = build_tool_specs(
        {"read_file": READ},
        child_ids=["agent-0001"],
        recipient_ids=["agent-0001"],
        wait_ids=["agent-0001"],
        stable=True,
    )
    assert json.dumps(empty, sort_keys=True) == json.dumps(busy, sort_keys=True)
    assert {spec["function"]["name"] for spec in empty} >= {"send", "wait", "finish"}
