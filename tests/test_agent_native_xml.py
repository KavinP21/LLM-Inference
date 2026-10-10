"""Qwen3-Coder calls preserve typed arguments and the closed action boundary."""

import json

import pytest
from forge_llm.agents.native_tools import (
    build_tool_specs,
    parse_native_action,
    parse_native_actions,
)
from forge_llm.agents.protocol import ProtocolError


@pytest.fixture
def specs():
    return build_tool_specs(
        {
            "edit_lines": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "minimum": 1},
                    "text": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["path", "start", "text"],
                "additionalProperties": False,
            },
            "read_file": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        child_ids=["child-1"],
        recipient_ids=["child-1"],
    )


def call(name, parameters):
    args = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in parameters)
    return f"<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>"


def test_typed_xml_preserves_multiline_python_indentation(specs):
    source = "def mean(values):\n    if not values:\n        raise ValueError('empty')\n    return sum(values) / len(values)\n"
    action = parse_native_action(
        call(
            "edit_lines",
            [
                ("path", "stats.py"),
                ("start", "1"),
                ("text", source),
                ("dry_run", "False"),
            ],
        ),
        specs,
    )
    assert action == {
        "action": "tool",
        "name": "edit_lines",
        "args": {"path": "stats.py", "start": 1, "text": source, "dry_run": False},
    }


def test_array_null_and_structured_finish_follow_catalog_types(specs):
    spawn = parse_native_action(
        call(
            "spawn",
            [
                ("task", "Review the source."),
                ("role", "None"),
                ("tools", '["read_file"]'),
            ],
        ),
        specs,
    )
    assert spawn["tools"] == ["read_file"]
    assert "role" not in spawn
    wait = parse_native_action(call("wait", [("agents", '["child-1"]')]), specs)
    assert wait == {"action": "wait", "agents": ["child-1"]}
    finished = parse_native_action(
        call("finish", [("result", '{"passed":false,"count":2}')]), specs
    )
    assert json.loads(finished["result"]) == {"passed": False, "count": 2}


def test_xml_batch_and_existing_json_calls_keep_same_canonical_actions(specs):
    xml = (
        call("read_file", [("path", "one.py")])
        + "\n"
        + call("read_file", [("path", "two.py")])
    )
    actions = parse_native_actions(xml, specs)
    legacy = '<tool_call>{"name":"read_file","arguments":{"path":"one.py"}}</tool_call>'
    assert actions[0] == parse_native_action(legacy, specs)
    assert actions[1]["args"] == {"path": "two.py"}


@pytest.mark.parametrize(
    "wire",
    [
        call("missing", []),
        call("read_file", [("path", "one.py"), ("path", "two.py")]),
        call("read_file", [("extra", "one.py")]),
        call("read_file", []),
        call("read_file", [("path", "one.py")])[:-5],
        "<tool_call><function=read_file><parameter=path>one.py</function></tool_call>",
        "<tool_call><function=read_file><parameter=path>one.py</parameter></wrong></tool_call>",
        "<tool_call><function=read_file>prose<parameter=path>one.py</parameter></function></tool_call>",
        call(
            "read_file",
            [("path", "<!DOCTYPE x [<!ENTITY steal SYSTEM 'file:///private'>]>")],
        ),
        call("read_file", [("path", "<parameter=other>value</parameter>")]),
        "thinking\n" + call("read_file", [("path", "one.py")]),
        "<think>" + call("read_file", [("path", "one.py")]) + "</think>",
        call("read_file", [("path", "one.py")]) + " suffix",
    ],
)
def test_unknown_duplicate_incomplete_or_prose_xml_is_rejected(specs, wire):
    with pytest.raises(ProtocolError):
        parse_native_actions(wire, specs)


@pytest.mark.parametrize(
    "key,value",
    [("start", "true"), ("start", "1.5"), ("start", "0"), ("dry_run", "yes")],
)
def test_scalar_coercion_cannot_bypass_schema(specs, key, value):
    parameters = {"path": "one.py", "start": "1", "text": "pass"}
    parameters[key] = value
    with pytest.raises(ProtocolError):
        parse_native_action(call("edit_lines", parameters.items()), specs)


def test_bad_json_object_and_duplicate_json_keys_are_not_recovered_as_strings(specs):
    for value in ('{"a":1,"a":2}', '{"passed":NaN}', "{malformed}"):
        with pytest.raises(ProtocolError):
            parse_native_action(call("finish", [("result", value)]), specs)


def test_literal_entities_in_string_are_not_expanded(specs):
    action = parse_native_action(call("read_file", [("path", "a&amp;b.py")]), specs)
    assert action["args"]["path"] == "a&amp;b.py"


def test_unwrapped_function_block_is_not_a_native_call(specs):
    with pytest.raises(ProtocolError):
        parse_native_action(
            "<function=read_file><parameter=path>one.py</parameter></function>", specs
        )


def test_batch_validates_every_call_and_obeys_bounds(specs):
    valid = call("read_file", [("path", "one.py")])
    invalid = call("read_file", [("path", "two.py"), ("path", "three.py")])
    with pytest.raises(ProtocolError):
        parse_native_actions(valid + invalid, specs)
    with pytest.raises(ProtocolError):
        parse_native_actions(valid * 5, specs)


def test_actual_qwen35_explanatory_finish_prefix_is_explicitly_supported(specs):
    # Shape and wording from the retained delegated-document model_response.
    prefix = (
        "I have both files analyzed. Let me synthesize the required JSON:\n\n"
        "**From report_a.md:**\n- Baseline throughput: 32 tokens/s\n"
        "- Candidate (eligible) throughput: 64 tokens/s\n\n"
        "**From report_b.md:**\n- Observed throughput: 128 tokens/s\n"
        "- Excluded observation: Yes (unrelated GPU job overlapped)\n"
        "- Energy measured: No\n- Release-ready: No\n\n"
    )
    payload = '{"eligible_throughput":64,"baseline_throughput":32,"speedup":2.0,"release_ready":false,"energy_measured":false}'
    wire = prefix + call("finish", [("result", payload)])
    with pytest.raises(ProtocolError):
        parse_native_actions(wire, specs)
    parsed = parse_native_actions(wire, specs, allow_xml_text=True)
    assert parsed == [parse_native_action(call("finish", [("result", payload)]), specs)]
    assert json.loads(parsed[0]["result"])["release_ready"] is False
    assert parse_native_action(wire, specs, allow_xml_text=True) == parsed[0]


def test_opt_in_plain_text_surrounds_only_fully_validated_xml_frames(specs):
    first = call("read_file", [("path", "report_a.md")])
    second = call("read_file", [("path", "report_b.md")])
    wire = (
        "I will inspect both reports.\n"
        + first
        + "\nThen inspect the second report.\n"
        + second
        + "\nThese are the requested calls."
    )
    assert parse_native_actions(
        wire, specs, allow_xml_text=True
    ) == parse_native_actions(first + second, specs)


@pytest.mark.parametrize(
    "wire",
    [
        'Explanation\n{"name":"read_file","arguments":{"path":"one.py"}}',
        'Explanation\n<tool_call>{"name":"read_file","arguments":{"path":"one.py"}}</tool_call>',
        '<tool_call>{"name":"read_file","arguments":{"path":"one.py"}}</tool_call>\nExplanation',
        "Explanation\n<tool_call><function=read_file><parameter=path>one.py</parameter></function>",
        "Explanation\n<function=read_file><parameter=path>one.py</parameter></function>",
        "Explanation\n<tool_call><function=read_file><parameter=path>one.py</function></tool_call>",
        "Explanation\n<tool_call><function=read_file><parameter=path>one.py</parameter><tool_call></function></tool_call>",
        "<think>Explanation</think>\n" + call("read_file", [("path", "one.py")]),
        "```xml\n" + call("read_file", [("path", "one.py")]) + "\n```",
        "An example is `" + call("read_file", [("path", "one.py")]) + "`.",
        call("read_file", [("path", "one.py")]) + "\n</tool_call>",
        call("read_file", [("path", "one.py")]) + "\n<tool_call",
        "Explanation\n" + call("missing", []),
        "Explanation\n" + call("read_file", [("path", "one.py"), ("path", "two.py")]),
    ],
)
def test_xml_text_option_does_not_recover_json_markup_or_malformed_calls(specs, wire):
    with pytest.raises(ProtocolError):
        parse_native_actions(wire, specs, allow_xml_text=True)


def test_text_mode_keeps_whole_response_and_call_budgets(specs):
    valid = call("read_file", [("path", "one.py")])
    with pytest.raises(ProtocolError):
        parse_native_actions(
            "x" * 100 + valid, specs, max_chars=100, allow_xml_text=True
        )
    with pytest.raises(ProtocolError):
        parse_native_actions("Explanation\n" + valid * 5, specs, allow_xml_text=True)
    with pytest.raises(ValueError):
        parse_native_actions(valid, specs, allow_xml_text="True")
    strict_json = '{"name":"read_file","arguments":{"path":"one.py"}}'
    assert parse_native_actions(
        strict_json, specs, allow_xml_text=True
    ) == parse_native_actions(strict_json, specs)
