from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from forge_llm.agents.native_tools import (
    build_tool_specs,
    parse_native_action,
    validate_tool_specs,
)
from forge_llm.agents.protocol import ProtocolError, parse_action

SCHEMAS = {
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 256},
            "start_line": {"type": "integer", "minimum": 1, "maximum": 10000},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "run_tests": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
                "uniqueItems": True,
            },
        },
        "required": ["paths"],
        "additionalProperties": False,
    },
}


def catalog(**kwargs):
    return build_tool_specs(
        SCHEMAS,
        child_ids=["child-1", "child-2"],
        recipient_ids=["parent", "child-1"],
        **kwargs,
    )


def call(name, args, *, wrapped=True):
    text = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return "<tool_call>\n" + text + "\n</tool_call>" if wrapped else text


@pytest.mark.parametrize("wrapped", [True, False])
def test_native_registered_call_becomes_exact_canonical_action(wrapped):
    result = parse_native_action(
        call("read_file", {"path": "calc.py", "max_lines": 20}, wrapped=wrapped),
        catalog(),
    )
    assert result == {
        "action": "tool",
        "name": "read_file",
        "args": {"path": "calc.py", "max_lines": 20},
    }


@pytest.mark.parametrize(
    "name,args,expected",
    [
        (
            "spawn",
            {
                "task": "Check arithmetic",
                "role": None,
                "dependencies": ["child-1"],
                "tools": ["read_file"],
            },
            {
                "action": "spawn",
                "task": "Check arithmetic",
                "dependencies": ["child-1"],
                "tools": ["read_file"],
            },
        ),
        (
            "send",
            {"to": "parent", "message": "The measured total is 17."},
            {"action": "send", "to": "parent", "message": "The measured total is 17."},
        ),
        ("wait", {"agents": ["child-1"]}, {"action": "wait", "agents": ["child-1"]}),
        (
            "finish",
            {"result": "Answer: 17."},
            {"action": "finish", "result": "Answer: 17."},
        ),
    ],
)
def test_native_control_translation_preserves_protocol_contract(name, args, expected):
    assert parse_native_action(call(name, args), catalog()) == expected


@pytest.mark.parametrize(
    "name,args",
    [
        ("spawn", {"task": "Check", "agent_id": "made-up"}),
        ("spawn", {"task": "Check", "dependencies": ["made-up"]}),
        ("spawn", {"task": "Check", "tools": ["write_file"]}),
        ("spawn", {"task": "Check", "tools": ["read_file", "read_file"]}),
        ("spawn", {"task": "Check", "dependencies": "child-1"}),
        ("send", {"to": "made-up", "message": "hello"}),
        ("wait", {"agents": ["parent"]}),
        ("wait", {"agents": []}),
        ("finish", {"result": 42}),
        ("read_file", {"path": "calc.py", "start_line": True}),
        ("read_file", {"path": "calc.py", "max_lines": 0}),
        ("read_file", {"path": "calc.py", "start_line": 1.0}),
        ("read_file", {"path": "calc.py", "python_expression": "open('secret')"}),
        ("read_file", {"path": ""}),
        ("run_tests", {"paths": ["test.py", "test.py"]}),
    ],
)
def test_calls_cannot_invent_arguments_ids_or_tool_grants(name, args):
    with pytest.raises(ProtocolError):
        parse_native_action(
            call(name, args), catalog(allowed_tools=["read_file", "run_tests"])
        )


@pytest.mark.parametrize(
    "text",
    [
        'I will read the file. <tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>',
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call> Done.',
        '```json\n{"name":"read_file","arguments":{"path":"a"}}\n```',
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}}',
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call><tool_call>{"name":"finish","arguments":{"result":"done"}}</tool_call>',
        '{"name":"read_file","arguments":{"path":"a"}} {"name":"finish","arguments":{"result":"done"}}',
        '{"name":"read_file","arguments":"{\\"path\\":\\"a\\"}"}',
        '{"name":"read_file","arguments":{"path":"a"},"id":"call_1"}',
        '{"action":"read_file","args":{"path":"a"}}',
        '[{"name":"read_file","arguments":{"path":"a"}}]',
        '{"name":"read_file","name":"finish","arguments":{"result":"done"}}',
        '{"name":"read_file","arguments":{"path":"a","path":"b"}}',
        '{"name":"read_file","arguments":{"path":"a","max_lines":NaN}}',
        '{"name":"read_file","arguments":{"path":"a","max_lines":1e999}}',
    ],
)
def test_no_prose_fuzzy_extraction_multiple_calls_or_invalid_json(text):
    with pytest.raises(ProtocolError):
        parse_native_action(text, catalog())


def test_actual_action_menu_hides_controls_and_ungranted_tools():
    specs = build_tool_specs(SCHEMAS, allowed_tools=["read_file"], controls=["finish"])
    assert [s["function"]["name"] for s in specs] == ["read_file", "finish"]
    for name, args in (
        ("run_tests", {"paths": ["test.py"]}),
        ("spawn", {"task": "check"}),
        ("send", {"to": "parent", "message": "hello"}),
    ):
        with pytest.raises(ProtocolError, match="not currently advertised"):
            parse_native_action(call(name, args), specs)


def test_empty_actual_id_sets_allow_only_empty_spawn_dependencies_and_grants():
    specs = build_tool_specs({}, controls=["spawn", "send", "wait", "finish"])
    assert [s["function"]["name"] for s in specs] == ["spawn", "finish"]
    assert parse_native_action(
        call("spawn", {"task": "check", "dependencies": [], "tools": []}), specs
    ) == {"action": "spawn", "task": "check", "dependencies": [], "tools": []}
    with pytest.raises(ProtocolError):
        parse_native_action(
            call("spawn", {"task": "check", "dependencies": ["fake"]}), specs
        )


def test_wait_catalog_uses_only_undelivered_direct_children():
    specs = catalog(wait_ids=["child-2"])
    with pytest.raises(ProtocolError):
        parse_native_action(call("wait", {"agents": ["child-1"]}), specs)
    assert parse_native_action(call("wait", {"agents": ["child-2"]}), specs)[
        "agents"
    ] == ["child-2"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_tools": ["unregistered"]},
        {"controls": ["execute"]},
        {"wait_ids": ["parent"]},
        {"child_ids": "child-1"},
        {"recipient_ids": ["a", "a"]},
    ],
)
def test_builder_rejects_bad_runtime_configuration(kwargs):
    with pytest.raises(ValueError):
        build_tool_specs(SCHEMAS, **kwargs)


def test_normalized_catalog_is_not_mutated_by_schema_source_changes():
    schemas = copy.deepcopy(SCHEMAS)
    specs = build_tool_specs(schemas, controls=[])
    schemas["read_file"]["properties"]["path"]["type"] = "integer"
    assert parse_native_action(call("read_file", {"path": "a"}), specs)["args"] == {
        "path": "a"
    }


def test_large_integer_fails_protocol_bounds_without_float_overflow():
    with pytest.raises(ProtocolError):
        parse_native_action(
            call("read_file", {"path": "a", "start_line": 10**400}), catalog()
        )


def test_unicode_and_literal_tags_are_data_inside_one_call():
    result = "计算结果：17，字符串 <tool_call>sample</tool_call>。"
    assert (
        parse_native_action(call("finish", {"result": result}), catalog())["result"]
        == result
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda spec: spec.update(executable="run_me"),
        lambda spec: spec["function"].update(callable="eval"),
        lambda spec: spec["function"].update(description=17),
        lambda spec: spec["function"].update(name="read file"),
        lambda spec: spec["function"]["parameters"].update(additionalProperties=True),
        lambda spec: spec["function"]["parameters"].update(type=["object", {}]),
        lambda spec: spec["function"]["parameters"].update(required=["missing"]),
        lambda spec: spec["function"]["parameters"].update(custom_validator="eval"),
        lambda spec: spec["function"]["parameters"]["properties"]["path"].update(
            pattern="["
        ),
        lambda spec: spec["function"]["parameters"]["properties"]["start_line"].update(
            minimum="bad"
        ),
    ],
)
def test_worker_schema_validator_rejects_open_or_malformed_definitions(change):
    specs = build_tool_specs(SCHEMAS, controls=[])
    change(specs[0])
    with pytest.raises(ValueError):
        validate_tool_specs(specs)


def test_schema_transport_count_bytes_finiteness_and_copy_bounds():
    specs = build_tool_specs(SCHEMAS, controls=[])
    normalized = validate_tool_specs(specs)
    normalized[0]["function"]["description"] = "changed"
    assert specs[0]["function"]["description"] != "changed"
    with pytest.raises(ValueError):
        validate_tool_specs([specs[0]] * 33)
    with pytest.raises(ValueError):
        validate_tool_specs([specs[0], specs[0]])
    specs[0]["function"]["description"] = "汉" * 30000
    with pytest.raises(ValueError, match="64 KiB"):
        validate_tool_specs(specs)
    specs = build_tool_specs(SCHEMAS, controls=[])
    specs[0]["function"]["parameters"]["properties"]["start_line"]["default"] = float(
        "inf"
    )
    with pytest.raises(ValueError, match="finite"):
        validate_tool_specs(specs)


def test_local_qwen_pretrained_native_template_and_decode_preserve_calls():
    directory = Path(
        "/Users/kavinprabhakar/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
    )
    if not (directory / "tokenizer_config.json").is_file():
        pytest.skip("optional cached Qwen tokenizer integration")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    specs = build_tool_specs(SCHEMAS, controls=["finish"])
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Read calc.py."}],
        tools=specs,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<tools>" in prompt and "<tool_call>" in prompt
    assert '"name": "read_file"' in prompt
    text = call("read_file", {"path": "calc.py"}) + "<|im_end|>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded.strip() == call("read_file", {"path": "calc.py"})
    assert parse_native_action(decoded, specs)["args"] == {"path": "calc.py"}


def test_native_finish_accepts_nested_json_object_and_preserves_legacy_result_string():
    result = {
        "summary": "总计：17",
        "evidence": [{"path": "prices.csv", "rows": 3}],
        "verified": True,
        "missing": None,
        "metrics": {"value": 17.5},
    }
    action = parse_native_action(call("finish", {"result": result}), catalog())
    assert action == {
        "action": "finish",
        "result": json.dumps(result, ensure_ascii=False, allow_nan=False),
    }
    assert json.loads(action["result"]) == result
    schema = next(spec for spec in catalog() if spec["function"]["name"] == "finish")
    assert schema["function"]["parameters"]["properties"]["result"]["type"] == [
        "string",
        "object",
    ]
    assert (
        parse_native_action(call("finish", {"result": {}}), catalog())["result"] == "{}"
    )


@pytest.mark.parametrize("result", [42, 17.5, True, None, [1, 2], "", "   "])
def test_native_finish_rejects_scalar_array_and_empty_string_results(result):
    with pytest.raises(ProtocolError):
        parse_native_action(call("finish", {"result": result}), catalog())


@pytest.mark.parametrize("result", [42, {"answer": 17}, [1, 2]])
def test_legacy_finish_remains_strict_string_only(result):
    with pytest.raises(ProtocolError):
        parse_action(json.dumps({"action": "finish", "result": result}))


def test_native_finish_object_canonical_length_bound_and_escaping():
    accepted = {"value": "a" * 23987}  # Default JSON separators add 13 characters.
    assert len(json.dumps(accepted)) == 24000
    assert (
        len(
            parse_native_action(call("finish", {"result": accepted}), catalog())[
                "result"
            ]
        )
        == 24000
    )
    with pytest.raises(ProtocolError, match="canonical characters"):
        parse_native_action(
            call("finish", {"result": {"value": "a" * 23988}}), catalog()
        )
    # Escaping the normalized JSON text for the internal legacy validation can
    # exceed the native wire limit even when both native and canonical bounds fit.
    quote_heavy = {str(i): '"' * 5 for i in range(1000)}
    native = call("finish", {"result": quote_heavy})
    normalized = json.dumps(quote_heavy, ensure_ascii=False, allow_nan=False)
    assert len(native) <= 32768 and len(normalized) <= 24000
    assert len(json.dumps({"action": "finish", "result": normalized})) > 32768
    assert json.loads(parse_native_action(native, catalog())["result"]) == quote_heavy


def test_native_finish_object_depth_and_nonfinite_values_fail_closed():
    nested = {}
    for _ in range(34):
        nested = {"nested": nested}
    with pytest.raises(ProtocolError, match="nesting"):
        parse_native_action(call("finish", {"result": nested}), catalog())
    with pytest.raises(ProtocolError, match="finite"):
        parse_native_action(
            '{"name":"finish","arguments":{"result":{"metrics":{"value":1e999}}}}',
            catalog(),
        )
    with pytest.raises(ProtocolError):
        parse_native_action(
            '{"name":"finish","arguments":{"result":{"answer":17},"extra":"bad"}}',
            catalog(),
        )


def test_native_batch_preserves_two_typed_independent_spawn_calls():
    from forge_llm.agents.native_tools import parse_native_actions

    text = (
        call("spawn", {"task": "Inspect shipping.md", "tools": ["read_file"]})
        + "\n"
        + call("spawn", {"task": "Inspect pricing.md", "role": "reviewer"})
    )
    actions = parse_native_actions(text, catalog())
    assert actions == [
        {"action": "spawn", "task": "Inspect shipping.md", "tools": ["read_file"]},
        {"action": "spawn", "task": "Inspect pricing.md", "role": "reviewer"},
    ]
    with pytest.raises(ProtocolError):
        parse_native_action(text, catalog())  # Existing single-call API stays strict.


def test_native_batch_supports_raw_single_call_and_literal_tags_in_json_strings():
    from forge_llm.agents.native_tools import parse_native_actions

    result = 'Literal </tool_call><tool_call> tags and escaped "quoted" text.'
    text = (
        call("finish", {"result": result})
        + "\n\t"
        + call("read_file", {"path": "data.txt"})
    )
    actions = parse_native_actions(text, catalog())
    assert actions[0] == {"action": "finish", "result": result}
    assert actions[1] == {
        "action": "tool",
        "name": "read_file",
        "args": {"path": "data.txt"},
    }
    assert parse_native_actions(
        call("read_file", {"path": "data.txt"}, wrapped=False), catalog()
    ) == [actions[1]]
    # Validation returns typed calls only; the runtime decides permitted batch
    # combinations and rejects finish mixed with another action before effects.


@pytest.mark.parametrize(
    "second",
    [
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}</tool_call>',
        call("read_file", {"path": "a", "invented_argument": 1}),
        call("spawn", {"task": "check", "tools": ["ungranted_tool"]}),
        call("unknown_tool", {}),
        call("read_file", {"path": "a", "start_line": True}),
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}} extraJSON </tool_call>',
        "Some prose " + call("read_file", {"path": "a"}),
        call("read_file", {"path": "a"}) + " trailing prose",
    ],
)
def test_native_batch_invalid_late_call_returns_no_partial_list(second):
    from forge_llm.agents.native_tools import parse_native_actions

    text = call("read_file", {"path": "first.txt"}) + second
    returned = None
    with pytest.raises(ProtocolError):
        returned = parse_native_actions(text, catalog())
    assert returned is None


def test_native_batch_count_and_whole_response_size_are_bounded():
    from forge_llm.agents.native_tools import parse_native_actions

    one = call("read_file", {"path": "a"})
    assert len(parse_native_actions(one * 4, catalog())) == 4
    with pytest.raises(ProtocolError, match="allowance"):
        parse_native_actions(one * 5, catalog())
    with pytest.raises(ProtocolError, match="allowance"):
        parse_native_actions(one * 2, catalog(), max_calls=1)
    with pytest.raises(ProtocolError, match="at most"):
        parse_native_actions(one * 2, catalog(), max_chars=len(one))
    for bad in (0, 5, True, 1.5):
        with pytest.raises(ValueError):
            parse_native_actions(one, catalog(), max_calls=bad)


@pytest.mark.parametrize(
    "second",
    [
        '<tool_call>{"name":"finish","arguments":{"result":{"value":1e999}}}</tool_call>',
        '<tool_call>{"name":"read_file","name":"finish","arguments":{"result":"bad"}}</tool_call>',
        '<tool_call>{"name":"finish","arguments":{"result":{"nested":NaN}}}</tool_call>',
    ],
)
def test_native_batch_rejects_nonfinite_and_duplicate_json_in_late_calls(second):
    from forge_llm.agents.native_tools import parse_native_actions

    with pytest.raises(ProtocolError):
        parse_native_actions(call("read_file", {"path": "a"}) + second, catalog())


def test_native_batch_bounded_depth_applies_to_generic_tool_arguments_too():
    from forge_llm.agents.native_tools import parse_native_actions

    schemas = {
        "inspect_json": {
            "type": "object",
            "properties": {
                "payload": {"type": "object", "additionalProperties": True},
            },
            "required": ["payload"],
            "additionalProperties": False,
        }
    }
    specs = build_tool_specs(schemas, controls=[])
    nested = {}
    for _ in range(36):
        nested = {"nested": nested}
    with pytest.raises(ProtocolError, match="nesting"):
        parse_native_actions(
            call("inspect_json", {"payload": {}})
            + call("inspect_json", {"payload": nested}),
            specs,
        )


@pytest.mark.parametrize("keyword", ["enum", "const"])
@pytest.mark.parametrize(
    "actual", [{"total": True}, {"total": [True]}, {"total": {"value": True}}]
)
def test_schema_recursive_equality_keeps_booleans_distinct_from_numbers(
    keyword, actual
):
    from forge_llm.agents.native_tools import _validate

    expected = (
        {"total": 1}
        if type(actual["total"]) is bool
        else (
            {"total": [1]}
            if isinstance(actual["total"], list)
            else {"total": {"value": 1}}
        )
    )
    schema = {keyword: [expected] if keyword == "enum" else expected}
    with pytest.raises(ProtocolError):
        _validate(actual, schema, "final answer")
    _validate(expected, schema, "final answer")


@pytest.mark.parametrize("keyword", ["enum", "const"])
def test_schema_recursive_numeric_equality_and_object_order_follow_json_semantics(
    keyword,
):
    from forge_llm.agents.native_tools import _validate

    expected = {"numbers": [1, {"zero": 0}], "verified": True}
    equivalent = {"verified": True, "numbers": [1.0, {"zero": -0.0}]}
    schema = {keyword: [expected] if keyword == "enum" else expected}
    _validate(equivalent, schema)
    with pytest.raises(ProtocolError):
        _validate({"numbers": [1, {"zero": 0}], "verified": 1}, schema)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_schema_validation_checks_nonfinite_data_inside_unspecified_properties(number):
    from forge_llm.agents.native_tools import _validate

    schema = {"type": "object", "additionalProperties": True}
    with pytest.raises(ProtocolError, match="finite"):
        _validate({"unexpected": [{"numeric": number}]}, schema, "final answer")


def test_schema_unique_items_uses_the_same_json_equality():
    from forge_llm.agents.native_tools import _validate

    schema = {"type": "array", "uniqueItems": True}
    _validate([True, 1], schema)
    _validate([{"nested": False}, {"nested": 0}], schema)
    with pytest.raises(ProtocolError, match="duplicate"):
        _validate([1, 1.0], schema)
    with pytest.raises(ProtocolError, match="duplicate"):
        _validate([{"a": 1, "b": True}, {"b": True, "a": 1.0}], schema)


def test_schema_validation_non_json_or_cyclic_python_data_fails_closed():
    from forge_llm.agents.native_tools import _validate

    cyclic = []
    cyclic.append(cyclic)
    for value in (cyclic, {1: "not a string key"}, {"value": b"not JSON"}):
        with pytest.raises(ProtocolError):
            _validate(value, {})
