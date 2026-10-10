"""Closed native function calls for pretrained tool-aware chat templates.

Definitions use Hugging Face/OpenAI function schemas. Calls become the existing
agent actions only after validating the currently advertised function catalog.
No prose, fuzzy names, executable expressions, or multiple calls are recovered.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .protocol import ProtocolError, _unique_object, parse_action

_CONTROLS = frozenset({"spawn", "send", "wait", "finish"})
_RESERVED = _CONTROLS | {"tool"}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "description",
        "title",
        "default",
        "$schema",
        "pattern",
    }
)
_TYPES = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)


def _collection(
    values: Collection[str], field: str, *, names: bool = False
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a collection of identifiers")  # noqa: TRY004 - JSON schema errors share a ValueError contract.
    identifiers = list(values)
    if any(not isinstance(v, str) or not v or len(v) > 128 for v in identifiers):
        raise ValueError(f"{field} contains an invalid identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field} contains duplicate identifiers")
    if names and any(not _NAME.fullmatch(v) for v in identifiers):
        raise ValueError(
            "native function names require letters/digits/underscores/hyphens and at most 64 characters"
        )
    return sorted(identifiers)


def _check_schema(schema: Any, *, depth: int = 0) -> None:
    if not isinstance(schema, dict) or depth > 32:
        raise ValueError("native tools require bounded JSON parameter schemas")
    unsupported = schema.keys() - _SCHEMA_KEYS
    if unsupported:
        raise ValueError(f"unsupported native schema keywords: {sorted(unsupported)}")
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if (
            not isinstance(kinds, list)
            or not kinds
            or any(not isinstance(k, str) or k not in _TYPES for k in kinds)
        ):
            raise ValueError("native schema contains an unsupported type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or any(
        not isinstance(k, str) for k in properties
    ):
        raise ValueError("schema properties must be a string-keyed object")
    for nested in properties.values():
        _check_schema(nested, depth=depth + 1)
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(k, str) for k in required)
        or len(set(required)) != len(required)
    ):
        raise ValueError("schema required fields must be unique strings")
    if set(required) - properties.keys():
        raise ValueError("schema requires fields absent from its properties")
    if "items" in schema:
        _check_schema(schema["items"], depth=depth + 1)
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        _check_schema(additional, depth=depth + 1)
    elif type(additional) is not bool:
        raise ValueError("additionalProperties must be boolean or a schema")
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = schema[key]
            if not isinstance(options, list) or not options:
                raise ValueError(f"{key} requires at least one schema")
            for option in options:
                _check_schema(option, depth=depth + 1)
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and (
            type(schema[key]) not in (int, float)
            or (type(schema[key]) is float and not math.isfinite(schema[key]))
        ):
            raise ValueError(f"{key} must be finite numeric")
    for low, high in (
        ("minItems", "maxItems"),
        ("minLength", "maxLength"),
        ("minimum", "maximum"),
    ):
        if low in schema and high in schema and schema[low] > schema[high]:
            raise ValueError(f"{low} exceeds {high}")
    if "uniqueItems" in schema and type(schema["uniqueItems"]) is not bool:
        raise ValueError("uniqueItems must be boolean")
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise ValueError("enum must be a nonempty list")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError("pattern must be a string")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise ValueError("schema pattern is invalid") from exc


def _object(properties: dict, required: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _string(limit: int, *, nullable: bool = False) -> dict:
    return {
        "type": ["string", "null"] if nullable else "string",
        "minLength": 1,
        "maxLength": limit,
    }


def _ids(identifiers: Sequence[str], *, nonempty: bool = False) -> dict:
    item = _string(128)
    if identifiers:
        item["enum"] = list(identifiers)
    return {
        "type": "array",
        "items": item,
        "minItems": int(nonempty),
        "maxItems": min(64, len(identifiers)),
        "uniqueItems": True,
    }


def build_tool_specs(
    tool_schemas: Mapping[str, dict],
    *,
    descriptions: Mapping[str, str] | None = None,
    allowed_tools: Collection[str] | None = None,
    controls: Collection[str] = ("spawn", "send", "wait", "finish"),
    child_ids: Collection[str] = (),
    recipient_ids: Collection[str] = (),
    wait_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Advertise only registered/granted tools and currently legal controls.

    Callers supply parameter schemas explicitly; descriptions are never parsed
    or evaluated to discover arguments. IDs must come from the runtime's actual
    agents, and the tool grant must be the current agent's complete allowlist.
    Empty recipient/wait sets omit those functions. ``controls`` comes from the
    current action menu, so depth/capacity restrictions can omit spawning.
    """
    if not isinstance(tool_schemas, Mapping):
        raise ValueError("tool_schemas must map registered names to parameter schemas")  # noqa: TRY004 - JSON schema errors share a ValueError contract.
    all_names = _collection(list(tool_schemas), "registered tools", names=True)
    if set(all_names) & _RESERVED:
        raise ValueError("registered tools collide with reserved agent controls")
    granted = _collection(
        all_names if allowed_tools is None else allowed_tools,
        "allowed_tools",
        names=True,
    )
    if set(granted) - set(all_names):
        raise ValueError("granted tools require explicit registered parameter schemas")
    chosen = set(_collection(controls, "controls"))
    if chosen - _CONTROLS:
        raise ValueError("unknown native agent control")
    children = _collection(child_ids, "child_ids")
    recipients = _collection(recipient_ids, "recipient_ids")
    waiting = children if wait_ids is None else _collection(wait_ids, "wait_ids")
    if set(waiting) - set(children):
        raise ValueError("wait IDs must be actual direct child IDs")
    descriptions = descriptions or {}
    specs = []

    def add(name, description, parameters):
        _check_schema(parameters)
        if (
            parameters.get("type") != "object"
            or parameters.get("additionalProperties") is not False
        ):
            raise ValueError(
                "native function parameters require a closed object schema"
            )
        if not isinstance(description, str):
            raise ValueError("native function descriptions must be strings")  # noqa: TRY004 - Invalid JSON metadata, not an executable function signature.
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": copy.deepcopy(parameters),
                },
            }
        )

    for name in granted:
        add(
            name,
            descriptions.get(name, f"Call the registered {name} tool."),
            tool_schemas[name],
        )
    if "spawn" in chosen:
        add(
            "spawn",
            "Create one independent child subtask. For a review, omit tools for read-only defaults. Describe the question and deliverable; do not invent source contents. Use actual returned child IDs for later communication. Tool grants cannot exceed your own.",
            _object(
                {
                    "task": _string(12_000),
                    "role": _string(256, nullable=True),
                    "dependencies": _ids(children),
                    "tools": _ids(granted),
                },
                ["task"],
            ),
        )
    if "send" in chosen and recipients:
        recipient = _string(128)
        recipient["enum"] = recipients
        add(
            "send",
            "Send evidence or instructions to one currently permitted actual agent ID.",
            _object({"to": recipient, "message": _string(8_000)}, ["to", "message"]),
        )
    if "wait" in chosen and waiting:
        add(
            "wait",
            "Wait for outcomes from these actual direct children; do not invent IDs or tasks.",
            _object({"agents": _ids(waiting, nonempty=True)}, ["agents"]),
        )
    if "finish" in chosen:
        result_schema = _string(24_000)
        result_schema["type"] = ["string", "object"]
        result_schema["additionalProperties"] = True
        add(
            "finish",
            "Return the completed task result as a nonempty string or JSON object. Objects become JSON text, limited to 24000 characters and 32 nesting levels. This terminates your agent.",
            _object({"result": result_schema}, ["result"]),
        )
    return validate_tool_specs(specs)


def validate_tool_specs(specs: Sequence[dict]) -> list[dict[str, Any]]:
    """Normalize bounded, finite standard schemas for worker transport.

    No callable or description reflection is performed. The JSON round trip
    also disconnects caller-owned schema objects from queued requests.
    """
    if not isinstance(specs, (list, tuple)) or len(specs) > 32:
        raise ValueError(
            "native tools require a list of at most 32 function definitions"
        )
    try:
        encoded = json.dumps(
            specs, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("native tools must be finite JSON metadata") from exc
    if len(encoded.encode("utf-8")) > 64 << 10:
        raise ValueError("native tool definitions exceed 64 KiB")
    normalized = json.loads(encoded)
    names = set()
    for spec in normalized:
        if (
            not isinstance(spec, dict)
            or set(spec) != {"type", "function"}
            or spec["type"] != "function"
        ):
            raise ValueError("invalid native function catalog")
        function = spec["function"]
        if not isinstance(function, dict) or set(function) != {
            "name",
            "description",
            "parameters",
        }:
            raise ValueError("invalid native function definition")
        name = function["name"]
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in names
            or name == "tool"
        ):
            raise ValueError("invalid or duplicate native function name")
        if not isinstance(function["description"], str):
            raise ValueError("native function descriptions must be strings")  # noqa: TRY004 - Uniform worker validation error contract.
        _check_schema(function["parameters"])
        if (
            function["parameters"].get("type") != "object"
            or function["parameters"].get("additionalProperties") is not False
        ):
            raise ValueError(
                "native function parameters require a closed object schema"
            )
        names.add(name)
    return normalized


def _matches_type(value: Any, kind: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }[kind]()


def _validate(value: Any, schema: dict, location: str = "arguments") -> None:
    # Validate all nested data, including additionalProperties=true values that
    # have no individual property schema. Booleans are JSON booleans, not 0/1.
    _json_semantic_key(value)
    _validate_schema(value, schema, location)


def _json_semantic_key(value: Any) -> tuple:
    """Hashable JSON identity with recursive type and finite-number semantics.

    JSON numbers compare by value (1 equals 1.0), while booleans stay distinct.
    Object key order is immaterial; array order matters. Iterative traversal
    avoids recursion limits and also rejects cycles/non-JSON Python values.
    """
    ready: dict[int, tuple] = {}
    active: set[int] = set()
    pending = [(value, False)]
    while pending:
        item, exiting = pending.pop()
        identifier = id(item)
        kind = type(item)
        if kind in (dict, list):
            if exiting:
                active.remove(identifier)
                ready[identifier] = (
                    (
                        "object",
                        frozenset(
                            (key, ready[id(nested)]) for key, nested in item.items()
                        ),
                    )
                    if kind is dict
                    else ("array", tuple(ready[id(nested)] for nested in item))
                )
            elif identifier not in ready:
                if identifier in active:
                    raise ProtocolError("JSON data must not contain cycles")
                if kind is dict and any(type(key) is not str for key in item):
                    raise ProtocolError("JSON object keys must be strings")
                active.add(identifier)
                pending.append((item, True))
                pending.extend(
                    (nested, False)
                    for nested in (item.values() if kind is dict else item)
                )
        elif kind in (int, float):
            if kind is float and not math.isfinite(item):
                raise ProtocolError("JSON numbers must be finite")
            ready[identifier] = ("number", item)
        elif kind is bool:
            ready[identifier] = ("boolean", item)
        elif kind is str:
            ready[identifier] = ("string", item)
        elif item is None:
            ready[identifier] = ("null",)
        else:
            raise ProtocolError("value must contain only JSON data")
    return ready[id(value)]


def _validate_schema(value: Any, schema: dict, location: str = "arguments") -> None:
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_matches_type(value, kind) for kind in kinds):
            raise ProtocolError(f"{location} has the wrong JSON type")
    for key in ("anyOf", "oneOf"):
        if key in schema:
            matches = 0
            for option in schema[key]:
                try:
                    _validate_schema(value, option, location)
                    matches += 1
                except ProtocolError:
                    pass
            if matches == 0 or (key == "oneOf" and matches != 1):
                raise ProtocolError(f"{location} violates {key}")
    if "enum" in schema and not any(
        _json_semantic_key(value) == _json_semantic_key(option)
        for option in schema["enum"]
    ):
        raise ProtocolError(f"{location} is outside its current permitted values")
    if "const" in schema and (
        _json_semantic_key(value) != _json_semantic_key(schema["const"])
    ):
        raise ProtocolError(f"{location} violates const")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ProtocolError(f"{location} is missing required fields")
        extra = value.keys() - properties.keys()
        additional = schema.get("additionalProperties", True)
        if extra and additional is False:
            raise ProtocolError(
                f"{location} contains unadvertised fields: {sorted(extra)}"
            )
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], f"{location}.{key}")
            elif isinstance(additional, dict):
                _validate_schema(item, additional, f"{location}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", len(value)
        ):
            raise ProtocolError(f"{location} violates its array length bounds")
        if schema.get("uniqueItems"):
            identities = [_json_semantic_key(item) for item in value]
            if len(set(identities)) != len(identities):
                raise ProtocolError(f"{location} contains duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], f"{location}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get(
            "maxLength", len(value)
        ):
            raise ProtocolError(f"{location} violates its text length bounds")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ProtocolError(f"{location} does not match its required pattern")
    elif type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            raise ProtocolError("JSON numbers must be finite")
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ProtocolError(f"{location} violates numeric bounds")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ProtocolError(f"{location} violates exclusiveMinimum")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ProtocolError(f"{location} violates exclusiveMaximum")


def parse_native_action(
    text: str, tool_specs: Sequence[dict], *, max_chars: int = 32_768
) -> dict[str, Any]:
    """Accept exactly one native call and validate against its current catalog."""
    if not isinstance(text, str) or len(text) > max_chars:
        raise ProtocolError(
            f"native response must be text of at most {max_chars} characters"
        )
    catalog = {
        spec["function"]["name"]: spec["function"]["parameters"]
        for spec in validate_tool_specs(tool_specs)
    }
    body = text.strip()
    if body.startswith("<tool_call>"):
        if not body.endswith("</tool_call>"):
            raise ProtocolError("native call requires one complete tool_call wrapper")
        body = body[len("<tool_call>") : -len("</tool_call>")].strip()
    try:
        call = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"invalid JSON constant: {value}")
            ),
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError(f"invalid native call JSON: {exc}") from exc
    if (
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
    ):
        raise ProtocolError("native call requires exactly name and arguments")
    stack = [(call, 0)]
    while stack:
        item, depth = stack.pop()
        # Two framing levels (call -> arguments -> result) leave the advertised
        # 32 levels for a structured result. Other arguments are also bounded.
        if depth > 34:
            raise ProtocolError("native call JSON exceeds 34 nesting levels")
        if type(item) is float and not math.isfinite(item):
            raise ProtocolError("JSON numbers must be finite")
        if isinstance(item, dict):
            stack.extend((value, depth + 1) for value in item.values())
        elif isinstance(item, list):
            stack.extend((value, depth + 1) for value in item)
    name, arguments = call["name"], call["arguments"]
    if name not in catalog:
        raise ProtocolError(f"native function is not currently advertised: {name}")
    _validate(arguments, catalog[name])
    if name == "finish" and isinstance(arguments["result"], dict):
        pending = [(arguments["result"], 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 32:
                raise ProtocolError("native finish result exceeds 32 nesting levels")
            if isinstance(item, dict):
                pending.extend((value, depth + 1) for value in item.values())
            elif isinstance(item, list):
                pending.extend((value, depth + 1) for value in item)
        normalized = json.dumps(
            arguments["result"], ensure_ascii=False, allow_nan=False
        )
        if len(normalized) > 24_000:
            raise ProtocolError(
                "native finish result exceeds 24000 canonical characters"
            )
        arguments = {**arguments, "result": normalized}
    canonical = (
        {**arguments, "action": name}
        if name in _CONTROLS
        else {"action": "tool", "name": name, "args": arguments}
    )
    canonical_wire = json.dumps(canonical, ensure_ascii=False, allow_nan=False)
    # The model's wire response was already bounded above. Serializing an object
    # into the legacy result string introduces escaped quotes; do not confuse
    # that internal expansion with the native response's own size allowance.
    return parse_action(canonical_wire, max_chars=max(max_chars, len(canonical_wire)))


def parse_native_actions(
    text: str,
    tool_specs: Sequence[dict],
    *,
    max_calls: int = 4,
    max_chars: int = 32_768,
) -> list[dict[str, Any]]:
    """Validate a bounded native batch before returning any canonical actions.

    Multiple calls require contiguous complete XML wrappers. JSON decoding
    determines the payload boundary, so literal closing tags inside quoted
    arguments cannot terminate a call. A bare JSON call remains a single-call
    alternative. Runtime batch policy decides which validated actions may run
    together; this helper performs no effects or argument repair.
    """
    if type(max_calls) is not int or not 1 <= max_calls <= 4:
        raise ValueError("native batches permit between one and four calls")
    if not isinstance(text, str) or len(text) > max_chars:
        raise ProtocolError(
            f"native response must be text of at most {max_chars} characters"
        )
    specs = validate_tool_specs(tool_specs)
    body = text.strip()
    if not body.startswith("<tool_call>"):
        return [parse_native_action(body, specs, max_chars=max_chars)]
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ProtocolError(f"invalid JSON constant: {value}")
        ),
    )

    def skip_whitespace(position: int) -> int:
        while position < len(body) and body[position] in " \t\r\n":
            position += 1
        return position

    actions = []
    position = 0
    while position < len(body):
        if not body.startswith("<tool_call>", position):
            raise ProtocolError(
                "native batches require contiguous tool_call wrappers without prose"
            )
        if len(actions) >= max_calls:
            raise ProtocolError(f"native batch exceeds its {max_calls}-call allowance")
        start = skip_whitespace(position + len("<tool_call>"))
        try:
            _, end = decoder.raw_decode(body, start)
        except (ValueError, RecursionError) as exc:
            raise ProtocolError(f"invalid native batch JSON: {exc}") from exc
        position = skip_whitespace(end)
        if not body.startswith("</tool_call>", position):
            raise ProtocolError("native call requires an exact closing tool_call tag")
        # Retain only decoded JSON, not an extracted prose fragment. Validation
        # of all calls completes before the caller receives the returned list.
        actions.append(parse_native_action(body[start:end], specs, max_chars=max_chars))
        position = skip_whitespace(position + len("</tool_call>"))
    return actions
