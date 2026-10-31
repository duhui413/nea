"""Shared stage schemas and a fail-closed validator for their keyword subset.

This deliberately supports the project's contracts, not the full JSON Schema
standard.  Unsupported keywords are rejected even in an unvisited anyOf branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Protocol

from nea.method.gaps import d1_schema, d3_schema
from nea.method.grounding import grounding_schema
from nea.method.templates import critique_schema, diagnose_schema, judge_schema, plan_schema, validation_schema
from nea.schemas import SchemaError, Step1Plan


VALIDATION_PROTOCOL = "structured-stage-schema-0022-v1"
_KEYWORDS = frozenset({"type", "properties", "required", "additionalProperties", "items",
                       "enum", "minItems", "maxItems", "uniqueItems", "minLength",
                       "minimum", "maximum", "anyOf", "description"})
_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


class SchemaValidationError(SchemaError):
    """An instance or schema violates the supported structural contract."""


class _Job(Protocol):
    stage: str
    base: str

    def arg(self, name: str, default: Any = None) -> Any: ...


def stage_schema(job: _Job) -> dict[str, Any] | None:
    """Build the exact concrete schema used by prompts, requests and validation."""
    if job.stage in {"answer", "b_answer", "revise"}:
        return None
    if job.stage == "b_plan":
        return plan_schema()
    if job.stage == "plan":
        return plan_schema()
    if job.stage == "ground":
        return grounding_schema(job.arg("candidates", ()))
    if job.stage == "gaps":
        detector = job.arg("detector")
        if detector == "D1":
            return d1_schema()
        if detector == "D3":
            return d3_schema(job.arg("candidates", ()))
        raise ValueError("structured gaps stage requires D1 or D3")
    if job.stage == "diagnose":
        return diagnose_schema(job.arg("gap_ids", ()), job.arg("detour_indices", ()),
                               job.arg("bridge_indices", ()))
    if job.stage == "critique":
        return critique_schema()
    if job.stage == "validate":
        return validation_schema()
    if job.stage == "judge":
        return judge_schema()
    raise ValueError(f"no schema for stage {job.stage!r}")


def _canonical(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def schema_hash(schema: Mapping[str, Any]) -> str:
    """SHA256 of canonical UTF-8 JSON, without a trailing newline."""
    return hashlib.sha256(_canonical(schema).encode("utf-8")).hexdigest()


def structured_output_instruction(schema: Mapping[str, Any]) -> str:
    """Explicit compatibility instructions; also valid for native schema mode."""
    from nea.prompts import DEFAULT_PROMPTS_DIR, PromptStore
    return "\n" + PromptStore(DEFAULT_PROMPTS_DIR).text("structured_output") + "\n" + _canonical(schema) + "\n"



def _fail(path: str, rule: str) -> None:
    # Do not put model output or supplied property values into persisted errors.
    raise SchemaValidationError(f"{path}: {rule}")


def _json_value(value: Any, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, path)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, path)
        return
    _fail(path, "non_json_value")


def _check_schema(schema: Any, path: str = "schema") -> None:
    if not isinstance(schema, dict) or set(schema) - _KEYWORDS:
        _fail(path, "unsupported_schema_keyword_or_shape")
    if "type" in schema and (not isinstance(schema["type"], str) or schema["type"] not in _TYPES):
        _fail(path, "unsupported_schema_type")
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict) or not all(isinstance(key, str) for key in props):
            _fail(path, "invalid_properties")
        for child in props.values():
            _check_schema(child, path + ".properties")
    if "required" in schema:
        required = schema["required"]
        if (not isinstance(required, list) or not all(isinstance(key, str) for key in required)
                or len(set(required)) != len(required)):
            _fail(path, "invalid_required")
    for key in ("additionalProperties", "uniqueItems"):
        if key in schema and type(schema[key]) is not bool:
            _fail(path, "unsupported_" + key)
    for key in ("minItems", "maxItems", "minLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            _fail(path, "invalid_" + key)
    for key in ("minimum", "maximum"):
        if key in schema and (type(schema[key]) not in {int, float} or not math.isfinite(schema[key])):
            _fail(path, "invalid_" + key)
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            _fail(path, "invalid_enum")
        _json_value(schema["enum"], path)
        if any(_equal(left, right) for index, left in enumerate(schema["enum"])
               for right in schema["enum"][index + 1:]):
            _fail(path, "duplicate_enum")
    if "description" in schema and not isinstance(schema["description"], str):
        _fail(path, "invalid_description")
    if "items" in schema:
        _check_schema(schema["items"], path + ".items")
    if "anyOf" in schema:
        if not isinstance(schema["anyOf"], list) or not schema["anyOf"]:
            _fail(path, "invalid_anyOf")
        for branch in schema["anyOf"]:
            _check_schema(branch, path + ".anyOf")


def _equal(left: Any, right: Any) -> bool:
    """JSON equality keeps booleans distinct from numeric 0 and 1."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> None:
    kind = schema.get("type")
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "integer": type(value) is int,
               "number": type(value) in {int, float}, "boolean": type(value) is bool,
               "null": value is None}
    if kind is not None and not matches[kind]:
        _fail(path, "type")
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        _fail(path, "enum")
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate(value, branch, path)
                break
            except SchemaValidationError:
                pass
        else:
            _fail(path, "anyOf")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if set(schema.get("required", ())) - set(value):
            _fail(path, "required")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            _fail(path, "additionalProperties")
        for key in sorted(set(value) & set(props)):
            _validate(value[key], props[key], path + ".property")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            _fail(path, "array_length")
        if schema.get("uniqueItems") and any(_equal(left, right) for index, left in enumerate(value)
                                             for right in value[index + 1:]):
            _fail(path, "uniqueItems")
        if "items" in schema:
            for item in value:
                _validate(item, schema["items"], path + ".item")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        _fail(path, "minLength")
    if type(value) in {int, float}:
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            _fail(path, "number_range")


def validate_json_schema(value: Any, schema: Mapping[str, Any]) -> None:
    """Reject invalid JSON values, unknown keywords and nonconforming instances."""
    _check_schema(schema)
    _json_value(value, "instance")
    _validate(value, schema, "instance")
