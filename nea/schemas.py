"""Strict baseline planning schema; other stage checks live with transport."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, ClassVar, Collection, Mapping
_UNKNOWN_IDENTIFIER_SENTINELS = frozenset({"unknown", "none", "null", "n/a", "未知", "不明"})

class SchemaError(ValueError):
    """Raised when an LLM structured output violates its persisted contract."""

def _strict_keys(data: Mapping[str, Any], required: set[str], label: str) -> None:
    actual = set(data)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise SchemaError(f"{label} has invalid keys ({'; '.join(details)})")

def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be an object")
    return value

def _string(value: Any, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise SchemaError(f"{field} must not be empty")
    return normalized

def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{field} must be a list of strings")
    return [_string(item, f"{field}[{index}]") for index, item in enumerate(value)]

def _identifier(value: Any, field: str) -> str:
    """Validate an auditable identifier rather than silently dropping it."""

    if not isinstance(value, str):
        raise SchemaError(f"{field} must be an identifier string")
    identifier = value.strip()
    if not identifier:
        raise SchemaError(f"{field} must not be empty")
    if identifier.casefold() in _UNKNOWN_IDENTIFIER_SENTINELS:
        raise SchemaError(f"{field} must be a real identifier, not {identifier!r}")
    return identifier

def _integer(
    value: Any,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field} must be an integer")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise SchemaError(f"{field} must be between {minimum} and {maximum}")
    return value

def _enum(value: Any, field: str, allowed: Collection[str]) -> str:
    parsed = _string(value, field)
    if parsed not in allowed:
        raise SchemaError(f"{field} must be one of {sorted(allowed)}")
    return parsed

@dataclass(frozen=True)
class Step1Subquery:
    query: str
    target_concept_id: str
    slot: str
    level: int

    SLOTS: ClassVar[tuple[str, ...]] = ("obj", "pre", "bridge", "target", "example")

    @classmethod
    def schema(cls) -> dict[str, Any]:
        properties = {
            "query": {"type": "string", "minLength": 1},
            "target_concept_id": {"type": "string", "minLength": 1},
            "slot": {"type": "string", "enum": list(cls.SLOTS)},
            "level": {"type": "integer", "minimum": 0, "maximum": 2},
        }
        return {"type": "object", "additionalProperties": False,
                "required": list(properties), "properties": properties}

    @classmethod
    def from_dict(cls, value: Any) -> "Step1Subquery":
        data = _object(value, "step1 subquery")
        _strict_keys(data, {"query", "target_concept_id", "slot", "level"}, "step1 subquery")
        return cls(
            query=_string(data["query"], "query"),
            target_concept_id=_identifier(data["target_concept_id"], "target_concept_id"),
            slot=_enum(data["slot"], "slot", cls.SLOTS),
            level=_integer(data["level"], "level", 0, 2),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class Step1Plan:
    stage_id: str
    planning_mode: str
    rewritten_query: str
    subqueries: list[Step1Subquery]
    answer_outline: list[str]
    retrieval_queries: list[str]
    assumptions: list[str]
    warnings: list[str]

    FIELDS: ClassVar[set[str]] = {
        "stage_id", "planning_mode", "rewritten_query", "subqueries",
        "answer_outline", "retrieval_queries", "assumptions", "warnings",
    }

    @classmethod
    def schema(cls, base: str | None = None) -> dict[str, Any]:
        """The arm-B planning contract, with PE's empty retrieval requirement."""
        if base not in {None, "pe", "rag"}:
            raise ValueError("baseline planning base must be pe or rag")
        properties: dict[str, Any] = {
            "stage_id": {"type": "string", "enum": ["step1"]},
            "planning_mode": {"type": "string", "enum": ["baseline"]},
            "rewritten_query": {"type": "string", "minLength": 1},
            "subqueries": {"type": "array", "items": Step1Subquery.schema()},
        }
        for field in ("answer_outline", "retrieval_queries", "assumptions", "warnings"):
            properties[field] = {"type": "array", "items": {"type": "string", "minLength": 1}}
        if base == "pe":
            properties["retrieval_queries"]["maxItems"] = 0
        return {"type": "object", "additionalProperties": False,
                "required": sorted(cls.FIELDS), "properties": properties}

    @classmethod
    def from_dict(cls, value: Any) -> "Step1Plan":
        data = _object(value, "step1 output")
        _strict_keys(data, cls.FIELDS, "step1 output")
        raw_subqueries = data["subqueries"]
        if not isinstance(raw_subqueries, list):
            raise SchemaError("subqueries must be a list")
        return cls(
            stage_id=_enum(data["stage_id"], "stage_id", {"step1"}),
            planning_mode=_enum(data["planning_mode"], "planning_mode", {"baseline", "cognitive"}),
            rewritten_query=_string(data["rewritten_query"], "rewritten_query"),
            subqueries=[Step1Subquery.from_dict(item) for item in raw_subqueries],
            answer_outline=_strings(data["answer_outline"], "answer_outline"),
            retrieval_queries=_strings(data["retrieval_queries"], "retrieval_queries"),
            assumptions=_strings(data["assumptions"], "assumptions"),
            warnings=_strings(data["warnings"], "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "planning_mode": self.planning_mode,
            "rewritten_query": self.rewritten_query,
            "subqueries": [item.to_dict() for item in self.subqueries],
            "answer_outline": list(self.answer_outline),
            "retrieval_queries": list(self.retrieval_queries),
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }
