"""Registered formal operators, isolated from frozen operator-v1 identities."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from nea.method import operators as legacy
from nea.runtime.records import digest

OPERATOR_VERSION = "nea-answer-operator-formal8-v1"
MODULES = ("T_EXP", "D_EXP", "D_CON", "T_CON")
BUDGET = {"max_rounds": 1, "max_candidates": 1, "max_growth": 1.2,
          "max_actions": 3, "max_validation_calls": 1, "max_graph_nodes": 24}
ABLATIONS = {
    "without_term_expansion": {"modules": ("D_EXP", "D_CON", "T_CON"), "shared": True},
    "without_derivation_expansion": {"modules": ("T_EXP", "D_CON", "T_CON"), "shared": True},
    "without_term_constraint": {"modules": ("T_EXP", "D_EXP", "D_CON"), "shared": True},
    "without_derivation_constraint": {"modules": ("T_EXP", "D_EXP", "T_CON"), "shared": True},
    "expansion_only": {"modules": ("T_EXP", "D_EXP"), "shared": True},
    "constraint_only": {"modules": ("D_CON", "T_CON"), "shared": True},
    "modules_only": {"modules": MODULES, "shared": False},
    "shared_only": {"modules": (), "shared": True},
}
SENSITIVITIES = {
    "actions_1": {"max_actions": 1}, "actions_5": {"max_actions": 5},
    "growth_1_0": {"max_growth": 1.0}, "growth_1_5": {"max_growth": 1.5},
    "rounds_2": {"max_rounds": 2},
    "graph_nodes_12": {"max_graph_nodes": 12},
    "graph_nodes_48": {"max_graph_nodes": 48},
}


def registered_budget(phase: str, comparison: str = "main") -> dict[str, Any]:
    if phase not in {"live", "test"}:
        raise ValueError("formal phase must be live or test")
    if comparison == "main" or (comparison.startswith("ablation:")
                               and comparison[9:] in ABLATIONS):
        return dict(BUDGET)
    if comparison.startswith("sensitivity:") and comparison[12:] in SENSITIVITIES:
        if phase != "live":
            raise ValueError("registered budget sensitivity requires development/live")
        return {**BUDGET, **SENSITIVITIES[comparison[12:]]}
    raise ValueError("unknown formal comparison")


@dataclass(frozen=True)
class FormalOperatorContext(legacy.OperatorContext):
    """The same pure safety rules with the explicitly registered E6 limits.

    A context is always one round. The host orchestrates a second round with
    a fresh diagnosis and keeps the total length cap tied to the first parent.
    """

    def __post_init__(self) -> None:
        limits = {**BUDGET, **dict(self.budget)}
        if (type(limits["max_growth"]) not in {int, float}
                or limits["max_growth"] not in {1.0, 1.2, 1.5}
                or type(limits["max_actions"]) is not int
                or limits["max_actions"] not in {1, 3, 5}
                or any(type(limits[key]) is not int or limits[key] != 1
                       for key in ("max_rounds", "max_candidates", "max_validation_calls"))
                or type(limits["max_graph_nodes"]) is not int
                or limits["max_graph_nodes"] not in {12, 24, 48}):
            raise ValueError("unregistered formal single-round operator budget")
        maximum = int(len(self.parent_answer) * limits["max_growth"])
        cap = limits.get("max_chars", maximum)
        if type(cap) is not int or not 0 < cap <= maximum:
            raise ValueError("formal max_chars exceeds registered growth budget")
        # Delegate evidence, task and citation checks to the existing pure
        # context, using its original limits only during its initialization.
        baseline = {**limits, "max_actions": min(3, limits["max_actions"]),
                    "max_growth": min(1.2, limits["max_growth"])}
        baseline.pop("max_chars", None)
        object.__setattr__(self, "budget", baseline)
        super().__post_init__()
        object.__setattr__(self, "budget", deepcopy({**limits, "max_chars": cap}))

    @property
    def context_sha256(self) -> str:
        return digest({"question": self.question, "parent_sha256": self.parent_sha256,
                       "evidence": list(self.evidence), "obligations": list(self.obligations),
                       "budget": dict(self.budget), "versions": dict(self.versions),
                       "base": self.base, "operator_version": OPERATOR_VERSION})


def select_actions(ctx: FormalOperatorContext, diagnosis: Mapping[str, Any], *,
                   enabled_modules: Sequence[str] = MODULES, shared: bool = True,
                   generic: bool = False) -> dict[str, Any]:
    source = deepcopy(dict(diagnosis))
    excluded = []
    if not shared and not generic:
        actions = source.get("actions", [])
        excluded = [{"index": index, "reason": "shared_correction_disabled"}
                    for index, row in enumerate(actions)
                    if isinstance(row, Mapping)
                    and row.get("type", row.get("defect_type")) in legacy.SHARED_TYPES]
        source["actions"] = [row for row in actions if not isinstance(row, Mapping)
                             or row.get("type", row.get("defect_type")) not in legacy.SHARED_TYPES]
    result = legacy.select_actions(ctx, source, enabled_modules=enabled_modules, generic=generic)
    result["rejections"] = excluded + result["rejections"]
    return result


def finalize(ctx: FormalOperatorContext, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {**legacy.finalize(ctx, *args, **kwargs), "operator_version": OPERATOR_VERSION}


diagnose_payload = legacy.diagnose_payload
revise_payload = legacy.revise_payload
validate_payload = legacy.validate_payload
validate_candidate = legacy.validate_candidate
