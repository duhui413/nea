"""The versioned runtime contract for the 0022 operator pipeline.

The historical ca-v3 registry stays unchanged.  This contract shares its
transport and journal, but never shares an execution identity with it.
"""
from __future__ import annotations

from typing import Any, Mapping


PROTOCOL = "nea-0022-v1"
METHOD_VERSION = PROTOCOL
DIRECTORY = "operator-v1"
PHASES = ("live", "test")
ARMS = ("B", "P", "L", "F", "SR")
RESEARCH_VERSION = "nea-0032-same-parent-v1"
RESEARCH_STAGES = {"R_F": ("diagnose", "revise", "validate"),
                   "R_SR": ("critique", "revise", "validate")}
STAGES = {
    "B": ("answer",),
    "P": ("prepare", "plan", "answer"),
    "L": ("ground", "plan", "answer"),
    "F": ("diagnose", "revise", "validate"),
    "SR": ("critique", "revise", "validate"),
    "EVAL": ("judge",),
}
MARKDOWN_STAGES = frozenset({"answer", "revise"})
ROLE_STAGE = {"query_task_analysis": "prepare", "target_grounding": "ground",
              "explanation_obligation_planning": "plan", "answer_generation": "answer",
              "explanation_diagnosis": "diagnose", "generic_answer_critique": "critique",
              "answer_revision": "revise", "revision_validation": "validate",
              "answer_quality_evaluation": "judge"}


def is_operator_identity(identity: Mapping[str, Any]) -> bool:
    return identity.get("protocol_version") == PROTOCOL


def is_operator_job(job: Any) -> bool:
    return job.arg("operator_protocol") == PROTOCOL


def execution_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """All locked inputs except the location-specific run id enter cache keys."""
    return {key: value for key, value in identity.items() if key != "run_id"}


def source_split(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError("operator phase must be live or test")
    return "dev" if phase == "live" else "test"


def check_arm_stage(arm: str, stage: str, split: str) -> None:
    if split not in {"dev", "test"}:
        raise ValueError("operator question split must be dev or test")
    if arm not in STAGES or stage not in STAGES[arm]:
        raise ValueError(f"operator arm x stage {arm!r} x {stage!r} is not permitted")


def check_research_job(identity: Mapping[str, Any], arm: str, stage: str, role: str) -> None:
    """Narrow, manifest-bound development research; no default-arm expansion."""
    from nea.providers import config
    contract = identity.get("research_contract")
    if not isinstance(contract, dict) or identity.get("split") != "live":
        raise ValueError("research jobs require their explicit live manifest contract")
    generator = config.experiment_role_models_for(identity["generator"])["student1"]
    independent = "teacher2" if config.experiment_provider_name_for_model(generator) == "mimo" else "teacher1"
    lane = contract.get("lane")
    expected = {"version": RESEARCH_VERSION, "lane": lane,
                "arms": ["R_F", "R_SR"], "parent_arm": "L",
                "validation_role": independent if lane == "independent" else "student1"}
    if contract != expected or lane not in {"generation", "independent", "evaluation"}:
        raise ValueError("unknown same-parent research contract")
    allowed = ((arm in RESEARCH_STAGES and stage in RESEARCH_STAGES[arm] and role == "student1")
               if lane == "generation" else
               (arm in RESEARCH_STAGES and stage == "validate" and role == independent)
               if lane == "independent" else
               (arm == "EVAL" and stage == "judge" and role in {"teacher1", "teacher2"}))
    if not allowed:
        raise ValueError("job is outside the research lane arm/stage/role whitelist")


def role_for_stage(stage: str) -> tuple[str, ...]:
    if stage not in {value for values in STAGES.values() for value in values}:
        raise ValueError(f"unknown operator stage {stage!r}")
    return ("teacher1", "teacher2") if stage == "judge" else ("student1",)


def aggregate_telemetry(phase_root: Any) -> dict[str, Any]:
    """Report paid attempts including failures; absent prices remain unknown.

    Token counts and elapsed time are observed quantities.  This repository
    has no versioned provider price catalogue, so no currency amount is
    invented from tokens.  Reserved calls without a measurement are explicit.
    """
    import json
    from pathlib import Path
    from nea.runtime.records import ConflictError, digest

    root = Path(phase_root)
    reservations = list((root / "attempts").glob("*/*-reserved.json"))
    totals = {"request_count": len(reservations), "measured_request_count": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "known_usage_request_count": 0,
              "unknown_usage_request_count": 0, "elapsed_ms": 0.0,
              "cost": None, "cost_status": "price_unavailable", "by_model": {}}
    for path in sorted((root / "telemetry").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf8"))
        body = {key: value for key, value in row.items() if key != "telemetry_hash"}
        reservation = root / "attempts" / row.get("key", "") / f"{row.get('attempt', 0):04d}-reserved.json"
        if (row.get("telemetry_hash") != digest(body) or row.get("protocol") != PROTOCOL
                or not reservation.is_file()):
            raise ConflictError("Operator telemetry is corrupt or has no reserved request")
        reserved = json.loads(reservation.read_text(encoding="utf8"))
        if reserved.get("request_hash") != row.get("request_hash"):
            raise ConflictError("Operator telemetry request identity differs")
        totals["measured_request_count"] += 1
        totals["elapsed_ms"] += row["elapsed_ms"]
        model = totals["by_model"].setdefault(row["model"], {"request_count": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "elapsed_ms": 0.0, "unknown_usage_request_count": 0})
        model["request_count"] += 1
        model["elapsed_ms"] += row["elapsed_ms"]
        if row["usage"] is None:
            model["unknown_usage_request_count"] += 1
            totals["unknown_usage_request_count"] += 1
        else:
            totals["known_usage_request_count"] += 1
            for name in ("prompt_tokens", "completion_tokens"):
                totals[name] += row["usage"][name]
                model[name] += row["usage"][name]
    missing = totals["request_count"] - totals["measured_request_count"]
    if missing < 0:
        raise ConflictError("More telemetry measurements than reserved requests")
    totals["unmeasured_request_count"] = missing
    totals["unknown_usage_request_count"] += missing
    totals["elapsed_ms"] = round(totals["elapsed_ms"], 3)
    for model in totals["by_model"].values():
        model["elapsed_ms"] = round(model["elapsed_ms"], 3)
    return totals
