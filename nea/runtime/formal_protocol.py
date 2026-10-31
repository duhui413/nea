"""Isolated, manifest-bound contract for the eight-arm paper experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nea.prompts import PromptStore, ACTIVE_PROMPTS
from nea.runtime.operator_protocol import ROLE_STAGE as LEGACY_ROLES
from nea.runtime.records import digest

PROTOCOL = METHOD_VERSION = "nea-formal8-v1"
DIRECTORY = "operator-v2"
ARMS = ("B", "P", "L", "F", "SR", "SRL", "FB", "CV")
PARENTS = {"F": "L", "SRL": "L", "SR": "B", "FB": "B", "CV": "B"}
ROLE_STAGE = {**LEGACY_ROLES,
    "formal_explanation_diagnosis": "diagnose",
    "cove_verification_questions": "cv_questions",
    "cove_verification_answer": "cv_answer", "cove_revision": "revise",
    "formal_concept_measurement": "measure", "formal_defect_audit": "audit",
    "formal_perturbation_rewrite": "perturb"}
STAGES = {"B": ("answer",), "P": ("prepare", "plan", "answer"),
    "L": ("ground", "plan", "answer"), "F": ("diagnose", "revise", "validate"),
    "FB": ("diagnose", "revise", "validate"),
    "SR": ("critique", "revise", "validate"), "SRL": ("critique", "revise", "validate"),
    "CV": ("cv_questions", "cv_answer", "revise"),
    "EVAL": ("judge", "measure", "audit"),
    "PERTURB": ("perturb", "audit", "diagnose")}
MARKDOWN_STAGES = frozenset({"answer", "revise"})
EXTRA_ROLES = tuple(role for role in ROLE_STAGE if role not in LEGACY_ROLES)


def source_split(phase: str) -> str:
    if phase not in {"live", "test", "case"}:
        raise ValueError("formal phase must be live, test or independent case")
    return "dev" if phase == "live" else phase


def is_operator_identity(identity: Mapping[str, Any]) -> bool:
    return identity.get("protocol_version") == PROTOCOL


def is_operator_job(job: Any) -> bool:
    return job.arg("operator_protocol") == PROTOCOL


def execution_identity(identity: Mapping[str, Any]) -> dict:
    return {key: value for key, value in identity.items() if key != "run_id"}


def check_arm_stage(arm: str, stage: str, split: str) -> None:
    if split not in {"dev", "test", "case"} or stage not in STAGES.get(arm, ()):
        raise ValueError("unregistered formal arm/stage/split")


def role_for_stage(stage: str) -> tuple[str, ...]:
    if stage not in set(ROLE_STAGE.values()):
        raise ValueError("unregistered formal stage")
    if stage == "audit":
        return ("student1", "teacher1", "teacher2")
    return ("teacher1", "teacher2") if stage in {"judge", "measure"} else ("student1",)


def phase_directory(run_root: Path, generator: str, phase: str) -> Path:
    from nea.runtime.identity import generator_directory
    source_split(phase)
    return generator_directory(run_root / DIRECTORY, generator) / phase


class FormalPromptStore(PromptStore):
    """Register new prompts without changing the old prompt inventory or hashes."""
    def __init__(self, directory: Path):
        super().__init__(directory)
        for role in EXTRA_ROLES:
            template = (directory / (role + ".md")).read_text(encoding="utf-8-sig")
            self._templates[role] = template
            self._metadata[role] = self._parse_metadata(template, role)

    def active_hashes(self) -> dict[str, str]:
        return {**super().active_hashes(),
                **{role + ".md": digest((self.directory / (role + ".md")).read_bytes())
                   for role in EXTRA_ROLES}}


def method_source_hash() -> str:
    """Bind both the reused engine and every formal implementation dependency."""
    from nea.pipeline.operator_workflow import method_source_hash as previous_hash
    root = Path(__file__).resolve().parents[2]
    paths = sorted({*root.glob("nea/*/formal_*.py"),
                    root / "scripts/run_formal_experiment.py",
                    root / "scripts/run_diagnosis_perturbation.py"})
    return digest({"operator_base": previous_hash(), "formal": {
        path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths if path.is_file()}})


def aggregate_telemetry(phase_root: Path) -> dict:
    """Account for failures and unknown usage, and expose cost by real arm."""
    import json
    from nea.runtime.records import ConflictError
    reservations = list((phase_root / "attempts").glob("*/*-reserved.json"))
    totals = {"request_count": len(reservations), "measured_request_count": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "elapsed_ms": 0.0,
              "known_usage_request_count": 0, "unknown_usage_request_count": 0,
              "cost": None, "cost_status": "NOT_CONFIGURED", "by_model": {}, "by_arm": {}, "by_comparison": {},
              "structure_failures": 0, "first_response_count": 0, "first_structure_success_count": 0}
    for path in sorted((phase_root / "telemetry").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf8"))
        body = {k: v for k, v in row.items() if k != "telemetry_hash"}
        reserved = phase_root / "attempts" / row["key"] / f"{row['attempt']:04d}-reserved.json"
        if row.get("protocol") != PROTOCOL or digest(body) != row.get("telemetry_hash") or not reserved.is_file():
            raise ConflictError("formal telemetry identity or journal differs")
        if json.loads(reserved.read_text(encoding="utf8"))["request_hash"] != row["request_hash"]:
            raise ConflictError("formal telemetry request differs")
        totals["measured_request_count"] += 1
        totals["structure_failures"] += row["result"] == "structure_error"
        structured = row["stage"] not in MARKDOWN_STAGES
        totals["first_response_count"] += row["attempt"] == 1 and structured
        totals["first_structure_success_count"] += row["attempt"] == 1 and structured and row["result"] == "success"
        groups = [totals]
        for field, key in (("by_model", row["model"]), ("by_arm", row["arm"])):
            group = totals[field].setdefault(key, {"request_count": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "elapsed_ms": 0.0, "known_usage_request_count": 0,
                "unknown_usage_request_count": 0})
            group["request_count"] += 1
            groups.append(group)
        comparison = totals["by_comparison"].setdefault(row.get("comparison", "main"), {})
        group = comparison.setdefault(row["arm"], {"request_count": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "elapsed_ms": 0.0, "known_usage_request_count": 0,
            "unknown_usage_request_count": 0})
        group["request_count"] += 1
        groups.append(group)
        for group in groups:
            group["elapsed_ms"] += row["elapsed_ms"]
            group["known_usage_request_count" if row["usage"] is not None else "unknown_usage_request_count"] += 1
            if row["usage"] is not None:
                for name in ("prompt_tokens", "completion_tokens"):
                    group[name] += row["usage"][name]
    missing = totals["request_count"] - totals["measured_request_count"]
    if missing < 0:
        raise ConflictError("more formal telemetry than reserved sends")
    totals["unmeasured_request_count"] = missing
    totals["unknown_usage_request_count"] += missing
    totals["accounting_complete"] = missing == 0 and totals["unknown_usage_request_count"] == 0
    totals["accounting_note"] = ("complete observed attempts" if totals["accounting_complete"] else
        "observed token/time totals are lower bounds; unmatched reservations cannot be allocated to arms")
    totals["first_structure_success_scope"] = "first attempts of structured stages only; excludes answer and revise Markdown"
    totals["first_structure_success_rate"] = (totals["first_structure_success_count"] / totals["first_response_count"]
        if totals["first_response_count"] else None)
    totals["elapsed_ms"] = round(totals["elapsed_ms"], 3)
    for field in ("by_model", "by_arm"):
        for group in totals[field].values():
            group["elapsed_ms"] = round(group["elapsed_ms"], 3)
    for comparison in totals["by_comparison"].values():
        for group in comparison.values():
            group["elapsed_ms"] = round(group["elapsed_ms"], 3)
    return totals
