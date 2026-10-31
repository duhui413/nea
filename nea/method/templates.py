"""File-backed role rendering and single-source output schemas (0022)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from nea.prompts import PromptStore, DEFAULT_PROMPTS_DIR, ALIASES, ROLES
from nea.method.gaps import d1_schema, d3_schema
from nea.method.grounding import grounding_schema
from nea.evaluation.judging import RUBRIC_ANCHORS, rubric_payload

INPUT_MARKER = "## 本题输入\n"
BASELINE_PAYLOAD_MARKER = INPUT_MARKER
REPAIR_NOTE = "\n" + PromptStore(DEFAULT_PROMPTS_DIR).text("repair_output") + "\n{reason}\n"
MODULE_BY_DEFECT = {"term_gap": "T_EXP", "step_gap": "D_EXP", "logical_jump": "D_CON",
                    "term_excess": "T_CON", "step_excess": "D_CON",
                    "factual_error": "shared", "task_omission": "shared"}


def obj(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def string(*values):
    return {"type": "string", **({"enum": list(values)} if values else {"minLength": 1})}


def array(items, **limits):
    return {"type": "array", "items": items, **limits}


def plan_schema():
    return obj({"route": string("direct", "decompose"),
                "needed_prerequisites": array(string(), maxItems=6),
                "obligations": array(string(), minItems=1),
                "given": array(string()), "uncertain": array(string())})


def finding_schema():
    return obj({"span": {"type": "string"}, "question_anchor": {"type": "string"},
                "reason": string(), "evidence_status": string("supported", "reasoning", "conflict", "unknown"),
                "evidence_ids": array(string(), uniqueItems=True)})


def diagnose_schema(gap_ids=None, detour_indices=None, bridge_indices=None):
    # Old keyword arguments remain callable; surface signal lists no longer gate semantics.
    branches = []
    for defect, module in MODULE_BY_DEFECT.items():
        for expands in (False, True):
            actions = ("add", "compress", "bridge", "rewrite") if defect in {
                "task_omission", "term_gap", "step_gap"} else ("add", "bridge") if expands else ("compress", "rewrite")
            if defect in {"task_omission", "term_gap", "step_gap"} and not expands:
                continue
            properties = {**finding_schema()["properties"], "type": string(defect),
                          "module": string(module), "action": string(*actions), "instruction": string()}
            if expands:
                properties.update(question_anchor=string(), necessity_reason=string())
            branches.append(obj(properties))
    actions = array({"anyOf": branches}, minItems=1, maxItems=3)
    return {"anyOf": [obj({"status": string("issues"), "actions": actions}),
                      obj({"status": string("no_issue", "uncertain"),
                           "actions": array({"anyOf": branches}, maxItems=0)})]}


def diagnose_schema_args(payload: Mapping[str, Any]) -> dict:
    return {}


def critique_schema():
    issues = {"anyOf": [obj({**finding_schema()["properties"], "scope_effect": string("correction"),
                               "necessity_reason": {"type": "string"}}),
                         obj({**finding_schema()["properties"], "scope_effect": string("add_obligation"),
                              "question_anchor": string(), "necessity_reason": string()})]}
    return {"anyOf": [obj({"status": string("issues"), "issues": array(issues, minItems=1, maxItems=3)}),
                      obj({"status": string("no_issue", "uncertain"), "issues": array(issues, maxItems=0)})]}


def validation_schema():
    return obj({"status": string("pass", "fail", "uncertain"), "reason": string(),
                "findings": array(finding_schema())})


def judge_schema():
    return obj({"scores": obj({dim: {"type": "integer", "minimum": 1, "maximum": 5} for dim in sorted(RUBRIC_ANCHORS)}),
                "reasons": obj({dim: finding_schema() for dim in sorted(RUBRIC_ANCHORS)}),
                "status": string("issues", "no_issue", "uncertain"),
                "findings": array(finding_schema())})


def role_schema(role, payload=None, d3_candidates=None):
    payload = payload or {}
    if role == "gaps_d1":
        return d1_schema()
    if role == "gaps_d3":
        return d3_schema(d3_candidates if d3_candidates is not None else [r["concept_id"] for r in payload.get("candidates", [])])
    role = ALIASES.get(role, role)
    if role in {"query_task_analysis", "explanation_obligation_planning"}:
        return plan_schema()
    if role == "target_grounding":
        return grounding_schema([r["concept_id"] for r in payload.get("candidates", [])])
    return {"explanation_diagnosis": diagnose_schema, "generic_answer_critique": critique_schema,
            "revision_validation": validation_schema, "answer_quality_evaluation": judge_schema}.get(role, lambda: None)()


def template_hashes(prompts_dir: Path = DEFAULT_PROMPTS_DIR):
    store = PromptStore(prompts_dir)
    # Include shared instructions and rubric in every role identity; conservative invalidation.
    sources = json.dumps(store.active_hashes(), sort_keys=True)
    return {name: hashlib.sha256((sources + name + store.text(name)).encode()).hexdigest()
            for name in (*ROLES, *ALIASES)}


def baseline_template_hashes(prompts_dir: Path):
    hashes = template_hashes(prompts_dir)
    return {"b_plan": hashes["step1"], "b_answer": hashes["step2"]}


def render(template: str, payload: Mapping[str, Any], *, d3_candidates=None,
           repair_reason=None, prompts_dir: Path = DEFAULT_PROMPTS_DIR):
    return PromptStore(prompts_dir).render(template, dict(payload), repair_reason,
        schema=role_schema(template, payload, d3_candidates))


def judge_payload(question: str, answer: str):
    return {"question": question, "candidate_answer": answer, "rubric": rubric_payload()}


class BaselineRenderer:
    """Stable B call interface; now shares the task-first roles with P/L."""
    def __init__(self, prompts_dir: Path):
        self.store = PromptStore(prompts_dir)
        self.prompts_dir = prompts_dir

    def plan_prompt(self, *, base, question, chunks, repair_reason=None):
        return self.store.render("step1", {"question": question, "retrieval": base.upper(),
            "knowledge_chunks": chunks if base == "rag" else []}, repair_reason)

    def answer_prompt(self, *, base, question, plan, chunks, repair_reason=None):
        return self.store.render("step2", {"question": question, "retrieval": base.upper(),
            "obligations": plan.get("obligations", []), "plan": dict(plan),
            "knowledge_chunks": chunks if base == "rag" else []}, repair_reason)
