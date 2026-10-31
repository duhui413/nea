"""Pure need-aware job builders: per-question preparation, contexts, prompts, jobs.

Nothing here performs file, model or network I/O; the caller injects the
graph, the candidate index, the T/S components and the RAG candidate pools.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from nea.evaluation.judging import RUBRIC_HASH
from nea.runtime.records import answer_sha256, digest
from nea.runtime.protocol import BASELINE_RENDERER_ID
from nea.method.rag import select_shared_rag_context

from nea.method import gaps as gapmod
from nea.method.graph import Graph, link_concepts
from nea.pipeline.registry import (
    ARM_DETECTOR,
    EVAL_ARM,
    EVAL_STAGE,
    GENERATION_ROLE,
    PROTOCOL,
    REVISION_LIMITS,
)
from nea.method.templates import (
    BaselineRenderer,
    judge_payload,
    diagnose_schema_args,
    render,
    template_hashes,
    baseline_template_hashes,
)
from nea.runtime.transport import ExperimentJob


RAG_TOP_K = 8
MAX_NODE_BLOCKS = 4
ROOT = "ROOT"
NO_CALIBRATION = "NOT_AVAILABLE"


def static_payloads(prompts_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-template static inputs; their hashes enter the run manifest."""
    hashes = template_hashes(prompts_dir)
    rows = {name: {"protocol": PROTOCOL, "template": name, "template_sha256": value,
                   "rag_policy": "shared ordinary pool; grounded arms: ordinary(8-k)+k node blocks",
                   "revision_limits": REVISION_LIMITS}
            for name, value in hashes.items() if name != "common"}
    for stage, value in baseline_template_hashes(prompts_dir).items():
        rows[f"v1_{stage}"] = {"protocol": PROTOCOL, "template": f"v1:{stage}",
                               "template_sha256": value, "renderer": BASELINE_RENDERER_ID,
                               "rag_policy": "shared ordinary pool top-8"}
    return rows


def all_template_hashes(prompts_dir: Path) -> dict[str, str]:
    return {**{name: value for name, value in template_hashes(prompts_dir).items()},
            **{f"v1_{stage}": value for stage, value in baseline_template_hashes(prompts_dir).items()}}


@dataclass(frozen=True)
class QuestionPrep:
    row: dict[str, Any]
    question_ids: tuple[str, ...]
    terminology: float
    structure: float
    candidates: tuple[dict[str, str], ...]
    pool: tuple[dict[str, Any], ...]

    @property
    def qid(self) -> str:
        return self.row["qid"]

    @property
    def question(self) -> str:
        return self.row["question"]


def _public_chunk(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"chunk_id": str(row["chunk_id"]), "concept_id": str(row.get("concept_id", "")),
            "concept_name": str(row.get("concept_name", "")), "content": str(row.get("content", ""))}


def prepare_question(row: Mapping[str, Any], graph: Graph, index: Any,
                     terminology: Callable[[str], float], structure: Callable[[str], float],
                     pool: Sequence[Mapping[str, Any]] | None) -> QuestionPrep:
    question = row["question"]
    ranked: list[dict[str, Any]] = []
    if pool is not None:
        # Shared ordinary ranking (score desc, dense desc, chunk id): local-path rule.
        context = select_shared_rag_context(list(pool))
        by_id = {str(item["chunk_id"]): item for item in pool}
        order = list(context["chunk_ids"]) + sorted(
            (key for key in by_id if key not in set(context["chunk_ids"])),
            key=lambda key: (-float(by_id[key].get("rerank_score", by_id[key].get("score", 0)) or 0),
                             -float(by_id[key].get("dense_score", 0) or 0), key))
        ranked = [_public_chunk(by_id[key]) for key in order]
    return QuestionPrep(row=dict(row), question_ids=tuple(link_concepts(question, graph)),
                        terminology=float(terminology(question)), structure=float(structure(question)),
                        candidates=tuple(index.rank(question)), pool=tuple(ranked))


def ordinary_context(prep: QuestionPrep, base: str) -> list[dict[str, Any]]:
    return list(prep.pool[:RAG_TOP_K]) if base == "rag" else []


def node_block(graph: Graph, node_id: str) -> dict[str, Any]:
    node = graph.nodes[node_id]
    return {"chunk_id": node_id, "concept_id": node_id, "concept_name": node["name"],
            "content": node["explain"]}


def grounded_context(prep: QuestionPrep, base: str, graph: Graph, target_ids: Sequence[str],
                     ordered_gaps: Sequence[str]) -> list[dict[str, Any]]:
    """ordinary top (8-k) + k target/gap node blocks, k = min(|G_q|+1, 4), deduplicated."""
    if base != "rag":
        return []
    if not target_ids:
        return ordinary_context(prep, base)
    k = min(len(ordered_gaps) + 1, MAX_NODE_BLOCKS)
    nodes: list[str] = []
    for node in [target_ids[0], *ordered_gaps]:
        if node not in nodes:
            nodes.append(node)
    nodes = nodes[:k]
    ordinary = [chunk for chunk in prep.pool if chunk["chunk_id"] not in set(nodes)][:RAG_TOP_K - len(nodes)]
    return ordinary + [node_block(graph, node) for node in nodes]


def _hashes(prompts_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    payloads = static_payloads(prompts_dir)
    return all_template_hashes(prompts_dir), {name: digest(value) for name, value in payloads.items()}


class JobFactory:
    """Build ExperimentJob objects with the right template/static hashes."""

    def __init__(self, prompts_dir: Path, split: str):
        self.templates, self.statics = _hashes(prompts_dir)
        self.prompts_dir = prompts_dir
        self.split = split
        self.b_renderer = BaselineRenderer(prompts_dir)

    def render(self, role, payload, **kwargs):
        return render(role, payload, prompts_dir=self.prompts_dir, **kwargs)

    def job(self, prep: QuestionPrep, base: str, arm: str, stage: str, template: str, prompt: str,
            *, parent: str, calibration: str = NO_CALIBRATION, role: str = GENERATION_ROLE,
            call_index: int = 1, **schema_args: Any) -> ExperimentJob:
        return ExperimentJob(qid=prep.qid, split=self.split, base=base, arm=arm, stage=stage, role=role,
                     prompt=prompt, template_hash=self.templates[template],
                     static_payload_hash=self.statics[template], parent_hash=parent,
                     calibration_hash=calibration, rubric_hash=RUBRIC_HASH, call_index=call_index,
                     schema_args=tuple(sorted(schema_args.items())))

    # --- wave 1 -------------------------------------------------------
    def b_plan(self, prep: QuestionPrep, base: str) -> ExperimentJob:
        prompt = self.b_renderer.plan_prompt(base=base, question=prep.question,
                                             chunks=ordinary_context(prep, base))
        return self.job(prep, base, "B", "b_plan", "v1_b_plan", prompt, parent=ROOT)

    def p_plan(self, prep: QuestionPrep, base: str) -> ExperimentJob:
        prompt = self.render("plan", {"question": prep.question, "retrieval": base.upper()})
        return self.job(prep, base, "P", "plan", "plan", prompt, parent=ROOT)

    def ground(self, prep: QuestionPrep, base: str, arm: str, graph: Graph) -> ExperimentJob:
        candidates = [{"concept_id": row["concept_id"], "name": graph.name(row["concept_id"]),
                       "chapter": graph.nodes[row["concept_id"]]["chapter"]} for row in prep.candidates]
        prompt = self.render("ground", {"question": prep.question, "candidates": candidates})
        return self.job(prep, base, arm, "ground", "ground", prompt, parent=digest(list(prep.candidates)),
                        candidates=tuple(row["concept_id"] for row in candidates))

    # --- wave 2 -------------------------------------------------------
    def b_answer(self, prep: QuestionPrep, base: str, plan: Mapping[str, Any], plan_text: str) -> ExperimentJob:
        effective = dict(plan)
        if base == "pe":
            effective["retrieval_queries"] = []
        chunks = ordinary_context(prep, base)
        prompt = self.b_renderer.answer_prompt(base=base, question=prep.question, plan=effective,
                                               chunks=chunks)
        return self.job(prep, base, "B", "b_answer", "v1_b_answer", prompt,
                        parent=answer_sha256(plan_text),
                        allowed_citations=tuple(chunk["chunk_id"] for chunk in chunks))

    def gaps(self, prep: QuestionPrep, base: str, arm: str, detector: str, graph: Graph,
             grounding: Mapping[str, Any], c2: Mapping[str, Any]) -> ExperimentJob:
        targets = [{"concept_id": node, "name": graph.name(node)} for node in grounding["grounded_target_ids"]]
        parent = digest({"grounding": grounding["grounded_target_ids"], "c2": c2["candidate_ids"]})
        if detector == "D1":
            prompt = self.render("gaps_d1", {"question": prep.question,
                                        "grounded_targets": [row["name"] for row in targets]})
            return self.job(prep, base, arm, "gaps", "gaps_d1", prompt, parent=parent, detector="D1")
        candidates = [{"concept_id": node, "name": graph.name(node)} for node in c2["candidate_ids"]]
        prompt = self.render("gaps_d3", {"question": prep.question, "grounded_targets": targets,
                                    "candidates": candidates}, d3_candidates=list(c2["candidate_ids"]))
        return self.job(prep, base, arm, "gaps", "gaps_d3", prompt, parent=parent, detector="D3",
                        candidates=tuple(c2["candidate_ids"]))

    # --- answers (shared need-aware answer template) ---------------------------
    def answer(self, prep: QuestionPrep, base: str, arm: str, payload: Mapping[str, Any],
               parent: str) -> ExperimentJob:
        allowed = tuple(chunk["chunk_id"] for chunk in payload["knowledge_chunks"])
        return self.job(prep, base, arm, "answer", "answer", self.render("answer", payload), parent=parent,
                        allowed_citations=allowed)

    # --- revision -----------------------------------------------------
    def diagnose(self, prep: QuestionPrep, base: str, arm: str, payload: Mapping[str, Any],
                 parent: str, calibration: str) -> ExperimentJob:
        return self.job(prep, base, arm, "diagnose", "diagnose", self.render("diagnose", payload),
                        parent=parent, calibration=calibration, **diagnose_schema_args(payload))

    def critique(self, prep: QuestionPrep, base: str, answer: str, parent: str) -> ExperimentJob:
        prompt = self.render("sr_critique", {"question": prep.question, "answer": answer})
        return self.job(prep, base, "SR", "critique", "sr_critique", prompt, parent=parent)

    def revise(self, prep: QuestionPrep, base: str, arm: str, payload: Mapping[str, Any], parent: str,
               calibration: str, call_index: int, allowed: Sequence[str]) -> ExperimentJob:
        template = "sr_revise" if arm == "SR" else "revise"
        return self.job(prep, base, arm, "revise", template, self.render(template, payload), parent=parent,
                        calibration=calibration, call_index=call_index, allowed_citations=tuple(allowed))

    # --- evaluation ---------------------------------------------------
    def judge(self, prep: QuestionPrep, answer: str, teacher: str) -> ExperimentJob:
        prompt = self.render("judge", judge_payload(prep.question, answer))
        return self.job(prep, "-", EVAL_ARM, EVAL_STAGE, "judge", prompt, parent=answer_sha256(answer),
                        role=teacher)


def p_answer_payload(prep: QuestionPrep, base: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Arm P: no grounding, no candidates, no KG; prerequisites are self-judged names."""
    return {"question": prep.question, "retrieval": base.upper(),
            "knowledge_chunks": ordinary_context(prep, base),
            "obligations": list(plan.get("obligations", [])), "plan": dict(plan)}


def grounded_answer_payload(prep: QuestionPrep, base: str, graph: Graph, grounding: Mapping[str, Any],
                            detection: Mapping[str, Any], order: Mapping[str, Any],
                            rep: Mapping[str, Any]) -> dict[str, Any]:
    targets = list(grounding["grounded_target_ids"])
    return {"question": prep.question, "retrieval": base.upper(),
            "knowledge_chunks": grounded_context(prep, base, graph, targets, order["order"]),
            "obligations": list(detection.get("obligations", [])),
            "support_concepts": [{"concept_id": node, "name": graph.name(node)} for node in order["order"]],
            "grounded_targets": [{"concept_id": node, "name": graph.name(node)} for node in targets]}


def detector_for(arm: str, frozen_detector: str) -> str:
    return frozen_detector if arm == "L" else ARM_DETECTOR[arm]


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
