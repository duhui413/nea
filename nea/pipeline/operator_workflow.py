"""0022 task-first five-arm pipeline; legacy runs retain their own engine."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nea.evaluation.judging import RUBRIC_HASH, rubric_payload
from nea.evaluation.measurement import METRIC_VERSION
from nea.method.templates import role_schema
from nea.pipeline.artifacts import RunDirectoryLock
from nea.pipeline.registry import BASES, JUDGE_ROLES
from nea.prompts import PromptStore
from nea.providers import config
from nea.runtime.identity import operator_manifest_identity, operator_phase_directory
from nea.runtime.operator_protocol import PROTOCOL, METHOD_VERSION
from nea.runtime.records import AppendOnlyRun, answer_sha256, digest, write_once
from nea.runtime.transport import ExperimentJob, execute, TransportStopped

ARMS = ("B", "P", "L", "F", "SR")
MODULES = ("T_EXP", "D_EXP", "D_CON", "T_CON")
BUDGET = {"max_rounds": 1, "max_candidates": 1, "max_growth": 1.2, "max_actions": 3,
          "max_validation_calls": 1, "max_graph_nodes": 24}
ROLE_STAGE = {"query_task_analysis": "prepare", "target_grounding": "ground",
              "explanation_obligation_planning": "plan", "answer_generation": "answer",
              "explanation_diagnosis": "diagnose", "generic_answer_critique": "critique",
              "answer_revision": "revise", "revision_validation": "validate",
              "answer_quality_evaluation": "judge"}
COMPARISONS = {"primary": [["F", "B"], ["F", "P"]],
               "supplementary": [["L", "P"], ["F", "L"], ["F", "SR"]],
               "paired_unit": "qid averaged over two generators and PE/RAG",
               "unit": "qid", "seed": 20260927, "bootstrap_replicates": 20000}


def public_evidence(pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reuse the existing BGE ranking; never inject extra target body blocks."""
    from nea.method.rag import select_shared_rag_context
    selected = select_shared_rag_context(list(pool))
    by_id = {str(item["chunk_id"]): item for item in pool}
    return [{"chunk_id": key, "concept_id": str(by_id[key].get("concept_id", key)),
             "concept_name": str(by_id[key].get("concept_name", "")),
             "source": str(by_id[key].get("source", "knowledge_base/concepts.jsonl")),
             "chapter": str(by_id[key].get("chapter", "")),
             "content": str(by_id[key]["content"])} for key in selected["chunk_ids"]]


def graph_support(graph: Any, target_ids: Sequence[str], max_nodes: int = 24) -> dict[str, Any]:
    """Names and structural edges only, including for PE."""
    ids = list(dict.fromkeys(target_ids))
    edges = []
    for target in target_ids:
        for edge in graph.incoming(target):
            if edge["src"] not in ids and len(ids) >= max_nodes:
                continue
            if edge["src"] not in ids:
                ids.append(edge["src"])
            edges.append(dict(edge))
    return {"targets": list(target_ids),
            "nodes": [{"concept_id": key, "name": graph.name(key), "level": graph.level(key)}
                      for key in ids], "edges": edges, "body_policy": "no_extra_body"}


def method_source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    files = ["nea/pipeline/operator_workflow.py", "nea/pipeline/operator_entry.py",
             "nea/method/operators.py", "nea/runtime/operator_protocol.py",
             "nea/runtime/transport.py", "nea/runtime/identity.py", "nea/runtime/recovery.py",
             "nea/runtime/generation_recovery.py",
             "nea/runtime/structured.py", "nea/method/templates.py", "nea/prompts.py",
             "nea/method/graph.py", "nea/method/grounding.py", "nea/method/rag.py",
             "nea/evaluation/operator_analysis.py", "nea/evaluation/measurement.py"]
    return digest({name: digest((root / name).read_bytes()) for name in files})


def select_operator_rows(rows: Sequence[Mapping[str, Any]], phase: str,
                         sample_size: int | None = None,
                         selected_qids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One scope rule shared by retrieval preparation and model execution."""
    ordered = sorted(rows, key=lambda row: row["qid"])
    if selected_qids is not None:
        if phase != "live" or sample_size is not None:
            raise ValueError("explicit qids require live and cannot combine with sample_size")
        if (isinstance(selected_qids, (str, bytes)) or not selected_qids
                or any(not isinstance(qid, str) or not qid for qid in selected_qids)
                or len(selected_qids) != len(set(selected_qids))):
            raise ValueError("explicit qids must be nonempty unique strings")
        selected = set(selected_qids)
        if not selected.issubset({row["qid"] for row in ordered}):
            raise ValueError("explicit qids must belong to the live development split")
        return [row for row in ordered if row["qid"] in selected]
    if sample_size is not None:
        if type(sample_size) is not int or not 1 <= sample_size <= len(ordered):
            raise ValueError("sample_size exceeds this phase's sealed question count")
        ordered = ordered[:sample_size]
    return ordered


@dataclass
class OperatorEnvironment:
    run_root: Path
    run_id: str
    generator: str
    question_set: Any
    graph: Any
    candidates: Mapping[str, Sequence[Mapping[str, Any]]]
    rag_pools: Mapping[str, Sequence[Mapping[str, Any]]]
    prompts_dir: Path
    router: Any
    phase: str = "test"
    sample_size: int | None = None
    selected_qids: Sequence[str] | None = None
    structure_attempts_by_provider: Mapping[str, int] | None = None
    asset_identity: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "REAL"
    budget: Mapping[str, Any] = field(default_factory=lambda: dict(BUDGET))
    enabled_modules: tuple[str, ...] = MODULES


class OperatorPhase:
    def __init__(self, env: OperatorEnvironment):
        self.env, self.split = env, env.phase
        if env.phase not in {"live", "test"}:
            raise ValueError("operator phase must be live or test")
        if env.origin not in {"REAL", "OFFLINE_FIXTURE"}:
            raise ValueError("unknown operator data origin")
        if env.origin == "REAL" and (env.question_set.sealed_problems()
                or env.question_set.counts != {"dev": 30, "test": 270}
                or (env.question_set.manifest or {}).get("version") != "nea-questions-0022-v1"):
            raise ValueError("real operator runs require the accepted 30/270 asset version")
        self.source_split = "dev" if env.phase == "live" else "test"
        self.rows = select_operator_rows(env.question_set.split_rows(self.source_split), env.phase,
                                         env.sample_size, env.selected_qids)
        if env.structure_attempts_by_provider is not None:
            policy = env.structure_attempts_by_provider
            if (env.phase != "live" or set(policy) != {"mimo", "minimax"}
                    or type(policy["mimo"]) is not int or policy["mimo"] != 3
                    or type(policy["minimax"]) is not int or not 1 <= policy["minimax"] <= 12):
                raise ValueError("custom structure budget requires live, MiMo=3, and MiniMax in 1..12")
        if not self.rows or not set(env.enabled_modules).issubset(MODULES):
            raise ValueError("empty phase or invalid module mask")
        if dict(env.budget) != BUDGET:
            raise ValueError("0022 main protocol uses the registered one-candidate budget")
        if env.phase == "test" and env.enabled_modules != MODULES:
            raise ValueError("module ablations require a separately registered optional comparison")
        self.store = PromptStore(env.prompts_dir)
        # Generic schemas enter the manifest; instance-dependent schemas also enter request keys.
        schemas = {role: role_schema(role, {}) for role in ROLE_STAGE}
        self.templates = {role: digest(self.store.text(role).encode("utf8")) for role in ROLE_STAGE}
        self.static = {role: digest({"protocol": PROTOCOL, "role": role, "budget": BUDGET,
                                    "evidence_policy": "shared-top8-no-extra-body"}) for role in ROLE_STAGE}
        qids = [row["qid"] for row in self.rows]
        locked = {"source_split": self.source_split, "qids": qids,
                  "qid_targets": {row["qid"]: row["authoring_target_id"] for row in self.rows},
                  "question_set_version": (env.question_set.manifest or {}).get("version"),
                  "question_metadata_hash": digest(env.question_set.manifest),
                  "selected_exposure": {qid: (env.question_set.manifest or {}).get("exposure", {}).get(qid, {"status": "unknown"}) for qid in qids},
                  "asset_report": (env.question_set.manifest or {}).get("report", {}),
                  "selected_questions_hash": digest(self.rows), "asset_identity": dict(env.asset_identity),
                  "prompt_files": self.store.active_hashes(), "schema_hashes": {k: digest(v) for k, v in schemas.items()},
                  "shared_template_hashes": {"repair_output": digest(self.store.text("repair_output").encode("utf8"))},
                  "method_source_hash": method_source_hash(), "budget": dict(env.budget),
                  "arms": list(ARMS), "modules": list(env.enabled_modules), "comparisons": COMPARISONS,
                  "evidence_policy": "shared-top8-no-extra-body", "nli": "disabled", "ner": "absent",
                  "optional_comparisons": [], "measurement_version": METRIC_VERSION,
                  "unimplemented_measurements": ["LTS_NEA_v1"], "sample_size": env.sample_size,
                  "origin": env.origin,
                  "context_hash": digest({qid: {"evidence": public_evidence(env.rag_pools[qid]),
                                                 "candidates": list(env.candidates[qid])} for qid in qids})}
        if env.selected_qids is not None:
            locked.update(selection_mode="explicit_qids", selected_qids=qids)
        if env.structure_attempts_by_provider is not None:
            locked["structure_attempts_by_provider"] = dict(env.structure_attempts_by_provider)
        identity = operator_manifest_identity(generator=env.generator, run_id=env.run_id, phase=env.phase,
            template_hashes=self.templates, static_payload_hashes=self.static,
            question_set_sha256=env.question_set.sha256, **locked)
        from nea.runtime.identity import model_config_identity
        identity["model_config_hash"] = digest(getattr(env.router, "model_identity",
                                                     model_config_identity(env.generator)))
        if getattr(env.router, "mimo_batch", False):
            identity["mimo_transport"] = "batch"
            from nea.runtime.batch_transport import source_hash
            identity["mimo_batch_source_hash"] = source_hash()
        phase_root = operator_phase_directory(env.run_root, env.generator, env.phase)
        from nea.runtime.source_compatibility import bind_realtime_identity
        identity = bind_realtime_identity(phase_root, identity, mimo_batch=getattr(env.router, "mimo_batch", False))
        self.run = AppendOnlyRun(phase_root, identity)
        self.parent_index: dict[tuple[str, str, str], str] = {}
        self.stop_reason: str | None = None
        self.job_results: dict[ExperimentJob, Any] = {}
        self.request_keys: dict[ExperimentJob, str] = {}
        self.generation_recovery = None

    def job(self, row: Mapping[str, Any], base: str, arm: str, role: str,
            payload: Mapping[str, Any], *, parent: str = "ROOT", teacher: str = "student1") -> ExperimentJob:
        schema = role_schema(role, payload)
        prompt = self.store.render(role, dict(payload), schema=schema)
        args = {"operator_protocol": PROTOCOL, "output_schema": json.dumps(schema, sort_keys=True),
                "schema_name": role, "repair_instruction": self.store.text("repair_output"),
                "allowed_citations": tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", []))}
        if role in {"answer_quality_evaluation", "revision_validation", "explanation_diagnosis", "generic_answer_critique"}:
            args.update(locatable_answer=payload.get("candidate_answer", "") if role in {
                            "answer_quality_evaluation", "revision_validation"} else payload.get("answer", ""),
                        locatable_parent=payload.get("answer", "") if role == "revision_validation" else "",
                        locatable_question=payload["question"],
                        allowed_evidence_ids=tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", [])))
        return ExperimentJob(qid=row["qid"], split=self.source_split, base=base, arm=arm,
            stage=ROLE_STAGE[role], role=teacher, prompt=prompt, template_hash=self.templates[role],
            static_payload_hash=self.static[role], parent_hash=parent, calibration_hash="NO_CALIBRATION",
            rubric_hash=RUBRIC_HASH, schema_args=tuple(sorted(args.items())))

    def call(self, jobs: Sequence[ExperimentJob]) -> dict[ExperimentJob, Any]:
        if not jobs:
            return {}
        if self.stop_reason:
            return {job: None for job in jobs}
        if self.generation_recovery is not None:
            with RunDirectoryLock(self.run.root):
                self.generation_recovery.import_jobs(jobs,
                    question_splits=self.env.question_set.qid_splits(), phase_split=self.source_split,
                    parent_index=self.parent_index)
        try:
            wave = execute(self.run, self.env.generator, jobs, router=self.env.router,
                question_splits=self.env.question_set.qid_splits(), phase_split=self.source_split,
                parent_index=self.parent_index,
                max_in_flight=getattr(self.env.router, "max_in_flight", 2))
            results = {job: wave["results"].get(wave["key_by_job"][job]) for job in jobs}
            self.request_keys.update(wave["key_by_job"])
        except TransportStopped as exc:
            # Preserve partial successes; resumable requests remain pending in their journal.
            from nea.runtime.transport import prepare
            _, keys = prepare(self.run, self.env.generator, jobs,
                question_splits=self.env.question_set.qid_splits(), phase_split=self.source_split,
                parent_index=self.parent_index, _model_identity=getattr(self.env.router, "model_identity", None))
            results = {job: (self.run.success(key) or {}).get("payload") for job, key in keys.items()}
            self.request_keys.update(keys)
            self.stop_reason = exc.code
        self.job_results.update(results)
        return results

    def base_payload(self, row: Mapping[str, Any], base: str) -> dict[str, Any]:
        chunks = public_evidence(self.env.rag_pools[row["qid"]]) if base == "rag" else []
        return {"question": row["question"], "retrieval": base.upper(), "knowledge_chunks": chunks,
                "citation_map": {item["chunk_id"]: item for item in chunks}}

    def run_phase(self) -> dict[str, Any]:
        # The following waves use real transport and immutable request successes.
        from nea.method import operators as op
        with RunDirectoryLock(self.run.root):
            self.run.initialize()
        cells = [(row["qid"], base) for row in self.rows for base in BASES]
        by_qid = {row["qid"]: row for row in self.rows}
        payloads = {key: self.base_payload(by_qid[key[0]], key[1]) for key in cells}
        tasks = {key: self.job(by_qid[key[0]], key[1], "P", "query_task_analysis", payloads[key]) for key in cells}
        task_out = self.call(list(tasks.values()))
        task_plans = {key: task_out[job]["parsed"] for key, job in tasks.items() if task_out[job] is not None}
        ground_jobs = {}
        for key, task in task_plans.items():
            candidates = [{"concept_id": item["concept_id"], "name": self.env.graph.name(item["concept_id"])}
                          for item in self.env.candidates[key[0]]]
            ground_jobs[key] = self.job(by_qid[key[0]], key[1], "L", "target_grounding",
                {**payloads[key], "task": task, "obligations": task.get("obligations", []), "candidates": candidates},
                parent=digest(task))
        ground_out = self.call(list(ground_jobs.values()))
        supports = {}
        grounding = {}
        for key, job in ground_jobs.items():
            result = ground_out[job]
            parsed = result["parsed"] if result else {"grounded_target_ids": []}
            allowed = {item["concept_id"] for item in self.env.candidates[key[0]]}
            ids = parsed.get("grounded_target_ids", [])
            ids = ids if isinstance(ids, list) and len(ids) <= 2 and set(ids).issubset(allowed) else []
            supports[key] = graph_support(self.env.graph, ids)
            grounding[key] = {"grounded_target_ids": ids, "status": "selected" if ids else "abstained",
                              "reason": "structure_failure" if result is None else "model_decision"}
        planning = {}
        for key, task in task_plans.items():
            for arm in ("P", "L"):
                payload = {**payloads[key], "task": task, "obligations": task.get("obligations", []),
                           "given": task.get("given", []), "uncertain": task.get("uncertain", [])}
                if arm == "L":
                    payload["graph_support"] = supports[key]
                planning[(*key, arm)] = self.job(by_qid[key[0]], key[1], arm,
                    "explanation_obligation_planning", payload, parent=digest(payload))
        plans_out = self.call(list(planning.values()))
        plans = {key: plans_out[job]["parsed"] for key, job in planning.items() if plans_out[job] is not None}
        answers = {}
        answer_jobs = {}
        for key in cells:
            for arm in ("B", "P", "L"):
                if arm != "B" and (*key, arm) not in plans:
                    continue
                payload = dict(payloads[key])
                if arm != "B":
                    payload.update(task=task_plans[key], obligations=task_plans[key].get("obligations", []),
                                   explanation_plan=plans[(*key, arm)])
                if arm == "L":
                    payload["graph_support"] = supports[key]
                answer_jobs[(*key, arm)] = self.job(by_qid[key[0]], key[1], arm,
                    "answer_generation", payload, parent=digest(payload))
        answer_out = self.call(list(answer_jobs.values()))
        for key, job in answer_jobs.items():
            if answer_out[job] is not None:
                answers[key] = answer_out[job]["parsed"]
        self.parent_index = {key: answer_sha256(text) for key, text in answers.items()}
        states = {}
        diagnosis_jobs = {}
        final_revisions = {}
        for key in cells:
            for arm, parent_arm in (("F", "L"), ("SR", "B")):
                parent = answers.get((*key, parent_arm))
                if parent is None:
                    continue
                ctx = op.OperatorContext(question=by_qid[key[0]]["question"], parent_answer=parent,
                    evidence=tuple(payloads[key]["knowledge_chunks"]),
                    obligations=tuple(task_plans.get(key, {}).get("obligations", [])) if arm == "F" else (),
                    budget=dict(self.env.budget), versions={"method": METHOD_VERSION}, base=key[1])
                states[(*key, arm)] = ctx
                payload = op.diagnose_payload(ctx, generic=arm == "SR")
                diagnosis_jobs[(*key, arm)] = self.job(by_qid[key[0]], key[1], arm,
                    "explanation_diagnosis" if arm == "F" else "generic_answer_critique",
                    payload, parent=answer_sha256(parent))
        diagnosis_out = self.call(list(diagnosis_jobs.values()))
        selected = {}
        candidate_jobs = {}
        for key, job in diagnosis_jobs.items():
            ctx, result = states[key], diagnosis_out[job]
            if result is None:
                final_revisions[key] = op.finalize(ctx, status="diagnosis_failed")
                continue
            actions = op.select_actions(ctx, result["parsed"], enabled_modules=self.env.enabled_modules,
                                        generic=key[2] == "SR")
            selected[key] = actions
            if not actions["actions"]:
                final_revisions[key] = op.finalize(ctx, status=actions["status"])
                continue
            candidate_jobs[key] = self.job(by_qid[key[0]], key[1], key[2], "answer_revision",
                op.revise_payload(ctx, actions["actions"], generic=key[2] == "SR"), parent=answer_sha256(ctx.parent_answer))
        candidates_out = self.call(list(candidate_jobs.values()))
        candidates = {}
        validation_jobs = {}
        for key, job in candidate_jobs.items():
            ctx, result = states[key], candidates_out[job]
            candidate = None if result is None else result["parsed"]
            if candidate is None:
                final_revisions[key] = op.finalize(ctx, status="revision_failed", actions=selected[key]["actions"])
                continue
            okay, reason = op.validate_candidate(ctx, candidate)
            if not okay:
                final_revisions[key] = op.finalize(ctx, candidate=candidate, status=reason,
                                                 actions=selected[key]["actions"])
                continue
            candidates[key] = candidate
            validation_jobs[key] = self.job(by_qid[key[0]], key[1], key[2], "revision_validation",
                op.validate_payload(ctx, candidate, selected[key]["actions"], generic=key[2] == "SR"),
                parent=answer_sha256(ctx.parent_answer))
        validation_out = self.call(list(validation_jobs.values()))
        for key, job in validation_jobs.items():
            result = validation_out[job]
            final_revisions[key] = op.finalize(states[key], candidate=candidates[key],
                semantic=None if result is None else result["parsed"], actions=selected[key]["actions"],
                status="validation_failed" if result is None else None)
        for key, revision in final_revisions.items():
            answers[key] = revision["answer"]
        judge_jobs = {}
        for key, answer in answers.items():
            # Complete evaluation input is arm-independent; runtime de-duplicates its full identity.
            payload = {"question": by_qid[key[0]]["question"], "candidate_answer": answer,
                       "rubric": rubric_payload(), "knowledge_chunks": payloads[key[:2]]["knowledge_chunks"],
                       "citation_map": payloads[key[:2]]["citation_map"]}
            for teacher in JUDGE_ROLES:
                judge_jobs[(*key, teacher)] = self.job(by_qid[key[0]], "-", "EVAL",
                    "answer_quality_evaluation", payload, parent=answer_sha256(answer), teacher=teacher)
        judges_out = self.call(list(judge_jobs.values()))
        judgments = []
        evaluation_sources = {}
        for (qid, base, arm, teacher), job in sorted(judge_jobs.items()):
            result = judges_out[job]
            request_key = self.request_keys.get(job)
            source = evaluation_sources.get(request_key) if request_key else None
            if request_key and source is None:
                evaluation_sources[request_key] = [qid, base, arm, teacher]
            judgments.append({"qid": qid, "generator": self.env.generator, "base": base, "arm": arm,
                "teacher": teacher, "answer_sha256": answer_sha256(answers[(qid, base, arm)]),
                "scores": None if result is None else result["parsed"]["scores"],
                "parsed": None if result is None else result["parsed"],
                "status": "missing" if result is None else "ok", "evaluation_input_hash": digest(job.prompt),
                "request_key": request_key, "reused_from": source})
        for qid, base in cells:
            for arm in ARMS:
                if (qid, base, arm) in answers:
                    continue
                for teacher in JUDGE_ROLES:
                    judgments.append({"qid": qid, "generator": self.env.generator, "base": base, "arm": arm,
                        "teacher": teacher, "answer_sha256": None, "scores": None, "parsed": None,
                        "status": "not_scheduled_missing_answer", "request_key": None, "reused_from": None})
        judgments.sort(key=lambda row: (row["qid"], row["base"], row["arm"], row["teacher"]))
        rows = []
        for qid, base in cells:
            for arm in ARMS:
                key = (qid, base, arm)
                row = {"qid": qid, "generator": self.env.generator, "base": base, "arm": arm,
                       "split": self.source_split, "phase": self.split,
                       "authoring_target_id": by_qid[qid]["authoring_target_id"],
                       "evidence": payloads[(qid, base)]["knowledge_chunks"],
                       "evidence_hash": digest(payloads[(qid, base)]["knowledge_chunks"])}
                if key not in answers:
                    row.update(status="missing", reason=self.stop_reason or "generation_failed")
                else:
                    text = answers[key]
                    row.update(status="ok", answer=text, answer_sha256=answer_sha256(text), chars=len(text))
                if arm in {"P", "L"}:
                    row["task"] = task_plans.get((qid, base))
                    row["plan"] = plans.get(key)
                if arm == "L":
                    row["grounding"] = grounding.get((qid, base))
                    row["graph_support"] = supports.get((qid, base))
                if key in final_revisions:
                    row["revision"] = {**final_revisions[key], "parent_arm": "L" if arm == "F" else "B"}
                    stage_jobs = {"diagnose": diagnosis_jobs.get(key), "revise": candidate_jobs.get(key),
                                  "validate": validation_jobs.get(key)}
                    row["revision"]["request_keys"] = {stage: self.request_keys.get(job) for stage, job in stage_jobs.items() if job}
                    row["revision"]["calls"] = {stage: len(list((self.run.root / "attempts" / self.request_keys[job]).glob("*-reserved.json")))
                        if job in self.request_keys else 0 for stage, job in stage_jobs.items()}
                    row["diagnosis"] = (diagnosis_out[diagnosis_jobs[key]] or {}).get("parsed")
                    row["action_filter"] = selected.get(key)
                rows.append(row)
        costs = ledger_costs(self.run)
        missing = [[row["qid"], row["base"], row["arm"]] for row in rows if row["status"] != "ok"]
        judge_missing = len(rows) * 2 - sum(row["scores"] is not None for row in judgments)
        summary = {"protocol": PROTOCOL, "phase": self.split, "split": self.split,
                   "generator": self.env.generator, "qids": [row["qid"] for row in self.rows],
                   "qid_targets": self.run.identity["qid_targets"], "question_set_sha256": self.env.question_set.sha256,
                   "question_set_version": self.run.identity["question_set_version"], "origin": self.env.origin,
                   "complete": not missing and not judge_missing and not self.stop_reason,
                   "missing": missing, "judgments_missing": judge_missing, "stop_reason": self.stop_reason,
                   "answers": len(rows) - len(missing), "answer_positions": len(rows), "costs": costs,
                   "optional_comparisons": [], "nli_calls": 0, "ner_calls": 0,
                   "measurements": {"existing_version": METRIC_VERSION, "status": "NOT_MEASURED",
                                    "LTS_NEA_v1": "NOT_IMPLEMENTED"}}
        with RunDirectoryLock(self.run.root):
            if summary["complete"]:
                write_once(self.run.root / "answers.json", {"rows": rows})
                write_once(self.run.root / "judgments.json", {"rows": judgments})
                write_once(self.run.root / "phase_summary.json", summary)
            else:
                # Each incomplete snapshot is immutable, while a later restart may complete the phase.
                snapshot = {"answers": rows, "judgments": judgments, "summary": summary}
                saved = sorted((self.run.root / "incomplete").glob("*.json"))
                if not saved or json.loads(saved[-1].read_text(encoding="utf8")) != snapshot:
                    write_once(self.run.root / "incomplete" / f"{len(saved) + 1:06d}.json", snapshot)
        return summary


def ledger_costs(run: AppendOnlyRun) -> dict[str, Any]:
    """Count all paid attempts, including invalid replies and transport failures."""
    attempts = sorted((run.root / "attempts").glob("*/*-reserved.json"))
    outcomes = []
    unknown = 0
    for reserved in attempts:
        path = reserved.with_name(reserved.name.replace("-reserved", "-outcome"))
        if path.is_file():
            outcomes.append(json.loads(path.read_text(encoding="utf8")))
        else:
            unknown += 1
    usage = [row["usage"] for row in outcomes if row.get("usage") is not None]
    from nea.runtime.operator_protocol import aggregate_telemetry
    telemetry = aggregate_telemetry(run.root)
    latencies = telemetry["measured_request_count"]
    return {"request_count": len(attempts), "prompt_tokens": sum(row["prompt_tokens"] for row in usage),
            "completion_tokens": sum(row["completion_tokens"] for row in usage),
            "unknown_usage_count": unknown + sum(row.get("usage") is None for row in outcomes),
            "latency_ms": telemetry["elapsed_ms"], "unknown_latency_count": len(attempts) - latencies,
            "structure_failures": sum(row["result"] == "structure_error" for row in outcomes),
            "transport_failures": sum(row["result"] == "transport_error" for row in outcomes),
            "unresolved_requests": unknown, "cost": None, "pricing_status": "NOT_CONFIGURED"}
