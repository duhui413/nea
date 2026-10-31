"""Eight-arm formal workflow with same-parent, registered comparisons.

All model work uses the isolated formal transport. The old workflow, prompt
registry and its source identity are deliberately left untouched.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nea.evaluation.judging import RUBRIC_HASH
from nea.method import formal_operators as op
from nea.method.templates import role_schema as legacy_role_schema, obj, string, array
from nea.pipeline.artifacts import RunDirectoryLock
from nea.pipeline.operator_workflow import (OperatorEnvironment, graph_support,
    public_evidence, select_operator_rows)
from nea.pipeline.registry import BASES
from nea.providers import config
from nea.runtime.identity import model_config_identity
from nea.runtime.formal_protocol import ARMS, ROLE_STAGE, method_source_hash as formal_source_hash
from nea.runtime.records import AppendOnlyRun, answer_sha256, digest, write_once

PROTOCOL = "nea-formal8-v1"
PARENTS = {"F": "L", "SRL": "L", "FB": "B", "SR": "B", "CV": "B"}
BUDGET, MODULES = op.BUDGET, op.MODULES
ABLATIONS, SENSITIVITIES = op.ABLATIONS, op.SENSITIVITIES


def role_schema(role: str, payload: Mapping[str, Any] | None = None) -> dict | None:
    payload = payload or {}
    if role == "formal_explanation_diagnosis":
        schema = deepcopy(legacy_role_schema("explanation_diagnosis", payload))
        schema["anyOf"][0]["properties"]["actions"]["maxItems"] = 5
        return schema
    if role == "cove_verification_questions":
        return obj({"verification_questions": array(obj({"question_id": string("v1", "v2", "v3"),
                     "question": string()}), maxItems=3)})
    if role == "cove_verification_answer":
        ids = [item["question_id"] for item in payload.get("verification_questions", [])]
        return obj({"verification_answers": array(obj({"question_id": string(*ids),
            "answer": string(), "status": string("answered", "uncertain")}), maxItems=3)})
    if role == "cove_revision":
        return None
    if role.startswith("formal_"):
        from nea.evaluation.formal_measurement import role_schema as measurement_schema
        return measurement_schema(role, payload)
    return legacy_role_schema(role, payload)


@dataclass
class FormalEnvironment(OperatorEnvironment):
    run_ablations: bool = True
    run_sensitivity: bool = True


class FormalPhase:
    def __init__(self, env: FormalEnvironment):
        from nea.runtime.formal_protocol import FormalPromptStore
        self.env, self.split = env, env.phase
        if env.phase not in {"live", "test", "case"}:
            raise ValueError("formal phase must be live, test or case")
        if env.origin not in {"REAL", "OFFLINE_FIXTURE"}:
            raise ValueError("unknown formal data origin")
        if dict(env.budget) != BUDGET or tuple(env.enabled_modules) != MODULES:
            raise ValueError("main protocol uses its default budget and all modules; use registered comparisons")
        if env.structure_attempts_by_provider is not None:
            raise ValueError("formal structure attempts are fixed by the formal protocol")
        self.source_split = "dev" if env.phase == "live" else env.phase
        self.rows = select_operator_rows(env.question_set.split_rows(self.source_split), env.phase,
                                        env.sample_size, env.selected_qids)
        if not self.rows:
            raise ValueError("empty formal phase")
        self.store = FormalPromptStore(env.prompts_dir)
        self.templates = {role: digest(self.store.text(role).encode("utf8")) for role in ROLE_STAGE}
        self.static = {role: digest({"protocol": PROTOCOL, "role": role,
            "budget": BUDGET, "ablations": ABLATIONS, "sensitivities": SENSITIVITIES,
            "evidence_policy": "shared-top8-no-extra-body"}) for role in ROLE_STAGE}
        schemas = {role: role_schema(role) for role in ROLE_STAGE}
        qids = [row["qid"] for row in self.rows]
        metadata = env.question_set.manifest or {}
        generator = config.experiment_role_models_for(env.generator)["student1"]
        identity = {"protocol_version": PROTOCOL, "method_version": PROTOCOL,
            "run_id": env.run_id, "split": env.phase, "question_split": self.source_split,
            "generator": generator, "origin": env.origin, "arms": list(ARMS),
            "parents": PARENTS, "budget": BUDGET, "modules": list(MODULES),
            "question_set_sha256": env.question_set.sha256,
            "question_set_version": metadata.get("version"),
            "question_metadata_hash": digest(metadata), "selected_questions_hash": digest(self.rows),
            "qids": qids, "qid_targets": {row["qid"]: row["authoring_target_id"] for row in self.rows},
            "qid_metadata": {row["qid"]: {"question_type": row.get("style"),
                "graph_level": env.graph.level(row["authoring_target_id"]),
                "has_gap": "gap" in row.get("gold_labels", {}).values()} for row in self.rows},
            "selected_exposure": {qid: metadata.get("exposure", {}).get(qid, {"status": "unknown"}) for qid in qids},
            "asset_identity": dict(env.asset_identity), "template_hashes": self.templates,
            "static_payload_hashes": self.static, "prompt_files": self.store.active_hashes(),
            "schema_hashes": {key: digest(value) for key, value in schemas.items()},
            "shared_template_hashes": {"repair_output": digest(self.store.text("repair_output").encode("utf8"))},
            "method_source_hash": formal_source_hash(), "rubric_hash": RUBRIC_HASH,
            "model_config_hash": digest(getattr(env.router, "model_identity", model_config_identity(env.generator))),
            "comparisons_registry": {"ablations": ABLATIONS, "sensitivities": SENSITIVITIES},
            "run_ablations": env.run_ablations and env.phase != "case",
            "run_sensitivity": env.run_sensitivity and env.phase == "live",
            "sample_size": env.sample_size, "selected_qids": env.selected_qids,
            "context_hash": digest({qid: {"evidence": public_evidence(env.rag_pools[qid]),
                "candidates": list(env.candidates[qid])} for qid in qids}),
            "evidence_policy": "shared-top8-no-extra-body", "nli": "disabled", "ner": "absent"}
        if getattr(env.router, "mimo_batch", False):
            identity["mimo_transport"] = "batch"
            from nea.runtime.batch_transport import source_hash
            identity["mimo_batch_source_hash"] = source_hash()
        phase_root = env.run_root / "operator-v2" / generator.replace("/", "__") / env.phase
        from nea.runtime.source_compatibility import bind_realtime_identity
        identity = bind_realtime_identity(phase_root, identity, mimo_batch=getattr(env.router, "mimo_batch", False))
        from nea.runtime.batch_migration import bind
        identity = bind(phase_root, identity, env.router)
        self.run = AppendOnlyRun(phase_root, identity)
        self.parent_index, self.request_keys, self.job_results = {}, {}, {}
        self.stop_reason = None
        self._stage_jobs: dict[tuple, dict[str, Any]] = {}
        self._scheduled_jobs = set()
        self._by_qid = {row["qid"]: row for row in self.rows}

    def job(self, row: Mapping[str, Any], base: str, arm: str, role: str,
            payload: Mapping[str, Any], *, parent: str = "ROOT", teacher: str = "student1",
            comparison: str = "main", round_index: int = 1):
        from nea.runtime.formal_transport import ExperimentJob
        schema = role_schema(role, payload)
        args = {"operator_protocol": PROTOCOL, "output_schema": json.dumps(schema, sort_keys=True),
            "schema_name": role, "repair_instruction": self.store.text("repair_output"),
            "allowed_citations": tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", [])),
            "comparison": comparison, "round_index": round_index,
            "measurement_concept_ids": tuple(item.get("concept_id", item.get("id")) for item in payload.get("concepts", [])),
            "verification_question_ids": tuple(item["question_id"] for item in payload.get("verification_questions", [])),
            "locatable_answer": payload.get("candidate_answer", payload.get("answer", "")),
            "locatable_parent": payload.get("answer", "") if role == "revision_validation" else "",
            "locatable_question": payload.get("question", ""),
            "allowed_evidence_ids": tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", []))}
        return ExperimentJob(qid=row["qid"], split=self.source_split, base=base, arm=arm,
            stage=ROLE_STAGE[role], role=teacher, prompt=self.store.render(role, dict(payload), schema=schema),
            template_hash=self.templates[role], static_payload_hash=self.static[role], parent_hash=parent,
            calibration_hash="NO_CALIBRATION", rubric_hash=RUBRIC_HASH, call_index=round_index,
            schema_args=tuple(sorted(args.items())))

    def call(self, jobs: Sequence[Any]) -> dict[Any, Any]:
        from nea.runtime.formal_transport import execute, prepare, TransportStopped
        if not jobs or self.stop_reason:
            return {job: None for job in jobs}
        try:
            wave = execute(self.run, self.env.generator, jobs, router=self.env.router,
                question_splits=self.env.question_set.qid_splits(), phase_split=self.source_split,
                parent_index=self.parent_index, max_in_flight=getattr(self.env.router, "max_in_flight", 2))
            self.request_keys.update(wave["key_by_job"])
            results = {job: wave["results"].get(wave["key_by_job"][job]) for job in jobs}
        except TransportStopped as exc:
            _, keys = prepare(self.run, self.env.generator, jobs,
                question_splits=self.env.question_set.qid_splits(), phase_split=self.source_split,
                parent_index=self.parent_index, _model_identity=getattr(self.env.router, "model_identity", None))
            self.request_keys.update(keys)
            results = {job: (self.run.success(key) or {}).get("payload") for job, key in keys.items()}
            self.stop_reason = exc.code
        self.job_results.update(results)
        return results

    def base_payload(self, row: Mapping[str, Any], base: str) -> dict[str, Any]:
        chunks = public_evidence(self.env.rag_pools[row["qid"]]) if base == "rag" else []
        return {"question": row["question"], "retrieval": base.upper(), "knowledge_chunks": chunks,
                "citation_map": {item["chunk_id"]: item for item in chunks}}

    def _wave(self, jobs: Mapping[tuple, Any]) -> dict[tuple, Any]:
        self._scheduled_jobs.update(jobs.values())
        results = self.call(list(jobs.values()))
        return {key: results[job]["parsed"] for key, job in jobs.items() if results[job] is not None}

    def _context(self, key, answer, budget=None, *, generic=False, max_chars=None):
        limits = {**(budget or BUDGET), "max_rounds": 1}
        if max_chars is not None:
            limits["max_chars"] = min(max_chars, int(len(answer) * limits["max_growth"]))
        return op.FormalOperatorContext(question=self._by_qid[key[0]]["question"], parent_answer=answer,
            evidence=tuple(self._payloads[key[:2]]["knowledge_chunks"]),
            obligations=() if generic else tuple(self._tasks.get(key[:2], {}).get("obligations", [])),
            budget=limits, versions={"method": PROTOCOL}, base=key[1])

    def _revision_input_hash(self, ctx, actions):
        return digest({"context": ctx.context_sha256, "actions": actions,
            "revision_template": self.templates["answer_revision"],
            "validation_template": self.templates["revision_validation"],
            "revision_contract": self.static["answer_revision"],
            "validation_contract": self.static["revision_validation"]})

    def _operators(self, contexts, *, comparison="main", diagnoses=None, selected=None, round_index=1):
        """Batched diagnosis/revision/validation with reusable fixed parent evidence."""
        jobs, stages = {}, {key: {} for key in contexts}
        for key, ctx in contexts.items():
            self.parent_index[(*key[:2], PARENTS[key[2]], comparison, round_index)] = ctx.parent_sha256
        if diagnoses is None:
            for key, ctx in contexts.items():
                generic = key[2] in {"SR", "SRL"}
                jobs[key] = self.job(self._by_qid[key[0]], key[1], key[2],
                    "generic_answer_critique" if generic else "formal_explanation_diagnosis",
                    op.diagnose_payload(ctx, generic=generic), parent=ctx.parent_sha256,
                    comparison=comparison, round_index=round_index)
                stages[key]["diagnose"] = jobs[key]
            diagnoses = self._wave(jobs)
        selected = dict(selected or {})
        final, revisions = {}, {}
        for key, ctx in contexts.items():
            if key not in diagnoses:
                final[key] = op.finalize(ctx, status="diagnosis_failed")
                continue
            chosen = selected.setdefault(key, op.select_actions(ctx, diagnoses[key], generic=key[2] in {"SR", "SRL"}))
            if not chosen["actions"]:
                final[key] = op.finalize(ctx, status=chosen["status"], diagnosis=diagnoses[key], rejections=chosen["rejections"])
                continue
            revisions[key] = self.job(self._by_qid[key[0]], key[1], key[2], "answer_revision",
                op.revise_payload(ctx, chosen["actions"], generic=key[2] in {"SR", "SRL"}),
                parent=ctx.parent_sha256, comparison=comparison, round_index=round_index)
            stages[key]["revise"] = revisions[key]
        candidates = self._wave(revisions)
        validations = {}
        for key, job in revisions.items():
            ctx, chosen = contexts[key], selected[key]
            candidate = candidates.get(key)
            valid, reason = op.validate_candidate(ctx, candidate)
            if not valid:
                final[key] = op.finalize(ctx, candidate=candidate, status=reason if candidate is not None else "revision_failed",
                    actions=chosen["actions"], diagnosis=diagnoses[key], rejections=chosen["rejections"])
                continue
            validations[key] = self.job(self._by_qid[key[0]], key[1], key[2], "revision_validation",
                op.validate_payload(ctx, candidate, chosen["actions"], generic=key[2] in {"SR", "SRL"}),
                parent=ctx.parent_sha256, comparison=comparison, round_index=round_index)
            stages[key]["validate"] = validations[key]
        verdicts = self._wave(validations)
        for key in validations:
            final[key] = op.finalize(contexts[key], candidates[key], verdicts.get(key),
                status=None if key in verdicts else "validation_failed", actions=selected[key]["actions"],
                diagnosis=diagnoses[key], rejections=selected[key]["rejections"])
        for key, record in final.items():
            record["request_keys"] = {stage: self.request_keys.get(job) for stage, job in stages[key].items()}
            record["calls"] = {stage: len(list((self.run.root / "attempts" / self.request_keys[job]).glob("*-reserved.json")))
                if job in self.request_keys else 0 for stage, job in stages[key].items()}
            record["parent_arm"] = PARENTS[key[2]]
            record["comparison"] = comparison
            record["action_filter"] = selected.get(key)
            record["input_hash"] = self._revision_input_hash(contexts[key], record.get("actions", []))
        return final, diagnoses

    def _cove(self, answers):
        jobs, contexts, stages = {}, {}, {}
        for cell in self._cells:
            parent = answers.get((*cell, "B"))
            if parent is None:
                continue
            key = (*cell, "CV")
            contexts[key] = self._context(key, parent, generic=True)
            payload = {**self._payloads[cell], "answer": parent, "max_verification_questions": 3}
            jobs[key] = self.job(self._by_qid[cell[0]], cell[1], "CV", "cove_verification_questions",
                                 payload, parent=answer_sha256(parent))
            stages[key] = {"cv_questions": jobs[key]}
        questions = self._wave(jobs)
        independent = {}
        for key, result in questions.items():
            rows = result["verification_questions"]
            if len({item["question_id"] for item in rows}) != len(rows):
                continue
            # Deliberately omit the draft, planning, obligations and NEA labels.
            payload = {**self._payloads[key[:2]], "verification_questions": rows}
            independent[key] = self.job(self._by_qid[key[0]], key[1], "CV", "cove_verification_answer",
                payload, parent=contexts[key].parent_sha256)
            stages[key]["cv_answer"] = independent[key]
        verification = self._wave(independent)
        revisions = {}
        for key, result in verification.items():
            expected = [item["question_id"] for item in questions[key]["verification_questions"]]
            actual = [item["question_id"] for item in result["verification_answers"]]
            if sorted(expected) != sorted(actual):
                continue
            ctx = contexts[key]
            payload = {**self._payloads[key[:2]], "answer": ctx.parent_answer,
                "verification_questions": questions[key]["verification_questions"],
                "verification_answers": result["verification_answers"], "max_chars": ctx.budget["max_chars"],
                "allowed_citations": list(ctx.allowed_citations)}
            revisions[key] = self.job(self._by_qid[key[0]], key[1], "CV", "cove_revision", payload,
                                     parent=ctx.parent_sha256)
            stages[key]["revise"] = revisions[key]
        candidates = self._wave(revisions)
        final = {}
        for key, ctx in contexts.items():
            candidate = candidates.get(key)
            valid, reason = op.validate_candidate(ctx, candidate)
            if candidate is None:
                reason = "cove_stage_failed"
            answer = candidate if valid else ctx.parent_answer
            candidate_record = None if candidate is None else {"answer": candidate,
                "answer_sha256": answer_sha256(candidate), "status": "accepted" if valid else reason,
                "semantic": None}
            final[key] = {"answer": answer, "answer_sha256": answer_sha256(answer),
                "parent_sha256": ctx.parent_sha256, "context_sha256": ctx.context_sha256,
                "operator_version": "nea-cove-v1", "parent_arm": "B", "status": "accepted" if valid else "parent_retained",
                "revised": valid, "fallback_reason": None if valid else reason,
                "candidate_record": candidate_record, "candidate_records": [] if candidate_record is None else [candidate_record],
                "verification_questions": questions.get(key), "verification_answers": verification.get(key),
                "validation_policy": "independent-verification-answers-and-structural-rules",
                "request_keys": {stage: self.request_keys.get(job) for stage, job in stages[key].items()},
                "calls": {stage: len(list((self.run.root / "attempts" / self.request_keys[job]).glob("*-reserved.json")))
                    if job in self.request_keys else 0 for stage, job in stages[key].items()}, "comparison": "main"}
        return final

    def _answer_row(self, key, answer=None, *, revision=None, comparison="main", reused_from=None):
        source = self._by_qid[key[0]]
        evidence = self._payloads[key[:2]]["knowledge_chunks"]
        row = {"qid": key[0], "base": key[1], "arm": key[2], "generator": self.env.generator,
            "question": source["question"], "authoring_target_id": source["authoring_target_id"],
            "split": self.source_split, "phase": self.split, "comparison": comparison,
            "evidence": evidence, "evidence_hash": digest(evidence), "reused_from": reused_from}
        if answer is None:
            row.update(status="missing", reason=self.stop_reason or "generation_failed")
        else:
            row.update(status="ok", answer=answer, answer_sha256=answer_sha256(answer), chars=len(answer))
        if revision is not None:
            row.update(revision=revision, parent_sha256=revision["parent_sha256"],
                       diagnosis=revision.get("diagnosis"), action_filter=revision.get("action_filter"))
        return row

    def _ablation_rows(self, contexts, main, diagnoses):
        rows = []
        cached = {(key[:2], value["input_hash"]): ("F", value)
                  for key, value in main.items() if key[2] == "F"}
        for name, mask in ABLATIONS.items():
            comparison = "ablation:" + name
            pending, selected, resolved, reuse = {}, {}, {}, {}
            for key, ctx in contexts.items():
                if key[2] != "F":
                    continue
                diagnosis = diagnoses.get(key)
                if diagnosis is None:
                    continue
                chosen = op.select_actions(ctx, diagnosis, enabled_modules=mask["modules"], shared=mask["shared"])
                source = main[key]
                input_hash = self._revision_input_hash(ctx, chosen["actions"])
                cache_key = (key[:2], input_hash)
                if cache_key in cached:
                    source_name, source_record = cached[cache_key]
                    resolved[key], reuse[key] = deepcopy(source_record), source_name
                elif not chosen["actions"]:
                    resolved[key], reuse[key] = op.finalize(ctx, status=chosen["status"], diagnosis=diagnosis,
                        rejections=chosen["rejections"]), "L"
                else:
                    pending[key], selected[key] = ctx, chosen
                if key in resolved:
                    resolved[key].update(input_hash=input_hash, actions=chosen["actions"],
                        action_filter=chosen, parent_arm="L", comparison=comparison,
                        request_keys={}, calls={})
            generated, _ = self._operators(pending, comparison=comparison, diagnoses=diagnoses, selected=selected)
            resolved.update(generated)
            for key, record in resolved.items():
                cached.setdefault((key[:2], record["input_hash"]), (comparison, record))
            for cell in self._cells:
                key = (*cell, "F")
                revision = resolved.get(key)
                rows.append(self._answer_row(key, revision["answer"] if revision else None,
                    revision=revision, comparison=comparison, reused_from=reuse.get(key)))
        return rows

    def _sensitivity_rows(self, answers, contexts, main, diagnoses, supports, grounded):
        rows = []
        for name, change in SENSITIVITIES.items():
            comparison = "sensitivity:" + name
            budget = op.registered_budget("live", comparison)
            pending, reuse, resolved, variant_parents, variant_plans, variant_support = {}, {}, {}, {}, {}, {}
            if "max_graph_nodes" in change:
                altered = {cell: graph_support(self.env.graph, grounded.get(cell, []), change["max_graph_nodes"])
                           for cell in self._tasks}
                changed = {cell for cell in altered if altered[cell] != supports.get(cell)}
                plan_jobs = {cell: self.job(self._by_qid[cell[0]], cell[1], "L", "explanation_obligation_planning",
                    {**self._payloads[cell], "task": self._tasks[cell], "obligations": self._tasks[cell].get("obligations", []),
                     "given": self._tasks[cell].get("given", []), "uncertain": self._tasks[cell].get("uncertain", []),
                     "graph_support": altered[cell]}, parent=digest(altered[cell]), comparison=comparison) for cell in changed}
                plans = self._wave(plan_jobs)
                answer_jobs = {cell: self.job(self._by_qid[cell[0]], cell[1], "L", "answer_generation",
                    {**self._payloads[cell], "task": self._tasks[cell], "obligations": self._tasks[cell].get("obligations", []),
                     "explanation_plan": plans[cell], "graph_support": altered[cell]}, parent=digest(plans[cell]),
                     comparison=comparison) for cell in plans}
                parents = self._wave(answer_jobs)
                variant_parents, variant_plans, variant_support = parents, plans, altered
                for cell in self._cells:
                    key = (*cell, "F")
                    if cell not in changed and key in main:
                        resolved[key], reuse[key] = deepcopy(main[key]), "F"
                    elif cell in parents:
                        pending[key] = self._context(key, parents[cell], budget)
                generated, _ = self._operators(pending, comparison=comparison)
            elif name == "rounds_2":
                for key, ctx in contexts.items():
                    if key[2] != "F":
                        continue
                    first = main[key]
                    if not first["revised"]:
                        resolved[key], reuse[key] = deepcopy(first), "F"
                    else:
                        pending[key] = self._context(key, first["answer"], budget, max_chars=ctx.budget["max_chars"])
                generated, _ = self._operators(pending, comparison=comparison, round_index=2)
                for key, second in generated.items():
                    first = main[key]
                    second["round_records"] = [deepcopy(first), deepcopy(second)]
                    second["parent_sha256"] = contexts[key].parent_sha256
                    second["revised"] = second["answer_sha256"] != contexts[key].parent_sha256
                    second["status"] = "accepted" if second["revised"] else "parent_retained"
                    second["candidate_records"] = first.get("candidate_records", []) + second.get("candidate_records", [])
            else:
                pending = {key: self._context(key, ctx.parent_answer, budget)
                           for key, ctx in contexts.items() if key[2] == "F"}
                generated, _ = self._operators(pending, comparison=comparison, diagnoses=diagnoses)
            resolved.update(generated)
            for cell in self._cells:
                key = (*cell, "F")
                revision = resolved.get(key)
                if revision and key in reuse:
                    revision.update(comparison=comparison, request_keys={}, calls={})
                row = self._answer_row(key, revision["answer"] if revision else None,
                    revision=revision, comparison=comparison, reused_from=reuse.get(key))
                row["budget"] = budget
                if cell in variant_support:
                    row.update(graph_support=variant_support[cell], plan=variant_plans.get(cell),
                        parent_answer=variant_parents.get(cell, answers.get((*cell, "L"))))
                rows.append(row)
        return rows

    def run_phase(self) -> dict[str, Any]:
        self._scheduled_jobs = set()
        with RunDirectoryLock(self.run.root):
            self.run.initialize()
        self._cells = [(row["qid"], base) for row in self.rows for base in BASES]
        self._payloads = {cell: self.base_payload(self._by_qid[cell[0]], cell[1]) for cell in self._cells}
        tasks = {cell: self.job(self._by_qid[cell[0]], cell[1], "P", "query_task_analysis", self._payloads[cell])
                 for cell in self._cells}
        self._tasks = self._wave(tasks)
        ground_jobs = {cell: self.job(self._by_qid[cell[0]], cell[1], "L", "target_grounding",
            {**self._payloads[cell], "task": task, "obligations": task.get("obligations", []),
             "candidates": [{"concept_id": item["concept_id"], "name": self.env.graph.name(item["concept_id"])}
                            for item in self.env.candidates[cell[0]]]}, parent=digest(task)) for cell, task in self._tasks.items()}
        ground = self._wave(ground_jobs)
        grounded = {cell: ground.get(cell, {}).get("grounded_target_ids", []) for cell in self._tasks}
        supports = {cell: graph_support(self.env.graph, ids) for cell, ids in grounded.items()}
        plan_jobs = {}
        for cell, task in self._tasks.items():
            for arm in ("P", "L"):
                payload = {**self._payloads[cell], "task": task, "obligations": task.get("obligations", []),
                           "given": task.get("given", []), "uncertain": task.get("uncertain", [])}
                if arm == "L":
                    payload["graph_support"] = supports[cell]
                plan_jobs[(*cell, arm)] = self.job(self._by_qid[cell[0]], cell[1], arm,
                    "explanation_obligation_planning", payload, parent=digest(payload))
        plans = self._wave(plan_jobs)
        answer_jobs = {}
        for cell in self._cells:
            for arm in ("B", "P", "L"):
                key = (*cell, arm)
                if arm != "B" and key not in plans:
                    continue
                payload = dict(self._payloads[cell])
                if arm != "B":
                    payload.update(task=self._tasks[cell], obligations=self._tasks[cell].get("obligations", []), explanation_plan=plans[key])
                if arm == "L":
                    payload["graph_support"] = supports[cell]
                answer_jobs[key] = self.job(self._by_qid[cell[0]], cell[1], arm, "answer_generation", payload, parent=digest(payload))
        answers = self._wave(answer_jobs)
        self.parent_index = {key: answer_sha256(text) for key, text in answers.items()}
        contexts = {(*cell, arm): self._context((*cell, arm), answers[(*cell, parent)], generic=arm in {"SR", "SRL"})
                    for cell in self._cells for arm, parent in PARENTS.items()
                    if arm != "CV" and (*cell, parent) in answers}
        revisions, diagnoses = self._operators(contexts)
        revisions.update(self._cove(answers))
        answers.update({key: value["answer"] for key, value in revisions.items()})
        rows = []
        for cell in self._cells:
            for arm in ARMS:
                key = (*cell, arm)
                row = self._answer_row(key, answers.get(key), revision=revisions.get(key))
                if arm in {"P", "L"}:
                    row.update(task=self._tasks.get(cell), plan=plans.get(key))
                if arm == "L":
                    row.update(grounding={"grounded_target_ids": grounded.get(cell, [])}, graph_support=supports.get(cell))
                if key in answer_jobs:
                    row["request_key"] = self.request_keys.get(answer_jobs[key])
                rows.append(row)
        ablations = self._ablation_rows(contexts, revisions, diagnoses) if self.run.identity["run_ablations"] else []
        sensitivity = self._sensitivity_rows(answers, contexts, revisions, diagnoses, supports, grounded) if self.run.identity["run_sensitivity"] else []
        validation = []
        for row in rows:
            revision = row.get("revision", {})
            if row["arm"] != "F" or revision.get("fallback_reason") not in {"semantic_rejected", "semantic_uncertain"}:
                continue
            candidate = revision["candidate_record"]
            alternative = self._answer_row((row["qid"], row["base"], "F"), candidate["answer"],
                revision=revision, comparison="without_verification", reused_from="F.candidate")
            alternative["protected_answer_sha256"] = row["answer_sha256"]
            alternative["generation_calls"] = 0
            validation.append(alternative)
        missing = [[row["qid"], row["base"], row["arm"], row["comparison"]]
                   for row in rows + ablations + sensitivity if row["status"] != "ok"]
        method_failures = [[row["qid"], row["base"], row["arm"], row["comparison"]]
            for row in rows + ablations + sensitivity if str(row.get("revision", {}).get("fallback_reason", "")).endswith("_failed")]
        stage_failures = sorted(({"qid": job.qid, "base": job.base, "arm": job.arm,
            "comparison": job.arg("comparison", "main"), "stage": job.stage,
            "round_index": job.arg("round_index", 1), "request_key": self.request_keys.get(job)} for job in self._scheduled_jobs
            if self.job_results.get(job) is None),
            key=lambda row: (row["qid"], row["base"], row["arm"], row["comparison"], row["stage"], row["round_index"], row["request_key"] or ""))
        for row in stage_failures:
            cell = [row["qid"], row["base"], row["arm"], row["comparison"]]
            if cell not in method_failures:
                method_failures.append(cell)
        method_failures.sort()
        summary = {"protocol": PROTOCOL, "phase": self.split, "split": self.split,
            "generator": self.env.generator, "origin": self.env.origin, "qids": [row["qid"] for row in self.rows],
            "qid_targets": self.run.identity["qid_targets"], "question_set_sha256": self.env.question_set.sha256,
            "question_set_version": self.run.identity["question_set_version"], "answer_positions": len(rows),
            "answers": sum(row["status"] == "ok" for row in rows), "missing": missing,
            "method_failures": method_failures, "stage_failures": stage_failures,
            "generation_complete": not missing and not method_failures and not self.stop_reason,
            "complete": False, "judgments_missing": len(rows) * 2, "stop_reason": self.stop_reason,
            "nli_calls": 0, "ner_calls": 0,
            "comparison_positions": {"ablations": len(ablations), "sensitivity": len(sensitivity),
                                     "validation_ablation": len(validation)}}
        result = {"answers": rows, "judgments": [], "summary": summary, "ablations": ablations,
                  "sensitivity": sensitivity, "validation_ablation": validation}
        with RunDirectoryLock(self.run.root):
            if summary["generation_complete"]:
                write_once(self.run.root / "answers.json", {"rows": rows})
                write_once(self.run.root / "generation_summary.json", summary)
                for filename, records in (("ablations.json", ablations), ("sensitivity.json", sensitivity),
                                           ("validation-ablation.json", validation)):
                    write_once(self.run.root / filename, {"rows": records})
            else:
                write_once(self.run.root / "snapshots" / digest(result) / "generation.json", result)
        return result
