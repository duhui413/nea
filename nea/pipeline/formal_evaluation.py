"""Append-only formal scoring, need measurements and blind audits."""
from __future__ import annotations

from dataclasses import replace
import json
import threading
from typing import Any, Mapping, Sequence

from nea.evaluation.formal_measurement import (AUDIT_ARMS, PROTOCOL,
    blind_concept_payload, classification_agreement, concept_metrics, cross_family_teacher,
    select_t90, summarize_concept_metrics)
from nea.evaluation.judging import rubric_payload
from nea.pipeline.artifacts import RunDirectoryLock, run_directory_guard
from nea.runtime.records import answer_sha256, digest, write_once


def _job(phase, row, role, payload, *, parent, teacher):
    job = phase.job(row, "-", "EVAL", role, payload, parent=parent, teacher=teacher)
    args = dict(job.schema_args)
    if role == "formal_concept_measurement":
        args["measurement_concept_ids"] = tuple(item["concept_id"] for item in payload["concepts"])
    if role in {"formal_defect_audit", "answer_quality_evaluation"}:
        args.update(locatable_answer=payload["candidate_answer"], locatable_question=payload["question"],
                    allowed_evidence_ids=tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", [])))
    return replace(job, schema_args=tuple(sorted(args.items())))


def _parsed(results, job):
    result = results.get(job)
    return result.get("parsed") if result else None


def _identity(phase, record):
    return {"qid": record["qid"], "generator": phase.env.generator, "base": record["base"],
            "arm": record["arm"], "comparison": record.get("comparison", "main"),
            "answer_sha256": answer_sha256(record["answer"]) if record.get("answer") else None}


def _save(phase, name, rows, **extra):
    complete = all(row.get("status") == "ok" for row in rows) and not phase.stop_reason
    artifact = {"protocol": PROTOCOL, "origin": phase.env.origin,
                "status": "COMPLETE" if complete else "NOT_RUN", "rows": rows, **extra}
    with run_directory_guard(phase.run.root):
        target = (phase.run.root / f"{name}.json" if complete else
                  phase.run.root / "incomplete" / f"{name}-{digest(artifact)}.json")
        write_once(target, artifact)
    return artifact


def _run_stages(functions, names):
    """Run independent evaluation stages in named threads (the name selects the stage's transport lock)."""
    outputs, errors = [None] * len(functions), []

    def run(index):
        try:
            outputs[index] = functions[index]()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
    threads = [threading.Thread(target=run, args=(i,), name="formal8-stage-" + names[i]) for i in range(len(functions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return outputs


def run_evaluations(phase: Any, records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
                    *, scoring_only: bool = False) -> dict:
    """Use an initialized FormalPhase; resumed requests reuse its success journal.

    Extra records with a non-main ``comparison`` receive quality scores only.
    They share the arm-independent evaluator identity with main answers.
    """
    if records is None:
        records = json.loads((phase.run.root / "answers.json").read_text(encoding="utf8"))["rows"]
        if not scoring_only:
            for filename in ("ablations.json", "sensitivity.json", "validation-ablation.json"):
                path = phase.run.root / filename
                if path.is_file():
                    records.extend(json.loads(path.read_text(encoding="utf8"))["rows"])
    if isinstance(records, Mapping):
        records = [row for kind in ("answers", "ablations", "sensitivity", "validation_ablation")
                   for row in records.get(kind, [])]
    records = sorted((dict(row) for row in records), key=lambda row:
                     (row["qid"], row["base"], row.get("comparison", "main"), row["arm"]))
    questions = {row["qid"]: row for row in phase.rows}
    main = [row for row in records if row.get("comparison", "main") == "main"]
    scoring = []
    for record in records:
        row = questions[record["qid"]]
        for teacher in ("teacher1", "teacher2"):
            job = None
            if record.get("answer"):
                payload = {"question": row["question"], "candidate_answer": record["answer"],
                           "rubric": rubric_payload(), "knowledge_chunks": record.get("evidence", [])}
                job = _job(phase, row, "answer_quality_evaluation", payload,
                           parent=answer_sha256(record["answer"]), teacher=teacher)
            scoring.append((record, teacher, job))
    results = phase.call(list(dict.fromkeys(job for _, _, job in scoring if job)))
    judgments = []
    for record, teacher, job in scoring:
        parsed = _parsed(results, job)
        judgments.append({**_identity(phase, record), "teacher": teacher, "parsed": parsed,
            "scores": parsed["scores"] if parsed else None, "status": "ok" if parsed else "missing",
            "evaluation_input_hash": digest(job.prompt) if job else None,
            "request_key": phase.request_keys.get(job), "reused_from": None})
    independent_judgments = {row["request_key"]: row for row in judgments if row["parsed"] is not None}
    quality_name = ("quality/" + digest([_identity(phase, row) for row in records]) + "/judgments"
                    if scoring_only else "judgments")
    judgment_artifact = _save(phase, quality_name, judgments,
        invalid_localization_fields=sum((row["parsed"] or {}).get("localization", {}).get("invalid_fields", 0)
                                        for row in independent_judgments.values()),
        position_invalid_localization_fields=sum((row["parsed"] or {}).get("localization", {}).get("invalid_fields", 0)
                                                 for row in judgments))
    if scoring_only or phase.env.phase == "case":
        return {"judgments": judgments, "measurements": [], "audits": [],
                "summary": {"protocol": PROTOCOL, "origin": phase.env.origin,
                    "status": judgment_artifact["status"], "judgments": len(judgments),
                    "judgments_missing": sum(row["status"] != "ok" for row in judgments),
                    "measurements": 0, "audits": 0, "stop_reason": phase.stop_reason}}

    independent = cross_family_teacher(phase.env.generator)
    teachers = ("teacher1", "teacher2") if phase.env.phase in {"live", "dev"} else (independent,)
    def stage_measurements():
        pending = []
        for record in main:
            row = questions[record["qid"]]
            for teacher in teachers:
                job = None
                if record.get("answer") and row["gold_labels"]:
                    payload = blind_concept_payload(row["question"], record["answer"],
                                                   row["gold_labels"], phase.env.graph.nodes)
                    job = _job(phase, row, "formal_concept_measurement", payload,
                               parent=answer_sha256(record["answer"]), teacher=teacher)
                pending.append((record, teacher, job))
        results = phase.call(list(dict.fromkeys(job for _, _, job in pending if job)))
        measurements = []
        for record, teacher, job in pending:
            labels = questions[record["qid"]]["gold_labels"]
            parsed = ({"concepts": []} if record.get("answer") and not labels else _parsed(results, job))
            measurements.append({**_identity(phase, record), "teacher": teacher, "primary": teacher == independent,
                "parsed": parsed, "status": "ok" if parsed is not None else "missing",
                "request_key": phase.request_keys.get(job),
                **(concept_metrics(labels, parsed) if parsed is not None else {"NCov": None, "ROE": None})})
        measurement_artifact = _save(phase, "measurements", measurements,
            summary=summarize_concept_metrics(measurements), agreement=classification_agreement(measurements))
        return measurements, measurement_artifact


    def stage_audits():
        # T90 is selected from the full trusted test split, independently of this run's subset.
        audit_qids = set(select_t90(phase.env.question_set.split_rows("test")))
        audits, audit_jobs = [], []
        for record in main:
            if record["qid"] not in audit_qids or record["arm"] not in AUDIT_ARMS:
                continue
            question = questions[record["qid"]]
            job = None
            if record.get("answer"):
                payload = {"question": question["question"], "candidate_answer": record["answer"],
                           "knowledge_chunks": record.get("evidence", [])}
                job = _job(phase, question, "formal_defect_audit", payload,
                           parent=answer_sha256(record["answer"]), teacher=independent)
            audit_jobs.append((record, job))
        results = phase.call(list(dict.fromkeys(job for _, job in audit_jobs if job)))
        for record, job in audit_jobs:
            parsed = _parsed(results, job)
            audits.append({**_identity(phase, record), "teacher": independent,
                "status": "ok" if parsed is not None else "missing", "parsed": parsed,
                "defects": sorted({item["type"] for item in (parsed or {}).get("defects", [])}),
                "request_key": phase.request_keys.get(job)})
        audit_artifact = _save(phase, "audits", audits, selected_qids=sorted(audit_qids))
        return audits, audit_artifact

    # The two stages read only the finished answers, so they run side by side (own transport locks).
    (measurements, measurement_artifact), (audits, audit_artifact) = _run_stages(
        [stage_measurements, stage_audits], ("measure", "audit"))
    summary = {"protocol": PROTOCOL, "origin": phase.env.origin,
        "status": "COMPLETE" if all(item["status"] == "COMPLETE" for item in
            (judgment_artifact, measurement_artifact, audit_artifact)) else "NOT_RUN",
        "judgments": len(judgments), "judgments_missing": sum(row["status"] != "ok" for row in judgments),
        "measurements": len(measurements), "audits": len(audits),
        "stop_reason": phase.stop_reason}
    with run_directory_guard(phase.run.root):
        path = (phase.run.root / "evaluation_summary.json" if summary["status"] == "COMPLETE" else
                phase.run.root / "incomplete" / f"evaluation-summary-{digest(summary)}.json")
        write_once(path, summary)
    return {"judgments": judgments, "measurements": measurements,
            "audits": audits, "summary": summary}


evaluate_phase = run_evaluations
