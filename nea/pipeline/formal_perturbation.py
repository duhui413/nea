"""Deterministic controlled perturbations and append-only detector evaluation."""
from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any, Mapping, Sequence

from nea.evaluation.formal_measurement import (DEFECT_TYPES, PROTOCOL, cross_family_teacher,
                                             select_perturbation_sources, selection_hash)
from nea.pipeline.artifacts import RunDirectoryLock, run_directory_guard
from nea.runtime.records import answer_sha256, digest, write_once

PERTURBATION_KINDS = (*DEFECT_TYPES, "null")


def ancestor_closure(graph: Any, targets: Sequence[str]) -> set[str]:
    """Traverse the already-loaded trusted graph; never revalidate its assets."""
    closure, pending = set(targets), list(targets)
    while pending:
        for edge in graph.incoming(pending.pop()):
            if edge["src"] not in closure:
                closure.add(edge["src"])
                pending.append(edge["src"])
    return closure


def unrelated_concept(graph: Any, targets: Sequence[str], qid: str) -> Mapping[str, Any] | None:
    excluded = ancestor_closure(graph, targets)
    chapters = {graph.nodes[target].get("chapter") for target in targets}
    choices = [node for key, node in graph.nodes.items() if key not in excluded
               and node.get("chapter") not in chapters and str(node.get("explain", "")).strip()]
    return min(choices, key=lambda node: (selection_hash(qid + "|unrelated|" + node["id"]), node["id"])) if choices else None


def paragraph_spans(answer: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S(?:[^\n]|\n(?!\s*\n))*", answer)]


def deterministic_insert(answer: str, insertion: str, key: str) -> tuple[str, dict]:
    if not answer.strip() or not insertion.strip():
        raise ValueError("insertion requires nonempty source and added content")
    boundaries = [0, *[end for _, end in paragraph_spans(answer)]]
    position = boundaries[int(selection_hash(key), 16) % len(boundaries)]
    before, after = ("\n\n" if position else ""), ("\n\n" if position < len(answer) else "")
    text = answer[:position] + before + insertion + after + answer[position:]
    start = position + len(before)
    return text, {"source_start": position, "source_end": position, "start": start,
                  "end": start + len(insertion), "before_span": "", "after_span": insertion,
                  "operation": "insert"}


def duplicate_paragraph(answer: str, key: str) -> str:
    spans = paragraph_spans(answer)
    if not spans:
        raise ValueError("cannot choose a paragraph from an empty answer")
    start, end = spans[int(selection_hash(key + "|duplicate"), 16) % len(spans)]
    return answer[start:end]


def apply_model_edit(answer: str, kind: str, edit: Mapping[str, Any], *, key: str,
                     duplicate: str = "") -> tuple[str | None, dict]:
    """Apply a unique exact local edit and retain the removed text as evidence."""
    if kind not in PERTURBATION_KINDS or kind == "term_excess":
        raise ValueError("unsupported model perturbation")
    if not edit["applicable"]:
        if any(edit[field] for field in ("before_span", "after_span", "term")):
            raise ValueError("inapplicable perturbation must have empty edit fields")
        return None, {"operation": "inapplicable", "reason": edit["reason"]}
    before, after = edit["before_span"], edit["after_span"]
    if not before.strip() or answer.count(before) != 1:
        raise ValueError("perturbation before_span must occur exactly once in source")
    if kind == "step_excess":
        if before != duplicate or not after.strip():
            raise ValueError("repetition must paraphrase the selected original paragraph")
        text, injection = deterministic_insert(answer, after, key + "|repeat")
        injection.update(copied_source_start=answer.index(before), copied_source_end=answer.index(before) + len(before),
                         copied_source_text=before, reason=edit["reason"])
        return text, injection
    if before == after:
        raise ValueError("perturbation edit must change source text")
    if kind in {"term_gap", "step_gap", "logical_jump"} and len(after) >= len(before):
        raise ValueError("deletion perturbation must remove source content")
    position = answer.index(before)
    text = answer[:position] + after + answer[position + len(before):]
    if not text.strip():
        raise ValueError("perturbation cannot delete the entire answer")
    if kind == "term_gap" and (not edit["term"].strip() or edit["term"] not in answer or edit["term"] not in text):
        raise ValueError("term-gap deletion must preserve the original term name")
    return text, {"source_start": position, "source_end": position + len(before),
                  "start": position, "end": position + len(after), "before_span": before,
                  "after_span": after, "operation": "replace", "term": edit["term"], "reason": edit["reason"]}


def location_hit(answer: str, span: str, injection: Mapping[str, Any]) -> bool:
    """Overlap or one neighboring sentence; deleted text has a zero-width anchor."""
    if not span or span not in answer:
        return False
    units = [(match.start(), match.end()) for match in re.finditer(r"[^。！？!?\n]+[。！？!?\n]*", answer)]
    if not units:
        return False
    start, end = injection["start"], injection["end"]
    affected = [index for index, (a, b) in enumerate(units)
                if (a <= start < b if start == end else a < end and b > start)]
    if not affected:
        affected = [min(range(len(units)), key=lambda index: abs(units[index][0] - start))]
    for match in re.finditer(re.escape(span), answer):
        located = [index for index, (a, b) in enumerate(units) if a < match.end() and b > match.start()]
        if any(abs(left - right) <= 1 for left in located for right in affected):
            return True
    return False


def detection_metrics(answer: str, expected_type: str | None, injection: Mapping[str, Any],
                       diagnosis: Mapping[str, Any] | None) -> dict:
    if diagnosis is None or diagnosis.get("status") == "uncertain":
        return {"type_detected": None, "location_hit": None, "false_positive": None,
                "measurement_status": "missing" if diagnosis is None else "uncertain"}
    defects = diagnosis.get("defects", diagnosis.get("actions", []))
    detected = [row for row in defects if row.get("type") == expected_type]
    return {"type_detected": bool(detected) if expected_type else None,
            "location_hit": any(location_hit(answer, row.get("span", ""), injection) for row in detected)
                if expected_type else None,
            "false_positive": bool(defects) if expected_type is None else None,
            "measurement_status": "ok"}


def summarize_perturbations(rows: Sequence[Mapping[str, Any]]) -> dict:
    result = {}
    for kind in PERTURBATION_KINDS:
        selected = [row for row in rows if row["kind"] == kind]
        injected_count = sum(row.get("answer") is not None for row in selected)
        lanes = {}
        for lane in ("same_family", "cross_family"):
            metrics = [item for row in selected for item in row.get("diagnostics", []) if item["lane"] == lane]
            available = [item for item in metrics if item.get("measurement_status") == "ok"]
            lanes[lane] = {"evaluated": len(available), "missing_or_uncertain": injected_count - len(available)}
            for field, label in (("type_detected", "type_detection_rate"),
                                 ("location_hit", "location_hit_rate"), ("false_positive", "false_positive_rate")):
                values = [row[field] for row in available if row.get(field) is not None]
                lanes[lane][label] = sum(values) / len(values) if values else None
        result[kind] = {"selected": len(selected), "injected": injected_count,
                        "inapplicable": sum(row.get("status") == "inapplicable" for row in selected), "lanes": lanes}
    return result


def _job(phase, source, role, payload, teacher, *, extra=None):
    question = next(row for row in phase.rows if row["qid"] == source["qid"])
    job = phase.job(question, source["base"], "PERTURB", role, payload,
                    parent=answer_sha256(source["answer"]), teacher=teacher)
    args = dict(job.schema_args)
    args.update(locatable_answer=payload.get("candidate_answer", payload.get("answer", "")),
                locatable_question=payload["question"],
                allowed_evidence_ids=tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", [])))
    args.update(extra or {})
    return replace(job, schema_args=tuple(sorted(args.items())))


def run_perturbations(phases: Sequence[Any], *, limit: int = 200) -> dict:
    """Select once across generators, then write each phase's own E4 artifact."""
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("formal perturbation source limit must be within 1..200")
    if not phases or len({phase.env.generator for phase in phases}) != len(phases):
        raise ValueError("perturbation requires distinct generator phases")
    if any(phase.env.phase != "test" for phase in phases):
        raise ValueError("controlled perturbation sources come from the formal test split")
    missing_sources = [phase.env.generator for phase in phases if not (phase.run.root / "answers.json").is_file()]
    if missing_sources:
        # Selecting before all source phases finish would change T200 on resume.
        artifact = {"protocol": PROTOCOL, "status": "NOT_RUN", "rows": [], "sources": [],
                    "missing_source_generators": missing_sources,
                    "reason": "source_generation_incomplete", "summary": summarize_perturbations([])}
        for phase in phases:
            with run_directory_guard(phase.run.root):
                write_once(phase.run.root / "incomplete" / f"perturbations-{digest(artifact)}.json",
                           {**artifact, "origin": phase.env.origin})
        return artifact
    phase_by_generator = {phase.env.generator: phase for phase in phases}
    sources = []
    for phase in phases:
        rows = json.loads((phase.run.root / "answers.json").read_text(encoding="utf8"))["rows"]
        parents = {(row["qid"], row["base"], row["arm"]): row for row in rows}
        for row in rows:
            if row["arm"] not in {"F", "FB"} or (row.get("diagnosis") or {}).get("status") != "no_issue":
                continue
            parent_arm = "L" if row["arm"] == "F" else "B"
            parent = parents.get((row["qid"], row["base"], parent_arm), {})
            if parent.get("answer"):
                task_record = parents.get((row["qid"], row["base"], "P"), parent)
                sources.append({**parent, "generator": phase.env.generator, "split": "test",
                    "source_arm": parent_arm, "diagnosis": row["diagnosis"],
                    "obligations": (task_record.get("task") or {}).get("obligations", [])})
    selected = select_perturbation_sources(sources, limit)
    injected = []
    pending = {key: [] for key in phase_by_generator}
    for source in selected:
        phase = phase_by_generator[source["generator"]]
        question = next(row for row in phase.rows if row["qid"] == source["qid"])
        source_id = digest({key: source[key] for key in ("qid", "generator", "base", "answer")})
        for kind in PERTURBATION_KINDS:
            record = {"qid": source["qid"], "generator": source["generator"], "base": source["base"],
                "source_arm": source["source_arm"], "source_sha256": answer_sha256(source["answer"]),
                "source_id": source_id, "kind": kind, "expected_type": None if kind == "null" else kind,
                "answer": None, "injection": None, "status": "missing", "diagnostics": []}
            if kind == "term_excess":
                concept = unrelated_concept(phase.env.graph, question["gold_target_ids"], source["qid"])
                if concept:
                    record["answer"], record["injection"] = deterministic_insert(source["answer"],
                        concept["name"] + "：" + concept["explain"], source_id + "|unrelated")
                    record["injection"]["concept_id"] = concept["id"]
                    record["status"] = "injected"
                else:
                    record.update(status="inapplicable", injection={"reason": "no_cross_chapter_nonancestor"})
            else:
                duplicate = duplicate_paragraph(source["answer"], source_id) if kind == "step_excess" else ""
                payload = {"question": question["question"], "candidate_answer": source["answer"],
                           "kind": kind, "duplicate_paragraph": duplicate}
                job = _job(phase, source, "formal_perturbation_rewrite", payload, "student1",
                    extra={"perturbation_kind": kind, "duplicate_paragraph": duplicate})
                pending[source["generator"]].append((record, source, duplicate, job))
            injected.append((record, source))
    def run_generator(generator):
        phase = phase_by_generator[generator]
        work = pending[generator]
        results = phase.call(list(dict.fromkeys(item[-1] for item in work)))
        for record, source, duplicate, job in work:
            result = results.get(job)
            record["perturbation_request_key"] = phase.request_keys.get(job)
            if result:
                record["answer"], record["injection"] = apply_model_edit(source["answer"], record["kind"],
                    result["parsed"], key=record["source_id"], duplicate=duplicate)
                record["status"] = "injected" if record["answer"] else "inapplicable"
        diagnostics = []
        for record, source in injected:
            if record["answer"] is None or source["generator"] != generator:
                continue
            question = next(row for row in phase.rows if row["qid"] == source["qid"])
            for lane, teacher, role in (("same_family", "student1", "formal_explanation_diagnosis"),
                                       ("cross_family", cross_family_teacher(source["generator"]), "formal_defect_audit")):
                payload = {"question": question["question"], "knowledge_chunks": source.get("evidence", [])}
                payload["answer" if lane == "same_family" else "candidate_answer"] = record["answer"]
                if lane == "same_family":
                    from nea.method.formal_operators import BUDGET, FormalOperatorContext, diagnose_payload
                    context = FormalOperatorContext(question=question["question"], parent_answer=record["answer"],
                        evidence=tuple(source.get("evidence", [])), obligations=tuple(source["obligations"]),
                        budget=dict(BUDGET), versions={"method": PROTOCOL}, base=source["base"])
                    payload = diagnose_payload(context)
                job = _job(phase, source, role, payload, teacher)
                diagnostics.append((record, lane, teacher, job))
        results = phase.call(list(dict.fromkeys(item[-1] for item in diagnostics)))
        for record, lane, teacher, job in diagnostics:
            parsed = (results.get(job) or {}).get("parsed")
            record["diagnostics"].append({"lane": lane, "teacher": teacher, "parsed": parsed,
                "status": "ok" if parsed else "missing", "request_key": phase.request_keys.get(job),
                **detection_metrics(record["answer"], record["expected_type"], record["injection"], parsed)})

    # Each generator's rewrite -> diagnose chain is independent of the other's, so both run concurrently.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(phase_by_generator)), thread_name_prefix="formal8-perturb") as chains:
        for future in [chains.submit(run_generator, generator) for generator in phase_by_generator]:
            future.result()
    rows = []
    for record, source in injected:
        if record["answer"]:
            record["answer_sha256"] = answer_sha256(record["answer"])
            record["status"] = "ok" if len(record["diagnostics"]) == 2 and all(
                item["status"] == "ok" for item in record["diagnostics"]) else "missing"
        rows.append(record)
    complete = (all(row["status"] in {"ok", "inapplicable"} for row in rows)
                and not any(phase.stop_reason for phase in phases))
    source_manifest = [{key: row[key] for key in ("qid", "generator", "base", "source_arm")}
                       | {"answer_sha256": answer_sha256(row["answer"])} for row in selected]
    for generator, phase in phase_by_generator.items():
        local_rows = [row for row in rows if row["generator"] == generator]
        artifact = {"protocol": PROTOCOL, "origin": phase.env.origin,
            "status": "COMPLETE" if complete else "NOT_RUN",
            "source_selection": source_manifest, "source_limit": limit, "rows": local_rows,
            "summary": summarize_perturbations(local_rows)}
        with run_directory_guard(phase.run.root):
            target = (phase.run.root / "perturbations.json" if artifact["status"] == "COMPLETE" else
                      phase.run.root / "incomplete" / f"perturbations-{digest(artifact)}.json")
            write_once(target, artifact)
    return {"status": "COMPLETE" if complete else "NOT_RUN", "rows": rows,
            "summary": summarize_perturbations(rows), "sources": source_manifest}
