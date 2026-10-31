"""need-aware phase engine: dev/test generation waves, internal signals, revision,
blind evaluation payloads, freeze.  Every model call goes through
``transport.execute`` (successful-result reuse and reply validation).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from nea.pipeline.artifacts import RunDirectoryLock
from nea.pipeline.sampling import check_mode, mode_fields
from nea.runtime.records import AppendOnlyRun, answer_sha256, digest, write_once

from nea.method import gaps as gapmod, revision as rev
from nea.question_set.contract import QuestionSet
from nea.method.graph import Graph
from nea.method.grounding import select_targets
from nea.runtime.identity import generator_directory, manifest_identity, phase_directory, model_config_identity
from nea.providers.config import local_runtime_identity
from nea.pipeline.planner import (
    NO_CALIBRATION,
    JobFactory,
    all_template_hashes,
    detector_for,
    grounded_answer_payload,
    p_answer_payload,
    prepare_question,
    static_payloads,
)
from nea.pipeline.registry import (
    BASES,
    DEFAULT_DETECTOR,
    FEEDBACK_MASK,
    GENERATORS,
    JUDGE_ROLES,
    PARENT_ARM,
    PROTOCOL,
    SIGNALS,
    arms_for_split,
)
from nea.method.signals import SignalComponents, compute_signals, fit_thresholds, triggers
from nea.runtime.transport import MAX_IN_FLIGHT, RouterLike, ExperimentJob, execute


GROUNDED_ARMS = ("L", "KP", "D1", "D2")
RELEASE_RULES = {"f_minus_b_min": 0.0, "f_minus_p_min": 0.0, "f_b_chars_ratio_max": 1.3,
                 "grounding_any_hit_min": 0.80, "detector_f1_min": 0.50}
DETECTOR_CALLS = {"D1": 1, "D2": 0, "D3": 1}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_missing_judgments(answers: Sequence[Mapping[str, Any]],
                            judgments: Sequence[Mapping[str, Any]]) -> int:
    """Include judge slots that could not be scheduled for missing answers."""
    return len(answers) * len(JUDGE_ROLES) - sum(row["scores"] is not None for row in judgments)


@dataclass
class Environment:
    run_root: Path
    run_id: str
    generator: str
    question_set: QuestionSet
    graph: Graph
    index: Any
    terminology: Callable[[str], float]
    structure: Callable[[str], float]
    components: SignalComponents
    rag_pools: Mapping[str, Sequence[Mapping[str, Any]]]
    prompts_dir: Path
    router: RouterLike
    smoke: bool = False
    sample_size: int | None = None
    prepared_questions: Mapping[str, Any] | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"calls": 0, "reused": 0,
                                                            "structure_failures": 0, "exhausted": 0})


def append_progress(env: Any, split: str, event: str, **details: Any) -> dict[str, Any]:
    """Immutable progress events; invocation counts never alter phase summaries."""
    directory = generator_directory(env.run_root / "progress", env.generator) / split
    with RunDirectoryLock(directory):
        existing = sorted(directory.glob("*.json"))
        if any(not path.stem.isdigit() for path in existing):
            raise ValueError("Unknown progress record")
        sequence = max((int(path.stem) for path in existing), default=0) + 1
        record = {"version": 1, "sequence": sequence, "generator": env.generator, "split": split,
                  "event": event, "counts": dict(env.counts),
                  "completed": event == "phase_completed", "failed": event in {"phase_failed", "wave_failed"},
                  **details}
        write_once(directory / f"{sequence:06d}.json", record)
    print(json.dumps({"progress": record}, ensure_ascii=False, sort_keys=True), flush=True)
    return record


class PhaseIncompleteError(RuntimeError):
    """Safe completeness diagnostics; exhausted requests remain terminal."""

    code = "phase_incomplete"

    def __init__(self, missing: Sequence[Sequence[str]], judgments_missing: int):
        self.missing = [list(row) for row in missing]
        self.judgments_missing = judgments_missing
        super().__init__(f"Phase incomplete: {len(self.missing)} missing answers, "
                         f"{judgments_missing} missing judgments; inspect failure records before continuing")


class PhaseGroupError(RuntimeError):
    def __init__(self, results: dict[str, Any], failures: dict[str, Any]):
        super().__init__("Phase incomplete for: " + ", ".join(sorted(failures)))
        self.results, self.failures = results, failures


def run_phases(phases: Sequence[Any]) -> dict[str, Any]:
    """Let both generator groups settle, preserving each group's own failure."""
    if not 1 <= len(phases) <= 2 or len({phase.env.generator for phase in phases}) != len(phases):
        raise ValueError("Require one or two distinct generator phases")

    def run_one(phase: Any) -> Any:
        try:
            append_progress(phase.env, phase.split, "phase_started")
            result = phase.run_phase()
            append_progress(phase.env, phase.split, "phase_completed")
        except Exception as exc:
            try:
                details = ({"missing": exc.missing, "judgments_missing": exc.judgments_missing}
                           if isinstance(exc, PhaseIncompleteError) else {})
                append_progress(phase.env, phase.split, "phase_failed",
                                code=getattr(exc, "code", type(exc).__name__), **details)
            except Exception:
                # A progress-directory failure must not hide the original
                # startup or transport failure; the collector also prints it.
                pass
            raise
        return result

    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="nea-phase") as executor:
        futures = {executor.submit(run_one, phase): phase.env.generator for phase in phases}
        for future in as_completed(futures):
            generator = futures[future]
            try:
                results[generator] = future.result()
            except Exception as exc:
                failures[generator] = {"type": type(exc).__name__, "code": getattr(exc, "code", "phase_error")}
                if isinstance(exc, PhaseIncompleteError):
                    failures[generator].update(missing=exc.missing, judgments_missing=exc.judgments_missing)
                if isinstance(exc, OSError):
                    for name in ("errno", "winerror", "filename"):
                        value = getattr(exc, name, None)
                        if value is not None:
                            failures[generator][name] = str(value) if name == "filename" else value
                if getattr(exc, "retry_at", None) is not None:
                    failures[generator]["retry_at_unix_utc"] = exc.retry_at
                print(json.dumps({"failed_group": {"generator": generator, **failures[generator]}},
                                 ensure_ascii=False, sort_keys=True), flush=True)
    if failures:
        raise PhaseGroupError(dict(sorted(results.items())), dict(sorted(failures.items())))
    return dict(sorted(results.items()))


class Phase:
    def __init__(self, env: Environment, split: str, freeze: Mapping[str, Any] | None):
        raise RuntimeError("0022 phase orchestration pending: new prompts require diagnose/revise/validate integration")
        self.env, self.split = env, split
        mode_fields(smoke=env.smoke, sample_size=env.sample_size)
        if env.smoke and env.question_set.counts != {"dev": 3, "test": 3}:
            raise ValueError("Smoke requires exactly dev3/test3")
        if env.sample_size is not None and env.question_set.counts != {"dev": env.sample_size, "test": env.sample_size}:
            raise ValueError("Exploratory question counts differ from sample_size")
        if split == "test":
            if not freeze or (not (env.smoke or env.sample_size is not None)
                              and freeze.get("freeze", {}).get("release_gate", {}).get("release_test") is not True):
                raise ValueError("test requires a freeze whose release gate passed")
            body = freeze["freeze"]
            check_mode(body, smoke=env.smoke, sample_size=env.sample_size)
            if body.get("local_runtime") != local_runtime_identity():
                raise ValueError("Calibration local runtime differs; run dev/freeze with CUDA fp16")
            if body["question_set_sha256"] != env.question_set.sha256:
                raise ValueError("Calibration uses a different question set")
            self.detector = body["detector"]["selected"]
            self.calibration = freeze["freeze_hash"]
            self.thresholds = body["thresholds"][env.generator]
            self.validated = frozenset(body["validated_signals"])
        else:
            self.detector, self.calibration = DEFAULT_DETECTOR, NO_CALIBRATION
            self.thresholds, self.validated = {}, frozenset(SIGNALS)
        identity = manifest_identity(
            generator=env.generator, run_id=env.run_id, split=split,
            template_hashes=all_template_hashes(env.prompts_dir),
            static_payload_hashes={name: digest(value) for name, value in static_payloads(env.prompts_dir).items()},
            calibration_hash=self.calibration, question_set_sha256=env.question_set.sha256,
            detector=self.detector)
        identity.update(mode_fields(smoke=env.smoke, sample_size=env.sample_size))
        self.run = AppendOnlyRun(phase_directory(env.run_root, env.generator, split), identity)
        self.factory = JobFactory(env.prompts_dir, split)
        self.arms = arms_for_split(split)
        self.rows = env.question_set.split_rows(split)
        self.parent_index: dict[tuple[str, str, str], str] = {}
        self._wave_number = 0
        if not self.rows:
            raise ValueError(f"question set has no {split} rows")

    def call(self, jobs: list[ExperimentJob]) -> dict[ExperimentJob, dict[str, Any] | None]:
        if not jobs:
            return {}
        self._wave_number += 1
        details = {"wave": self._wave_number, "stages": sorted({job.stage for job in jobs}), "jobs": len(jobs)}
        append_progress(self.env, self.split, "wave_started", **details)
        try:
            result = execute(self.run, self.env.generator, jobs, router=self.env.router,
                             question_splits=self.env.question_set.qid_splits(), phase_split=self.split,
                             parent_index=self.parent_index,
                             max_in_flight=getattr(self.env.router, "max_in_flight", MAX_IN_FLIGHT))
        except Exception as exc:
            for name, value in getattr(exc, "counts", {}).items():
                self.env.counts[name] = self.env.counts.get(name, 0) + value
            append_progress(self.env, self.split, "wave_failed", code=getattr(exc, "code", type(exc).__name__), **details)
            raise
        for name, value in result["counts"].items():
            self.env.counts[name] = self.env.counts.get(name, 0) + value
        append_progress(self.env, self.split, "wave_completed", **details)
        return {job: result["results"].get(result["key_by_job"][job]) for job in jobs}

    # ------------------------------------------------------------------
    def run_phase(self) -> dict[str, Any]:
        env, graph = self.env, self.env.graph
        self.run.initialize()
        preps = env.prepared_questions
        if preps is None:
            preps = {row["qid"]: prepare_question(row, graph, env.index, env.terminology, env.structure,
                                                  env.rag_pools.get(row["qid"]))
                     for row in self.rows}
        if set(preps) != {row["qid"] for row in self.rows}:
            raise ValueError("Prepared questions differ from phase scope")
        cells = [(qid, base) for qid in sorted(preps) for base in BASES]
        grounded_arms = [arm for arm in GROUNDED_ARMS if arm in self.arms]
        records: dict[tuple[str, str, str], dict[str, Any]] = {}

        # Wave 1: B plan, P plan, grounding.
        wave: dict[tuple[str, str, str], ExperimentJob] = {}
        for qid, base in cells:
            prep = preps[qid]
            wave[(qid, base, "B")] = self.factory.b_plan(prep, base)
            wave[(qid, base, "P")] = self.factory.p_plan(prep, base)
            for arm in grounded_arms:
                wave[(qid, base, arm)] = self.factory.ground(prep, base, arm, graph)
        out1 = self.call(list(wave.values()))

        # Wave 2: B/P answers, grounding decisions, gap-detection calls.
        answers: dict[tuple[str, str, str], ExperimentJob] = {}
        gap_jobs: dict[tuple[str, str, str], ExperimentJob] = {}
        context: dict[tuple[str, str, str], dict[str, Any]] = {}
        for qid, base in cells:
            prep = preps[qid]
            plan = out1[wave[(qid, base, "B")]]
            if plan is None:
                records[(qid, base, "B")] = {"status": "missing", "reason": "b_plan_structure_exhausted"}
            else:
                answers[(qid, base, "B")] = self.factory.b_answer(prep, base, plan["parsed"], plan["text"])
            plan = out1[wave[(qid, base, "P")]]
            if plan is None:
                records[(qid, base, "P")] = {"status": "missing", "reason": "plan_structure_exhausted"}
            else:
                payload = p_answer_payload(prep, base, plan["parsed"])
                context[(qid, base, "P")] = {"route": payload["route"],
                                             "knowledge_chunks": payload["knowledge_chunks"]}
                answers[(qid, base, "P")] = self.factory.answer(prep, base, "P", payload,
                                                                parent=answer_sha256(plan["text"]))
            for arm in grounded_arms:
                result = out1[wave[(qid, base, arm)]]
                grounding = select_targets(result["parsed"] if result else None, prep.candidates,
                                           failure=None if result else "structure_invalid_after_retries")
                c2 = gapmod.local_candidates(graph, grounding["grounded_target_ids"])
                rep = gapmod.query_representation(terminology=prep.terminology, structure=prep.structure,
                                                  question_concept_ids=prep.question_ids,
                                                  c2_ids=c2["candidate_ids"])
                detector = detector_for(arm, self.detector)
                item = {"grounding": grounding, "c2": c2, "rep": rep, "detector": detector}
                if detector == "KP":
                    item["detection"] = gapmod.detect_kp(c2)
                elif detector == "D2":
                    item["detection"] = gapmod.detect_d2(c2, prep.question_ids)
                elif detector == "D3" and not c2["candidate_ids"]:
                    item["detection"] = gapmod.detect_d3({"labels": {}})
                else:
                    gap_jobs[(qid, base, arm)] = self.factory.gaps(prep, base, arm, detector, graph,
                                                                   grounding, c2)
                context[(qid, base, arm)] = item
        out2 = self.call(list(answers.values()) + list(gap_jobs.values()))

        # Wave 3: grounded answers (shared need-aware answer template).
        initial: dict[tuple[str, str, str], str] = {}
        for key, job in answers.items():
            result = out2[job]
            if result is None:
                records[key] = {"status": "missing", "reason": "answer_structure_exhausted"}
            else:
                initial[key] = result["parsed"]
                if key[2] == "B":
                    context[key] = {"knowledge_chunks": [
                        {"chunk_id": cid} for cid in job.arg("allowed_citations", ())]}
        grounded_answers: dict[tuple[str, str, str], ExperimentJob] = {}
        for qid, base in cells:
            prep = preps[qid]
            for arm in grounded_arms:
                key = (qid, base, arm)
                item = context[key]
                if "detection" not in item:
                    result = out2[gap_jobs[key]]
                    if result is None:
                        records[key] = {"status": "missing", "reason": "gaps_structure_exhausted"}
                        continue
                    item["detection"] = (gapmod.detect_d1(result["parsed"], graph,
                                                          item["grounding"]["grounded_target_ids"])
                                         if item["detector"] == "D1" else gapmod.detect_d3(result["parsed"]))
                item["order"] = gapmod.order_gaps(item["detection"]["gap_ids"], graph)
                payload = grounded_answer_payload(prep, base, graph, item["grounding"], item["detection"],
                                                  item["order"], item["rep"])
                item["knowledge_chunks"] = payload["knowledge_chunks"]
                grounded_answers[key] = self.factory.answer(
                    prep, base, arm, payload,
                    parent=digest({"grounding": item["grounding"]["grounded_target_ids"],
                                   "detection": item["detection"], "order": item["order"]["order"]}))
        out3 = self.call(list(grounded_answers.values()))
        for key, job in grounded_answers.items():
            result = out3[job]
            if result is None:
                records[key] = {"status": "missing", "reason": "answer_structure_exhausted"}
            else:
                initial[key] = result["parsed"]

        # Internal signals (method only; never paper measurement fields).
        levels = {node: data["level"] for node, data in graph.nodes.items()}
        signals: dict[tuple[str, str, str], dict[str, Any]] = {}
        for key, text in initial.items():
            if key[2] not in grounded_arms:
                continue
            item, prep = context[key], preps[key[0]]
            detection = item["detection"]
            targets = item["grounding"]["grounded_target_ids"]
            slot_of = {**{node: "pre" for node in detection["gap_ids"]}, **{node: "target" for node in targets}}
            signals[key] = compute_signals(
                text, gap_ids=detection["gap_ids"], given_ids=detection["given_ids"],
                assumed_ids=detection["assumed_ids"], target_ids=targets,
                question_ids=list(prep.question_ids), slot_of=slot_of, levels=levels,
                components=env.components)
        if self.split == "dev":
            self.thresholds = {base: fit_thresholds(
                signals[(qid, b, "L")] for (qid, b, arm) in signals if arm == "L" and b == base)
                for base in BASES}
        calibration_by_base = {base: digest({"source": self.calibration, "thresholds": self.thresholds[base],
                                             "validated": sorted(self.validated)}) for base in BASES}

        # Parent index for F/M-* (L initial) and SR (B initial): checked per job.
        self.parent_index = {key: answer_sha256(text) for key, text in initial.items()}

        # Wave 4: diagnose (F, M-*) and critique (SR).
        revision_arms = [arm for arm in self.arms if arm in PARENT_ARM]
        pending: dict[tuple[str, str, str], dict[str, Any]] = {}
        finals: dict[tuple[str, str, str], dict[str, Any]] = {}
        first: dict[tuple[str, str, str], ExperimentJob] = {}
        for qid, base in cells:
            prep = preps[qid]
            for arm in revision_arms:
                key, parent_key = (qid, base, arm), (qid, base, PARENT_ARM[arm])
                if parent_key not in initial:
                    records[key] = {"status": "missing", "reason": "parent_answer_missing"}
                    continue
                parent_text = initial[parent_key]
                allowed = [row["chunk_id"] for row in context[parent_key]["knowledge_chunks"]]
                state = {"initial": parent_text, "parent_arm": PARENT_ARM[arm],
                         "limits": rev.constraints(parent_text, allowed), "records": [],
                         "calibration": NO_CALIBRATION if arm == "SR" else calibration_by_base[base]}
                if arm == "SR":
                    first[key] = self.factory.critique(prep, base, parent_text, answer_sha256(parent_text))
                else:
                    sig = signals[parent_key]
                    mask = FEEDBACK_MASK[arm] & self.validated
                    fired = triggers(sig, self.thresholds[base], mask)
                    state["triggers"] = fired
                    if not any(fired.values()):
                        finals[key] = rev.finalize(parent_text, [], reason_if_none="no_trigger")
                        continue
                    units = sig["units"]
                    payload = {"question": prep.question, "answer": parent_text,
                               "triggered_signals": sorted(name for name, value in fired.items() if value),
                               "uncovered_gaps": [{"concept_id": node, "name": graph.name(node)}
                                                  for node in sig["uncovered_gaps"]],
                               "detour_units": [{"unit_index": i, "text": units[i]} for i in sig["detour_units"]],
                               "weak_transitions": [{"unit_index": i, "text": units[i]}
                                                    for i in sig["weak_transitions"]]}
                    first[key] = self.factory.diagnose(prep, base, arm, payload, answer_sha256(parent_text),
                                                       state["calibration"])
                pending[key] = state
        out4 = self.call(list(first.values()))
        for key, job in first.items():
            state, result = pending[key], out4[job]
            if result is None:
                finals[key] = rev.finalize(state["initial"], [], reason_if_none="diagnosis_structure_exhausted")
                del pending[key]
                continue
            if key[2] == "SR":
                state["issues"] = [issue for issue in result["parsed"]["issues"] if issue.strip()]
                if not state["issues"]:
                    finals[key] = rev.finalize(state["initial"], [], reason_if_none="no_issue")
                    del pending[key]
            else:
                sig = signals[(key[0], key[1], "L")]
                state["actions"] = rev.select_actions(
                    result["parsed"], uncovered_gaps=sig["uncovered_gaps"], detour_units=sig["detour_units"],
                    weak_transitions=sig["weak_transitions"], triggered=state["triggers"])
                if not state["actions"]:
                    finals[key] = rev.finalize(state["initial"], [], reason_if_none="no_executable_action")
                    del pending[key]

        # Waves 5-7: revise calls (<=3 real calls per cell: one call per revise
        # key, structure failures included; length retry <=1), first valid wins.
        while pending:
            jobs: dict[tuple[str, str, str], ExperimentJob] = {}
            for key, state in sorted(pending.items()):
                prep = preps[key[0]]
                previous = state["records"][-1]["status"] if state["records"] else None
                common = {"question": prep.question, "answer": state["initial"], **state["limits"],
                          "previous_failure": previous}
                payload = ({**common, "issues": state["issues"]} if key[2] == "SR"
                           else {**common, "actions": state["actions"]})
                jobs[key] = self.factory.revise(prep, key[1], key[2], payload, answer_sha256(state["initial"]),
                                                state["calibration"], len(state["records"]) + 1,
                                                state["limits"]["allowed_citations"])
            out = self.call(list(jobs.values()))
            for key, job in jobs.items():
                state, result = pending[key], out[job]
                if result is None:
                    # One real call per revise key: a structurally bad revise
                    # response is one of this cell's <= 3 revise calls.
                    state["records"].append({"call_index": job.call_index, "status": "invalid_structure",
                                             "answer": "", "answer_sha256": ""})
                else:
                    text = result["parsed"]
                    okay, reason = rev.validate_revision(state["initial"], text, state["limits"])
                    state["records"].append({"call_index": job.call_index, "status": reason,
                                             "answer": text, "answer_sha256": answer_sha256(text)})
                verdict = rev.decision(state["records"])
                if verdict != "continue":
                    finals[key] = rev.finalize(state["initial"], state["records"],
                                               actions=state.get("actions", ()))
                    del pending[key]

        # Final answers and blind evaluation payloads.
        final_text: dict[tuple[str, str, str], str] = {}
        for key, text in initial.items():
            final_text[key] = text
        for key, result in finals.items():
            final_text[key] = result["answer"]
        judge_jobs: dict[tuple[str, str, str, str], ExperimentJob] = {}
        for key in sorted(final_text):
            if key[2] not in self.arms:
                continue
            for teacher in JUDGE_ROLES:
                judge_jobs[(*key, teacher)] = self.factory.judge(preps[key[0]], final_text[key], teacher)
        out5 = self.call(list(judge_jobs.values()))
        judgments = []
        for (qid, base, arm, teacher), job in sorted(judge_jobs.items()):
            result = out5[job]
            judgments.append({"qid": qid, "generator": env.generator, "base": base, "arm": arm,
                              "teacher": teacher, "answer_sha256": answer_sha256(final_text[(qid, base, arm)]),
                              "scores": None if result is None else result["parsed"]["scores"]})

        rows = []
        for qid, base in cells:
            for arm in self.arms:
                key = (qid, base, arm)
                row: dict[str, Any] = {"qid": qid, "split": self.split, "generator": env.generator,
                                       "base": base, "arm": arm}
                if key not in final_text:
                    row.update(records.get(key, {"status": "missing", "reason": "not_generated"}))
                    rows.append(row)
                    continue
                text = final_text[key]
                row.update({"status": "ok", "answer": text, "answer_sha256": answer_sha256(text),
                            "chars": len(text)})
                if arm in grounded_arms:
                    item = context[key]
                    row["internal"] = {"grounding": {k: item["grounding"][k] for k in
                                                     ("grounded_target_ids", "grounding_source", "fallback_reason")},
                                       "candidate_ids": [c["concept_id"] for c in item["grounding"]["candidates"]],
                                       "c2": item["c2"], "query_representation": item["rep"],
                                       "detection": item["detection"], "order": item["order"],
                                       "signals": {name: signals[key][name] for name in SIGNALS}}
                if arm == "P":
                    row["internal"] = {"route": context[key]["route"]}
                if key in finals:
                    result = finals[key]
                    row["revision"] = {"parent_arm": PARENT_ARM[arm], "parent_sha256": result["parent_sha256"],
                                       "revised": result["revised"], "fallback_reason": result["fallback_reason"],
                                       "calls": result["calls"], "actions": result["actions"],
                                       "candidate_status": [r["status"] for r in result["candidate_records"]],
                                       "triggers": pending_triggers(arm, key, signals, self)}
                rows.append(row)
        summary = {"protocol": PROTOCOL, "split": self.split, "generator": env.generator,
                   "question_set_sha256": env.question_set.sha256, "detector": self.detector,
                   "calibration_hash": self.calibration, "thresholds": self.thresholds,
                   "validated_signals": sorted(self.validated),
                   "feedback_calibration_by_base": calibration_by_base,
                   "answers": len([row for row in rows if row["status"] == "ok"]),
                   "missing": [[row["qid"], row["base"], row["arm"], row.get("reason")]
                               for row in rows if row["status"] != "ok"],
                   "judgments_missing": count_missing_judgments(rows, judgments),
                   "origin": "REAL"}
        summary.update(mode_fields(smoke=env.smoke, sample_size=env.sample_size))
        if summary["missing"] or summary["judgments_missing"]:
            raise PhaseIncompleteError(summary["missing"], summary["judgments_missing"])
        with RunDirectoryLock(self.run.root):
            write_once(self.run.root / "answers.json", {"rows": rows})
            write_once(self.run.root / "judgments.json", {"rows": judgments})
            write_once(self.run.root / "phase_summary.json", summary)
        return summary


def pending_triggers(arm: str, key: tuple[str, str, str], signals: Mapping[Any, Any], phase: Phase) -> Any:
    if arm == "SR":
        return None
    sig = signals.get((key[0], key[1], "L"))
    if sig is None:
        return None
    return triggers(sig, phase.thresholds[key[1]], FEEDBACK_MASK[arm] & phase.validated)


# ----------------------------------------------------------------------
def load_phase(run_root: Path, generator: str, split: str) -> dict[str, Any]:
    directory = phase_directory(run_root, generator, split)
    paths = {name: directory / f"{name}.json" for name in ("answers", "judgments", "phase_summary")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"{split} outputs for {generator} are incomplete")
    if read_json(paths["phase_summary"]).get("origin") != "REAL":
        raise ValueError("Only real phase outputs can be analyzed")
    manifest = read_json(directory / "run_manifest.json")
    return {"answers": read_json(paths["answers"])["rows"], "judgments": read_json(paths["judgments"])["rows"],
            "summary": read_json(paths["phase_summary"]), "manifest": manifest,
            "directory": str(directory)}


def overall_by_answer(judgments: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    """Mean of the two judges' overall_quality per (generator, base, arm, qid)."""
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for row in judgments:
        if row["scores"] is None:
            continue
        grouped.setdefault((row["generator"], row["base"], row["arm"], row["qid"]), []).append(
            row["scores"]["overall_quality"])
    return {key: fmean(values) for key, values in grouped.items() if len(values) == len(JUDGE_ROLES)}


def freeze(run_root: Path, question_set: QuestionSet, *, smoke: bool = False,
           sample_size: int | None = None) -> dict[str, Any]:
    mode_fields(smoke=smoke, sample_size=sample_size)
    phases = {generator: load_phase(run_root, generator, "dev") for generator in GENERATORS}
    for phase in phases.values():
        for source in (phase["manifest"], phase["summary"]):
            check_mode(source, smoke=smoke, sample_size=sample_size)
    if smoke and question_set.counts != {"dev": 3, "test": 3}:
        raise ValueError("Smoke requires exactly dev3/test3")
    if sample_size is not None and question_set.counts != {"dev": sample_size, "test": sample_size}:
        raise ValueError("Exploratory freeze counts differ from sample_size")
    if any(phase["manifest"].get("model_config_hash") != digest(model_config_identity(generator))
           for generator, phase in phases.items()):
        raise ValueError("dev model/runtime configuration differs; use a new CUDA fp16 run")
    if any(phase["summary"]["question_set_sha256"] != question_set.sha256 for phase in phases.values()):
        raise ValueError("dev outputs were produced from a different question set")
    gold = question_set.by_qid
    counts: dict[str, list[dict[str, int]]] = {name: [] for name in ("D1", "D2", "D3")}
    grounding_rows = []
    thresholds: dict[str, dict[str, Any]] = {}
    chars: dict[str, list[int]] = {"F": [], "B": []}
    for generator, phase in sorted(phases.items()):
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for row in phase["answers"]:
            by_arm.setdefault(row["arm"], []).append(row)
        for arm, name in (("D1", "D1"), ("D2", "D2"), ("L", "D3")):
            for row in by_arm.get(arm, []):
                if row["status"] != "ok":
                    continue
                gold_gaps = [k for k, v in gold[row["qid"]]["gold_labels"].items() if v == "gap"]
                counts[name].append(gapmod.set_f1(row["internal"]["detection"]["gap_ids"], gold_gaps))
        for row in by_arm.get("L", []):
            if row["status"] == "ok":
                chosen = row["internal"]["grounding"]["grounded_target_ids"]
                targets = set(gold[row["qid"]]["gold_target_ids"])
                grounding_rows.append({"top1": chosen[0] in targets, "any": bool(set(chosen) & targets)})
        thresholds[generator] = phase["summary"]["thresholds"]
        for arm in ("F", "B"):
            chars[arm] += [row["chars"] for row in by_arm.get(arm, []) if row["status"] == "ok"]
    f1 = {name: round(gapmod.micro_f1(rows), 6) for name, rows in counts.items()}
    selected = sorted(f1, key=lambda name: (-f1[name], DETECTOR_CALLS[name], name != "D3", name))[0]
    scores: dict[tuple[str, str, str, str], float] = {}
    for phase in phases.values():
        scores.update(overall_by_answer(phase["judgments"]))

    def contrast(left: str, right: str) -> float | None:
        diffs = [scores[(g, b, left, q)] - scores[(g, b, right, q)] for (g, b, a, q) in scores
                 if a == left and (g, b, right, q) in scores]
        return round(fmean(diffs), 6) if diffs else None

    any_hit = round(fmean(row["any"] for row in grounding_rows), 6) if grounding_rows else None
    top1 = round(fmean(row["top1"] for row in grounding_rows), 6) if grounding_rows else None
    ratio = round(fmean(chars["F"]) / fmean(chars["B"]), 6) if chars["F"] and chars["B"] else None
    f_b, f_p = contrast("F", "B"), contrast("F", "P")
    checks = {"f_minus_b": f_b is not None and f_b >= RELEASE_RULES["f_minus_b_min"],
              "f_minus_p": f_p is not None and f_p >= RELEASE_RULES["f_minus_p_min"],
              "f_b_chars_ratio": ratio is not None and ratio <= RELEASE_RULES["f_b_chars_ratio_max"],
              "grounding_any_hit": any_hit is not None and any_hit >= RELEASE_RULES["grounding_any_hit_min"],
              "detector_f1": f1[selected] >= RELEASE_RULES["detector_f1_min"]}
    validated = list(SIGNALS)
    body = {"protocol": PROTOCOL, "question_set_sha256": question_set.sha256,
            "local_runtime": local_runtime_identity(),
            "generators": list(GENERATORS), "origin": "REAL",
            "detector": {"selected": selected, "micro_f1": f1, "calls_per_question": DETECTOR_CALLS},
            "grounding": {"top1_accuracy": top1, "any_hit_accuracy": any_hit, "n": len(grounding_rows)},
            "thresholds": thresholds, "validated_signals": validated, "validity_origin": "DEV_FITTED_INTERNAL_SIGNALS",
            "release_gate": {"rules": RELEASE_RULES, "point_estimates": {
                "F-B": f_b, "F-P": f_p, "F/B_chars": ratio, "grounding_any_hit": any_hit,
                "detector_f1": f1[selected]}, "checks": checks, "release_test": all(checks.values())}}
    body.update(mode_fields(smoke=smoke, sample_size=sample_size))
    document = {"freeze": body, "freeze_hash": digest(body)}
    run_root.mkdir(parents=True, exist_ok=True)
    with RunDirectoryLock(run_root):
        write_once(run_root / "freeze.json", document)
    return document


def load_freeze(run_root: Path) -> dict[str, Any]:
    document = read_json(run_root / "freeze.json")
    if document["freeze"].get("local_runtime") != local_runtime_identity():
        raise ValueError("Calibration local runtime differs; run dev/freeze with CUDA fp16")
    return document


def generator_root(run_root: Path, generator: str) -> Path:
    return generator_directory(run_root, generator)
