"""Manifest-bound analysis for the 0022 test -> analyze operator protocol.

This reader never loads development calibration or a freeze artifact.  Existing
statistical procedures remain unchanged; only complete, real 270-question runs
with both generators can reach confirmatory analysis.  Failed answer positions,
fallbacks and missing independent judgments remain visible in the report.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from nea.evaluation.judging import RUBRIC_ANCHORS
from nea.evaluation.statistics import BOOTSTRAP_REPLICATES, SEED
from nea.pipeline.registry import ARMS_TEST, BASES, GENERATORS, JUDGE_ROLES
from nea.runtime.records import answer_sha256


PROTOCOL = "nea-0022-v1"
QUESTION_VERSION = "nea-questions-0022-v1"
DIMENSIONS = tuple(RUBRIC_ANCHORS)
PRIMARY = (("H1", "F", "B"), ("H2", "F", "P"))
SECONDARY = (("L-P", "L", "P"), ("F-L", "F", "L"), ("F-SR", "F", "SR"))
CONTRASTS = (("F-B", "F", "B"), ("F-P", "F", "P"), *SECONDARY)
PARENTS = {"F": "L", "SR": "B"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(value: Any, path: Path) -> list[dict[str, Any]]:
    result = value.get("rows") if isinstance(value, Mapping) else value
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ValueError(f"rows are missing or malformed: {path}")
    return result


def _identity(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest.get("identity", manifest)


def _qid_list(source: Mapping[str, Any]) -> list[str]:
    values = source.get("qids", source.get("selected_qids", []))
    if not isinstance(values, list) or any(not isinstance(qid, str) or not qid for qid in values):
        raise ValueError("manifest qids must be a list of nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("manifest repeats a qid")
    return values


def _load(run_root: Path, phase: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    phases, absent = {}, []
    for generator in GENERATORS:
        directory = run_root / "operator-v1" / generator.replace("/", "__") / phase
        names = ("answers", "judgments", "phase_summary", "run_manifest")
        missing = [name for name in names if not (directory / f"{name}.json").is_file()]
        absent.extend({"generator": generator, "phase": phase, "artifact": name} for name in missing)
        if "run_manifest" in missing:
            continue
        manifest = _read(directory / "run_manifest.json")
        snapshot, snapshot_path = None, None
        if missing:
            snapshots = sorted((directory / "incomplete").glob("*.json"), key=lambda path: path.name)
            if snapshots:
                snapshot_path = snapshots[-1]
                snapshot = _read(snapshot_path)
                if not isinstance(snapshot, Mapping):
                    raise ValueError("incomplete operator snapshot is malformed")
        summary = (_read(directory / "phase_summary.json") if "phase_summary" not in missing
                   else snapshot.get("summary") if snapshot else None)
        if summary is None:
            # The manifest still fixes all expected positions even if execution
            # stopped before the first recoverable phase snapshot was written.
            identity = _identity(manifest)
            summary = {"protocol": PROTOCOL, "phase": phase, "generator": generator,
                       "origin": identity.get("origin", "UNKNOWN"), "complete": False}
        if not isinstance(manifest, Mapping) or not isinstance(summary, Mapping):
            raise ValueError(f"malformed operator manifest/summary: {directory}")
        identity = _identity(manifest)
        if summary.get("origin") != identity.get("origin"):
            raise ValueError("summary and manifest have different artifact origins")
        for source in (identity, summary):
            protocol = source.get("protocol", source.get("protocol_version"))
            if protocol != PROTOCOL:
                raise ValueError("analysis refuses a historical or unknown protocol")
            if source.get("phase", source.get("split")) != phase:
                raise ValueError("live and test analysis identities cannot be mixed")
            if source.get("generator") != generator:
                raise ValueError("generator directory and identity differ")
        phases[generator] = {
            "manifest": manifest, "identity": identity, "summary": summary,
            "answers": _rows(_read(directory / "answers.json"), directory / "answers.json")
                if "answers" not in missing else list(snapshot.get("answers", [])) if snapshot else [],
            "judgments": _rows(_read(directory / "judgments.json"), directory / "judgments.json")
                if "judgments" not in missing else list(snapshot.get("judgments", [])) if snapshot else [],
            "directory": str(directory),
            "incomplete_snapshot": str(snapshot_path) if snapshot_path else None,
        }
    return phases, absent


def _scores(row: Mapping[str, Any]) -> Mapping[str, int] | None:
    value = row.get("scores")
    parsed = row.get("parsed")
    if value is not None and isinstance(parsed, Mapping) and parsed.get("scores") not in (None, value):
        raise ValueError("stored independent scores conflict with their raw parsed judgment")
    if value is None and isinstance(parsed, Mapping):
        value = parsed.get("scores")
    if value is None:
        return None
    if (not isinstance(value, Mapping) or set(value) != set(DIMENSIONS)
            or any(type(score) is not int or not 1 <= score <= 5 for score in value.values())):
        raise ValueError("independent evaluation requires six integer scores in 1..5")
    return value


def _mean_contrasts(means: Mapping[tuple[str, str, str, str], Mapping[str, float]],
                    qids: Sequence[str], cells: Sequence[tuple[str, str]]) -> dict[str, Any]:
    complete = [qid for qid in qids if all((generator, base, arm, qid) in means
                for generator, base in cells for arm in ARMS_TEST)]
    results: dict[str, Any] = {}
    for label, left, right in CONTRASTS:
        differences = {
            dimension: {qid: fmean(means[(g, b, left, qid)][dimension]
                                  - means[(g, b, right, qid)][dimension] for g, b in cells)
                        for qid in complete}
            for dimension in DIMENSIONS
        }
        results[label] = {
            "n_qids": len(complete), "status": "DESCRIPTIVE" if complete else "NOT_READY",
            "dimensions": {dimension: {
                "effect": round(fmean(values.values()), 6) if values else None,
                "declined_qids": sum(value < 0 for value in values.values()),
                "improved_qids": sum(value > 0 for value in values.values()),
                "tied_qids": sum(value == 0 for value in values.values()),
            } for dimension, values in differences.items()},
        }
    return {"n_complete_qids": len(complete), "complete_qids": complete,
            "comparisons": results, "unit": "qid", "cell_aggregation": "equal mean of generator x base cells"}


def _revision_report(answers: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
                     means: Mapping[tuple[str, str, str, str], Mapping[str, float]]) -> dict[str, Any]:
    states: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    rows = []
    harm_counts = {arm: {dimension: 0 for dimension in DIMENSIONS} for arm in PARENTS}
    improved = {arm: {dimension: 0 for dimension in DIMENSIONS} for arm in PARENTS}
    paired = Counter()
    missing = []
    missing_parent_metadata = []
    for key, answer in sorted(answers.items()):
        generator, base, arm, qid = key
        if arm not in PARENTS:
            continue
        revision = answer.get("revision") or {}
        if revision.get("parent_arm", PARENTS[arm]) != PARENTS[arm]:
            raise ValueError("revision has a different parent than the registered comparison")
        state = revision.get("status", revision.get("state"))
        if not state:
            if revision.get("fallback_reason"):
                state = "fallback"
            elif revision.get("revised") is True:
                state = "accepted"
            else:
                state = "unchanged"
        states[str(state)] += 1
        reason = revision.get("fallback_reason", revision.get("reason"))
        if reason:
            reasons[str(reason)] += 1
        outcome = ("accepted" if state == "accepted" or revision.get("revised") is True
                   else str(reason) if reason in {"no_issue", "uncertain", "no_executable_action"}
                   else "failure" if str(reason or "").endswith("failed") else "fallback")
        outcomes[outcome] += 1
        parent = (generator, base, PARENTS[arm], qid)
        parent_hash = revision.get("parent_sha256", revision.get("parent_hash"))
        if not revision.get("parent_arm") or not parent_hash:
            missing_parent_metadata.append(list(key))
        if (parent_hash and parent in answers
                and parent_hash != answers[parent].get("answer_sha256")):
            raise ValueError("revision parent hash differs from its paired parent answer")
        item = {"qid": qid, "generator": generator, "base": base, "arm": arm,
                "parent_arm": PARENTS[arm], "revision_state": str(state),
                "revision_outcome": outcome, "fallback_reason": reason, "status": "NOT_EVALUATED"}
        if key not in means or parent not in means:
            missing.append([generator, base, arm, qid])
        else:
            delta = {dimension: round(means[key][dimension] - means[parent][dimension], 6)
                     for dimension in DIMENSIONS}
            item.update(status="EVALUATED", score_delta=delta,
                        declined_dimensions=[dimension for dimension in DIMENSIONS if delta[dimension] < 0])
            paired[arm] += 1
            for dimension in DIMENSIONS:
                harm_counts[arm][dimension] += int(delta[dimension] < 0)
                improved[arm][dimension] += int(delta[dimension] > 0)
        rows.append(item)
    return {"definition": "independent two-teacher mean child minus parent, per generator x base x qid",
            "parents": PARENTS, "states": dict(sorted(states.items())),
            "outcomes": dict(sorted(outcomes.items())), "missing_parent_metadata": missing_parent_metadata,
            "fallback_reasons": dict(sorted(reasons.items())), "paired_cells": dict(paired),
            "declined_cells": harm_counts, "improved_cells": improved,
            "missing_score_pairs": missing, "rows": rows,
            "interpretation": "Score declines are observed modification harm; semantic validation is not independent quality evidence."}


def _cost_report(phases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source = [{"generator": generator, "costs": dict(phase["summary"].get("costs") or {})}
              for generator, phase in sorted(phases.items())]
    aliases = {"request_count": ("request_count", "requests"),
               "prompt_tokens": ("prompt_tokens",), "completion_tokens": ("completion_tokens",),
               "latency_ms": ("latency_ms", "total_latency_ms"), "cost": ("cost", "cost_usd"),
               "unknown_usage_count": ("unknown_usage_count",),
               "unknown_latency_count": ("unknown_latency_count",),
               "structure_failures": ("structure_failures",), "transport_failures": ("transport_failures",),
               "unresolved_requests": ("unresolved_requests",)}
    totals = {}
    for name, fields in aliases.items():
        values = [next((row["costs"][key] for key in fields if key in row["costs"]), None) for row in source]
        known = [value for value in values if type(value) in {int, float} and value >= 0]
        totals[name] = round(sum(known), 6) if len(known) == len(source) and source else None
    statuses = sorted({str(row["costs"].get("pricing_status", "NOT_CONFIGURED")) for row in source})
    return {"phases": source, "totals": totals,
            "pricing_status": statuses[0] if len(statuses) == 1 else "MIXED_OR_MISSING",
            "usage_complete": totals["unknown_usage_count"] == 0,
            "note": "Unknown monetary costs remain null; tokens and summed request latency are not money or wall-clock duration."}


def _collect(run_root: Path, phase: str, question_set: Any = None) -> dict[str, Any]:
    phases, missing_artifacts = _load(run_root, phase)
    split = "test" if phase == "test" else "dev"
    expected_rows = question_set.split_rows(split) if question_set is not None else []
    qids = sorted(row["qid"] for row in expected_rows)
    targets = {row["qid"]: row["authoring_target_id"] for row in expected_rows}
    hashes, versions, origins = set(), set(), set()
    declared_sets = []
    locked_fields = ("method_version", "method_source_hash", "prompt_files", "schema_hashes", "budget",
                     "comparisons", "question_set_version", "selected_questions_hash", "asset_identity",
                     "context_hash", "optional_comparisons", "evidence_policy", "modules", "arms", "nli", "ner",
                     "question_metadata_hash", "shared_template_hashes", "selected_exposure", "asset_report")
    fixed_contract = None
    manifest_missing_fields = []
    for generator, current in sorted(phases.items()):
        identity, summary = current["identity"], current["summary"]
        current_contract = {key: identity.get(key) for key in locked_fields}
        if fixed_contract is not None and current_contract != fixed_contract:
            raise ValueError("generator manifests have different methods, prompts, schemas, budgets or comparisons")
        fixed_contract = current_contract
        manifest_missing_fields.extend({"generator": generator, "field": key}
                                       for key in locked_fields if key not in identity)
        declared = _qid_list(identity) or _qid_list(summary)
        if not declared:
            raise ValueError("operator manifest must fix the selected qids")
        declared_sets.append(set(declared))
        for source in (identity, summary):
            value = source.get("question_set_sha256")
            if value:
                hashes.add(str(value))
            value = source.get("question_set_version", source.get("asset_version"))
            if value:
                versions.add(str(value))
            source_targets = source.get("qid_targets", {})
            if not isinstance(source_targets, Mapping):
                raise ValueError("qid_targets must be an object")
            for qid, target in source_targets.items():
                if qid in targets and targets[qid] != target:
                    raise ValueError("analysis target mapping conflicts with the question set")
                targets[qid] = str(target)
        origins.add(str(summary.get("origin", "UNKNOWN")))
    if len(hashes) > 1 or len(versions) > 1:
        raise ValueError("analysis cannot pool different question assets")
    if question_set is not None and hashes and hashes != {question_set.sha256}:
        raise ValueError("operator outputs use a different question set")
    if declared_sets and any(values != declared_sets[0] for values in declared_sets):
        raise ValueError("generator manifests select different qids")
    if qids and declared_sets and declared_sets[0] != set(qids):
        # A fixed subset is valid for development or descriptive test analysis.
        if not declared_sets[0] <= set(qids):
            raise ValueError("manifest qids lie outside the selected phase's question set")
        qids = sorted(declared_sets[0])
    elif not qids and declared_sets:
        qids = sorted(declared_sets[0])
    selected = set(qids)
    answers, judges = {}, {}
    uncertain, locatable, locatable_reasons = 0, 0, 0
    reason_evidence_states = Counter()
    for generator, current in sorted(phases.items()):
        for row in current["answers"]:
            key = (generator, row.get("base"), row.get("arm"), row.get("qid"))
            if (row.get("generator", generator) != generator or key[1] not in BASES
                    or key[2] not in ARMS_TEST or key[3] not in selected or key in answers):
                raise ValueError("duplicate or out-of-scope answer position")
            answers[key] = row
            target = row.get("authoring_target_id")
            if target:
                if key[3] in targets and targets[key[3]] != target:
                    raise ValueError("answer target mapping conflicts with manifest")
                targets[key[3]] = str(target)
        for row in current["judgments"]:
            key = (generator, row.get("base"), row.get("arm"), row.get("qid"), row.get("teacher"))
            if (row.get("generator", generator) != generator or key[1] not in BASES
                    or key[2] not in ARMS_TEST or key[3] not in selected or key[4] not in JUDGE_ROLES
                    or key in judges):
                raise ValueError("duplicate or out-of-scope independent judgment")
            judges[key] = row
            parsed = row.get("parsed") or {}
            uncertain += int(parsed.get("status") == "uncertain"
                             or bool(parsed.get("uncertain", row.get("uncertain", False))))
            for dimension, reason in parsed.get("reasons", {}).items():
                if isinstance(reason, Mapping):
                    reason_evidence_states[(dimension, str(reason.get("evidence_status", "missing")))] += 1
                    locatable_reasons += int(bool(reason.get("span") or reason.get("question_anchor")))
            findings = parsed.get("findings", row.get("findings", []))
            locatable += sum(bool(item.get("answer_span", item.get("span")) or item.get("question_anchor")) for item in findings
                             if isinstance(item, Mapping))
    missing_answers, missing_judgments, failures, means, chars = [], [], [], {}, {}
    for generator in GENERATORS:
        for base in BASES:
            for arm in ARMS_TEST:
                for qid in qids:
                    key = (generator, base, arm, qid)
                    row = answers.get(key)
                    if row is None or row.get("status") != "ok" or not str(row.get("answer", "")).strip():
                        missing_answers.append(list(key))
                        if row is not None:
                            failures.append({"generator": generator, "base": base, "arm": arm, "qid": qid,
                                             "status": row.get("status"), "reason": row.get("reason")})
                    else:
                        if row.get("answer_sha256") != answer_sha256(row["answer"]):
                            raise ValueError("recorded answer hash does not bind its actual text")
                        chars[key] = len(row["answer"])
                    values = []
                    for teacher in JUDGE_ROLES:
                        judgment = judges.get((*key, teacher))
                        score = None if judgment is None else _scores(judgment)
                        if score is None:
                            missing_judgments.append([*key, teacher])
                        else:
                            if not row or judgment.get("answer_sha256") != row.get("answer_sha256"):
                                raise ValueError("independent judgment belongs to a different answer")
                            values.append(score)
                    if len(values) == len(JUDGE_ROLES) and key in chars:
                        means[key] = {dimension: fmean(value[dimension] for value in values) for dimension in DIMENSIONS}
    cells = [(generator, base) for generator in GENERATORS for base in BASES]
    pooled = _mean_contrasts(means, qids, cells)
    per_cell = [{"generator": generator, "base": base,
                 **_mean_contrasts(means, qids, [(generator, base)])} for generator, base in cells]
    per_generator = [{"generator": generator,
                      **_mean_contrasts(means, qids, [(generator, base) for base in BASES])}
                     for generator in GENERATORS]
    per_base = [{"base": base,
                 **_mean_contrasts(means, qids, [(generator, base) for generator in GENERATORS])} for base in BASES]
    expected_answers = len(qids) * len(GENERATORS) * len(BASES) * len(ARMS_TEST)
    revision_report = _revision_report(answers, means)
    execution_failures = [{"generator": generator, "complete": current["summary"].get("complete"),
                           "stop_reason": current["summary"].get("stop_reason")}
                          for generator, current in sorted(phases.items())
                          if current["summary"].get("complete") is not True or current["summary"].get("stop_reason")]
    integrity = {"expected_answer_positions": expected_answers,
                 "recorded_answer_positions": len(answers), "usable_answer_positions": len(chars),
                 "expected_judgment_positions": expected_answers * len(JUDGE_ROLES),
                 "recorded_judgment_positions": len(judges),
                 "scored_answer_positions": len(means), "missing_artifacts": missing_artifacts,
                 "missing_answers": missing_answers, "missing_judgments": missing_judgments,
                 "failures": failures, "missing_target_qids": sorted(selected - targets.keys()),
                 "manifest_missing_fields": manifest_missing_fields,
                 "incomplete_snapshots": [current["incomplete_snapshot"] for current in phases.values()
                                          if current["incomplete_snapshot"]],
                 "missing_parent_metadata": revision_report["missing_parent_metadata"],
                 "phase_execution_failures": execution_failures,
                 "complete": bool(qids) and not (missing_artifacts or missing_answers or missing_judgments
                                                   or revision_report["missing_parent_metadata"] or execution_failures)}
    report = {"protocol": PROTOCOL, "phase": phase, "status": "DESCRIPTIVE",
              "data_origin": next(iter(origins)) if len(origins) == 1 else "MIXED_OR_MISSING",
              "question_set_sha256": next(iter(hashes)) if hashes else None,
              "question_set_version": next(iter(versions)) if versions else None,
              "n_qids": len(qids), "qids": qids, "integrity": integrity,
              "primary_contrasts": ["F-B", "F-P"], "secondary_contrasts": [name for name, _, _ in SECONDARY],
              "pooled": pooled, "by_cell": per_cell, "by_generator": per_generator, "by_base": per_base,
              "modification_harm": revision_report, "costs": _cost_report(phases),
              "evaluation_evidence": {"uncertain_judgments": uncertain, "locatable_findings": locatable,
                  "locatable_dimension_reasons": locatable_reasons,
                  "dimension_reason_evidence_status": {
                      dimension: dict(sorted((status, count) for (name, status), count in reason_evidence_states.items()
                                             if name == dimension)) for dimension in DIMENSIONS},
                  "raw_parsed_records": [item["incomplete_snapshot"] or str(Path(item["directory"]) / "judgments.json")
                                         for item in phases.values()]},
              "optional_experiments": {"module_ablations": [], "SR-L": [], "NLI": []},
              "new_measurements": {"LTS_NEA_v1": {"status": "NOT_IMPLEMENTED",
                  "reason": "formula, independent reference and validity are not defined"}},
              "exposure_disclosure": {
                  "question_metadata_hash": (fixed_contract or {}).get("question_metadata_hash"),
                  "selected_exposure": (fixed_contract or {}).get("selected_exposure", {}),
                  "asset_report": (fixed_contract or {}).get("asset_report", {}),
                  "source": "locked run manifest fields copied from accepted question metadata",
                  "claim_rule": "Known exposure and unknown history remain disclosed; a formal test label does not establish unseen questions."},
              "comparison_limits": "F-SR has different parents (L/B); L-P alone does not identify graph-edge causality; case3 is excluded."}
    return {"report": report, "means": means, "chars": chars, "targets": targets, "phases": phases}


def analyze_operator_run(run_root: Path, question_set: Any = None) -> dict[str, Any]:
    """Analyze test records; incomplete or fixture runs yield explicit reports.

    Passing the already loaded QuestionSet avoids reading frozen assets again.
    Without it, qids must be fixed in the run artifacts; targets are descriptive metadata.
    """
    state = _collect(Path(run_root), "test", question_set)
    report, means, chars, targets = (state[key] for key in ("report", "means", "chars", "targets"))
    conditions = {
        "real_outputs": report["data_origin"] == "REAL",
        "full_270_questions": report["n_qids"] == 270,
        "both_generators": set(state["phases"]) == set(GENERATORS),
        "complete_positions_and_judgments": report["integrity"]["complete"],
        "new_asset_version": report["question_set_version"] == QUESTION_VERSION,
        "complete_manifest_contract": not report["integrity"]["manifest_missing_fields"],
        "at_least_two_questions": len(report["qids"]) >= 2,
    }
    report["formal_eligibility"] = {"eligible": all(conditions.values()), "conditions": conditions}
    report["bootstrap"] = {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED,
                           "unit": "qid", "quantile": "linear p*(N-1)"}
    if not report["integrity"]["complete"]:
        report["status"] = "NOT_READY_MISSING_RESULTS"
    if not all(conditions.values()):
        report["formal"] = {"status": "NOT_RUN", "reason": "requires complete real 270-question two-generator test with the accepted asset version"}
        return report
    # Reuse the established inference implementation only for an eligible real
    # run; descriptive readers stay independent of the historical phase engine.
    from nea.evaluation.analysis import bootstrap_mean, holm, length_sensitivity

    qids = report["pooled"]["complete_qids"]
    cells = [(generator, base) for generator in GENERATORS for base in BASES]
    quality = {(qid, arm): fmean(means[(g, b, arm, qid)]["overall_quality"] for g, b in cells)
               for qid in qids for arm in ARMS_TEST}

    def differences(left: str, right: str) -> dict[str, float]:
        return {qid: quality[(qid, left)] - quality[(qid, right)] for qid in qids}

    primary, open_gate = {}, True
    for name, left, right in PRIMARY:
        if not open_gate:
            primary[name] = {"contrast": f"{left}-{right}", "status": "NOT_TESTED"}
            continue
        row = bootstrap_mean(differences(left, right), replicates=BOOTSTRAP_REPLICATES, seed=SEED)
        open_gate = row["ci95"][0] > 0
        primary[name] = {"contrast": f"{left}-{right}", **row,
                         "status": "SUPPORTED" if open_gate else "NOT_SUPPORTED"}
    secondary = {name: bootstrap_mean(differences(left, right),
                                     replicates=BOOTSTRAP_REPLICATES, seed=SEED)
                 for name, left, right in SECONDARY}
    corrections = holm({name: row["p_two_sided"] for name, row in secondary.items()})
    for name in secondary:
        secondary[name]["holm"] = corrections[name]
    length_rows = [{"qid": qid, "arm": arm, "y": quality[(qid, arm)],
                    "chars": fmean(chars[(g, b, arm, qid)] for g, b in cells)}
                   for qid in qids for arm in ARMS_TEST]
    report.update(status="READY_FOR_CONFIRMATORY_ANALYSIS", formal={
        "status": "ESTIMATED", "unit": "qid", "n_qids": len(qids),
        "n_targets": len({targets[qid] for qid in qids if qid in targets}),
        "primary": {"procedure": "fixed_sequence", "alpha": 0.05, **primary},
        "secondary": {"procedure": "holm", **secondary},
        "length_sensitivity": length_sensitivity(length_rows,
                                                   replicates=BOOTSTRAP_REPLICATES, seed=SEED),
        "claim_rule": "H2 is interpreted only if H1 lower CI > 0; interval crossing 0 does not establish improvement."})
    # These strata disclose uncertainty without adding another family of
    # confirmatory hypotheses or treating the four cells as independent qids.
    for field, cell_selector in (
        ("by_generator", lambda item: [(item["generator"], base) for base in BASES]),
        ("by_base", lambda item: [(generator, item["base"]) for generator in GENERATORS]),
        ("by_cell", lambda item: [(item["generator"], item["base"])]),
    ):
        for item in report[field]:
            stratum = cell_selector(item)
            item["uncertainty"] = {
                "status": "AUXILIARY_INTERVALS", "unit": "qid",
                "metric": "overall_quality", "comparisons": {
                    label: bootstrap_mean(
                        {qid: fmean(means[(g, b, left, qid)]["overall_quality"]
                                    - means[(g, b, right, qid)]["overall_quality"] for g, b in stratum)
                         for qid in qids}, replicates=BOOTSTRAP_REPLICATES, seed=SEED)
                    for label, left, right in CONTRASTS},
                "interpretation": "Stratified descriptive uncertainty; no additional confirmatory significance claim.",
            }
    return report


def compare_operator_run(run_root: Path, phase: str = "live") -> dict[str, Any]:
    """Descriptive paired comparisons for live or a fixed test subset."""
    if phase not in {"live", "test"}:
        raise ValueError("operator comparison phase must be live or test")
    report = _collect(Path(run_root), phase)["report"]
    if not report["integrity"]["complete"]:
        report["status"] = "NOT_READY_MISSING_RESULTS"
    report["formal"] = {"status": "NOT_RUN", "reason": "compare is descriptive; live cannot produce formal test inference"}
    return report
