"""Deterministic, manifest-bound analysis and paper data for formal8.

Only saved run records are read: this module never loads frozen assets or calls
a model.  Descriptive diagnostics remain available for interrupted/fixture runs,
but every paper export is explicitly NOT_RUN until its real input is complete.
Both formal and supplementary comparisons resample whole questions.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from nea.evaluation.analysis import bootstrap_mean, question_draw_counts, fixed_effects_ols, holm
from nea.evaluation.judging import RUBRIC_ANCHORS
from nea.evaluation.statistics import BOOTSTRAP_REPLICATES, SEED, linear_quantile
from nea.pipeline.registry import BASES, GENERATORS, JUDGE_ROLES
from nea.runtime.records import ConflictError, answer_sha256, canonical_bytes, digest, write_once

PROTOCOL = "nea-formal8-v1"
ARMS = ("B", "P", "L", "F", "SR", "SRL", "FB", "CV")
PARENTS = {"F": "L", "SRL": "L", "FB": "B", "SR": "B", "CV": "B"}
DIMENSIONS = tuple(RUBRIC_ANCHORS)
PRIMARY = (("F-B", "F", "B"), ("F-L", "F", "L"), ("F-SRL", "F", "SRL"))
SECONDARY = (("FB-B", "FB", "B"), ("FB-SR", "FB", "SR"), ("F-CV", "F", "CV"),
             ("F-SR", "F", "SR"), ("SRL-L", "SRL", "L"), ("L-P", "L", "P"),
             ("P-B", "P", "B"), ("CV-B", "CV", "B"))
CONTRASTS = PRIMARY + SECONDARY
# Spec 0045: GPlan+CoVe (CoVe on the Host+GPlan parent) was run after the eight registered arms.
# Its rows live in the same answers/judgments files but never enter the eight-arm statistics,
# the fixed sequence or the Holm family; they are reported as a separate supplementary section.
# Spec 0046 adds Self-Refine, CoVe and NEA-Refine on the Host+Plan parent (SRP, CVP, FP) to the same section.
SUPPLEMENT_ARMS = ("CVL", "SRP", "CVP", "FP")
SUPPLEMENT_PARENTS = {"CVL": "L", "SRP": "P", "CVP": "P", "FP": "P"}
SUPPLEMENT_COMPARISONS = (("F-CVL", "F", "CVL"), ("CVL-L", "CVL", "L"),
                          ("CVL-SRL", "CVL", "SRL"), ("CVL-CV", "CVL", "CV"),
                          ("SRP-P", "SRP", "P"), ("CVP-P", "CVP", "P"), ("FP-P", "FP", "P"),
                          ("FP-SRP", "FP", "SRP"), ("FP-CVP", "FP", "CVP"),
                          ("FP-F", "FP", "F"), ("SRP-SRL", "SRP", "SRL"), ("CVP-CVL", "CVP", "CVL"),
                          ("FP-FB", "FP", "FB"))
DEFECTS = ("term_gap", "step_gap", "logical_jump", "term_excess", "step_excess")
PAPER_ITEMS = {
    "table-06": ("Main results", "main"),
    "table-07": ("Need-aware metrics", "need_metrics"),
    "table-08": ("Ablations", "ablations"),
    "table-09": ("Diagnosis validity", "perturbations"),
    "table-10": ("Efficiency", "efficiency"),
    "table-11": ("Fixed cases and error patterns", "cases"),
    "figure-04": ("Main results", "main"),
    "figure-05": ("Residual defects", "defects"),
    "figure-06": ("Parameter sensitivity", "sensitivity"),
    "figure-07": ("Quality and cost", "efficiency"),
    "figure-08": ("Stratified results", "strata"),
}
LOCKED_FIELDS = ("method_version", "method_source_hash", "prompt_files", "schema_hashes",
                 "budget", "comparisons_registry", "question_set_version", "selected_questions_hash",
                 "asset_identity", "arms", "question_metadata_hash", "context_hash")


# Spec 0044: the live (E6 sensitivity) experiment is abolished and cost accounting is reported as a lower bound.
WAIVED_CONDITIONS = frozenset({"complete_development_sensitivity", "complete_efficiency_records"})


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _not_run(reason: str, **fields: Any) -> dict[str, Any]:
    return {"status": "NOT_RUN", "reason": reason, **fields}


def _identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return value.get("identity", value)


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in ("generator", "base", "arm", "qid"))


def _valid_number(value: Any) -> bool:
    return type(value) in {float, int} and math.isfinite(value)


def _scores(row: Mapping[str, Any]) -> Mapping[str, int] | None:
    parsed = row.get("parsed") or {}
    scores = row.get("scores", parsed.get("scores"))
    if row.get("status") not in {None, "ok", "success", "COMPLETE"}:
        return None
    if scores is None:
        return None
    if parsed.get("scores") not in (None, scores):
        raise ValueError("judgment scores conflict with parsed result")
    if (not isinstance(scores, Mapping) or set(scores) != set(DIMENSIONS)
            or any(type(score) is not int or not 1 <= score <= 5 for score in scores.values())):
        raise ValueError("judgment requires six integer scores in 1..5")
    return scores


def _estimate(values: Mapping[str, float]) -> dict[str, Any]:
    if len(values) < 2:
        return _not_run("at least two paired questions required", n_qids=len(values),
                        effect=_round(fmean(values.values())) if values else None)
    return {"status": "ESTIMATED", **bootstrap_mean(values,
            replicates=BOOTSTRAP_REPLICATES, seed=SEED)}


def statistical_summary(quality: Mapping[tuple[str, str], float],
                        correctness: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    """Pure statistical calculation; this utility does not certify run provenance.

    Inputs are already equally averaged over the four generator/base cells and
    the two independent judges.  The qid, never a cell or judgment, is the unit.
    """
    qids = sorted(qid for qid in {qid for qid, _ in quality} if all((qid, arm) in quality and
                   (qid, arm) in correctness for arm in ARMS))

    def delta(left: str, right: str, scores=quality) -> dict[str, float]:
        return {qid: scores[qid, left] - scores[qid, right] for qid in qids}

    primary, gate = {}, True
    for label, left, right in PRIMARY:
        if not gate:
            primary[label] = {"status": "NOT_TESTED", "reason": "previous fixed-sequence hypothesis did not pass"}
            continue
        item = _estimate(delta(left, right))
        gate = item.get("status") == "ESTIMATED" and item["ci95"][0] > 0
        primary[label] = {**item, "status": "SUPPORTED" if gate else
                          "NOT_SUPPORTED" if item.get("status") == "ESTIMATED" else "NOT_RUN"}
    secondary = {label: _estimate(delta(left, right)) for label, left, right in SECONDARY}
    adjusted = holm({label: item["p_two_sided"] for label, item in secondary.items()
                     if item.get("status") == "ESTIMATED"})
    # A missing member cannot silently shrink the registered multiplicity family.
    if len(adjusted) == len(SECONDARY):
        for label, correction in adjusted.items():
            secondary[label]["holm"] = correction
    noninferiority = {}
    for left, right in (("F", "B"), ("F", "L")):
        item = _estimate(delta(left, right, correctness))
        noninferiority[f"{left}-{right}"] = {**item, "margin": -0.10,
            "noninferior": item["ci95"][0] > -0.10 if item.get("status") == "ESTIMATED" else None}
    return {"calculation_only": True, "unit": "qid", "n_qids": len(qids),
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED,
                          "unit": "qid", "quantile": "linear p*(N-1)"},
            "primary": {"procedure": "fixed_sequence", "alpha": 0.05, "comparisons": primary},
            "secondary": {"procedure": "holm", "family_size": len(SECONDARY), "comparisons": secondary},
            "correctness_noninferiority": noninferiority,
            "claim_rule": "F-L is interpreted only after F-B passes; F-SRL only after both preceding lower bounds exceed zero."}


def _length_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The registered question-FE model extended to all eight arms."""
    if len({row["qid"] for row in rows}) < 2:
        return _not_run("at least two complete questions required")
    fit = fixed_effects_ols(rows, ARMS)
    if fit["beta"] is None:
        return {"status": "NOT_IDENTIFIED", "reason": "length and arm are collinear after question fixed effects"}
    names, blocks = fit["names"], fit["blocks"]
    qids, dim = sorted(blocks), len(names)
    xtx = np.array([blocks[qid][0].T @ blocks[qid][0] for qid in qids])
    xty = np.array([blocks[qid][0].T @ blocks[qid][1] for qid in qids])
    counts = question_draw_counts(len(qids), BOOTSTRAP_REPLICATES, SEED)
    draw_x, draw_y = np.einsum("rc,cij->rij", counts, xtx), np.einsum("rc,ci->ri", counts, xty)
    solvable = np.linalg.matrix_rank(draw_x) == dim
    if not solvable.any():
        return {"status": "NOT_IDENTIFIED", "reason": "all bootstrap fits singular"}
    betas = np.linalg.solve(draw_x[solvable], draw_y[solvable][..., None])[..., 0]

    def coefficient(beta, arm):
        return beta[..., names.index(f"arm:{arm}")] if arm != "B" else np.zeros(beta.shape[:-1])

    comparisons = {}
    for label, left, right in CONTRASTS:
        draws = np.sort(coefficient(betas, left) - coefficient(betas, right))
        comparisons[label] = {"effect": _round(coefficient(fit["beta"], left) - coefficient(fit["beta"], right)),
            "ci95": [_round(linear_quantile(draws, p)) for p in (0.025, 0.975)]}
    return {"status": "ESTIMATED", "model": "Y ~ question FE + arm + log(chars)",
            "reference_arm": "B", "comparisons": comparisons,
            "coefficients": {name: _round(value) for name, value in zip(names, fit["beta"])},
            "singular_replicates": int((~solvable).sum())}


def _factorial(quality, qids, targets):
    contrasts = {
        "support_main": (("L", 1 / 3), ("SRL", 1 / 3), ("F", 1 / 3),
                         ("B", -1 / 3), ("SR", -1 / 3), ("FB", -1 / 3)),
        "generic_revision_main": (("SR", 0.5), ("SRL", 0.5), ("B", -0.5), ("L", -0.5)),
        "nea_revision_main": (("FB", 0.5), ("F", 0.5), ("B", -0.5), ("L", -0.5)),
        "support_x_generic": (("SRL", 1), ("L", -1), ("SR", -1), ("B", 1)),
        "support_x_nea": (("F", 1), ("L", -1), ("FB", -1), ("B", 1)),
        "support_x_nea_vs_generic": (("F", 1), ("SRL", -1), ("FB", -1), ("SR", 1)),
    }
    return {"status": "DESCRIPTIVE", "design": "parent B/L x none/generic/NEA revision",
            "cells": [{"parent": parent, "revision": revision, "arm": arm,
                       "overall_quality": _round(fmean(quality[qid, arm] for qid in qids)) if qids else None}
                      for parent, revision, arm in (("B", "none", "B"), ("B", "generic", "SR"),
                          ("B", "NEA", "FB"), ("L", "none", "L"), ("L", "generic", "SRL"), ("L", "NEA", "F"))],
            "contrasts": {label: _estimate({qid: sum(weight * quality[qid, arm] for arm, weight in terms)
                          for qid in qids}) for label, terms in contrasts.items()},
            "interpretation": "Exploratory factorial contrasts; no additional confirmatory hypothesis family."}


def _revision_report(answers, means):
    rows, errors = [], []
    for key, answer in sorted(answers.items()):
        generator, base, arm, qid = key
        if arm not in PARENTS:
            continue
        revision, parent = answer.get("revision") or {}, answers.get((generator, base, PARENTS[arm], qid))
        declared_hash = revision.get("parent_sha256", revision.get("parent_hash", answer.get("parent_sha256")))
        if (revision.get("parent_arm") != PARENTS[arm] or not parent or
                declared_hash != parent.get("answer_sha256")):
            errors.append(list(key))
        parent_key = (generator, base, PARENTS[arm], qid)
        changed = bool(parent and answer.get("answer_sha256") != parent.get("answer_sha256"))
        delta = (means[key]["overall_quality"] - means[parent_key]["overall_quality"]
                 if key in means and parent_key in means else None)
        rows.append({"qid": qid, "generator": generator, "base": base, "arm": arm,
                     "parent_arm": PARENTS[arm], "revised": changed,
                     "state": revision.get("status", revision.get("state")),
                     "reason": revision.get("fallback_reason", revision.get("reason")),
                     "overall_delta": _round(delta), "harmed": delta < 0 if delta is not None else None})
    summaries = []
    for arm in PARENTS:
        selected = [row for row in rows if row["arm"] == arm]
        revised = [row for row in selected if row["revised"]]
        scored = [row for row in revised if row["harmed"] is not None]
        summaries.append({"arm": arm, "n_cells": len(selected), "revised_cells": len(revised),
            "scored_revised_cells": len(scored), "harmed_cells": sum(row["harmed"] for row in scored),
            "harm_rate": _round(fmean(row["harmed"] for row in scored)) if scored else None,
            "revised_rate": _round(len(revised) / len(selected)) if selected else None,
            "missing_revision_scores": len(revised) - len(scored)})
    return {"definition": "score declines among answers whose final text differs from their registered parent",
            "rows": rows, "by_arm": summaries, "parent_errors": errors}


def _complete_extras(extras: Mapping[str, Any], origins: set[str]) -> dict[str, Any]:
    result = {}
    for name, document in sorted(extras.items()):
        if not isinstance(document, Mapping):
            result[name] = {"complete": False, "reason": "missing artifact envelope"}
            continue
        rows = document.get("rows", [])
        complete = (document.get("status") == "COMPLETE" and document.get("origin") == "REAL"
                    and origins == {"REAL"} and document.get("protocol", document.get("protocol_version")) == PROTOCOL
                    and isinstance(rows, list)
                    and not any(row.get("status") in {"failed", "missing", "NOT_RUN", "incomplete", "stopped"}
                                for row in rows if isinstance(row, Mapping)))
        result[name] = {"complete": complete, "n_rows": len(rows) if isinstance(rows, list) else 0,
                        "reason": None if complete else "missing, non-real, incomplete or stopped artifact"}
    return result


def _attach_variant_scores(extras, judgments):
    """Bind every variant's independent scores to its comparison and text."""
    copied = {name: {**document, "rows": [dict(row) for row in document.get("rows", [])]}
              for name, document in extras.items()}
    index = {}
    for row in judgments:
        key = (*_key(row), row.get("comparison"), row.get("teacher"))
        if key in index:
            raise ValueError("duplicate variant judgment position")
        index[key] = row
    expected = set()
    for name in ("ablations", "validation_ablation", "sensitivity"):
        for row in copied.get(name, {}).get("rows", []):
            row.pop("mean_scores", None)
            row["judgments"] = []
            for teacher in JUDGE_ROLES:
                key = (*_key(row), row.get("comparison"), teacher)
                expected.add(key)
                score = index.get(key)
                if score is None:
                    continue
                if score.get("answer_sha256") != row.get("answer_sha256"):
                    raise ValueError("variant judgment belongs to a different answer")
                row["judgments"].append(score)
    if set(index) - expected:
        raise ValueError("orphan variant judgment")
    return copied


def _extra_coverage(extras, checks, answers, qids, metadata, phase):
    """Check exact assignments, independent judge identity and answer hashes."""
    from nea.evaluation.formal_measurement import (cross_family_teacher,
        select_t90, select_perturbation_sources)
    from nea.method.formal_operators import ABLATIONS, SENSITIVITIES

    def check(name, expected, positions, valid=True):
        item = checks.setdefault(name, {"complete": False, "reason": "missing artifact"})
        unique = set(positions)
        item.update(expected_positions=len(expected), recorded_positions=len(positions),
                    missing_positions=len(expected - unique), unexpected_positions=len(unique - expected),
                    duplicate_positions=len(positions) - len(unique))
        item["complete"] = bool(item["complete"] and valid and unique == expected and len(unique) == len(positions))
        if not item["complete"]:
            item["reason"] = "artifact provenance, assignment, answer binding or score coverage incomplete"

    measurement_rows = extras.get("measurements", {}).get("rows", [])
    primary = [row for row in measurement_rows if row.get("primary") is True]
    expected = {(*key, cross_family_teacher(key[0])) for key in answers}
    positions = [(*_key(row), row.get("teacher")) for row in primary]
    def valid_metric(metric):
        if not isinstance(metric, Mapping):
            return False
        numerator, denominator = metric.get("numerator"), metric.get("denominator")
        return (type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator
                and metric.get("applicable") is (denominator > 0)
                and (metric.get("value") is None if denominator == 0 else
                     _valid_number(metric.get("value")) and abs(metric["value"] - numerator / denominator) < 1e-8))
    valid = all(_key(row) in answers and row.get("answer_sha256") == answers[_key(row)].get("answer_sha256")
                and all(valid_metric(row.get(metric)) for metric in ("NCov", "ROE")) for row in primary)
    check("measurements", expected, positions, valid)
    selected = set(select_t90([{"qid": qid, "style": metadata.get(qid, {}).get("question_type")} for qid in qids]))
    expected = {(g, b, arm, qid, cross_family_teacher(g)) for g in GENERATORS for b in BASES
                for arm in ("B", "L", "SR", "SRL", "FB", "F") for qid in selected}
    audit_rows = extras.get("audits", {}).get("rows", [])
    positions = [(*_key(row), row.get("teacher")) for row in audit_rows]
    valid = (phase != "test" or len(selected) == 90) and all(_key(row) in answers and
        row.get("answer_sha256") == answers[_key(row)].get("answer_sha256") for row in audit_rows)
    check("audits", expected, positions, valid)
    for name, registry, prefix in (("ablations", ABLATIONS, "ablation:"),
                                   ("sensitivity", SENSITIVITIES, "sensitivity:")):
        expected = {(g, b, "F", qid, prefix + variant) for g in GENERATORS for b in BASES
                    for qid in qids for variant in registry}
        rows = extras.get(name, {}).get("rows", [])
        positions = [(*_key(row), row.get("comparison")) for row in rows]
        valid = all(row.get("status") == "ok" and row.get("answer_sha256") == answer_sha256(row["answer"])
                    and len(row.get("judgments", [])) == 2 and
                    all(_scores(judgment) is not None for judgment in row["judgments"]) for row in rows)
        check(name, expected, positions, valid)
    expected = {key for key, row in answers.items() if key[2] == "F" and
                (row.get("revision") or {}).get("fallback_reason") in {"semantic_rejected", "semantic_uncertain"}}
    rows = extras.get("validation_ablation", {}).get("rows", [])
    valid = all(row.get("generation_calls") == 0 and len(row.get("judgments", [])) == 2
                and all(_scores(judgment) is not None for judgment in row["judgments"])
                and row.get("protected_answer_sha256") == answers.get(_key(row), {}).get("answer_sha256")
                and row.get("answer_sha256") == answer_sha256(row["answer"])
                and row.get("answer") == (answers.get(_key(row), {}).get("revision", {}).get("candidate_record") or {}).get("answer")
                for row in rows)
    check("validation_ablation", expected, [_key(row) for row in rows], valid)
    sources = []
    for key, row in answers.items():
        diagnosis = row.get("diagnosis") or {}
        if key[2] in {"F", "FB"} and diagnosis.get("status") == "no_issue":
            parent = answers.get((key[0], key[1], PARENTS[key[2]], key[3]))
            if parent:
                sources.append({**parent, "diagnosis": diagnosis, "source_arm": PARENTS[key[2]], "split": "test"})
    selected_sources = select_perturbation_sources(sources, 200)
    expected = {(row["generator"], row["base"], row["qid"], answer_sha256(row["answer"]), kind)
                for row in selected_sources for kind in (*DEFECTS, "null")}
    rows = extras.get("perturbations", {}).get("rows", [])
    positions = [(row.get("generator"), row.get("base"), row.get("qid"), row.get("source_sha256"), row.get("kind")) for row in rows]
    valid = all(row.get("status") == "inapplicable" or
        (row.get("status") == "ok" and row.get("injection") is not None and
         row.get("answer_sha256") == answer_sha256(row["answer"]) and
         {item.get("lane") for item in row.get("diagnostics", [])} == {"same_family", "cross_family"}
         and all(item.get("status") == "ok" for item in row["diagnostics"])) for row in rows)
    check("perturbations", expected, positions, valid)
    rows = extras.get("cases", {}).get("rows", [])
    expected = {(g, b, arm, qid) for g in GENERATORS for b in BASES for arm in ARMS for qid in ("Q410", "Q240", "Q213")}
    valid = all(row.get("status") == "ok" and row.get("answer_sha256") == answer_sha256(row["answer"]) for row in rows)
    check("cases", expected, [_key(row) for row in rows], valid)


def _metric_rows(extras):
    rows = [row for row in extras.get("measurements", {}).get("rows", [])
            if row.get("primary") is True and row.get("status") in {"ok", "success", "COMPLETE"}]
    result = []
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        for name in ("NCov", "ROE"):
            per_qid = defaultdict(list)
            for row in selected:
                metric = row.get(name) or {}
                if metric.get("applicable") is True and _valid_number(metric.get("value")):
                    per_qid[row["qid"]].append(float(metric["value"]))
            # Missing cells do not silently receive extra weight.
            complete = {qid: fmean(values) for qid, values in per_qid.items() if len(values) == 4}
            result.append({"arm": arm, "metric": name, "value": _round(fmean(complete.values())) if complete else None,
                "n_applicable_qids": len(complete), "n_recorded_qids": len({row["qid"] for row in selected}),
                "n_na_qids": len({row["qid"] for row in selected if (row.get(name) or {}).get("applicable") is False}),
                "n_incomplete_qids": sum(len(values) != 4 for values in per_qid.values())})
    return result


def _defect_rows(extras):
    rows = extras.get("audits", {}).get("rows", [])
    results = []
    for arm in ("B", "L", "SR", "SRL", "FB", "F"):
        assigned = [row for row in rows if row.get("arm") == arm]
        selected = [row for row in assigned if row.get("status") in {"ok", "success", "COMPLETE"}
                    and (row.get("parsed") or {}).get("status") in {"issues", "no_issue"}]
        for kind in DEFECTS:
            hits = sum(kind in row.get("defects", []) for row in selected)
            results.append({"arm": arm, "defect": kind, "n_answers": len(selected), "n_assigned": len(assigned),
                            "missing_or_uncertain": len(assigned) - len(selected), "n_residual": hits,
                            "residual_rate": _round(hits / len(selected)) if selected else None})
    return results


def _perturbation_rows(extras):
    rows = extras.get("perturbations", {}).get("rows", [])
    groups = defaultdict(list, {(kind, lane): [] for kind in (*DEFECTS, "null")
                               for lane in ("same_family", "cross_family")})
    for row in rows:
        for diagnostic in row.get("diagnostics", []):
            groups[row.get("kind"), diagnostic.get("lane")].append((row, diagnostic))
    result = []
    for (kind, lane), items in sorted(groups.items()):
        valid = [(row, d) for row, d in items if row.get("status") in {"ok", "success", "COMPLETE"}
                 and d.get("status") in {"ok", "success", "COMPLETE"} and d.get("measurement_status") == "ok"]
        null = bool(valid) and all(not row.get("expected_type") for row, _ in valid)
        assigned = [row for row in rows if row.get("kind") == kind]
        inapplicable = sum(row.get("status") == "inapplicable" for row in assigned)
        missing = max(0, len(assigned) - inapplicable - len(valid))
        status = "ESTIMATED" if valid else "NOT_RUN" if missing else "NA"
        result.append({"kind": kind, "lane": lane, "n_assigned": len(assigned), "n_valid": len(valid),
            "status": status, "reason": None if valid else "missing_or_uncertain_diagnosis" if missing else
                "no_eligible_source_answers" if not assigned else "no_applicable_perturbations",
            "n_inapplicable": inapplicable, "missing_or_uncertain": missing,
            "type_detection_rate": _round(fmean(bool(d.get("type_detected")) for _, d in valid)) if valid and not null else None,
            "location_hit_rate": _round(fmean(bool(d.get("location_hit")) for _, d in valid)) if valid and not null else None,
            "null_false_positive_rate": _round(fmean(bool(d.get("false_positive")) for _, d in valid)) if valid and null else None})
    return result


def _variant_rows(document, means, targets, *, require_four=True):
    groups = defaultdict(list)
    for row in document.get("rows", []):
        groups[str(row.get("comparison", row.get("variant", "unknown")))].append(row)
    result = []
    for name, rows in sorted(groups.items()):
        qids = defaultdict(list)
        unscored = 0
        for row in rows:
            key = _key(row)
            scores = row.get("mean_scores")
            if not isinstance(scores, Mapping):
                judgments = row.get("judgments", [])
                valid = [_scores(judgment) for judgment in judgments]
                scores = ({dimension: fmean(item[dimension] for item in valid) for dimension in DIMENSIONS}
                          if len(valid) == 2 and all(valid) else None)
            baseline = means.get((key[0], key[1], "F", key[3]))
            if scores and baseline and _valid_number(scores.get("overall_quality")):
                qids[key[3]].append(float(scores["overall_quality"]) - baseline["overall_quality"])
            else:
                unscored += 1
        paired = {qid: fmean(values) for qid, values in qids.items() if not require_four or len(values) == 4}
        result.append({"comparison": name, "n_positions": len(rows), "unscored_positions": unscored,
                       "reused_positions": sum(bool(row.get("reused_from", row.get("reuse_from"))) for row in rows),
                       "effect_vs_F": _estimate(paired),
                       "population": "all four cells" if require_four else "cells with a rejected or uncertain candidate"})
    return result


def _efficiency_rows(answers, means, summaries, extras):
    accounting_failures = []
    for generator in GENERATORS:
        telemetry = summaries.get(generator, {}).get("telemetry", {})
        if (telemetry.get("accounting_complete") is not True or
                telemetry.get("unmeasured_request_count", 0) != 0 or
                telemetry.get("unknown_usage_request_count", 0) != 0):
            accounting_failures.append({"generator": generator,
                "reason": "missing_or_incomplete_phase_accounting",
                "accounting_complete": telemetry.get("accounting_complete"),
                "unmeasured_request_count": telemetry.get("unmeasured_request_count"),
                "unknown_usage_request_count": telemetry.get("unknown_usage_request_count")})
    sources = extras.get("costs", {}).get("rows", [])
    if not sources:
        sources = []
        for generator, summary in summaries.items():
            telemetry = summary.get("telemetry", {})
            grouped = telemetry.get("by_comparison", {}).get("main", {})
            by_arm = grouped.get("by_arm", grouped) if grouped else summary.get("costs_by_arm", {})
            for arm, cost in by_arm.items():
                if isinstance(cost, Mapping):
                    sources.append({"generator": generator, "arm": arm, **cost,
                                    "latency_ms": cost.get("latency_ms", cost.get("elapsed_ms"))})
    result = []
    for arm in ARMS:
        selected = [row for row in sources if row.get("arm") == arm]
        cell_answers = [(key, row) for key, row in answers.items() if key[2] == arm]
        revisions = [row.get("revision") or {} for _, row in cell_answers]
        triggered = sum((row.get("calls") or {}).get("revise", 0) > 0 or row.get("candidate_count", 0) > 0
                        for row in revisions)
        accepted = sum(row.get("revised") is True or row.get("status") == "accepted" for row in revisions)
        item = {"arm": arm, "n_answers": len(cell_answers),
                "cost_status": "UNKNOWN" if selected and accounting_failures else "RECORDED" if selected else "NOT_RUN",
                "accounting_complete": not accounting_failures,
                "accounting_failures": accounting_failures,
                "observed_lower_bound": {},
                "revision_trigger_rate": _round(triggered / len(revisions)) if revisions and arm in PARENTS else None,
                "revision_acceptance_rate": _round(accepted / triggered) if triggered else None,
                "parent_retention_rate": _round(1 - accepted / len(revisions)) if revisions and arm in PARENTS else None,
                "mean_overall_quality": _round(fmean(means[key]["overall_quality"] for key, _ in cell_answers if key in means))
                                         if any(key in means for key, _ in cell_answers) else None}
        for name in ("request_count", "prompt_tokens", "completion_tokens", "latency_ms"):
            values = [row.get(name) for row in selected]
            observed = _round(sum(values)) if values and all(_valid_number(value) and value >= 0 for value in values) else None
            unknown = any(row.get("unknown_usage_request_count", 0) > 0 for row in selected) if name in {
                "prompt_tokens", "completion_tokens"} else any(row.get("unmeasured_request_count", 0) > 0 for row in selected)
            item["observed_lower_bound"][name] = observed
            item[name] = observed if not accounting_failures and not unknown else None
        item["monetary_cost"] = None
        item["pricing_status"] = "NOT_CONFIGURED"
        result.append(item)
    by_arm = {row["arm"]: row for row in result}
    for row in result:
        row["attribution"] = "main comparison incremental requests; shared preparation charged to its recorded owner"
        row["parent_arm"] = PARENTS.get(row["arm"])
        for name in ("request_count", "prompt_tokens", "completion_tokens", "latency_ms"):
            parent_value = by_arm[PARENTS[row["arm"]]][name] if row["arm"] in PARENTS else 0
            row[f"with_parent_{name}"] = (row[name] + parent_value
                if row[name] is not None and parent_value is not None else None)
    return result


def analyze_records(manifests: Mapping[str, Mapping[str, Any]], summaries: Mapping[str, Mapping[str, Any]],
                    answers: Sequence[Mapping[str, Any]], judgments: Sequence[Mapping[str, Any]], *,
                    phase: str = "test", extras: Mapping[str, Mapping[str, Any]] | None = None,
                    missing_artifacts: Sequence[str] = (),
                    development_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze saved records without reading a question asset or certifying fixtures."""
    if phase not in {"test", "dev", "live"}:
        raise ValueError("formal analysis phase must be test, dev or live")
    variant_judgments = [row for row in judgments if row.get("comparison", "main") != "main"]
    judgments = [row for row in judgments if row.get("comparison", "main") == "main"]
    extras = _attach_variant_scores(extras or {}, variant_judgments)
    identities = {generator: _identity(value) for generator, value in manifests.items()}
    qids, targets, metadata, exposure = set(), {}, {}, {}
    origins, errors, missing_contract = set(), [], []
    declared_sets, contracts = [], []
    for generator, identity in sorted(identities.items()):
        declared = identity.get("qids", identity.get("selected_qids", []))
        if (not isinstance(declared, list) or any(not isinstance(qid, str) or not qid for qid in declared)
                or len(declared) != len(set(declared))):
            raise ValueError("manifest qids must be unique nonempty strings")
        declared_sets.append(set(declared))
        qids.update(declared)
        origins.add(str(identity.get("origin", "UNKNOWN")))
        if (identity.get("protocol_version", identity.get("protocol")) != PROTOCOL
                or identity.get("method_version") != PROTOCOL
                or identity.get("generator") != generator
                or identity.get("phase", identity.get("split")) != phase
                or set(identity.get("arms", [])) != set(ARMS)):
            errors.append(f"invalid_identity:{generator}")
        summary = summaries.get(generator, {})
        if (summary.get("origin") != identity.get("origin") or
                summary.get("protocol", summary.get("protocol_version")) != PROTOCOL
                or summary.get("generator") != generator
                or summary.get("phase", summary.get("split")) != phase):
            errors.append(f"summary_identity:{generator}")
        missing_contract.extend(f"{generator}:{name}" for name in LOCKED_FIELDS if name not in identity)
        contracts.append({name: identity.get(name) for name in LOCKED_FIELDS})
        for qid, target in identity.get("qid_targets", {}).items():
            if not isinstance(target, str) or not target.strip():
                raise ValueError("target cluster IDs must be nonempty strings")
            if qid in targets and targets[qid] != target:
                raise ValueError("conflicting target cluster identities")
            targets[qid] = str(target)
        for source, destination in ((identity.get("qid_metadata", {}), metadata),
                                    (identity.get("selected_exposure", {}), exposure)):
            for qid, value in source.items():
                if qid in destination and destination[qid] != value:
                    raise ValueError("conflicting question metadata in run manifests")
                destination[qid] = value
    if declared_sets and any(items != declared_sets[0] for items in declared_sets):
        errors.append("different_generator_question_sets")
    if contracts and any(contract != contracts[0] for contract in contracts):
        errors.append("different_generator_contracts")
    qids = sorted(qids)
    supplement_answers = [row for row in answers if row.get("arm") in SUPPLEMENT_ARMS]
    supplement_judgments = [row for row in judgments if row.get("arm") in SUPPLEMENT_ARMS]
    answers = [row for row in answers if row.get("arm") not in SUPPLEMENT_ARMS]
    judgments = [row for row in judgments if row.get("arm") not in SUPPLEMENT_ARMS]
    answer_map, judgment_map = {}, {}
    for row in answers:
        key = _key(row)
        if key in answer_map or key[0] not in GENERATORS or key[1] not in BASES or key[2] not in ARMS or key[3] not in qids:
            raise ValueError("duplicate or out-of-scope answer position")
        answer_map[key] = row
        if row.get("origin", identities.get(key[0], {}).get("origin")) != "REAL":
            origins.add(str(row.get("origin", identities.get(key[0], {}).get("origin", "UNKNOWN"))))
    for row in judgments:
        key = (*_key(row), row.get("teacher"))
        if key in judgment_map or key[4] not in JUDGE_ROLES or key[:4] not in answer_map:
            raise ValueError("duplicate, orphan or out-of-scope judgment position")
        judgment_map[key] = row
        if row.get("origin", identities.get(key[0], {}).get("origin")) != "REAL":
            origins.add(str(row.get("origin", identities.get(key[0], {}).get("origin", "UNKNOWN"))))
    means, cross, chars, missing_answers, missing_judgments = {}, {}, {}, [], []
    cells = [(generator, base) for generator in GENERATORS for base in BASES]
    for generator, base in cells:
        for arm in ARMS:
            for qid in qids:
                key = (generator, base, arm, qid)
                answer = answer_map.get(key)
                if not answer or answer.get("status") != "ok" or not str(answer.get("answer", "")).strip():
                    missing_answers.append(list(key))
                else:
                    if answer.get("answer_sha256") != answer_sha256(answer["answer"]):
                        raise ValueError("answer hash does not bind its text")
                    chars[key] = len(answer["answer"])
                values = []
                for teacher in JUDGE_ROLES:
                    judgment = judgment_map.get((*key, teacher))
                    scores = _scores(judgment) if judgment else None
                    if scores is None:
                        missing_judgments.append([*key, teacher])
                        continue
                    if not answer or judgment.get("answer_sha256") != answer.get("answer_sha256"):
                        raise ValueError("judgment belongs to a different answer hash")
                    values.append(scores)
                    cross_teacher = "teacher2" if generator == GENERATORS[0] else "teacher1"
                    if teacher == cross_teacher and key in chars:
                        cross[key] = scores
                if key in chars and len(values) == 2:
                    means[key] = {dimension: fmean(score[dimension] for score in values) for dimension in DIMENSIONS}
    complete_qids = [qid for qid in qids if all((g, b, arm, qid) in means for g, b in cells for arm in ARMS)]
    quality = {(qid, arm): fmean(means[g, b, arm, qid]["overall_quality"] for g, b in cells)
               for qid in complete_qids for arm in ARMS}
    correctness = {(qid, arm): fmean(means[g, b, arm, qid]["answer_correctness"] for g, b in cells)
                   for qid in complete_qids for arm in ARMS}
    revisions = _revision_report(answer_map, means)
    extra_checks = _complete_extras(extras, origins)
    _extra_coverage(extras, extra_checks, answer_map, qids, metadata, phase)
    execution_complete = all(summaries.get(generator, {}).get("complete") is True
                             and not summaries.get(generator, {}).get("stop_reason") for generator in GENERATORS)
    integrity = {"expected_answer_positions": len(qids) * 32, "recorded_answer_positions": len(answer_map),
                 "expected_judgment_positions": len(qids) * 64, "recorded_judgment_positions": len(judgment_map),
                 "missing_answers": missing_answers, "missing_judgments": missing_judgments,
                 "missing_artifacts": sorted(missing_artifacts), "identity_errors": sorted(errors),
                 "missing_contract_fields": sorted(missing_contract), "parent_errors": revisions["parent_errors"],
                 "missing_target_qids": sorted(set(qids) - targets.keys()), "execution_complete": execution_complete,
                 "extra_artifacts": extra_checks}
    main_complete = bool(qids) and not (missing_answers or missing_judgments or errors or missing_contract
                                       or revisions["parent_errors"] or missing_artifacts) and execution_complete
    conditions = {"real_origin": origins == {"REAL"}, "full_test_scope": phase == "test" and len(qids) == 270,
                  "both_generators": set(identities) == set(GENERATORS), "complete_main_records": main_complete,
                  "at_least_two_questions": len(complete_qids) >= 2,
                  "accepted_question_version": all(identity.get("question_set_version") == "nea-questions-0022-v1"
                                                   for identity in identities.values()) and bool(identities)}
    conditions["not_sampled_or_exploratory"] = all(identity.get("sample_size") is None
        and not identity.get("smoke") and not identity.get("exploratory") for identity in identities.values())
    # Whole-experiment readiness includes the independent measurements, variants,
    # diagnosis experiment and cases, not just answer/judgment counts.
    required = ("measurements", "ablations", "validation_ablation", "audits", "perturbations", "cases")
    conditions["complete_registered_experiments"] = all(extra_checks.get(name, {}).get("complete") for name in required)
    conditions["complete_strata_metadata"] = all(isinstance(metadata.get(qid), Mapping) and
        metadata[qid].get("question_type") in {"direct", "prerequisite", "application"}
        and metadata[qid].get("graph_level") is not None and type(metadata[qid].get("has_gap")) is bool for qid in qids)
    cost_rows = _efficiency_rows(answer_map, means, summaries, extras)
    conditions["complete_efficiency_records"] = all(row["cost_status"] == "RECORDED" and
        all(row[field] is not None for field in ("request_count", "prompt_tokens", "completion_tokens", "latency_ms")) for row in cost_rows)
    method_identity = {key: next(iter(identities.values()), {}).get(key) for key in
                       ("method_version", "method_source_hash", "prompt_files", "schema_hashes", "budget", "comparisons_registry")}
    development = development_analysis or {}
    conditions["complete_development_sensitivity"] = (development.get("data_origin") == "REAL"
        and development.get("phase") in {"live", "dev"} and development.get("n_qids") == 30
        and development.get("method_identity") == method_identity
        and development.get("sensitivity", {}).get("status") == "ESTIMATED")
    # Spec 0044: live/E6 sensitivity and complete cost accounting are waived; their values stay in the record.
    eligible = all(value for name, value in conditions.items() if name not in WAIVED_CONDITIONS)
    real_main_ready = main_complete and origins == {"REAL"}
    inference = (statistical_summary(quality, correctness) if eligible else
                 _not_run("requires complete real formal8 test and all registered experiment records"))
    arm_rows = [{"arm": arm, "dimension": dimension, "n_qids": len(complete_qids),
                 "mean": _round(fmean(fmean(means[g, b, arm, qid][dimension] for g, b in cells)
                                     for qid in complete_qids)) if complete_qids else None,
                 "mean_chars": _round(fmean(chars[g, b, arm, qid] for qid in complete_qids for g, b in cells))
                               if complete_qids else None}
                for arm in ARMS for dimension in DIMENSIONS]
    main = {"status": "ESTIMATED" if eligible else "NOT_RUN", "rows": arm_rows, "inference": inference,
            "factorial": _factorial(quality, complete_qids, targets) if eligible else _not_run("formal inference unavailable"),
            "revision_harm": revisions["by_arm"]}
    strata = _strata(means, complete_qids, metadata, targets, eligible)
    metric_rows = _metric_rows(extras)
    length_rows = [{"qid": qid, "arm": arm, "y": quality[qid, arm],
                    "chars": fmean(chars[g, b, arm, qid] for g, b in cells)} for qid in complete_qids for arm in ARMS]
    report = {"protocol": PROTOCOL, "phase": phase, "status": "ESTIMATED" if eligible else "NOT_RUN",
        "data_origin": next(iter(origins)) if len(origins) == 1 else "MIXED_OR_MISSING",
        "n_qids": len(qids), "n_complete_qids": len(complete_qids), "qids": qids,
        "method_identity": method_identity,
        "formal_eligibility": {"eligible": eligible, "conditions": conditions, "waived": sorted(WAIVED_CONDITIONS)}, "integrity": integrity,
        "formal": inference, "main": main,
        "need_metrics": {"rows": metric_rows,
                                "agreement": extras.get("measurements", {}).get("agreement")},
        "ablations": {"rows": _variant_rows(extras.get("ablations", {}), means, targets) if eligible else [],
                      "validation_protection": _variant_rows(extras.get("validation_ablation", {}), means, targets,
                                                              require_four=False) if eligible else []},
        "perturbations": {"rows": _perturbation_rows(extras)},
        "defects": {"rows": _defect_rows(extras)},
        "sensitivity": {"rows": _variant_rows(extras.get("sensitivity", {}), means, targets)
                        if origins == {"REAL"} and execution_complete else []},
        "efficiency": {"rows": cost_rows,
                       "note": "Latency is summed request time, not elapsed run time; unknown costs stay null."},
        "strata": {"rows": strata}, "cases": _case_report(answer_map, revisions, extras),
        "length_sensitivity": _length_sensitivity(length_rows) if eligible else _not_run("formal inference unavailable"),
        "revision_records": revisions,
        "evaluation_evidence": {"invalid_localization_fields": sum((row.get("parsed") or {}).get("localization", {}).get("invalid_fields", 0)
                for row in judgments), "first_structure_success": {generator: {
                    key: summary.get("telemetry", {}).get(key) for key in
                    ("first_response_count", "first_structure_success_count", "first_structure_success_rate")}
                    for generator, summary in sorted(summaries.items())}},
        "development_preview": {"status": "DESCRIPTIVE" if real_main_ready and phase in {"live", "dev"} else "NOT_RUN",
            "correctness": {f"F-{parent}": _estimate({qid: correctness[qid, "F"] - correctness[qid, parent]
                    for qid in complete_qids}) for parent in ("B", "L", "SRL")}
                    if real_main_ready and phase in {"live", "dev"} else {},
            "interpretation": "Development-only directional results; no confirmatory hypothesis claim."},
        "robustness": _robustness(cross, quality, correctness, complete_qids, targets, exposure) if eligible else _not_run("formal inference unavailable"),
        "export_rule": "Only complete real results may populate paper data; exposure exclusion stays in the run analysis record."}
    for name in ("need_metrics", "ablations", "perturbations", "defects", "efficiency", "strata", "cases"):
        report[name]["status"] = "ESTIMATED" if eligible else "NOT_RUN"
    report["sensitivity"]["status"] = ("ESTIMATED" if phase in {"live", "dev"} and main_complete
        and len(qids) == 30 and conditions["not_sampled_or_exploratory"]
        and origins == {"REAL"} and extra_checks.get("sensitivity", {}).get("complete") else "NOT_RUN")
    if phase == "test" and conditions["complete_development_sensitivity"]:
        report["sensitivity"] = {**development["sensitivity"], "source_phase": development["phase"],
                                 "source_analysis_sha256": digest(development), "source_n_qids": 30}
    if supplement_answers or supplement_judgments:
        # Absent supplement rows leave every earlier report byte-identical.
        report["main"]["supplement"] = _supplement(supplement_answers, supplement_judgments, answer_map,
                                                   means, chars, qids, targets, eligible)
    return report


def _supplement(answers, judgments, main_answers, main_means, main_chars, qids, targets, eligible):
    """Spec 0045 supplementary arm(s): same averaging and bootstrap as the main table, separate family."""
    cells = [(generator, base) for generator in GENERATORS for base in BASES]
    answer_map, judgment_map, parent_errors = {}, {}, []
    for row in answers:
        key = _key(row)
        if key in answer_map or key[0] not in GENERATORS or key[1] not in BASES or key[3] not in qids:
            raise ValueError("duplicate or out-of-scope supplementary answer position")
        if row.get("status") == "ok" and row.get("answer_sha256") != answer_sha256(row["answer"]):
            raise ValueError("supplementary answer hash does not bind its text")
        parent = main_answers.get((key[0], key[1], SUPPLEMENT_PARENTS[key[2]], key[3]))
        if (row.get("parent_arm") != SUPPLEMENT_PARENTS[key[2]] or not parent
                or row.get("parent_sha256") != parent.get("answer_sha256")):
            parent_errors.append(list(key))
        answer_map[key] = row
    for row in judgments:
        key = (*_key(row), row.get("teacher"))
        if key in judgment_map or key[4] not in JUDGE_ROLES or key[:4] not in answer_map:
            raise ValueError("duplicate, orphan or out-of-scope supplementary judgment position")
        if row.get("answer_sha256") != answer_map[key[:4]].get("answer_sha256"):
            raise ValueError("supplementary judgment belongs to a different answer hash")
        judgment_map[key] = row
    means, chars, missing = {}, {}, []
    for arm in SUPPLEMENT_ARMS:
        for generator, base in cells:
            for qid in qids:
                key = (generator, base, arm, qid)
                answer = answer_map.get(key)
                scores = [_scores(judgment_map[(*key, teacher)]) if (*key, teacher) in judgment_map else None
                          for teacher in JUDGE_ROLES]
                if not answer or answer.get("status") != "ok" or any(score is None for score in scores):
                    missing.append(list(key))
                    continue
                chars[key] = len(answer["answer"])
                means[key] = {dimension: fmean(score[dimension] for score in scores) for dimension in DIMENSIONS}
    expected = len(SUPPLEMENT_ARMS) * len(cells) * len(qids)
    record = {"arms": list(SUPPLEMENT_ARMS), "parents": dict(SUPPLEMENT_PARENTS),
              "answer_positions": len(answer_map), "judgment_positions": len(judgment_map),
              "expected_answer_positions": expected, "expected_judgment_positions": expected * len(JUDGE_ROLES),
              "missing_positions": missing, "parent_errors": parent_errors,
              "family": "supplementary comparisons; outside the fixed sequence and the Holm family"}
    if missing or parent_errors or not eligible:
        return {**_not_run("supplementary records incomplete or main analysis not eligible"), **record}
    all_means, all_chars = {**main_means, **means}, {**main_chars, **chars}

    def per_qid(arm, dimension):
        return {qid: fmean(all_means[g, b, arm, qid][dimension] for g, b in cells) for qid in qids}

    rows = [{"arm": arm, "dimension": dimension, "n_qids": len(qids),
             "mean": _round(fmean(per_qid(arm, dimension).values())),
             "mean_chars": _round(fmean(all_chars[g, b, arm, qid] for qid in qids for g, b in cells))}
            for arm in SUPPLEMENT_ARMS for dimension in DIMENSIONS]
    comparisons = {}
    for label, left, right in SUPPLEMENT_COMPARISONS:
        comparisons[label] = {}
        for dimension in ("overall_quality", "answer_correctness"):
            high, low = per_qid(left, dimension), per_qid(right, dimension)
            comparisons[label][dimension] = _estimate({qid: high[qid] - low[qid] for qid in qids})
    revision, requests = {}, {}
    for arm in SUPPLEMENT_ARMS:
        selected = [row for key, row in sorted(answer_map.items()) if key[2] == arm]
        revised = sum(row["answer_sha256"] != row["parent_sha256"] for row in selected)
        calls = Counter()
        for row in selected:
            calls.update((row.get("revision") or {}).get("calls") or {})
        revision[arm] = {"positions": len(selected), "revised": revised, "revised_rate": _round(revised / len(selected))}
        requests[arm] = {"by_stage": dict(sorted(calls.items())),
                         "per_answer": _round(sum(calls.values()) / len(selected))}
    return {"status": "ESTIMATED", **record, "rows": rows, "comparisons": comparisons,
            "revision": revision, "requests": requests}


def _strata(means, qids, metadata, targets, infer):
    cells = [(g, b) for g in GENERATORS for b in BASES]
    groups = [("generator", generator, qids, [(generator, base) for base in BASES]) for generator in GENERATORS]
    groups += [("host", base, qids, [(generator, base) for generator in GENERATORS]) for base in BASES]
    for field in ("question_type", "graph_level", "has_gap"):
        values = sorted({str(metadata.get(qid, {}).get(field, "UNKNOWN")) for qid in qids})
        groups += [(field, value, [qid for qid in qids if str(metadata.get(qid, {}).get(field, "UNKNOWN")) == value], cells)
                   for value in values]
    rows = []
    for dimension, value, selected, selected_cells in groups:
        for label, left, right in PRIMARY:
            delta = {qid: fmean(means[g, b, left, qid]["overall_quality"] - means[g, b, right, qid]["overall_quality"]
                               for g, b in selected_cells) for qid in selected}
            rows.append({"stratum": dimension, "value": value, "comparison": label,
                         **(_estimate(delta) if infer and value != "UNKNOWN" else
                            {"status": "DESCRIPTIVE", "n_qids": len(selected),
                             "effect": _round(fmean(delta.values())) if delta else None})})
    return rows


def _robustness(cross, quality, correctness, qids, targets, exposure):
    cells = [(g, b) for g in GENERATORS for b in BASES]
    cross_quality = {(qid, arm): fmean(cross[g, b, arm, qid]["overall_quality"] for g, b in cells)
               for qid in qids for arm in ARMS}
    cross_correctness = {(qid, arm): fmean(cross[g, b, arm, qid]["answer_correctness"] for g, b in cells)
                   for qid in qids for arm in ARMS}
    # Metadata must explicitly identify exposure; unknown is disclosed, never
    # guessed from qid prefixes or treated as evidence of being unseen.
    def exposed(value):
        return isinstance(value, Mapping) and (value.get("used_for_development") is True
            or value.get("development_exposed") is True or value.get("priority_known_development") is True
            or str(value.get("development_feedback", "")).startswith("known_") or value.get("status") in
            {"development", "development_exposed", "exposed", "used_in_development"})
    excluded = sorted(qid for qid in qids if exposed(exposure.get(qid)))
    remaining = set(qids) - set(excluded)
    return {"cross_family_only": statistical_summary(cross_quality, cross_correctness),
            "exposure_exclusion": {"excluded_qids": excluded, "known_metadata_qids": len(set(qids) & exposure.keys()),
                "status": "ESTIMATED" if len(exposure) >= len(qids) else "INCOMPLETE_METADATA",
                "analysis": statistical_summary(
                    {key: value for key, value in quality.items() if key[0] in remaining},
                    {key: value for key, value in correctness.items() if key[0] in remaining}),
                "run_record_only": True}}


def _case_report(answers, revisions, extras):
    case_rows = extras.get("cases", {}).get("rows", [])
    indexed = {_key(row): row for row in case_rows}
    enriched = []
    for row in case_rows:
        generator, base, arm, qid = _key(row)
        revision = row.get("revision") or {}
        parent = indexed.get((generator, base, PARENTS.get(arm), qid), {})
        enriched.append({**row, "parent_answer": parent.get("answer"),
                         "diagnosis": row.get("diagnosis", revision.get("diagnosis")),
                         "actions": revision.get("actions"), "final_answer": row.get("answer")})
    fixed = [{"qid": qid, "status": "RECORDED" if any(row.get("qid") == qid for row in case_rows) else "NOT_RUN",
              "records": [row for row in enriched if row.get("qid") == qid]} for qid in ("Q410", "Q240", "Q213")]
    failures = defaultdict(list, {kind: [] for kind in ("rule_rejection", "validation_rejection",
                                                       "validation_uncertain", "quality_decline")})
    for row in revisions["rows"]:
        state = str(row.get("state", ""))
        reason = str(row.get("reason") or state)
        if row.get("harmed"):
            failures["quality_decline"].append(row)
        if "uncertain" in reason:
            failures["validation_uncertain"].append(row)
        elif reason == "semantic_rejected" or "valid" in reason and ("reject" in reason or "fail" in reason):
            failures["validation_rejection"].append(row)
        elif any(token in reason for token in ("rule", "growth", "length", "citation", "constraint")) or (
                reason == "no_executable_action" and (answers.get(_key(row), {}).get("revision") or {}).get("rejections")):
            failures["rule_rejection"].append(row)
    patterns = [{"pattern": pattern, "count": len(rows), "examples": sorted(rows,
                key=lambda row: (row["qid"], row["generator"], row["base"], row["arm"]))[:3]}
                for pattern, rows in sorted(failures.items())]
    for pattern in patterns:
        pattern["examples"] = [{**example,
            "parent_answer": answers.get((example["generator"], example["base"], example["parent_arm"], example["qid"]), {}).get("answer"),
            "final_answer": answers.get((example["generator"], example["base"], example["arm"], example["qid"]), {}).get("answer"),
            "revision": answers.get((example["generator"], example["base"], example["arm"], example["qid"]), {}).get("revision")}
            for example in pattern["examples"]]
    return {"rows": fixed, "error_patterns": patterns,
            "interpretation": "Deterministically selected observed cases; no generated explanation of unobserved errors."}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(document: Any) -> list[dict[str, Any]]:
    rows = document.get("rows", []) if isinstance(document, Mapping) else document
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("artifact rows must be objects")
    return rows


def analyze_formal_run(run_root: Path, phase: str = "test") -> dict[str, Any]:
    """Read only operator-v2 files; absent phases produce a NOT_RUN report."""
    root, manifests, summaries, answers, judgments = Path(run_root), {}, {}, [], []
    missing, extra_documents = [], defaultdict(list)
    extra_names = {"measurements": "measurements", "audits": "audits",
        "perturbations": "perturbations", "ablations": "ablations", "sensitivity": "sensitivity",
        "validation_ablation": "validation-ablation", "costs": "costs", "cases": "cases"}
    for generator in GENERATORS:
        directory = root / "operator-v2" / generator.replace("/", "__") / phase
        for name, destination in (("run_manifest", manifests), ("phase_summary", summaries)):
            path = directory / f"{name}.json"
            if path.is_file():
                destination[generator] = _read_json(path)
            else:
                missing.append(f"{generator}/{phase}/{name}.json")
        for name, destination in (("answers", answers), ("judgments", judgments)):
            path = directory / f"{name}.json"
            if path.is_file():
                for row in _rows(_read_json(path)):
                    destination.append({"generator": generator, **row})
            else:
                missing.append(f"{generator}/{phase}/{name}.json")
        for name, filename in extra_names.items():
            path = directory / f"{filename}.json"
            if path.is_file():
                document = _read_json(path)
                if isinstance(document, Mapping):
                    if name in {"ablations", "sensitivity", "validation_ablation"}:
                        # Generation artifacts use the immutable phase manifest
                        # as their envelope; score binding is checked separately.
                        identity = _identity(manifests.get(generator, {}))
                        summary = summaries.get(generator, {})
                        document = {"protocol": identity.get("protocol_version"),
                            "origin": identity.get("origin"),
                            "status": "COMPLETE" if summary.get("generation_complete") is True else "NOT_RUN",
                            **document}
                    extra_documents[name].append(document)
    extras = {}
    for name, documents in extra_documents.items():
        extras[name] = {"protocol": PROTOCOL if all(item.get("protocol", item.get("protocol_version")) == PROTOCOL
                                                      for item in documents) else "UNKNOWN",
            "origin": "REAL" if all(item.get("origin") == "REAL" for item in documents) else "MIXED_OR_MISSING",
            "status": "COMPLETE" if len(documents) == len(GENERATORS)
                and all(item.get("status") == "COMPLETE" for item in documents) else "NOT_RUN",
            "rows": [row for document in documents for row in _rows(document)]}
        if name == "measurements":
            extras[name]["agreement"] = [document.get("agreement") for document in documents]
    development = analyze_formal_run(root, phase="live") if phase == "test" else None
    return analyze_records(manifests, summaries, answers, judgments, phase=phase, extras=extras,
                           missing_artifacts=missing, development_analysis=development)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = sorted({key for row in rows for key in row}) or ["status", "reason"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                         if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    return output.getvalue().encode("utf-8")


def _write_bytes_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ConflictError(f"Existing paper artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def export_paper_tables(report: Mapping[str, Any], run_root: Path) -> Path:
    """Publish immutable content-addressed table/figure JSON and CSV pairs.

    New evidence creates a new analysis snapshot inside the same experiment.
    Re-exporting identical input leaves every byte unchanged.  Fixture and
    incomplete reports export NOT_RUN without any synthetic result numbers.
    """
    snapshot = digest(report)
    directory = Path(run_root) / "paper-tables" / str(report.get("phase", "test")) / snapshot
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (title, section) in PAPER_ITEMS.items():
        source = report.get(section, {})
        ready = report.get("data_origin") == "REAL" and source.get("status") == "ESTIMATED"
        payload = {"paper_item": name, "title": title, "protocol": PROTOCOL,
                   "analysis_sha256": snapshot, "data_origin": report.get("data_origin"),
                   "status": "ESTIMATED" if ready else "NOT_RUN",
                   "data": source if ready else _not_run("complete real results not available")}
        write_once(directory / f"{name}.json", payload)
        rows = source.get("rows", []) if ready else [{"status": "NOT_RUN", "reason": "complete real results not available"}]
        # Auxiliary columns (inference and error patterns) are
        # represented as explicit records, so CSV and JSON cover the same data.
        if ready:
            rows = [{"record_type": "row", **row} for row in rows]
            rows += [{"record_type": key, "value": value} for key, value in sorted(source.items()) if key not in {"rows", "status"}]
        _write_bytes_once(directory / f"{name}.csv", _csv_bytes(rows))
        for extension in ("json", "csv"):
            path = directory / f"{name}.{extension}"
            files[path.name] = digest(path.read_bytes())
    write_once(directory / "manifest.json", {"protocol": PROTOCOL, "analysis_sha256": snapshot,
        "phase": report.get("phase"), "status": report.get("status"), "files": files,
        "expected_json_csv_pairs": 11, "exposure_exclusion_exported": False})
    # Robustness exclusions intentionally stay in this run record, not tables.
    write_once(Path(run_root) / "operator-v2" / "analysis" / snapshot / "analysis.json", dict(report))
    return directory
