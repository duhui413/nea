"""Descriptive paired comparisons for isolated exploratory dev/test samples."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from nea.evaluation.judging import RUBRIC_ANCHORS
from nea.pipeline.registry import BASES, GENERATORS, JUDGE_ROLES
from nea.pipeline.sampling import check_mode, validate_sample_scope
from nea.pipeline.workflow import load_freeze, load_phase, read_json
from nea.runtime.records import digest


def _statistics(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {"n": len(rows), "means": {arm: round(fmean(row[arm] for row in rows), 6)
                                      for arm in ("B", "P", "F")},
            "F-B": round(fmean(row["F"] - row["B"] for row in rows), 6),
            "F-P": round(fmean(row["F"] - row["P"] for row in rows), 6)}


def compare_phases(phases: Mapping[tuple[str, str], Mapping[str, Any]], scope: Mapping[str, Any],
                   freeze_hash: str) -> dict[str, Any]:
    """Pair by qid within every generator/base cell; never infer significance."""
    size = validate_sample_scope(scope)
    expected_phases = {(generator, split) for generator in GENERATORS for split in ("dev", "test")}
    if set(phases) != expected_phases:
        raise ValueError("Comparison requires both generators and both splits")
    cells, pooled = [], []
    for split in ("dev", "test"):
        all_pairs = []
        qids = scope["qids"][split]
        for generator in GENERATORS:
            grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
            for row in phases[(generator, split)]["judgments"]:
                if row["generator"] != generator or row["qid"] not in qids or row["base"] not in BASES:
                    raise ValueError("Judgment lies outside the exploratory scope")
                teacher, scores = row["teacher"], row["scores"]
                if (teacher not in JUDGE_ROLES or not isinstance(scores, dict) or set(scores) != set(RUBRIC_ANCHORS)
                        or any(type(value) is not int or not 1 <= value <= 5 for value in scores.values())):
                    raise ValueError("Exploratory comparison requires complete six-dimension judgments")
                if row["arm"] not in {"B", "P", "F"}:
                    continue
                key = (row["qid"], row["base"], row["arm"])
                if teacher in grouped[key]:
                    raise ValueError("Duplicate teacher judgment")
                grouped[key][teacher] = scores["overall_quality"]
            for base in BASES:
                pairs = []
                for qid in qids:
                    pair = {}
                    for arm in ("B", "P", "F"):
                        values = grouped[(qid, base, arm)]
                        if set(values) != set(JUDGE_ROLES):
                            raise ValueError("Comparison is missing a paired B/P/F score")
                        pair[arm] = fmean(values[teacher] for teacher in JUDGE_ROLES)
                    pairs.append(pair)
                cells.append({"split": split, "generator": generator, "base": base, **_statistics(pairs)})
                all_pairs.extend(pairs)
        pooled.append({"split": split, **_statistics(all_pairs)})
    return {"status": "EXPLORATORY", "mode": "exploratory", "sample_size": size,
            "question_set_sha256": scope["question_set_sha256"], "freeze_hash": freeze_hash,
            "cells": cells, "pooled": pooled,
            "interpretation": "Descriptive paired point estimates only; no confirmatory inference or test-driven tuning."}


def compare_run(run_root: Path) -> dict[str, Any]:
    scope = read_json(run_root / "sample_scope.json")
    size = validate_sample_scope(scope)
    frozen = load_freeze(run_root)
    body = frozen["freeze"]
    if digest(body) != frozen["freeze_hash"]:
        raise ValueError("Freeze hash differs")
    check_mode(body, sample_size=size)
    if body["question_set_sha256"] != scope["question_set_sha256"]:
        raise ValueError("Freeze and sample scope differ")
    phases = {}
    for generator in GENERATORS:
        for split in ("dev", "test"):
            phase = load_phase(run_root, generator, split)
            for source in (phase["manifest"], phase["summary"]):
                check_mode(source, sample_size=size)
                if source["question_set_sha256"] != scope["question_set_sha256"]:
                    raise ValueError("Phase and sample scope differ")
                if split == "test" and source["calibration_hash"] != frozen["freeze_hash"]:
                    raise ValueError("Test phases do not share the exploratory freeze")
            if phase["summary"].get("missing") or phase["summary"].get("judgments_missing"):
                raise ValueError("Exploratory phase is incomplete")
            phases[(generator, split)] = phase
    return compare_phases(phases, scope, frozen["freeze_hash"])
