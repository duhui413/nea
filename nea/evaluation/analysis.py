"""Pre-registered need-aware test analysis (spec 0054 section 4, implemented in 0055).

Unit: qid.  Y(qid, arm) = mean over the four generator x base cells of the
two judges' mean overall score.  Primary: fixed sequence H1 F-B then H2 F-P,
question bootstrap (20,000 replicates, seed 20260927, linear quantile
p*(N-1)); H2 is ``NOT_TESTED`` unless H1's 95% lower bound is > 0.
Secondary L-P, F-L, F-SR use Holm on two-sided bootstrap p-values.  Length
sensitivity: question fixed-effects OLS ``Y ~ arm + log(chars)``.
Non-default replicates/seed can only yield ``EXPLORATORY`` (G6c).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nea.evaluation.statistics import (
    BOOTSTRAP_REPLICATES,
    SEED,
    formal_parameters,
    linear_quantile,
)

from nea.question_set.contract import QuestionSet
from nea.pipeline.registry import ARMS_TEST, BASES, GENERATORS, PROTOCOL
from nea.pipeline.sampling import check_mode
from nea.pipeline.workflow import load_freeze, load_phase, overall_by_answer


PRIMARY = (("H1", "F", "B"), ("H2", "F", "P"))
SECONDARY = (("L-P", "L", "P"), ("F-L", "F", "L"), ("F-SR", "F", "SR"))
ALPHA = 0.05
REFERENCE_ARM = "B"


def _question_draw_indices(n_questions: int, replicates: int, seed: int) -> np.ndarray:
    if type(n_questions) is not int or n_questions < 1 or type(replicates) is not int or replicates < 1:
        raise ValueError("positive question and replicate counts required")
    return np.random.default_rng(seed).integers(0, n_questions, size=(replicates, n_questions))


def question_draw_counts(n_questions: int, replicates: int, seed: int) -> np.ndarray:
    """Resample N whole questions with replacement in each replicate."""
    indices = _question_draw_indices(n_questions, replicates, seed)
    counts = np.zeros((replicates, n_questions), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(replicates), n_questions), indices.reshape(-1)), 1.0)
    return counts


def bootstrap_mean(differences: Mapping[str, float], *,
                   replicates: int, seed: int) -> dict[str, Any]:
    values = np.array([differences[qid] for qid in sorted(differences)], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two questions with finite paired differences required")
    indices = _question_draw_indices(len(values), replicates, seed)
    draws = np.sort(values[indices].mean(axis=1))
    effect = float(values.mean())
    below, above = float(np.mean(draws <= 0)), float(np.mean(draws >= 0))
    return {"effect": round(effect, 6), "n_qids": len(values), "unit": "qid",
            "ci95": [round(linear_quantile(draws, 0.025), 6), round(linear_quantile(draws, 0.975), 6)],
            "p_two_sided": round(min(1.0, 2 * min(below, above)), 6)}


def holm(p_values: Mapping[str, float], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, dict[str, Any]] = {}
    still = True
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (m - rank)
        rejected = still and p <= threshold
        still = still and rejected
        result[name] = {"p": p, "holm_threshold": round(threshold, 6), "rejected": rejected}
    return result


def fixed_effects_ols(rows: Sequence[Mapping[str, Any]], arms: Sequence[str],
                      reference: str = REFERENCE_ARM) -> dict[str, Any]:
    """Within-qid OLS of Y on arm dummies (vs reference) and log(chars)."""
    others = [arm for arm in arms if arm != reference]
    names = [f"arm:{arm}" for arm in others] + ["log_chars"]
    by_qid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    blocks = {}
    for qid, items in sorted(by_qid.items()):
        x = np.array([[1.0 if row["arm"] == arm else 0.0 for arm in others] + [np.log(row["chars"])]
                      for row in items], dtype=np.float64)
        y = np.array([row["y"] for row in items], dtype=np.float64)
        blocks[qid] = (x - x.mean(axis=0), y - y.mean())
    xtx = sum(x.T @ x for x, _ in blocks.values())
    xty = sum(x.T @ y for x, y in blocks.values())
    if np.linalg.matrix_rank(xtx) < len(names):
        # Length is collinear with arm: the adjustment is not identified.
        return {"names": names, "beta": None, "blocks": blocks}
    beta = np.linalg.solve(xtx, xty)
    return {"names": names, "beta": beta, "blocks": blocks}


def _adjusted(beta: np.ndarray, names: Sequence[str], left: str, right: str) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}

    def coef(arm: str) -> np.ndarray:
        return beta[..., index[f"arm:{arm}"]] if arm != REFERENCE_ARM else np.zeros(beta.shape[:-1])
    return coef(left) - coef(right)


def length_sensitivity(rows: Sequence[Mapping[str, Any]], *,
                       replicates: int, seed: int) -> dict[str, Any]:
    if len({row["qid"] for row in rows}) < 2:
        return {"model": "Y ~ question FE + arm + log(chars)", "status": "NOT_IDENTIFIED",
                "reason": "at least two complete questions required"}
    fit = fixed_effects_ols(rows, ARMS_TEST)
    names, blocks = fit["names"], fit["blocks"]
    if fit["beta"] is None:
        return {"model": "Y ~ question FE + arm + log(chars)", "status": "NOT_IDENTIFIED",
                "reason": "log(chars) is collinear with arm after question fixed effects"}
    qids = sorted(blocks)
    dim = len(names)
    question_xtx = np.array([blocks[qid][0].T @ blocks[qid][0] for qid in qids])
    question_xty = np.array([blocks[qid][0].T @ blocks[qid][1] for qid in qids])
    multiplicity = question_draw_counts(len(qids), replicates, seed)
    xtx = np.einsum("rc,cij->rij", multiplicity, question_xtx)
    xty = np.einsum("rc,ci->ri", multiplicity, question_xty)
    solvable = np.linalg.matrix_rank(xtx) == dim
    if not solvable.any():
        return {"model": "Y ~ question FE + arm + log(chars)", "status": "NOT_IDENTIFIED",
                "reason": "every bootstrap replicate is rank deficient"}
    betas = np.linalg.solve(xtx[solvable], xty[solvable][..., None])[..., 0]
    report = {"model": "Y ~ question FE + arm + log(chars)", "status": "ESTIMATED",
              "reference_arm": REFERENCE_ARM,
              "coefficients": {name: round(float(value), 6) for name, value in zip(names, fit["beta"])},
              "singular_replicates": int((~solvable).sum()), "adjusted": {}}
    for label, left, right in (("F-B", "F", "B"), ("F-P", "F", "P"), *[(n, a, b) for n, a, b in SECONDARY]):
        point = float(_adjusted(fit["beta"], names, left, right))
        draws = np.sort(_adjusted(betas, names, left, right))
        report["adjusted"][label] = {"effect": round(point, 6), "ci95": [
            round(linear_quantile(draws, 0.025), 6), round(linear_quantile(draws, 0.975), 6)]}
    return report


def analyze_scores(scores: Mapping[tuple[str, str, str, str], float],
                   chars: Mapping[tuple[str, str, str, str], int],
                   qid_targets: Mapping[str, str], *, generators: Sequence[str] = GENERATORS,
                   replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> dict[str, Any]:
    cells = [(g, b) for g in generators for b in BASES]
    complete, missing = [], []
    for qid in sorted(qid_targets):
        absent = [[g, b, arm] for arm in ARMS_TEST for g, b in cells if (g, b, arm, qid) not in scores]
        (missing if absent else complete).append((qid, absent))
    y = {(qid, arm): float(np.mean([scores[(g, b, arm, qid)] for g, b in cells]))
         for qid, _ in complete for arm in ARMS_TEST}
    length = {(qid, arm): float(np.mean([chars[(g, b, arm, qid)] for g, b in cells]))
              for qid, _ in complete for arm in ARMS_TEST}
    qids = [qid for qid, _ in complete]

    def diffs(left: str, right: str) -> dict[str, float]:
        return {qid: y[(qid, left)] - y[(qid, right)] for qid in qids}

    report: dict[str, Any] = {"protocol": PROTOCOL, "data_origin": "REAL", "n_qids": len(qid_targets),
                              "n_complete_qids": len(qids),
                              "n_targets": len({qid_targets[q] for q in qids}),
                              "missing": [[qid, absent] for qid, absent in missing],
                              "bootstrap": {"replicates": replicates, "seed": seed,
                                            "unit": "qid", "quantile": "linear p*(N-1)",
                                            "formal": formal_parameters(replicates, seed)}}
    primary: dict[str, Any] = {}
    if len(qids) >= 2:
        gate_open = True
        for name, left, right in PRIMARY:
            if not gate_open:
                primary[name] = {"contrast": f"{left}-{right}", "status": "NOT_TESTED"}
                continue
            result = bootstrap_mean(diffs(left, right), replicates=replicates, seed=seed)
            passed = result["ci95"][0] > 0
            primary[name] = {"contrast": f"{left}-{right}", **result,
                             "status": "SUPPORTED" if passed else "NOT_SUPPORTED"}
            gate_open = passed
        secondary = {name: bootstrap_mean(diffs(left, right), replicates=replicates, seed=seed)
                     for name, left, right in SECONDARY}
        adjusted = holm({name: row["p_two_sided"] for name, row in secondary.items()})
        for name in secondary:
            secondary[name]["holm"] = adjusted[name]
        rows = [{"qid": qid, "arm": arm, "y": y[(qid, arm)], "chars": length[(qid, arm)]}
                for qid in qids for arm in ARMS_TEST]
        report.update({"primary": {"procedure": "fixed_sequence", "alpha": ALPHA, **primary},
                       "secondary": {"procedure": "holm", **secondary},
                       "length_sensitivity": length_sensitivity(rows,
                                                                replicates=replicates, seed=seed)})
    if not formal_parameters(replicates, seed):
        status = "EXPLORATORY"
    elif missing or len(qids) < 2:
        status = "NOT_READY_MISSING_SCORES"
    else:
        status = "READY_FOR_CONFIRMATORY_ANALYSIS"
    report["status"] = status
    report["claim_rule"] = ("H2 is interpreted only if H1's 95% lower bound > 0; if the length-adjusted "
                            "interval crosses 0 the gain cannot be separated from length.")
    return report


def analyze_run(run_root: Path, question_set: QuestionSet, *,
                replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> dict[str, Any]:
    frozen = load_freeze(run_root)
    if (frozen["freeze"].get("smoke") or frozen["freeze"].get("exploratory")
            or frozen["freeze"].get("sample_size") is not None
            or (run_root / "sample_scope.json").exists()):
        raise ValueError("Smoke and exploratory runs cannot produce formal statistical analysis")
    phases = {generator: load_phase(run_root, generator, "test") for generator in GENERATORS}
    for generator, phase in phases.items():
        # M1: every test phase must come from this freeze and this question set.
        for source in (phase["summary"], phase["manifest"]):
            check_mode(source)
            if source.get("question_set_sha256") != question_set.sha256:
                raise ValueError(f"test outputs of {generator} used a different question set than the freeze")
    scores: dict[tuple[str, str, str, str], float] = {}
    chars: dict[tuple[str, str, str, str], int] = {}
    for generator, phase in phases.items():
        scores.update(overall_by_answer(phase["judgments"]))
        for row in phase["answers"]:
            if row["status"] == "ok":
                chars[(generator, row["base"], row["arm"], row["qid"])] = row["chars"]
    qid_targets = {row["qid"]: row["authoring_target_id"] for row in question_set.split_rows("test")}
    report = analyze_scores(scores, chars, qid_targets, replicates=replicates, seed=seed)
    report["question_set_sha256"] = question_set.sha256
    report["freeze_hash"] = frozen["freeze_hash"]
    return report
