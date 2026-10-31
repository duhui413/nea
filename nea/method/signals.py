"""Method-internal signals CER_G, DR_G, RCC, LTS with injectable components.

These signals steer revision only.  Their denominators come from the method's
own G_q, so they are never paper measurement fields (anti-circularity).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from nea.metrics import combine_lts, segment_explanation_units


SLOT_ORDER = ("obj", "pre", "bridge", "target", "example")
TRANSITION_MARKERS = ("因此", "所以", "由于", "从而", "于是", "这意味着", "进而", "基于此", "在此基础上")
QUANTILES = {"cer_g": 0.25, "rcc": 0.25, "dr_g": 0.75, "lts": 0.50}
DIRECTION = {"cer_g": "below", "rcc": "below", "dr_g": "above", "lts": "above"}


def _bigrams(text: str) -> set[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def default_pair_evidence(left: str, right: str) -> tuple[float, bool]:
    """Character-bigram Jaccard as semantic support; explicit connective as transition."""
    a, b = _bigrams(left), _bigrams(right)
    similarity = len(a & b) / len(a | b) if a | b else 0.0
    return round(similarity, 6), any(marker in right for marker in TRANSITION_MARKERS)


@dataclass(frozen=True)
class SignalComponents:
    """Unit segmentation, concept linking and RCC/LTS pair evidence (injectable)."""

    segment: Callable[[str], list[str]]
    link: Callable[[str], list[str]]
    pair_evidence: Callable[[str, str], tuple[float, bool]] = default_pair_evidence

    @classmethod
    def default(cls, link: Callable[[str], list[str]]) -> "SignalComponents":
        return cls(segment=lambda answer: segment_explanation_units(answer), link=link)


def compute_signals(answer: str, *, gap_ids: Sequence[str], given_ids: Sequence[str],
                    assumed_ids: Sequence[str], target_ids: Sequence[str],
                    question_ids: Sequence[str], slot_of: Mapping[str, str],
                    levels: Mapping[str, int | None],
                    components: SignalComponents) -> dict[str, Any]:
    units = [unit for unit in components.segment(answer) if unit.strip()]
    linked = [components.link(unit) for unit in units]
    main = [ids[0] if ids else None for ids in linked]
    gaps = set(gap_ids)
    covered = sorted({node for ids in linked for node in ids} & gaps)
    cer = None if not gaps else round(len(covered) / len(gaps), 6)
    allowed = gaps | set(given_ids) | set(assumed_ids) | set(target_ids) | set(question_ids)
    total_chars = sum(len(unit) for unit in units)
    detour_units = [index for index, node in enumerate(main) if node is not None and node not in allowed]
    detour_chars = sum(len(units[index]) for index in detour_units)
    dr = round(detour_chars / total_chars, 6) if total_chars else None
    pairs = []
    for index in range(len(units) - 1):
        similarity, transition = components.pair_evidence(units[index], units[index + 1])
        left_slot = slot_of.get(main[index]) if main[index] else None
        right_slot = slot_of.get(main[index + 1]) if main[index + 1] else None
        forward = (left_slot is not None and right_slot is not None
                   and 0 <= SLOT_ORDER.index(right_slot) - SLOT_ORDER.index(left_slot) <= 1)
        left_level = levels.get(main[index]) if main[index] else None
        right_level = levels.get(main[index + 1]) if main[index + 1] else None
        penalty = None
        if left_level is not None and right_level is not None:
            delta = right_level - left_level
            penalty = float(min(1, max(0, delta - 1) + max(0, -delta - 1)))
        pairs.append({"left": index, "right": index + 1, "semantic_support": similarity,
                      "transition": transition, "slot_forward": forward,
                      "rcc_pair": round(0.5 * similarity + 0.5 * float(forward and transition), 6),
                      "hierarchy_penalty": penalty})
    rcc = round(fmean(row["rcc_pair"] for row in pairs), 6) if pairs else None
    if pairs:
        semantic = [row["semantic_support"] for row in pairs]
        mean = fmean(semantic)
        variance = fmean((value - mean) ** 2 for value in semantic)
        discontinuity = fmean(1.0 - value for value in semantic)
        known = [row["hierarchy_penalty"] for row in pairs if row["hierarchy_penalty"] is not None]
        hierarchy = fmean(known) if known else 0.0
        lts = combine_lts(variance, discontinuity, hierarchy)
    else:
        lts = None
    weak = [row["left"] for row in pairs if not (row["slot_forward"] and row["transition"])]
    return {"cer_g": cer, "dr_g": dr, "rcc": rcc, "lts": lts,
            "n_units": len(units), "chars": len(answer),
            "uncovered_gaps": sorted(gaps - set(covered)), "detour_units": detour_units,
            "weak_transitions": weak[:3], "unit_main_concepts": main, "units": units}


def linear_quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no applicable values")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fit_thresholds(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """CER_G q25, RCC q25, DR_G q75, LTS q50 over L initial answers of one cell."""
    materialised = list(rows)
    thresholds: dict[str, float | None] = {}
    for name, fraction in QUANTILES.items():
        values = [float(row[name]) for row in materialised if row.get(name) is not None]
        thresholds[name] = round(linear_quantile(values, fraction), 6) if values else None
    return thresholds


def triggers(signals: Mapping[str, Any], thresholds: Mapping[str, float | None],
             mask: Iterable[str]) -> dict[str, bool]:
    fired: dict[str, bool] = {}
    for name in sorted(mask):
        value, threshold = signals.get(name), thresholds.get(name)
        if value is None or threshold is None:
            fired[name] = False
        elif DIRECTION[name] == "below":
            fired[name] = float(value) < float(threshold)
        else:
            fired[name] = float(value) > float(threshold)
    return fired
