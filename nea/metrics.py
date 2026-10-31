"""Explanation segmentation and LTS aggregation."""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Mapping
from nea.data import KnowledgeBase
from nea.cognitive import clamp01


_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "、": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "…": "...",
    }
)

_FORMULA_SYMBOL_REPLACEMENTS = (
    (r"\leqslant", "<="),
    (r"\geqslant", ">="),
    (r"\leq", "<="),
    (r"\geq", ">="),
    (r"\neq", "!="),
    (r"\ne", "!="),
    (r"\times", "*"),
    (r"\cdot", "*"),
    (r"\div", "/"),
    (r"\leftrightarrow", "<->"),
    (r"\rightarrow", "->"),
    (r"\Rightarrow", "->"),
    (r"\to", "->"),
    ("≤", "<="),
    ("⩽", "<="),
    ("≥", ">="),
    ("⩾", ">="),
    ("≠", "!="),
    ("×", "*"),
    ("∙", "*"),
    ("⋅", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("∕", "/"),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
    ("→", "->"),
    ("⇒", "->"),
    ("↔", "<->"),
)

def _normalise_surface(value: str) -> str:
    """Apply Formula-(31)'s non-semantic Unicode/punctuation normalization."""

    normalized = unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_TRANSLATION)
    for source, target in _FORMULA_SYMBOL_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\r?\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()

def _metric_alias_catalog(kb: KnowledgeBase) -> dict[str, Any]:
    """Compile the unambiguous KB alias -> canonical-name normalizer once."""

    cached = getattr(kb, "_nea_metric_alias_catalog", None)
    if isinstance(cached, dict):
        return cached
    signature = tuple(
        sorted(
            (
                str(chunk.chunk_id),
                str(chunk.concept_name),
                tuple(str(alias) for alias in chunk.aliases),
            )
            for chunk in kb.chunks
        )
    )
    canonical_names: dict[str, set[str]] = {}
    for chunk in kb.chunks:
        canonical = _normalise_surface(chunk.concept_name)
        for label in (chunk.concept_name, *chunk.aliases):
            normalized_label = _normalise_surface(str(label))
            if normalized_label:
                canonical_names.setdefault(normalized_label.casefold(), set()).add(canonical)

    # Ambiguous aliases are deliberately left unchanged: choosing between two
    # concepts would turn a normalization step into an un-auditable linker.
    replacements = {
        label: next(iter(names))
        for label, names in canonical_names.items()
        if len(names) == 1
    }
    ordered_labels = sorted(replacements, key=lambda value: (-len(value), value))
    pattern = (
        re.compile(
            r"(?<![a-z0-9_])(?:" + "|".join(re.escape(value) for value in ordered_labels) + r")(?![a-z0-9_])",
            flags=re.IGNORECASE,
        )
        if ordered_labels
        else None
    )
    catalog = {
        "signature": signature,
        "replacements": replacements,
        "pattern": pattern,
    }
    setattr(kb, "_nea_metric_alias_catalog", catalog)
    return catalog

def _normalise_concept_aliases(value: str, kb: KnowledgeBase | None) -> str:
    if kb is None or not value:
        return value
    catalog = _metric_alias_catalog(kb)
    pattern = catalog["pattern"]
    if pattern is None:
        return value

    def replace(match: re.Match[str]) -> str:
        key = _normalise_surface(match.group(0)).casefold()
        return str(catalog["replacements"].get(key, match.group(0)))

    return pattern.sub(replace, value)

def _normalise_answer(answer: str, *, kb: KnowledgeBase | None = None) -> str:
    value = _normalise_surface(answer)
    return _normalise_concept_aliases(value, kb)

def _segment_normalised_answer(value: str) -> list[str]:
    """Formula (31) splitter after the shared deterministic normalizer."""

    if not value:
        return []
    blocks = [part.strip() for part in re.split(r"\n\s*(?:#{1,6}\s+|(?:\d+[.)]|[一二三四五六七八九十]+[,.])\s+|[-*•]\s+)", value) if part.strip()]
    if len(blocks) >= 2:
        return blocks
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", value) if part.strip()]
    if len(paragraphs) >= 2:
        return paragraphs
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;])(?:\s+|(?=[\u3400-\u9fff]))", value)
        if part.strip()
    ]
    return sentences if len(sentences) >= 2 else paragraphs

def segment_explanation_units(answer: str, *, kb: KnowledgeBase | None = None) -> list[str]:
    """Formula (31) splitter: headings → numbering → bullets → paragraphs → sentences."""

    return _segment_normalised_answer(_normalise_answer(answer, kb=kb))

def combine_lts(variance: float, discontinuity: float, hierarchy: float) -> float:
    """Mean of the three clamped components used by the current internal LTS."""
    return round(clamp01((clamp01(variance) + clamp01(discontinuity) + clamp01(hierarchy)) / 3), 6)
