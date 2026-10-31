"""Sealed paper questions; split comes from each row, never inferred from qid.

The knowledge-base metadata is one JSONL row.
Hashes sort rows by qid, sort JSON keys and use compact UTF-8 lines.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


QID_PATTERN = {"dev": re.compile(r"V3-D(\d{3})"), "test": re.compile(r"V3-T(\d{3})")}
QID_RANGE = {"dev": (1, 60), "test": (1, 240)}
STYLES = frozenset({"direct", "prerequisite", "application"})
LABELS = frozenset({"gap", "given", "assumed"})
ROW_FIELDS = frozenset({"qid", "split", "question", "style", "authoring_target_id",
                        "gold_target_ids", "gold_labels", "status"})
MANIFEST_FIELDS = frozenset({"sha256", "n_dev", "n_test", "sealed_at_commit"})
CONCEPT_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*_\d{3,}")


def canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(row), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def normalized_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get("qid", "")))
    return hashlib.sha256(b"".join(canonical_row_bytes(row) for row in ordered)).hexdigest()


def validate_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != ROW_FIELDS:
        missing = sorted(ROW_FIELDS - set(row)) if isinstance(row, Mapping) else []
        extra = sorted(set(row) - ROW_FIELDS) if isinstance(row, Mapping) else []
        raise ValueError(f"question row fields differ from contract: missing={missing} extra={extra}")
    qid, split = row["qid"], row["split"]
    if split not in {"dev", "test"}:
        raise ValueError(f"{qid!r}: split must be dev or test")
    historical = "dev" if str(qid).startswith("V3-D") else "test"
    match = QID_PATTERN[historical].fullmatch(str(qid))
    low, high = QID_RANGE[historical]
    if match is None or not low <= int(match.group(1)) <= high:
        raise ValueError(f"{qid!r}: qid is outside the {split} range")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{qid}: question must be nonempty text")
    if row["style"] not in STYLES:
        raise ValueError(f"{qid}: unknown style {row['style']!r}")
    if row["status"] != "sealed":
        raise ValueError(f"{qid}: status must be sealed")
    target = row["authoring_target_id"]
    if not isinstance(target, str) or not CONCEPT_ID.fullmatch(target):
        raise ValueError(f"{qid}: authoring_target_id must be a concept id")
    gold = row["gold_target_ids"]
    if (not isinstance(gold, list) or not 1 <= len(gold) <= 2 or len(set(gold)) != len(gold)
            or not all(isinstance(item, str) and CONCEPT_ID.fullmatch(item) for item in gold)):
        raise ValueError(f"{qid}: gold_target_ids must be 1-2 unique concept ids")
    labels = row["gold_labels"]
    if not isinstance(labels, Mapping) or any(
            not isinstance(key, str) or not CONCEPT_ID.fullmatch(key) or value not in LABELS
            for key, value in labels.items()):
        raise ValueError(f"{qid}: gold_labels must map concept ids to gap/given/assumed")
    if set(labels) & set(gold):
        raise ValueError(f"{qid}: a gold target cannot carry a prerequisite label")
    return {**dict(row), "gold_target_ids": list(gold), "gold_labels": dict(sorted(labels.items()))}


@dataclass(frozen=True)
class QuestionSet:
    rows: tuple[dict[str, Any], ...]
    sha256: str
    manifest: dict[str, Any] | None

    @property
    def by_qid(self) -> dict[str, dict[str, Any]]:
        return {row["qid"]: row for row in self.rows}

    def split_rows(self, split: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["split"] == split]

    def qid_splits(self) -> dict[str, str]:
        return {row["qid"]: row["split"] for row in self.rows}

    @property
    def counts(self) -> dict[str, int]:
        return {"dev": len(self.split_rows("dev")), "test": len(self.split_rows("test"))}

    def sealed_problems(self) -> list[str]:
        """Reasons the set is not sealed (empty list = sealed and consistent)."""
        problems: list[str] = []
        if self.manifest is None:
            return ["question_set_manifest_missing"]
        if not MANIFEST_FIELDS <= set(self.manifest):
            problems.append("question_set_manifest_fields")
        if self.manifest.get("sha256") != self.sha256:
            problems.append("question_set_sha256_mismatch")
        if (self.manifest.get("n_dev") != self.counts["dev"]
                or self.manifest.get("n_test") != self.counts["test"]):
            problems.append("question_set_counts_mismatch")
        version = self.manifest.get("version")
        if version is not None:
            if version != "nea-questions-0022-v1":
                problems.append("question_set_unknown_version")
            if self.counts != {"dev": 30, "test": 270}:
                problems.append("question_set_0022_counts")
            exposure = self.manifest.get("exposure", {})
            if set(exposure) != set(self.by_qid) or any(
                    exposure.get(r["qid"], {}).get("current_split") != r["split"] for r in self.rows):
                problems.append("question_set_exposure_mismatch")
        elif self.counts != {"dev": 60, "test": 240}:
            problems.append("question_set_legacy_counts")
        commit = self.manifest.get("sealed_at_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            problems.append("question_set_sealed_at_commit")
        if any(row["status"] != "sealed" for row in self.rows):
            problems.append("question_set_rows_not_sealed")
        return problems


def parse_question_set(data: bytes, manifest: Mapping[str, Any] | None = None) -> QuestionSet:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"question_set line {number} is not JSON") from exc
        row = validate_row(raw)
        if row["qid"] in seen:
            raise ValueError(f"duplicate qid {row['qid']}")
        seen.add(row["qid"])
        rows.append(row)
    if not rows:
        raise ValueError("question set is empty")
    ordered = tuple(sorted(rows, key=lambda row: row["qid"]))
    return QuestionSet(rows=ordered, sha256=normalized_sha256(ordered),
                       manifest=None if manifest is None else dict(manifest))


def load_question_set(path: Path) -> QuestionSet:
    """Read sealed questions and their sibling questions_meta.jsonl."""
    manifest_path = path.with_name("questions_meta.jsonl")
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8").strip())
        if not isinstance(manifest, dict):
            raise ValueError("questions_meta.jsonl must contain one object")
    return parse_question_set(path.read_bytes(), manifest)
