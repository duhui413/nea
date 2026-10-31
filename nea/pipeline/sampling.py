"""Deterministic exploratory subsets and explicit run-mode isolation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nea.question_set.contract import QuestionSet, normalized_sha256
from nea.runtime.records import write_once


def validate_sample_size(size: int) -> int:
    if type(size) is not int or not 1 <= size <= 60:
        raise ValueError("sample_size must be an integer within 1..60")
    return size


def mode_fields(*, smoke: bool = False, sample_size: int | None = None) -> dict[str, Any]:
    if smoke and sample_size is not None:
        raise ValueError("Smoke and exploratory sample modes are mutually exclusive")
    if sample_size is not None:
        return {"mode": "exploratory", "exploratory": True,
                "sample_size": validate_sample_size(sample_size)}
    return {"smoke": True} if smoke else {}


def check_mode(document: Mapping[str, Any], *, smoke: bool = False,
               sample_size: int | None = None) -> None:
    expected = mode_fields(smoke=smoke, sample_size=sample_size)
    actual = {key: document[key] for key in ("mode", "exploratory", "sample_size", "smoke")
              if key in document and document[key] is not False}
    if actual != expected:
        raise ValueError("Smoke, exploratory and formal run identities cannot be mixed")


def select_sample(question_set: QuestionSet, size: int) -> tuple[QuestionSet, dict[str, Any]]:
    size = validate_sample_size(size)
    rows = tuple(row for split in ("dev", "test")
                 for row in sorted(question_set.split_rows(split), key=lambda row: row["qid"])[:size])
    subset = QuestionSet(rows, normalized_sha256(rows), None)
    if subset.counts != {"dev": size, "test": size}:
        raise ValueError("Source question set does not contain the requested sample")
    scope = {**mode_fields(sample_size=size), "source_question_set_sha256": question_set.sha256,
             "question_set_sha256": subset.sha256, "n_dev": size, "n_test": size,
             "qids": {split: [row["qid"] for row in subset.split_rows(split)]
                      for split in ("dev", "test")}}
    return subset, scope


def validate_sample_scope(scope: Mapping[str, Any]) -> int:
    size = validate_sample_size(scope.get("sample_size"))
    check_mode(scope, sample_size=size)
    if scope.get("n_dev") != size or scope.get("n_test") != size:
        raise ValueError("Sample scope counts differ")
    qids = scope.get("qids")
    if not isinstance(qids, dict) or set(qids) != {"dev", "test"}:
        raise ValueError("Sample scope requires dev/test qids")
    for split in ("dev", "test"):
        values = qids[split]
        if (not isinstance(values, list) or len(values) != size
                or not all(isinstance(value, str) for value in values)
                or values != sorted(set(values))):
            raise ValueError("Sample scope qids must be unique and sorted")
    if set(qids["dev"]) & set(qids["test"]):
        raise ValueError("Sample dev/test qids overlap")
    for name in ("source_question_set_sha256", "question_set_sha256"):
        value = scope.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Sample scope hash is malformed")
    return size


def scope_question_set(run_root: Path, question_set: QuestionSet, *, smoke: bool = False,
                       sample_size: int | None = None) -> QuestionSet:
    """Publish or verify a scope while the caller holds the run-directory lock."""
    import json

    mode_fields(smoke=smoke, sample_size=sample_size)
    smoke_path, sample_path = run_root / "smoke_scope.json", run_root / "sample_scope.json"
    if smoke_path.exists() and not smoke:
        raise ValueError("Smoke outputs cannot be used by another run mode")
    if sample_path.exists() and sample_size is None:
        raise ValueError("Exploratory outputs cannot be used by another run mode")
    for path in sorted(run_root.glob("*/*/run_manifest.json")):
        check_mode(json.loads(path.read_text(encoding="utf8")), smoke=smoke, sample_size=sample_size)
    if sample_size is not None:
        subset, scope = select_sample(question_set, sample_size)
        write_once(sample_path, scope)
        return subset
    if smoke:
        selected = tuple(row for split in ("dev", "test")
                         for row in sorted(question_set.split_rows(split), key=lambda row: row["qid"])[:3])
        subset = QuestionSet(selected, normalized_sha256(selected), None)
        if subset.counts != {"dev": 3, "test": 3}:
            raise ValueError("Smoke requires dev3/test3")
        write_once(smoke_path, {"smoke": True, "source_question_set_sha256": question_set.sha256,
                               "qids": {split: [row["qid"] for row in subset.split_rows(split)]
                                        for split in ("dev", "test")}})
        return subset
    return question_set
