"""Bounded answer operators for a workflow or an injected host.

These helpers perform no network, model, graph or filesystem I/O.  The host
supplies its allowed evidence and a callable for the registered prompt roles.
Rules establish that an edit can be executed; a separate semantic verdict is
still required before its text can replace the parent answer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from nea.method.citations import extract_citations
from nea.runtime.records import answer_sha256, digest


MODULE_BY_TYPE = {
    "term_gap": "T_EXP",
    "step_gap": "D_EXP",
    "logical_jump": "D_CON",
    "term_excess": "T_CON",
    "step_excess": "D_CON",
}
MODULES = frozenset(MODULE_BY_TYPE.values())
SHARED_TYPES = frozenset({"factual_error", "task_omission"})
MAX_ACTIONS = 3
MAX_GROWTH = 1.2
OPERATOR_VERSION = "nea-answer-operator-0032-v1"


def _obligation_id(row: Mapping[str, Any]) -> str:
    return str(row.get("obligation_id", row.get("id", "")))


@dataclass(frozen=True)
class OperatorContext:
    question: str
    parent_answer: str
    evidence: Sequence[Mapping[str, Any]] = ()
    obligations: Sequence[str | Mapping[str, Any]] = ()
    budget: Mapping[str, Any] = field(default_factory=dict)
    versions: Mapping[str, Any] = field(default_factory=dict)
    base: str = "pe"

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("operator question must be nonempty")
        answer_sha256(self.parent_answer)
        if self.base not in {"pe", "rag"}:
            raise ValueError("operator base must be pe or rag")
        if self.base == "pe" and self.evidence:
            raise ValueError("PE operators cannot receive external evidence bodies")
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in self.evidence:
            if not isinstance(source, Mapping):
                raise ValueError("operator evidence must be source records")
            chunk_id, body = source.get("chunk_id"), source.get("content")
            if (not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen
                    or not isinstance(body, str) or not body.strip()):
                raise ValueError("operator evidence requires unique IDs and nonempty bodies")
            seen.add(chunk_id)
            chunks.append(deepcopy(dict(source)))
        rows = tuple(deepcopy(dict(row)) if isinstance(row, Mapping) else row
                     for row in self.obligations)
        if any(not isinstance(row, (str, Mapping)) or (isinstance(row, str) and not row.strip())
               for row in rows):
            raise ValueError("operator obligations must be nonempty task requirements")
        # The registered prepare schema returns strings.  Hosts that already
        # have an obligation table may supply stable IDs; strings get IDs in
        # program metadata without asking a model to invent them.
        ids = [_obligation_id(row) if isinstance(row, Mapping) else f"obligation_{index + 1}"
               for index, row in enumerate(rows)]
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("operator obligations require unique IDs")
        limits = {"max_growth": MAX_GROWTH, "max_actions": MAX_ACTIONS,
                  "max_candidates": 1, **dict(self.budget)}
        growth = limits["max_growth"]
        if (type(growth) not in {int, float} or not 1 <= growth <= MAX_GROWTH
                or type(limits["max_actions"]) is not int
                or not 0 <= limits["max_actions"] <= MAX_ACTIONS
                or type(limits["max_candidates"]) is not int or limits["max_candidates"] != 1):
            raise ValueError("operator budget exceeds one-candidate contract")
        if any(name in limits and (type(limits[name]) is not int or limits[name] != 1)
               for name in ("max_rounds", "max_validation_calls")):
            raise ValueError("operator permits one revision round and one semantic validation")
        maximum = int(len(self.parent_answer) * growth)
        if "max_chars" in limits:
            if type(limits["max_chars"]) is not int or not 0 < limits["max_chars"] <= maximum:
                raise ValueError("operator max_chars exceeds parent growth budget")
        else:
            limits["max_chars"] = maximum
        object.__setattr__(self, "evidence", tuple(chunks))
        object.__setattr__(self, "obligations", rows)
        object.__setattr__(self, "budget", deepcopy(limits))
        object.__setattr__(self, "versions", deepcopy(dict(self.versions)))

    @property
    def parent_sha256(self) -> str:
        return answer_sha256(self.parent_answer)

    @property
    def allowed_citations(self) -> tuple[str, ...]:
        return tuple(sorted(row["chunk_id"] for row in self.evidence))

    @property
    def context_sha256(self) -> str:
        return digest({"question": self.question, "parent_sha256": self.parent_sha256,
                       "evidence": list(self.evidence), "obligations": list(self.obligations),
                       "budget": dict(self.budget), "versions": dict(self.versions),
                       "base": self.base, "operator_version": OPERATOR_VERSION})


def prepare_payload(ctx: OperatorContext) -> dict[str, Any]:
    """Allow a host to reuse its task representation instead of calling prepare."""
    return {"question": ctx.question, "retrieval": ctx.base.upper(),
            "knowledge_chunks": deepcopy(list(ctx.evidence)),
            "obligations": deepcopy(list(ctx.obligations)),
            "constraints": deepcopy(dict(ctx.budget)), "versions": deepcopy(dict(ctx.versions))}


def diagnose_payload(ctx: OperatorContext, *, generic: bool = False) -> dict[str, Any]:
    payload = {**prepare_payload(ctx), "answer": ctx.parent_answer,
            "allowed_citations": list(ctx.allowed_citations),
            "citation_map": {row["chunk_id"]: deepcopy(dict(row)) for row in ctx.evidence},
            "parent_sha256": ctx.parent_sha256, "context_sha256": ctx.context_sha256}
    if generic:
        payload.pop("obligations", None)
    return payload


def revise_payload(ctx: OperatorContext, actions: Sequence[Mapping[str, Any]], *,
                   generic: bool = False) -> dict[str, Any]:
    return {**diagnose_payload(ctx, generic=generic), "issues" if generic else "actions": deepcopy(list(actions)),
            "max_chars": ctx.budget["max_chars"]}


def validate_payload(ctx: OperatorContext, candidate: str,
                     actions: Sequence[Mapping[str, Any]], *, generic: bool = False) -> dict[str, Any]:
    return {**revise_payload(ctx, actions, generic=generic), "parent_answer": ctx.parent_answer,
            "candidate_answer": candidate, "candidate_sha256": answer_sha256(candidate)}


def unique_exact_span(body: str, text: str) -> bool:
    """Require one start position, including overlapping occurrences."""
    start = body.find(text)
    return bool(text) and start >= 0 and body.find(text, start + 1) < 0


def _anchor(row: Mapping[str, Any], ctx: OperatorContext) -> tuple[str, int, int] | None:
    """Resolve an exact, unambiguous span in the answer or the question."""
    question_anchor = row.get("question_anchor", "")
    if not isinstance(question_anchor, str) or (question_anchor and (
            not question_anchor.strip() or question_anchor not in ctx.question)):
        return None
    raw = row.get("anchor", row.get("span", row.get("location")))
    if not raw and row.get("question_anchor"):
        raw = {"text": row["question_anchor"], "source": "question"}
    if isinstance(raw, str):
        anchor = {"text": raw, "source": row.get("anchor_source", "answer")}
    elif isinstance(raw, Mapping):
        anchor = dict(raw)
    else:
        return None
    source = anchor.get("source", "answer")
    source = "answer" if source == "parent_answer" else source
    if source not in {"answer", "question"}:
        return None
    body = ctx.parent_answer if source == "answer" else ctx.question
    text = anchor.get("text", anchor.get("quote", ""))
    if not isinstance(text, str) or not text.strip():
        return None
    start, end = anchor.get("start"), anchor.get("end")
    if start is not None or end is not None:
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(body) or body[start:end] != text):
            return None
    else:
        if not unique_exact_span(body, text):
            return None
        start = body.index(text)
        end = start + len(text)
    return source, start, end


def action_requirement_error(row: Mapping[str, Any], question: str, *, generic: bool = False) -> str | None:
    """Check supplied task provenance, not the semantic truth of necessity."""
    if generic:
        if row.get("scope_effect") not in {"correction", "add_obligation"}:
            return "invalid_scope_effect"
        expands = row["scope_effect"] == "add_obligation"
    else:
        kind = row.get("defect_type", row.get("type", row.get("kind")))
        expands = row.get("action") in {"add", "bridge"} or kind in {
            "task_omission", "term_gap", "step_gap"}
    if expands:
        anchor = row.get("question_anchor", "")
        if not isinstance(anchor, str) or not anchor.strip() or anchor not in question:
            return "missing_task_anchor"
        necessity = row.get("necessity_reason", "")
        if not isinstance(necessity, str) or not necessity.strip():
            return "missing_necessity_reason"
    return None


def _support(row: Mapping[str, Any], ctx: OperatorContext) -> bool:
    state = row.get("evidence_status")
    if state not in {"supported", "reasoning"}:
        return False
    ids = row.get("evidence_ids", row.get("source_ids", []))
    if (not isinstance(ids, (list, tuple)) or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= set(ctx.allowed_citations)):
        return False
    # A supported external claim must name an allowed block.  A PE diagnosis
    # may instead locate the question or state its explicit reasoning basis.
    basis = row.get("basis", row.get("support_basis", row.get("evidence_basis", row.get("reason", ""))))
    if isinstance(basis, Mapping):
        kind, quote = basis.get("source", basis.get("type")), basis.get("text", basis.get("quote", ""))
        if kind in {"question", "query"}:
            return isinstance(quote, str) and bool(quote) and quote in ctx.question
        if kind in {"reasoning", "explicit_reasoning"}:
            return isinstance(quote, str) and bool(quote.strip())
        if kind in {"evidence", "allowed_evidence"}:
            return bool(ids) and isinstance(quote, str) and bool(quote.strip())
        return False
    if ids:
        return isinstance(basis, str) and bool(basis.strip())
    if state == "reasoning":
        return isinstance(basis, str) and bool(basis.strip())
    query_anchor = row.get("question_anchor", "")
    return (isinstance(query_anchor, str) and bool(query_anchor) and query_anchor in ctx.question
            and isinstance(basis, str) and bool(basis.strip()))


def select_actions(ctx: OperatorContext, diagnosis: Mapping[str, Any], *,
                   enabled_modules: Sequence[str] = tuple(sorted(MODULES)),
                   generic: bool = False) -> dict[str, Any]:
    """Reject unsupported, unlocated, conflicting or disabled actions."""
    enabled = set(enabled_modules)
    if not enabled <= MODULES:
        raise ValueError("unknown NEA module")
    status = diagnosis.get("status", "issues")
    if status in {"no_issue", "uncertain"}:
        if diagnosis.get("actions", diagnosis.get("issues", [])):
            return {"status": "invalid_diagnosis", "actions": [], "rejections": []}
        return {"status": status, "actions": [], "rejections": []}
    if status not in {"issues", "defects", "actionable"}:
        return {"status": "invalid_diagnosis", "actions": [], "rejections": []}
    rows = diagnosis.get("actions", diagnosis.get("defects", diagnosis.get("issues", [])))
    if not isinstance(rows, (list, tuple)):
        return {"status": "invalid_diagnosis", "actions": [], "rejections": []}
    selected: list[dict[str, Any]] = []
    spans: list[tuple[str, int, int]] = []
    rejected: list[dict[str, Any]] = []
    obligation_ids = {_obligation_id(row) if isinstance(row, Mapping) else f"obligation_{index + 1}"
                      for index, row in enumerate(ctx.obligations)}
    for index, raw in enumerate(rows):
        reason: str | None = None
        if not isinstance(raw, Mapping):
            rejected.append({"index": index, "reason": "invalid_action"})
            continue
        row = deepcopy(dict(raw))
        kind = "generic_issue" if generic else row.get("defect_type", row.get("type", row.get("kind")))
        module = MODULE_BY_TYPE.get(kind)
        if not generic and module is None and kind not in SHARED_TYPES:
            reason = "unknown_defect_type"
        elif module is not None and module not in enabled:
            reason = "module_disabled"
        elif not generic and row.get("module") != (module or "shared"):
            reason = "invalid_module_mapping"
        elif not generic and row.get("action") not in {"add", "compress", "bridge", "rewrite"}:
            reason = "unknown_action"
        instruction = row.get("reason", "") if generic else row.get("instruction", "")
        if isinstance(instruction, Mapping):
            instruction = instruction.get("instruction", "")
        if reason is None and (not isinstance(instruction, str) or not instruction.strip()):
            reason = "empty_instruction"
        if reason is None and (not isinstance(row.get("reason"), str) or not row["reason"].strip()):
            reason = "missing_reason"
        span = _anchor(row, ctx)
        if reason is None and span is None:
            reason = "invalid_anchor"
        if reason is None:
            reason = action_requirement_error(row, ctx.question, generic=generic)
        referenced = row.get("obligation_ids", [])
        if "obligation_id" in row:
            referenced = [row["obligation_id"]]
        if reason is None and (not isinstance(referenced, (list, tuple))
                               or any(not isinstance(item, str) for item in referenced)
                               or not set(referenced) <= obligation_ids):
            reason = "unknown_obligation"
        if reason is None and not _support(row, ctx):
            reason = "unsupported_action"
        if reason is None and span is not None:
            if any(source == span[0] and max(start, span[1]) < min(end, span[2])
                   for source, start, end in spans):
                reason = "conflicting_or_duplicate_action"
        if reason is None and len(selected) >= ctx.budget["max_actions"]:
            reason = "action_budget"
        if reason is not None:
            rejected.append({"index": index, "reason": reason})
            continue
        assert span is not None
        body = ctx.parent_answer if span[0] == "answer" else ctx.question
        row.update({"instruction": instruction,
                    "anchor": {"source": span[0], "start": span[1], "end": span[2],
                               "text": body[span[1]:span[2]]}})
        if not generic:
            row.update({"type": kind, "module": module or "shared"})
        selected.append(row)
        spans.append(span)
    return {"status": "actionable" if selected else "no_executable_action",
            "actions": selected, "rejections": rejected}


def validate_candidate(ctx: OperatorContext, text: Any) -> tuple[bool, str]:
    """Structural checks; valid conditions and citation support need semantics."""
    if not isinstance(text, str) or not text.strip():
        return False, "invalid_candidate"
    if text.strip() == ctx.parent_answer.strip():
        return False, "unchanged"
    if len(text) > ctx.budget["max_chars"]:
        return False, "length_exceeded"
    if text.lstrip().startswith(("```", "{", "[")):
        return False, "invalid_answer_format"
    if not set(extract_citations(text)) <= set(ctx.allowed_citations):
        return False, "unknown_citation"
    return True, "pending_semantic_validation"


def semantic_accepts(ctx: OperatorContext, candidate: str, verdict: Mapping[str, Any]) -> tuple[bool, str]:
    """Use only an affirmative semantic verdict for this parent and candidate."""
    if (verdict.get("parent_sha256", ctx.parent_sha256) != ctx.parent_sha256
            or verdict.get("candidate_sha256", answer_sha256(candidate)) != answer_sha256(candidate)):
        return False, "validation_identity_mismatch"
    status = verdict.get("status", verdict.get("verdict", "uncertain"))
    if status in {"uncertain", "unknown"}:
        return False, "semantic_uncertain"
    if status != "pass":
        return False, "semantic_rejected"
    # Validate the registered verdict schema.  Its role asks the four semantic
    # safety questions; the rules do not claim to prove those properties.
    from nea.method.templates import validation_schema
    from nea.runtime.structured import validate_json_schema
    try:
        validate_json_schema(dict(verdict), validation_schema())
    except ValueError:
        return False, "invalid_semantic_verdict"
    if not verdict["reason"].strip():
        return False, "invalid_semantic_verdict"
    for finding in verdict["findings"]:
        question_anchor = finding.get("question_anchor", "")
        if question_anchor and (not question_anchor.strip() or question_anchor not in ctx.question):
            return False, "semantic_findings_unresolved"
        located = _anchor(finding, ctx) is not None
        candidate_span = finding.get("span", "")
        if isinstance(candidate_span, str) and candidate_span and unique_exact_span(candidate, candidate_span):
            located = True
        if (not located
                or finding["evidence_status"] not in {"supported", "reasoning"}
                or not _support(finding, ctx)):
            return False, "semantic_findings_unresolved"
    return True, "accepted"


def finalize(ctx: OperatorContext, candidate: str | None = None,
             semantic: Mapping[str, Any] | None = None, *,
             actions: Sequence[Mapping[str, Any]] = (), status: str | None = None,
             diagnosis: Mapping[str, Any] | None = None,
             rejections: Sequence[Mapping[str, Any]] = (),
             calls: Mapping[str, int] | None = None) -> dict[str, Any]:
    accepted, reason = False, status or "no_candidate"
    if candidate is not None and status is None:
        accepted, reason = (validate_candidate(ctx, candidate) if actions else (False, "no_executable_action"))
        if accepted:
            accepted, reason = semantic_accepts(ctx, candidate, semantic or {})
    answer = candidate if accepted else ctx.parent_answer
    assert isinstance(answer, str)
    candidate_record = None
    if candidate is not None:
        candidate_record = {"answer": candidate, "answer_sha256": answer_sha256(candidate)
                            if candidate.strip() else None, "status": reason,
                            "semantic": deepcopy(dict(semantic)) if semantic is not None else None}
    return {"answer": answer, "answer_sha256": answer_sha256(answer),
            "parent_sha256": ctx.parent_sha256, "context_sha256": ctx.context_sha256,
            "operator_version": OPERATOR_VERSION, "versions": deepcopy(dict(ctx.versions)),
            "status": "accepted" if accepted else "parent_retained", "revised": accepted,
            "fallback_reason": None if accepted else reason, "candidate_record": candidate_record,
            "candidate_records": [] if candidate_record is None else [candidate_record],
            "actions": deepcopy(list(actions)), "rejections": deepcopy(list(rejections)),
            "diagnosis": deepcopy(dict(diagnosis)) if diagnosis is not None else None,
            "calls": dict(calls or {}), "candidate_count": int(candidate is not None)}


class RoleCaller(Protocol):
    def __call__(self, role: str, payload: Mapping[str, Any]) -> Any: ...


def run_operator(ctx: OperatorContext, call: RoleCaller, *, generic: bool = False,
                 enabled_modules: Sequence[str] = tuple(sorted(MODULES))) -> dict[str, Any]:
    """Minimal synchronous host adapter, with one candidate and safe fallback.

    A production host can call the same pure helpers in batches and use its
    own append-only transport, budgets and metrics.  This adapter records
    logical role invocations; the transport remains the request-cost authority.
    """
    roles = {"diagnose": "generic_answer_critique" if generic else "explanation_diagnosis",
             "revise": "answer_revision", "validate": "revision_validation"}
    calls = {"diagnose": 0, "revise": 0, "validate": 0}
    diagnosis: Mapping[str, Any] | None = None
    chosen: dict[str, Any] = {"actions": [], "rejections": []}
    candidate: str | None = None
    phase = "diagnose"
    try:
        calls[phase] += 1
        diagnosis = call(roles[phase], diagnose_payload(ctx, generic=generic))
        if not isinstance(diagnosis, Mapping):
            return finalize(ctx, status="invalid_diagnosis", calls=calls)
        chosen = select_actions(ctx, diagnosis, enabled_modules=enabled_modules, generic=generic)
        if not chosen["actions"]:
            return finalize(ctx, status=chosen["status"], diagnosis=diagnosis,
                            rejections=chosen["rejections"], calls=calls)
        phase = "revise"
        calls[phase] += 1
        result = call(roles[phase], revise_payload(ctx, chosen["actions"], generic=generic))
        candidate = result.get("answer") if isinstance(result, Mapping) else result
        valid, reason = validate_candidate(ctx, candidate)
        if not valid:
            return finalize(ctx, candidate if isinstance(candidate, str) else None,
                            status=reason, actions=chosen["actions"], diagnosis=diagnosis,
                            rejections=chosen["rejections"], calls=calls)
        phase = "validate"
        calls[phase] += 1
        semantic = call(roles[phase], validate_payload(ctx, candidate, chosen["actions"], generic=generic))
        if not isinstance(semantic, Mapping):
            semantic = {"status": "uncertain"}
        return finalize(ctx, candidate, semantic, actions=chosen["actions"], diagnosis=diagnosis,
                        rejections=chosen["rejections"], calls=calls)
    except Exception:
        # Exception text can contain source bodies or credentials; it is not
        # copied into results.  The host transport keeps its sanitized failure.
        return finalize(ctx, candidate, {"status": "uncertain"} if phase == "validate" else None,
                        status=f"{phase}_failed", actions=chosen["actions"], diagnosis=diagnosis,
                        rejections=chosen["rejections"], calls=calls)


def local_nli_check(*args: Any, enabled: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Closed optional interface: no dependency imports, weights or inference."""
    if enabled:
        raise ValueError("NLI is not enabled for the 0022 main workflow")
    return {"status": "disabled", "requests": 0}
