"""Real requests, bounded structure retries and successful-result reuse."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from nea.pipeline.artifacts import RunDirectoryLock
from nea.providers import config
from nea.schemas import SchemaError, Step1Plan
from nea.providers.client import LLMTransportError
from nea.runtime.structured import (stage_schema, schema_hash, validate_json_schema,
                                    structured_output_instruction)
from nea.evaluation.judging import RUBRIC_ANCHORS
from nea.method.citations import extract_citations
from nea.runtime.records import AppendOnlyRun, ConflictError, canonical_bytes, digest, write_once
from nea.runtime import operator_protocol

from nea.method.gaps import validate_d1_structure, validate_d3_structure
from nea.method.grounding import validate_grounding_structure
from nea.runtime.identity import model_config_identity
from nea.pipeline.registry import (
    EVAL_STAGE,
    PARENT_ARM,
    PROTOCOL,
    check_arm_stage,
    role_for_stage,
    structure_attempts,
)
from nea.method.templates import REPAIR_NOTE


MAX_IN_FLIGHT = 2  # default for routers that do not declare max_in_flight

# Only local validator rules may enter retry prompts or persisted failure evidence.
# Do not copy arbitrary exception messages, output values, or citation IDs here.
from nea.prompts import DEFAULT_PROMPTS_DIR
_VALIDATION_REASONS = json.loads((DEFAULT_PROMPTS_DIR / "validation_reasons.json").read_text(encoding="utf-8"))
_VALIDATION_REASON_HASH = hashlib.sha256((DEFAULT_PROMPTS_DIR / "validation_reasons.json").read_bytes()).hexdigest()
_SAFE_BUSINESS_MESSAGES = frozenset({
    "response must be a JSON object", "plan must be {route, needed_prerequisites}",
    "needed_prerequisites must be <= 6 names", "arm B step1 planning_mode must be baseline",
    "diagnose must be {actions: [...<=3]}", "diagnose action malformed",
    "critique must be {issues: [<=3 strings]}", "judge must score all six rubric dimensions",
    "rubric scores must be integers 1..5", "grounding response must be {grounded_target_ids: [...]} only",
    "grounded_target_ids must be a list of strings", "D1 response must be {gap_concepts: [...]} only",
    "gap_concepts must be a list of names", "D3 response must be {labels: {...}} only",
    "D3 must label exactly every local candidate", "D3 labels must be gap/given/assumed",
})
_SAFE_SCHEMA_REASON = re.compile(
    r"(?:instance|schema)(?:\.(?:properties|items|anyOf|property|item))*: "
    r"(?:non_json_value|unsupported_schema_keyword_or_shape|unsupported_schema_type|invalid_properties|"
    r"invalid_required|unsupported_additionalProperties|unsupported_uniqueItems|invalid_minItems|"
    r"invalid_maxItems|invalid_minLength|invalid_minimum|invalid_maximum|invalid_enum|duplicate_enum|"
    r"invalid_description|invalid_anyOf|type|enum|anyOf|required|additionalProperties|array_length|"
    r"uniqueItems|minLength|number_range)"
)


def _safe_business_reason(exc: Exception) -> str:
    message = str(exc)
    if message in _SAFE_BUSINESS_MESSAGES:
        return "business_constraint: " + message
    if re.fullmatch(r"(?:stage_id|planning_mode|rewritten_query|query|target_concept_id|slot|"
                    r"answer_outline|retrieval_queries|assumptions|warnings)(?:\[\d+\])? must not be empty", message):
        return "nonempty_string_required: Fill every required string with non-whitespace text."
    if message.startswith("target_concept_id must be a real identifier, not "):
        return "real_identifier_required: Use a real target_concept_id, not an unknown or null placeholder."
    return _VALIDATION_REASONS["business_validation_error"]


class RouterLike(Protocol):
    async def complete(self, *, role: str, prompt: str, temperature: float,
                       max_tokens: int, seed: int, timeout: float,
                       output_schema: dict | None = None,
                       schema_name: str | None = None) -> tuple[Any, list[Any]]: ...


class OutputValidationError(SchemaError):
    def __init__(self, message: str, *, code: str, validation_reason: str | None = None):
        super().__init__(message)
        self.code = code
        self.validation_reason = validation_reason or _VALIDATION_REASONS.get(
            code, _VALIDATION_REASONS["business_validation_error"])


@dataclass(frozen=True)
class ExperimentJob:
    qid: str
    split: str
    base: str
    arm: str
    stage: str
    role: str
    prompt: str
    template_hash: str
    static_payload_hash: str
    parent_hash: str
    calibration_hash: str
    rubric_hash: str
    call_index: int = 1
    schema_args: tuple[tuple[str, Any], ...] = field(default=())

    def arg(self, name: str, default: Any = None) -> Any:
        return dict(self.schema_args).get(name, default)

    def __hash__(self) -> int:
        # Operator jobs carry their concrete schema as a dictionary.  Equality
        # remains the dataclass field comparison; a canonical digest lets the
        # same jobs continue to serve as result-map keys.
        return hash(digest({name: getattr(self, name) for name in self.__dataclass_fields__}))


def _job_schema(job: ExperimentJob) -> dict[str, Any] | None:
    if not operator_protocol.is_operator_job(job):
        return stage_schema(job)
    if "output_schema" not in dict(job.schema_args):
        raise ValueError("operator job requires an explicit output_schema")
    schema = job.arg("output_schema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise ValueError("operator output_schema is not valid JSON") from exc
    if job.stage in operator_protocol.MARKDOWN_STAGES:
        if schema is not None:
            raise ValueError("operator answer/revise stages require Markdown output")
    elif not isinstance(schema, dict):
        raise ValueError("structured operator stages require their concrete output schema")
    return schema


def _check_operator_schema(run: AppendOnlyRun, job: ExperimentJob, schema: dict | None) -> None:
    """Only registered role contracts, with grounding's explicit candidate enum."""
    from nea.method.templates import role_schema
    role = job.arg("schema_name")
    if operator_protocol.ROLE_STAGE.get(role) != job.stage:
        raise ConflictError("operator schema role does not match its stage")
    generic = role_schema(role, {})
    if run.identity.get("schema_hashes", {}).get(role) != digest(generic):
        raise ConflictError("operator schema role is not registered in the run manifest")
    expected = generic
    if role == "target_grounding":
        try:
            candidates = schema["properties"]["grounded_target_ids"]["items"].get("enum", [])
            expected = role_schema(role, {"candidates": [{"concept_id": key} for key in candidates]})
        except (KeyError, TypeError, ValueError) as exc:
            raise ConflictError("operator grounding schema differs from its registered contract") from exc
    if canonical_bytes(schema) != canonical_bytes(expected):
        raise ConflictError("operator output schema differs from its registered role contract")
    if run.identity.get("template_hashes", {}).get(role) != job.template_hash:
        raise ConflictError("operator template hash does not match its registered role")
    if run.identity.get("static_payload_hashes", {}).get(role) != job.static_payload_hash:
        raise ConflictError("operator static contract hash does not match its registered role")
    repair = job.arg("repair_instruction")
    if (not isinstance(repair, str) or not repair.strip()
            or run.identity.get("shared_template_hashes", {}).get("repair_output") != digest(repair.encode("utf8"))):
        raise ConflictError("operator repair instruction is not registered in the run manifest")
    # Safe validation reasons currently have one file-backed source.  Custom
    # prompt trees can vary role instructions while this runtime dictionary
    # must match the source whose actual text enters retry prompts.
    actual_reasons = hashlib.sha256((DEFAULT_PROMPTS_DIR / "validation_reasons.json").read_bytes()).hexdigest()
    if (run.identity.get("prompt_files", {}).get("validation_reasons.json") != actual_reasons
            or actual_reasons != _VALIDATION_REASON_HASH):
        raise ConflictError("operator validation reasons differ from the locked runtime source")


def _operator_locatable(job: ExperimentJob, parsed: dict) -> None:
    """A blind reason locates supplied text and cites only supplied references."""
    if job.stage not in {"judge", "validate", "diagnose", "critique"}:
        return
    from nea.method.operators import action_requirement_error, unique_exact_span
    diagnostic = job.stage in {"diagnose", "critique"}
    answer = job.arg("locatable_answer", "")
    parent = job.arg("locatable_parent", "") if job.stage == "validate" else ""
    question = job.arg("locatable_question", "")
    allowed = set(job.arg("allowed_evidence_ids", ()))
    rows = [(f"findings[{index}]", row) for index, row in enumerate(parsed.get("findings", ()))]
    if diagnostic:
        field = "actions" if job.stage == "diagnose" else "issues"
        rows = [(f"{field}[{index}]", row) for index, row in enumerate(parsed.get(field, ()))]
    if job.stage == "judge":
        rows.extend((f"reasons.{dimension}", parsed["reasons"][dimension])
                    for dimension in sorted(RUBRIC_ANCHORS))
    for location, row in rows:
        span, anchor = row.get("span", ""), row.get("question_anchor", "")
        span_ok = bool(span.strip()) and (span in answer or span in parent)
        anchor_ok = bool(anchor.strip()) and anchor in question
        # Only schema-controlled paths and fixed instructions enter repair feedback.
        # Never echo model-supplied text, citation IDs or arbitrary exception messages.
        problem = None
        if span and not span_ok:
            source = ("parent answer" if diagnostic else "candidate_answer or parent answer"
                      if job.stage == "validate" else "candidate_answer")
            problem = f"{location}.span: copy one exact contiguous substring from {source}; preserve Markdown, punctuation and whitespace; do not paraphrase or join excerpts."
        elif diagnostic and span and not unique_exact_span(answer, span):
            problem = f"{location}.span: choose a short exact substring that occurs exactly once in parent answer."
        elif anchor and not anchor_ok:
            problem = f"{location}.question_anchor: copy one exact contiguous substring from question, or use an empty string when span supplies the location."
        elif diagnostic and not span and anchor_ok and not unique_exact_span(question, anchor):
            problem = f"{location}.question_anchor: without span, choose a substring occurring exactly once in question."
        elif not (span_ok or anchor_ok):
            problem = f"{location}: supply at least one nonempty exact span or question_anchor; use empty strings for unused location fields."
        elif not set(row.get("evidence_ids", ())).issubset(allowed):
            problem = f"{location}.evidence_ids: use only supplied knowledge_chunks chunk_id values; use an empty array when no evidence is supplied."
        if problem is None and diagnostic:
            requirement = action_requirement_error(row, question, generic=job.stage == "critique")
            if requirement == "missing_task_anchor":
                problem = f"{location}.question_anchor: added obligations require an exact nonempty question excerpt."
            elif requirement == "missing_necessity_reason":
                problem = f"{location}.necessity_reason: explain why omitting this added obligation prevents answering that question excerpt."
            elif requirement is not None:
                problem = f"{location}.scope_effect: use correction or add_obligation, matching the proposed change."
        if problem:
            raise OutputValidationError("operator finding does not locate supplied text or evidence",
                                        code="business_validation_error",
                                        validation_reason="location_constraint: " + problem
                                        + " Recheck every supplied location and task basis; prefer short verbatim excerpts without crossing lines.")


def _markdown(text: str, allowed: Iterable[str] | None) -> Any:
    answer = text.strip()
    if not answer:
        reason = _VALIDATION_REASONS["empty_answer"]
        raise OutputValidationError(reason, code="business_validation_error", validation_reason=reason)
    if answer.startswith(("```", "{", "[")):
        reason = _VALIDATION_REASONS["json_or_code_fence_prefix"]
        raise OutputValidationError(reason, code="business_validation_error", validation_reason=reason)
    if allowed is not None:
        unknown = sorted(set(extract_citations(answer)) - set(allowed))
        if unknown:
            reason = _VALIDATION_REASONS["unavailable_citation"]
            raise OutputValidationError(reason, code="business_validation_error", validation_reason=reason)
    return answer


def _json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise SchemaError("response must be a JSON object")
    return parsed


def _plan(text: str) -> Any:
    parsed = _json_object(text)
    if set(parsed) != {"route", "needed_prerequisites"} or parsed["route"] not in {"direct", "decompose"}:
        raise SchemaError("plan must be {route, needed_prerequisites}")
    names = parsed["needed_prerequisites"]
    if not isinstance(names, list) or len(names) > 6 or not all(isinstance(n, str) and n.strip() for n in names):
        raise SchemaError("needed_prerequisites must be <= 6 names")
    return parsed


def _b_plan(text: str) -> Any:
    plan = Step1Plan.from_dict(_json_object(text))
    if plan.planning_mode != "baseline":
        raise SchemaError("arm B step1 planning_mode must be baseline")
    return plan.to_dict()


def _diagnose(text: str) -> Any:
    parsed = _json_object(text)
    actions = parsed.get("actions")
    if set(parsed) != {"actions"} or not isinstance(actions, list) or len(actions) > 3:
        raise SchemaError("diagnose must be {actions: [...<=3]}")
    for row in actions:
        if (not isinstance(row, dict) or row.get("type") not in {"add_gap", "compress_detour", "add_bridge"}
                or not isinstance(row.get("instruction"), str)
                or not row["instruction"].strip()
                or set(row) - {"type", "instruction", "concept_id", "unit_index"}):
            raise SchemaError("diagnose action malformed")
    return parsed


def _critique(text: str) -> Any:
    parsed = _json_object(text)
    issues = parsed.get("issues")
    if set(parsed) != {"issues"} or not isinstance(issues, list) or len(issues) > 3 or not all(
            isinstance(item, str) for item in issues):
        raise SchemaError("critique must be {issues: [<=3 strings]}")
    return parsed


def _judge(text: str) -> Any:
    parsed = _json_object(text)
    scores = parsed.get("scores")
    if set(parsed) != {"scores"} or not isinstance(scores, dict) or set(scores) != set(RUBRIC_ANCHORS):
        raise SchemaError("judge must score all six rubric dimensions")
    if any(type(value) is not int or not 1 <= value <= 5 for value in scores.values()):
        raise SchemaError("rubric scores must be integers 1..5")
    return parsed


def validate_structure(job: ExperimentJob, text: str) -> Any:
    """Stage structure check; raises on failure, returns the parsed value."""
    stage = job.stage
    schema = _job_schema(job)
    if schema is not None:
        try:
            parsed = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise OutputValidationError("response is not strict JSON", code="json_syntax_error") from exc
        try:
            validate_json_schema(parsed, schema)
        except SchemaError as exc:
            message = str(exc)
            reason = ("schema_constraint: " + message if _SAFE_SCHEMA_REASON.fullmatch(message)
                      and len(message) <= 550 else _VALIDATION_REASONS["schema_validation_error"])
            raise OutputValidationError(reason, code="schema_validation_error", validation_reason=reason) from exc
    if operator_protocol.is_operator_job(job):
        if schema is not None:
            if not isinstance(parsed, dict):
                raise OutputValidationError("operator structured output must be an object",
                                            code="schema_validation_error")
            _operator_locatable(job, parsed)
            return parsed
        return _markdown(text, job.arg("allowed_citations") if stage == "answer" else None)
    try:
        if stage in {"answer", "b_answer"}:
            return _markdown(text, job.arg("allowed_citations"))
        if stage == "revise":
            # Structure only: non-empty Markdown, not JSON.  Citation legality,
            # length and kept conditions are candidate checks in
            # ``revision.validate_revision`` and count within the <= 3 calls.
            return _markdown(text, None)
        if stage == "validate":
            return parsed
        if stage == "b_plan":
            return parsed
        if stage == "plan":
            return parsed
        if stage == "ground":
            return validate_grounding_structure(text)
        if stage == "gaps":
            if job.arg("detector") == "D1":
                return validate_d1_structure(text)
            return validate_d3_structure(text, list(job.arg("candidates", ())))
        if stage == "diagnose":
            return parsed
        if stage == "critique":
            return parsed
        if stage == EVAL_STAGE:
            return parsed
    except OutputValidationError:
        raise
    except (json.JSONDecodeError, SchemaError, KeyError, TypeError, ValueError) as exc:
        reason = _safe_business_reason(exc)
        raise OutputValidationError(reason, code="business_validation_error", validation_reason=reason) from exc
    raise ValueError(f"no structure validator for stage {stage!r}")


def request_options() -> dict[str, Any]:
    raw = dict(config.REQUEST_OPTIONS)
    if set(raw) != {"temperature", "seed", "max_output_tokens", "timeout_seconds"}:
        raise ValueError("providers.config request options incomplete")
    return raw


def validate_structure_policy(manifest: Mapping[str, Any]) -> None:
    """Only explicitly registered live runs may change provider retry budgets."""
    if "structure_attempts_by_provider" not in manifest:
        return
    policy = manifest["structure_attempts_by_provider"]
    if (not operator_protocol.is_operator_identity(manifest) or manifest.get("split") != "live"
            or not isinstance(policy, dict) or set(policy) != {"mimo", "minimax"}
            or type(policy["mimo"]) is not int or policy["mimo"] != 3
            or type(policy["minimax"]) is not int or not 1 <= policy["minimax"] <= 12):
        raise ConflictError("Custom structure budget requires operator/live, MiMo=3 and MiniMax in 1..12")


def structure_limit(manifest: Mapping[str, Any], stage: str, provider: str) -> int:
    validate_structure_policy(manifest)
    if stage == "revise":
        return 1
    return manifest.get("structure_attempts_by_provider", {}).get(provider, structure_attempts(stage))


def _quota_directory(run: AppendOnlyRun):
    # Main operator runs share one stop across both generator phase directories.
    # Standalone research hosts keep the same protection within their ledger.
    root = run.root
    if (operator_protocol.is_operator_identity(run.identity) and len(root.parents) >= 3
            and root.parents[1].name == operator_protocol.DIRECTORY
            and root.parents[2].name == run.identity.get("run_id")):
        root = root.parents[2]
    return root / "provider_quota_stops"


def _saved_quota_stop(run: AppendOnlyRun, router: RouterLike) -> None:
    for path in sorted(_quota_directory(run).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf8"))
        body = {key: value for key, value in row.items() if key != "stop_hash"}
        if (row.get("version") != 1 or row.get("code") != "provider_quota_exhausted"
                or row.get("run_id") != run.identity.get("run_id")
                or row.get("provider") not in config.EXPERIMENT_PROVIDER_ORDER
                or row.get("stop_hash") != digest(body)):
            raise ConflictError("Provider quota stop record differs or is corrupt")
        pool = getattr(router, "pool", None)
        if pool is not None:
            pool.stop_for_quota(row["provider"])
        raise TransportStopped("Run stopped after provider quota exhaustion", code=row["code"])


def _save_quota_stop(run: AppendOnlyRun, entry: Prepared, request_hash: str) -> None:
    body = {"version": 1, "code": "provider_quota_exhausted", "run_id": run.identity.get("run_id"),
            "provider": entry.request["model_request"]["provider"], "key": entry.key,
            "request_hash": request_hash, "generator": run.identity.get("generator")}
    write_once(_quota_directory(run) / f"{entry.key}.json", {**body, "stop_hash": digest(body)})


@dataclass(frozen=True)
class Prepared:
    job: ExperimentJob
    key: str
    request: dict[str, Any]


def success_key(run: AppendOnlyRun, job: ExperimentJob, model_request: Mapping[str, Any]) -> str:
    if operator_protocol.is_operator_identity(run.identity):
        return digest({"protocol": operator_protocol.PROTOCOL,
                       "execution_identity": operator_protocol.execution_identity(run.identity),
                       "qid": job.qid, "split": job.split,
                       "arm": job.arm if job.stage == EVAL_STAGE else f"{job.base}:{job.arm}",
                       "stage": job.stage, "model_request": dict(model_request),
                       "template_hash": job.template_hash, "static_payload_hash": job.static_payload_hash,
                       "parent_answer_hash": job.parent_hash, "rubric_hash": job.rubric_hash,
                       "method_hash": job.arg("method_hash"), "contract_hash": job.arg("contract_hash")})
    return digest({"protocol": PROTOCOL, "qid": job.qid, "split": job.split,
                   "arm": job.arm if job.stage == EVAL_STAGE else f"{job.base}:{job.arm}",
                   "stage": job.stage, "model_request": dict(model_request),
                   "template_hash": job.template_hash, "static_payload_hash": job.static_payload_hash,
                   "parent_answer_hash": job.parent_hash, "calibration_hash": job.calibration_hash,
                   "rubric_hash": job.rubric_hash})


def check_parent(job: ExperimentJob, parent_index: Mapping[tuple[str, str, str], str] | None) -> None:
    """F/M-* must descend from the same-question same-base L initial answer, SR from B."""
    parents = {**PARENT_ARM, **{arm: "L" for arm in operator_protocol.RESEARCH_STAGES}}
    if job.arm not in parents:
        return
    if parent_index is None:
        raise ValueError(f"{job.arm} jobs need the initial-answer parent index")
    expected = parent_index.get((job.qid, job.base, parents[job.arm]))
    if expected is None or job.parent_hash != expected:
        raise ValueError(f"{job.qid}/{job.base}/{job.arm}: parent hash is not the "
                         f"{parents[job.arm]} initial answer; refusing to schedule")


def prepare(run: AppendOnlyRun, generator: str, jobs: Iterable[ExperimentJob], *,
            question_splits: Mapping[str, str], phase_split: str,
            parent_index: Mapping[tuple[str, str, str], str] | None = None,
            _model_identity: Mapping[str, Any] | None = None) -> tuple[list[Prepared], dict[ExperimentJob, str]]:
    """Validate identity/whitelist/question-set membership and derive keys."""
    options = request_options()
    identity = ((model_config_identity(generator, mimo_batch=True) if run.identity.get("mimo_transport") == "batch"
                 else model_config_identity(generator)) if _model_identity is None else _model_identity)
    manifest = run.identity
    validate_structure_policy(manifest)
    operator = operator_protocol.is_operator_identity(manifest)
    expected_protocol = operator_protocol.PROTOCOL if operator else PROTOCOL
    expected_split = operator_protocol.source_split(manifest["split"]) if operator else manifest.get("split")
    if (manifest.get("protocol_version") != expected_protocol
            or manifest.get("generator") != config.experiment_role_models_for(generator)["student1"]
            or manifest.get("model_config_hash") != digest(identity)):
        raise ConflictError("run manifest protocol/generator/model configuration differs")
    if operator and (manifest.get("method_version") != operator_protocol.METHOD_VERSION
                     or manifest.get("question_split") != expected_split or phase_split != expected_split):
        raise ConflictError("operator live/test method or question scope differs")
    if operator and manifest.get("origin", "REAL") not in {"REAL", "OFFLINE_FIXTURE"}:
        raise ConflictError("unknown operator artifact origin")
    prepared: dict[str, Prepared] = {}
    key_by_job: dict[ExperimentJob, str] = {}
    selected = manifest.get("selected_qids") if manifest.get("selection_mode") == "explicit_qids" else None
    if selected is not None and (not operator or manifest.get("split") != "live"
            or not isinstance(selected, list) or not selected or selected != manifest.get("qids")
            or any(not isinstance(qid, str) or not qid for qid in selected)
            or len(selected) != len(set(selected))):
        raise ConflictError("Explicit live selection differs from the locked question scope")
    if manifest.get("selection_mode") == "explicit_qids" and selected is None:
        raise ConflictError("Explicit live selection is missing its locked question scope")
    for job in jobs:
        if selected is not None and job.qid not in selected:
            raise ConflictError("Job is outside the explicit live question selection")
        if operator != operator_protocol.is_operator_job(job):
            raise ConflictError("operator and historical request identities cannot be mixed")
        if job.arg("operator_protocol") not in {None, operator_protocol.PROTOCOL}:
            raise ConflictError("unknown operator request protocol")
        split = question_splits.get(job.qid)
        if split is None:
            raise ValueError(f"qid {job.qid!r} is outside the loaded question set")
        if job.split != split or split != phase_split:
            raise ValueError(f"qid {job.qid!r} split {split!r} is not the {phase_split} phase split")
        research = operator and "research_contract" in manifest
        if research:
            operator_protocol.check_research_job(manifest, job.arm, job.stage, job.role)
        else:
            (operator_protocol.check_arm_stage if operator else check_arm_stage)(job.arm, job.stage, split)
        if operator and job.stage in {"revise", "validate"} and (type(job.call_index) is not int or job.call_index != 1):
            raise ValueError("operator protocol permits one revision candidate and one validation job")
        check_parent(job, parent_index)
        if not research and job.role not in (operator_protocol.role_for_stage if operator else role_for_stage)(job.stage):
            raise ValueError(f"role {job.role!r} cannot run stage {job.stage!r}")
        if job.stage != EVAL_STAGE and job.base not in {"pe", "rag"}:
            raise ValueError("generation jobs need base pe or rag")
        if job.rubric_hash != manifest.get("rubric_hash"):
            raise ConflictError("job rubric differs from the run manifest")
        if job.template_hash not in set(manifest.get("template_hashes", {}).values()):
            raise ConflictError("job template hash is not registered in the run manifest")
        if operator and job.static_payload_hash not in set(manifest.get("static_payload_hashes", {}).values()):
            raise ConflictError("operator static contract hash is not registered in the run manifest")
        schema = _job_schema(job)
        if operator:
            _check_operator_schema(run, job, schema)
        output_mode = config.structured_output_mode(identity[job.role]["model"]) if schema is not None else "markdown"
        prompt = job.prompt if operator else job.prompt + (structured_output_instruction(schema) if schema is not None else "")
        model_request = {**identity[job.role], "temperature": options["temperature"],
                         "seed": options["seed"], "max_tokens": options["max_output_tokens"],
                         "timeout": options["timeout_seconds"], "call_index": job.call_index,
                         "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                         "output_mode": output_mode, "schema_hash": schema_hash(schema) if schema is not None else None,
                         "schema": schema, "structured_output_protocol": config.STRUCTURED_OUTPUT_PROTOCOL}
        key = success_key(run, job, model_request)
        request = {"model_request": model_request, "prompt": prompt}
        entry = Prepared(job=job, key=key, request=request)
        if key in prepared and canonical_bytes(prepared[key].request) != canonical_bytes(request):
            raise ConflictError("same success key has conflicting request content")
        prepared.setdefault(key, entry)
        key_by_job[job] = key
    return sorted(prepared.values(), key=lambda row: row.key), key_by_job


def _settle(response: Any, records: list[Any], entry: Prepared) -> dict[str, Any]:
    text = getattr(response, "text", None)
    usage = getattr(response, "usage", None)
    if not isinstance(text, str):
        raise RuntimeError("unknown provider response; no success saved")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                            for k in ("prompt_tokens", "completion_tokens")):
        raise RuntimeError("provider usage unknown; no success saved")
    if len(records) != 1:
        raise RuntimeError("hidden provider retry or unknown call count")
    record = records[0].to_dict() if hasattr(records[0], "to_dict") else dict(records[0])
    model_request = entry.request["model_request"]
    if (record.get("status") != "ok" or record.get("role") != entry.job.role
            or record.get("model") != model_request["model"]
            or record.get("provider") != model_request["provider"]):
        raise RuntimeError("provider completion status/identity unknown")
    return {"text": text,
            "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens")},
            "call": {k: record[k] for k in ("role", "provider", "model", "status", "attempt",
                                            "credential_slot", "latency_ms", "prompt_tokens", "completion_tokens",
                                            "transport", "batch_id", "custom_id")
                     if k in record}}


def _save_telemetry(run: AppendOnlyRun, entry: Prepared, attempt: int | None, request_hash: str,
                    *, result: str, start: float, settled: dict | None = None) -> None:
    """Separate append-only measurements preserve the historical journal schema."""
    if attempt is None or not operator_protocol.is_operator_identity(run.identity):
        return
    model = entry.request["model_request"]
    provider_latency = settled.get("call", {}).get("latency_ms") if settled else None
    fixture = run.identity.get("origin") == "OFFLINE_FIXTURE"
    elapsed = (provider_latency if provider_latency is not None else 0.0) if fixture else round((time.monotonic() - start) * 1000, 3)
    body = {"version": 1, "protocol": operator_protocol.PROTOCOL, "key": entry.key,
            "request_hash": request_hash, "attempt": attempt, "result": result,
            "qid": entry.job.qid, "base": entry.job.base, "arm": entry.job.arm,
            "stage": entry.job.stage, "role": entry.job.role,
            "model": model["model"], "provider": model["provider"],
            "usage": settled["usage"] if settled else None,
            "elapsed_ms": elapsed, "timing_source": "synthetic_fixture" if fixture else "monotonic_with_admission",
            "provider_latency_ms": provider_latency,
            "cost": None, "cost_status": "price_unavailable"}
    write_once(run.root / "telemetry" / entry.key / f"{attempt:04d}.json",
               {**body, "telemetry_hash": digest(body)})


def validate_completion(response: Any) -> None:
    """A HTTP-successful refusal or truncated answer is never a saved success."""
    raw = getattr(response, "raw", None) or {}
    choice = (raw.get("choices") or [{}])[0]
    finish = getattr(response, "finish_reason", None) or choice.get("finish_reason")
    if getattr(response, "refusal", False) or choice.get("message", {}).get("refusal") or finish == "content_filter":
        raise OutputValidationError("provider refused the response", code="refusal")
    if finish == "length":
        raise OutputValidationError("provider truncated the response", code="output_truncated")
    if finish not in {None, "stop"}:
        raise LLMTransportError("unexpected completion finish reason", code="provider_protocol_error")


class TransportStopped(RuntimeError):
    def __init__(self, message: str, *, code: str = "transport_error", retry_at: float | None = None,
                 counts: dict[str, int] | None = None):
        super().__init__(message)
        self.code = code
        self.retry_at = retry_at
        self.counts = dict(counts or {})


async def _run_entries(run: AppendOnlyRun, entries: list[Prepared], router: RouterLike,
                       origin: str, max_in_flight: int) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    options = request_options()
    results: dict[str, dict[str, Any]] = {}
    counts = {"calls": 0, "reused": 0, "structure_failures": 0, "exhausted": 0,
              "transport_failures": 0, "unknown": 0}
    errors: list[BaseException] = []
    stop = asyncio.Event()

    async def one(entry: Prepared) -> None:
        request_hash = digest(entry.request)
        model_request = entry.request["model_request"]
        limit = structure_limit(run.identity, entry.job.stage, model_request["provider"])
        existing = run.success(entry.key)
        rows = run.attempts(entry.key, request_hash)
        terminal = run.terminal(entry.key, request_hash)
        if ((terminal and terminal.get("code") == "provider_quota_exhausted")
                or any(row["outcome"] is not None and row["outcome"].get("code") == "provider_quota_exhausted"
                       for row in rows)):
            _save_quota_stop(run, entry, request_hash)
            _saved_quota_stop(run, router)
        if existing is not None:
            if (existing.get("key") != entry.key or existing.get("payload", {}).get("origin") != origin
                    or existing.get("request_hash") != request_hash
                    or existing.get("payload_hash") != digest(existing.get("payload")) or terminal):
                raise ConflictError("Existing success has conflicting/corrupt content")
            if rows and rows[-1]["outcome"] is None:
                number = rows[-1]["reservation"]["attempt"]
                if existing["payload"].get("attempt_number") != number:
                    raise ConflictError("Success does not belong to the unresolved attempt")
                run.finish_attempt(entry.key, request_hash, number, result="success", code="success",
                                   usage=existing["payload"]["usage"])
            elif rows and rows[-1]["outcome"]["result"] != "success":
                raise ConflictError("Success contradicts the attempt journal")
            results[entry.key] = {"status": "reused", "payload": existing["payload"]}
            counts["reused"] += 1
            return
        resume_number = None
        if rows and rows[-1]["outcome"] is None and getattr(router, "pending_attempt", lambda *args: False)(
                entry.key, request_hash, rows[-1]["reservation"]["attempt"]):
            resume_number = rows[-1]["reservation"]["attempt"]
            rows = rows[:-1]
        if rows and rows[-1]["outcome"] is None:
            number = rows[-1]["reservation"]["attempt"]
            run.finish_attempt(entry.key, request_hash, number, result="unknown", code="unknown_result",
                               retryable=True)
            counts["unknown"] += 1
            rows = run.attempts(entry.key, request_hash)
        if any(row["outcome"]["result"] == "success" for row in rows):
            raise ConflictError("Journal success has no matching saved response")
        if terminal:
            if terminal["status"] == "structure_exhausted":
                if sum(r["outcome"]["result"] == "structure_error" for r in rows) < limit:
                    raise ConflictError("Structure terminal state contradicts the attempt journal")
                results[entry.key] = {"status": "exhausted", "payload": None}
                counts["exhausted"] += 1
                return
            if terminal["status"] == "transport_exhausted" and sum(
                    r["outcome"]["result"] in {"transport_error", "unknown"} for r in rows) < config.TRANSPORT_MAX_FAILURES:
                raise ConflictError("Transport terminal state contradicts the attempt journal")
            if terminal["status"] == "permanent_error" and (not rows
                    or rows[-1]["outcome"]["result"] != "transport_error" or rows[-1]["outcome"]["retryable"]):
                raise ConflictError("Permanent terminal state contradicts the attempt journal")
            raise TransportStopped("Request stopped permanently; choose a new run after fixing its cause",
                                   code=terminal["code"])
        structural = sum(row["outcome"]["result"] == "structure_error" for row in rows)
        failed_transport = sum(row["outcome"]["result"] in {"transport_error", "unknown"} for row in rows)
        reason = next((row["outcome"].get("validation_reason") or row["outcome"]["code"] for row in reversed(rows)
                       if row["outcome"]["result"] == "structure_error"), None)
        last = rows[-1]["outcome"] if rows else None
        restored_delay = bool(rows)
        while not stop.is_set():
            if structural >= limit:
                run.put_terminal(entry.key, request_hash, "structure_exhausted", "structure_exhausted")
                results[entry.key] = {"status": "exhausted", "payload": None}
                counts["exhausted"] += 1
                return
            if last and last["result"] == "transport_error" and not last["retryable"]:
                run.put_terminal(entry.key, request_hash, "permanent_error", last["code"])
                raise TransportStopped("Non-retryable provider/protocol error", code=last["code"])
            if failed_transport >= config.TRANSPORT_MAX_FAILURES:
                run.put_terminal(entry.key, request_hash, "transport_exhausted", "transport_exhausted")
                raise TransportStopped("Transport retry budget exhausted", code="transport_exhausted")
            retry_at = last.get("retry_at") if last else None
            if retry_at is not None:
                delay = max(0.0, retry_at - time.time())
                if delay and (restored_delay or delay > config.TRANSPORT_MAX_WAIT_SECONDS):
                    raise TransportStopped("Provider requested a later retry; restart at or after retry_at",
                                           code="retry_deferred", retry_at=retry_at)
                if delay:
                    await asyncio.sleep(delay)
                    if stop.is_set():
                        return
            restored_delay = False
            prompt = entry.request["prompt"]
            if reason:
                prompt += ("\n" + entry.job.arg("repair_instruction") + "\n" + reason + "\n"
                           if operator_protocol.is_operator_job(entry.job) else REPAIR_NOTE.format(reason=reason))
            number = None

            def before_send():
                nonlocal number, resume_number
                if stop.is_set():
                    raise TransportStopped("Transport wave already stopped", code="transport_stopped")
                _saved_quota_stop(run, router)
                if number is not None:
                    raise RuntimeError("duplicate send reservation")
                is_resume = resume_number is not None
                number = resume_number if is_resume else run.reserve_attempt(entry.key, request_hash)
                resume_number = None
                if not is_resume:
                    counts["calls"] += 1

            use_admission = getattr(router, "uses_send_admission", lambda role: False)(entry.job.role)
            if not use_admission:
                # Existing providers retain resume-v3 semantics and identity.
                before_send()
            settled = None
            started = time.monotonic()
            try:
                response, records = await router.complete(
                    role=entry.job.role, prompt=prompt, temperature=options["temperature"],
                    max_tokens=options["max_output_tokens"], seed=options["seed"],
                    timeout=options["timeout_seconds"], output_schema=model_request["schema"],
                    schema_name=(entry.job.arg("schema_name", entry.job.stage)
                                 if operator_protocol.is_operator_job(entry.job) else
                                 entry.job.stage + ("_" + entry.job.arg("detector", "") if entry.job.stage == "gaps" else "")),
                    **({"before_send": before_send} if use_admission else {}),
                    **({"batch_request": {"key": entry.key, "request_hash": request_hash,
                                          "attempt": resume_number or len(run.attempts(entry.key, request_hash)) + 1}}
                       if getattr(router, "mimo_batch", False) and model_request["provider"] == "mimo" else {}))
                if number is None:
                    raise RuntimeError("completion without a send reservation")
                settled = _settle(response, records, entry)
                validate_completion(response)
                parsed = validate_structure(entry.job, settled["text"])
            except OutputValidationError as exc:
                _save_telemetry(run, entry, number, request_hash, result="structure_error",
                                start=started, settled=settled)
                structural += 1
                counts["structure_failures"] += 1
                reason = exc.validation_reason
                run.finish_attempt(entry.key, request_hash, number, result="structure_error", code=exc.code,
                                   usage=settled["usage"] if settled else None, validation_reason=reason)
                run.put_failure(entry.key, request_hash, {"attempt": number, "code": exc.code,
                    "validation_reason": reason, "qid": entry.job.qid, "base": entry.job.base,
                    "arm": entry.job.arm, "stage": entry.job.stage,
                    "usage": settled["usage"] if settled else None,
                    "output_mode": model_request["output_mode"], "schema_hash": model_request["schema_hash"],
                    **({"response": settled, "response_hash": digest(settled)}
                       if settled is not None and operator_protocol.is_operator_job(entry.job) else {})})
                last = None
                continue
            except Exception as exc:
                from nea.runtime.batch_transport import BatchDeferred
                if isinstance(exc, BatchDeferred):
                    if exc.code == "provider_quota_exhausted":
                        _save_quota_stop(run, entry, request_hash)
                    raise
                if getattr(router, "mimo_batch", False) and model_request["provider"] == "mimo":
                    usage = getattr(exc, "details", {}).get("usage", {})
                    if all(type(usage.get(field)) is int and usage[field] >= 0
                           for field in ("prompt_tokens", "completion_tokens")):
                        settled = {"usage": {field: usage[field] for field in ("prompt_tokens", "completion_tokens")}}
                if number is None:
                    # Admission or journal write failed before HTTP submission.
                    # Do not manufacture a paid attempt or transport failure.
                    if getattr(exc, "code", None) == "provider_quota_exhausted":
                        if getattr(router, "mimo_batch", False):
                            _save_quota_stop(run, entry, request_hash)
                        raise TransportStopped("Run stopped before send after provider quota exhaustion",
                                               code="provider_quota_exhausted") from None
                    raise
                code = getattr(exc, "code", "transport_error")
                if code == "provider_quota_exhausted":
                    # Persist the shared stop before journal settlement so a
                    # restart cannot lose an already observed quota result.
                    _save_quota_stop(run, entry, request_hash)
                _save_telemetry(run, entry, number, request_hash, result="transport_error",
                                start=started, settled=settled)
                failed_transport += 1
                counts["transport_failures"] += 1
                details = getattr(exc, "details", {})
                status = details.get("http_status")
                retryable = isinstance(exc, LLMTransportError) and (
                    code in {"connection_error", "timeout"}
                    or code == "http_status_error" and (status == 429 or type(status) is int and 500 <= status <= 599))
                delay = details.get("retry_after_seconds")
                if delay is None:
                    delay = config.TRANSPORT_BACKOFF_SECONDS[min(failed_transport - 1,
                                                                 len(config.TRANSPORT_BACKOFF_SECONDS) - 1)]
                retry_at = time.time() + delay if retryable else None
                run.finish_attempt(entry.key, request_hash, number, result="transport_error", code=code,
                                   usage=settled["usage"] if settled else None,
                                   retryable=retryable, retry_at=retry_at)
                run.put_failure(entry.key, request_hash, {"attempt": number, "code": code,
                    "qid": entry.job.qid, "base": entry.job.base, "arm": entry.job.arm, "stage": entry.job.stage,
                    "http_status": status, "retry_at": retry_at,
                    "output_mode": model_request["output_mode"], "schema_hash": model_request["schema_hash"]})
                if code == "provider_quota_exhausted":
                    run.put_terminal(entry.key, request_hash, "permanent_error", code)
                    raise TransportStopped("Run stopped after provider quota exhaustion", code=code)
                last = {"result": "transport_error", "retryable": retryable, "retry_at": retry_at, "code": code}
                continue
            _save_telemetry(run, entry, number, request_hash, result="success", start=started, settled=settled)
            payload = {**settled, "parsed": parsed, "attempt": structural + 1, "attempt_number": number,
                       "origin": origin, "output_mode": model_request["output_mode"],
                       "schema_hash": model_request["schema_hash"],
                       "sent_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       "qid": entry.job.qid, "base": entry.job.base, "arm": entry.job.arm,
                       "stage": entry.job.stage, "role": entry.job.role, "call_index": entry.job.call_index}
            run.put_success(entry.key, request_hash, payload)
            run.finish_attempt(entry.key, request_hash, number, result="success", code="success", usage=settled["usage"])
            results[entry.key] = {"status": "completed", "payload": payload}
            return

    async def worker(queue: asyncio.Queue) -> None:
        while not stop.is_set():
            try:
                entry = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await one(entry)
            except Exception as exc:
                errors.append(exc)
                if (getattr(router, "resume_only", False)
                        and getattr(exc, "code", None) in {"batch_pending", "batch_submission_unknown"}):
                    continue
                stop.set()
                return

    # Provider lanes prevent workers waiting for one busy provider from blocking
    # ready jobs on another. The shared router owns the actual global key caps.
    limits = getattr(router, "provider_limits", None)
    lanes: dict[str, asyncio.Queue] = {}
    for entry in entries:
        provider = entry.request["model_request"]["provider"] if limits and (max_in_flight > 1 or getattr(router, "mimo_batch", False)) else "all"
        lanes.setdefault(provider, asyncio.Queue()).put_nowait(entry)
    workers = [worker(queue) for provider, queue in lanes.items()
               for _ in range(min(queue.qsize(), limits[provider] if provider != "all" else max_in_flight))]
    await asyncio.gather(*workers)
    if errors:
        error = errors[0]
        raise TransportStopped(f"Transport stopped: {getattr(error, 'code', type(error).__name__)}",
                               code=getattr(error, "code", "transport_error"),
                               retry_at=getattr(error, "retry_at", None), counts=counts) from error
    return results, counts


def execute(run: AppendOnlyRun, generator: str, jobs: Iterable[ExperimentJob], *, router: RouterLike,
            question_splits: Mapping[str, str], phase_split: str,
            parent_index: Mapping[tuple[str, str, str], str] | None = None,
            max_in_flight: int = MAX_IN_FLIGHT) -> dict[str, Any]:
    """Run one wave of jobs; returns ``{key: payload|None}`` plus counters."""
    if type(max_in_flight) is not int or not 1 <= max_in_flight <= config.EXPERIMENT_MAX_IN_FLIGHT:
        raise ValueError(f"max_in_flight must be within 1..{config.EXPERIMENT_MAX_IN_FLIGHT} (the configured need-aware cap)")
    entries, key_by_job = prepare(run, generator, jobs, question_splits=question_splits,
                                  phase_split=phase_split, parent_index=parent_index,
                                  _model_identity=getattr(router, "model_identity", None))
    _saved_quota_stop(run, router)
    origin = run.identity.get("origin", "REAL") if operator_protocol.is_operator_identity(run.identity) else "REAL"
    if not entries:
        return {"results": {}, "key_by_job": {}, "counts": {"calls": 0, "reused": 0,
                                                            "structure_failures": 0, "exhausted": 0}}
    execution_lock = run.root.with_name(run.root.name + "-transport-execute")
    with RunDirectoryLock(execution_lock):
        if getattr(router, "mimo_batch", False) or getattr(router, "resume_mimo_batches", False):
            from nea.runtime.batch_transport import BatchWaveRouter
            router = BatchWaveRouter(router, run)
        results, counts = asyncio.run(_run_entries(run, entries, router, origin, max_in_flight))
    return {"results": {key: row["payload"] for key, row in results.items()},
            "key_by_job": key_by_job, "counts": counts}


def job_key(run: AppendOnlyRun, generator: str, job: ExperimentJob) -> str:
    """Recompute one job's success key (used for resume audits)."""
    return prepare(run, generator, [job], question_splits={job.qid: job.split},
                   phase_split=job.split)[1][job]


__all__ = ["ExperimentJob", "execute", "prepare", "validate_structure", "job_key", "TransportStopped"]
