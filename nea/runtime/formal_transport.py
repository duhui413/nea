"""Eight-arm transport using the existing append-only journal and provider pool.

The bounded send loop is versioned separately so importing this protocol does
not change old method hashes, retry policies, or request identities.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from typing import Any, Iterable, Mapping

from nea.pipeline.artifacts import RunDirectoryLock
from nea.providers import config
from nea.runtime import formal_recovery as recovery
from nea.providers.client import LLMTransportError
from nea.runtime import formal_protocol as operator_protocol
from nea.runtime.identity import model_config_identity
from nea.runtime.records import AppendOnlyRun, ConflictError, canonical_bytes, digest, write_once
from nea.runtime.structured import validate_json_schema, schema_hash
from nea.runtime.transport import (
    ExperimentJob, Prepared, RouterLike, TransportStopped, OutputValidationError,
    _markdown, _operator_locatable, _settle, validate_completion, request_options,
    _VALIDATION_REASONS, _SAFE_SCHEMA_REASON)
from nea.schemas import SchemaError

MAX_IN_FLIGHT = 2


def _job_schema(job: ExperimentJob) -> dict | None:
    if not operator_protocol.is_operator_job(job):
        raise ConflictError("Only formal protocol jobs may enter the formal transport")
    schema = job.arg("output_schema")
    schema = json.loads(schema) if isinstance(schema, str) else schema
    if job.stage in operator_protocol.MARKDOWN_STAGES:
        if schema is not None:
            raise ConflictError("Markdown stage has a structured output schema")
    elif not isinstance(schema, dict):
        raise ConflictError("Structured stage needs an explicit schema")
    return schema


def _check_schema(run: AppendOnlyRun, job: ExperimentJob, schema: dict | None) -> None:
    from nea.pipeline.formal_workflow import role_schema
    role = job.arg("schema_name")
    if operator_protocol.ROLE_STAGE.get(role) != job.stage:
        raise ConflictError("formal role/stage mismatch")
    if run.identity.get("schema_hashes", {}).get(role) != digest(role_schema(role, {})):
        raise ConflictError("formal schema not bound to manifest")
    try:
        payload = json.JSONDecoder().raw_decode(job.prompt.split("## 本题输入\n", 1)[1].lstrip())[0]
    except (ValueError, IndexError) as exc:
        raise ConflictError("formal prompt has no canonical payload") from exc
    if canonical_bytes(schema) != canonical_bytes(role_schema(role, payload)):
        raise ConflictError("formal concrete schema differs from registered role")
    chunks = tuple(item["chunk_id"] for item in payload.get("knowledge_chunks", []))
    validation_inputs = {"locatable_answer": payload.get("candidate_answer", payload.get("answer", "")),
        "locatable_parent": payload.get("answer", "") if role == "revision_validation" else "",
        "locatable_question": payload.get("question", ""), "allowed_evidence_ids": chunks,
        "allowed_citations": chunks,
        "measurement_concept_ids": tuple(item.get("concept_id", item.get("id")) for item in payload.get("concepts", [])),
        "verification_question_ids": tuple(item["question_id"] for item in payload.get("verification_questions", []))}
    if role == "formal_perturbation_rewrite":
        validation_inputs.update(perturbation_kind=payload.get("kind"), duplicate_paragraph=payload.get("duplicate_paragraph", ""))
    if any(canonical_bytes(job.arg(key)) != canonical_bytes(value) for key, value in validation_inputs.items()):
        raise ConflictError("formal local validation inputs differ from the bound prompt")
    if run.identity.get("template_hashes", {}).get(role) != job.template_hash:
        raise ConflictError("formal template not bound to manifest")
    if run.identity.get("static_payload_hashes", {}).get(role) != job.static_payload_hash:
        raise ConflictError("formal static payload not bound to manifest")
    repair = job.arg("repair_instruction")
    if not isinstance(repair, str) or run.identity.get("shared_template_hashes", {}).get("repair_output") != digest(repair.encode("utf8")):
        raise ConflictError("formal repair instruction not bound to manifest")


def validate_structure(job: ExperimentJob, text: str) -> Any:
    schema = _job_schema(job)
    if schema is None:
        return _markdown(text, job.arg("allowed_citations") if job.stage == "answer" else None)
    try:
        parsed = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError) as exc:
        raise OutputValidationError("response is not strict JSON", code="json_syntax_error") from exc
    try:
        validate_json_schema(parsed, schema)
    except SchemaError as exc:
        message = str(exc)
        reason = ("schema_constraint: " + message if _SAFE_SCHEMA_REASON.fullmatch(message)
                  and len(message) <= 550 else _VALIDATION_REASONS["schema_validation_error"])
        raise OutputValidationError(reason, code="schema_validation_error", validation_reason=reason) from exc
    if not isinstance(parsed, dict):
        raise OutputValidationError("structured result is not an object", code="schema_validation_error")
    if job.stage != "judge":
        _operator_locatable(job, parsed)
    try:
        from nea.evaluation.formal_measurement import validate_formal_payload
        return validate_formal_payload(job, parsed)
    except (SchemaError, ValueError, KeyError, TypeError) as exc:
        raise OutputValidationError("formal result violates its role contract", code="business_validation_error") from exc


def structure_limit(manifest: Mapping[str, Any], stage: str, provider: str) -> int:
    # One candidate is a method budget; malformed revisions never buy another.
    return 1 if stage == "revise" else 3


def check_parent(job: ExperimentJob, parent_index: Mapping | None) -> None:
    parent = operator_protocol.PARENTS.get(job.arm)
    if parent is None:
        return
    key = (job.qid, job.base, parent, job.arg("comparison", "main"), job.arg("round_index", 1))
    expected = None if parent_index is None else parent_index.get(key, parent_index.get((job.qid, job.base, parent)))
    if job.parent_hash != expected:
        raise ConflictError("formal request has the wrong same-cell parent answer")


def success_key(run: AppendOnlyRun, job: ExperimentJob, model_request: Mapping) -> str:
    return digest({"protocol": operator_protocol.PROTOCOL,
        "execution_identity": operator_protocol.execution_identity(run.identity),
        "qid": job.qid, "split": job.split, "arm": job.arm,
        "base": job.base, "stage": job.stage, "model_request": dict(model_request),
        "template_hash": job.template_hash, "static_payload_hash": job.static_payload_hash,
        "parent_answer_hash": job.parent_hash, "rubric_hash": job.rubric_hash,
        "comparison": job.arg("comparison", "main"), "round_index": job.arg("round_index", 1)})


def prepare(run: AppendOnlyRun, generator: str, jobs: Iterable[ExperimentJob], *,
            question_splits: Mapping[str, str], phase_split: str,
            parent_index: Mapping | None = None, _model_identity: Mapping | None = None):
    options = request_options()
    identity = ((model_config_identity(generator, mimo_batch=True) if run.identity.get("mimo_transport") == "batch"
                 else model_config_identity(generator)) if _model_identity is None else _model_identity)
    manifest = run.identity
    if (not operator_protocol.is_operator_identity(manifest)
            or manifest.get("method_version") != operator_protocol.METHOD_VERSION
            or manifest.get("generator") != config.experiment_role_models_for(generator)["student1"]
            or manifest.get("model_config_hash") != digest(identity)
            or manifest.get("question_split") != phase_split
            or operator_protocol.source_split(manifest.get("split")) != phase_split
            or manifest.get("origin") not in {"REAL", "OFFLINE_FIXTURE"}):
        raise ConflictError("formal manifest protocol/model/scope differs")
    from nea.pipeline.formal_workflow import ABLATIONS, SENSITIVITIES
    prepared, key_by_job = {}, {}
    for job in jobs:
        if (not operator_protocol.is_operator_job(job) or job.qid not in manifest.get("qids", ())
                or question_splits.get(job.qid) != phase_split or job.split != phase_split):
            raise ConflictError("formal request is outside the locked scope")
        operator_protocol.check_arm_stage(job.arm, job.stage, job.split)
        if job.role not in operator_protocol.role_for_stage(job.stage):
            raise ConflictError("formal role not permitted for stage")
        comparison = job.arg("comparison", "main")
        if comparison != "main":
            family, _, name = comparison.partition(":")
            if (family == "ablation" and name in ABLATIONS):
                pass
            elif family == "sensitivity" and name in SENSITIVITIES and phase_split == "dev":
                pass
            else:
                raise ConflictError("unregistered comparison or test budget variant")
        maximum_round = 2 if comparison == "sensitivity:rounds_2" else 1
        if type(job.call_index) is not int or not 1 <= job.call_index <= maximum_round:
            raise ConflictError("formal revision round exceeds registered budget")
        check_parent(job, parent_index)
        if (job.base not in ({"pe", "rag", "-"} if job.arm == "EVAL" else {"pe", "rag"})
                or job.rubric_hash != manifest.get("rubric_hash")):
            raise ConflictError("formal host/rubric differs")
        schema = _job_schema(job)
        _check_schema(run, job, schema)
        model_request = {**identity[job.role], "temperature": options["temperature"],
            "seed": options["seed"], "max_tokens": options["max_output_tokens"],
            "timeout": options["timeout_seconds"], "call_index": job.call_index,
            "prompt_sha256": hashlib.sha256(job.prompt.encode("utf8")).hexdigest(),
            "output_mode": config.structured_output_mode(identity[job.role]["model"]) if schema is not None else "markdown",
            "schema_hash": schema_hash(schema) if schema is not None else None,
            "schema": schema, "structured_output_protocol": config.STRUCTURED_OUTPUT_PROTOCOL}
        key = success_key(run, job, model_request)
        entry = Prepared(job=job, key=key, request={"model_request": model_request, "prompt": job.prompt})
        if key in prepared and canonical_bytes(prepared[key].request) != canonical_bytes(entry.request):
            raise ConflictError("conflicting formal request identity")
        prepared.setdefault(key, entry)
        key_by_job[job] = key
    return sorted(prepared.values(), key=lambda entry: entry.key), key_by_job


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
    if recovery.enabled(run):
        retry_at = recovery.retry_at(run)
        if retry_at > time.time():
            raise TransportStopped("Registered one-hour cooldown", code="retry_deferred", retry_at=retry_at)
        return
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
            "comparison": entry.job.arg("comparison", "main"), "round_index": entry.job.arg("round_index", 1),
            "model": model["model"], "provider": model["provider"],
            "usage": settled["usage"] if settled else None,
            "elapsed_ms": elapsed, "timing_source": "synthetic_fixture" if fixture else "monotonic_with_admission",
            "provider_latency_ms": provider_latency,
            "cost": None, "cost_status": "price_unavailable"}
    write_once(run.root / "telemetry" / entry.key / f"{attempt:04d}.json",
               {**body, "telemetry_hash": digest(body)})




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
        recovering = recovery.enabled(run)
        if not recovering and ((terminal and terminal.get("code") == "provider_quota_exhausted")
                or any(row["outcome"] is not None and row["outcome"].get("code") == "provider_quota_exhausted"
                       for row in rows)):
            _save_quota_stop(run, entry, request_hash)
            _saved_quota_stop(run, router)
        if existing is not None:
            if (existing.get("key") != entry.key or existing.get("payload", {}).get("origin") != origin
                    or existing.get("request_hash") != request_hash
                    or existing.get("payload_hash") != digest(existing.get("payload")) or (terminal and not recovering)):
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
        if terminal and not recovering:
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
            raise TransportStopped("Batch remains incomplete; recovery requires a separately registered repair",
                                   code=terminal["code"])
        prior_attempts = recovery.grant(run, entry.key, request_hash, rows) if recovering else 0
        round_attempts = len(rows) - prior_attempts
        structural = sum(row["outcome"]["result"] == "structure_error" for row in rows[prior_attempts:])
        failed_transport = sum(row["outcome"]["result"] in {"transport_error", "unknown"} for row in rows)
        reason = next((row["outcome"].get("validation_reason") or row["outcome"]["code"] for row in reversed(rows)
                       if row["outcome"]["result"] == "structure_error"), None)
        last = rows[-1]["outcome"] if rows else None
        restored_delay = bool(rows)
        while not stop.is_set():
            if recovering and round_attempts >= 8:
                results[entry.key] = {"status": "exhausted", "payload": None}
                counts["exhausted"] += 1
                return
            if not recovering and structural >= limit:
                run.put_terminal(entry.key, request_hash, "structure_exhausted", "structure_exhausted")
                results[entry.key] = {"status": "exhausted", "payload": None}
                counts["exhausted"] += 1
                return
            if not recovering and last and last["result"] == "transport_error" and not last["retryable"]:
                run.put_terminal(entry.key, request_hash, "permanent_error", last["code"])
                raise TransportStopped("Non-retryable provider/protocol error", code=last["code"])
            if not recovering and failed_transport >= config.TRANSPORT_MAX_FAILURES:
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
                nonlocal number, round_attempts, resume_number
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
                round_attempts += 1

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
                        if recovering and number is not None:
                            recovery.defer(run, entry, request_hash, number, exc.code)
                        else:
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
                if code == "provider_quota_exhausted" and not recovering:
                    # Persist the shared stop before journal settlement so a
                    # restart cannot lose an already observed quota result.
                    _save_quota_stop(run, entry, request_hash)
                _save_telemetry(run, entry, number, request_hash, result="transport_error",
                                start=started, settled=settled)
                failed_transport += 1
                counts["transport_failures"] += 1
                details = getattr(exc, "details", {})
                status = details.get("http_status")
                lane_of = getattr(router, "lane_for", None)
                relay_protocol = (code == "provider_protocol_error" and lane_of is not None
                                  and lane_of(model_request["model"]) == config.MINIMAX_RELAY_LANE)
                retryable = isinstance(exc, LLMTransportError) and (
                    code in {"connection_error", "timeout"} or relay_protocol
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
                # Relay lanes (MiniMax relay, ezlook MiMo) rate-limit transiently: retry 429 with the provider's
                # Retry-After/backoff instead of the one-hour registered cooldown. Quota exhaustion still stops.
                relay_429 = (status == 429 and retryable and (
                    (lane_of is not None and lane_of(model_request["model"]) == config.MINIMAX_RELAY_LANE)
                    or (model_request["provider"] == "mimo"
                        and getattr(getattr(router, "pool", None), "mimo_api", None) == "ezlook")))
                if recovering and (code == "provider_quota_exhausted" or not retryable or (status == 429 and not relay_429)):
                    recovery.defer(run, entry, request_hash, number, code)
                    raise TransportStopped("Registered cooldown after provider failure", code=code)
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
        mr = entry.request["model_request"]
        provider = ((router.lane_for(mr["model"]) if hasattr(router, "lane_for") else mr["provider"])
                    if limits and (max_in_flight > 1 or getattr(router, "mimo_batch", False)) else "all")
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
    # The controlled-perturbation thread runs beside evaluation, over disjoint request keys.
    thread_name = threading.current_thread().name
    suffix = ("-perturb" if thread_name.startswith("formal8-perturb")
              else "-" + thread_name[len("formal8-"):] if thread_name.startswith("formal8-stage-") else "")
    execution_lock = run.root.with_name(run.root.name + "-transport-execute" + suffix)
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
