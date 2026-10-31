"""Persistent MiMo batches at experiment dependency-wave boundaries.

The ordinary transport still owns validation, retries, and the attempt journal.
This adapter owns asynchronous remote work, including the ambiguous-create gap.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nea.providers import config
from nea.providers.client import LLMResponse, LLMTransportError
from nea.runtime.records import ConflictError, canonical_bytes, digest, write_once


def source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    names = ("nea/providers/config.py", "nea/providers/pool.py", "nea/providers/router.py",
             "nea/providers/batch.py", "nea/runtime/batch_transport.py")
    return digest({name: digest((root / name).read_bytes()) for name in names})


class BatchDeferred(LLMTransportError):
    """Remote work is unresolved: keep its reservation open, never resubmit."""


class BatchWaveRouter:
    def __init__(self, router, run):
        self.router, self.run, self.pool = router, run, router.pool
        self.roles, self.generator = router.roles, router.generator
        self.mimo_batch = True
        self.resume_only = getattr(router, "resume_mimo_batches", False)
        self.batch_policy = getattr(self.pool, "batch_policy", config.MIMO_BATCH_POLICY)
        self.provider_limits = (dict(router.provider_limits) if self.resume_only else
                                {**router.provider_limits, "mimo": self.batch_policy["max_requests"]})
        self.max_in_flight = router.max_in_flight
        self.model_identity = router.model_identity
        self.root = run.root / "batches"
        self.client = self.pool.batch_client
        self._queued = []
        self._callbacks = {}
        self._tasks = {}
        self._flush_task = None
        # A crash between manifest and index publication must not duplicate rows.
        for path in sorted(self.root.glob("jobs/*/input.json")):
            manifest = json.loads(path.read_text(encoding="utf8"))
            self._validate_manifest(path.parent.name, manifest)
            self._index(path.parent.name, manifest)

    def uses_send_admission(self, role):
        return True

    def lane_for(self, model):
        inner = getattr(self.router, "lane_for", None)
        return inner(model) if inner else config.experiment_provider_name_for_model(model)

    def _index_path(self, key, attempt):
        return self.root / "requests" / key / f"{attempt:04d}.json"

    def _index(self, chunk, manifest):
        for row in manifest["requests"]:
            write_once(self._index_path(row["key"], row["attempt"]),
                       {"chunk": chunk, "request_hash": row["request_hash"],
                        "body_hash": digest(row["body"]), "custom_id": row["custom_id"]})

    def _validate_manifest(self, chunk, manifest):
        if (digest(manifest) != chunk or manifest.get("execution_identity_hash") != digest(self.run.identity)
                or manifest.get("version") != self.batch_policy["version"]):
            raise ConflictError("Batch input identity differs")

    def pending_attempt(self, key, request_hash, number):
        path = self._index_path(key, number)
        if not path.exists():
            return False
        index = json.loads(path.read_text(encoding="utf8"))
        if index.get("request_hash") != request_hash:
            raise ConflictError("Batch request identity differs")
        return True

    async def complete(self, *, batch_request=None, before_send=None, **kwargs):
        provider = config.experiment_provider_name_for_model(self.roles[kwargs["role"]])
        if provider != "mimo":
            return await self.router.complete(before_send=before_send, **kwargs)
        if batch_request is None or before_send is None:
            raise ValueError("Batch completion requires a persistent request and reservation")
        key, attempt = batch_request["key"], batch_request["attempt"]
        model = self.roles[kwargs["role"]]
        body = config.chat_request_body(model=model, prompt=kwargs["prompt"],
            temperature=kwargs["temperature"], max_tokens=kwargs["max_tokens"], seed=kwargs["seed"],
            extra_body=config.structured_extra_body(model, kwargs.get("output_schema"), kwargs.get("schema_name")))
        row = {**batch_request, "body": body, "custom_id": key + "-" + str(attempt)}
        self._callbacks[(key, attempt)] = before_send
        index_path = self._index_path(key, attempt)
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf8"))
            if index["request_hash"] != row["request_hash"] or index["body_hash"] != digest(body):
                raise ConflictError("Resumed Batch body differs")
            directory = self.root / "jobs" / index["chunk"]
            if self.resume_only and not any((directory / name).exists() for name in
                                            ("submitted.json", "creating.json", "results.json")):
                return await self.router.complete(before_send=before_send, **kwargs)
            task = self._task(index["chunk"])
        else:
            if self.resume_only:
                return await self.router.complete(before_send=before_send, **kwargs)
            future = asyncio.get_running_loop().create_future()
            self._queued.append((row, future))
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush())
            task = future
        result = (await task)[row["custom_id"]]
        if "error" in result:
            error = result["error"]
            if error["code"] == "provider_quota_exhausted":
                self.pool.stop_for_quota("mimo")
            raise LLMTransportError("MiMo Batch request failed", code=error["code"], details=error["details"])
        response = LLMResponse(text=result["text"], usage=result["usage"],
                               finish_reason=result["finish_reason"], refusal=result["refusal"])
        return response, [{"role": kwargs["role"], "provider": "mimo", "model": model,
            "status": "ok", "attempt": 1, "transport": "batch", "batch_id": result["batch_id"],
            "custom_id": row["custom_id"], **response.usage}]

    def _task(self, chunk):
        if chunk not in self._tasks:
            self._tasks[chunk] = asyncio.create_task(self._drive(chunk))
        return self._tasks[chunk]

    async def _flush(self):
        # Every independent MiMo worker enters before this collection boundary.
        await asyncio.sleep(0)
        queued, self._queued = self._queued, []
        from nea.providers.batch import partition_requests, request_row
        rows = sorted((row for row, _ in queued), key=lambda row: row["custom_id"])
        lookup = {row["custom_id"]: row for row in rows}
        try:
            groups = partition_requests([request_row(row["custom_id"], row["body"]) for row in rows],
                max_requests=self.batch_policy["max_requests"], max_input_bytes=self.batch_policy["max_input_bytes"])
            futures = {row["custom_id"]: future for row, future in queued}
            for group in groups:
                manifest = {"version": self.batch_policy["version"],
                    "execution_identity_hash": digest(self.run.identity),
                    "requests": [lookup[row["custom_id"]] for row in group]}
                chunk = digest(manifest)
                write_once(self.root / "jobs" / chunk / "input.json", manifest)
                self._index(chunk, manifest)
                task = self._task(chunk)
                for row in group:
                    future = futures[row["custom_id"]]
                    task.add_done_callback(lambda done, target=future: self._resolve(done, target))
        except Exception as exc:
            for _, future in queued:
                if not future.done():
                    future.set_exception(exc)

    @staticmethod
    def _resolve(task, future):
        if future.done():
            return
        if task.cancelled():
            future.cancel()
        elif task.exception() is not None:
            future.set_exception(task.exception())
        else:
            future.set_result(task.result())

    def _reserve(self, manifest):
        for row in manifest["requests"]:
            rows = self.run.attempts(row["key"], row["request_hash"])
            if len(rows) == row["attempt"] and rows[-1]["outcome"] is None:
                # Let the ordinary engine adopt its existing attempt number.
                callback = self._callbacks.get((row["key"], row["attempt"]))
                if callback:
                    callback()
                continue
            if len(rows) != row["attempt"] - 1:
                # Already settled results need no new reservation.
                if len(rows) >= row["attempt"]:
                    continue
                raise ConflictError("Batch attempt sequence differs")
            self.pool._check_quota_stop()
            callback = self._callbacks.get((row["key"], row["attempt"]))
            if callback:
                callback()
            else:
                self.run.reserve_attempt(row["key"], row["request_hash"])

    async def _drive(self, chunk):
        from nea.providers.batch import encode_jsonl, request_row, decode_result
        directory = self.root / "jobs" / chunk
        manifest = json.loads((directory / "input.json").read_text(encoding="utf8"))
        self._validate_manifest(chunk, manifest)
        saved = directory / "results.json"
        if saved.exists():
            self._reserve(manifest)
            return self._read_results(saved, manifest)
        if self.resume_only and not any((directory / name).exists() for name in
                                        ("submitted.json", "creating.json")):
            raise ConflictError("Resume-only Batch router cannot submit a new task")
        if self.client is None:
            self.client = self.pool.legacy_batch_client()
        acquired = False
        try:
            while not self.pool.batch_semaphore.acquire(blocking=False):
                await asyncio.sleep(0.1)
            acquired = True
            self.pool._check_quota_stop()
            uploaded = directory / "uploaded.json"
            submitted = directory / "submitted.json"
            creating = directory / "creating.json"
            if submitted.exists():
                task = json.loads(submitted.read_text(encoding="utf8"))
                self._reserve(manifest)
            elif creating.exists():
                raise BatchDeferred("Batch creation outcome is unknown; attach its server task ID before resuming",
                                    code="batch_submission_unknown")
            else:
                if uploaded.exists():
                    input_id = json.loads(uploaded.read_text(encoding="utf8"))["input_file_id"]
                else:
                    content = encode_jsonl([request_row(row["custom_id"], row["body"]) for row in manifest["requests"]])
                    input_id = await asyncio.to_thread(self.client.upload, content, chunk + ".jsonl")
                    write_once(uploaded, {"input_file_id": input_id, "input_sha256": digest(content)})
                self._reserve(manifest)
                write_once(creating, {"input_file_id": input_id, "name": chunk})
                try:
                    task = await asyncio.to_thread(self.client.create, input_id, chunk)
                except LLMTransportError as exc:
                    status = exc.details.get("http_status")
                    if (type(status) is int and 400 <= status < 500) or exc.code == "provider_quota_exhausted":
                        if exc.code == "provider_quota_exhausted":
                            self.pool.stop_for_quota("mimo")
                        results = {row["custom_id"]: {"error": {"code": exc.code, "details": exc.details}}
                                   for row in manifest["requests"]}
                        write_once(directory / "rejected.json", {"code": exc.code, "details": exc.details})
                        write_once(saved, {"input_hash": chunk, "results": results, "results_hash": digest(results)})
                        return results
                    raise BatchDeferred("Batch creation outcome is unknown; attach its server task ID before resuming",
                                        code="batch_submission_unknown") from None
                except Exception:
                    # Even a lost HTTP response might conceal a paid task.
                    raise BatchDeferred("Batch creation outcome is unknown; attach its server task ID before resuming",
                                        code="batch_submission_unknown") from None
                write_once(submitted, task)
            failures = 0
            while True:
                try:
                    task = await asyncio.to_thread(self.client.retrieve, task["id"])
                    failures = 0
                except LLMTransportError as exc:
                    if exc.code == "provider_quota_exhausted":
                        self.pool.stop_for_quota("mimo")
                        raise BatchDeferred("Provider quota is exhausted; the remote task remains pending",
                                            code="provider_quota_exhausted") from None
                    failures += 1
                    if self.resume_only or failures >= config.TRANSPORT_MAX_FAILURES:
                        raise BatchDeferred("Batch status is unavailable; resume the same run to poll again",
                                            code="batch_pending") from None
                    await asyncio.sleep(self.batch_policy["poll_seconds"])
                    continue
                status = str(task.get("status", "")).lower()
                if status not in {"validating", "created", "in_progress", "finalizing", "cancelling",
                                  "completed", "failed", "expired", "cancelled"}:
                    raise BatchDeferred("Batch returned an unknown status", code="batch_pending")
                status_hash = digest(task)
                # Keep Windows paths below MAX_PATH; the complete hash stays in content.
                write_once(directory / "status" / (status_hash[:16] + ".json"),
                           {**task, "status_hash": status_hash})
                if status in {"completed", "failed", "expired", "cancelled"}:
                    break
                if self.resume_only:
                    raise BatchDeferred("Existing Batch task is pending; independent realtime work may continue",
                                        code="batch_pending")
                await asyncio.sleep(self.batch_policy["poll_seconds"])
            expected = {row["custom_id"]: row for row in manifest["requests"]}
            results = {}
            for field in ("output_file_id", "error_file_id"):
                if not task.get(field):
                    continue
                try:
                    raw = await asyncio.to_thread(self.client.content, task[field])
                    lines = raw.decode("utf-8-sig").splitlines()
                    for line in lines:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        custom_id = row.get("custom_id") if isinstance(row, dict) else None
                        if custom_id not in expected or custom_id in results:
                            raise ConflictError("Batch result has an unknown or duplicate custom_id")
                        try:
                            response = decode_result(row, model=expected[custom_id]["body"]["model"])
                            results[custom_id] = {"text": response.text, "usage": response.usage,
                                "finish_reason": response.finish_reason, "refusal": response.refusal,
                                "batch_id": task["id"]}
                        except LLMTransportError as exc:
                            if exc.code == "provider_quota_exhausted":
                                self.pool.stop_for_quota("mimo")
                            results[custom_id] = {"error": {"code": exc.code, "details": exc.details},
                                                  "batch_id": task["id"]}
                except (LLMTransportError, ValueError, UnicodeError):
                    raise BatchDeferred("Batch results could not be downloaded or decoded; resume the same run",
                                        code="batch_pending") from None
            for custom_id in expected.keys() - results.keys():
                results[custom_id] = {"error": {"code": "http_status_error",
                    "details": {"http_status": 503, "batch_status": status}}, "batch_id": task["id"]}
            if task.get("error_code") == "provider_quota_exhausted":
                self.pool.stop_for_quota("mimo")
                for custom_id in expected.keys() - {key for key, value in results.items() if "text" in value}:
                    results[custom_id]["error"]["code"] = "provider_quota_exhausted"
            write_once(saved, {"input_hash": chunk, "results": results, "results_hash": digest(results)})
            return results
        except LLMTransportError as exc:
            if exc.code == "provider_quota_exhausted":
                self.pool.stop_for_quota("mimo")
            raise
        finally:
            if acquired:
                self.pool.batch_semaphore.release()

    def _read_results(self, path, manifest):
        record = json.loads(path.read_text(encoding="utf8"))
        results = record.get("results")
        if (record.get("input_hash") != digest(manifest) or record.get("results_hash") != digest(results)
                or set(results) != {row["custom_id"] for row in manifest["requests"]}):
            raise ConflictError("Saved Batch results differ")
        return results


def attach_task(phase_root: Path, chunk: str, batch_id: str, client) -> None:
    """Explicitly reconcile a lost create response against the uploaded file."""
    if len(chunk) != 64 or any(char not in "0123456789abcdef" for char in chunk):
        raise ValueError("chunk must be the local 64-character Batch input hash")
    directory = phase_root / "batches" / "jobs" / chunk
    intent = json.loads((directory / "creating.json").read_text(encoding="utf8"))
    task = client.retrieve(batch_id)
    if task.get("id") != batch_id or task.get("input_file_id") != intent["input_file_id"]:
        raise ConflictError("Server Batch task does not match the uploaded input file")
    write_once(directory / "submitted.json", task)
