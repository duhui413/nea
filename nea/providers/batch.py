"""Independent, synchronous MiMo Batch API transport with no hidden retries.

Request bodies continue to come from ``providers.config.chat_request_body``.
The caller owns the append-only task journal and must treat a failed ``create``
as potentially submitted until it can reconcile the server task ID.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

from nea.providers import config
from nea.providers.client import LLMResponse, LLMTransportError, provider_error_code, retry_after_seconds


_MAX_CONTROL_BYTES = 4 * 1024 * 1024
_MAX_ERROR_BYTES = 64 * 1024
_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}\Z")
_TASK_STATUSES = frozenset({
    "created", "validating", "in_progress", "finalizing", "completed", "failed", "expired",
    "cancelling", "cancelled",
})
_PROVIDER_CODES = frozenset({
    "invalid_request_error", "rate_limit_exceeded", "rate_limit_error",
    "server_error", "internal_server_error", "inference_failed", "batch_expired", "batch_cancelled",
    "request_timeout", "invalid_request", "invalid_json", "model_not_found",
    "insufficient_quota", "insufficient_balance", "quota_exhausted",
})
_TRANSIENT_ITEM_CODES = frozenset({
    "server_error", "internal_server_error", "inference_failed", "batch_expired", "batch_cancelled",
})
_TASK_TIMESTAMPS = (
    "created_at", "in_progress_at", "expires_at", "finalizing_at", "completed_at",
    "failed_at", "expired_at", "cancelling_at", "cancelled_at",
)


def _protocol_error(*, usage: dict[str, int] | None = None) -> LLMTransportError:
    return LLMTransportError(
        "MiMo Batch returned an invalid response", code="provider_protocol_error",
        details={"usage": dict(usage)} if usage is not None else {},
    )


def _identifier(value: Any) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise _protocol_error()
    return value


def safe_usage(value: Any) -> dict[str, int]:
    """Keep known token counts; absent, invalid or unknown counts remain absent."""
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(value.get(field)) is int and value[field] >= 0}


def _safe_provider_codes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in ("code", "type")
            if type(value.get(field)) is str and value[field] in _PROVIDER_CODES}


def sanitize_task(payload: Any) -> dict[str, Any]:
    """Public task fields only: no server messages, metadata or credential data."""
    if not isinstance(payload, dict):
        raise _protocol_error()
    result: dict[str, Any] = {"id": _identifier(payload.get("id"))}
    status = payload.get("status")
    if type(status) is not str or status.lower() not in _TASK_STATUSES:
        raise _protocol_error()
    result["status"] = status.lower()
    for field in ("input_file_id", "output_file_id", "error_file_id"):
        value = payload.get(field)
        if value is not None:
            result[field] = _identifier(value)
        else:
            result[field] = None
    counts = payload.get("request_counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise _protocol_error()
        result["request_counts"] = {
            field: counts[field] for field in ("total", "completed", "failed")
            if type(counts.get(field)) is int and counts[field] >= 0
        }
    for field in _TASK_TIMESTAMPS:
        value = payload.get(field)
        if type(value) is int and value >= 0:
            result[field] = value
    # Messages and arbitrary machine-code strings can contain secrets. Retain
    # only a closed set of recognized codes, with no provider text or params.
    errors = payload.get("errors")
    if isinstance(errors, dict) and isinstance(errors.get("data"), list):
        result["errors"] = {"data": [_safe_provider_codes(value) for value in errors["data"]]}
        if any(provider_error_code({"error": value}, model="") for value in errors["data"]):
            result["error_code"] = "provider_quota_exhausted"
    if provider_error_code(payload, model=""):
        result["error_code"] = "provider_quota_exhausted"
    return result


def request_row(custom_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap the existing chat body without changing its model/options/schema."""
    if type(custom_id) is not str or not custom_id or len(custom_id) > 256:
        raise ValueError("batch custom_id must be a nonempty string of at most 256 characters")
    if not isinstance(body, Mapping):
        raise ValueError("batch request body must be an object")
    return config.batch_input_row(custom_id, dict(body))


def _encoded_row(row: Mapping[str, Any]) -> bytes:
    if not isinstance(row, Mapping):
        raise ValueError("batch request row must be an object")
    custom_id = row.get("custom_id")
    if (type(custom_id) is not str or not custom_id or len(custom_id) > 256
            or row.get("method") != "POST" or row.get("url") != config.MIMO_BATCH_REQUEST_PATH
            or not isinstance(row.get("body"), dict)
            or set(row) != {"custom_id", "method", "url", "body"}):
        raise ValueError("invalid batch request row")
    try:
        return (json.dumps(dict(row), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("batch request row is not finite UTF-8 JSON") from None


def encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Deterministic UTF-8 JSONL, including one terminating newline per row."""
    seen: set[str] = set()
    result = []
    for row in rows:
        encoded = _encoded_row(row)
        custom_id = row["custom_id"]
        if custom_id in seen:
            raise ValueError("duplicate batch custom_id")
        seen.add(custom_id)
        result.append(encoded)
    return b"".join(result)


def partition_requests(
    rows: Iterable[Mapping[str, Any]], *, max_requests: int | None = None,
    max_input_bytes: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Stable-order sharding by both request count and actual UTF-8 file bytes."""
    max_requests = config.MIMO_BATCH_POLICY["max_requests"] if max_requests is None else max_requests
    max_input_bytes = config.MIMO_BATCH_POLICY["max_input_bytes"] if max_input_bytes is None else max_input_bytes
    if type(max_requests) is not int or not 1 <= max_requests <= 100 or type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("batch partition limits must be positive integers")
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    seen: set[str] = set()
    for row in rows:
        encoded = _encoded_row(row)
        custom_id = row["custom_id"]
        if custom_id in seen:
            raise ValueError("duplicate batch custom_id")
        seen.add(custom_id)
        if len(encoded) > max_input_bytes:
            raise ValueError("one batch request exceeds the input byte limit")
        if current and (len(current) >= max_requests or current_bytes + len(encoded) > max_input_bytes):
            result.append(current)
            current, current_bytes = [], 0
        current.append(dict(row))
        current_bytes += len(encoded)
    if current:
        result.append(current)
    return result


def decode_result(row: Any, model: str) -> LLMResponse:
    """Decode one matched result; failures retain known usage for accounting."""
    if not isinstance(row, dict):
        raise _protocol_error()
    response = row.get("response")
    response = response if isinstance(response, dict) else {}
    body = response.get("body")
    body = body if isinstance(body, dict) else {}
    usage = safe_usage(body.get("usage"))
    quota = provider_error_code(body, model=model) or provider_error_code(row, model=model)
    if quota:
        raise LLMTransportError("Provider quota is exhausted", code=quota, details={"usage": usage})
    status = response.get("status_code")
    error = row.get("error")
    if error is not None or (type(status) is int and not 200 <= status < 300):
        details: dict[str, Any] = {"usage": usage}
        if type(status) is int and 100 <= status <= 599:
            details["http_status"] = status
        provider_codes = _safe_provider_codes(error if isinstance(error, dict) else body.get("error"))
        details.update({"provider_" + key: value for key, value in provider_codes.items()})
        code = "payment_required" if status == 402 else "http_status_error" if type(status) is int else "batch_item_error"
        # Official error-file rows can have response=null and inference_failed.
        # Infer retryability only from recognized codes and only if the actual
        # response has no status; a real HTTP 400 must remain a terminal 400.
        if status is None:
            known_codes = set(provider_codes.values())
            if "request_timeout" in known_codes:
                code = "timeout"
            elif known_codes & _TRANSIENT_ITEM_CODES:
                code = "http_status_error"
                details["http_status"] = 503
            elif known_codes & {"rate_limit_exceeded", "rate_limit_error"}:
                code = "http_status_error"
                details["http_status"] = 429
        raise LLMTransportError("MiMo Batch item failed", code=code, details=details)
    if type(status) is not int or not 200 <= status < 300:
        raise _protocol_error(usage=usage)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _protocol_error(usage=usage)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _protocol_error(usage=usage)
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and (type(finish_reason) is not str or finish_reason not in {
            "stop", "length", "content_filter", "tool_calls", "function_call"}):
        raise _protocol_error(usage=usage)
    refusal = bool(message.get("refusal")) or finish_reason == "content_filter"
    content = message.get("content")
    if content is None and (refusal or finish_reason == "length"):
        content = ""
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    if not isinstance(content, str):
        raise _protocol_error(usage=usage)
    return LLMResponse(text=content, usage=usage, finish_reason=finish_reason, refusal=refusal)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MiMoBatchClient:
    """Fixed-endpoint Batch HTTP client; all POSTs are executed exactly once."""

    def __init__(self, api_key: str, *, timeout: float | None = None):
        if type(api_key) is not str or not api_key.strip():
            raise ValueError("MiMo Batch api_key is required")
        timeout = config.MIMO_BATCH_POLICY["request_timeout_seconds"] if timeout is None else timeout
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("MiMo Batch timeout must be finite and positive")
        self._api_key = api_key
        self.timeout = float(timeout)
        self._open = urllib.request.build_opener(_NoRedirect()).open

    @property
    def endpoint(self) -> str:
        return config.MIMO_BATCH_ENDPOINT.rstrip("/")

    def _request(
        self, path: str, *, method: str = "GET", data: bytes | None = None,
        content_type: str | None = None, as_json: bool = True,
    ) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.endpoint + path, data=data, method=method, headers=headers)
        limit = _MAX_CONTROL_BYTES if as_json else _MAX_OUTPUT_BYTES
        try:
            with self._open(request, timeout=self.timeout) as response:
                raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise _protocol_error()
                if not as_json:
                    return raw
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError, RecursionError):
                    raise _protocol_error() from None
        except urllib.error.HTTPError as exc:
            delay = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
            try:
                raw = exc.read(_MAX_ERROR_BYTES + 1)
                error = json.loads(raw) if len(raw) <= _MAX_ERROR_BYTES else None
            except (OSError, ValueError, UnicodeError, RecursionError):
                error = None
            code = provider_error_code(error, model="") or ("payment_required" if exc.code == 402 else "http_status_error")
            raise LLMTransportError(
                "MiMo Batch endpoint returned an HTTP error", code=code,
                details={"http_status": exc.code, "retry_after_seconds": delay},
            ) from None
        except urllib.error.URLError:
            raise LLMTransportError("MiMo Batch endpoint could not be reached", code="connection_error") from None
        except TimeoutError:
            raise LLMTransportError("MiMo Batch request timed out", code="timeout") from None
        except (ConnectionError, http.client.IncompleteRead, OSError):
            raise LLMTransportError("MiMo Batch connection ended before the response completed", code="connection_error") from None
        if not isinstance(payload, dict):
            raise _protocol_error()
        quota = provider_error_code(payload, model="")
        if quota:
            raise LLMTransportError("Provider quota is exhausted", code=quota) from None
        return payload

    def upload(self, content: bytes, filename: str) -> str:
        if type(content) is not bytes or not content:
            raise ValueError("batch input must be nonempty bytes")
        if len(content) > config.MIMO_BATCH_POLICY["max_input_bytes"]:
            raise ValueError("batch input exceeds the byte limit")
        if (type(filename) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", filename)
                or not filename.endswith(".jsonl")):
            raise ValueError("batch filename must be a safe JSONL basename")
        boundary = "nea_batch_" + hashlib.sha256(content).hexdigest()
        # The generated hexadecimal delimiter is checked against the input,
        # so even deliberately crafted JSON cannot alter the multipart body.
        while boundary.encode("ascii") in content:
            boundary = "nea_batch_" + hashlib.sha256(boundary.encode("ascii")).hexdigest()
        separator = ("--" + boundary + "\r\n").encode("ascii")
        data = (separator + b'Content-Disposition: form-data; name="purpose"\r\n\r\n'
                + config.MIMO_BATCH_FILE_PURPOSE.encode("ascii") + b"\r\n"
                + separator + f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("ascii")
                + b"Content-Type: application/jsonl\r\n\r\n" + content
                + ("\r\n--" + boundary + "--\r\n").encode("ascii"))
        payload = self._request(config.MIMO_BATCH_API_PATHS["files"], method="POST", data=data,
                                content_type="multipart/form-data; boundary=" + boundary)
        return _identifier(payload.get("id"))

    def create(self, input_file_id: str, name: str) -> dict[str, Any]:
        _identifier(input_file_id)
        if type(name) is not str or not name or len(name) > 256:
            raise ValueError("batch name must be a nonempty string of at most 256 characters")
        body = config.batch_create_body(input_file_id, name)
        data = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sanitize_task(self._request(config.MIMO_BATCH_API_PATHS["batches"], method="POST", data=data,
                                           content_type="application/json; charset=utf-8"))

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return sanitize_task(self._request(config.MIMO_BATCH_API_PATHS["batches"] + "/" + _identifier(batch_id)))

    def content(self, file_id: str) -> bytes:
        return self._request(config.MIMO_BATCH_API_PATHS["files"] + "/" + _identifier(file_id)
                             + config.MIMO_BATCH_API_PATHS["content"], as_json=False)
