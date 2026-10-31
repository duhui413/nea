"""Minimal OpenAI-compatible transport used by the experiment pipeline."""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from nea.schemas import SchemaError
from nea.providers import config


_MAX_ERROR_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_QUOTA_CODES = frozenset({"insufficient_quota", "insufficient_balance", "quota_exhausted"})


def provider_error_code(payload: Any, *, model: str) -> str | None:
    """Recognize explicit machine codes only; provider prose is never retained."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    error = error if isinstance(error, dict) else {}
    if any(type(error.get(field)) is str and error[field] in _QUOTA_CODES
           for field in ("code", "type")):
        return "provider_quota_exhausted"
    # MiMo Token Plan uses code=429/type=limitation for both rate and quota
    # errors. Require explicit exhaustion evidence; never treat all 429s as quota.
    if (model.startswith("mimo-") and str(error.get("code")) == "429"
            and error.get("type") == "limitation"):
        message = error.get("message")
        if isinstance(message, str) and "quota" in message.lower() and "exhausted" in message.lower():
            return "provider_quota_exhausted"
    base = payload.get("base_resp")
    if (model in config.EXPERIMENT_MODEL_EXTRA_BODY
            and config.experiment_provider_name_for_model(model) == "minimax"
            and isinstance(base, dict) and type(base.get("status_code")) is int
            and base["status_code"] == 1008):
        return "provider_quota_exhausted"
    return None


def _read_error_payload(response: Any) -> Any:
    try:
        raw = response.read(_MAX_ERROR_BYTES + 1)
        return json.loads(raw) if len(raw) <= _MAX_ERROR_BYTES else None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any] | None = None
    finish_reason: str | None = None
    refusal: bool = False


class LLMTransportError(RuntimeError):
    """A provider or network failure, separate from schema validation errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def retry_after_seconds(value: str | None, *, now: float | None = None) -> float | None:
    """Accept only a finite delta or an HTTP date, never persist raw headers."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return None
            seconds = parsed.timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class OpenAICompatibleClient:
    """Dependency-free asynchronous OpenAI-compatible chat client."""

    def __init__(self, *, base_url: str, api_key: str):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        timeout: float,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await asyncio.to_thread(
            self._complete_sync,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
            extra_body=extra_body,
        )

    def _complete_sync(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        timeout: float,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        body = json.dumps(
            config.chat_request_body(model=model, prompt=prompt, temperature=temperature,
                                     max_tokens=max_tokens, seed=seed, extra_body=extra_body),
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        raise ValueError("response too large")
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError):
                    raise LLMTransportError(
                        "OpenAI-compatible endpoint returned invalid JSON",
                        code="provider_protocol_error",
                    ) from None
        except urllib.error.HTTPError as exc:
            delay = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
            code = provider_error_code(_read_error_payload(exc), model=model)
            code = code or ("payment_required" if exc.code == 402 else "http_status_error")
            raise LLMTransportError(
                f"OpenAI-compatible endpoint returned HTTP {exc.code}",
                code=code,
                details={"http_status": exc.code, "retry_after_seconds": delay},
            ) from None
        except urllib.error.URLError as exc:
            raise LLMTransportError(
                "OpenAI-compatible endpoint could not be reached",
                code="connection_error",
            ) from exc
        except TimeoutError as exc:
            raise LLMTransportError(
                "OpenAI-compatible endpoint request timed out",
                code="timeout",
            ) from exc
        except (ConnectionError, http.client.IncompleteRead) as exc:
            raise LLMTransportError(
                "OpenAI-compatible endpoint connection ended before the response completed",
                code="connection_error",
            ) from exc

        quota = provider_error_code(payload, model=model)
        if quota:
            raise LLMTransportError("Provider quota is exhausted", code=quota) from None
        if not isinstance(payload, dict):
            raise LLMTransportError("Provider response is not an object", code="provider_protocol_error")
        choices = payload.get("choices") or []
        if not choices:
            raise LLMTransportError(
                "OpenAI-compatible endpoint response has no choices",
                code="provider_protocol_error",
            )
        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason")
        refusal = bool(message.get("refusal")) or finish_reason == "content_filter"
        content = message.get("content")
        if content is None and (refusal or finish_reason == "length"):
            content = ""
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise LLMTransportError(
                "OpenAI-compatible endpoint response content is not text",
                code="provider_protocol_error",
            )
        return LLMResponse(
            text=content,
            usage=dict(payload.get("usage") or {}),
            raw=payload,
            finish_reason=finish_reason,
            refusal=refusal,
        )


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one strict JSON object without accepting Markdown fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        raise SchemaError("structured output must not use a Markdown code fence")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SchemaError("structured output must be one strict JSON object") from exc
    if not isinstance(value, dict):
        raise SchemaError("structured output must decode to an object")
    return value
