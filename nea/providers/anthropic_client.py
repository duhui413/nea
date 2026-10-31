"""Dependency-free asynchronous client for Anthropic-style ``/messages`` relays (spec 0042)."""
from __future__ import annotations

import asyncio
import http.client
import json
import urllib.error
import urllib.request
from typing import Any

from nea.providers import config
from nea.providers.client import (LLMResponse, LLMTransportError, _MAX_RESPONSE_BYTES, _read_error_payload,
                                  provider_error_code, retry_after_seconds)

_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"}


def normalize_usage(usage: Any) -> dict[str, int]:
    """Map Anthropic usage onto the prompt/completion fields used by telemetry."""
    usage = usage if isinstance(usage, dict) else {}
    fields = {key: usage.get(key) for key in ("input_tokens", "output_tokens",
                                              "cache_read_input_tokens", "cache_creation_input_tokens")}
    clean = {key: value for key, value in fields.items() if type(value) is int and value >= 0}
    prompt = sum(clean.get(key, 0) for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    completion = clean.get("output_tokens", 0)
    return {**clean, "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class AnthropicMessagesClient:
    """Same ``complete`` contract as the OpenAI-compatible client; seed and OpenAI-only extras are not sent."""

    def __init__(self, *, base_url: str, api_key: str):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/messages") else self.base_url + "/messages"

    async def complete(self, *, model: str, prompt: str, temperature: float, max_tokens: int, seed: int,
                       timeout: float, extra_body: dict[str, Any] | None = None) -> LLMResponse:
        return await asyncio.to_thread(self._complete_sync, model=model, prompt=prompt, temperature=temperature,
                                       max_tokens=max_tokens, timeout=timeout, extra_body=extra_body)

    @staticmethod
    def request_body(*, model: str, prompt: str, temperature: float, max_tokens: int,
                     extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"model": config.MINIMAX_RELAY_REQUEST_MODELS.get(model, model), "max_tokens": max_tokens,
                "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
        if (extra_body or {}).get("thinking") == {"type": "disabled"}:
            body["thinking"] = {"type": "disabled"}
        return body

    def _complete_sync(self, *, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float,
                       extra_body: dict[str, Any] | None = None) -> LLMResponse:
        body = json.dumps(self.request_body(model=model, prompt=prompt, temperature=temperature,
                                            max_tokens=max_tokens, extra_body=extra_body),
                          ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "x-api-key": self._api_key, "anthropic-version": config.MINIMAX_RELAY_ANTHROPIC_VERSION,
            "Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        raise ValueError("response too large")
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError):
                    raise LLMTransportError("Messages endpoint returned invalid JSON",
                                            code="provider_protocol_error") from None
        except urllib.error.HTTPError as exc:
            delay = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
            code = provider_error_code(_read_error_payload(exc), model=model)
            code = code or ("payment_required" if exc.code == 402 else "http_status_error")
            raise LLMTransportError(f"Messages endpoint returned HTTP {exc.code}", code=code,
                                    details={"http_status": exc.code, "retry_after_seconds": delay}) from None
        except urllib.error.URLError as exc:
            raise LLMTransportError("Messages endpoint could not be reached", code="connection_error") from exc
        except TimeoutError as exc:
            raise LLMTransportError("Messages endpoint request timed out", code="timeout") from exc
        except (ConnectionError, http.client.IncompleteRead) as exc:
            raise LLMTransportError("Messages endpoint connection ended before the response completed",
                                    code="connection_error") from exc
        quota = provider_error_code(payload, model=model)
        if quota:
            raise LLMTransportError("Provider quota is exhausted", code=quota) from None
        if not isinstance(payload, dict) or payload.get("type") == "error":
            raise LLMTransportError("Messages endpoint returned an error or non-object body",
                                    code="provider_protocol_error")
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise LLMTransportError("Messages endpoint response has no content blocks", code="provider_protocol_error")
        text = "".join(block.get("text", "") for block in blocks
                       if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
        stop = payload.get("stop_reason")
        refusal = stop == "refusal"
        if not text and not (refusal or stop == "max_tokens"):
            raise LLMTransportError("Messages endpoint response has no text content", code="provider_protocol_error")
        return LLMResponse(text=text, usage=normalize_usage(payload.get("usage")), raw=payload,
                           finish_reason=_STOP_REASONS.get(stop, stop), refusal=refusal)
