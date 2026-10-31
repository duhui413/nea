"""Direct role routing, one in-flight request per provider key."""
from __future__ import annotations
import asyncio
import math
import time
from pathlib import Path
from nea.providers import config
from nea.providers.pool import ProviderPool
from nea.providers.budget import estimate_tokens
from nea.providers.client import LLMTransportError

class RoleRouter:
    def __init__(self, generator: str, env_file: Path, pool: ProviderPool | None = None):
        self.roles = config.experiment_role_models_for(generator)
        self.generator = self.roles["student1"]
        self.pool = pool if pool is not None else ProviderPool(env_file)
        self.mimo_batch = self.pool.batch_mode
        self.provider_limits = dict(self.pool.provider_limits)
        if config.MIMO_OFFICIAL_LANE in self.provider_limits:
            # Scheduling lane "mimo" feeds both MiMo routes; admission picks the route per request.
            self.provider_limits["mimo"] += self.provider_limits[config.MIMO_OFFICIAL_LANE]
        self.max_in_flight = self.pool.max_in_flight

    @property
    def model_identity(self):
        from nea.runtime.identity import model_config_identity
        if hasattr(self, "logical_model_identity"):
            return self.logical_model_identity
        return model_config_identity(self.generator, mimo_api=self.pool.mimo_api, batch_policy=self.pool.batch_policy,
                                     mimo_concurrency=self.pool.mimo_concurrency,
                                     minimax_concurrency=self.pool.minimax_concurrency,
                                     minimax_m3_api=self.pool.minimax_m3_api,
                                     minimax_relay_concurrency=self.pool.minimax_relay_concurrency,
                                     mimo_official_concurrency=self.pool.mimo_official_concurrency)

    def lane_for(self, model):
        return self.pool.lane_for(model)

    def uses_send_admission(self, role):
        # Every provider checks the shared quota stop before reserving a send.
        return True

    async def complete(self, *, role, prompt, temperature, max_tokens, seed, timeout,
                       output_schema=None, schema_name=None, before_send=None):
        model = self.roles[role]
        provider = config.experiment_provider_name_for_model(model)
        if self.mimo_batch and provider == "mimo":
            raise RuntimeError("MiMo Batch requests require a persistent experiment wave")
        extra_body = config.structured_extra_body(model, output_schema, schema_name)
        body = config.chat_request_body(model=model, prompt=prompt, temperature=temperature,
                                       max_tokens=max_tokens, seed=seed, extra_body=extra_body)
        lane = self.pool.admission_lane(model)
        limited = self.pool.rate_limited(lane)
        tokens = estimate_tokens(body) if limited else None
        async with self.pool.admit(lane, self.generator, tokens, model=model) as admission:
            admission.mark_sent(before_send)
            start = time.monotonic()
            pending = asyncio.create_task(admission.client.complete(model=model, prompt=prompt,
                temperature=temperature, max_tokens=max_tokens, seed=seed, timeout=timeout,
                extra_body=extra_body))
            response = None
            cancelled = False
            try:
                # Even repeated outer cancellation must not release urllib's slot.
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        break
                response = pending.result()
            except LLMTransportError as exc:
                official_failed = lane == config.MIMO_OFFICIAL_LANE and (
                    exc.code == "provider_quota_exhausted" or exc.details.get("http_status") in {401, 402, 403})
                if official_failed:
                    # A failing official route must not stop the whole run: drop it and let the request retry on ezlook.
                    self.pool.disable_official()
                    raise LLMTransportError("Official MiMo route disabled after a hard failure", code="connection_error",
                                            details={"official_route_disabled": True}) from None
                if exc.code == "provider_quota_exhausted":
                    # A known quota result outranks an outer cancellation:
                    # propagate it so transport can persist the global stop.
                    self.pool.stop_for_quota(provider)
                    raise
                if limited and exc.details.get("http_status") == 429:
                    delay = exc.details.get("retry_after_seconds")
                    if (type(delay) not in {int, float} or not math.isfinite(delay) or delay < 0):
                        policy = config.MIMO_RATE_POLICY if provider == "mimo" else config.SILICONFLOW_RATE_POLICY
                        delay = policy["window_seconds"]
                    admission.limited(delay)
                    # Runtime persists precisely the same effective cooldown.
                    exc.details["retry_after_seconds"] = delay
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
            finally:
                admission.settle(response.usage if response is not None else None)
            if cancelled:
                raise asyncio.CancelledError
            record = {"role": role, "provider": provider, "model": model, "status": "ok",
                      "attempt": 1, "credential_slot": admission.slot, "latency_ms": round((time.monotonic()-start)*1000, 1),
                      **response.usage}
            return response, [record]
