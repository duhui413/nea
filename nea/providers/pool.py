"""Thread-safe credential slots shared by generator groups and event loops.

Each provider serves its nonempty generator queues in round-robin order.
Admission for one provider never waits for another provider's busy slots.
No request, retry, credential identifier or credential hash is persisted here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nea.providers import config
from nea.providers.client import LLMTransportError, OpenAICompatibleClient
from nea.providers.budget import Charge, TokenBudget


@dataclass(eq=False, repr=False)
class _Ticket:
    provider: str
    generator: str
    slot: int | None = None
    released: bool = False
    ready: asyncio.Future[int] | None = None
    tokens: int | None = None
    charge: Charge | None = None
    sent: bool = False
    model: str | None = None
    budget_key: tuple | None = None
    requested: float = 0.0
    wait_reason: str | None = None


class Admission:
    """One reserved capacity unit; settlement and cancellation stay in the shared pool."""
    def __init__(self, pool, ticket):
        self._pool, self._ticket = pool, ticket
        self.slot = pool._slot_indices[ticket.provider][ticket.slot] + 1
        self.client = pool._clients[ticket.provider][ticket.slot]

    def mark_sent(self, before_send=None):
        with self._pool._condition:
            self._pool._check_quota_stop()
            if self._ticket.released:
                raise RuntimeError("provider admission was released")
            if before_send is not None:
                before_send()
            self._ticket.sent = True

    def settle(self, usage=None):
        self._pool._settle(self._ticket, usage)

    def limited(self, delay):
        self._pool._limited(self._ticket, delay)


class ProviderPool:
    """Credential identity and provider capacity are separate, across asyncio loops."""

    def __init__(self, env_file: Path, *, client_factory: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, schedule=None,
                 mimo_batch: bool = False, batch_client_factory=None,
                 mimo_api: str | None = None, mimo_batch_size: int = 5, mimo_concurrency: int | None = None, minimax_concurrency: int | None = None,
                 minimax_m3_api: str | None = None, minimax_relay_concurrency: int | None = None,
                 mimo_official_concurrency: int | None = None):
        factory = client_factory or OpenAICompatibleClient
        if client_factory is None:
            from nea.providers.anthropic_client import AnthropicMessagesClient
            relay_factory = AnthropicMessagesClient
        else:
            relay_factory = client_factory
        self.mimo_api = config.resolve_mimo_api(mimo_api, mimo_batch=mimo_batch)
        mimo_batch = self.mimo_api == "batch"
        self.batch_mode = mimo_batch
        self.mimo_concurrency = config.mimo_concurrency(mimo_concurrency, self.mimo_api)
        self._gate_lock = threading.Lock()
        self._gate_next = 0.0
        self.minimax_concurrency = config.minimax_concurrency(minimax_concurrency)
        self.minimax_m3_api = config.resolve_minimax_m3_api(minimax_m3_api)
        self.mimo_official_concurrency = config.mimo_official_concurrency(mimo_official_concurrency)
        self._official_disabled = False
        self.minimax_relay_concurrency = config.minimax_relay_concurrency(minimax_relay_concurrency)
        self._env_file = env_file
        self.batch_policy = {**config.MIMO_BATCH_POLICY, "max_requests": config.batch_size(mimo_batch_size)}
        self.batch_client = None
        self.batch_semaphore = threading.Semaphore(config.MIMO_BATCH_POLICY["max_active_batches"])
        if mimo_batch:
            from nea.providers.batch import MiMoBatchClient
            self.batch_client = (batch_client_factory or MiMoBatchClient)(
                api_key=config.load_mimo_batch_credential(env_file))
        credentials = config.load_credential_pools(env_file, mimo_batch=mimo_batch, mimo_api=self.mimo_api,
                                               minimax_m3_api=self.minimax_m3_api,
                                               mimo_official=self.mimo_official_concurrency > 0)
        self._slot_indices = {provider: getattr(keys, "slot_indices", tuple(range(len(keys))))
                              for provider, keys in credentials.items()}
        self._clients = {
            provider: tuple((relay_factory if provider == config.MINIMAX_RELAY_LANE else factory)(base_url=config.credential_endpoint(provider, slot, mimo_api=self.mimo_api if provider == "mimo" else "standard"), api_key=key)
                            for slot, key in zip(self._slot_indices[provider], keys))
            for provider, keys in credentials.items()
        }
        self.provider_limits = {provider: len(clients) for provider, clients in self._clients.items()}
        if "minimax" in self.provider_limits and self.minimax_concurrency is not None:
            self.provider_limits["minimax"] = min(self.provider_limits["minimax"], self.minimax_concurrency)
        self._key_capacity = {provider: 1 for provider in self._clients}
        if self.mimo_api in ("standard", "ezlook") and "mimo" in self._clients:
            self.provider_limits["mimo"] = self.mimo_concurrency
            self._key_capacity["mimo"] = self.mimo_concurrency
        if config.MIMO_OFFICIAL_LANE in self._clients:
            self.provider_limits[config.MIMO_OFFICIAL_LANE] = self.mimo_official_concurrency
            self._key_capacity[config.MIMO_OFFICIAL_LANE] = self.mimo_official_concurrency
        if config.MINIMAX_RELAY_LANE in self._clients:
            self.provider_limits[config.MINIMAX_RELAY_LANE] = self.minimax_relay_concurrency
            self._key_capacity[config.MINIMAX_RELAY_LANE] = self.minimax_relay_concurrency
        self._inflight = {provider: [0] * len(clients) for provider, clients in self._clients.items()}
        self.max_in_flight = sum(self.provider_limits.values())
        if not (1 if mimo_batch else 2) <= self.max_in_flight <= config.EXPERIMENT_MAX_IN_FLIGHT:
            raise ValueError("provider pool credential capacity is outside configured bounds")
        self._condition = threading.Condition()
        self._quota_provider: str | None = None
        self._groups = tuple(config.EXPERIMENT_GENERATORS)
        self._queues: dict[str, dict[str, deque[_Ticket]]] = {
            provider: {group: deque() for group in self._groups} for provider in self._clients}
        self._next_group = {provider: 0 for provider in self._clients}
        self._free = {provider: deque(range(len(clients))) for provider, clients in self._clients.items()}
        self._clock = clock
        self._schedule = schedule or self._timer
        self._budgets = {(p, s): TokenBudget() for p, clients in self._clients.items()
                         if p == "siliconflow" for s in range(len(clients))}
        for lane in config.MIMO_LANES:
            if lane in self._clients and self.rate_limited(lane):
                self._budgets.update({(lane, model): TokenBudget(config.MIMO_RATE_POLICY)
                    for model in config.EXPERIMENT_MODEL_EXTRA_BODY if model.startswith("mimo-")})
        self._cooldown = {key: 0.0 for key in self._budgets}
        self._startup = clock() + config.SILICONFLOW_RATE_POLICY["startup_seconds"]
        self._mimo_startup = clock() + config.MIMO_RATE_POLICY["startup_seconds"]
        self._alarms: dict[str, tuple[float, Any, object]] = {}
        self._diagnostics = {p: {"wait_counts": {}, "wait_seconds": 0.0} for p in self._clients}

    def legacy_batch_client(self):
        """Load the dedicated key only when recovering an existing Batch task."""
        with self._condition:
            if self.batch_client is None:
                from nea.providers.batch import MiMoBatchClient
                self.batch_client = MiMoBatchClient(config.load_mimo_batch_credential(self._env_file))
            return self.batch_client

    def lane_for(self, model: str) -> str:
        """Scheduling lane of a model (static); admission may still split MiMo between its two routes."""
        return config.lane_for_model(model, self.minimax_m3_api)

    def admission_lane(self, model: str) -> str:
        """Pool provider that will serve this request: the less loaded MiMo route when both are enabled."""
        lane = self.lane_for(model)
        if lane != "mimo" or config.MIMO_OFFICIAL_LANE not in self._clients or self._official_disabled:
            return lane
        with self._condition:
            def load(name):
                queued = sum(len(queue) for queue in self._queues[name].values())
                return (sum(self._inflight[name]) + queued) / max(1, self.provider_limits[name])
            return config.MIMO_OFFICIAL_LANE if load(config.MIMO_OFFICIAL_LANE) <= load("mimo") else "mimo"

    def disable_official(self) -> None:
        """The official route failed hard (quota/auth): keep MiMo on ezlook for the rest of this process."""
        self._official_disabled = True

    def rate_limited(self, provider: str) -> bool:
        return (provider == "siliconflow" or (provider == "mimo" and self.mimo_api == "standard")
                or provider == config.MIMO_OFFICIAL_LANE)

    def _budget_key(self, provider, slot, ticket):
        return (provider, ticket.model if provider in config.MIMO_LANES else slot)

    @staticmethod
    def _timer(delay, callback):
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()
        return timer

    def public_diagnostics(self):
        with self._condition:
            return {p: {"wait_counts": dict(d["wait_counts"]),
                        "wait_seconds": round(d["wait_seconds"], 3),
                        **({"stop_code": "provider_quota_exhausted",
                            "quota_provider": self._quota_provider} if self._quota_provider else {})}
                    for p, d in self._diagnostics.items()}

    def _check_quota_stop(self):
        if self._quota_provider is not None:
            raise LLMTransportError("Provider pool stopped after quota exhaustion",
                                    code="provider_quota_exhausted")

    def stop_for_quota(self, provider: str):
        """Stop every queue, across generators; existing sent requests settle."""
        with self._condition:
            if provider not in self._clients and not (self.batch_mode and provider == "mimo"):
                raise ValueError("quota provider is outside the configured pool")
            self._quota_provider = self._quota_provider or provider
            for current, queues in self._queues.items():
                self._alarm(current, None)
                for queue in queues.values():
                    for ticket in queue:
                        if ticket.ready is not None:
                            ticket.ready.get_loop().call_soon_threadsafe(self._wake_quota, ticket.ready)
            self._condition.notify_all()

    @staticmethod
    def _wake_quota(ready):
        if not ready.done():
            ready.set_exception(LLMTransportError("Provider pool stopped after quota exhaustion",
                                                 code="provider_quota_exhausted"))

    def _alarm(self, provider, deadline):
        previous = self._alarms.get(provider)
        if previous and previous[0] == deadline:
            return
        if previous:
            previous[1].cancel()
            del self._alarms[provider]
        if deadline is None:
            return
        generation = object()

        def wake():
            with self._condition:
                alarm = self._alarms.get(provider)
                if alarm is not None and alarm[2] is generation:
                    del self._alarms[provider]
                    self._dispatch(provider)

        handle = self._schedule(max(0.0, deadline - self._clock()), wake)
        self._alarms[provider] = (deadline, handle, generation)

    def _slot_ready(self, provider, slot, ticket, now):
        key = self._budget_key(provider, slot, ticket)
        budget = self._budgets.get(key)
        if budget is None:
            return now, None
        ready = budget.ready_at(ticket.tokens, now)
        startup = self._mimo_startup if provider in config.MIMO_LANES else self._startup
        if now < startup:
            return max(startup, ready or now, self._cooldown[key]), "startup"
        if now < self._cooldown[key]:
            return max(self._cooldown[key], ready or now), "cooldown"
        return ready, "token" if ready is None or ready > now else None

    def _dispatch(self, provider: str) -> None:
        """Called while holding the condition; no unavailable group blocks admission."""
        if self._quota_provider is not None:
            return
        queues = self._queues[provider]
        now = self._clock()
        deadlines = []
        while self._free[provider] and sum(self._inflight[provider]) < self.provider_limits[provider]:
            for offset in range(len(self._groups)):
                index = (self._next_group[provider] + offset) % len(self._groups)
                queue = queues[self._groups[index]]
                if queue:
                    available = None
                    # Independent model budgets must not block each other within a generator.
                    for candidate in (queue if provider in config.MIMO_LANES else [queue[0]]):
                        for slot in self._free[provider]:
                            ready, reason = self._slot_ready(provider, slot, candidate, now)
                            if ready == now:
                                available, ticket = slot, candidate
                                break
                            candidate.wait_reason = reason
                            if ready is not None:
                                deadlines.append(ready)
                        if available is not None:
                            break
                    if available is None:
                        continue
                    queue.remove(ticket)
                    ticket.slot = available
                    self._free[provider].remove(available)
                    self._inflight[provider][available] += 1
                    if self._inflight[provider][available] < self._key_capacity[provider]:
                        self._free[provider].append(available)
                    ticket.budget_key = self._budget_key(provider, available, ticket)
                    budget = self._budgets.get(ticket.budget_key)
                    if budget is not None:
                        ticket.charge = budget.reserve(ticket.tokens, now)
                    self._record_wait(ticket, now)
                    if ticket.ready is not None:
                        ticket.ready.get_loop().call_soon_threadsafe(self._wake, ticket.ready, ticket.slot)
                    self._next_group[provider] = (index + 1) % len(self._groups)
                    break
            else:
                break
        # Only waiters on free slots need a deadline; busy slots dispatch on release.
        waiting = any(queues[g] for g in self._groups)
        at_capacity = sum(self._inflight[provider]) >= self.provider_limits[provider]
        if waiting and (not self._free[provider] or at_capacity):
            for queue in queues.values():
                if queue and queue[0].wait_reason is None:
                    queue[0].wait_reason = "capacity"
        self._alarm(provider, min(deadlines) if waiting and deadlines and not at_capacity else None)
        self._condition.notify_all()

    def _record_wait(self, ticket, now):
        if ticket.wait_reason:
            diagnostics = self._diagnostics[ticket.provider]
            counts = diagnostics["wait_counts"]
            counts[ticket.wait_reason] = counts.get(ticket.wait_reason, 0) + 1
            diagnostics["wait_seconds"] += max(0.0, now - ticket.requested)
            ticket.wait_reason = None

    @staticmethod
    def _wake(ready: asyncio.Future[int], slot: int) -> None:
        if not ready.done():
            ready.set_result(slot)

    def _request(self, provider: str, generator: str,
                 ready: asyncio.Future[int] | None = None, tokens: int | None = None, model: str | None = None) -> _Ticket:
        if provider not in self._clients:
            raise ValueError("provider is outside the configured pool")
        group = config.experiment_role_models_for(generator)["student1"]
        if self.rate_limited(provider):
            if provider in config.MIMO_LANES:
                if model not in config.EXPERIMENT_MODEL_EXTRA_BODY or not model.startswith("mimo-"):
                    raise ValueError("MiMo admission requires the request model")
                TokenBudget(config.MIMO_RATE_POLICY).validate(tokens)
            else:
                TokenBudget().validate(tokens)
        ticket = _Ticket(provider, group, ready=ready, tokens=tokens, model=model, requested=self._clock())
        with self._condition:
            self._check_quota_stop()
            self._queues[provider][group].append(ticket)
            self._dispatch(provider)
        return ticket

    def _wait(self, ticket: _Ticket) -> int:
        with self._condition:
            while ticket.slot is None and not ticket.released:
                self._check_quota_stop()
                self._condition.wait()
            if ticket.released:
                raise RuntimeError("provider lease was cancelled")
            assert ticket.slot is not None
            return ticket.slot

    def _release(self, ticket: _Ticket) -> None:
        with self._condition:
            if ticket.released:
                return
            ticket.released = True
            if ticket.slot is None:
                self._queues[ticket.provider][ticket.generator].remove(ticket)
                self._record_wait(ticket, self._clock())
            else:
                if ticket.charge is not None:
                    budget = self._budgets[ticket.budget_key]
                    if ticket.sent:
                        budget.settle(ticket.charge, None, self._clock())
                    else:
                        budget.refund(ticket.charge)
                self._inflight[ticket.provider][ticket.slot] -= 1
                if ticket.slot not in self._free[ticket.provider]:
                    self._free[ticket.provider].append(ticket.slot)
            self._dispatch(ticket.provider)

    def _settle(self, ticket, usage):
        with self._condition:
            if ticket.charge is not None:
                self._budgets[ticket.budget_key].settle(ticket.charge, usage, self._clock())
                self._dispatch(ticket.provider)

    def _limited(self, ticket, delay):
        with self._condition:
            key = ticket.budget_key
            if key in self._cooldown:
                self._cooldown[key] = max(self._cooldown[key], self._clock() + delay)
                self._dispatch(ticket.provider)

    @asynccontextmanager
    async def lease(self, provider: str, generator: str):
        """Yield ``(one_based_slot, client)``; cancellation cannot strand a queue slot."""
        # Wake the owning event loop directly.  Blocking executor threads for
        # queued leases could starve the same executor used for HTTP requests.
        async with self.admit(provider, generator) as admission:
            yield admission.slot, admission.client

    async def _ezlook_gate(self) -> None:
        """Space request starts so the relay's per-second limit is not hit; concurrency stays high."""
        with self._gate_lock:
            now = time.monotonic()
            start = max(now, self._gate_next)
            self._gate_next = start + config.EZLOOK_MIN_START_INTERVAL_SECONDS
        if start > now:
            await asyncio.sleep(start - now)

    @asynccontextmanager
    async def admit(self, provider: str, generator: str, tokens: int | None = None, *, model: str | None = None):
        ready = asyncio.get_running_loop().create_future()
        ticket = self._request(provider, generator, ready, tokens, model)
        try:
            await ready
            with self._condition:
                self._check_quota_stop()
            if provider == "mimo" and self.mimo_api == "ezlook":
                await self._ezlook_gate()
            yield Admission(self, ticket)
        except LLMTransportError as exc:
            if exc.code == "provider_quota_exhausted":
                self.stop_for_quota(provider)
            raise
        finally:
            self._release(ticket)
