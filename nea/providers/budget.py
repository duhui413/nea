"""Conservative in-memory token charges; callers serialize access in the pool."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nea.providers import config


class AdmissionError(RuntimeError):
    code = "token_budget_request_too_large"


def estimate_tokens(body: dict) -> int:
    maximum = body["max_tokens"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_tokens must be a positive integer")
    envelope = {k: v for k, v in body.items()
                if k not in {"model", "temperature", "max_tokens", "seed"}}
    encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return len(encoded) + config.SILICONFLOW_RATE_POLICY["envelope_overhead_tokens"] + maximum


def actual_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    if any(type(usage.get(k)) is not int or usage[k] < 0
           for k in ("prompt_tokens", "completion_tokens")):
        return None
    total = usage.get("total_tokens")
    return max(usage["prompt_tokens"] + usage["completion_tokens"],
               total if type(total) is int and total >= 0 else 0)


@dataclass(eq=False)
class Charge:
    started: float
    tokens: int
    previous_send: float
    settled: float | None = None


class TokenBudget:
    def __init__(self, policy=None):
        policy = config.SILICONFLOW_RATE_POLICY if policy is None else policy
        self.rpm = policy.get("rpm")
        self.limit = policy["tpm_per_key"]
        self.window = policy["window_seconds"]
        self.charges: list[Charge] = []
        self.next_send = 0.0

    def validate(self, tokens: int) -> None:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("token reservation must be a positive integer")
        if tokens > self.limit:
            raise AdmissionError("Request exceeds the local token budget")

    def ready_at(self, tokens: int, now: float) -> float | None:
        self.validate(tokens)
        self.charges = [c for c in self.charges
                        if c.settled is None or c.settled + self.window > now]
        needed = sum(c.tokens for c in self.charges) + tokens - self.limit
        deadline = max(now, self.next_send)
        if self.rpm is not None and len(self.charges) >= self.rpm:
            settled = sorted(c.settled for c in self.charges if c.settled is not None)
            excess = len(self.charges) - self.rpm + 1
            if len(settled) < excess:
                return None
            deadline = max(deadline, settled[excess - 1] + self.window)
        if needed > 0:
            for charge in sorted((c for c in self.charges if c.settled is not None),
                                 key=lambda c: c.settled):
                needed -= charge.tokens
                deadline = max(deadline, charge.settled + self.window)
                if needed <= 0:
                    break
            if needed > 0:
                return None  # A pending HTTP attempt must settle first.
        return deadline

    def reserve(self, tokens: int, now: float) -> Charge:
        if self.ready_at(tokens, now) != now:
            raise RuntimeError("token budget is not ready")
        charge = Charge(now, tokens, self.next_send)
        self.charges.append(charge)
        self.next_send = now + tokens * self.window / self.limit
        return charge

    def refund(self, charge: Charge) -> None:
        # Only an unsubmitted exclusive lease may refund its charge.
        if charge.settled is None and charge in self.charges:
            latest = self.charges[-1] is charge
            self.charges.remove(charge)
            if latest:
                self.next_send = charge.previous_send

    def settle(self, charge: Charge, usage: Any, now: float) -> None:
        if charge.settled is not None:
            return
        actual = actual_tokens(usage)
        if actual is not None:
            self.next_send = max(now, self.next_send + (actual - charge.tokens) * self.window / self.limit)
            charge.tokens = actual
        charge.settled = now
