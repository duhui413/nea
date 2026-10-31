"""Immutable successes and an append-only attempt journal for safe recovery."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
MAX_STRUCTURE_ATTEMPTS = 3

class ConflictError(RuntimeError):
    pass

def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf8")

def digest(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_bytes(value)).hexdigest()

def answer_sha256(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("answer text must be nonempty")
    return hashlib.sha256(text.encode("utf8")).hexdigest()

def write_once(path: Path, payload: Any) -> bool:
    data = canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ConflictError(f"Existing artifact differs: {path}")
        return False
    # The caller holds the phase lock. Atomic publication avoids partial successes.
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    return True


class AppendOnlyRun:
    """Resume immutable successes within one fully matching execution identity."""
    def __init__(self, root: Path, identity: dict):
        self.root, self.identity = root, identity

    _write_once = staticmethod(write_once)

    def initialize(self) -> None:
        path = self.root / "run_manifest.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf8"))
            if canonical_bytes(old) != canonical_bytes(self.identity):
                raise ConflictError("Existing run identity differs; choose a new run id")
        else:
            self._write_once(path, self.identity)

    def success(self, key: str) -> dict | None:
        path = self.root / "success" / f"{key}.json"
        return json.loads(path.read_text(encoding="utf8")) if path.exists() else None

    def put_success(self, key: str, request_hash: str, payload: dict) -> bool:
        return self._write_once(self.root / "success" / f"{key}.json",
            {"key": key, "request_hash": request_hash, "payload": payload, "payload_hash": digest(payload)})

    def put_failure(self, key: str, request_hash: str, payload: dict) -> None:
        """Append safe failure evidence, without raw responses or credentials."""
        directory = self.root / "failures" / key
        number = len(list(directory.glob("*.json"))) + 1
        self._write_once(directory / f"{number:04d}.json",
                         {"key": key, "request_hash": request_hash, **payload})

    def _event(self, key: str, request_hash: str, number: int, event: str, **fields: Any) -> dict:
        body = {"version": 2 if event == "outcome" else 1, "key": key, "request_hash": request_hash,
                "attempt": number, "event": event, **fields}
        return {**body, "event_hash": digest(body)}

    def attempts(self, key: str, request_hash: str) -> list[dict]:
        directory = self.root / "attempts" / key
        paths = sorted(directory.glob("*.json"))
        reservations = sorted(directory.glob("*-reserved.json"))
        expected = set()
        result = []
        for number, path in enumerate(reservations, 1):
            if path.name != f"{number:04d}-reserved.json":
                raise ConflictError("Attempt journal sequence is corrupt")
            expected.add(path.name)
            reserved = self._read_event(path, key, request_hash, number, "reserved")
            outcome_path = directory / f"{number:04d}-outcome.json"
            outcome = None
            if outcome_path.exists():
                expected.add(outcome_path.name)
                outcome = self._read_event(outcome_path, key, request_hash, number, "outcome")
            elif number != len(reservations):
                raise ConflictError("Only the last attempt may be unresolved")
            result.append({"reservation": reserved, "outcome": outcome})
        if {p.name for p in paths} != expected:
            raise ConflictError("Unknown attempt journal file")
        return result

    @staticmethod
    def _read_event(path: Path, key: str, request_hash: str, number: int, event: str) -> dict:
        try:
            document = json.loads(path.read_text(encoding="utf8"))
            body = {k: v for k, v in document.items() if k != "event_hash"}
            fields = {"version", "key", "request_hash", "attempt", "event", "event_hash"}
            if event == "outcome":
                fields |= {"result", "code", "usage", "retryable", "retry_at"}
                if document.get("version") == 2:
                    fields.add("validation_reason")
            if (set(document) != fields or document["event_hash"] != digest(body)
                    or type(document["version"]) is not int
                    or document["version"] not in ({1, 2} if event == "outcome" else {1}) or document["key"] != key
                    or document["request_hash"] != request_hash or document["attempt"] != number
                    or document["event"] != event):
                raise ValueError("identity")
            if event == "outcome":
                import math
                if (document["result"] not in {"success", "structure_error", "transport_error", "unknown"}
                        or type(document["retryable"]) is not bool or not isinstance(document["code"], str)
                        or (document["retry_at"] is not None and
                            (type(document["retry_at"]) not in {int, float}
                             or not math.isfinite(document["retry_at"]) or document["retry_at"] < 0))):
                    raise ValueError("outcome")
                usage = document["usage"]
                if usage is not None and (not isinstance(usage, dict)
                        or set(usage) != {"prompt_tokens", "completion_tokens"}
                        or any(type(v) is not int or v < 0 for v in usage.values())):
                    raise ValueError("usage")
                reason = document.get("validation_reason")
                if reason is not None and (document["result"] != "structure_error"
                        or not isinstance(reason, str) or not reason or len(reason) > 600
                        or any(ord(char) < 32 or ord(char) == 127 for char in reason)):
                    raise ValueError("validation reason")
            return document
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ConflictError("Attempt journal is corrupt or has a different request identity") from exc

    def reserve_attempt(self, key: str, request_hash: str) -> int:
        rows = self.attempts(key, request_hash)
        if rows and rows[-1]["outcome"] is None:
            raise ConflictError("Unresolved attempt must be recovered before a new call")
        number = len(rows) + 1
        self._write_once(self.root / "attempts" / key / f"{number:04d}-reserved.json",
                         self._event(key, request_hash, number, "reserved"))
        return number

    def finish_attempt(self, key: str, request_hash: str, number: int, *, result: str, code: str,
                       usage: dict | None = None, retryable: bool = False, retry_at: float | None = None,
                       validation_reason: str | None = None) -> None:
        rows = self.attempts(key, request_hash)
        if not rows or number != len(rows):
            raise ConflictError("Attempt outcome has no matching last reservation")
        document = self._event(key, request_hash, number, "outcome", result=result, code=code,
                               usage=usage, retryable=retryable, retry_at=retry_at,
                               validation_reason=validation_reason)
        path = self.root / "attempts" / key / f"{number:04d}-outcome.json"
        self._write_once(path, document)
        self._read_event(path, key, request_hash, number, "outcome")

    def terminal(self, key: str, request_hash: str) -> dict | None:
        path = self.root / "terminal" / f"{key}.json"
        if not path.exists():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf8"))
            body = {k: v for k, v in document.items() if k != "state_hash"}
            if (set(document) != {"version", "key", "request_hash", "status", "code", "state_hash"}
                    or document["version"] != 1 or document["key"] != key
                    or document["request_hash"] != request_hash or document["state_hash"] != digest(body)
                    or document["status"] not in {"structure_exhausted", "transport_exhausted", "permanent_error"}
                    or not isinstance(document["code"], str)):
                raise ValueError("terminal state")
            return document
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ConflictError("Terminal state is corrupt or has a different request identity") from exc

    def put_terminal(self, key: str, request_hash: str, status: str, code: str) -> None:
        body = {"version": 1, "key": key, "request_hash": request_hash, "status": status, "code": code}
        self._write_once(self.root / "terminal" / f"{key}.json", {**body, "state_hash": digest(body)})
