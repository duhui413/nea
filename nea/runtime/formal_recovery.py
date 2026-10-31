"""Explicit batch recovery grants; immutable attempts and first valid outputs."""
from __future__ import annotations
import json
import time
from pathlib import Path
from nea.runtime.records import ConflictError, digest, write_once

POLICY = {"version": 1, "retries": 7, "cooldown_seconds": 3600,
          "selection": "first_valid_response", "authorization": "user-2026-10-07"}


def batch_root(run):
    root = run.root
    return root.parents[2] if root.parents[1].name == "operator-v2" else root


def enabled(run):
    path = batch_root(run) / "recovery-policy.json"
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf8")) != POLICY:
        raise ConflictError("Unknown recovery policy")
    return True


def grant(run, key, request_hash, rows):
    rounds = sorted((batch_root(run) / "supervision" / "rounds").glob("*.json"))
    if not rounds:
        raise ConflictError("Recovery has no registered supervisor round")
    round_id = rounds[-1].stem
    path = run.root / "recovery" / key / (round_id + ".json")
    if path.exists():
        body = json.loads(path.read_text(encoding="utf8"))
        if body["request_hash"] != request_hash or body["key"] != key:
            raise ConflictError("Recovery grant input differs")
    else:
        body = {"key": key, "request_hash": request_hash, "round": round_id,
                "prior_attempts": len(rows), "additional_attempts": 8}
        write_once(path, body)
    return body["prior_attempts"]


def defer(run, entry, request_hash, attempt, code):
    body = {"key": entry.key, "request_hash": request_hash, "attempt": attempt,
            "code": code, "retry_at": time.time() + POLICY["cooldown_seconds"]}
    write_once(batch_root(run) / "supervision" / "cooldowns" /
               (digest([str(run.root), entry.key, attempt]) + ".json"), body)


def retry_at(run):
    return max((json.loads(p.read_text(encoding="utf8"))["retry_at"]
                for p in (batch_root(run) / "supervision" / "cooldowns").glob("*.json")), default=0)


def bind_imported_case(phase):
    receipt_path = batch_root(phase.run) / "case-import.json"
    if phase.env.phase != "case" or not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    manifest_path = phase.run.root / "run_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf8"))
    rel = manifest_path.relative_to(batch_root(phase.run)).as_posix()
    if receipt["files"].get(rel) != digest(manifest_path.read_bytes()):
        raise ConflictError("Imported case manifest differs from receipt")
    differences = {key for key in set(old) | set(phase.run.identity)
                   if digest(old.get(key)) != digest(phase.run.identity.get(key))}
    if differences - {"run_id", "method_source_hash"}:
        raise ConflictError("Imported case inputs differ: " + str(sorted(differences)))
    phase.run.identity = old
