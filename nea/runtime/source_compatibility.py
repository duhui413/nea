"""Exact, reviewed realtime-source compatibility for optional Batch extensions.

Only this audited old/new source pair can preserve a pre-existing manifest.
Every other locked field must remain identical. Batch never uses this bridge.
"""
from __future__ import annotations

import json
from pathlib import Path

from nea.runtime.records import canonical_bytes, digest, write_once

# Source pairs reviewed in 0035/0037; model identities must still be identical.
REALTIME_SOURCE_PAIRS: dict[str, str] = {'de62256bc37e9bba9ddefc6362472a5070bf78e6a048fd4b1197272737ac0ca8': 'f8a170929ad3e99f582f683491da524a34416ca32bb1a760d347f604f9175c73', 'e631e0bc9b36e47d3e82450e867bc67b5124d977e0d3020b3fc5ab3b94f6de01': 'f133c96dea7b97d8c498e74ad4f0334f4fd47f9bf3c37d5c51fbbc714ec2f104'}


def bind_realtime_identity(root: Path, identity: dict, *, mimo_batch: bool = False) -> dict:
    path = root / "run_manifest.json"
    if mimo_batch or not path.is_file():
        return identity
    old = json.loads(path.read_text(encoding="utf8"))
    previous, current = old.get("method_source_hash"), identity.get("method_source_hash")
    if previous == current or REALTIME_SOURCE_PAIRS.get(previous) != current:
        return identity
    candidate = {**identity, "method_source_hash": previous}
    if canonical_bytes(candidate) != canonical_bytes(old):
        return identity
    receipt = {"spec": "0035-mimo-batch-inference", "transport": "realtime",
        "original_method_source_hash": previous, "current_method_source_hash": current,
        "execution_identity_hash": digest(old),
        "scope": "reviewed optional Batch dispatch only; all other manifest fields identical"}
    write_once(root / "source_compatibility" / (digest(receipt) + ".json"), receipt)
    return old
