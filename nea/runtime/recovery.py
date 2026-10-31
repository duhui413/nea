"""Explicit cross-run recovery of immutable completed request states.

The caller must hold the repository CLI lock and the target phase lock. This
module never alters the source or resets a journal within an existing run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nea.pipeline.sampling import check_mode, validate_sample_scope
from nea.pipeline.registry import structure_attempts, check_arm_stage, role_for_stage
from nea.providers import config
from nea.runtime.identity import phase_directory, operator_phase_directory
from nea.runtime import operator_protocol
from nea.runtime.records import AppendOnlyRun, ConflictError, canonical_bytes, digest, write_once


_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _phase_directory(run_root: Path, identity: dict) -> Path:
    factory = operator_phase_directory if operator_protocol.is_operator_identity(identity) else phase_directory
    return factory(run_root, identity["generator"], identity["split"])


def _document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise ValueError("noncanonical object")
        return value, raw
    except (OSError, ValueError, TypeError) as exc:
        raise ConflictError("Recovery source contains a malformed or noncanonical JSON artifact") from exc


def _scope(source: Path, target_root: Path, identity: dict) -> dict[str, str]:
    """Equal scope bytes, plus identity binding, also when this phase is absent."""
    names = ("sample_scope.json", "smoke_scope.json")
    required = "sample_scope.json" if identity.get("exploratory") else "smoke_scope.json" if identity.get("smoke") else None
    hashes = {}
    for name in names:
        left, right = source / name, target_root / name
        if name == required and (not left.is_file() or not right.is_file()):
            raise ConflictError("Recovery source and target require the same scope")
        if name != required and (left.exists() or right.exists()):
            raise ConflictError("Recovery cannot mix formal, smoke and exploratory scopes")
        if name != required:
            continue
        document, raw = _document(left)
        _, target_raw = _document(right)
        if raw != target_raw:
            raise ConflictError("Recovery source and target scopes differ")
        if name == "sample_scope.json":
            size = validate_sample_scope(document)
            check_mode(identity, sample_size=size)
            if document["question_set_sha256"] != identity["question_set_sha256"]:
                raise ConflictError("Recovery scope and manifest question identities differ")
        else:
            check_mode(identity, smoke=True)
        hashes[name] = digest(raw)
    return hashes


def _keys(root: Path, directory: str, *, folders: bool = False) -> set[str]:
    path = root / directory
    if not path.exists():
        return set()
    found = set()
    for item in path.iterdir():
        key = item.name if folders else item.stem
        if (not _HASH.fullmatch(key) or item.is_symlink() or item.is_dir() != folders
                or not folders and item.suffix != ".json"):
            raise ConflictError("Recovery source contains unknown request-state files")
        if folders and any(not f.is_file() or f.suffix != ".json" or f.is_symlink() for f in item.iterdir()):
            raise ConflictError("Recovery source contains unfinished or unknown journal files")
        found.add(key)
    return found


def _verify_receipt(receipt: dict, source: Path, target: AppendOnlyRun, scope_hashes: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if (receipt.get("receipt_hash") != digest(body) or receipt.get("version") != 1
            or receipt.get("source_run_id") != source.name
            or receipt.get("target_run_id") != target.identity["run_id"]
            or receipt.get("generator") != target.identity["generator"]
            or receipt.get("split") != target.identity["split"]
            or receipt.get("target_manifest_hash") != digest(target.identity)
            or receipt.get("scope_hashes") != scope_hashes):
        raise ConflictError("Recovery receipt identity differs or is corrupt")
    source_phase = _phase_directory(source, target.identity)
    source_hash = receipt.get("source_manifest_hash")
    if source_hash is not None and digest((source_phase / "run_manifest.json").read_bytes()) != source_hash:
        raise ConflictError("Recovery source manifest changed")
    for row in receipt["copied_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"success", "terminal", "attempts", "telemetry", "failures"}:
            raise ConflictError("Recovery receipt contains an invalid artifact path")
        if digest((source_phase / relative).read_bytes()) != row["sha256"] or digest((target.root / relative).read_bytes()) != row["sha256"]:
            raise ConflictError("Previously imported artifact differs from its recovery receipt")
    return receipt


def import_phase(source_run_root: Path, target: AppendOnlyRun) -> dict[str, Any]:
    """Import only completed successes/structure terminals, validating before writes."""
    source = Path(source_run_root)
    identity = target.identity
    if "structure_attempts_by_provider" in identity:
        raise ConflictError("Custom structure budgets support same-directory resume only")
    operator = operator_protocol.is_operator_identity(identity)
    target_root = target.root.parents[2] if operator else target.root.parent.parent
    if (not source.is_dir() or source.resolve() == target_root.resolve()
            or identity.get("run_id") != target_root.name):
        raise ConflictError("Recovery requires an existing, different source run and matching target run_id")
    if identity.get("split") not in (operator_protocol.PHASES if operator else {"dev", "test"}):
        raise ConflictError("Recovery phase is not supported by its protocol")
    if operator and (identity.get("method_version") != operator_protocol.METHOD_VERSION
                     or identity.get("question_split") != operator_protocol.source_split(identity["split"])):
        raise ConflictError("Recovery operator method or question scope differs")
    if operator and identity.get("origin", "REAL") not in {"REAL", "OFFLINE_FIXTURE"}:
        raise ConflictError("Recovery operator artifact origin is unknown")
    scope_hashes = _scope(source, target_root, identity)
    manifest_path = target.root / "run_manifest.json"
    if manifest_path.exists() and _document(manifest_path)[0] != identity:
        raise ConflictError("Recovery target manifest differs")
    receipt_path = target.root / "recovery_receipt.json"
    if receipt_path.exists():
        return _verify_receipt(_document(receipt_path)[0], source, target, scope_hashes)

    source_phase = _phase_directory(source, identity)
    source_manifest = source_phase / "run_manifest.json"
    source_exists = source_manifest.is_file()
    if not source_exists:
        raise ConflictError("Source phase does not exist; start this phase without --resume-from")
    files: dict[str, bytes] = {}
    omitted = []
    source_identity, raw = _document(source_manifest)
    if "structure_attempts_by_provider" in source_identity:
        raise ConflictError("Custom structure budgets support same-directory resume only")
    expected = {**identity, "run_id": source.name}
    if source_identity != expected:
        raise ConflictError("Recovery manifests may differ only in run_id")
    source_manifest_hash = digest(raw)
    source_run = AppendOnlyRun(source_phase, source_identity)

    success_keys = _keys(source_phase, "success")
    terminal_keys = _keys(source_phase, "terminal")
    attempt_keys = _keys(source_phase, "attempts", folders=True)
    failure_keys = _keys(source_phase, "failures", folders=True)
    telemetry_keys = _keys(source_phase, "telemetry", folders=True) if operator else set()
    all_keys = success_keys | terminal_keys | attempt_keys | failure_keys | telemetry_keys
    for key in sorted(all_keys):
        if key not in attempt_keys:
            raise ConflictError("Recovery request has no attempt journal")
        reserved = source_phase / "attempts" / key / "0001-reserved.json"
        first, _ = _document(reserved)
        request_hash = first.get("request_hash")
        if not isinstance(request_hash, str) or not _HASH.fullmatch(request_hash):
            raise ConflictError("Recovery request hash is malformed")
        rows = source_run.attempts(key, request_hash)
        terminal = source_run.terminal(key, request_hash)
        success = source_run.success(key)
        outcomes = [row["outcome"] for row in rows]
        completed = all(outcome is not None for outcome in outcomes)
        structural = sum(outcome is not None and outcome["result"] == "structure_error" for outcome in outcomes)
        transport = sum(outcome is not None and outcome["result"] in {"transport_error", "unknown"} for outcome in outcomes)
        if structural > 3 or transport > config.TRANSPORT_MAX_FAILURES:
            raise ConflictError("Recovery source exceeded its request budget")
        telemetry_files = []
        for path in sorted((source_phase / "telemetry" / key).glob("*.json")) if operator else ():
            document, _ = _document(path)
            body = {name: value for name, value in document.items() if name != "telemetry_hash"}
            if (document.get("telemetry_hash") != digest(body) or document.get("key") != key
                    or document.get("request_hash") != request_hash or document.get("protocol") != operator_protocol.PROTOCOL
                    or type(document.get("attempt")) is not int or not 1 <= document["attempt"] <= len(rows)
                    or path.name != f"{document['attempt']:04d}.json"):
                raise ConflictError("Recovery telemetry identity differs or is corrupt")
            telemetry_files.append(path.relative_to(source_phase).as_posix())
        copied = []
        if success is not None:
            payload = success.get("payload", {})
            usage = payload.get("usage")
            if (set(success) != {"key", "request_hash", "payload", "payload_hash"}
                    or success["key"] != key or success["request_hash"] != request_hash
                    or success["payload_hash"] != digest(payload)
                    or payload.get("origin") != (identity.get("origin", "REAL") if operator else "REAL")
                    or not isinstance(payload.get("text"), str) or not isinstance(usage, dict)
                    or set(usage) != {"prompt_tokens", "completion_tokens"}
                    or any(type(value) is not int or value < 0 for value in usage.values())
                    or terminal is not None or not completed or not outcomes
                    or outcomes[-1]["result"] != "success"
                    or any(outcome["result"] == "success" for outcome in outcomes[:-1])
                    or outcomes[-1]["usage"] != usage or payload.get("attempt_number") != len(rows)
                    or payload.get("attempt") != structural + 1
                    or structural + 1 > structure_attempts(payload.get("stage"))):
                raise ConflictError("Recovery success and its complete journal disagree")
            try:
                stage, role = payload["stage"], payload["role"]
                question_split = identity["question_split"] if operator else identity["split"]
                (operator_protocol.check_arm_stage if operator else check_arm_stage)(payload["arm"], stage, question_split)
                model = config.experiment_role_models_for(identity["generator"])[role]
                markdown = stage in (operator_protocol.MARKDOWN_STAGES if operator else {"answer", "b_answer", "revise"})
                if (role not in (operator_protocol.role_for_stage if operator else role_for_stage)(stage)
                        or payload["call"]["model"] != model
                        or payload["call"]["provider"] != config.experiment_provider_name_for_model(model)
                        or payload["call"]["status"] != "ok"
                        or payload["output_mode"] != ("markdown" if markdown else config.structured_output_mode(model))
                        or (payload["schema_hash"] is not None if markdown else not isinstance(payload["schema_hash"], str)
                            or not _HASH.fullmatch(payload["schema_hash"]))):
                    raise ValueError("response identity")
            except (KeyError, TypeError, ValueError) as exc:
                raise ConflictError("Recovery success response identity differs") from exc
            copied.append(f"success/{key}.json")
        elif terminal is not None and terminal["status"] == "structure_exhausted":
            if (not completed or not outcomes or outcomes[-1]["result"] != "structure_error"
                    or structural not in {1, 3} or any(outcome["result"] == "success" for outcome in outcomes)):
                raise ConflictError("Recovery structure terminal and its journal disagree")
            if operator:
                failures = [_document(path)[0] for path in sorted((source_phase / "failures" / key).glob("*.json"))]
                stages = {document.get("stage") for document in failures}
                if len(stages) != 1 or structural != structure_attempts(next(iter(stages))):
                    raise ConflictError("Recovery operator structure terminal exceeded its stage budget")
            copied.append(f"terminal/{key}.json")
        else:
            if any(outcome is not None and outcome["result"] == "success" for outcome in outcomes):
                raise ConflictError("Source journal has a success without its response")
            if terminal is not None:
                if (not completed or terminal["status"] == "transport_exhausted" and transport != config.TRANSPORT_MAX_FAILURES
                        or terminal["status"] == "permanent_error" and (not outcomes
                            or outcomes[-1]["result"] != "transport_error" or outcomes[-1]["retryable"])):
                    raise ConflictError("Source transport terminal contradicts its journal")
            omitted.append({"key": key, "request_hash": request_hash,
                            "status": terminal["status"] if terminal else "unresolved" if not completed else "incomplete"})
        if copied:
            copied.extend(telemetry_files)
            if operator:
                copied.extend(path.relative_to(source_phase).as_posix()
                              for path in sorted((source_phase / "failures" / key).glob("*.json")))
            copied.extend(path.relative_to(source_phase).as_posix()
                          for path in sorted((source_phase / "attempts" / key).glob("*.json")))
            for relative in copied:
                _, raw = _document(source_phase / relative)
                files[relative] = raw

    # Check all destination collisions before copying any request or manifest.
    copied_keys = {Path(relative).stem for relative in files
                   if relative.startswith(("success/", "terminal/"))}
    for key in copied_keys:
        existing = [target.root / directory / f"{key}.json" for directory in ("success", "terminal")]
        existing += list((target.root / "attempts" / key).glob("*"))
        if operator:
            existing += list((target.root / "telemetry" / key).glob("*"))
            existing += list((target.root / "failures" / key).glob("*"))
        if any(path.exists() and path.relative_to(target.root).as_posix() not in files for path in existing):
            raise ConflictError("Recovery target has additional state for an imported request")
    for relative, raw in files.items():
        destination = target.root / relative
        if destination.exists() and destination.read_bytes() != raw:
            raise ConflictError("Recovery target already contains a conflicting request artifact")
    body = {"version": 1, "source_run_id": source.name, "target_run_id": identity["run_id"],
            "generator": identity["generator"], "split": identity["split"], "source_phase_exists": source_exists,
            "source_manifest_hash": source_manifest_hash, "target_manifest_hash": digest(identity),
            "scope_hashes": scope_hashes,
            "copied_files": [{"path": relative, "sha256": digest(raw)} for relative, raw in sorted(files.items())],
            "omitted_requests": omitted}
    receipt = {**body, "receipt_hash": digest(body)}
    target.initialize()
    for relative, raw in sorted(files.items()):
        write_once(target.root / relative, json.loads(raw))
    write_once(receipt_path, receipt)
    return receipt
