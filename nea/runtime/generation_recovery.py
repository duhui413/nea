"""Explicit live continuation across judge/validator-only prompt repairs.

The source stays immutable. Matching generation journals are re-keyed with
per-request provenance; evaluation and semantic validation are never imported.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from nea.pipeline.registry import structure_attempts
from nea.providers import config
from nea.runtime.identity import model_config_identity, operator_phase_directory
from nea.runtime.records import AppendOnlyRun, ConflictError, digest, write_once
from nea.runtime.recovery import _document
from nea.runtime.transport import prepare, validate_structure

STAGES = ("prepare", "ground", "plan", "answer", "diagnose", "critique", "revise")
REPAIRED_ROLES = ("answer_quality_evaluation", "revision_validation")


class GenerationRecovery:
    def __init__(self, source_root: Path, target: AppendOnlyRun, *, endpoint_expansion=False):
        if "structure_attempts_by_provider" in target.identity:
            raise ConflictError("Custom structure budgets support same-directory resume only")
        self.target = target
        self.endpoint_expansion = endpoint_expansion
        self.stages = (*STAGES, "validate", "judge") if endpoint_expansion else STAGES
        self.source_model_identity = None
        self.root = operator_phase_directory(source_root, target.identity["generator"], "live")
        self.manifest, raw = _document(self.root / "run_manifest.json")
        if "structure_attempts_by_provider" in self.manifest:
            raise ConflictError("Custom structure budgets support same-directory resume only")
        self.source_hash = digest(raw)
        self.source = AppendOnlyRun(self.root, self.manifest)
        if (self.root.resolve() == target.root.resolve() or target.identity["split"] != "live"
                or self.manifest["run_id"] != source_root.name):
            raise ConflictError("Generation reuse requires a different live source")
        left, right = deepcopy(self.manifest), deepcopy(target.identity)
        if endpoint_expansion:
            current = model_config_identity(target.identity["generator"])
            legacy = deepcopy(current)
            for role in config.experiment_role_models_for(target.identity["generator"]):
                legacy[role].pop("credential_endpoints", None)
            if (self.manifest["model_config_hash"] != digest(legacy)
                    or target.identity["model_config_hash"] != digest(current)):
                raise ConflictError("Endpoint continuation permits only newly declared credential endpoints")
            self.source_model_identity = legacy
            left["model_config_hash"] = right["model_config_hash"]
        for item in (left, right):
            for field in ("run_id", "method_source_hash", "generation_reuse"):
                item.pop(field, None)
            for role in (() if endpoint_expansion else REPAIRED_ROLES):
                item["template_hashes"].pop(role, None)
                item["prompt_files"].pop(role + ".md", None)
        if left != right:
            raise ConflictError("Generation reuse permits only judge/validate prompts and runtime repair changes")

    def import_jobs(self, jobs, **kwargs):
        if digest((self.root / "run_manifest.json").read_bytes()) != self.source_hash:
            raise ConflictError("Generation source manifest changed")
        selected = [job for job in jobs if job.stage in self.stages]
        if not selected:
            return
        old, old_keys = prepare(self.source, self.manifest["generator"], selected,
                                _model_identity=self.source_model_identity, **kwargs)
        new, new_keys = prepare(self.target, self.target.identity["generator"], selected, **kwargs)
        old_by_key, new_by_key = {x.key: x for x in old}, {x.key: x for x in new}
        for job in selected:
            previous, current = old_by_key[old_keys[job]], new_by_key[new_keys[job]]
            normalized = deepcopy(current.request)
            if self.endpoint_expansion:
                normalized["model_request"].pop("credential_endpoints", None)
            if previous.request != normalized:
                raise ConflictError("Cannot reuse a changed generation request")
            self._copy(previous.key, current.key, digest(previous.request), digest(current.request), job)

    def _copy(self, old_key, new_key, source_request_hash, request_hash, job):
        receipt_path = self.target.root / "generation_reuse" / (new_key + ".json")
        if receipt_path.exists():
            receipt, _ = _document(receipt_path)
            if (receipt["source_manifest_hash"] != self.source_hash
                    or receipt["source_key"] != old_key or receipt["request_hash"] != request_hash
                    or receipt["source_request_hash"] != source_request_hash
                    or receipt["receipt_hash"] != digest({k: v for k, v in receipt.items() if k != "receipt_hash"})):
                raise ConflictError("Generation receipt identity differs")
            for item in receipt["files"]:
                if (digest((self.root / item["source"]).read_bytes()) != item["source_hash"]
                        or digest((self.target.root / item["target"]).read_bytes()) != item["target_hash"]):
                    raise ConflictError("Imported generation artifact changed")
            return
        rows = self.source.attempts(old_key, source_request_hash)
        if not rows:
            return
        outcomes = [r["outcome"] for r in rows]
        structural = sum(r is not None and r["result"] == "structure_error" for r in outcomes)
        transport = sum(r is None or r["result"] in {"transport_error", "unknown"} for r in outcomes)
        if structural > structure_attempts(job.stage) or transport > config.TRANSPORT_MAX_FAILURES:
            raise ConflictError("Source request exceeded its attempt budget")
        success = self.source.success(old_key)
        terminal = self.source.terminal(old_key, source_request_hash)
        if success is not None:
            payload = success["payload"]
            if (success["key"] != old_key or success["request_hash"] != source_request_hash
                    or success["payload_hash"] != digest(payload) or terminal is not None
                    or any(r is None for r in outcomes) or outcomes[-1]["result"] != "success"
                    or any(r["result"] == "success" for r in outcomes[:-1])
                    or payload["attempt_number"] != len(rows) or payload["attempt"] != structural + 1
                    or payload["usage"] != outcomes[-1]["usage"]
                    or payload["origin"] != self.target.identity["origin"]
                    or any(payload[name] != getattr(job, name) for name in ("qid", "base", "arm", "stage", "role"))):
                raise ConflictError("Source success and journal disagree")
            if validate_structure(job, payload["text"]) != payload["parsed"]:
                raise ConflictError("Source parsed output differs from its text")
        elif any(r is not None and r["result"] == "success" for r in outcomes):
            raise ConflictError("Source success is missing")
        if terminal is not None:
            if (terminal["status"] == "structure_exhausted" and structural != structure_attempts(job.stage)
                    or terminal["status"] == "transport_exhausted" and transport != config.TRANSPORT_MAX_FAILURES
                    or terminal["status"] == "permanent_error" and (outcomes[-1] is None
                        or outcomes[-1]["result"] != "transport_error" or outcomes[-1]["retryable"])):
                raise ConflictError("Source terminal contradicts its budget")
        files, evidence = [], []
        for folder in ("attempts", "failures", "telemetry", "success", "terminal"):
            paths = ([self.root / folder / (old_key + ".json")] if folder in {"success", "terminal"}
                     else sorted((self.root / folder / old_key).glob("*.json")))
            for source in paths:
                if not source.exists():
                    continue
                value, raw = _document(source)
                if value.get("key") != old_key or value.get("request_hash") != source_request_hash:
                    raise ConflictError("Source artifact request identity differs")
                changed = {**value, "key": new_key, "request_hash": request_hash}
                for hash_field in ("event_hash", "state_hash", "telemetry_hash"):
                    if hash_field in value:
                        if value[hash_field] != digest({k: v for k, v in value.items() if k != hash_field}):
                            raise ConflictError("Source artifact hash differs")
                        changed[hash_field] = digest({k: v for k, v in changed.items() if k != hash_field})
                relative = source.relative_to(self.root).as_posix()
                destination = relative.replace(old_key, new_key)
                path = self.target.root / destination
                if path.exists() and _document(path)[0] != changed:
                    raise ConflictError("Generation reuse target contains conflicting state")
                files.append((path, changed))
                evidence.append({"source": relative, "target": destination,
                                 "source_hash": digest(raw), "target_hash": digest(changed)})
        receipt = {"version": 1, "source_run_id": self.manifest["run_id"],
                   "source_manifest_hash": self.source_hash, "source_key": old_key,
                   "target_key": new_key, "request_hash": request_hash,
                   "source_request_hash": source_request_hash,
                   "mode": "endpoint_expansion" if self.endpoint_expansion else "generation_only",
                   "stage": job.stage, "files": evidence}
        for path, value in files:
            write_once(path, value)
        write_once(receipt_path, {**receipt, "receipt_hash": digest(receipt)})
