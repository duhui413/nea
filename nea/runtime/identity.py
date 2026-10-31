"""need-aware output-directory isolation and immutable run identity."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from nea.providers.config import (
    REQUEST_OPTIONS,
    EXPERIMENT_MODEL_EXTRA_BODY,
    EXPERIMENT_PROVIDER_ENDPOINTS,
    experiment_provider_name_for_model,
    experiment_role_models_for,
    local_runtime_identity,
    structured_output_mode,
    STRUCTURED_OUTPUT_PROTOCOL,
    EXECUTION_POLICY_VERSION,
    TRANSPORT_MAX_FAILURES,
    TRANSPORT_BACKOFF_SECONDS,
    TRANSPORT_MAX_WAIT_SECONDS,
)
from nea.evaluation.judging import RUBRIC_HASH
from nea.providers import config
from nea.runtime.records import digest

from nea.pipeline.registry import PROTOCOL


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Windows device names are reserved with any extension ("NUL", "nul.txt").
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[0-9]|LPT[0-9])(\..*)?$", re.IGNORECASE)
PHASE_SPLITS = ("dev", "test")


def _norm(path: Path) -> str:
    return os.path.normcase(str(path))


def validate_run_root(repo_root: Path, output_root: Path | None, run_id: str | None) -> Path:
    """Return ``<repo>/outputs/<run_id>`` or raise; never creates anything."""
    if output_root is None or not run_id:
        raise ValueError("need-aware requires explicit --output-root and --run-id")
    if not _SEGMENT.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe path segment")
    if _RESERVED.fullmatch(run_id) or run_id.endswith("."):
        raise ValueError("run_id must not be a Windows reserved device name or end with '.'")
    expected = (repo_root / "outputs").resolve(strict=False)
    if _norm(output_root.resolve(strict=False)) != _norm(expected):
        raise ValueError(f"need-aware output root must be exactly {expected}")
    lexical = Path(os.path.abspath(output_root))
    if _norm(lexical) != _norm(expected):
        raise ValueError("need-aware output root must not use an alias, '..' or a symlink")
    run_root = expected / run_id
    if _norm(run_root.resolve(strict=False)) != _norm(run_root):
        raise ValueError("need-aware run directory must not be an existing junction or symlink")
    return run_root


def model_config_identity(generator: str, *, mimo_batch: bool = False, mimo_api: str | None = None, batch_policy: dict | None = None, mimo_concurrency: int | None = None, minimax_concurrency: int | None = None, minimax_m3_api: str | None = None, minimax_relay_concurrency: int | None = None, mimo_official_concurrency: int | None = None) -> dict[str, Any]:
    """Public model identity used by successful request keys."""
    mode = config.resolve_mimo_api(mimo_api, mimo_batch=mimo_batch)
    mimo_batch = mode == "batch"
    roles_identity = {}
    for role, model in experiment_role_models_for(generator).items():
        provider = experiment_provider_name_for_model(model)
        roles_identity[role] = {"model": model, "provider": provider, "endpoint": EXPERIMENT_PROVIDER_ENDPOINTS[provider],
                                "credential_endpoints": [config.credential_endpoint(provider, slot)
                                    for slot in range(len(config.EXPERIMENT_PROVIDER_CREDENTIAL_NAMES[provider]))],
                                "extra_body": EXPERIMENT_MODEL_EXTRA_BODY[model],
                                "structured_output_mode": structured_output_mode(model)}
        if provider == "mimo" and mode == "token-plan":
            roles_identity[role].update(endpoint=config.MIMO_TOKEN_PLAN_ENDPOINT,
                credential_endpoints=[config.MIMO_TOKEN_PLAN_ENDPOINT], transport="token-plan")
        if config.lane_for_model(model, minimax_m3_api) == config.MINIMAX_RELAY_LANE:
            roles_identity[role].update(endpoint=config.MINIMAX_RELAY_ENDPOINT,
                credential_endpoints=[config.MINIMAX_RELAY_ENDPOINT], transport="minimax-relay",
                protocol=config.MINIMAX_RELAY_PROTOCOL, request_model=config.MINIMAX_RELAY_REQUEST_MODELS[model])
        if provider == "mimo" and mode == "ezlook":
            roles_identity[role].update(endpoint=config.EZLOOK_ENDPOINT,
                credential_endpoints=[config.EZLOOK_ENDPOINT], transport="ezlook")
            if config.mimo_official_concurrency(mimo_official_concurrency):
                roles_identity[role].update(credential_endpoints=[config.EZLOOK_ENDPOINT, config.MIMO_STANDARD_ENDPOINT],
                                            transport="ezlook+official")
        if mimo_batch and provider == "mimo":
            roles_identity[role].update(endpoint=config.MIMO_BATCH_ENDPOINT,
                                       credential_endpoints=[config.MIMO_BATCH_ENDPOINT], transport="batch")
    result = {**roles_identity, "request_options": dict(REQUEST_OPTIONS), "local_runtime": local_runtime_identity(),
            "structured_output_protocol": STRUCTURED_OUTPUT_PROTOCOL,
            "execution_policy": {"version": EXECUTION_POLICY_VERSION, "transport_max_failures": TRANSPORT_MAX_FAILURES,
                                 "backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
                                 "max_wait_seconds": TRANSPORT_MAX_WAIT_SECONDS}}
    if any(role["provider"] == "siliconflow" for role in roles_identity.values()):
        result["execution_policy"]["version"] = config.SILICONFLOW_EXECUTION_POLICY_VERSION
        result["execution_policy"]["siliconflow_rate_policy"] = deepcopy(config.SILICONFLOW_RATE_POLICY)
    if minimax_concurrency is not None:
        result["execution_policy"]["minimax_concurrency"] = config.minimax_concurrency(minimax_concurrency)
    result["execution_policy"]["mimo_api"] = mode
    if config.mimo_official_concurrency(mimo_official_concurrency):
        result["execution_policy"]["mimo_official_concurrency"] = config.mimo_official_concurrency(mimo_official_concurrency)
    if config.resolve_minimax_m3_api(minimax_m3_api) in config.MINIMAX_RELAY_MODE_MODELS:
        result["execution_policy"]["minimax_m3_api"] = config.resolve_minimax_m3_api(minimax_m3_api)
        result["execution_policy"]["minimax_relay_concurrency"] = config.minimax_relay_concurrency(minimax_relay_concurrency)
    if mode in ("standard", "ezlook"):
        result["execution_policy"]["mimo_rate_policy"] = deepcopy(config.MIMO_RATE_POLICY)
        if mode == "ezlook":
            result["execution_policy"]["ezlook_min_start_interval_seconds"] = config.EZLOOK_MIN_START_INTERVAL_SECONDS
        result["execution_policy"]["mimo_concurrency"] = config.mimo_concurrency(mimo_concurrency, mode)
    if mimo_batch:
        result["execution_policy"]["mimo_batch"] = deepcopy(batch_policy or config.MIMO_BATCH_POLICY)
    return result


def generator_directory(run_root: Path, generator: str) -> Path:
    canonical = experiment_role_models_for(generator)["student1"]
    path = run_root / canonical.replace("/", "__")
    if _norm(path.resolve(strict=False)) != _norm(path):
        raise ValueError("need-aware generator directory must not be a junction or symlink")
    return path


def phase_directory(run_root: Path, generator: str, split: str) -> Path:
    if split not in PHASE_SPLITS:
        raise ValueError("phase split must be dev or test")
    return generator_directory(run_root, generator) / split


def operator_phase_directory(run_root: Path, generator: str, phase: str) -> Path:
    """Keep operator live/test states outside every historical phase directory."""
    from nea.runtime.operator_protocol import DIRECTORY, source_split
    source_split(phase)
    return generator_directory(run_root / DIRECTORY, generator) / phase


def operator_manifest_identity(*, generator: str, run_id: str, phase: str,
                               template_hashes: dict, static_payload_hashes: dict,
                               question_set_sha256: str, **locked_fields: Any) -> dict[str, Any]:
    """Build the immutable identity; the pipeline supplies data/schema/budgets."""
    from nea.runtime.operator_protocol import METHOD_VERSION, PROTOCOL as OPERATOR_PROTOCOL, source_split
    result = {"protocol_version": OPERATOR_PROTOCOL, "method_version": METHOD_VERSION,
              "run_id": run_id, "split": phase, "question_split": source_split(phase),
              "generator": experiment_role_models_for(generator)["student1"],
              "template_hashes": template_hashes, "static_payload_hashes": static_payload_hashes,
              "rubric_hash": RUBRIC_HASH, "question_set_sha256": question_set_sha256,
              "model_config_hash": digest(model_config_identity(generator))}
    if set(locked_fields) & set(result):
        raise ValueError("locked operator fields cannot replace the runtime identity")
    return {**result, **locked_fields}


def manifest_identity(*, generator, run_id, split, template_hashes, static_payload_hashes,
                      calibration_hash, question_set_sha256, detector):
    return {"protocol_version": PROTOCOL, "run_id": run_id, "split": split,
            "generator": experiment_role_models_for(generator)["student1"],
            "template_hashes": template_hashes, "static_payload_hashes": static_payload_hashes,
            "rubric_hash": RUBRIC_HASH, "calibration_hash": calibration_hash,
            "question_set_sha256": question_set_sha256, "detector": detector,
            "model_config_hash": digest(model_config_identity(generator))}
