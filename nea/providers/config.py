"""The sole definition of experiment models, endpoints and request bodies."""
from __future__ import annotations
import os
from pathlib import Path

LOCAL_MODEL_PATHS = {"retrieval_embedding": "BAAI/bge-small-zh-v1.5", "reranker": "BAAI/bge-reranker-base"}
LOCAL_MODEL_DEVICE = "cuda"
LOCAL_MODEL_DTYPE = "fp16"
LOCAL_MODEL_SEED = 23
LOCAL_MODEL_ATTENTION = "eager"


def local_runtime_identity() -> dict:
    return {"models": dict(LOCAL_MODEL_PATHS), "device": LOCAL_MODEL_DEVICE,
            "dtype": LOCAL_MODEL_DTYPE, "seed": LOCAL_MODEL_SEED,
            "attention": LOCAL_MODEL_ATTENTION, "grounding_score_dtype": "fp32",
            "grounding_device": "cuda", "grounding_matmul_dtype": "fp16",
            "grounding_normalization_dtype": "fp32", "grounding_precision_version": "cuda-mixed-v1",
            "milvus_vector_dtype": "float16", "milvus_pi_dtype": "float64"}


EXPERIMENT_ACTIVE_FAMILIES = ("mimo", "minimax")
EXPERIMENT_GENERATORS = ("mimo-v2.6-flash", "MiniMax-M3.1-Flash-Preview")
EXPERIMENT_MODEL_ASSIGNMENTS = dict(zip(("student1", "student2", "teacher1", "teacher2"),
    (*EXPERIMENT_GENERATORS, "mimo-v2.6-pro", "MiniMax-M3")))
EXPERIMENT_PROVIDER_ORDER = ("mimo", "minimax")
EXPERIMENT_PROVIDER_ENDPOINTS = {"mimo": "https://api.xiaomimimo.com/v1",
                                  "minimax": "https://api.minimax.cn/v1"}
# MiMo credentials are isolated by billing/transport mode.
MIMO_TOKEN_PLAN_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_TOKEN_PLAN_CREDENTIAL_NAME = "MIMO_TOKEN_PLAN_API_KEY"
MIMO_STANDARD_ENDPOINT = "https://api.xiaomimimo.com/v1"
# ezlook is a third-party relay exposing the same two MiMo models (single key, one in-flight request).
EZLOOK_ENDPOINT = "https://mimo.ezlook.top/v1"
EZLOOK_CREDENTIAL_NAME = "EZLOOK_API_KEY"
EZLOOK_MODELS = ("mimo-v2.6-flash", "mimo-v2.6-pro")
# MiniMax relay (specs 0041/0042): both MiniMax models are routed there over the Anthropic Messages protocol
# (Flash-Preview has no OpenAI-protocol channel on the relay). One key, total in-flight shared by both models.
MINIMAX_RELAY_LANE = "minimax-relay"
MINIMAX_RELAY_ENDPOINT = "http://118.89.124.131:3001/v1"
MINIMAX_RELAY_CREDENTIAL_NAME = "MINIMAX_RELAY_API_KEY"
MINIMAX_RELAY_MODELS = ("MiniMax-M3", "MiniMax-M3.1-Flash-Preview")
MINIMAX_RELAY_REQUEST_MODELS = {"MiniMax-M3": "MiniMax-M3", "MiniMax-M3.1-Flash-Preview": "MiniMax M3.1-Flash-Preview"}
# Relay modes (spec 0046): "relay" routes both MiniMax models, "relay-m3" only MiniMax-M3 (Flash stays official).
MINIMAX_RELAY_MODE_MODELS = {"relay": MINIMAX_RELAY_MODELS, "relay-m3": ("MiniMax-M3",)}
MINIMAX_RELAY_PROTOCOL = "anthropic-messages"
MINIMAX_RELAY_ANTHROPIC_VERSION = "2023-06-01"
MINIMAX_RELAY_DEFAULT_CONCURRENCY = 8
MINIMAX_RELAY_MAX_CONCURRENCY = 32
# Batch is a separate transport, never a third realtime credential slot.
MIMO_BATCH_ENDPOINT = "https://batch-api-cn.xiaomimimo.com/v1"
MIMO_BATCH_CREDENTIAL_NAME = "MIMO_BATCH_API_KEY"
MIMO_BATCH_REQUEST_PATH = "/v1/chat/completions"
MIMO_BATCH_API_PATHS = {"files": "/files", "batches": "/batches", "content": "/content"}
MIMO_BATCH_FILE_PURPOSE = "batch"
MIMO_BATCH_POLICY = {"version": "mimo-batch-v1", "completion_window": "24h",
    "max_requests": 5, "max_input_bytes": 120 * 1024 * 1024,
    "poll_seconds": 60.0, "max_active_batches": 1, "request_timeout_seconds": 300.0}
FORMAL_OVERNIGHT_BATCH_MAX_REQUESTS = 5
EXPERIMENT_MODEL_EXTRA_BODY = {
    "mimo-v2.6-flash": {"thinking": {"type": "disabled"}},
    "mimo-v2.6-pro": {"thinking": {"type": "disabled"}},
    "MiniMax-M3.1-Flash-Preview": {"reasoning_split": True, "reasoning_effort": "low"},
    "MiniMax-M3": {"thinking": {"type": "disabled"}, "reasoning_split": True},
}
EXPERIMENT_PROVIDER_CREDENTIAL_NAMES = {
    "mimo": ("MIMO_API_KEY", *(f"MIMO_API_KEY_{i}" for i in range(2, 9))),
    "minimax": ("MINIMAX_API_KEY", "MINIMAX_API_KEY_2"),
}
EXPERIMENT_MAX_IN_FLIGHT = 96
MIMO_DEFAULT_CONCURRENCY = 8
MIMO_MAX_CONCURRENCY = 8
# ezlook limits request STARTS (about 2 per second for the key), not concurrency: spaced starts allow more in flight.
EZLOOK_MAX_CONCURRENCY = 24
EZLOOK_MIN_START_INTERVAL_SECONDS = 0.7
# Optional second MiMo route (spec 0043): the official standard endpoint beside ezlook, with its own token budgets.
MIMO_OFFICIAL_LANE = "mimo-official"
MIMO_OFFICIAL_MAX_CONCURRENCY = 16
MIMO_LANES = ("mimo", MIMO_OFFICIAL_LANE)
MIMO_RATE_POLICY = {"version": "shared-model-settled60-v1", "tpm_per_key": 10_000_000,
    "rpm": 100, "window_seconds": 60.0, "startup_seconds": 60.0,
    "scope": "per-model-across-standard-keys", "max_in_flight_per_key": 8}
EXECUTION_POLICY_VERSION = "resume-v3"
TRANSPORT_MAX_FAILURES = 3
TRANSPORT_BACKOFF_SECONDS = (2.0, 4.0)
TRANSPORT_MAX_WAIT_SECONDS = 60.0
REQUEST_OPTIONS = {"temperature": 0.2, "seed": 23, "max_output_tokens": 16384, "timeout_seconds": 300.0}

# Inactive until an explicit model/provider/credential plan enables SiliconFlow.
SILICONFLOW_RATE_POLICY = {
    "version": "conservative-settled60-v1", "tpm_per_key": 2_000_000,
    "window_seconds": 60.0, "max_in_flight_per_key": 1,
    "envelope_overhead_tokens": 256, "startup_seconds": 60.0,
    "estimate": "utf8-input-envelope-plus-full-output-v1",
}
SILICONFLOW_EXECUTION_POLICY_VERSION = "resume-v4-token-budget"


def chat_request_body(*, model, prompt, temperature, max_tokens, seed, extra_body=None):
    """One source for the exact body used by admission and HTTP transport."""
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            **(extra_body or {})}


def batch_input_row(custom_id: str, body: dict) -> dict:
    return {"custom_id": custom_id, "method": "POST", "url": MIMO_BATCH_REQUEST_PATH, "body": dict(body)}


def batch_create_body(input_file_id: str, name: str) -> dict:
    return {"input_file_id": input_file_id, "endpoint": MIMO_BATCH_REQUEST_PATH,
            "completion_window": MIMO_BATCH_POLICY["completion_window"], "name": name}

# Native support is evidence-based, never inferred from a successful HTTP status.
STRUCTURED_OUTPUT_PROTOCOL = "stage-schema-v1"
EXPERIMENT_STRUCTURED_OUTPUT_MODES = {
    "mimo-v2.6-flash": "json_schema",
    "mimo-v2.6-pro": "json_schema",
    "MiniMax-M3.1-Flash-Preview": "prompt_json",
    "MiniMax-M3": "prompt_json",
}


def structured_output_mode(model: str) -> str:
    mode = EXPERIMENT_STRUCTURED_OUTPUT_MODES[model]
    if mode not in {"json_schema", "json_object", "prompt_json"}:
        raise ValueError("unknown structured output mode")
    return mode


def structured_extra_body(model: str, schema: dict | None, name: str | None = None) -> dict:
    from copy import deepcopy
    body = deepcopy(EXPERIMENT_MODEL_EXTRA_BODY[model])
    if schema is not None:
        mode = structured_output_mode(model)
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": name or "stage_output", "strict": True, "schema": deepcopy(schema)}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
    return body

def experiment_role_models_for(generator: str) -> dict[str, str]:
    aliases = dict(zip(EXPERIMENT_ACTIVE_FAMILIES, EXPERIMENT_GENERATORS))
    generator = aliases.get(generator, generator)
    if generator not in EXPERIMENT_GENERATORS:
        raise ValueError("generator must be mimo or minimax")
    other = next(model for model in EXPERIMENT_GENERATORS if model != generator)
    return {**EXPERIMENT_MODEL_ASSIGNMENTS, "student1": generator, "student2": other}

def experiment_provider_name_for_model(model: str) -> str:
    if model not in EXPERIMENT_MODEL_EXTRA_BODY:
        raise ValueError("model is outside the paper experiment group")
    return "mimo" if model.startswith("mimo-") else "minimax"

def credential_endpoint(provider: str, slot: int, *, mimo_api: str = "standard") -> str:
    """Zero-based configured slot; endpoints never come from the environment."""
    if provider == MINIMAX_RELAY_LANE:
        if slot != 0:
            raise ValueError("unknown credential slot")
        return MINIMAX_RELAY_ENDPOINT
    if provider == MIMO_OFFICIAL_LANE:
        if not 0 <= slot < len(EXPERIMENT_PROVIDER_CREDENTIAL_NAMES["mimo"]):
            raise ValueError("unknown credential slot")
        return MIMO_STANDARD_ENDPOINT
    if not 0 <= slot < len(EXPERIMENT_PROVIDER_CREDENTIAL_NAMES[provider]):
        raise ValueError("unknown credential slot")
    if provider == "mimo":
        if mimo_api == "token-plan" and slot == 0:
            return MIMO_TOKEN_PLAN_ENDPOINT
        if mimo_api == "ezlook" and slot == 0:
            return EZLOOK_ENDPOINT
        if mimo_api != "standard":
            raise ValueError("invalid realtime MiMo mode or slot")
        return MIMO_STANDARD_ENDPOINT
    return EXPERIMENT_PROVIDER_ENDPOINTS[provider]


class CredentialKeys(tuple):
    """Keys with their original configured positions after empty-slot filtering."""
    def __new__(cls, entries):
        entries = tuple(entries)
        obj = super().__new__(cls, (key for _, key in entries))
        obj.slot_indices = tuple(slot for slot, _ in entries)
        return obj


def load_credential_pools(env_file: Path, *, mimo_batch: bool = False, mimo_api: str | None = None,
                          minimax_m3_api: str | None = None, mimo_official: bool = False) -> dict[str, tuple[str, ...]]:
    """Read only fixed credential slots, with environment precedence and deduplication."""
    mode = resolve_mimo_api(mimo_api, mimo_batch=mimo_batch)
    credential_names = {**EXPERIMENT_PROVIDER_CREDENTIAL_NAMES}
    if mode == "token-plan":
        credential_names["mimo"] = (MIMO_TOKEN_PLAN_CREDENTIAL_NAME,)
    elif mode == "ezlook":
        credential_names["mimo"] = (EZLOOK_CREDENTIAL_NAME,)
    if resolve_minimax_m3_api(minimax_m3_api) in MINIMAX_RELAY_MODE_MODELS:
        credential_names[MINIMAX_RELAY_LANE] = (MINIMAX_RELAY_CREDENTIAL_NAME,)
    if mimo_official:
        credential_names[MIMO_OFFICIAL_LANE] = EXPERIMENT_PROVIDER_CREDENTIAL_NAMES["mimo"]
    names = {name for slots in credential_names.values() for name in slots}
    values = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() in names:
                values[name.strip()] = value.strip().strip('"').strip("'")
    result: dict[str, tuple[str, ...]] = {}
    missing = []
    for provider, slots in credential_names.items():
        if mode == "batch" and provider == "mimo":
            continue
        keys = tuple(os.environ.get(name, values.get(name, "")).strip() for name in slots)
        if not any(keys):
            missing.append(slots[0])
        seen = set()
        entries = []
        for slot, key in enumerate(keys):
            if key and key not in seen:
                entries.append((slot, key))
                seen.add(key)
        result[provider] = CredentialKeys(entries)
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(missing))
    return result


def load_mimo_batch_credential(env_file: Path) -> str:
    """Only the dedicated whitelist slot; an empty process value overrides .env."""
    name = MIMO_BATCH_CREDENTIAL_NAME
    value = ""
    if name in os.environ:
        value = os.environ[name].strip()
    elif env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            field, sep, raw = line.strip().partition("=")
            if sep and field.strip() == name:
                value = raw.strip().strip('"').strip("'")
    if not value:
        raise ValueError("Missing credentials: " + name)
    return value


def load_credentials(env_file: Path) -> dict[str, str]:
    """Compatibility view for callers that only need each provider's primary key."""
    return {provider: keys[0] for provider, keys in load_credential_pools(env_file).items()}


# Concept-text correction (spec 0019). Kept outside the paper experiment group above.
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
DEEPSEEK_CREDENTIAL_NAME = "DEEPSEEK_API_KEY"
CORRECTION_MODEL = "deepseek-v4-pro"
CORRECTION_EXTRA_BODY = {"thinking": {"type": "disabled"}}
CORRECTION_REQUEST_OPTIONS = {"temperature": 0.0, "seed": 23, "max_output_tokens": 8192, "timeout_seconds": 300.0}
CORRECTION_MAX_IN_FLIGHT = 4


def load_deepseek_credential(env_file: Path) -> str:
    value = os.environ.get(DEEPSEEK_CREDENTIAL_NAME, "").strip()
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, sep, raw = line.strip().partition("=")
            if sep and name.strip() == DEEPSEEK_CREDENTIAL_NAME:
                value = raw.strip().strip('"').strip("'")
    if not value:
        raise ValueError("Missing credentials: " + DEEPSEEK_CREDENTIAL_NAME)
    return value


def resolve_mimo_api(mode: str | None = None, *, mimo_batch: bool = False) -> str:
    if mode not in (None, "standard", "token-plan", "ezlook", "batch"):
        raise ValueError("unknown MiMo API mode")
    if mimo_batch and mode not in (None, "batch"):
        raise ValueError("--mimo-batch conflicts with --mimo-api")
    return "batch" if mimo_batch else mode or "standard"


def resolve_minimax_m3_api(mode: str | None = None) -> str:
    if mode not in (None, "official", *MINIMAX_RELAY_MODE_MODELS):
        raise ValueError("unknown MiniMax-M3 API mode")
    return mode or "official"


def lane_for_model(model: str, minimax_m3_api: str | None = None) -> str:
    """Scheduling lane; provider identity in records stays the logical family."""
    if model in MINIMAX_RELAY_MODE_MODELS.get(resolve_minimax_m3_api(minimax_m3_api), ()):
        return MINIMAX_RELAY_LANE
    return experiment_provider_name_for_model(model)


def minimax_relay_concurrency(value=None):
    value = MINIMAX_RELAY_DEFAULT_CONCURRENCY if value is None else value
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise ValueError("MiniMax relay concurrency must be an integer") from None
    if isinstance(value, bool) or str(number) != str(value) or not 1 <= number <= MINIMAX_RELAY_MAX_CONCURRENCY:
        raise ValueError(f"MiniMax relay concurrency must be between 1 and {MINIMAX_RELAY_MAX_CONCURRENCY}")
    return number


def mimo_concurrency(value=None, mode=None):
    """1-8 for batch; the official standard route up to 16 and the ezlook relay up to EZLOOK_MAX_CONCURRENCY."""
    limit = EZLOOK_MAX_CONCURRENCY if mode == "ezlook" else MIMO_OFFICIAL_MAX_CONCURRENCY if mode == "standard" else MIMO_MAX_CONCURRENCY
    value = MIMO_DEFAULT_CONCURRENCY if value is None else value
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise ValueError("MiMo concurrency must be an integer") from None
    if isinstance(value, bool) or str(number) != str(value) or not 1 <= number <= limit:
        raise ValueError(f"MiMo concurrency must be between 1 and {limit}")
    return number


def mimo_official_concurrency(value=None):
    """0 = the official lane is off; otherwise 1..MIMO_OFFICIAL_MAX_CONCURRENCY."""
    if value is None or (type(value) is int and value == 0):
        return 0
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise ValueError("MiMo official concurrency must be an integer") from None
    if isinstance(value, bool) or str(number) != str(value) or not 1 <= number <= MIMO_OFFICIAL_MAX_CONCURRENCY:
        raise ValueError(f"MiMo official concurrency must be between 1 and {MIMO_OFFICIAL_MAX_CONCURRENCY}")
    return number


def mimo_concurrency_cli(value):
    """Command-line syntax check (the mode-specific ceiling is enforced where the mode is known)."""
    return mimo_concurrency(value, "ezlook")


def minimax_concurrency(value=None):
    if value is None:
        return None
    if isinstance(value, bool) or str(value) not in {"1", "2"}:
        raise ValueError("MiniMax total concurrency must be 1 or 2")
    return int(value)


def add_mimo_arguments(parser):
    parser.add_argument("--minimax-concurrency", type=minimax_concurrency, default=None,
                        help="total MiniMax in-flight cap, 1 or 2; default: distinct key count")
    parser.add_argument("--mimo-concurrency", type=mimo_concurrency_cli, default=None,
                        help="total standard MiMo in-flight limit, independent of key count (default: 8, range: 1-8; ezlook: 1-24)")
    parser.add_argument("--mimo-api", choices=("standard", "token-plan", "ezlook", "batch"),
                        help="MiMo API mode (default: standard); keep the same mode on resume")
    parser.add_argument("--mimo-official-concurrency", type=mimo_official_concurrency, default=None,
                        help="also route MiMo to the official endpoint with this total in-flight limit (1-16; default: off)")
    parser.add_argument("--minimax-m3-api", choices=("official", "relay", "relay-m3"), default=None,
                        help="route MiniMax to the relay: relay = both models, relay-m3 = MiniMax-M3 only (default: official); keep the same mode on resume")
    parser.add_argument("--minimax-relay-concurrency", type=minimax_relay_concurrency, default=None,
                        help="total relay in-flight limit for routed MiniMax models (default: 8, range: 1-32)")
    parser.add_argument("--mimo-batch-size", type=batch_size, default=5,
                        help="requests per Batch task, 1-100 (default: 5)")


def batch_size(value):
    import argparse
    try:
        size = int(value)
        if not 1 <= size <= 100:
            raise ValueError()
        return size
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Batch size must be between 1 and 100") from None

MIMO_SMOKE_OPTIONS = {"temperature": 0.2, "seed": 23, "max_tokens": 32, "timeout": 30.0}
MIMO_SMOKE_PROMPT = "Reply with exactly OK."
MIMO_SMOKE_MODELS = ("mimo-v2.6-flash", "mimo-v2.6-pro")
