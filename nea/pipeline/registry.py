"""Paper arms, stage rules and parent relations."""

from __future__ import annotations

from typing import Any

from nea.providers.config import EXPERIMENT_GENERATORS

from nea.runtime.records import MAX_STRUCTURE_ATTEMPTS


PROTOCOL = "ca-v3"
GENERATORS = tuple(EXPERIMENT_GENERATORS)  # spec 0060/0002: MiMo + MiniMax
BASES = ("pe", "rag")
JUDGE_ROLES = ("teacher1", "teacher2")
GENERATION_ROLE = "student1"
ARMS_TEST = ("B", "P", "L", "F", "SR")
ARMS_DEV_ONLY = ("KP", "D1", "D2", "M-CER", "M-DR", "M-RCC", "M-LTS")
ALL_ARMS = ARMS_TEST + ARMS_DEV_ONLY
DETECTORS = ("D1", "D2", "D3")
DEFAULT_DETECTOR = "D3"
SIGNALS = ("cer_g", "dr_g", "rcc", "lts")
# The evaluation calls are not an experimental arm.  They use one explicit
# pseudo-arm so the whitelist still rejects every unlisted combination.
EVAL_ARM = "EVAL"
EVAL_STAGE = "judge"

GROUNDED_STAGES = ("ground", "gaps", "answer")
REVISION_STAGES = ("diagnose", "revise")
STAGE_WHITELIST: dict[str, tuple[str, ...]] = {
    "B": ("b_plan", "b_answer"),
    "P": ("plan", "answer"),
    "L": GROUNDED_STAGES,
    "KP": GROUNDED_STAGES,
    "D1": GROUNDED_STAGES,
    "D2": GROUNDED_STAGES,
    "F": REVISION_STAGES,
    "M-CER": REVISION_STAGES,
    "M-DR": REVISION_STAGES,
    "M-RCC": REVISION_STAGES,
    "M-LTS": REVISION_STAGES,
    "SR": ("critique", "revise"),
    EVAL_ARM: (EVAL_STAGE,),
}
PARENT_ARM = {"F": "L", "M-CER": "L", "M-DR": "L", "M-RCC": "L", "M-LTS": "L", "SR": "B"}
FEEDBACK_MASK = {
    "F": frozenset(SIGNALS),
    "M-CER": frozenset(SIGNALS) - {"cer_g"},
    "M-DR": frozenset(SIGNALS) - {"dr_g"},
    "M-RCC": frozenset(SIGNALS) - {"rcc"},
    "M-LTS": frozenset(SIGNALS) - {"lts"},
}
# Which gap detector each grounded arm uses; L follows the frozen choice.
ARM_DETECTOR = {"KP": "KP", "D1": "D1", "D2": "D2"}
REVISION_LIMITS = {"max_rounds": 1, "max_calls": 3, "length_retry": 1,
                   "max_growth": 1.2, "max_actions": 3}
# Real router calls allowed per success key.  A revise key gets exactly one
# call: a structurally bad revise response is one of the <= 3 revise calls of
# its cell (recorded as ``invalid_structure``), never a hidden extra retry.
# Every other stage keeps the local-path structure-retry limit (1 call + 2 retries).
STRUCTURE_ATTEMPTS_BY_STAGE = {"revise": 1}


def structure_attempts(stage: str) -> int:
    return STRUCTURE_ATTEMPTS_BY_STAGE.get(stage, MAX_STRUCTURE_ATTEMPTS)














def arms_for_split(split: str) -> tuple[str, ...]:
    if split == "dev":
        return ALL_ARMS
    if split == "test":
        return ARMS_TEST
    raise ValueError("split must be dev or test")


def check_arm_stage(arm: str, stage: str, split: str) -> None:
    """Reject every arm x stage (x split) combination not in the whitelist."""
    if split not in {"dev", "test"}:
        raise ValueError(f"unknown split {split!r}")
    allowed = STAGE_WHITELIST.get(arm)
    if allowed is None:
        raise ValueError(f"unknown need-aware arm {arm!r}")
    if stage not in allowed:
        raise ValueError(f"arm x stage {arm} x {stage} is not in the need-aware whitelist")
    if arm in ARMS_DEV_ONLY and split != "dev":
        raise ValueError(f"arm {arm} is dev-only and cannot run on {split}")


def role_for_stage(stage: str) -> tuple[str, ...]:
    return JUDGE_ROLES if stage == EVAL_STAGE else (GENERATION_ROLE,)
