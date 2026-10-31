from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
ROLES = ("common_rules", "query_task_analysis", "target_grounding",
         "explanation_obligation_planning", "answer_generation", "explanation_diagnosis",
         "answer_revision", "revision_validation", "generic_answer_critique",
         "answer_quality_evaluation", "structured_output", "repair_output")
PROMPT_FILES_BY_STAGE = {role: role + ".md" for role in ROLES}
ALIASES = {"step1": "query_task_analysis", "step2": "answer_generation",
           "step3": "explanation_diagnosis", "step4": "answer_revision",
           "step5": "answer_quality_evaluation", "plan": "explanation_obligation_planning",
           "ground": "target_grounding", "answer": "answer_generation",
           "gaps_d1": "explanation_obligation_planning", "gaps_d3": "explanation_obligation_planning",
           "diagnose": "explanation_diagnosis", "revise": "answer_revision",
           "sr_revise": "answer_revision", "sr_critique": "generic_answer_critique",
           "judge": "answer_quality_evaluation", "validate": "revision_validation"}
ACTIVE_PROMPTS = tuple(PROMPT_FILES_BY_STAGE.values()) + ("quality_rubric.json", "validation_reasons.json")
LLM_PROMPT_STAGES = frozenset(ROLES) | frozenset(ALIASES)
RULE_CONTRACT_STAGES = frozenset()


REQUIRED_METADATA = {
    "template_version",
    "stage_id",
    "stage_kind",
    "input_schema_version",
    "output_schema_version",
}


class PromptStore:
    def __init__(self, directory: Path):
        self.directory = directory
        missing = [name for name in ACTIVE_PROMPTS if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"missing registered prompt files: {missing}")
        self._templates = {
            stage: (directory / filename).read_text(encoding="utf-8-sig")
            for stage, filename in PROMPT_FILES_BY_STAGE.items()
        }
        self._metadata = {
            stage: self._parse_metadata(template, stage)
            for stage, template in self._templates.items()
        }

    @staticmethod
    def _parse_metadata(template: str, stage: str) -> dict[str, str]:
        """Read the tiny YAML front matter without making YAML a runtime dependency."""
        lines = template.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError(f"prompt {stage} must start with YAML front matter")
        closing = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if closing is None:
            raise ValueError(f"prompt {stage} has unclosed YAML front matter")
        metadata: dict[str, str] = {}
        for line in lines[1:closing]:
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        missing = REQUIRED_METADATA - set(metadata)
        if missing:
            raise ValueError(f"prompt {stage} metadata missing {sorted(missing)}")
        if metadata["stage_id"] != stage:
            raise ValueError(f"prompt {stage} has mismatched stage_id={metadata['stage_id']!r}")
        expected_kind = "rule_contract" if stage in RULE_CONTRACT_STAGES else "llm_prompt"
        if metadata["stage_kind"] != expected_kind:
            raise ValueError(
                f"prompt {stage} must declare stage_kind={expected_kind!r}; "
                f"found {metadata['stage_kind']!r}"
            )
        for field in ("input_schema_version", "output_schema_version"):
            if not metadata[field].isdigit() or int(metadata[field]) < 1:
                raise ValueError(f"prompt {stage} has invalid {field}")
        return metadata

    def active_hashes(self) -> dict[str, str]:
        return {
            name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            for name in ACTIVE_PROMPTS
        }

    def source_hashes(self) -> dict[str, str]:
        source = self.directory / "source"
        if not source.exists():
            return {}
        return {
            str(path.relative_to(self.directory)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*.md"))
        }

    def metadata(self) -> dict[str, dict[str, str]]:
        return {stage: dict(values) for stage, values in self._metadata.items()}

    @staticmethod
    def is_llm_stage(step: str) -> bool:
        return step in LLM_PROMPT_STAGES

    def text(self, role: str) -> str:
        role = ALIASES.get(role, role)
        if role not in self._templates:
            raise ValueError(f"unknown prompt role {role!r}")
        return self._templates[role]

    def render(self, step: str, payload: dict[str, Any], repair_note: str | None = None,
               *, schema: dict[str, Any] | None = None) -> str:
        role = ALIASES.get(step, step)
        if schema is None:
            from nea.method.templates import role_schema
            schema = role_schema(step, payload)
        sections = [self.text("common_rules"), self.text(role)]
        if schema is not None:
            sections += [self.text("structured_output"), json.dumps(schema, ensure_ascii=False, sort_keys=True)]
        sections += ["## 本题输入\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)]
        if repair_note:
            sections += [self.text("repair_output"), repair_note[:1200]]
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
