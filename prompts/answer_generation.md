---
template_version: nea-0022-v1
stage_id: answer_generation
stage_kind: llm_prompt
input_schema_version: 2
output_schema_version: 2
---

直接回答原问题，按任务义务组织答案。B/P/L共有相同的正确性、切题和必要信息要求；输入计划、概念及图仅作可调整的支撑建议。无计划时自行围绕原任务组织，不编造计划。
不使用固定概念槽位，不限定每概念句数，不强制例子或两个解释单元；必要前置和推导按任务展开。纠正问题或证据中的错误前提并解释依据。区分结论、证明条件与不确定性。只输出完整答案正文（Markdown），不输出内部诊断、自评分或JSON。
