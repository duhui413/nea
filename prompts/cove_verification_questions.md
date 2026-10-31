---
template_version: nea-cove-questions-v1
stage_id: cove_verification_questions
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

根据原问题、草稿答案与允许的共享证据，提出能够独立验证草稿中重要事实、成立条件或推理步骤的问题。最多3个，每个问题自足、明确、可回答，不包含其期望答案，不复述草稿结论作为前提。不使用任何NEA模块、诊断标签或任务义务表。PE输入不含检索正文，不得假装查阅外部资料。
输出verification_questions数组，每项包含question_id（v1、v2、v3）和question。确无可核验问题时可输出空数组。
