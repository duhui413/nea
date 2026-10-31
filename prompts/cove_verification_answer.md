---
template_version: nea-cove-independent-answer-v1
stage_id: cove_verification_answer
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

独立回答每个验证问题，只参考原问题及本题允许的共享证据。输入有意不包含草稿答案；不能猜测或迎合草稿的结论，不得调用外部检索。每个问题输出question_id、answer以及status（answered或uncertain）。证据不足则如实说明不确定。必须恰好回答输入中的全部问题，保持question_id对应，不添加额外问题。不使用NEA标签或义务表。
