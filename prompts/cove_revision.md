---
template_version: nea-cove-revision-v1
stage_id: cove_revision
stage_kind: llm_prompt
input_schema_version: 1
output_schema_version: 1
---

根据原问题、草稿答案与独立验证问答，作一次有界修订。纠正验证问答实际揭示的错误，保留原答案中的正确条件和有效推理。不要因为不确定验证结论引入新的肯定断言。只使用本题允许的共享证据，PE不包含检索正文；引文ID只可取allowed_citations。输出完整答案正文，不含JSON、审阅报告、内部标签或义务列表；字符数不得超过max_chars。没有实际问题时保持草稿。
